// Host test for the Vulkan backend's per-dispatch descriptor-update templates
// (src/backends/vk/descriptor_update_template.h, plan P2.2).
//
// This test covers:
// - the packed payload layout: entry offsets are multiples of every payload
//   structure's size, strides are that size, and the payload bytes are a
//   byte-for-byte copy of the caller's VkDescriptorBufferInfo /
//   VkDescriptorImageInfo / VkBufferView values in write order
// - multi-descriptor writes (descriptorCount > 1) keep their contiguous payload
//   array, array elements and counts
// - every shape the template ABI cannot represent fails closed and leaves the
//   plan empty, so the caller can only fall back to vkUpdateDescriptorSets
//   (a chained pNext such as the acceleration-structure write, an unsupported
//   descriptor type, a null payload, an empty count, two descriptor sets,
//   overlapping ranges, an oversized payload)
// - shape matching: a plan is applied only to a write list with identical
//   bindings, array elements, counts and descriptor types
// - the cache: one vkCreateDescriptorUpdateTemplate per distinct shape, reuse
//   across dispatches, and a destroy path that reaches every live template
//   exactly once
// - the single funnel: the legacy branch issues exactly one
//   vkUpdateDescriptorSets with the caller's own writes and never touches the
//   template entry points; the template branch issues no legacy call at all;
//   an empty write list issues nothing
// Everything runs on the host. No Vulkan device and no driver: the three
// template entry points and vkUpdateDescriptorSets are volk's function-pointer
// globals, so the test installs recording stubs into them.
#include "ut/ut.hpp"

#include "descriptor_update_template.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;

namespace {

constexpr auto kMaxTestEntries = 8u;

// Fake Vulkan handles. Distinct addresses are distinct handles as far as the
// planner and the stubs can tell; using objects instead of integer casts keeps
// the initializers portable across the platforms where a Vulkan handle is a
// pointer.
struct FakeHandlePool {
    int slots[16]{};
};
FakeHandlePool g_handles;

template<typename Handle>
[[nodiscard]] Handle fake_handle(uint32_t slot) noexcept {
    return reinterpret_cast<Handle>(&g_handles.slots[slot % 16u]);
}

[[nodiscard]] VkDescriptorSet fake_set(uint32_t slot) noexcept {
    return fake_handle<VkDescriptorSet>(slot);
}
[[nodiscard]] VkDescriptorSetLayout fake_layout(uint32_t slot) noexcept {
    return fake_handle<VkDescriptorSetLayout>(slot);
}
[[nodiscard]] VkDescriptorUpdateTemplate fake_template(uint32_t slot) noexcept {
    return fake_handle<VkDescriptorUpdateTemplate>(slot);
}
[[nodiscard]] VkBuffer fake_buffer(uint32_t slot) noexcept {
    return fake_handle<VkBuffer>(slot);
}
[[nodiscard]] VkImageView fake_image_view(uint32_t slot) noexcept {
    return fake_handle<VkImageView>(slot);
}

// --- recording stubs for the volk entry-point globals ----------------------
struct StubCalls {
    uint32_t legacy_calls{0u};
    uint32_t legacy_write_count{0u};
    const VkWriteDescriptorSet *legacy_writes{nullptr};
    uint32_t legacy_copy_count{0u};

    uint32_t create_calls{0u};
    uint32_t update_calls{0u};
    uint32_t destroy_calls{0u};
    // The allocation callbacks of the create call and of every destroy call. A
    // backend must hand the *same* pointer to both: Vulkan requires the callbacks
    // of a destroy to match the ones the object was created with.
    const VkAllocationCallbacks *create_callbacks{nullptr};
    std::vector<const VkAllocationCallbacks *> destroy_callbacks;

    // Last create info, decoded.
    VkDescriptorUpdateTemplateType template_type{
        VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET};
    VkDescriptorUpdateTemplateCreateFlags create_flags{0u};
    VkDescriptorSetLayout create_layout{VK_NULL_HANDLE};
    VkPipelineBindPoint create_bind_point{VK_PIPELINE_BIND_POINT_COMPUTE};
    uint32_t entry_count{0u};
    std::array<VkDescriptorUpdateTemplateEntry, kMaxTestEntries> entries{};

    // Last update call.
    VkDescriptorSet last_set{VK_NULL_HANDLE};
    VkDescriptorUpdateTemplate last_handle{VK_NULL_HANDLE};
    size_t expected_data_size{0u};
    std::vector<std::byte> data;

    // Handles handed out by the create stub, in creation order.
    std::vector<VkDescriptorUpdateTemplate> created;
};
StubCalls g_stub;

void reset_stub() noexcept {
    g_stub = StubCalls{};
}

VKAPI_ATTR void VKAPI_CALL stub_update_descriptor_sets(
    VkDevice, uint32_t write_count, const VkWriteDescriptorSet *writes,
    uint32_t copy_count, const VkCopyDescriptorSet *) noexcept {
    ++g_stub.legacy_calls;
    g_stub.legacy_write_count = write_count;
    g_stub.legacy_writes = writes;
    g_stub.legacy_copy_count = copy_count;
}

VKAPI_ATTR VkResult VKAPI_CALL stub_create_template(
    VkDevice, const VkDescriptorUpdateTemplateCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkDescriptorUpdateTemplate *out) noexcept {
    ++g_stub.create_calls;
    g_stub.create_callbacks = callbacks;
    g_stub.template_type = info->templateType;
    g_stub.create_flags = info->flags;
    g_stub.create_layout = info->descriptorSetLayout;
    g_stub.create_bind_point = info->pipelineBindPoint;
    g_stub.entry_count = info->descriptorUpdateEntryCount;
    g_stub.entries.fill(VkDescriptorUpdateTemplateEntry{});
    auto count = std::min<size_t>(info->descriptorUpdateEntryCount, kMaxTestEntries);
    for (size_t i = 0u; i < count; ++i) {
        g_stub.entries[i] = info->pDescriptorUpdateEntries[i];
    }
    auto handle = fake_template(static_cast<uint32_t>(g_stub.create_calls));
    g_stub.created.emplace_back(handle);
    *out = handle;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL stub_update_with_template(
    VkDevice, VkDescriptorSet set, VkDescriptorUpdateTemplate handle,
    const void *data) noexcept {
    ++g_stub.update_calls;
    g_stub.last_set = set;
    g_stub.last_handle = handle;
    auto size = g_stub.expected_data_size;
    g_stub.data.assign(size, std::byte{0});
    if (size != 0u && data != nullptr) {
        std::memcpy(g_stub.data.data(), data, size);
    }
}

VKAPI_ATTR void VKAPI_CALL stub_destroy_template(
    VkDevice, VkDescriptorUpdateTemplate,
    const VkAllocationCallbacks *callbacks) noexcept {
    ++g_stub.destroy_calls;
    g_stub.destroy_callbacks.emplace_back(callbacks);
}

// The allocation callbacks the test pretends to be the device's own. Nothing
// calls through them (the stubs never allocate), they only have to be a stable,
// non-null identity so a create/destroy mismatch is observable.
VkAllocationCallbacks g_test_callbacks{};

// Installs the stubs. Returns the API bundle the funnel should use.
[[nodiscard]] DescriptorUpdateTemplateApi install_stubs() noexcept {
    ::vkUpdateDescriptorSets = stub_update_descriptor_sets;
    ::vkCreateDescriptorUpdateTemplate = stub_create_template;
    ::vkUpdateDescriptorSetWithTemplate = stub_update_with_template;
    ::vkDestroyDescriptorUpdateTemplate = stub_destroy_template;
    return DescriptorUpdateTemplateApi{
        .create = stub_create_template,
        .destroy = stub_destroy_template,
        .update = stub_update_with_template,
        .callbacks = &g_test_callbacks};
}

// --- write-list helpers ---------------------------------------------------
// luisa::span is std::span or eastl::span depending on the build, so every list
// is converted through this explicit (pointer, count) form.
template<size_t N>
[[nodiscard]] luisa::span<const VkWriteDescriptorSet>
as_writes(const std::array<VkWriteDescriptorSet, N> &writes) noexcept {
    return luisa::span<const VkWriteDescriptorSet>{writes.data(), writes.size()};
}

[[nodiscard]] VkWriteDescriptorSet
make_buffer_write(VkDescriptorSet set, uint32_t binding, uint32_t array_element,
                  const VkDescriptorBufferInfo *infos, uint32_t count,
                  VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) noexcept {
    return VkWriteDescriptorSet{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        set,
        binding,
        array_element,
        count,
        type,
        nullptr,
        infos,
        nullptr};
}

[[nodiscard]] VkWriteDescriptorSet
make_image_write(VkDescriptorSet set, uint32_t binding,
                 const VkDescriptorImageInfo *infos, uint32_t count,
                 VkDescriptorType type) noexcept {
    return VkWriteDescriptorSet{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        set,
        binding,
        0u,
        count,
        type,
        infos,
        nullptr,
        nullptr};
}

[[nodiscard]] bool bytes_equal(const std::byte *packed, const void *expected,
                               size_t size) noexcept {
    return std::memcmp(packed, expected, size) == 0;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_descriptor_update_plan_buffer_only"_test = [] {
        auto set = fake_set(0u);
        VkDescriptorBufferInfo info{fake_buffer(1u), 64u, 256u};
        std::array writes{make_buffer_write(set, 0u, 0u, &info, 1u)};
        DescriptorUpdatePlan plan;
        expect(plan_descriptor_update(as_writes(writes), plan) == DescriptorUpdatePlanStatus::kOk);
        expect(plan.entries.size() == 1u);
        auto const &entry = plan.entries.front();
        expect(entry.dstBinding == 0u);
        expect(entry.dstArrayElement == 0u);
        expect(entry.descriptorCount == 1u);
        expect(entry.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        expect(entry.offset == 0u);
        expect(entry.stride == sizeof(VkDescriptorBufferInfo))
            << "the stride of a buffer-info payload is its own size";
        expect(plan.data_size == sizeof(VkDescriptorBufferInfo));
        expect(descriptor_update_plan_matches(plan, as_writes(writes)));
        std::array<std::byte, sizeof(VkDescriptorBufferInfo)> packed{};
        pack_descriptor_update(plan, as_writes(writes), packed.data());
        expect(bytes_equal(packed.data(), &info, sizeof(info)))
            << "the packed payload must be the exact VkDescriptorBufferInfo";
    };

    "vk_descriptor_update_plan_mixed_buffer_and_images"_test = [] {
        auto set = fake_set(2u);
        VkDescriptorBufferInfo uniform{fake_buffer(3u), 0u, 128u};
        VkDescriptorImageInfo sampled{
            VkSampler{nullptr}, fake_image_view(4u),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo storage{
            VkSampler{nullptr}, fake_image_view(5u),
            VK_IMAGE_LAYOUT_GENERAL};
        std::array writes{
            make_buffer_write(set, 1u, 0u, &uniform, 1u,
                              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
            make_image_write(set, 2u, &sampled, 1u,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            make_image_write(set, 3u, &storage, 1u,
                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)};
        DescriptorUpdatePlan plan;
        expect(plan_descriptor_update(as_writes(writes), plan) == DescriptorUpdatePlanStatus::kOk);
        expect(plan.entries.size() == 3u);
        for (auto const &entry : plan.entries) {
            expect(entry.stride != 0u);
            expect(entry.offset % entry.stride == 0u)
                << "VkDescriptorUpdateTemplateEntry requires offset to be a "
                   "multiple of the payload structure's size";
        }
        expect(plan.entries[0u].offset == 0u);
        // Both payload structures are 24 bytes, so the mixed list stays
        // contiguous and each entry is 24-aligned as well.
        expect(plan.data_size == 3u * sizeof(VkDescriptorBufferInfo));
        std::array<std::byte, 3u * sizeof(VkDescriptorBufferInfo)> packed{};
        pack_descriptor_update(plan, as_writes(writes), packed.data());
        expect(bytes_equal(packed.data() + plan.entries[0u].offset, &uniform,
                           sizeof(uniform)));
        expect(bytes_equal(packed.data() + plan.entries[1u].offset, &sampled,
                           sizeof(sampled)));
        expect(bytes_equal(packed.data() + plan.entries[2u].offset, &storage,
                           sizeof(storage)))
            << "every image payload must land at its own entry's offset";
    };

    "vk_descriptor_update_plan_texel_buffer_alignment"_test = [] {
        // A VkBufferView payload is 8 bytes, so a following 24-byte
        // VkDescriptorBufferInfo must be padded up to the next multiple of 24 -
        // never left at an offset that is only 8-aligned.
        auto set = fake_set(6u);
        VkDescriptorBufferInfo first{fake_buffer(7u), 0u, 64u};
        VkBufferView view = reinterpret_cast<VkBufferView>(&g_handles.slots[8u]);
        VkDescriptorBufferInfo second{fake_buffer(9u), 32u, 96u};
        std::array writes{
            make_buffer_write(set, 0u, 0u, &first, 1u),
            VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1u, 0u,
                1u, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, nullptr, nullptr,
                &view},
            make_buffer_write(set, 2u, 0u, &second, 1u)};
        DescriptorUpdatePlan plan;
        expect(plan_descriptor_update(as_writes(writes), plan) == DescriptorUpdatePlanStatus::kOk);
        expect(plan.entries.size() == 3u);
        expect(plan.entries[1u].stride == sizeof(VkBufferView));
        expect(plan.entries[2u].offset == 2u * sizeof(VkDescriptorBufferInfo))
            << "the buffer-info payload after an 8-byte handle payload must be "
               "rounded up to a multiple of its own size";
        expect(plan.entries[2u].offset % plan.entries[2u].stride == 0u);
        expect(plan.data_size ==
               plan.entries[2u].offset + sizeof(VkDescriptorBufferInfo));
        std::array<std::byte, 128> packed{};
        pack_descriptor_update(plan, as_writes(writes), packed.data());
        expect(bytes_equal(packed.data() + plan.entries[1u].offset, &view,
                           sizeof(view)));
        expect(bytes_equal(packed.data() + plan.entries[2u].offset, &second,
                           sizeof(second)));
    };

    "vk_descriptor_update_plan_multi_descriptor_write"_test = [] {
        auto set = fake_set(10u);
        std::array<VkDescriptorBufferInfo, 3> infos{
            VkDescriptorBufferInfo{fake_buffer(11u), 0u, 16u},
            VkDescriptorBufferInfo{fake_buffer(12u), 16u, 16u},
            VkDescriptorBufferInfo{fake_buffer(13u), 32u, 16u}};
        std::array writes{
            make_buffer_write(set, 4u, 2u, infos.data(), 3u)};
        DescriptorUpdatePlan plan;
        expect(plan_descriptor_update(as_writes(writes), plan) == DescriptorUpdatePlanStatus::kOk);
        auto const &entry = plan.entries.front();
        expect(entry.dstArrayElement == 2u);
        expect(entry.descriptorCount == 3u);
        expect(entry.stride == sizeof(VkDescriptorBufferInfo));
        expect(plan.data_size == 3u * sizeof(VkDescriptorBufferInfo));
        std::array<std::byte, 3u * sizeof(VkDescriptorBufferInfo)> packed{};
        pack_descriptor_update(plan, as_writes(writes), packed.data());
        for (size_t i = 0u; i < infos.size(); ++i) {
            expect(bytes_equal(packed.data() + i * sizeof(VkDescriptorBufferInfo),
                               &infos[i], sizeof(VkDescriptorBufferInfo)))
                << "descriptor " << i << " of a multi-descriptor write";
        }
    };

    "vk_descriptor_update_plan_rejects_unrepresentable_shapes"_test = [] {
        auto set = fake_set(0u);
        VkDescriptorBufferInfo info{fake_buffer(1u), 0u, 16u};
        DescriptorUpdatePlan plan;

        // A chained write (what the acceleration-structure descriptor always is)
        // has no template representation this backend can fill safely: the plan
        // is refused and left empty so the caller keeps vkUpdateDescriptorSets.
        auto chained = make_buffer_write(set, 0u, 0u, &info, 1u);
        VkWriteDescriptorSetAccelerationStructureKHR accel{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            nullptr, 1u, nullptr};
        chained.pNext = &accel;
        std::array chained_writes{chained};
        expect(plan_descriptor_update(as_writes(chained_writes), plan) ==
               DescriptorUpdatePlanStatus::kChainedWrite);
        expect(plan.entries.empty() && plan.data_size == 0u)
            << "a rejected plan must be empty, never half-built";

        auto input = make_buffer_write(set, 0u, 0u, &info, 1u,
                                       VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        std::array unsupported{input};
        expect(plan_descriptor_update(as_writes(unsupported), plan) ==
               DescriptorUpdatePlanStatus::kUnsupportedDescriptorType);
        expect(plan.entries.empty());

        auto null_payload = make_buffer_write(set, 0u, 0u, nullptr, 1u);
        std::array null_writes{null_payload};
        expect(plan_descriptor_update(as_writes(null_writes), plan) ==
               DescriptorUpdatePlanStatus::kNullPayload);

        auto empty_count = make_buffer_write(set, 0u, 0u, &info, 0u);
        std::array empty_writes{empty_count};
        expect(plan_descriptor_update(as_writes(empty_writes), plan) ==
               DescriptorUpdatePlanStatus::kEmptyDescriptorCount);

        std::array two_sets{
            make_buffer_write(set, 0u, 0u, &info, 1u),
            make_buffer_write(fake_set(1u), 1u, 0u, &info, 1u)};
        expect(plan_descriptor_update(as_writes(two_sets), plan) ==
               DescriptorUpdatePlanStatus::kMultipleDescriptorSets);

        std::array<VkDescriptorBufferInfo, 2> contiguous{
            VkDescriptorBufferInfo{fake_buffer(2u), 0u, 16u},
            VkDescriptorBufferInfo{fake_buffer(3u), 16u, 16u}};
        std::array overlapping{
            make_buffer_write(set, 0u, 0u, contiguous.data(), 2u),
            make_buffer_write(set, 0u, 1u, contiguous.data(), 1u)};
        expect(plan_descriptor_update(as_writes(overlapping), plan) ==
               DescriptorUpdatePlanStatus::kOverlappingWrites)
            << "a template has no defined order, so overlapping ranges are "
               "refused instead of being applied in write order";

        // Distinct array elements of one binding do not overlap.
        std::array disjoint{
            make_buffer_write(set, 0u, 0u, contiguous.data(), 1u),
            make_buffer_write(set, 0u, 1u, contiguous.data() + 1u, 1u)};
        expect(plan_descriptor_update(as_writes(disjoint), plan) ==
               DescriptorUpdatePlanStatus::kOk);

        std::array huge{
            make_buffer_write(set, 0u, 0u, contiguous.data(),
                              std::numeric_limits<uint32_t>::max())};
        expect(plan_descriptor_update(as_writes(huge), plan) ==
               DescriptorUpdatePlanStatus::kPayloadTooLarge);

        std::array<VkWriteDescriptorSet, 0> none{};
        expect(plan_descriptor_update(as_writes(none), plan) ==
               DescriptorUpdatePlanStatus::kNoWrites);
    };

    "vk_descriptor_update_plan_shape_matching"_test = [] {
        auto set = fake_set(0u);
        VkDescriptorBufferInfo first{fake_buffer(1u), 0u, 16u};
        VkDescriptorBufferInfo second{fake_buffer(2u), 16u, 32u};
        std::array writes{
            make_buffer_write(set, 0u, 0u, &first, 1u),
            make_buffer_write(set, 1u, 0u, &second, 1u)};
        DescriptorUpdatePlan plan;
        expect(plan_descriptor_update(as_writes(writes), plan) == DescriptorUpdatePlanStatus::kOk);
        expect(descriptor_update_plan_matches(plan, as_writes(writes)));

        // Same shape, different descriptor *values*: still the same template.
        std::array<VkDescriptorBufferInfo, 2> other{
            VkDescriptorBufferInfo{fake_buffer(3u), 64u, 64u},
            VkDescriptorBufferInfo{fake_buffer(4u), 0u, 8u}};
        std::array same_shape{
            make_buffer_write(set, 0u, 0u, &other[0], 1u),
            make_buffer_write(set, 1u, 0u, &other[1], 1u)};
        expect(descriptor_update_plan_matches(plan, as_writes(same_shape)))
            << "a template is keyed by the write shape, not by the bound "
               "resources";

        std::array different_binding{
            make_buffer_write(set, 0u, 0u, &first, 1u),
            make_buffer_write(set, 2u, 0u, &second, 1u)};
        expect(!descriptor_update_plan_matches(plan, as_writes(different_binding)));
        std::array different_element{
            make_buffer_write(set, 0u, 0u, &first, 1u),
            make_buffer_write(set, 1u, 1u, &second, 1u)};
        expect(!descriptor_update_plan_matches(plan, as_writes(different_element)));
        std::array different_count{
            make_buffer_write(set, 0u, 0u, &first, 1u),
            make_buffer_write(set, 1u, 0u, &second, 1u,
                              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
        expect(!descriptor_update_plan_matches(plan, as_writes(different_count)))
            << "a different descriptor type must not reuse a template";
        std::array shorter{make_buffer_write(set, 0u, 0u, &first, 1u)};
        expect(!descriptor_update_plan_matches(plan, as_writes(shorter)));
        std::array<VkWriteDescriptorSet, 0> empty{};
        expect(!descriptor_update_plan_matches(plan, as_writes(empty)));
    };

    "vk_descriptor_update_template_cache_reuses_and_destroys"_test = [] {
        reset_stub();
        auto api = install_stubs();
        DescriptorUpdateTemplateCache cache;
        auto set0 = fake_set(0u);
        auto set1 = fake_set(1u);
        auto layout = fake_layout(2u);
        std::array<VkDescriptorBufferInfo, 2> payloads{
            VkDescriptorBufferInfo{fake_buffer(3u), 0u, 16u},
            VkDescriptorBufferInfo{fake_buffer(4u), 16u, 32u}};
        std::array writes{make_buffer_write(set0, 0u, 0u, &payloads[0], 1u)};
        DescriptorUpdatePlan plan;
        expect(plan_descriptor_update(as_writes(writes), plan) == DescriptorUpdatePlanStatus::kOk);
        g_stub.expected_data_size = plan.data_size;

        DescriptorSetUpdateRequest request{
            .device = reinterpret_cast<VkDevice>(&g_handles.slots[5u]),
            .set = set0,
            .layout = layout,
            .bind_point = VK_PIPELINE_BIND_POINT_COMPUTE,
            .api = api,
            .allow_templates = true};
        auto first = update_descriptor_set(request, cache, as_writes(writes));
        expect(first.used_template);
        expect(first.created_template) << "the first dispatch creates the template";
        expect(first.status == DescriptorUpdatePlanStatus::kOk);
        expect(g_stub.create_calls == 1u);
        expect(g_stub.update_calls == 1u);
        expect(g_stub.legacy_calls == 0u) << "the template path issues no legacy call";
        expect(g_stub.template_type ==
               VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET);
        expect(g_stub.create_flags == VkDescriptorUpdateTemplateCreateFlags{0u});
        expect(g_stub.create_layout == layout)
            << "the template must be created for the shader's set-0 layout";
        expect(g_stub.entry_count == 1u);
        expect(g_stub.entries[0u].dstBinding == 0u);
        expect(g_stub.entries[0u].descriptorCount == 1u);
        expect(g_stub.entries[0u].descriptorType ==
               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        expect(g_stub.last_set == set0);
        expect(g_stub.last_handle == g_stub.created.front());
        expect(g_stub.data.size() == plan.data_size);
        expect(bytes_equal(g_stub.data.data(), &payloads[0], sizeof(payloads[0])))
            << "the driver must receive the exact payload byte block";

        // Second dispatch of the same shader: a fresh set, the same template.
        request.set = set1;
        std::array second_writes{make_buffer_write(set1, 0u, 0u, &payloads[1], 1u)};
        auto second = update_descriptor_set(request, cache, as_writes(second_writes));
        expect(second.used_template);
        expect(!second.created_template)
            << "an identical write shape must reuse the cached template";
        expect(g_stub.create_calls == 1u);
        expect(g_stub.update_calls == 2u);
        expect(g_stub.last_set == set1);
        expect(g_stub.last_handle == g_stub.created.front());
        expect(bytes_equal(g_stub.data.data(), &payloads[1], sizeof(payloads[1])))
            << "a reused template must carry the new descriptor values";

        expect(g_stub.create_callbacks == &g_test_callbacks)
            << "vkCreateDescriptorUpdateTemplate must receive the device's own "
               "allocation callbacks";
        expect(cache.size() == 1u);
        cache.destroy(api, request.device);
        expect(g_stub.destroy_calls == 1u)
            << "every live template must be destroyed exactly once";
        expect(g_stub.destroy_callbacks.size() == 1u &&
               g_stub.destroy_callbacks.front() == g_stub.create_callbacks)
            << "vkDestroyDescriptorUpdateTemplate must receive the identical "
               "callbacks pointer the create call used";
        expect(cache.size() == 0u);
        cache.destroy(api, request.device);
        expect(g_stub.destroy_calls == 1u) << "a second destroy must be a no-op";
    };

    "vk_descriptor_update_template_cache_per_shape"_test = [] {
        reset_stub();
        auto api = install_stubs();
        DescriptorUpdateTemplateCache cache;
        auto set = fake_set(0u);
        auto layout = fake_layout(1u);
        DescriptorSetUpdateRequest request{
            .device = reinterpret_cast<VkDevice>(&g_handles.slots[6u]),
            .set = set,
            .layout = layout,
            .bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .api = api,
            .allow_templates = true};
        std::array<VkDescriptorBufferInfo, 2> payloads{
            VkDescriptorBufferInfo{fake_buffer(2u), 0u, 16u},
            VkDescriptorBufferInfo{fake_buffer(3u), 16u, 16u}};
        g_stub.expected_data_size = sizeof(VkDescriptorBufferInfo);
        std::array single{make_buffer_write(set, 0u, 0u, &payloads[0], 1u)};
        std::array pair{make_buffer_write(set, 0u, 0u, &payloads[0], 1u),
                        make_buffer_write(set, 1u, 0u, &payloads[1], 1u)};
        expect(update_descriptor_set(request, cache, as_writes(single)).used_template);
        expect(update_descriptor_set(request, cache, as_writes(pair)).used_template);
        expect(update_descriptor_set(request, cache, as_writes(single)).used_template);
        expect(g_stub.create_calls == 2u)
            << "a different write shape needs its own template";
        expect(g_stub.update_calls == 3u);
        expect(cache.size() == 2u);
        cache.destroy(api, request.device);
    };

    "vk_descriptor_update_funnel_legacy_fallback"_test = [] {
        reset_stub();
        auto api = install_stubs();
        DescriptorUpdateTemplateCache cache;
        auto set = fake_set(0u);
        VkDescriptorBufferInfo info{fake_buffer(1u), 0u, 16u};
        std::array writes{make_buffer_write(set, 0u, 0u, &info, 1u)};
        DescriptorSetUpdateRequest request{
            .device = reinterpret_cast<VkDevice>(&g_handles.slots[7u]),
            .set = set,
            .layout = fake_layout(2u),
            .bind_point = VK_PIPELINE_BIND_POINT_COMPUTE,
            .api = api,
            .allow_templates = false};

        // Switched off (VulkanDeviceConfigExt switch / no entry points).
        auto disabled = update_descriptor_set(request, cache, as_writes(writes));
        expect(!disabled.used_template);
        expect(g_stub.legacy_calls == 1u);
        expect(g_stub.legacy_write_count == 1u);
        expect(g_stub.legacy_writes == writes.data())
            << "the legacy call must receive exactly the caller's write list";
        expect(g_stub.legacy_copy_count == 0u);
        expect(g_stub.create_calls == 0u && g_stub.update_calls == 0u)
            << "a disabled device must not touch the template entry points";
        expect(cache.size() == 0u);

        // Allowed, but the shape cannot be represented (a chained write).
        reset_stub();
        request.allow_templates = true;
        auto chained = make_buffer_write(set, 0u, 0u, &info, 1u);
        VkWriteDescriptorSetAccelerationStructureKHR accel{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            nullptr, 1u, nullptr};
        chained.pNext = &accel;
        std::array chained_writes{chained};
        auto fallback = update_descriptor_set(request, cache, as_writes(chained_writes));
        expect(!fallback.used_template);
        expect(fallback.status == DescriptorUpdatePlanStatus::kChainedWrite);
        expect(g_stub.legacy_calls == 1u)
            << "an unrepresentable shape keeps the legacy call";
        expect(g_stub.create_calls == 0u);
        expect(cache.size() == 0u);

        // An empty write list issues nothing at all on either path.
        reset_stub();
        std::array<VkWriteDescriptorSet, 0> none{};
        auto empty = update_descriptor_set(request, cache, as_writes(none));
        expect(empty.status == DescriptorUpdatePlanStatus::kNoWrites);
        expect(!empty.used_template);
        expect(g_stub.legacy_calls == 0u && g_stub.update_calls == 0u);

        // An API bundle without entry points fails closed even when the caller
        // allows templates.
        reset_stub();
        request.api = DescriptorUpdateTemplateApi{};
        auto unavailable = update_descriptor_set(request, cache, as_writes(writes));
        expect(!unavailable.used_template);
        expect(g_stub.legacy_calls == 1u);
        expect(g_stub.create_calls == 0u);
    };

    "vk_descriptor_update_template_reuse_requires_eligibility"_test = [] {
        // Regression for the parent review: a cached shape must not be reused for
        // a write list that the planner would have *refused*. Before this test the
        // comparison looked at binding/element/count/type only, so a shape cached
        // while it had no pNext matched a later chained write and silently dropped
        // the chain.
        reset_stub();
        auto api = install_stubs();
        DescriptorUpdateTemplateCache cache;
        auto set = fake_set(0u);
        auto layout = fake_layout(1u);
        VkDescriptorBufferInfo info{fake_buffer(2u), 0u, 64u};
        std::array plain{make_buffer_write(set, 0u, 0u, &info, 1u)};
        DescriptorUpdatePlan plan;
        expect(plan_descriptor_update(as_writes(plain), plan) ==
               DescriptorUpdatePlanStatus::kOk);
        g_stub.expected_data_size = plan.data_size;
        DescriptorSetUpdateRequest request{
            .device = reinterpret_cast<VkDevice>(&g_handles.slots[3u]),
            .set = set,
            .layout = layout,
            .bind_point = VK_PIPELINE_BIND_POINT_COMPUTE,
            .api = api,
            .allow_templates = true};

        // 1. cache the plain shape
        auto first = update_descriptor_set(request, cache, as_writes(plain));
        expect(first.used_template);
        expect(g_stub.create_calls == 1u && g_stub.update_calls == 1u);
        expect(g_stub.legacy_calls == 0u);
        expect(descriptor_update_plan_matches(plan, as_writes(plain)));

        // 2. the same write, now carrying a pNext chain: the cached template must
        //    be refused and the exact legacy call must happen instead.
        auto chained = make_buffer_write(set, 0u, 0u, &info, 1u);
        VkWriteDescriptorSetAccelerationStructureKHR accel{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            nullptr, 1u, nullptr};
        chained.pNext = &accel;
        std::array chained_writes{chained};
        expect(!descriptor_update_plan_matches(plan, as_writes(chained_writes)))
            << "a cached plan must never match a write that grew a pNext chain";
        auto chained_result = update_descriptor_set(request, cache, as_writes(chained_writes));
        expect(!chained_result.used_template)
            << "a chained write must not be applied through a cached template";
        expect(chained_result.status == DescriptorUpdatePlanStatus::kChainedWrite);
        expect(!chained_result.created_template);
        expect(g_stub.create_calls == 1u && g_stub.update_calls == 1u)
            << "the refused shape must not create or use a template";
        expect(g_stub.legacy_calls == 1u)
            << "the chained write must reach vkUpdateDescriptorSets exactly once";
        expect(g_stub.legacy_writes == chained_writes.data() &&
               g_stub.legacy_write_count == 1u)
            << "and the legacy call must receive the chained write itself";

        // 3. the same write with its payload pointer nulled: same treatment.
        reset_stub();
        g_stub.expected_data_size = plan.data_size;
        auto nulled = make_buffer_write(set, 0u, 0u, nullptr, 1u);
        std::array null_writes{nulled};
        expect(!descriptor_update_plan_matches(plan, as_writes(null_writes)))
            << "a cached plan must never match a write whose payload became null";
        auto null_result = update_descriptor_set(request, cache, as_writes(null_writes));
        expect(!null_result.used_template);
        expect(null_result.status == DescriptorUpdatePlanStatus::kNullPayload);
        expect(g_stub.create_calls == 0u && g_stub.update_calls == 0u);
        expect(g_stub.legacy_calls == 1u);
        expect(g_stub.legacy_writes == null_writes.data());

        // 4. a descriptor type without a spec-defined template payload.
        reset_stub();
        auto input = make_buffer_write(set, 0u, 0u, &info, 1u,
                                       VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        std::array input_writes{input};
        expect(!descriptor_update_plan_matches(plan, as_writes(input_writes)));
        auto input_result = update_descriptor_set(request, cache, as_writes(input_writes));
        expect(!input_result.used_template);
        expect(input_result.status ==
               DescriptorUpdatePlanStatus::kUnsupportedDescriptorType);
        expect(g_stub.legacy_calls == 1u && g_stub.update_calls == 0u);

        // 5. the count is part of the shape too: the same binding with two
        //    descriptors is a different shape, so it gets its own template and the
        //    cached one keeps working afterwards.
        reset_stub();
        std::array<VkDescriptorBufferInfo, 2> pair_infos{
            VkDescriptorBufferInfo{fake_buffer(4u), 0u, 16u},
            VkDescriptorBufferInfo{fake_buffer(5u), 16u, 16u}};
        std::array pair_writes{make_buffer_write(set, 0u, 0u, pair_infos.data(), 2u)};
        DescriptorUpdatePlan pair_plan;
        expect(plan_descriptor_update(as_writes(pair_writes), pair_plan) ==
               DescriptorUpdatePlanStatus::kOk);
        g_stub.expected_data_size = pair_plan.data_size;
        expect(!descriptor_update_plan_matches(plan, as_writes(pair_writes)))
            << "the element count decides how much payload moves, so it is part "
               "of the cached shape";
        auto pair = update_descriptor_set(request, cache, as_writes(pair_writes));
        expect(pair.used_template && pair.created_template);
        expect(g_stub.create_calls == 1u);
        expect(g_stub.update_calls == 1u);
        expect(g_stub.legacy_calls == 0u);
        expect(cache.size() == 2u);

        // 6. the original shape still reuses its own template, i.e. the refusals
        //    did not poison or replace it.
        reset_stub();
        g_stub.expected_data_size = plan.data_size;
        auto again = update_descriptor_set(request, cache, as_writes(plain));
        expect(again.used_template);
        expect(!again.created_template);
        expect(g_stub.create_calls == 0u && g_stub.update_calls == 1u);
        expect(g_stub.legacy_calls == 0u);
        cache.destroy(api, request.device);
    };

    return 0;
}