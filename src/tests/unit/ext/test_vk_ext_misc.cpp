// Test for the Vulkan descriptor-heap foundation and the vendor-extension plan.
// This test covers the host-visible planning helpers of the extension group:
// - VK_EXT_descriptor_heap (detail::plan_descriptor_heap /
//   detail::descriptor_slot / detail::plan_push_data and the bindless-array
//   migration planner)
// - VK_KHR_internally_synchronized_queues, VK_ARM_tensors, VK_ARM_data_graph,
//   VK_QCOM_tile_shading, VK_QCOM_tile_memory_heap and
//   VK_EXT_external_memory_metal (the inert planning entries of
//   vk_vendor_extensions_plan.h)
//
// Every planner is exercised for the enabled and for the
// disabled/unsupported/inconsistent case: an optional extension is an
// optimization only, so the "not enabled" answer has to select the pre-existing
// descriptor-set path instead of failing, and a rejected plan must never look
// usable. No GPU is needed and no Vulkan entry point is called.
#include "ut/ut.hpp"
#include "vk_descriptor_heap.h"
#include "vk_vendor_extensions_plan.h"
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <string_view>
using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;

namespace vk_ext_misc_test_detail {

// A representative desktop device: 4 KiB implementation reservations, a 64-byte
// sampler heap alignment, power-of-two descriptor sizes, and no tensor support
// (VK_ARM_tensors is not enabled by default).
inline constexpr DescriptorHeapLimits base_limits{
    .sampler_heap_alignment = 64u,
    .resource_heap_alignment = 256u,
    .max_sampler_heap_size = 1u << 20u,
    .max_resource_heap_size = 1u << 30u,
    .min_sampler_heap_reserved_range = 4096u,
    .min_sampler_heap_reserved_range_with_embedded = 8192u,
    .min_resource_heap_reserved_range = 4096u,
    .sampler_descriptor_size = 32u,
    .image_descriptor_size = 48u,
    .buffer_descriptor_size = 24u,
    .tensor_descriptor_size = 0u,
    .sampler_descriptor_alignment = 32u,
    .image_descriptor_alignment = 16u,
    .buffer_descriptor_alignment = 8u,
    .tensor_descriptor_alignment = 0u,
    .max_push_data_size = 256u,
    .max_descriptor_heap_embedded_samplers = 64u,
    .sparse_descriptor_heaps = false,
    .protected_descriptor_heaps = false};

// The same device with VK_ARM_tensors (and its descriptor-heap tensor properties)
// enabled.
inline constexpr DescriptorHeapLimits tensor_limits{
    .sampler_heap_alignment = 64u,
    .resource_heap_alignment = 256u,
    .max_sampler_heap_size = 1u << 20u,
    .max_resource_heap_size = 1u << 30u,
    .min_sampler_heap_reserved_range = 4096u,
    .min_sampler_heap_reserved_range_with_embedded = 8192u,
    .min_resource_heap_reserved_range = 4096u,
    .sampler_descriptor_size = 32u,
    .image_descriptor_size = 48u,
    .buffer_descriptor_size = 24u,
    .tensor_descriptor_size = 64u,
    .sampler_descriptor_alignment = 32u,
    .image_descriptor_alignment = 16u,
    .buffer_descriptor_alignment = 8u,
    .tensor_descriptor_alignment = 32u,
    .max_push_data_size = 256u,
    .max_descriptor_heap_embedded_samplers = 64u,
    .sparse_descriptor_heaps = true,
    .protected_descriptor_heaps = false};

[[nodiscard]] constexpr DescriptorHeapRequest region_request(
    DescriptorClass klass, uint64_t slots, bool sampler_heap) noexcept {
    DescriptorHeapRequest request{};
    request.extension_enabled = true;
    request.sampler_heap = sampler_heap;
    request.region_count = 1u;
    request.regions[0] = DescriptorHeapRegionRequest{.klass = klass, .slots = slots};
    return request;
}

[[nodiscard]] constexpr DescriptorHeapRequest two_region_request(
    DescriptorClass first, uint64_t first_slots,
    DescriptorClass second, uint64_t second_slots) noexcept {
    DescriptorHeapRequest request{};
    request.extension_enabled = true;
    request.sampler_heap = false;
    request.region_count = 2u;
    request.regions[0] = DescriptorHeapRegionRequest{.klass = first, .slots = first_slots};
    request.regions[1] = DescriptorHeapRegionRequest{.klass = second, .slots = second_slots};
    return request;
}

// An independent restatement of the layout rule the planner implements:
//   heap = [0, reserved) + regions, each region aligned to its own descriptor
//   alignment with a stride of align_up(size, alignment), heap size rounded up to
//   the heap alignment.
struct ExpectedLayout {
    uint64_t reserved{0u};
    uint64_t first_offset{0u};
    uint64_t first_stride{0u};
    uint64_t payload{0u};
    uint64_t heap_size{0u};
};

[[nodiscard]] constexpr ExpectedLayout expected_single_region(
    uint64_t aligned_reserved, uint64_t descriptor_size,
    uint64_t descriptor_alignment, uint64_t slots,
    uint64_t heap_alignment) noexcept {
    auto stride = align_up(descriptor_size, descriptor_alignment);
    auto offset = align_up(aligned_reserved, descriptor_alignment);
    auto payload = offset + slots * stride;
    return ExpectedLayout{.reserved = aligned_reserved,
                          .first_offset = offset,
                          .first_stride = stride,
                          .payload = payload,
                          .heap_size = align_up(payload, heap_alignment)};
}

[[nodiscard]] constexpr bool contains_file_line(const char *text) noexcept {
    if (text == nullptr) { return false; }
    for (auto p = text; *p != '\0'; p++) {
        if (*p != ':') { continue; }
        // A `file:line` reference is what a blocker has to carry; accept the
        // `path.cpp:12` / `path.h:12` shape only so a random ':' cannot pass.
        auto has_file = false;
        for (auto q = p; q != text; q--) {
            if (q[-1] == '.') { has_file = true; }
        }
        auto next = p + 1;
        if (has_file && *next >= '0' && *next <= '9') { return true; }
    }
    return false;
}

}// namespace vk_ext_misc_test_detail

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    using namespace vk_ext_misc_test_detail;

    // -----------------------------------------------------------------------
    // Alignment arithmetic. `vkGetPhysicalDeviceDescriptorSizeEXT` may report
    // non power-of-two sizes, so every rounding step must work without bit
    // masking.
    // -----------------------------------------------------------------------
    "descriptor_heap_alignment_helpers"_test = [] {
        expect(align_up(0u, 64u) == 0u);
        expect(align_up(1u, 64u) == 64u);
        expect(align_up(64u, 64u) == 64u);
        expect(align_up(65u, 64u) == 128u);
        expect(align_up(100u, 1u) == 100u);
        // An alignment of 0 or 1 is a no-op instead of a division by zero.
        expect(align_up(100u, 0u) == 100u);
        // Non power-of-two sizes and alignments.
        expect(align_up(48u, 24u) == 48u);
        expect(align_up(50u, 24u) == 72u);
        expect(align_up(4096u, 12u) == 4104u);
        expect(align_up(21u, 12u) == 24u);
        // Saturating instead of wrapping: an overflowing request must never turn
        // into a small "legal" size.
        expect(align_up(std::numeric_limits<uint64_t>::max(), 8u) ==
               std::numeric_limits<uint64_t>::max());
        expect(align_up(std::numeric_limits<uint64_t>::max() - 1u, 8u) ==
               std::numeric_limits<uint64_t>::max());
        expect(align_down(0u, 64u) == 0u);
        expect(align_down(65u, 64u) == 64u);
        expect(align_down(64u, 64u) == 64u);
        expect(align_down(4096u, 12u) == 4092u);
        expect(align_down(100u, 0u) == 100u);
        // Per-class sizes and strides of the representative device.
        expect(descriptor_size(base_limits, DescriptorClass::kBuffer) == 24u);
        expect(descriptor_stride(base_limits, DescriptorClass::kBuffer) == 24u);
        expect(descriptor_stride(base_limits, DescriptorClass::kImage) == 48u);
        expect(descriptor_stride(base_limits, DescriptorClass::kSampler) == 32u);
        expect(descriptor_stride(base_limits, DescriptorClass::kTensor) == 0u);
        expect(descriptor_stride(tensor_limits, DescriptorClass::kTensor) == 64u);
        // A non power-of-two descriptor size is rounded up to its alignment.
        auto odd = base_limits;
        odd.buffer_descriptor_size = 20u;
        odd.buffer_descriptor_alignment = 12u;
        expect(descriptor_stride(odd, DescriptorClass::kBuffer) == 24u);
        expect(descriptor_slot_capacity(0u, 24u) == 0u);
        expect(descriptor_slot_capacity(24u, 0u) == 0u);
        expect(descriptor_slot_capacity(23u, 24u) == 0u);
        expect(descriptor_slot_capacity(24u, 24u) == 1u);
        expect(descriptor_slot_capacity(47u, 24u) == 1u);
        expect(descriptor_slot_capacity(48u, 24u) == 2u);
    };

    // -----------------------------------------------------------------------
    // The limits mirror of VkPhysicalDeviceDescriptorHeapPropertiesEXT.
    // -----------------------------------------------------------------------
    "descriptor_heap_limit_mirror"_test = [] {
        VkPhysicalDeviceDescriptorHeapPropertiesEXT properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
            .pNext = nullptr,
            .samplerHeapAlignment = 128u,
            .resourceHeapAlignment = 512u,
            .maxSamplerHeapSize = 4096u,
            .maxResourceHeapSize = 8192u,
            .minSamplerHeapReservedRange = 256u,
            .minSamplerHeapReservedRangeWithEmbedded = 1024u,
            .minResourceHeapReservedRange = 512u,
            .samplerDescriptorSize = 16u,
            .imageDescriptorSize = 32u,
            .bufferDescriptorSize = 64u,
            .samplerDescriptorAlignment = 8u,
            .imageDescriptorAlignment = 16u,
            .bufferDescriptorAlignment = 32u,
            .maxPushDataSize = 128u,
            .imageCaptureReplayOpaqueDataSize = 96u,
            .maxDescriptorHeapEmbeddedSamplers = 4u,
            .samplerYcbcrConversionCount = 8u,
            .sparseDescriptorHeaps = VK_FALSE,
            .protectedDescriptorHeaps = VK_TRUE};
        auto limits = descriptor_heap_limits(properties);
        expect(limits.sampler_heap_alignment == 128u);
        expect(limits.resource_heap_alignment == 512u);
        expect(limits.max_sampler_heap_size == 4096u);
        expect(limits.max_resource_heap_size == 8192u);
        expect(limits.min_sampler_heap_reserved_range == 256u);
        expect(limits.min_sampler_heap_reserved_range_with_embedded == 1024u);
        expect(limits.min_resource_heap_reserved_range == 512u);
        expect(limits.sampler_descriptor_size == 16u);
        expect(limits.image_descriptor_size == 32u);
        expect(limits.buffer_descriptor_size == 64u);
        expect(limits.sampler_descriptor_alignment == 8u);
        expect(limits.image_descriptor_alignment == 16u);
        expect(limits.buffer_descriptor_alignment == 32u);
        expect(limits.max_push_data_size == 128u);
        expect(limits.max_descriptor_heap_embedded_samplers == 4u);
        expect(!limits.sparse_descriptor_heaps);
        expect(limits.protected_descriptor_heaps);
        // VK_ARM_tensors publishes its own descriptor properties; without them the
        // tensor class has no size and cannot be planned.
        expect(limits.tensor_descriptor_size == 0u);
        expect(!plan_descriptor_heap(limits, region_request(DescriptorClass::kTensor, 1u, false)).usable);
        VkPhysicalDeviceDescriptorHeapTensorPropertiesARM tensor_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_TENSOR_PROPERTIES_ARM,
            .pNext = nullptr,
            .tensorDescriptorSize = 96u,
            .tensorDescriptorAlignment = 64u,
            .tensorCaptureReplayOpaqueDataSize = 0u};
        auto with_tensors = descriptor_heap_limits(properties, tensor_properties);
        expect(with_tensors.tensor_descriptor_size == 96u);
        expect(with_tensors.tensor_descriptor_alignment == 64u);
        expect(descriptor_stride(with_tensors, DescriptorClass::kTensor) == 128u);
        expect(plan_descriptor_heap(with_tensors,
                                    region_request(DescriptorClass::kTensor, 1u, false))
                   .usable);
    };

    // -----------------------------------------------------------------------
    // Resource-heap layout: alignment rounding of the reserved range, of every
    // region start and of the final heap size.
    // -----------------------------------------------------------------------
    "descriptor_heap_resource_layout"_test = [] {
        auto request = two_region_request(DescriptorClass::kBuffer, 1000u,
                                          DescriptorClass::kImage, 500u);
        auto plan = plan_descriptor_heap(base_limits, request);
        expect(plan.error == DescriptorHeapError::kNone);
        expect(plan.usable);
        expect(!plan.sampler_heap);
        // `descriptorHeapCaptureReplay` is never enabled, so no replay data is ever
        // produced - not even for an accepted plan.
        expect(!plan.capture_replay_structs);
        expect(plan.heap_alignment == 256u);
        expect(plan.heap_size_limit == (1u << 30u));
        expect(plan.payload_budget == (1u << 30u));
        expect(plan.reserved_range_offset == 0u);
        expect(plan.reserved_range_size == 4096u);
        expect(plan.region_count == 2u);
        // Region 0: buffers at the reserved range, stride 24.
        expect(plan.regions[0].klass == DescriptorClass::kBuffer);
        expect(plan.regions[0].first_slot == 0u);
        expect(plan.regions[0].slots == 1000u);
        expect(plan.regions[0].descriptor_size == 24u);
        expect(plan.regions[0].descriptor_alignment == 8u);
        expect(plan.regions[0].descriptor_stride == 24u);
        expect(plan.regions[0].byte_offset == 4096u);
        expect(plan.regions[0].fitting_slots == 1000u);
        // Region 1: images directly after the buffer region, stride 48.
        expect(plan.regions[1].klass == DescriptorClass::kImage);
        expect(plan.regions[1].first_slot == 1000u);
        expect(plan.regions[1].descriptor_stride == 48u);
        expect(plan.regions[1].byte_offset == 4096u + 1000u * 24u);
        expect(plan.regions[1].fitting_slots == 500u);
        auto payload = 4096u + 1000u * 24u + 500u * 48u;
        expect(plan.payload_bytes == payload);
        expect(plan.heap_size == align_up(payload, 256u));
        expect(plan.heap_size == 52224u);
        expect(plan.heap_size <= plan.heap_size_limit);
        expect(plan.slot_capacity == 1500u);
        // The same numbers through the independent restatement of the rule.
        auto first = expected_single_region(4096u, 24u, 8u, 1000u, 256u);
        expect(first.first_offset == plan.regions[0].byte_offset);
        expect(first.first_stride == plan.regions[0].descriptor_stride);
        // A region start is realigned to its own descriptor alignment: a 20-slot
        // image region after a 1-slot buffer region starts at an 16-byte boundary.
        auto odd = base_limits;
        odd.buffer_descriptor_size = 20u;
        odd.buffer_descriptor_alignment = 12u;
        auto odd_plan = plan_descriptor_heap(odd, two_region_request(
                                                      DescriptorClass::kBuffer, 1u,
                                                      DescriptorClass::kImage, 1u));
        expect(odd_plan.usable);
        expect(odd_plan.regions[0].descriptor_stride == 24u);
        expect(odd_plan.regions[0].byte_offset == 4104u);
        expect(odd_plan.regions[1].byte_offset == 4128u);
        expect(odd_plan.payload_bytes == 4128u + 48u);
        // One slot in a 256-byte-aligned heap still rounds the heap size up.
        auto one_slot = plan_descriptor_heap(base_limits,
                                             region_request(DescriptorClass::kBuffer, 1u, false));
        expect(one_slot.usable);
        expect(one_slot.payload_bytes == 4096u + 24u);
        expect(one_slot.heap_size == 4352u);
    };

    // -----------------------------------------------------------------------
    // Descriptor-offset computation for the four descriptor classes and the
    // sampler/resource heap separation.
    // -----------------------------------------------------------------------
    "descriptor_heap_slot_offsets"_test = [] {
        auto resource_plan = plan_descriptor_heap(
            tensor_limits,
            [&] {
                DescriptorHeapRequest request{};
                request.extension_enabled = true;
                request.region_count = 3u;
                request.regions[0] = DescriptorHeapRegionRequest{
                    .klass = DescriptorClass::kBuffer, .slots = 4u};
                request.regions[1] = DescriptorHeapRegionRequest{
                    .klass = DescriptorClass::kImage, .slots = 3u};
                request.regions[2] = DescriptorHeapRegionRequest{
                    .klass = DescriptorClass::kTensor, .slots = 2u};
                return request;
            }());
        expect(resource_plan.usable);
        // buffer slots
        auto buffer0 = descriptor_slot(resource_plan, DescriptorClass::kBuffer, 0u);
        expect(buffer0.valid);
        expect(buffer0.byte_offset == 4096u);
        expect(buffer0.descriptor_size == 24u);
        expect(buffer0.descriptor_stride == 24u);
        auto buffer3 = descriptor_slot(resource_plan, DescriptorClass::kBuffer, 3u);
        expect(buffer3.valid);
        expect(buffer3.byte_offset == 4096u + 3u * 24u);
        expect(!descriptor_slot(resource_plan, DescriptorClass::kBuffer, 4u).valid);
        // image slots start after the buffer region, aligned to 16 bytes
        auto image_offset = align_up(4096u + 4u * 24u, 16u);
        auto image0 = descriptor_slot(resource_plan, DescriptorClass::kImage, 0u);
        expect(image0.valid);
        expect(image0.byte_offset == image_offset);
        expect(image0.descriptor_stride == 48u);
        expect(descriptor_slot(resource_plan, DescriptorClass::kImage, 2u).byte_offset ==
               image_offset + 2u * 48u);
        expect(!descriptor_slot(resource_plan, DescriptorClass::kImage, 3u).valid);
        // tensor slots start after the image region, aligned to 32 bytes
        auto tensor_offset = align_up(image_offset + 3u * 48u, 32u);
        auto tensor0 = descriptor_slot(resource_plan, DescriptorClass::kTensor, 0u);
        expect(tensor0.valid);
        expect(tensor0.byte_offset == tensor_offset);
        expect(tensor0.descriptor_size == 64u);
        expect(tensor0.descriptor_stride == 64u);
        expect(descriptor_slot(resource_plan, DescriptorClass::kTensor, 1u).byte_offset ==
               tensor_offset + 64u);
        expect(!descriptor_slot(resource_plan, DescriptorClass::kTensor, 2u).valid);
        // A class with no region at all, and a sampler in the resource heap, are
        // both invalid.
        expect(!descriptor_slot(resource_plan, DescriptorClass::kSampler, 0u).valid);
        // Sampler offsets only exist in the sampler heap and never alias a resource
        // offset: the two heaps are separate VkDeviceAddress ranges.
        auto sampler_plan = plan_descriptor_heap(
            tensor_limits, region_request(DescriptorClass::kSampler, 16u, true));
        expect(sampler_plan.usable);
        expect(sampler_plan.sampler_heap);
        expect(sampler_plan.reserved_range_size == 4096u);
        auto sampler0 = descriptor_slot(sampler_plan, DescriptorClass::kSampler, 0u);
        expect(sampler0.valid);
        expect(sampler0.byte_offset == 4096u);
        expect(sampler0.descriptor_size == 32u);
        expect(sampler0.descriptor_stride == 32u);
        expect(descriptor_slot(sampler_plan, DescriptorClass::kSampler, 15u).byte_offset ==
               4096u + 15u * 32u);
        expect(!descriptor_slot(sampler_plan, DescriptorClass::kSampler, 16u).valid);
        expect(!descriptor_slot(sampler_plan, DescriptorClass::kBuffer, 0u).valid);
        expect(!descriptor_slot(sampler_plan, DescriptorClass::kImage, 0u).valid);
        // The two heaps use their own alignment and maximum.
        expect(sampler_plan.heap_alignment == 64u);
        expect(sampler_plan.heap_size_limit == (1u << 20u));
        expect(sampler_plan.heap_size == 4608u);
        expect(resource_plan.heap_alignment == 256u);
        expect(resource_plan.heap_size_limit == (1u << 30u));
        // A rejected plan answers no slots at all.
        DescriptorHeapRequest disabled{};
        disabled.region_count = 1u;
        disabled.regions[0] = DescriptorHeapRegionRequest{
            .klass = DescriptorClass::kBuffer, .slots = 4u};
        auto rejected = plan_descriptor_heap(tensor_limits, disabled);
        expect(!rejected.usable);
        expect(!descriptor_slot(rejected, DescriptorClass::kBuffer, 0u).valid);
    };

    // -----------------------------------------------------------------------
    // Capacity derived from each limit.
    // -----------------------------------------------------------------------
    "descriptor_heap_capacity_from_limits"_test = [] {
        // maxResourceHeapSize, minus the reserved range, divided by the stride.
        auto small = base_limits;
        small.max_resource_heap_size = 65536u;
        // (65536 - 4096) / 24 = 2560 exactly.
        expect(descriptor_slot_capacity(small.max_resource_heap_size -
                                            small.min_resource_heap_reserved_range,
                                        descriptor_stride(small, DescriptorClass::kBuffer)) == 2560u);
        auto full = plan_descriptor_heap(small, region_request(DescriptorClass::kBuffer, 2560u, false));
        expect(full.usable);
        expect(full.heap_size == 65536u);
        expect(full.slot_capacity == 2560u);
        auto over = plan_descriptor_heap(small, region_request(DescriptorClass::kBuffer, 2561u, false));
        expect(!over.usable);
        expect(over.error == DescriptorHeapError::kHeapTooSmall);
        expect(over.regions[0].fitting_slots == 2560u);
        expect(over.slot_capacity == 2560u);
        // The reported size of a rejected plan is the largest bindable one.
        expect(over.heap_size == 65536u);
        expect(over.heap_size <= over.heap_size_limit);
        // A caller cap below the device maximum reduces the capacity the same way.
        auto capped = small;
        DescriptorHeapRequest capped_request =
            region_request(DescriptorClass::kBuffer, 1194u, false);
        capped_request.heap_size_limit = 32768u;
        auto capped_plan = plan_descriptor_heap(capped, capped_request);
        expect(capped_plan.usable);
        expect(capped_plan.heap_size_limit == 32768u);
        expect(capped_plan.heap_size == 32768u);
        auto capped_over_request =
            region_request(DescriptorClass::kBuffer, 1195u, false);
        capped_over_request.heap_size_limit = 32768u;
        auto capped_over = plan_descriptor_heap(capped, capped_over_request);
        expect(!capped_over.usable);
        expect(capped_over.regions[0].fitting_slots == 1194u);
        // A larger implementation reservation shrinks the capacity.
        auto bigger_reserved = small;
        bigger_reserved.min_resource_heap_reserved_range = 8192u;
        auto bigger_reserved_plan =
            plan_descriptor_heap(bigger_reserved, region_request(DescriptorClass::kBuffer, 2560u, false));
        expect(!bigger_reserved_plan.usable);
        expect(bigger_reserved_plan.regions[0].fitting_slots == 2389u);
        // The heap alignment also costs capacity, because the final size has to be
        // rounded up inside the limit: a 1024-byte-aligned heap can only use the
        // 65536-byte budget below the 66000-byte limit.
        auto aligned = small;
        aligned.max_resource_heap_size = 66000u;
        aligned.resource_heap_alignment = 1024u;
        auto aligned_plan =
            plan_descriptor_heap(aligned, region_request(DescriptorClass::kBuffer, 2587u, false));
        expect(!aligned_plan.usable);
        expect(aligned_plan.payload_budget == 65536u);
        expect(aligned_plan.regions[0].fitting_slots == 2560u);
        expect(plan_descriptor_heap(aligned, region_request(DescriptorClass::kBuffer, 2560u, false)).usable);
        // A limit that is not a multiple of the heap alignment only exposes the
        // rounded-down budget.
        auto unaligned = small;
        auto unaligned_request = region_request(DescriptorClass::kBuffer, 1u, false);
        unaligned_request.heap_size_limit = 5000u;
        auto unaligned_plan = plan_descriptor_heap(unaligned, unaligned_request);
        expect(unaligned_plan.usable);
        expect(unaligned_plan.payload_budget == align_down(5000u, 256u));
        expect(unaligned_plan.payload_budget == 4864u);
        expect(unaligned_plan.heap_size == 4352u);
    };

    // -----------------------------------------------------------------------
    // Zero, oversized and inconsistent inputs.
    // -----------------------------------------------------------------------
    "descriptor_heap_refuses_inconsistent_inputs"_test = [] {
        // 1. The extension is not enabled: the pre-existing descriptor-set path
        //    stays in charge and nothing about the request matters.
        auto disabled = region_request(DescriptorClass::kBuffer, 64u, false);
        disabled.extension_enabled = false;
        auto disabled_plan = plan_descriptor_heap(base_limits, disabled);
        expect(!disabled_plan.usable);
        expect(disabled_plan.error == DescriptorHeapError::kExtensionDisabled);
        expect(disabled_plan.region_count == 0u);
        expect(!disabled_plan.capture_replay_structs);
        // 2. Capture replay is requested. The registry only enables the
        //    `descriptorHeap` feature bit, so a replayable heap is impossible.
        auto replay = region_request(DescriptorClass::kBuffer, 64u, false);
        replay.capture_replay_requested = true;
        auto replay_plan = plan_descriptor_heap(base_limits, replay);
        expect(!replay_plan.usable);
        expect(replay_plan.error == DescriptorHeapError::kCaptureReplayUnsupported);
        expect(!replay_plan.capture_replay_structs);
        // 3. The device reported no limits at all.
        auto no_limits = plan_descriptor_heap(DescriptorHeapLimits{},
                                              region_request(DescriptorClass::kBuffer, 1u, false));
        expect(!no_limits.usable);
        expect(no_limits.error == DescriptorHeapError::kNoDeviceLimits);
        // 4. No region, and a region with zero slots.
        DescriptorHeapRequest empty{};
        empty.extension_enabled = true;
        expect(plan_descriptor_heap(base_limits, empty).error ==
               DescriptorHeapError::kZeroSlots);
        expect(!plan_descriptor_heap(base_limits, empty).usable);
        auto zero_slots = plan_descriptor_heap(base_limits,
                                              region_request(DescriptorClass::kImage, 0u, false));
        expect(!zero_slots.usable);
        expect(zero_slots.error == DescriptorHeapError::kZeroSlots);
        // 5. Too many regions.
        DescriptorHeapRequest too_many{};
        too_many.extension_enabled = true;
        too_many.region_count = kMaxDescriptorHeapRegions + 1u;
        auto too_many_plan = plan_descriptor_heap(base_limits, too_many);
        expect(!too_many_plan.usable);
        expect(too_many_plan.error == DescriptorHeapError::kTooManyRegions);
        // 6. A descriptor class without an advertised size (tensors without
        //    VK_ARM_tensors).
        auto tensor_plan = plan_descriptor_heap(base_limits,
                                               region_request(DescriptorClass::kTensor, 4u, false));
        expect(!tensor_plan.usable);
        expect(tensor_plan.error == DescriptorHeapError::kDescriptorSizeUnavailable);
        // 7. The reserved range alone exceeds the limit.
        auto tiny = base_limits;
        auto tiny_request = region_request(DescriptorClass::kBuffer, 1u, false);
        tiny_request.heap_size_limit = 4096u;
        auto tiny_plan = plan_descriptor_heap(tiny, tiny_request);
        expect(!tiny_plan.usable);
        expect(tiny_plan.error == DescriptorHeapError::kHeapSizeLimitTooSmall);
        // 8. An absurd slot count saturates instead of wrapping, and can therefore
        //    never look legal.
        auto huge = plan_descriptor_heap(
            base_limits,
            region_request(DescriptorClass::kBuffer,
                           std::numeric_limits<uint64_t>::max() / 2u, false));
        expect(!huge.usable);
        expect(huge.error == DescriptorHeapError::kHeapTooSmall);
        expect(huge.payload_bytes == std::numeric_limits<uint64_t>::max());
        expect(huge.heap_size == huge.payload_budget);
        expect(huge.regions[0].fitting_slots <= huge.slot_capacity);
        // 9. Samplers and resources never share a heap.
        auto sampler_in_resource = plan_descriptor_heap(
            base_limits, region_request(DescriptorClass::kSampler, 4u, false));
        expect(!sampler_in_resource.usable);
        expect(sampler_in_resource.error == DescriptorHeapError::kSamplerInResourceHeap);
        auto resource_in_sampler = plan_descriptor_heap(
            base_limits, region_request(DescriptorClass::kBuffer, 4u, true));
        expect(!resource_in_sampler.usable);
        expect(resource_in_sampler.error == DescriptorHeapError::kResourceInSamplerHeap);
        // 10. Embedded samplers only exist in the sampler heap, only when the
        //     device advertises them, and only up to the advertised count.
        auto embedded_in_resource = region_request(DescriptorClass::kBuffer, 4u, false);
        embedded_in_resource.embedded_sampler_count = 1u;
        expect(plan_descriptor_heap(base_limits, embedded_in_resource).error ==
               DescriptorHeapError::kEmbeddedSamplersRequireSamplerHeap);
        auto no_embedded_support = base_limits;
        no_embedded_support.max_descriptor_heap_embedded_samplers = 0u;
        auto embedded_request = region_request(DescriptorClass::kSampler, 4u, true);
        embedded_request.embedded_sampler_count = 1u;
        expect(plan_descriptor_heap(no_embedded_support, embedded_request).error ==
               DescriptorHeapError::kEmbeddedSamplersUnsupported);
        auto too_many_embedded = embedded_request;
        too_many_embedded.embedded_sampler_count =
            base_limits.max_descriptor_heap_embedded_samplers + 1u;
        expect(plan_descriptor_heap(base_limits, too_many_embedded).error ==
               DescriptorHeapError::kTooManyEmbeddedSamplers);
        // Embedding a sampler selects the larger implementation reservation.
        auto embedded = plan_descriptor_heap(base_limits, embedded_request);
        expect(embedded.usable);
        expect(embedded.reserved_range_size == 8192u);
        expect(embedded.reserved_range_size >
               base_limits.min_sampler_heap_reserved_range);
        expect(embedded.regions[0].byte_offset == 8192u);
        expect(embedded.payload_bytes == 8192u + 4u * 32u);
        expect(embedded.heap_size == align_up(embedded.payload_bytes, 64u));
        // ... while the plain sampler heap uses the smaller one.
        expect(plan_descriptor_heap(base_limits,
                                    region_request(DescriptorClass::kSampler, 4u, true))
                   .reserved_range_size == 4096u);
    };

    // -----------------------------------------------------------------------
    // vkCmdPushDataEXT ranges.
    // -----------------------------------------------------------------------
    "descriptor_heap_push_data"_test = [] {
        auto plan = plan_push_data(true, 256u, 0u, 4u);
        expect(plan.valid);
        expect(plan.error == DescriptorHeapError::kNone);
        expect(plan.offset == 0u);
        expect(plan.size == 4u);
        expect(plan_push_data(true, 256u, 252u, 4u).valid);
        expect(plan_push_data(true, 256u, 248u, 8u).valid);
        expect(plan_push_data(false, 256u, 0u, 4u).error ==
               DescriptorHeapError::kExtensionDisabled);
        expect(plan_push_data(true, 0u, 0u, 4u).error ==
               DescriptorHeapError::kZeroPushData);
        expect(plan_push_data(true, 256u, 0u, 0u).error ==
               DescriptorHeapError::kZeroPushData);
        // Both the offset and the size have to be multiples of 4
        // (VUID-VkPushDataInfoEXT-offset-11418/11419).
        expect(plan_push_data(true, 256u, 2u, 4u).error ==
               DescriptorHeapError::kPushDataMisaligned);
        expect(plan_push_data(true, 256u, 0u, 6u).error ==
               DescriptorHeapError::kPushDataMisaligned);
        // offset + size must stay inside maxPushDataSize
        // (VUID-VkPushDataInfoEXT-offset-11243).
        expect(plan_push_data(true, 256u, 252u, 8u).error ==
               DescriptorHeapError::kPushDataOutOfRange);
        expect(plan_push_data(true, 256u, 256u, 4u).error ==
               DescriptorHeapError::kPushDataOutOfRange);
        expect(!plan_push_data(true, 256u, 256u, 4u).valid);
        // The offset is stored in a 32-bit field, so a value beyond it is out of
        // range rather than truncated.
        expect(plan_push_data(true, 1u << 20u, 0x1'0000'0000ull, 4u).error ==
               DescriptorHeapError::kPushDataOutOfRange);
    };

    // -----------------------------------------------------------------------
    // A device profile observed on real hardware (Windows, AMD/Intel class VK
    // 1.4 driver): a non power-of-two implementation reservation, a small heap
    // alignment, and a reservation 128x larger when embedded samplers are used.
    // The full bindless layout of this backend (three arrays of the default
    // 262144-slot capacity plus the immutable sampler set) has to fit inside it.
    // -----------------------------------------------------------------------
    "descriptor_heap_real_device_profile"_test = [] {
        DescriptorHeapLimits limits{
            .sampler_heap_alignment = 32u,
            .resource_heap_alignment = 32u,
            .max_sampler_heap_size = 131072u,
            .max_resource_heap_size = 33554432u,
            .min_sampler_heap_reserved_range = 512u,
            .min_sampler_heap_reserved_range_with_embedded = 65536u,
            .min_resource_heap_reserved_range = 96768u,
            .sampler_descriptor_size = 32u,
            .image_descriptor_size = 32u,
            .buffer_descriptor_size = 16u,
            .tensor_descriptor_size = 0u,
            .sampler_descriptor_alignment = 32u,
            .image_descriptor_alignment = 32u,
            .buffer_descriptor_alignment = 8u,
            .tensor_descriptor_alignment = 0u,
            .max_push_data_size = 256u,
            .max_descriptor_heap_embedded_samplers = 2032u,
            .sparse_descriptor_heaps = true,
            .protected_descriptor_heaps = false};
        // 2 buffer slots per 32 bytes of image stride: (32 MiB - 96768) / 16 is the
        // buffer capacity of a buffer-only heap.
        expect(descriptor_slot_capacity(limits.max_resource_heap_size -
                                            limits.min_resource_heap_reserved_range,
                                        descriptor_stride(limits, DescriptorClass::kBuffer)) == 2091104u);
        // The bindless layout: 262144 buffer slots plus two 262144-slot image
        // arrays.
        auto request = BindlessDescriptorHeapRequest{
            .extension_enabled = true,
            .bindless_capacity = 262144u,
            .sampler_count = 16u,
            .buffer_array = true,
            .tex2d_array = true,
            .tex3d_array = true,
            .acceleration_structures = false,
            .embed_samplers = false};
        auto plan = plan_bindless_descriptor_heap_migration(limits, request);
        expect(plan.possible);
        expect(plan.resource_heap.usable);
        expect(plan.resource_heap.reserved_range_offset == 0u);
        expect(plan.resource_heap.reserved_range_size == 96768u);
        expect(plan.buffer_array.byte_offset == 96768u);
        expect(plan.tex2d_array.byte_offset == 96768u + 262144u * 16u);
        expect(plan.tex3d_array.byte_offset == 96768u + 262144u * 16u + 262144u * 32u);
        expect(plan.resource_heap.payload_bytes ==
               96768u + 262144u * 16u + 2u * 262144u * 32u);
        expect(plan.resource_heap.heap_size == plan.resource_heap.payload_bytes);
        expect(plan.resource_heap.heap_size <= limits.max_resource_heap_size);
        expect(plan.sampler_heap.usable);
        expect(plan.sampler_heap.reserved_range_size == 512u);
        expect(plan.sampler_heap.heap_size == 1024u);
        // Embedding the 16 samplers is legal but costs the 64 KiB reservation. The
        // embedded descriptors live inside the reserved range, so the heap carries
        // no SAMPLER region and its payload *is* the reservation.
        auto embedded = request;
        embedded.embed_samplers = true;
        auto embedded_plan = plan_bindless_descriptor_heap_migration(limits, embedded);
        expect(embedded_plan.possible);
        expect(embedded_plan.sampler_heap.region_count == 0u);
        expect(embedded_plan.sampler_heap.reserved_range_size == 65536u);
        expect(embedded_plan.sampler_heap.payload_bytes == 65536u);
        expect(embedded_plan.sampler_heap.heap_size == 65536u);
        expect(!embedded_plan.sampler_array.present);
        expect(embedded_plan.sampler_heap.heap_size <= limits.max_sampler_heap_size);
        // The push-data limit of this device is the smallest value the extension
        // allows, so a 32-bit array-base index needs a 4-byte range.
        expect(plan_push_data(true, limits.max_push_data_size, 0u, 4u).valid);
        expect(!plan_push_data(true, limits.max_push_data_size, 0u, 260u).valid);
    };

    // -----------------------------------------------------------------------
    // The first-fit descriptor slot allocator: bump-order on a fresh pool,
    // first-fit reuse of released slots, neighbour merging, and the fail-closed
    // answers for out-of-pool / double releases.
    // -----------------------------------------------------------------------
    "descriptor_heap_slot_pool"_test = [] {
        DescriptorSlotPool pool{100u, 4u};
        expect(pool.first_slot() == 100u);
        expect(pool.capacity() == 4u);
        expect(pool.allocated() == 0u);
        expect(pool.free_range_count() == 1u);
        // A fresh pool behaves like a bump allocator.
        expect(pool.allocate() == 100u);
        expect(pool.allocate() == 101u);
        expect(pool.allocate() == 102u);
        expect(pool.allocate() == 103u);
        expect(pool.allocate() == kInvalidDescriptorSlot);
        expect(pool.allocated() == 4u);
        expect(pool.free_range_count() == 0u);
        // Releasing a slot makes it the next first-fit candidate.
        expect(pool.release(102u));
        expect(pool.allocated() == 3u);
        expect(pool.free_range_count() == 1u);
        expect(pool.allocate() == 102u);
        expect(pool.free_range_count() == 0u);
        // Out-of-pool releases, a slot that is not part of the pool, and a
        // double release are refused.
        expect(!pool.release(99u));
        expect(!pool.release(104u));
        expect(pool.is_allocated(100u));
        expect(pool.is_allocated(102u));
        expect(!pool.is_free(102u));
        expect(!pool.is_allocated(99u) && !pool.is_free(99u));
        // Freeing a run merges the neighbours into one range, and first-fit then
        // takes from the front of that range instead of splitting it.
        expect(pool.release(101u));
        expect(pool.release(102u));
        expect(pool.release(103u));
        expect(pool.free_range_count() == 1u);
        expect(pool.allocate() == 101u);
        expect(pool.allocate() == 102u);
        expect(pool.allocate() == 103u);
        expect(pool.free_range_count() == 0u);
        // A slot released between two free ranges merges both of them.
        DescriptorSlotPool merge_pool{0u, 4u};
        for (uint64_t i = 0u; i < 4u; i++) { expect(merge_pool.allocate() == i); }
        expect(merge_pool.release(0u));
        expect(merge_pool.release(3u));
        expect(merge_pool.release(2u));
        expect(merge_pool.free_range_count() == 2u);
        expect(merge_pool.release(1u));
        expect(merge_pool.free_range_count() == 1u);
        expect(merge_pool.allocate() == 0u);
        // Slot 1 is free, so releasing it again is refused.
        expect(!merge_pool.release(1u));
        // An empty pool allocates nothing.
        DescriptorSlotPool empty{0u, 0u};
        expect(empty.capacity() == 0u);
        expect(empty.allocate() == kInvalidDescriptorSlot);
        expect(!empty.release(0u));
        // A fragmented free set is still represented, and `reset` restores it.
        DescriptorSlotPool fragmented{0u, 8u};
        for (auto i = 0u; i < 8u; i++) { expect(fragmented.allocate() == i); }
        for (auto i = 0u; i < 8u; i += 2u) { expect(fragmented.release(i)); }
        expect(fragmented.free_range_count() == 4u);
        expect(fragmented.allocate() == 0u);
        expect(fragmented.allocate() == 2u);
        fragmented.reset(0u, 8u);
        expect(fragmented.allocated() == 0u);
        expect(fragmented.free_range_count() == 1u);
        expect(fragmented.allocate() == 0u);
        // Releasing a slot that would need a brand new range while the table is full
        // fails closed instead of silently dropping it.
        constexpr uint64_t fragment_slots = 2u * kMaxDescriptorSlotRanges + 2u;
        DescriptorSlotPool exhausted{0u, fragment_slots};
        for (uint64_t i = 0u; i < fragment_slots; i++) {
            expect(exhausted.allocate() == i);
        }
        for (uint32_t i = 0u; i < kMaxDescriptorSlotRanges; i++) {
            // Every second slot up to 126: one isolated range per release, so the
            // table fills up exactly.
            expect(exhausted.release(i * 2u));
        }
        expect(exhausted.free_range_count() == kMaxDescriptorSlotRanges);
        // Slot 128 is isolated (127 and 129 are still allocated), so releasing it
        // would need a 65th range.
        expect(!exhausted.release(fragment_slots - 2u));
        expect(exhausted.free_range_count() == kMaxDescriptorSlotRanges);
        // The slot is still allocated, i.e. the refusal did not lose it.
        expect(exhausted.is_allocated(fragment_slots - 2u));
        // ... while a slot that can merge into an existing range is still accepted.
        expect(exhausted.release(1u));
    };

    // -----------------------------------------------------------------------
    // The bindless-array migration design artifact.
    // -----------------------------------------------------------------------
    "descriptor_heap_bindless_migration"_test = [] {
        auto request = BindlessDescriptorHeapRequest{
            .extension_enabled = true,
            .bindless_capacity = 1024u,
            .sampler_count = 16u,
            .buffer_array = true,
            .tex2d_array = true,
            .tex3d_array = true,
            .acceleration_structures = false,
            .embed_samplers = false,
            .resource_heap_size_limit = 0u,
            .sampler_heap_size_limit = 0u};
        auto plan = plan_bindless_descriptor_heap_migration(base_limits, request);
        expect(plan.possible);
        expect(plan.error == DescriptorHeapError::kNone);
        expect(plan.resource_heap.usable);
        expect(plan.sampler_heap.usable);
        // Resource heap: one buffer region followed by the two image regions.
        expect(plan.resource_heap.region_count == 3u);
        expect(plan.resource_heap.regions[0].klass == DescriptorClass::kBuffer);
        expect(plan.resource_heap.regions[1].klass == DescriptorClass::kImage);
        expect(plan.resource_heap.regions[2].klass == DescriptorClass::kImage);
        // Every array reports its own mapping offset and stride, and none of them
        // overlaps the reserved range.
        expect(plan.buffer_array.present);
        expect(plan.buffer_array.byte_offset == 4096u);
        expect(plan.buffer_array.descriptor_stride == 24u);
        expect(plan.buffer_array.slots == 1024u);
        expect(plan.tex2d_array.present);
        expect(plan.tex2d_array.byte_offset == 4096u + 1024u * 24u);
        expect(plan.tex2d_array.descriptor_stride == 48u);
        expect(plan.tex3d_array.present);
        expect(plan.tex3d_array.byte_offset ==
               plan.tex2d_array.byte_offset + 1024u * 48u);
        expect(plan.tex3d_array.descriptor_stride == 48u);
        expect(!plan.acceleration_structure_array.present);
        // Monotone, non-overlapping, inside the heap.
        expect(plan.buffer_array.byte_offset >= plan.resource_heap.reserved_range_size);
        expect(plan.tex2d_array.byte_offset >=
               plan.buffer_array.byte_offset + plan.buffer_array.slots * plan.buffer_array.descriptor_stride);
        expect(plan.tex3d_array.byte_offset >=
               plan.tex2d_array.byte_offset + plan.tex2d_array.slots * plan.tex2d_array.descriptor_stride);
        expect(plan.tex3d_array.byte_offset + plan.tex3d_array.slots * plan.tex3d_array.descriptor_stride <=
               plan.resource_heap.heap_size);
        // Samplers live in the sampler heap only.
        expect(plan.sampler_array.present);
        expect(plan.sampler_array.klass == DescriptorClass::kSampler);
        expect(plan.sampler_array.slots == 16u);
        expect(plan.sampler_array.byte_offset == 4096u);
        expect(plan.sampler_array.descriptor_stride == 32u);
        expect(plan.sampler_heap.regions[0].klass == DescriptorClass::kSampler);
        expect(plan.sampler_heap.heap_size == 4608u);
        // The shader-side requirements are part of the artifact.
        expect(plan.requires_shader_binding_mapping);
        expect(plan.requires_pipeline_flags2);
        expect(plan.requires_heap_bind_commands);
        // Constant-offset mappings need no push data at all.
        expect(plan.required_push_data_bytes == 0u);
        // The blockers name the exact files and lines the follow-up wiring has to
        // change, and every one of them is non-empty.
        expect(bindless_descriptor_heap_blocker_count >= 6u);
        for (auto blocker : bindless_descriptor_heap_blockers) {
            expect(blocker != nullptr);
            expect(std::strlen(blocker) > 0u);
            expect(contains_file_line(blocker));
            expect(std::string_view{blocker}.starts_with("src/backends/vk/"));
        }
        // Embedding the samplers drops the SAMPLER region and uses the larger
        // implementation reservation instead.
        auto embedded = request;
        embedded.embed_samplers = true;
        auto embedded_plan = plan_bindless_descriptor_heap_migration(base_limits, embedded);
        expect(embedded_plan.possible);
        expect(embedded_plan.sampler_heap.usable);
        expect(embedded_plan.sampler_heap.region_count == 0u);
        expect(embedded_plan.sampler_heap.reserved_range_size == 8192u);
        expect(!embedded_plan.sampler_array.present);
        // Acceleration structures get their own buffer region when requested.
        auto with_accel = request;
        with_accel.acceleration_structures = true;
        auto with_accel_plan = plan_bindless_descriptor_heap_migration(base_limits, with_accel);
        expect(with_accel_plan.possible);
        expect(with_accel_plan.resource_heap.region_count == 4u);
        expect(with_accel_plan.acceleration_structure_array.present);
        expect(with_accel_plan.acceleration_structure_array.byte_offset ==
               4096u + 1024u * 24u);
        expect(with_accel_plan.tex2d_array.byte_offset ==
               4096u + 2u * 1024u * 24u);
        // A disabled extension makes the whole migration impossible, and an empty
        // array is not a migration candidate either.
        auto disabled = request;
        disabled.extension_enabled = false;
        auto disabled_plan = plan_bindless_descriptor_heap_migration(base_limits, disabled);
        expect(!disabled_plan.possible);
        expect(disabled_plan.error == DescriptorHeapError::kExtensionDisabled);
        auto no_arrays = request;
        no_arrays.buffer_array = false;
        no_arrays.tex2d_array = false;
        no_arrays.tex3d_array = false;
        auto no_arrays_plan = plan_bindless_descriptor_heap_migration(base_limits, no_arrays);
        expect(!no_arrays_plan.possible);
        expect(no_arrays_plan.error == DescriptorHeapError::kZeroSlots);
        // A capacity the device maximum cannot hold fails as a whole instead of
        // silently shrinking the bindless array. 96 bytes per slot x 2^26 slots is
        // far beyond the 1 GiB resource heap below.
        auto huge = request;
        huge.bindless_capacity = 1u << 26u;
        auto huge_plan = plan_bindless_descriptor_heap_migration(base_limits, huge);
        expect(!huge_plan.possible);
        expect(huge_plan.error == DescriptorHeapError::kHeapTooSmall);
        expect(huge_plan.resource_heap.regions[0].fitting_slots < huge.bindless_capacity);
        expect(huge_plan.resource_heap.regions[0].fitting_slots ==
               (base_limits.max_resource_heap_size -
                base_limits.min_resource_heap_reserved_range) /
                   24u);
    };

    // -----------------------------------------------------------------------
    // Vendor extension plan invariants.
    // -----------------------------------------------------------------------
    "vendor_extension_plan_is_inert"_test = [] {
        static_assert(std::size(kVendorExtensionPlans) ==
                      static_cast<size_t>(VendorExtensionId::kCount));
        expect(kVendorExtensionCount == 6u);
        expect(vendor_extension_plans_are_inert());
        // Every identifier resolves to exactly one entry, in table order.
        for (uint32_t i = 0u; i < kVendorExtensionCount; i++) {
            auto id = static_cast<VendorExtensionId>(i);
            auto *plan = vendor_extension_plan(id);
            expect(plan != nullptr);
            expect(plan->id == id);
            expect(!plan->active);
            expect(!plan->assumed_enabled);
            expect(!plan->reachable_from_backend);
            expect(vendor_extension_plan_is_inert(*plan));
            // The entry names the real extension.
            expect(std::string_view{plan->extension_name} ==
                   std::string_view{vendor_extension_id_name(id)});
            // The blocker carries a `file:line` reference and the unblock
            // requirement is spelled out.
            expect(contains_file_line(plan->blocker));
            expect(std::strlen(plan->unblock_requirement) > 0u);
            expect(std::strlen(plan->entry_points) > 0u);
            expect(std::strlen(plan->features) > 0u);
            // The gate is always `optional_extension_enabled`.
            expect(std::string_view{plan->enabled_gate}.find("optional_extension_enabled") !=
                   std::string_view::npos);
            expect(std::string_view{plan->registry_identifier}.find("OptionalExtension::") !=
                   std::string_view::npos);
        }
        expect(vendor_extension_plan(VendorExtensionId::kCount) == nullptr);
        // The names have to be the ones the loader reports, otherwise the registry
        // entry and the plan would drift apart.
        expect(std::string_view{kVendorExtensionPlans[0].extension_name} ==
               std::string_view{VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME});
        expect(std::string_view{kVendorExtensionPlans[1].extension_name} ==
               std::string_view{VK_ARM_TENSORS_EXTENSION_NAME});
        expect(std::string_view{kVendorExtensionPlans[2].extension_name} ==
               std::string_view{VK_ARM_DATA_GRAPH_EXTENSION_NAME});
        expect(std::string_view{kVendorExtensionPlans[3].extension_name} ==
               std::string_view{VK_QCOM_TILE_SHADING_EXTENSION_NAME});
        expect(std::string_view{kVendorExtensionPlans[4].extension_name} ==
               std::string_view{VK_QCOM_TILE_MEMORY_HEAP_EXTENSION_NAME});
        // VK_EXT_external_memory_metal is platform="metal", so its name macro does
        // not exist in this translation unit; only its structure types and handle
        // bits do.
        expect(std::string_view{kVendorExtensionPlans[5].extension_name} ==
               std::string_view{"VK_EXT_external_memory_metal"});
        expect(VK_STRUCTURE_TYPE_MEMORY_METAL_HANDLE_PROPERTIES_EXT !=
               VK_STRUCTURE_TYPE_MAX_ENUM);
        expect(VK_STRUCTURE_TYPE_IMPORT_MEMORY_METAL_HANDLE_INFO_EXT !=
               VK_STRUCTURE_TYPE_MAX_ENUM);
        expect((VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT |
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT |
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT) != 0u);
        // Each entry's documented feature structure really is a descriptor-heap
        // relevant Vulkan type, i.e. the plan does not invent one.
        expect(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM !=
               VK_STRUCTURE_TYPE_MAX_ENUM);
        expect(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM !=
               VK_STRUCTURE_TYPE_MAX_ENUM);
        expect(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_FEATURES_QCOM !=
               VK_STRUCTURE_TYPE_MAX_ENUM);
        expect(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_MEMORY_HEAP_FEATURES_QCOM !=
               VK_STRUCTURE_TYPE_MAX_ENUM);
        expect(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR !=
               VK_STRUCTURE_TYPE_MAX_ENUM);
        // No entry may claim a fast path: `active` is the flag
        // `Device::mark_optional_extension_active` would accompany.
        for (auto const &plan : kVendorExtensionPlans) {
            expect(!plan.active);
        }
    };

    // -----------------------------------------------------------------------
    // The planner is usable at compile time, which keeps it free of device state
    // and of any runtime dependency on the extension being present.
    // -----------------------------------------------------------------------
    static_assert(plan_descriptor_heap(base_limits,
                                       region_request(DescriptorClass::kBuffer, 1u, false))
                      .usable);
    static_assert(plan_descriptor_heap(base_limits,
                                       region_request(DescriptorClass::kSampler, 1u, true))
                      .usable);
    static_assert(!plan_descriptor_heap(base_limits,
                                        region_request(DescriptorClass::kSampler, 1u, false))
                       .usable);
    static_assert(plan_descriptor_heap(base_limits,
                                       region_request(DescriptorClass::kBuffer, 1u, false))
                      .heap_size == 4352u);
    static_assert(!plan_descriptor_heap(base_limits,
                                        region_request(DescriptorClass::kTensor, 1u, false))
                       .usable);
    static_assert(descriptor_slot(plan_descriptor_heap(
                                     base_limits,
                                     region_request(DescriptorClass::kBuffer, 2u, false)),
                                 DescriptorClass::kBuffer, 1u)
                      .byte_offset == 4096u + 24u);
    static_assert(!descriptor_slot(plan_descriptor_heap(
                                      base_limits,
                                      region_request(DescriptorClass::kBuffer, 2u, false)),
                                  DescriptorClass::kBuffer, 2u)
                       .valid);
    static_assert(plan_push_data(true, 256u, 4u, 4u).valid);
    static_assert(!plan_push_data(true, 256u, 4u, 253u).valid);
    // The slot pool is `constexpr`, so its ordering contract is verified at compile
    // time as well: a fresh pool bumps, an exhausted pool refuses, and only an
    // allocated slot can be released.
    static_assert([] {
        DescriptorSlotPool pool{8u, 2u};
        auto first = pool.allocate();
        auto second = pool.allocate();
        auto exhausted = pool.allocate();
        auto released = pool.release(8u);
        auto double_release = pool.release(8u);
        auto reused = pool.allocate();
        auto foreign = pool.release(10u);
        return first == 8u && second == 9u &&
               exhausted == kInvalidDescriptorSlot && released &&
               !double_release && reused == 8u && !foreign;
    }());
    static_assert(vendor_extension_plans_are_inert());
    static_assert(align_up(4096u, 12u) == 4104u);
    // The bindable device address range and the heap bind info the class records.
    static_assert(VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT != VK_STRUCTURE_TYPE_MAX_ENUM);
    static_assert(VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT != VK_STRUCTURE_TYPE_MAX_ENUM);
    static_assert(VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT != VK_STRUCTURE_TYPE_MAX_ENUM);
    // The 64-bit descriptor-heap usage bit is the 32-bit one, so a heap buffer can
    // be created through the 32-bit VkBufferUsageFlags mask.
    static_assert(static_cast<uint64_t>(VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT) ==
                  static_cast<uint64_t>(VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT));
    static_assert(sizeof(VkHostAddressRangeEXT) >= sizeof(void *) + sizeof(size_t));

    return 0;
}
