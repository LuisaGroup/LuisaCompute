// Host test for the Vulkan backend's set-0 push-descriptor path
// (src/backends/vk/push_descriptor.h, plan P2.3).
//
// This test covers:
// - the creation-time decision `push_descriptor_eligibility`: a device without
//   the extension / entry point / a usable maxPushDescriptors keeps classic set 0,
//   an empty set 0 keeps classic (vkCmdPushDescriptorSetKHR requires at least one
//   write), a set-0 descriptor count above maxPushDescriptors keeps classic
//   (VUID-VkDescriptorSetLayoutCreateInfo-flags-00281), and so does a set-0
//   binding with descriptor-binding flags or with a dynamic / inline-uniform
//   descriptor type (VUID-...-00280 / -02208)
// - the descriptor types push descriptors can express, and the ones they cannot
// - the write-chain filter: a pNext chain is accepted only for the
//   acceleration-structure write (the one legal chain of this backend), and a
//   mismatched type is reported as such instead of being handed to the driver
// - the single funnel: exactly one vkCmdPushDescriptorSetKHR with the caller's own
//   write list (pointer, count, set number), nothing recorded when the call is
//   refused, and the descriptor-count limit re-checked against the actual writes
// - the counter contract: the funnel counts exactly the calls it records, and a
// refused call increments nothing. That part is the only one gated by the submit
// profiler's private compile-time macro LUISA_VULKAN_PROFILE_SUBMIT (default 0):
// a default build keeps every functional test below and drops the counter one,
// because the counters it reads do not exist there.
// Everything runs on the host: the entry point is a parameter of the funnel, so
// the test supplies a recording stub and no Vulkan device or driver is involved.
#include "ut/ut.hpp"

#include "push_descriptor.h"

// The counter contract is part of the submit profiler, whose single gate is the
// private compile-time macro LUISA_VULKAN_PROFILE_SUBMIT (default 0 here and in
// src/backends/vk/submit_profile.h). Everything else in this file - eligibility,
// descriptor types, the write-chain filter and the funnel itself - is functional
// and stays compiled in a default build.
#if !defined(LUISA_VULKAN_PROFILE_SUBMIT)
#define LUISA_VULKAN_PROFILE_SUBMIT 0
#endif

#include <array>
#include <cstdint>
#include <vector>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;
#if LUISA_VULKAN_PROFILE_SUBMIT
namespace sp = lc::vk::submit_profile;
#endif

namespace {

// Fake Vulkan handles: distinct addresses are distinct objects for anything that
// only compares or forwards them.
struct FakeHandlePool {
    int slots[16]{};
};
FakeHandlePool g_handles;

template<typename Handle>
[[nodiscard]] Handle fake_handle(uint32_t slot) noexcept {
    return reinterpret_cast<Handle>(&g_handles.slots[slot % 16u]);
}

[[nodiscard]] VkCommandBuffer fake_command_buffer(uint32_t slot) noexcept {
    return fake_handle<VkCommandBuffer>(slot);
}

[[nodiscard]] VkPipelineLayout fake_pipeline_layout(uint32_t slot) noexcept {
    return fake_handle<VkPipelineLayout>(slot);
}

[[nodiscard]] VkBuffer fake_buffer(uint32_t slot) noexcept {
    return fake_handle<VkBuffer>(slot);
}

// --- recording stub for the push entry point --------------------------------
struct StubCalls {
    uint32_t push_calls{0u};
    VkCommandBuffer command_buffer{VK_NULL_HANDLE};
    VkPipelineBindPoint bind_point{VK_PIPELINE_BIND_POINT_COMPUTE};
    VkPipelineLayout layout{VK_NULL_HANDLE};
    uint32_t set{0u};
    uint32_t write_count{0u};
    const VkWriteDescriptorSet *writes{nullptr};
    // A copy of the writes the funnel passed, so the test can compare the fields
    // it promised to forward unchanged.
    std::vector<VkWriteDescriptorSet> copied_writes;
};

StubCalls g_stub;

void reset_stub() noexcept {
    g_stub = StubCalls{};
}

VKAPI_ATTR void VKAPI_CALL stub_push_descriptor_set(
    VkCommandBuffer command_buffer, VkPipelineBindPoint bind_point,
    VkPipelineLayout layout, uint32_t set, uint32_t write_count,
    const VkWriteDescriptorSet *writes) noexcept {
    ++g_stub.push_calls;
    g_stub.command_buffer = command_buffer;
    g_stub.bind_point = bind_point;
    g_stub.layout = layout;
    g_stub.set = set;
    g_stub.write_count = write_count;
    g_stub.writes = writes;
    g_stub.copied_writes.assign(writes, writes + write_count);
}

[[nodiscard]] PushDescriptorApi install_stub() noexcept {
    reset_stub();
    return PushDescriptorApi{.cmd = stub_push_descriptor_set};
}

// --- helpers ----------------------------------------------------------------

[[nodiscard]] VkWriteDescriptorSet make_buffer_write(
    uint32_t binding, uint32_t descriptor_count,
    const VkDescriptorBufferInfo *info,
    void *next = nullptr) noexcept {
    return VkWriteDescriptorSet{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        next,
        VK_NULL_HANDLE,// dstSet is ignored by a push (and never set here)
        binding,
        0u,
        descriptor_count,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr,
        info,
        nullptr};
}

[[nodiscard]] PushDescriptorEligibility eligible_device() noexcept {
    return PushDescriptorEligibility{
        .device_capable = true,
        .max_push_descriptors = 32u,
        .set_binding_count = 3u,
        .set_descriptor_count = 3u,
        .has_binding_flags = false,
        .has_dynamic_or_inline_uniform_binding = false};
}

void test_eligibility() {
    // The device-side gate: no extension, no entry point, no usable limit.
    auto info = eligible_device();
    expect(push_descriptor_eligibility(info) == PushDescriptorStatus::kOk)
        << "a capable device with a small set 0 may push";

    info = eligible_device();
    info.device_capable = false;
    expect(push_descriptor_eligibility(info) ==
           PushDescriptorStatus::kDeviceUnsupported)
        << "a device without the extension must keep classic set 0";

    info = eligible_device();
    info.max_push_descriptors = 0u;
    expect(push_descriptor_eligibility(info) ==
           PushDescriptorStatus::kDeviceUnsupported)
        << "an unreported maxPushDescriptors is not a licence to push";

    // An empty set 0 cannot be pushed: vkCmdPushDescriptorSetKHR requires
    // descriptorWriteCount > 0.
    info = eligible_device();
    info.set_binding_count = 0u;
    info.set_descriptor_count = 0u;
    expect(push_descriptor_eligibility(info) == PushDescriptorStatus::kEmptySet);

    info = eligible_device();
    info.set_binding_count = 1u;
    info.set_descriptor_count = 0u;
    expect(push_descriptor_eligibility(info) == PushDescriptorStatus::kEmptySet)
        << "a binding with descriptorCount 0 is still an unrepresentable push";

    // The limit itself, inclusive at the boundary.
    info = eligible_device();
    info.set_descriptor_count = 32u;
    expect(push_descriptor_eligibility(info) == PushDescriptorStatus::kOk)
        << "maxPushDescriptors is inclusive";
    info.set_descriptor_count = 33u;
    expect(push_descriptor_eligibility(info) ==
           PushDescriptorStatus::kDescriptorCountExceedsLimit)
        << "a set 0 above maxPushDescriptors must keep the classic layout";

    // Binding flags: set 0 is never bindless in this backend, and push
    // descriptors have no "update after bind" notion.
    info = eligible_device();
    info.has_binding_flags = true;
    expect(push_descriptor_eligibility(info) ==
           PushDescriptorStatus::kBindingFlagsNotPushable);

    // Dynamic / inline-uniform bindings are forbidden in a push layout.
    info = eligible_device();
    info.has_dynamic_or_inline_uniform_binding = true;
    expect(push_descriptor_eligibility(info) ==
           PushDescriptorStatus::kDynamicOrInlineUniformBinding);

    // The status names are part of the diagnostic output, so every value must be
    // distinguishable and never empty.
    for (auto status : {PushDescriptorStatus::kOk,
                        PushDescriptorStatus::kDeviceUnsupported,
                        PushDescriptorStatus::kEmptySet,
                        PushDescriptorStatus::kDescriptorCountExceedsLimit,
                        PushDescriptorStatus::kBindingFlagsNotPushable,
                        PushDescriptorStatus::kDynamicOrInlineUniformBinding,
                        PushDescriptorStatus::kNoWrites,
                        PushDescriptorStatus::kUnsupportedDescriptorType,
                        PushDescriptorStatus::kEmptyDescriptorCount,
                        PushDescriptorStatus::kUnsupportedWriteChain,
                        PushDescriptorStatus::kWriteChainMismatch,
                        PushDescriptorStatus::kPushCountExceedsLimit,
                        PushDescriptorStatus::kOverlappingWrites}) {
        auto name = luisa::string_view{push_descriptor_status_name(status)};
        expect(!name.empty() && name != "unknown")
            << "every status needs its own name";
    }
}

void test_descriptor_types() {
    // Everything a DSL set 0 can contain.
    for (auto type : {VK_DESCRIPTOR_TYPE_SAMPLER,
                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                      VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                      VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                      VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                      VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR}) {
        expect(push_descriptor_type_supported(type))
            << "push descriptors must accept every type the backend writes";
    }
    for (auto type : {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
                      VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK}) {
        expect(!push_descriptor_type_supported(type))
            << "a dynamic / inline-uniform descriptor is not pushable";
    }
}

void test_write_chain() {
    VkDescriptorBufferInfo info{fake_buffer(1u), 0u, 64u};
    // No chain: always fine.
    auto plain = make_buffer_write(0u, 1u, &info);
    expect(push_descriptor_detail::write_chain_supported(plain));

    // The one legal chain of this backend: the acceleration-structure write.
    VkWriteDescriptorSetAccelerationStructureKHR accel_write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .pNext = nullptr,
        .accelerationStructureCount = 1u,
        .pAccelerationStructures = nullptr};
    auto accel = plain;
    accel.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    accel.pNext = &accel_write;
    expect(push_descriptor_detail::write_chain_supported(accel))
        << "the hardware acceleration-structure write must be pushable";

    // The same chain on a buffer write does not describe what the chain says.
    auto mismatched = plain;
    mismatched.pNext = &accel_write;
    expect(!push_descriptor_detail::write_chain_supported(mismatched));

    // A chain longer than one node is not a shape this backend produces, and
    // Vulkan reads exactly one structure.
    VkWriteDescriptorSetInlineUniformBlock inline_block{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK,
        .pNext = nullptr,
        .dataSize = 4u,
        .pData = nullptr};
    auto chained = accel;
    accel_write.pNext = &inline_block;
    expect(!push_descriptor_detail::write_chain_supported(chained))
        << "a chain with a second node is refused";

    // An unknown sType is refused instead of being forwarded.
    VkBaseInStructure unknown{.sType = VK_STRUCTURE_TYPE_MAX_ENUM, .pNext = nullptr};
    auto unknown_write = plain;
    unknown_write.pNext = &unknown;
    expect(!push_descriptor_detail::write_chain_supported(unknown_write));
}

void test_funnel() {
    auto api = install_stub();
    VkDescriptorBufferInfo infos[2]{
        {fake_buffer(2u), 0u, 64u},
        {fake_buffer(3u), 16u, 32u}};
    std::array writes{
        make_buffer_write(0u, 1u, &infos[0]),
        make_buffer_write(1u, 1u, &infos[1])};
    auto request = PushDescriptorRecordRequest{
        .command_buffer = fake_command_buffer(1u),
        .bind_point = VK_PIPELINE_BIND_POINT_COMPUTE,
        .layout = fake_pipeline_layout(2u),
        .set = 0u,
        .max_push_descriptors = 32u,
        .api = api,
        .allow_push = true};

    // 1. The happy path records exactly one call with the caller's own list.
    auto result = push_descriptor_set(request, writes);
    expect(result.pushed && result.status == PushDescriptorStatus::kOk);
    expect(g_stub.push_calls == 1u);
    expect(g_stub.command_buffer == request.command_buffer);
    expect(g_stub.bind_point == request.bind_point);
    expect(g_stub.layout == request.layout);
    expect(g_stub.set == 0u);
    expect(g_stub.write_count == 2u);
    expect(g_stub.writes == writes.data())
        << "the write list must be forwarded, not rebuilt";
    expect(g_stub.copied_writes[0].dstBinding == 0u &&
           g_stub.copied_writes[1].dstBinding == 1u)
        << "bindings, counts and payloads must be passed through unchanged";
    expect(g_stub.copied_writes[0].pBufferInfo == &infos[0] &&
           g_stub.copied_writes[1].pBufferInfo == &infos[1]);

    // 2. A push without the switch or without the entry point records nothing.
    reset_stub();
    auto disabled = request;
    disabled.allow_push = false;
    auto off = push_descriptor_set(disabled, writes);
    expect(!off.pushed &&
           off.status == PushDescriptorStatus::kDeviceUnsupported);
    expect(g_stub.push_calls == 0u);

    auto no_entry_point = request;
    no_entry_point.api = PushDescriptorApi{};
    auto none = push_descriptor_set(no_entry_point, writes);
    expect(!none.pushed &&
           none.status == PushDescriptorStatus::kDeviceUnsupported);
    expect(g_stub.push_calls == 0u);

    // 3. An empty list is refused (the entry point's own valid usage).
    auto empty = push_descriptor_set(request, {});
    expect(!empty.pushed && empty.status == PushDescriptorStatus::kNoWrites);
    expect(g_stub.push_calls == 0u);

    // 4. Every write is re-checked, so a shape the layout should never have
    //    produced fails closed instead of reaching the driver.
    auto dynamic_write = make_buffer_write(0u, 1u, &infos[0]);
    dynamic_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    auto dynamic_result = push_descriptor_set(request, std::array{dynamic_write});
    expect(!dynamic_result.pushed &&
           dynamic_result.status ==
               PushDescriptorStatus::kUnsupportedDescriptorType);
    expect(g_stub.push_calls == 0u);

    auto zero_count = make_buffer_write(0u, 0u, &infos[0]);
    auto zero_result = push_descriptor_set(request, std::array{zero_count});
    expect(!zero_result.pushed &&
           zero_result.status == PushDescriptorStatus::kEmptyDescriptorCount);

    auto chained = make_buffer_write(0u, 1u, &infos[0]);
    VkWriteDescriptorSetAccelerationStructureKHR accel_write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .pNext = nullptr,
        .accelerationStructureCount = 1u,
        .pAccelerationStructures = nullptr};
    chained.pNext = &accel_write;
    auto chain_result = push_descriptor_set(request, std::array{chained});
    expect(!chain_result.pushed &&
           chain_result.status == PushDescriptorStatus::kWriteChainMismatch);

    VkBaseInStructure unknown{.sType = VK_STRUCTURE_TYPE_MAX_ENUM, .pNext = nullptr};
    auto unknown_write = make_buffer_write(0u, 1u, &infos[0]);
    unknown_write.pNext = &unknown;
    auto unknown_result = push_descriptor_set(request, std::array{unknown_write});
    expect(!unknown_result.pushed &&
           unknown_result.status ==
               PushDescriptorStatus::kUnsupportedWriteChain);

    // 5. The descriptor-count limit is re-checked on the real write list: 33
    //    descriptors in one push exceed a 32-descriptor device.
    VkDescriptorBufferInfo many_infos[33]{};
    for (auto i = 0u; i < 33u; ++i) {
        many_infos[i] = VkDescriptorBufferInfo{fake_buffer(i), 0u, 16u};
    }
    auto many = make_buffer_write(0u, 33u, many_infos);
    auto many_result = push_descriptor_set(request, std::array{many});
    expect(!many_result.pushed &&
           many_result.status == PushDescriptorStatus::kPushCountExceedsLimit)
        << "a push above maxPushDescriptors must not reach the entry point";
    expect(g_stub.push_calls == 0u);

    // 6. The acceleration-structure write itself goes through, with its chain.
    auto api_again = install_stub();
    VkWriteDescriptorSetAccelerationStructureKHR top_level{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .pNext = nullptr,
        .accelerationStructureCount = 1u,
        .pAccelerationStructures = nullptr};
    auto accel = make_buffer_write(0u, 1u, nullptr);
    accel.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    accel.pNext = &top_level;
    auto accel_request = request;
    accel_request.api = api_again;
    auto accel_result = push_descriptor_set(accel_request, std::array{accel});
    expect(accel_result.pushed &&
           accel_result.status == PushDescriptorStatus::kOk);
    expect(g_stub.push_calls == 1u);
    expect(g_stub.copied_writes[0].pNext == &top_level)
        << "the acceleration-structure chain must reach the call unchanged";

    // 7. A non-zero set number is forwarded as given (the caller binds the global
    //    sets from set + 1 on).
    reset_stub();
    auto set_two = request;
    set_two.set = 2u;
    auto set_two_result = push_descriptor_set(set_two, writes);
    expect(set_two_result.pushed && set_two_result.set == 2u);
    expect(g_stub.set == 2u);

    // 8. A write list that addresses the same descriptor twice with overlapping
    //    element ranges is refused instead of being handed to the driver: a push
    //    bank has no "last write wins" rule, and such a list used to crash inside
    //    the vendor driver (plan P2.3 review).
    reset_stub();
    VkDescriptorBufferInfo overlap_infos[2]{
        {fake_buffer(5u), 0u, 16u},
        {fake_buffer(6u), 0u, 16u}};
    auto first = make_buffer_write(2u, 1u, &overlap_infos[0]);
    auto overlapping = make_buffer_write(2u, 1u, &overlap_infos[1]);
    auto overlap_result = push_descriptor_set(request, std::array{first, overlapping});
    expect(!overlap_result.pushed &&
           overlap_result.status == PushDescriptorStatus::kOverlappingWrites);
    expect(g_stub.push_calls == 0u)
        << "an ambiguous write list must not reach the entry point";
    // Overlapping *array* ranges of one binding are just as ambiguous.
    auto array_a = make_buffer_write(1u, 2u, &overlap_infos[0]);
    array_a.dstArrayElement = 0u;
    auto array_b = make_buffer_write(1u, 2u, &overlap_infos[1]);
    array_b.dstArrayElement = 1u;
    auto array_overlap = push_descriptor_set(request, std::array{array_a, array_b});
    expect(!array_overlap.pushed &&
           array_overlap.status == PushDescriptorStatus::kOverlappingWrites);
    // Disjoint array elements of one binding stay legal.
    array_b.dstArrayElement = 2u;
    auto disjoint = push_descriptor_set(request, std::array{array_a, array_b});
    expect(disjoint.pushed && disjoint.status == PushDescriptorStatus::kOk);
    expect(g_stub.push_calls == 1u && g_stub.write_count == 2u);
    // A list with one write is never rejected by the overlap guard.
    auto single_write = push_descriptor_set(request, std::array{first});
    expect(single_write.pushed);
}

#if LUISA_VULKAN_PROFILE_SUBMIT
// The counter contract of plan P0.2: the funnel increments exactly when it
// records the native call, and a refused call increments nothing. The profiler is
// compiled in (LUISA_VULKAN_PROFILE_SUBMIT=1), so no setup is needed before the
// first call; the counter is read through the thread-local mirror, which needs no
// backend library.
void test_counters() {
    auto api = install_stub();
    VkDescriptorBufferInfo info{fake_buffer(4u), 0u, 32u};
    auto writes = std::array{make_buffer_write(0u, 1u, &info)};
    auto request = PushDescriptorRecordRequest{
        .command_buffer = fake_command_buffer(3u),
        .bind_point = VK_PIPELINE_BIND_POINT_COMPUTE,
        .layout = fake_pipeline_layout(4u),
        .set = 0u,
        .max_push_descriptors = 32u,
        .api = api,
        .allow_push = true};

    auto before = sp::mirror().push_descriptor_calls;
    auto pushed = push_descriptor_set(request, writes);
    expect(pushed.pushed);
    expect(sp::mirror().push_descriptor_calls == before + 1u)
        << "one recorded push is one counted call";

    // A refused call records nothing, so it must not be counted either.
    auto refused = push_descriptor_set(request, {});
    expect(!refused.pushed);
    expect(sp::mirror().push_descriptor_calls == before + 1u)
        << "a refused push is not a recorded call";

    // The shader-decision counters are separate units (shaders, not calls).
    auto shaders = sp::mirror().push_descriptor_shaders;
    auto classic = sp::mirror().push_descriptor_classic_shaders;
    sp::record_push_descriptor_shader(true);
    sp::record_push_descriptor_shader(false);
    expect(sp::mirror().push_descriptor_shaders == shaders + 1u &&
           sp::mirror().push_descriptor_classic_shaders == classic + 1u)
        << "the creation-time decision counts one shader, in exactly one bucket";
}
#endif// LUISA_VULKAN_PROFILE_SUBMIT

}// namespace

int main() {
    "vk_push_descriptor_eligibility"_test = [] { test_eligibility(); };
    "vk_push_descriptor_types"_test = [] { test_descriptor_types(); };
    "vk_push_descriptor_write_chain"_test = [] { test_write_chain(); };
    "vk_push_descriptor_funnel"_test = [] { test_funnel(); };
#if LUISA_VULKAN_PROFILE_SUBMIT
    "vk_push_descriptor_counters"_test = [] { test_counters(); };
#endif// LUISA_VULKAN_PROFILE_SUBMIT
    return 0;
}
