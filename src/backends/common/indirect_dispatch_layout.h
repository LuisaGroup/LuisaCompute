#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace lc {

// Vulkan's native XIR-to-SPIR-V path uses the same uint32 source ABI as the
// shared HLSL/DX path:
//
//   word 0: authored dispatch count relative to the host-selected record offset
//   record i:
//     logical dispatch size xyz, kernel id, physical group count xyz
//
// CUDA, HIP, and Metal retain backend-private source representations and are
// deliberately not described here. Vulkan materializes
// VkDispatchIndirectCommand records into private scratch storage. The logical
// size is authoritative there: preparation recomputes physical groups for the
// consuming shader's block size, so a writer cannot silently under-dispatch a
// target with a different block size. The source record stride is therefore not
// tied to Vulkan's tightly packed 12-byte command structure.
struct IndirectDispatchLayout {
#define LC_INDIRECT_LAYOUT(shader_name, cpp_name, value) \
    static constexpr uint32_t cpp_name = value;
#include "indirect_dispatch_layout.def"
#undef LC_INDIRECT_LAYOUT
    static constexpr size_t word_size = sizeof(uint32_t);
    static constexpr size_t header_size = header_word_count * word_size;
    static constexpr size_t record_size = record_word_count * word_size;
    static constexpr size_t vulkan_command_size =
        command_word_count * word_size;

    [[nodiscard]] static constexpr size_t record_word_offset(
        uint32_t record_index) noexcept {
        return header_word_count +
               static_cast<size_t>(record_index) * record_word_count;
    }

    [[nodiscard]] static constexpr bool try_total_size(
        size_t capacity, size_t &size) noexcept {
        if (capacity >
            (std::numeric_limits<size_t>::max() - header_size) /
                record_size) {
            return false;
        }
        size = header_size + capacity * record_size;
        return true;
    }
};

enum class IndirectDispatchPlanError : uint8_t {
    NONE,
    CAPACITY_EXCEEDS_UINT32,
    OFFSET_OUT_OF_RANGE,
    SCRATCH_SIZE_OVERFLOW
};

struct IndirectDispatchPlan {
    uint32_t source_record_offset{};
    uint32_t command_count{};
    size_t scratch_size_bytes{};
};

struct IndirectDispatchPlanResult {
    IndirectDispatchPlan plan{};
    IndirectDispatchPlanError error{IndirectDispatchPlanError::NONE};
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return error == IndirectDispatchPlanError::NONE;
    }
};

struct IndirectDispatchGroupCount {
    uint32_t value{};
    bool valid_block_size{};
};

[[nodiscard]] constexpr IndirectDispatchGroupCount
indirect_dispatch_group_count(uint32_t logical_size,
                              uint32_t block_size) noexcept {
    if (block_size == 0u) { return {}; }
    return {
        .value = logical_size / block_size +
                 (logical_size % block_size != 0u),
        .valid_block_size = true};
}

[[nodiscard]] constexpr uint32_t
indirect_dispatch_max_group_count_for_uint32_global_id(
    uint32_t block_size) noexcept {
    if (block_size == 0u) { return 0u; }
    auto limit =
        (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u) /
        block_size;
    return limit > std::numeric_limits<uint32_t>::max() ?
               std::numeric_limits<uint32_t>::max() :
               static_cast<uint32_t>(limit);
}

[[nodiscard]] constexpr IndirectDispatchPlanResult plan_indirect_dispatch(
    size_t capacity, uint32_t offset, uint32_t maximum_count) noexcept {
    if (capacity > std::numeric_limits<uint32_t>::max()) {
        return {.error =
                    IndirectDispatchPlanError::CAPACITY_EXCEEDS_UINT32};
    }
    if (offset > capacity) {
        return {.error = IndirectDispatchPlanError::OFFSET_OUT_OF_RANGE};
    }
    auto remaining = static_cast<uint32_t>(capacity - offset);
    auto count = maximum_count < remaining ? maximum_count : remaining;
    auto count_size = static_cast<size_t>(count);
    if (count_size != 0u &&
        IndirectDispatchLayout::vulkan_command_size >
            std::numeric_limits<size_t>::max() / count_size) {
        return {.error =
                    IndirectDispatchPlanError::SCRATCH_SIZE_OVERFLOW};
    }
    return {
        .plan = {
            .source_record_offset = offset,
            .command_count = count,
            .scratch_size_bytes =
                count_size *
                IndirectDispatchLayout::vulkan_command_size}};
}

enum class IndirectDispatchMode : uint32_t {
    DIRECT = 0u,
    INDIRECT = 1u
};

struct alignas(16) IndirectDispatchPushConstants {
    uint32_t logical_size_x{};
    uint32_t logical_size_y{};
    uint32_t logical_size_z{};
    uint32_t kernel_id{};
    uint32_t mode{};
    uint32_t source_record_index{};
    uint32_t reserved_0{};
    uint32_t reserved_1{};
};
static_assert(sizeof(IndirectDispatchPushConstants) == 32u);

struct alignas(16) IndirectDispatchPrepareConstants {
    uint32_t command_count{};
    uint32_t source_record_offset{};
    uint32_t target_block_size_x{};
    uint32_t target_block_size_y{};
    uint32_t target_block_size_z{};
    uint32_t max_group_count_x{};
    uint32_t max_group_count_y{};
    uint32_t max_group_count_z{};
    uint32_t command_base{};
    uint32_t reserved_0{};
    uint32_t reserved_1{};
    uint32_t reserved_2{};
};
static_assert(sizeof(IndirectDispatchPrepareConstants) == 48u);

}// namespace lc
