#include "memory_access_plan.h"

namespace lc::spirv {

namespace {

[[nodiscard]] constexpr bool is_power_of_two(size_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

}// namespace

SpirvAlignedMemoryAccessPlan plan_spirv_aligned_word_access(
    spv::MemoryAccessMask memory_access, size_t byte_alignment,
    size_t access_scalar_size) noexcept {
    SpirvAlignedMemoryAccessPlan plan{.mask = memory_access,
                                      .literal = 0u,
                                      .alignment = byte_alignment};
    // The word ABI requires the literal to be at least the largest scalar size
    // of the access (four bytes for a uint32 word) and at most the proven
    // alignment. A smaller or non-power-of-two proof is left unannotated instead
    // of being rounded up.
    auto literal = spirv_word_access_scalar_size;
    if (access_scalar_size > literal ||
        byte_alignment < literal ||
        !is_power_of_two(byte_alignment)) {
        return plan;
    }
    plan.mask = static_cast<spv::MemoryAccessMask>(
        memory_access | spv::MemoryAccessMask::Aligned);
    plan.literal = static_cast<uint32_t>(literal);
    return plan;
}

}// namespace lc::spirv