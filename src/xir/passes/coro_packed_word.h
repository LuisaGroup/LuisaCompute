#pragma once

#include <luisa/xir/passes/coro_cfg_distill.h>

namespace luisa::compute::xir::detail {

struct CoroPackedWordMasks {
    uint32_t stored{0u};
    uint32_t preserved{0u};
};

// The same physical read obligation must drive both split code generation
// and scheduler input metadata. For each word, D is its stored lane mask and
// P = L \ D its live, unstored mask. A store needs an old word iff D and P
// are both nonempty; otherwise the encoded word must start from zero.
[[nodiscard]] inline luisa::vector<CoroPackedWordMasks>
coro_packed_word_masks(
    const CoroCfgDistillResult &cfg,
    luisa::span<const size_t> stored_values,
    luisa::span<const size_t> live_values) noexcept {
    luisa::vector<CoroPackedWordMasks> masks(cfg.frame_slots.size());
    for (auto index : stored_values) {
        auto &value = cfg.frame_values[index];
        if (value.bit_offset) {
            masks[value.slot].stored |= uint32_t{1u} << *value.bit_offset;
        }
    }
    for (auto index : live_values) {
        auto &value = cfg.frame_values[index];
        if (value.bit_offset) {
            masks[value.slot].preserved |= uint32_t{1u} << *value.bit_offset;
        }
    }
    for (auto &mask : masks) { mask.preserved &= ~mask.stored; }
    return masks;
}

}// namespace luisa::compute::xir::detail
