#pragma once

#include <luisa/ast/op.h>

namespace lc::hlsl {

enum class HlslAtomicLowering {
    NATIVE,
    FLOAT_COMPARE_EXCHANGE,
    FLOAT_CAS_LOOP,
    UNSUPPORTED,
};

[[nodiscard]] constexpr HlslAtomicLowering plan_hlsl_atomic_lowering(
    luisa::compute::CallOp op, bool is_float32,
    bool is_spirv) noexcept {
    using luisa::compute::CallOp;
    if (!is_float32) { return HlslAtomicLowering::NATIVE; }
    if (is_spirv) { return HlslAtomicLowering::UNSUPPORTED; }
    switch (op) {
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            return HlslAtomicLowering::FLOAT_COMPARE_EXCHANGE;
        case CallOp::ATOMIC_FETCH_ADD:
        case CallOp::ATOMIC_FETCH_SUB:
        case CallOp::ATOMIC_FETCH_MIN:
        case CallOp::ATOMIC_FETCH_MAX:
            return HlslAtomicLowering::FLOAT_CAS_LOOP;
        default:
            return HlslAtomicLowering::NATIVE;
    }
}

}// namespace lc::hlsl
