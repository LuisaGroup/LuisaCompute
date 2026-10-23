#pragma once

#include <luisa/ast/op.h>

namespace lc::hlsl {

enum class HlslAtomicLowering {
    NATIVE,
    FLOAT_COMPARE_EXCHANGE,
    FLOAT_CAS_LOOP,
    UNSUPPORTED,
};

// SPIR-V float RMW operations use explicit native SPIR-V opcodes, not HLSL
// Interlocked overloads. Float compare-exchange remains unsupported here;
// its bitwise pointer reinterpretation needs a separate buffer-view contract.
// DXIL uses bitwise float compare-exchange loops for arithmetic RMW operations.
[[nodiscard]] constexpr HlslAtomicLowering plan_hlsl_atomic_lowering(
    luisa::compute::CallOp op, bool is_float32,
    bool is_spirv) noexcept {
    using luisa::compute::CallOp;
    if (!is_float32) { return HlslAtomicLowering::NATIVE; }
    if (is_spirv) {
        if (op == CallOp::ATOMIC_COMPARE_EXCHANGE) {
            return HlslAtomicLowering::UNSUPPORTED;
        }
        return HlslAtomicLowering::NATIVE;
    }
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
