#pragma once

#include <luisa/ast/op.h>

namespace lc::hlsl {

enum class HlslAtomicLowering {
    NATIVE,
    FLOAT_COMPARE_EXCHANGE,
    FLOAT_CAS_LOOP,
    // The DX route stores a scalar float64 buffer element as one uint64 word
    // (see CodegenUtility::IsFloat64WordElement), because DXC defines no
    // float64 Interlocked* overload at any shader model. Exchange and
    // compare-exchange map onto the SM 6.6 64-bit integer intrinsics and are
    // bit-preserving; add/sub/min/max retry a 64-bit CAS with float64
    // arithmetic. Every form keeps the hardware 8-byte element granularity.
    FLOAT64_WORD_EXCHANGE,
    FLOAT64_WORD_COMPARE_EXCHANGE,
    FLOAT64_WORD_CAS_LOOP,
    UNSUPPORTED,
};

// SPIR-V float RMW operations use explicit native SPIR-V opcodes, not HLSL
// Interlocked overloads. Float compare-exchange remains unsupported here;
// its bitwise pointer reinterpretation needs a separate buffer-view contract.
// DXIL uses bitwise float compare-exchange loops for arithmetic RMW operations.
//
// `float64_word_alias` states that the atomic addresses exactly one scalar
// float64 Buffer element on the DX route, i.e. that the 64-bit integer word the
// lowering needs really is the element's storage type.
[[nodiscard]] constexpr HlslAtomicLowering plan_hlsl_atomic_lowering(
    luisa::compute::CallOp op, bool is_float32, bool is_float64,
    bool float64_word_alias, bool is_spirv) noexcept {
    using luisa::compute::CallOp;
    if (is_float64) {
        // The HLSL-to-SPIR-V route keeps `double` elements and has no 64-bit
        // word to address; the DX route needs the exact scalar element shape,
        // because a groupshared or aggregate-member float64 has no word alias.
        if (is_spirv || !float64_word_alias) {
            return HlslAtomicLowering::UNSUPPORTED;
        }
        switch (op) {
            case CallOp::ATOMIC_EXCHANGE:
                return HlslAtomicLowering::FLOAT64_WORD_EXCHANGE;
            case CallOp::ATOMIC_COMPARE_EXCHANGE:
                return HlslAtomicLowering::FLOAT64_WORD_COMPARE_EXCHANGE;
            case CallOp::ATOMIC_FETCH_ADD:
            case CallOp::ATOMIC_FETCH_SUB:
            case CallOp::ATOMIC_FETCH_MIN:
            case CallOp::ATOMIC_FETCH_MAX:
                return HlslAtomicLowering::FLOAT64_WORD_CAS_LOOP;
            default:
                return HlslAtomicLowering::UNSUPPORTED;
        }
    }
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
