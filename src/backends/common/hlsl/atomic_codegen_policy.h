#pragma once

#include <luisa/ast/op.h>

namespace lc::hlsl {

enum class HlslAtomicLowering {
    NATIVE,
    FLOAT_COMPARE_EXCHANGE,
    FLOAT_CAS_LOOP,
    UNSUPPORTED,
};

// The bundled DXC lowers float atomics on typed buffers (RWBuffer /
// RWStructuredBuffer<float>) natively for both DXIL and Vulkan SPIR-V:
// InterlockedAdd/Min/Max/Exchange accept float operands and map to the
// corresponding atomic instructions. Only float compare-exchange still
// requires the bitwise CAS intrinsic, which DXC only accepts on
// RWByteAddressBuffer - keep it fail-closed on the SPIR-V route.
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
