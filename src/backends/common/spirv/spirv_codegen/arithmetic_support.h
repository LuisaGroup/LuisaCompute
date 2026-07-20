#pragma once

#include <luisa/xir/op.h>

namespace lc::spirv {

// GLSL.std.450 defines these transcendental operands as 16- or 32-bit
// floating-point values. Other GLSL.std.450 operations have different width
// contracts, so keep this list exact instead of rejecting float64 broadly.
[[nodiscard]] constexpr bool
spirv_glsl_transcendental_rejects_float64(
    luisa::compute::xir::ArithmeticOp op) noexcept {
    using luisa::compute::xir::ArithmeticOp;
    switch (op) {
        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::ATANH:
        case ArithmeticOp::COS:
        case ArithmeticOp::COSH:
        case ArithmeticOp::SIN:
        case ArithmeticOp::SINH:
        case ArithmeticOp::TAN:
        case ArithmeticOp::TANH:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::EXP10:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2:
        case ArithmeticOp::LOG10:
        case ArithmeticOp::POW: return true;
        default: return false;
    }
}

}// namespace lc::spirv
