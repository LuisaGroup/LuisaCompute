#include "entry.h"
#include "../../storage_buffer_metadata.h"
#include "arithmetic_support.h"
#include "texture_sampling.h"
#include <luisa/core/logging.h>
#include <luisa/xir/passes/integer_alignment.h>
#include <SPIRV/GLSL.std.450.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include "../../indirect_dispatch_layout.h"

namespace lc::spirv {

namespace {

struct FastMathFmaPlan {
    const xir::ArithmeticInst *multiply{nullptr};
    const xir::Value *addend{nullptr};

    [[nodiscard]] explicit operator bool() const noexcept {
        return multiply != nullptr;
    }
};

[[nodiscard]] FastMathFmaPlan plan_fast_math_fma(
    const xir::ArithmeticInst *add) noexcept {
    if (add == nullptr ||
        add->op() != xir::ArithmeticOp::BINARY_ADD ||
        add->type() == nullptr ||
        !add->type()->is_float_or_float_vector()) {
        return {};
    }
    auto *type = add->type();
    for (auto multiply_operand = 0u; multiply_operand < 2u;
         ++multiply_operand) {
        auto *candidate = add->operand(multiply_operand);
        if (!candidate->isa<xir::ArithmeticInst>()) { continue; }
        auto *multiply = static_cast<const xir::ArithmeticInst *>(candidate);
        auto *addend = add->operand(1u - multiply_operand);
        if (multiply->op() == xir::ArithmeticOp::BINARY_MUL &&
            multiply->type() == type &&
            multiply->operand(0u)->type() == type &&
            multiply->operand(1u)->type() == type &&
            addend->type() == type) {
            return {.multiply = multiply, .addend = addend};
        }
    }
    return {};
}

[[nodiscard]] bool is_deferred_fast_math_fma_multiply(
    const xir::ArithmeticInst *inst) noexcept {
    if (inst == nullptr) { return false; }
    const xir::Use *only_use = nullptr;
    for (auto *use : inst->use_list()) {
        if (only_use != nullptr) { return false; }
        only_use = use;
    }
    if (only_use == nullptr || only_use->user() == nullptr ||
        !only_use->user()->isa<xir::ArithmeticInst>()) {
        return false;
    }
    auto *add = static_cast<const xir::ArithmeticInst *>(
        only_use->user());
    auto plan = plan_fast_math_fma(add);
    return plan && plan.multiply == inst;
}

}// namespace

std::vector<spv::Id>
SpirvCodegenEntry::_emit_aggregate_access_indices(
    const SpirvAggregateIndexPlan &plan) noexcept {
    LUISA_ASSERT(plan.succeeded(),
                 "Cannot emit a failed SPIR-V aggregate index plan: {}",
                 plan.diagnostic);
    std::vector<spv::Id> indices;
    indices.reserve(plan.steps.size());
    for (auto &&step : plan.steps) {
        if (step.kind ==
            SpirvAggregateIndexKind::STRUCTURE_MEMBER) {
            LUISA_ASSERT(step.is_constant &&
                             step.constant_index <=
                                 std::numeric_limits<uint32_t>::max(),
                         "SPIR-V structure member index was not canonicalizable.");
            // OpAccessChain requires a structure member to be selected by a
            // 32-bit OpConstant. Never forward the XIR constant ID here: XIR
            // deliberately admits signed/unsigned constants of other widths.
            indices.emplace_back(_builder.makeUintConstant(
                static_cast<uint32_t>(step.constant_index)));
        } else {
            // Array/vector/matrix/buffer indices are ordinary integer IDs.
            // Preserve dynamic values and legal non-32-bit constants exactly.
            indices.emplace_back(_emit_value(step.index));
        }
    }
    return indices;
}

void SpirvCodegenEntry::_emit_arithmetic_inst(const xir::ArithmeticInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    auto t = inst->type();
    auto elem = t->is_vector() || t->is_matrix() ? t->element() : t;
    if (elem->is_float8() &&
        !spirv_fp8_transport_op_supported(inst->op())) {
        LUISA_ERROR_WITH_LOCATION(
            "SPIR-V backend does not support general arithmetic on FP8 types. "
            "Please up-convert to float16 or float32, perform arithmetic, and down-convert.");
    }
    auto is_float = elem->is_float();
    auto is_signed_int = elem->is_int();
    auto is_bool = elem->is_bool();
    auto is_scalar = t->is_scalar();

    auto operand = [&](size_t i) noexcept { return _emit_value(inst->operand(i)); };
    spv::Id id = spv::NoResult;

    auto require_uint32_bit_operation = [&]() noexcept {
        LUISA_ASSERT(
            (t->is_scalar() || t->is_vector()) && elem->is_uint32(),
            "SPIR-V bit operation '{}' requires a uint32 scalar/vector, got {}.",
            xir::to_string(inst->op()), t->description());
    };

    auto make_float_scalar_constant = [&](const Type *scalar_type, double value) noexcept -> spv::Id {
        if (scalar_type->is_float16()) { return _builder.makeFloat16Constant(static_cast<float>(value)); }
        if (scalar_type->is_float32()) { return _builder.makeFloatConstant(static_cast<float>(value)); }
        if (scalar_type->is_float64()) { return _builder.makeDoubleConstant(value); }
        LUISA_ERROR_WITH_LOCATION("Unsupported floating-point constant type {}.", scalar_type->description());
    };
    auto make_integer_constant = [&](const Type *value_type, uint64_t value) noexcept -> spv::Id {
        auto scalar_type = value_type->is_vector() ? value_type->element() : value_type;
        auto scalar_spv_type = _convert_type(scalar_type, Usage::READ);
        auto bit_width = static_cast<uint32_t>(scalar_type->size() * 8u);
        auto scalar = bit_width == 64u ?
                          _builder.makeInt64Constant(scalar_spv_type, value, false) :
                          _builder.makeIntConstant(scalar_spv_type, static_cast<uint32_t>(value), false);
        return value_type->is_vector() ?
                   _builder.smearScalar(spv::NoPrecision, scalar, _convert_type(value_type, Usage::READ)) :
                   scalar;
    };
    auto decode_constant_index = [](const xir::Constant *constant) noexcept {
        auto decoded = decode_spirv_aggregate_index_constant(constant);
        if (!decoded) {
            LUISA_ERROR_WITH_LOCATION("{}", decoded.diagnostic);
        }
        return decoded.value;
    };
    auto make_aggregate_index_plan = [&](const Type *aggregate_type,
                                         size_t first_index) noexcept {
        LUISA_ASSERT(first_index <= inst->operand_count(),
                     "SPIR-V arithmetic aggregate index offset {} exceeds "
                     "the {} operands of '{}'.",
                     first_index, inst->operand_count(),
                     xir::to_string(inst->op()));
        luisa::vector<const xir::Value *> index_values;
        index_values.reserve(inst->operand_count() - first_index);
        for (auto i = first_index; i < inst->operand_count(); ++i) {
            index_values.emplace_back(inst->operand(i));
        }
        auto plan = plan_spirv_aggregate_indices(
            aggregate_type, luisa::span{index_values});
        if (!plan) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid aggregate indices for SPIR-V arithmetic '{}': {}",
                xir::to_string(inst->op()), plan.diagnostic);
        }
        return plan;
    };

    auto make_glsl_call = [&](int builtin, spv::Id result_type, std::vector<spv::Id> ops) noexcept -> spv::Id {
        auto needs_8bit_promote = [&](spv::Id ty) noexcept -> bool {
            auto scalar = _builder.getScalarTypeId(ty);
            return _builder.getTypeClass(scalar) == spv::Op::OpTypeInt && _builder.getScalarTypeWidth(scalar) == 8;
        };
        bool promote = needs_8bit_promote(result_type);
        for (auto op : ops) promote = promote || needs_8bit_promote(_builder.getTypeId(op));
        if (promote) {
            spv::Id target_scalar;
            if (needs_8bit_promote(result_type)) {
                auto result_scalar = _builder.getScalarTypeId(result_type);
                target_scalar = _builder.isIntType(result_scalar) ? _builder.makeIntType(32) : _builder.makeUintType(32);
            } else {
                target_scalar = _builder.getScalarTypeId(result_type);
            }
            spv::Id target_type = result_type;
            if (_builder.isVectorType(result_type)) {
                target_type = _builder.makeVectorType(target_scalar, _builder.getNumTypeComponents(result_type));
            } else {
                target_type = target_scalar;
            }
            std::vector<spv::Id> wide_ops;
            for (auto op : ops) {
                wide_ops.push_back(_ensure_type(op, target_type));
            }
            spv::Id wide_result = _builder.createBuiltinCall(target_type, _glsl450, builtin, wide_ops);
            return _ensure_type(wide_result, result_type);
        }
        return _builder.createBuiltinCall(result_type, _glsl450, builtin, ops);
    };

    auto glsl = [&](int builtin, auto... args) noexcept -> spv::Id {
        return make_glsl_call(builtin, type, {args...});
    };

    auto glsl_typed = [&](int f_builtin, int s_builtin, int u_builtin, auto... args) noexcept -> spv::Id {
        int builtin = f_builtin;
        if (is_signed_int)
            builtin = s_builtin;
        else if (is_bool || elem->is_uint())
            builtin = u_builtin;
        return make_glsl_call(builtin, type, {args...});
    };

    auto unary = [&](spv::Op op) noexcept -> spv::Id {
        return _builder.createUnaryOp(op, type, operand(0));
    };
    auto binary = [&](spv::Op op) noexcept -> spv::Id {
        return _builder.createBinOp(op, type, operand(0), operand(1));
    };
    auto mark_no_contraction = [&](spv::Id arithmetic) noexcept {
        if (!_enable_fast_math) {
            _builder.addDecoration(
                arithmetic, spv::Decoration::NoContraction);
        }
        return arithmetic;
    };
    auto operand_matching_result_type = [&](size_t i) noexcept -> spv::Id {
        auto value = operand(i);
        auto operand_type = inst->operand(i)->type();
        if (t->is_vector() && operand_type->is_scalar()) {
            LUISA_ASSERT(operand_type == t->element(),
                         "Cannot broadcast {} to {}.",
                         operand_type->description(), t->description());
            value = _builder.smearScalar(spv::NoPrecision, value, type);
        }
        return value;
    };
    auto copy_float_sign = [&](spv::Id magnitude_value,
                               spv::Id sign_value) noexcept -> spv::Id {
        LUISA_ASSERT(is_float,
                     "SPIR-V floating sign copy requires a floating-point result.");
        auto bit_width = static_cast<uint32_t>(elem->size() * 8u);
        auto uint_scalar_type =
            _builder.makeUintType(static_cast<int32_t>(bit_width));
        auto uint_type = is_scalar ?
                             uint_scalar_type :
                             _builder.makeVectorType(
                                 uint_scalar_type, t->dimension());
        auto make_bit_constant = [&](uint64_t value) noexcept {
            auto scalar = bit_width == 64u ?
                              _builder.makeInt64Constant(
                                  uint_scalar_type, value, false) :
                              _builder.makeIntConstant(
                                  uint_scalar_type,
                                  static_cast<uint32_t>(value), false);
            return is_scalar ?
                       scalar :
                       _builder.smearScalar(
                           spv::NoPrecision, scalar, uint_type);
        };
        auto sign_bit_value = uint64_t{1} << (bit_width - 1u);
        auto sign_bit = make_bit_constant(sign_bit_value);
        auto magnitude_bits = make_bit_constant(sign_bit_value - 1u);
        auto value_bits = _builder.createUnaryOp(
            spv::Op::OpBitcast, uint_type, magnitude_value);
        auto source_sign_bits = _builder.createUnaryOp(
            spv::Op::OpBitcast, uint_type, sign_value);
        auto magnitude = _builder.createBinOp(
            spv::Op::OpBitwiseAnd, uint_type,
            value_bits, magnitude_bits);
        auto sign = _builder.createBinOp(
            spv::Op::OpBitwiseAnd, uint_type,
            source_sign_bits, sign_bit);
        auto copied = _builder.createBinOp(
            spv::Op::OpBitwiseOr, uint_type, magnitude, sign);
        return _builder.createUnaryOp(
            spv::Op::OpBitcast, type, copied);
    };
    auto strict_float32_constant = [&](double value) noexcept {
        auto scalar = make_float_scalar_constant(elem, value);
        return is_scalar ?
                   scalar :
                   _builder.smearScalar(
                       spv::NoPrecision,
                       scalar,
                       type);
    };
    auto strict_float32_mul =
        [&](spv::Id lhs, spv::Id rhs) noexcept {
            return mark_no_contraction(
                _builder.createBinOp(
                    spv::Op::OpFMul,
                    type,
                    lhs,
                    rhs));
        };
    auto strict_float32_add =
        [&](spv::Id lhs, spv::Id rhs) noexcept {
            return mark_no_contraction(
                _builder.createBinOp(
                    spv::Op::OpFAdd,
                    type,
                    lhs,
                    rhs));
        };
    auto strict_float32_sub =
        [&](spv::Id lhs, spv::Id rhs) noexcept {
            return mark_no_contraction(
                _builder.createBinOp(
                    spv::Op::OpFSub,
                    type,
                    lhs,
                    rhs));
        };
    auto exact_float32_fmod =
        [&](spv::Id x, spv::Id y) noexcept {
            LUISA_ASSERT(
                elem->is_float32() &&
                    (t->is_scalar() || t->is_vector()),
                "Exact SPIR-V fmod requires float32 scalar/vector operands.");

            // For a finite binary32 value, write the magnitude as
            //
            //   significand * 2^(scale - 149),
            //
            // where scale is max(biased_exponent - 1, 0). If |x| >= |y|,
            // the exact remainder is therefore
            //
            //   ((sig_x * 2^(scale_x - scale_y)) mod sig_y)
            //       * 2^(scale_y - 149).
            //
            // Compute the integer modulus with exponentiation by squaring.
            // The significands have at most 24 bits, so every intermediate
            // product fits in 48 bits. This avoids forming the enormous,
            // rounded floating-point quotient used by common OpFRem lowering.
            auto uint_scalar_type = _builder.makeUintType(32);
            auto uint_type = is_scalar ?
                                 uint_scalar_type :
                                 _builder.makeVectorType(
                                     uint_scalar_type, t->dimension());
            auto wide_uint_scalar_type = _builder.makeUintType(64);
            auto wide_uint_type = is_scalar ?
                                      wide_uint_scalar_type :
                                      _builder.makeVectorType(
                                          wide_uint_scalar_type,
                                          t->dimension());
            auto bool_type = is_scalar ?
                                 _builder.makeBoolType() :
                                 _builder.makeVectorType(
                                     _builder.makeBoolType(),
                                     t->dimension());
            auto make_u32 = [&](uint32_t value) noexcept {
                auto scalar = _builder.makeIntConstant(
                    uint_scalar_type, value, false);
                return is_scalar ?
                           scalar :
                           _builder.smearScalar(
                               spv::NoPrecision, scalar, uint_type);
            };
            auto make_u64 = [&](uint64_t value) noexcept {
                auto scalar = _builder.makeInt64Constant(
                    wide_uint_scalar_type, value, false);
                return is_scalar ?
                           scalar :
                           _builder.smearScalar(
                               spv::NoPrecision, scalar, wide_uint_type);
            };
            auto select_u32 = [&](spv::Id condition,
                                  spv::Id when_true,
                                  spv::Id when_false) noexcept {
                return _builder.createTriOp(
                    spv::Op::OpSelect, uint_type,
                    condition, when_true, when_false);
            };
            auto select_u64 = [&](spv::Id condition,
                                  spv::Id when_true,
                                  spv::Id when_false) noexcept {
                return _builder.createTriOp(
                    spv::Op::OpSelect, wide_uint_type,
                    condition, when_true, when_false);
            };

            auto zero = make_u32(0u);
            auto one = make_u32(1u);
            auto exponent_mask = make_u32(0xffu);
            auto fraction_mask = make_u32(0x007fffffu);
            auto magnitude_mask = make_u32(0x7fffffffu);
            auto sign_mask = make_u32(0x80000000u);
            auto hidden_bit = make_u32(0x00800000u);
            auto infinity_bits = make_u32(0x7f800000u);
            auto quiet_nan_bits = make_u32(0x7fc00000u);

            auto x_bits = _builder.createUnaryOp(
                spv::Op::OpBitcast, uint_type, x);
            auto y_bits = _builder.createUnaryOp(
                spv::Op::OpBitcast, uint_type, y);
            auto x_magnitude = _builder.createBinOp(
                spv::Op::OpBitwiseAnd, uint_type,
                x_bits, magnitude_mask);
            auto y_magnitude = _builder.createBinOp(
                spv::Op::OpBitwiseAnd, uint_type,
                y_bits, magnitude_mask);
            auto x_exponent = _builder.createBinOp(
                spv::Op::OpBitwiseAnd, uint_type,
                _builder.createBinOp(
                    spv::Op::OpShiftRightLogical, uint_type,
                    x_magnitude, make_u32(23u)),
                exponent_mask);
            auto y_exponent = _builder.createBinOp(
                spv::Op::OpBitwiseAnd, uint_type,
                _builder.createBinOp(
                    spv::Op::OpShiftRightLogical, uint_type,
                    y_magnitude, make_u32(23u)),
                exponent_mask);
            auto x_is_normal = _builder.createBinOp(
                spv::Op::OpINotEqual, bool_type, x_exponent, zero);
            auto y_is_normal = _builder.createBinOp(
                spv::Op::OpINotEqual, bool_type, y_exponent, zero);
            auto x_significand = _builder.createBinOp(
                spv::Op::OpBitwiseOr, uint_type,
                _builder.createBinOp(
                    spv::Op::OpBitwiseAnd, uint_type,
                    x_magnitude, fraction_mask),
                select_u32(x_is_normal, hidden_bit, zero));
            auto y_significand = _builder.createBinOp(
                spv::Op::OpBitwiseOr, uint_type,
                _builder.createBinOp(
                    spv::Op::OpBitwiseAnd, uint_type,
                    y_magnitude, fraction_mask),
                select_u32(y_is_normal, hidden_bit, zero));
            auto x_scale = select_u32(
                x_is_normal,
                _builder.createBinOp(
                    spv::Op::OpISub, uint_type, x_exponent, one),
                zero);
            auto y_scale = select_u32(
                y_is_normal,
                _builder.createBinOp(
                    spv::Op::OpISub, uint_type, y_exponent, one),
                zero);
            auto scales_ordered = _builder.createBinOp(
                spv::Op::OpUGreaterThanEqual, bool_type,
                x_scale, y_scale);
            auto exponent_delta = select_u32(
                scales_ordered,
                _builder.createBinOp(
                    spv::Op::OpISub, uint_type, x_scale, y_scale),
                zero);

            auto y_significand_is_zero = _builder.createBinOp(
                spv::Op::OpIEqual, bool_type, y_significand, zero);
            auto safe_y_significand = select_u32(
                y_significand_is_zero, one, y_significand);
            auto wide_x_significand = _builder.createUnaryOp(
                spv::Op::OpUConvert, wide_uint_type, x_significand);
            auto wide_y_significand = _builder.createUnaryOp(
                spv::Op::OpUConvert, wide_uint_type,
                safe_y_significand);
            auto remainder = _builder.createBinOp(
                spv::Op::OpUMod, wide_uint_type,
                wide_x_significand, wide_y_significand);
            auto factor = _builder.createBinOp(
                spv::Op::OpUMod, wide_uint_type,
                make_u64(2u), wide_y_significand);
            for (uint32_t bit = 0u; bit < 8u; ++bit) {
                auto bit_mask = make_u32(1u << bit);
                auto bit_set = _builder.createBinOp(
                    spv::Op::OpINotEqual, bool_type,
                    _builder.createBinOp(
                        spv::Op::OpBitwiseAnd, uint_type,
                        exponent_delta, bit_mask),
                    zero);
                auto multiplied = _builder.createBinOp(
                    spv::Op::OpIMul, wide_uint_type,
                    remainder, factor);
                auto reduced = _builder.createBinOp(
                    spv::Op::OpUMod, wide_uint_type,
                    multiplied, wide_y_significand);
                remainder = select_u64(bit_set, reduced, remainder);
                if (bit + 1u < 8u) {
                    factor = _builder.createBinOp(
                        spv::Op::OpUMod, wide_uint_type,
                        _builder.createBinOp(
                            spv::Op::OpIMul, wide_uint_type,
                            factor, factor),
                        wide_y_significand);
                }
            }
            auto remainder_bits = _builder.createUnaryOp(
                spv::Op::OpUConvert, uint_type, remainder);

            // Re-encode remainder * 2^(scale_y - 149) without performing a
            // floating-point operation, so subnormal results are preserved.
            auto remainder_is_zero = _builder.createBinOp(
                spv::Op::OpIEqual, bool_type, remainder_bits, zero);
            auto remainder_msb = _builder.createBuiltinCall(
                uint_type, _glsl450, GLSLstd450FindUMsb,
                {remainder_bits});
            auto safe_msb = select_u32(
                remainder_is_zero, zero, remainder_msb);
            auto exponent_sum = _builder.createBinOp(
                spv::Op::OpIAdd, uint_type,
                safe_msb, y_exponent);
            auto normal_result = _builder.createBinOp(
                spv::Op::OpLogicalAnd, bool_type,
                _builder.createUnaryOp(
                    spv::Op::OpLogicalNot, bool_type,
                    remainder_is_zero),
                _builder.createBinOp(
                    spv::Op::OpUGreaterThanEqual, bool_type,
                    exponent_sum, make_u32(24u)));
            auto normalized_significand = _builder.createBinOp(
                spv::Op::OpShiftLeftLogical, uint_type,
                remainder_bits,
                _builder.createBinOp(
                    spv::Op::OpISub, uint_type,
                    make_u32(23u), safe_msb));
            auto normal_bits = _builder.createBinOp(
                spv::Op::OpBitwiseOr, uint_type,
                _builder.createBinOp(
                    spv::Op::OpShiftLeftLogical, uint_type,
                    _builder.createBinOp(
                        spv::Op::OpISub, uint_type,
                        exponent_sum, make_u32(23u)),
                    make_u32(23u)),
                _builder.createBinOp(
                    spv::Op::OpBitwiseAnd, uint_type,
                    normalized_significand, fraction_mask));
            auto subnormal_shift = select_u32(
                y_is_normal,
                _builder.createBinOp(
                    spv::Op::OpISub, uint_type, y_exponent, one),
                zero);
            auto clamped_subnormal_shift = select_u32(
                _builder.createBinOp(
                    spv::Op::OpUGreaterThan, bool_type,
                    subnormal_shift, make_u32(23u)),
                make_u32(23u), subnormal_shift);
            auto subnormal_bits = _builder.createBinOp(
                spv::Op::OpShiftLeftLogical, uint_type,
                remainder_bits, clamped_subnormal_shift);
            auto magnitude_result = select_u32(
                normal_result, normal_bits, subnormal_bits);
            auto signed_result = _builder.createBinOp(
                spv::Op::OpBitwiseOr, uint_type,
                magnitude_result,
                _builder.createBinOp(
                    spv::Op::OpBitwiseAnd, uint_type,
                    x_bits, sign_mask));

            auto invalid = _builder.createBinOp(
                spv::Op::OpLogicalOr, bool_type,
                _builder.createBinOp(
                    spv::Op::OpUGreaterThanEqual, bool_type,
                    x_magnitude, infinity_bits),
                _builder.createBinOp(
                    spv::Op::OpLogicalOr, bool_type,
                    _builder.createBinOp(
                        spv::Op::OpUGreaterThan, bool_type,
                        y_magnitude, infinity_bits),
                    _builder.createBinOp(
                        spv::Op::OpIEqual, bool_type,
                        y_magnitude, zero)));
            auto compute_remainder = _builder.createBinOp(
                spv::Op::OpLogicalAnd, bool_type,
                _builder.createUnaryOp(
                    spv::Op::OpLogicalNot, bool_type, invalid),
                _builder.createBinOp(
                    spv::Op::OpLogicalAnd, bool_type,
                    _builder.createBinOp(
                        spv::Op::OpULessThan, bool_type,
                        y_magnitude, infinity_bits),
                    _builder.createBinOp(
                        spv::Op::OpUGreaterThanEqual, bool_type,
                        x_magnitude, y_magnitude)));
            auto result_bits = select_u32(
                invalid,
                quiet_nan_bits,
                select_u32(
                    compute_remainder, signed_result, x_bits));
            return _builder.createUnaryOp(
                spv::Op::OpBitcast, type, result_bits);
        };
    auto strict_float32_asin = [&](spv::Id x) noexcept {
        LUISA_ASSERT(
            elem->is_float32(),
            "Strict SPIR-V asin requires float32 operands.");
        auto magnitude = glsl(GLSLstd450FAbs, x);
        auto half = strict_float32_constant(0.5);
        auto one = strict_float32_constant(1.0);
        auto two = strict_float32_constant(2.0);
        auto half_pi = strict_float32_constant(
            1.57079632679489661923);
        auto bool_type = is_scalar ?
                             _builder.makeBoolType() :
                             _builder.makeVectorType(
                                 _builder.makeBoolType(),
                                 t->dimension());
        auto use_complement =
            _builder.createBinOp(
                spv::Op::OpFOrdGreaterThan,
                bool_type,
                magnitude,
                half);
        auto complement =
            strict_float32_mul(
                half,
                strict_float32_sub(one, magnitude));
        auto reduced_complement =
            glsl(GLSLstd450Sqrt, complement);
        auto reduced =
            _builder.createTriOp(
                spv::Op::OpSelect,
                type,
                use_complement,
                reduced_complement,
                magnitude);
        auto squared =
            strict_float32_mul(reduced, reduced);

        auto polynomial =
            strict_float32_constant(4.2163199048e-2);
        polynomial = strict_float32_add(
            strict_float32_mul(polynomial, squared),
            strict_float32_constant(2.4181311049e-2));
        polynomial = strict_float32_add(
            strict_float32_mul(polynomial, squared),
            strict_float32_constant(4.5470025998e-2));
        polynomial = strict_float32_add(
            strict_float32_mul(polynomial, squared),
            strict_float32_constant(7.4953002686e-2));
        polynomial = strict_float32_add(
            strict_float32_mul(polynomial, squared),
            strict_float32_constant(1.6666752422e-1));

        auto reduced_result =
            strict_float32_add(
                reduced,
                strict_float32_mul(
                    strict_float32_mul(reduced, squared),
                    polynomial));
        auto complement_result =
            strict_float32_sub(
                half_pi,
                strict_float32_mul(two, reduced_result));
        auto result_magnitude =
            _builder.createTriOp(
                spv::Op::OpSelect,
                type,
                use_complement,
                complement_result,
                reduced_result);
        return copy_float_sign(result_magnitude, x);
    };
    LUISA_ASSERT(
        !spirv_glsl_transcendental_rejects_float64(inst->op()) ||
            !elem->is_float64(),
        "SPIR-V dialect validation failed to reject float64 operands for "
        "GLSL.std.450 transcendental operation '{}'.",
        xir::to_string(inst->op()));

    switch (inst->op()) {
        case xir::ArithmeticOp::UNARY_MINUS:
            if (is_float)
                id = unary(spv::Op::OpFNegate);
            else
                id = unary(spv::Op::OpSNegate);
            break;
        case xir::ArithmeticOp::UNARY_BIT_NOT:
            if (is_bool)
                id = unary(spv::Op::OpLogicalNot);
            else
                id = unary(spv::Op::OpNot);
            break;
        case xir::ArithmeticOp::BINARY_ADD: {
            if (is_float && _enable_fast_math) {
                // Peephole: detect FMul+FAdd pattern and emit fused multiply-add (FMA).
                // BINARY_ADD(BINARY_MUL(a, b), c) -> Fma(a, b, c)
                // BINARY_ADD(c, BINARY_MUL(a, b)) -> Fma(a, b, c)
                if (auto fma = plan_fast_math_fma(inst)) {
                    // The multiply's operands dominate this add. A single-use
                    // multiply is deliberately deferred by _emit_instruction;
                    // multi-use multiplies remain materialized for their other
                    // consumers while this add still receives an FMA.
                    auto a = _emit_value(fma.multiply->operand(0u));
                    auto b = _emit_value(fma.multiply->operand(1u));
                    auto c = _emit_value(fma.addend);
                    id = _builder.createBuiltinCall(type, _glsl450, GLSLstd450Fma, {a, b, c});
                } else {
                    id = binary(spv::Op::OpFAdd);
                }
            } else {
                id = binary(is_float ?
                                spv::Op::OpFAdd :
                                spv::Op::OpIAdd);
            }
            break;
        }
        case xir::ArithmeticOp::BINARY_SUB:
            if (is_float)
                id = binary(spv::Op::OpFSub);
            else
                id = binary(spv::Op::OpISub);
            break;
        case xir::ArithmeticOp::BINARY_MUL: {
            id = binary(is_float ?
                            spv::Op::OpFMul :
                            spv::Op::OpIMul);
            break;
        }
        case xir::ArithmeticOp::BINARY_DIV:
            if (is_float) {
                id = binary(spv::Op::OpFDiv);
            } else if (is_signed_int) {
                id = binary(spv::Op::OpSDiv);
            } else
                id = binary(spv::Op::OpUDiv);
            break;
        case xir::ArithmeticOp::BINARY_MOD:
            if (elem->is_float32())
                id = exact_float32_fmod(operand(0), operand(1));
            else if (is_float)
                id = binary(spv::Op::OpFRem);
            else if (is_signed_int)
                id = binary(spv::Op::OpSRem);
            else
                id = binary(spv::Op::OpUMod);
            break;
        case xir::ArithmeticOp::BINARY_BIT_AND:
            if (is_bool)
                id = binary(spv::Op::OpLogicalAnd);
            else
                id = binary(spv::Op::OpBitwiseAnd);
            break;
        case xir::ArithmeticOp::BINARY_BIT_OR:
            if (is_bool)
                id = binary(spv::Op::OpLogicalOr);
            else
                id = binary(spv::Op::OpBitwiseOr);
            break;
        case xir::ArithmeticOp::BINARY_BIT_XOR:
            if (is_bool)
                id = binary(spv::Op::OpLogicalNotEqual);
            else
                id = binary(spv::Op::OpBitwiseXor);
            break;
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
            id = binary(spv::Op::OpShiftLeftLogical);
            break;
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
            if (is_signed_int)
                id = binary(spv::Op::OpShiftRightArithmetic);
            else
                id = binary(spv::Op::OpShiftRightLogical);
            break;
        case xir::ArithmeticOp::BINARY_ROTATE_LEFT: {
            auto a = operand(0);
            auto b = operand(1);
            auto width = t->is_scalar() ? t->size() * 8 : t->element()->size() * 8;
            auto shift_type = inst->operand(1)->type();
            auto shift_element = shift_type->is_vector() ? shift_type->element() : shift_type;
            auto shift_unsigned_scalar_type = _builder.makeUintType(
                static_cast<int32_t>(shift_element->size() * 8u));
            auto shift_unsigned_type = shift_type->is_vector() ?
                                           _builder.makeVectorType(shift_unsigned_scalar_type, shift_type->dimension()) :
                                           shift_unsigned_scalar_type;
            if (shift_element->is_int()) {
                b = _builder.createUnaryOp(spv::Op::OpBitcast, shift_unsigned_type, b);
            }
            auto width_scalar = shift_element->size() == 8u ?
                                    _builder.makeInt64Constant(shift_unsigned_scalar_type, width, false) :
                                    _builder.makeIntConstant(shift_unsigned_scalar_type, static_cast<uint32_t>(width), false);
            auto width_id = shift_type->is_vector() ?
                                _builder.smearScalar(spv::NoPrecision, width_scalar, shift_unsigned_type) :
                                width_scalar;
            auto b_mod = _builder.createBinOp(spv::Op::OpUMod, shift_unsigned_type, b, width_id);
            auto reverse_shift = _builder.createBinOp(
                spv::Op::OpUMod, shift_unsigned_type,
                _builder.createBinOp(spv::Op::OpISub, shift_unsigned_type, width_id, b_mod),
                width_id);
            auto left = _builder.createBinOp(spv::Op::OpShiftLeftLogical, type, a, b_mod);
            auto right = _builder.createBinOp(spv::Op::OpShiftRightLogical, type, a, reverse_shift);
            id = _builder.createBinOp(spv::Op::OpBitwiseOr, type, left, right);
            break;
        }
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT: {
            auto a = operand(0);
            auto b = operand(1);
            auto width = t->is_scalar() ? t->size() * 8 : t->element()->size() * 8;
            auto shift_type = inst->operand(1)->type();
            auto shift_element = shift_type->is_vector() ? shift_type->element() : shift_type;
            auto shift_unsigned_scalar_type = _builder.makeUintType(
                static_cast<int32_t>(shift_element->size() * 8u));
            auto shift_unsigned_type = shift_type->is_vector() ?
                                           _builder.makeVectorType(shift_unsigned_scalar_type, shift_type->dimension()) :
                                           shift_unsigned_scalar_type;
            if (shift_element->is_int()) {
                b = _builder.createUnaryOp(spv::Op::OpBitcast, shift_unsigned_type, b);
            }
            auto width_scalar = shift_element->size() == 8u ?
                                    _builder.makeInt64Constant(shift_unsigned_scalar_type, width, false) :
                                    _builder.makeIntConstant(shift_unsigned_scalar_type, static_cast<uint32_t>(width), false);
            auto width_id = shift_type->is_vector() ?
                                _builder.smearScalar(spv::NoPrecision, width_scalar, shift_unsigned_type) :
                                width_scalar;
            auto b_mod = _builder.createBinOp(spv::Op::OpUMod, shift_unsigned_type, b, width_id);
            auto reverse_shift = _builder.createBinOp(
                spv::Op::OpUMod, shift_unsigned_type,
                _builder.createBinOp(spv::Op::OpISub, shift_unsigned_type, width_id, b_mod),
                width_id);
            auto right = _builder.createBinOp(spv::Op::OpShiftRightLogical, type, a, b_mod);
            auto left = _builder.createBinOp(spv::Op::OpShiftLeftLogical, type, a, reverse_shift);
            id = _builder.createBinOp(spv::Op::OpBitwiseOr, type, left, right);
            break;
        }
        case xir::ArithmeticOp::BINARY_LESS: {
            auto op_elem = inst->operand(0)->type();
            op_elem = op_elem->is_vector() ? op_elem->element() : op_elem;
            if (op_elem->is_float())
                id = binary(spv::Op::OpFOrdLessThan);
            else if (op_elem->is_int())
                id = binary(spv::Op::OpSLessThan);
            else
                id = binary(spv::Op::OpULessThan);
            break;
        }
        case xir::ArithmeticOp::BINARY_GREATER: {
            auto op_elem = inst->operand(0)->type();
            op_elem = op_elem->is_vector() ? op_elem->element() : op_elem;
            if (op_elem->is_float())
                id = binary(spv::Op::OpFOrdGreaterThan);
            else if (op_elem->is_int())
                id = binary(spv::Op::OpSGreaterThan);
            else
                id = binary(spv::Op::OpUGreaterThan);
            break;
        }
        case xir::ArithmeticOp::BINARY_LESS_EQUAL: {
            auto op_elem = inst->operand(0)->type();
            op_elem = op_elem->is_vector() ? op_elem->element() : op_elem;
            if (op_elem->is_float())
                id = binary(spv::Op::OpFOrdLessThanEqual);
            else if (op_elem->is_int())
                id = binary(spv::Op::OpSLessThanEqual);
            else
                id = binary(spv::Op::OpULessThanEqual);
            break;
        }
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL: {
            auto op_elem = inst->operand(0)->type();
            op_elem = op_elem->is_vector() ? op_elem->element() : op_elem;
            if (op_elem->is_float())
                id = binary(spv::Op::OpFOrdGreaterThanEqual);
            else if (op_elem->is_int())
                id = binary(spv::Op::OpSGreaterThanEqual);
            else
                id = binary(spv::Op::OpUGreaterThanEqual);
            break;
        }
        case xir::ArithmeticOp::BINARY_EQUAL: {
            auto op_elem = inst->operand(0)->type();
            op_elem = op_elem->is_vector() ? op_elem->element() : op_elem;
            if (op_elem->is_float())
                id = binary(spv::Op::OpFOrdEqual);
            else if (op_elem->is_bool())
                id = binary(spv::Op::OpLogicalEqual);
            else
                id = binary(spv::Op::OpIEqual);
            break;
        }
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: {
            auto op_elem = inst->operand(0)->type();
            op_elem = op_elem->is_vector() ? op_elem->element() : op_elem;
            if (op_elem->is_float())
                id = binary(spv::Op::OpFUnordNotEqual);
            else if (op_elem->is_bool())
                id = binary(spv::Op::OpLogicalNotEqual);
            else
                id = binary(spv::Op::OpINotEqual);
            break;
        }
        case xir::ArithmeticOp::SELECT: {
            // XIR SELECT operands are (false_value, true_value, condition)
            auto cond = operand(2);
            auto cond_type = _builder.getTypeId(cond);
            auto is_bool_condition =
                _builder.isBoolType(cond_type) ||
                (_builder.isVectorType(cond_type) &&
                 _builder.isBoolType(
                     _builder.getContainedTypeId(cond_type)));
            LUISA_ASSERT(
                is_bool_condition,
                "SPIR-V arithmetic emission received a SELECT condition "
                "that did not lower to bool or bool vector.");
            id = _builder.createTriOp(spv::Op::OpSelect, type, cond, operand(1), operand(0));
            break;
        }

        // Boolean reductions
        case xir::ArithmeticOp::ALL:
            id = unary(spv::Op::OpAll);
            break;
        case xir::ArithmeticOp::ANY:
            id = unary(spv::Op::OpAny);
            break;

        // Math builtins via GLSL.std.450
        case xir::ArithmeticOp::CLAMP:
            // Luisa's floating min/max contract matches C fmin/fmax and the
            // LLVM minnum/maxnum intrinsics used by the CUDA, HIP, and
            // fallback XIR backends: a lone NaN yields the numeric operand.
            // GLSL FClamp inherits the implementation-dependent NaN behavior
            // of FMin/FMax, while NClamp has the required number-preferring
            // semantics.
            id = glsl_typed(GLSLstd450NClamp, GLSLstd450SClamp, GLSLstd450UClamp,
                            operand(0), operand(1), operand(2));
            break;
        case xir::ArithmeticOp::SATURATE:
            if (is_float) {
                auto zero = make_float_scalar_constant(elem, 0.0);
                auto one = make_float_scalar_constant(elem, 1.0);
                if (!is_scalar) {
                    zero = _builder.smearScalar(spv::NoPrecision, zero, type);
                    one = _builder.smearScalar(spv::NoPrecision, one, type);
                }
                id = glsl(GLSLstd450NClamp, operand(0), zero, one);
            } else {
                LUISA_NOT_IMPLEMENTED("SPIR-V saturate for integer types.");
            }
            break;
        case xir::ArithmeticOp::LERP:
            id = glsl(GLSLstd450FMix,
                      operand_matching_result_type(0),
                      operand_matching_result_type(1),
                      operand_matching_result_type(2));
            break;
        case xir::ArithmeticOp::STEP:
            id = glsl(GLSLstd450Step,
                      operand_matching_result_type(0),
                      operand_matching_result_type(1));
            break;
        case xir::ArithmeticOp::SMOOTHSTEP:
            id = glsl(GLSLstd450SmoothStep,
                      operand_matching_result_type(0),
                      operand_matching_result_type(1),
                      operand_matching_result_type(2));
            break;
        case xir::ArithmeticOp::ABS:
            if (is_float)
                id = glsl(GLSLstd450FAbs, operand(0));
            else if (is_signed_int)
                id = glsl(GLSLstd450SAbs, operand(0));
            else
                id = operand(0);// uint abs is identity
            break;
        case xir::ArithmeticOp::MIN:
            id = glsl_typed(GLSLstd450NMin, GLSLstd450SMin, GLSLstd450UMin,
                            operand(0), operand(1));
            break;
        case xir::ArithmeticOp::MAX:
            id = glsl_typed(GLSLstd450NMax, GLSLstd450SMax, GLSLstd450UMax,
                            operand(0), operand(1));
            break;
        case xir::ArithmeticOp::CLZ: {
            require_uint32_bit_operation();
            auto find_msb = glsl(GLSLstd450FindUMsb, operand(0));
            auto bit_width = static_cast<int32_t>(t->is_scalar() ? t->size() * 8 : t->element()->size() * 8);
            auto bit_width_id = make_integer_constant(t, static_cast<uint32_t>(bit_width));
            auto all_ones = bit_width == 64 ?
                                std::numeric_limits<uint64_t>::max() :
                                (uint64_t{1} << static_cast<uint32_t>(bit_width)) - 1u;
            auto minus_one = make_integer_constant(t, all_ones);
            auto bool_type = is_scalar ?
                                 _builder.makeBoolType() :
                                 _builder.makeVectorType(_builder.makeBoolType(), t->dimension());
            auto is_zero = _builder.createBinOp(spv::Op::OpIEqual, bool_type, find_msb, minus_one);
            auto one = make_integer_constant(t, 1u);
            auto clz_val = _builder.createBinOp(spv::Op::OpISub, type, _builder.createBinOp(spv::Op::OpISub, type, bit_width_id, one), find_msb);
            id = _builder.createTriOp(spv::Op::OpSelect, type, is_zero, bit_width_id, clz_val);
            break;
        }
        case xir::ArithmeticOp::CTZ: {
            require_uint32_bit_operation();
            auto find_lsb = glsl(GLSLstd450FindILsb, operand(0));
            auto bit_width = static_cast<int32_t>(t->is_scalar() ? t->size() * 8 : t->element()->size() * 8);
            auto bit_width_id = make_integer_constant(t, static_cast<uint32_t>(bit_width));
            auto all_ones = bit_width == 64 ?
                                std::numeric_limits<uint64_t>::max() :
                                (uint64_t{1} << static_cast<uint32_t>(bit_width)) - 1u;
            auto minus_one = make_integer_constant(t, all_ones);
            auto bool_type = is_scalar ?
                                 _builder.makeBoolType() :
                                 _builder.makeVectorType(_builder.makeBoolType(), t->dimension());
            auto is_zero = _builder.createBinOp(spv::Op::OpIEqual, bool_type, find_lsb, minus_one);
            id = _builder.createTriOp(spv::Op::OpSelect, type, is_zero, bit_width_id, find_lsb);
            break;
        }
        case xir::ArithmeticOp::POPCOUNT: {
            require_uint32_bit_operation();
            id = unary(spv::Op::OpBitCount);
            break;
        }
        case xir::ArithmeticOp::REVERSE: {
            require_uint32_bit_operation();
            id = unary(spv::Op::OpBitReverse);
            break;
        }
        case xir::ArithmeticOp::ISINF:
            id = unary(spv::Op::OpIsInf);
            break;
        case xir::ArithmeticOp::ISNAN:
            id = unary(spv::Op::OpIsNan);
            break;
        case xir::ArithmeticOp::ACOS:
            if (!_enable_fast_math && elem->is_float32()) {
                // Keep strict acos stable at both endpoints. Expressing it
                // as pi/2 - asin(x) loses every significant bit near +1;
                // range-reduce by magnitude instead, then select the
                // negative half of the domain without cancellation.
                auto x = operand(0);
                auto magnitude = glsl(GLSLstd450FAbs, x);
                auto zero = strict_float32_constant(0.0);
                auto half = strict_float32_constant(0.5);
                auto one = strict_float32_constant(1.0);
                auto two = strict_float32_constant(2.0);
                auto pi = strict_float32_constant(
                    3.14159265358979323846);
                auto bool_type = is_scalar ?
                                     _builder.makeBoolType() :
                                     _builder.makeVectorType(
                                         _builder.makeBoolType(),
                                         t->dimension());
                auto negative =
                    _builder.createBinOp(
                        spv::Op::OpFOrdLessThan,
                        bool_type,
                        x,
                        zero);
                auto reduced = glsl(
                    GLSLstd450Sqrt,
                    strict_float32_mul(
                        half,
                        strict_float32_sub(one, magnitude)));
                auto positive_result =
                    strict_float32_mul(
                        two,
                        strict_float32_asin(reduced));
                auto negative_result =
                    strict_float32_sub(pi, positive_result);
                id = _builder.createTriOp(
                    spv::Op::OpSelect,
                    type,
                    negative,
                    negative_result,
                    positive_result);
            } else {
                id = glsl(GLSLstd450Acos, operand(0));
            }
            break;
        case xir::ArithmeticOp::ACOSH:
            id = glsl(GLSLstd450Acosh, operand(0));
            break;
        case xir::ArithmeticOp::ASIN:
            id = !_enable_fast_math && elem->is_float32() ?
                     strict_float32_asin(operand(0)) :
                     glsl(GLSLstd450Asin, operand(0));
            break;
        case xir::ArithmeticOp::ASINH:
            id = glsl(GLSLstd450Asinh, operand(0));
            break;
        case xir::ArithmeticOp::ATAN:
            id = glsl(GLSLstd450Atan, operand(0));
            break;
        case xir::ArithmeticOp::ATAN2:
            id = glsl(GLSLstd450Atan2, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::ATANH:
            id = glsl(GLSLstd450Atanh, operand(0));
            break;
        case xir::ArithmeticOp::COS:
            id = glsl(GLSLstd450Cos, operand(0));
            break;
        case xir::ArithmeticOp::COSH:
            id = glsl(GLSLstd450Cosh, operand(0));
            break;
        case xir::ArithmeticOp::SIN:
            id = glsl(GLSLstd450Sin, operand(0));
            break;
        case xir::ArithmeticOp::SINH:
            id = glsl(GLSLstd450Sinh, operand(0));
            break;
        case xir::ArithmeticOp::TAN:
            id = glsl(GLSLstd450Tan, operand(0));
            break;
        case xir::ArithmeticOp::TANH:
            id = glsl(GLSLstd450Tanh, operand(0));
            break;
        case xir::ArithmeticOp::EXP:
            id = glsl(GLSLstd450Exp, operand(0));
            break;
        case xir::ArithmeticOp::EXP2:
            id = glsl(GLSLstd450Exp2, operand(0));
            break;
        case xir::ArithmeticOp::EXP10: {
            auto log2_10 = make_float_scalar_constant(elem, 3.321928094887362);
            spv::Id scaled;
            if (!is_scalar)
                scaled = _builder.createBinOp(spv::Op::OpVectorTimesScalar, type, operand(0), log2_10);
            else
                scaled = _builder.createBinOp(spv::Op::OpFMul, type, operand(0), log2_10);
            id = glsl(GLSLstd450Exp2, scaled);
            break;
        }
        case xir::ArithmeticOp::LOG:
            id = glsl(GLSLstd450Log, operand(0));
            break;
        case xir::ArithmeticOp::LOG2:
            id = glsl(GLSLstd450Log2, operand(0));
            break;
        case xir::ArithmeticOp::LOG10: {
            auto inv_log2_10 = make_float_scalar_constant(elem, 0.3010299956639812);
            auto log2_val = glsl(GLSLstd450Log2, operand(0));
            if (!is_scalar)
                id = _builder.createBinOp(spv::Op::OpVectorTimesScalar, type, log2_val, inv_log2_10);
            else
                id = _builder.createBinOp(spv::Op::OpFMul, type, log2_val, inv_log2_10);
            break;
        }
        case xir::ArithmeticOp::POW:
            id = glsl(GLSLstd450Pow, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::POW_INT: {
            auto exponent = operand(1);
            auto exponent_xir_type = inst->operand(1)->type();
            auto exponent_element = exponent_xir_type->is_vector() ? exponent_xir_type->element() : exponent_xir_type;
            auto exponent_type = _convert_type(exponent_xir_type, Usage::READ);
            auto exponent_scalar_type = _builder.getScalarTypeId(exponent_type);
            auto exponent_width = static_cast<uint32_t>(_builder.getScalarTypeWidth(exponent_scalar_type));
            auto exponent_unsigned_scalar_type = _builder.makeUintType(static_cast<int>(exponent_width));
            auto exponent_unsigned_type = exponent_xir_type->is_vector() ?
                                              _builder.makeVectorType(exponent_unsigned_scalar_type, exponent_xir_type->dimension()) :
                                              exponent_unsigned_scalar_type;
            auto bool_type = exponent_xir_type->is_vector() ?
                                 _builder.makeVectorType(_builder.makeBoolType(), exponent_xir_type->dimension()) :
                                 _builder.makeBoolType();
            auto make_exponent_constant = [&](uint64_t value) noexcept {
                auto scalar = exponent_width == 64u ?
                                  _builder.makeInt64Constant(exponent_unsigned_scalar_type, value, false) :
                                  _builder.makeIntConstant(exponent_unsigned_scalar_type, static_cast<uint32_t>(value), false);
                return exponent_xir_type->is_vector() ?
                           _builder.smearScalar(spv::NoPrecision, scalar, exponent_unsigned_type) :
                           scalar;
            };
            auto make_float_constant = [&](double value) noexcept {
                auto scalar = elem->is_float16() ?
                                  _builder.makeFloat16Constant(static_cast<float>(value)) :
                              elem->is_float64() ?
                                  _builder.makeDoubleConstant(value) :
                                  _builder.makeFloatConstant(static_cast<float>(value));
                return is_scalar ? scalar : _builder.smearScalar(spv::NoPrecision, scalar, type);
            };
            auto zero_unsigned = make_exponent_constant(0u);
            auto exponent_bits = _builder.createUnaryOp(spv::Op::OpBitcast, exponent_unsigned_type, exponent);
            spv::Id negative = spv::NoResult;
            spv::Id magnitude = exponent_bits;
            if (exponent_element->is_int()) {
                auto zero_signed_scalar = exponent_width == 64u ?
                                              _builder.makeInt64Constant(exponent_scalar_type, 0u, false) :
                                              _builder.makeIntConstant(exponent_scalar_type, 0u, false);
                auto zero_signed = exponent_xir_type->is_vector() ?
                                       _builder.smearScalar(spv::NoPrecision, zero_signed_scalar, exponent_type) :
                                       zero_signed_scalar;
                negative = _builder.createBinOp(spv::Op::OpSLessThan, bool_type, exponent, zero_signed);
                auto negated = _builder.createBinOp(spv::Op::OpISub, exponent_unsigned_type, zero_unsigned, exponent_bits);
                magnitude = _builder.createTriOp(spv::Op::OpSelect, exponent_unsigned_type, negative, negated, exponent_bits);
            }
            auto result = make_float_constant(1.0);
            auto factor = operand(0);
            for (uint32_t bit = 0u; bit < exponent_width; ++bit) {
                auto mask = make_exponent_constant(uint64_t{1} << bit);
                auto masked = _builder.createBinOp(spv::Op::OpBitwiseAnd, exponent_unsigned_type, magnitude, mask);
                auto bit_set = _builder.createBinOp(spv::Op::OpINotEqual, bool_type, masked, zero_unsigned);
                auto product = _builder.createBinOp(spv::Op::OpFMul, type, result, factor);
                result = _builder.createTriOp(spv::Op::OpSelect, type, bit_set, product, result);
                if (bit + 1u < exponent_width) {
                    factor = _builder.createBinOp(spv::Op::OpFMul, type, factor, factor);
                }
            }
            if (negative != spv::NoResult) {
                auto reciprocal = _builder.createBinOp(spv::Op::OpFDiv, type, make_float_constant(1.0), result);
                result = _builder.createTriOp(spv::Op::OpSelect, type, negative, reciprocal, result);
            }
            id = result;
            break;
        }
        case xir::ArithmeticOp::SQRT:
            id = glsl(GLSLstd450Sqrt, operand(0));
            break;
        case xir::ArithmeticOp::RSQRT:
            id = glsl(GLSLstd450InverseSqrt, operand(0));
            break;
        case xir::ArithmeticOp::CEIL:
            id = glsl(GLSLstd450Ceil, operand(0));
            break;
        case xir::ArithmeticOp::FLOOR:
            id = glsl(GLSLstd450Floor, operand(0));
            break;
        case xir::ArithmeticOp::FRACT:
            id = glsl(GLSLstd450Fract, operand(0));
            break;
        case xir::ArithmeticOp::TRUNC:
            id = glsl(GLSLstd450Trunc, operand(0));
            break;
        case xir::ArithmeticOp::ROUND: {
            auto value = operand(0);
            auto magnitude = glsl(GLSLstd450FAbs, value);
            auto integral = glsl(GLSLstd450Floor, magnitude);
            auto fraction = _builder.createBinOp(
                spv::Op::OpFSub, type, magnitude, integral);
            auto half = make_float_scalar_constant(elem, 0.5);
            auto one = make_float_scalar_constant(elem, 1.0);
            if (!is_scalar) {
                half = _builder.smearScalar(
                    spv::NoPrecision, half, type);
                one = _builder.smearScalar(
                    spv::NoPrecision, one, type);
            }
            auto bool_type = is_scalar ?
                                 _builder.makeBoolType() :
                                 _builder.makeVectorType(_builder.makeBoolType(), t->dimension());
            auto round_away = _builder.createBinOp(
                spv::Op::OpFOrdGreaterThanEqual, bool_type,
                fraction, half);
            auto next_integral = _builder.createBinOp(
                spv::Op::OpFAdd, type, integral, one);
            auto rounded_magnitude = _builder.createTriOp(
                spv::Op::OpSelect, type, round_away,
                next_integral, integral);
            id = copy_float_sign(rounded_magnitude, value);
            break;
        }
        case xir::ArithmeticOp::RINT:
            id = glsl(GLSLstd450RoundEven, operand(0));
            break;
        case xir::ArithmeticOp::FMA:
            id = glsl(GLSLstd450Fma, operand(0), operand(1), operand(2));
            break;
        case xir::ArithmeticOp::COPYSIGN:
            id = copy_float_sign(operand(0), operand(1));
            break;
        case xir::ArithmeticOp::CROSS:
            id = glsl(GLSLstd450Cross, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::DOT:
            id = mark_no_contraction(binary(spv::Op::OpDot));
            break;
        case xir::ArithmeticOp::LENGTH: {
            // Lower to native SPIR-V: sqrt(dot(v, v)). The native form lets
            // downstream optimizers CSE the dot product with sibling
            // LENGTH_SQUARED/NORMALIZE/DOT emissions, which an opaque
            // GLSL.std.450 ExtInst would block.
            auto a = operand(0);
            if (inst->operand(0)->type()->is_vector()) {
                auto dot = mark_no_contraction(
                    _builder.createBinOp(spv::Op::OpDot, type, a, a));
                id = make_glsl_call(GLSLstd450Sqrt, type, {dot});
            } else {
                id = glsl(GLSLstd450Length, a);
            }
            break;
        }
        case xir::ArithmeticOp::LENGTH_SQUARED: {
            auto a = operand(0);
            id = mark_no_contraction(
                _builder.createBinOp(spv::Op::OpDot, type, a, a));
            break;
        }
        case xir::ArithmeticOp::NORMALIZE: {
            // Lower to native SPIR-V: v * (1 / sqrt(dot(v, v))).
            auto a = operand(0);
            if (t->is_vector()) {
                auto scalar_type = _builder.getScalarTypeId(type);
                auto dot = mark_no_contraction(
                    _builder.createBinOp(spv::Op::OpDot, scalar_type, a, a));
                auto len = make_glsl_call(GLSLstd450Sqrt, scalar_type, {dot});
                auto one = make_float_scalar_constant(elem, 1.0);
                auto rcp = _builder.createBinOp(
                    spv::Op::OpFDiv, scalar_type, one, len);
                id = _builder.createBinOp(
                    spv::Op::OpVectorTimesScalar, type, a, rcp);
            } else {
                id = glsl(GLSLstd450Normalize, a);
            }
            break;
        }
        case xir::ArithmeticOp::FACEFORWARD:
            id = glsl(GLSLstd450FaceForward, operand(0), operand(1), operand(2));
            break;
        case xir::ArithmeticOp::REFLECT:
            id = glsl(GLSLstd450Reflect, operand(0), operand(1));
            break;

        // Reductions
        case xir::ArithmeticOp::REDUCE_SUM:
        case xir::ArithmeticOp::REDUCE_PRODUCT:
        case xir::ArithmeticOp::REDUCE_MIN:
        case xir::ArithmeticOp::REDUCE_MAX: {
            auto v = operand(0);
            auto operand_type = inst->operand(0)->type();
            auto elem_type = operand_type->element();
            auto elem_spv_type = _convert_type(elem_type, Usage::READ);
            auto dim = operand_type->dimension();
            auto extract = [&](uint32_t i) {
                return _builder.createCompositeExtract(v, elem_spv_type, i);
            };
            id = extract(0);
            for (auto i = 1u; i < dim; ++i) {
                auto comp = extract(i);
                switch (inst->op()) {
                    case xir::ArithmeticOp::REDUCE_SUM:
                        if (elem_type->is_float())
                            id = mark_no_contraction(
                                _builder.createBinOp(
                                    spv::Op::OpFAdd, elem_spv_type,
                                    id, comp));
                        else
                            id = _builder.createBinOp(spv::Op::OpIAdd, elem_spv_type, id, comp);
                        break;
                    case xir::ArithmeticOp::REDUCE_PRODUCT:
                        if (elem_type->is_float())
                            id = mark_no_contraction(
                                _builder.createBinOp(
                                    spv::Op::OpFMul, elem_spv_type,
                                    id, comp));
                        else
                            id = _builder.createBinOp(spv::Op::OpIMul, elem_spv_type, id, comp);
                        break;
                    case xir::ArithmeticOp::REDUCE_MIN:
                        if (elem_type->is_float())
                            id = make_glsl_call(GLSLstd450NMin, elem_spv_type, {id, comp});
                        else if (elem_type->is_int())
                            id = make_glsl_call(GLSLstd450SMin, elem_spv_type, {id, comp});
                        else
                            id = make_glsl_call(GLSLstd450UMin, elem_spv_type, {id, comp});
                        break;
                    case xir::ArithmeticOp::REDUCE_MAX:
                        if (elem_type->is_float())
                            id = make_glsl_call(GLSLstd450NMax, elem_spv_type, {id, comp});
                        else if (elem_type->is_int())
                            id = make_glsl_call(GLSLstd450SMax, elem_spv_type, {id, comp});
                        else
                            id = make_glsl_call(GLSLstd450UMax, elem_spv_type, {id, comp});
                        break;
                    default: break;
                }
            }
            break;
        }

        case xir::ArithmeticOp::OUTER_PRODUCT: {
            auto a = operand(0);
            auto b = operand(1);
            auto a_type = inst->operand(0)->type();
            auto b_type = inst->operand(1)->type();
            if (a_type->is_vector() && b_type->is_vector()) {
                id = mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpOuterProduct, type, a, b));
            } else {
                auto b_t = _builder.createUnaryOp(spv::Op::OpTranspose, type, b);
                id = mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpMatrixTimesMatrix, type, a, b_t));
            }
            break;
        }

        // Matrix operations
        case xir::ArithmeticOp::MATRIX_COMP_NEG: {
            auto mat = operand(0);
            auto rows = t->dimension();
            auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
            std::vector<spv::Id> new_rows;
            new_rows.reserve(rows);
            for (uint32_t i = 0u; i < rows; ++i) {
                auto row = _builder.createCompositeExtract(mat, row_type, i);
                new_rows.push_back(_builder.createUnaryOp(spv::Op::OpFNegate, row_type, row));
            }
            id = _builder.createCompositeConstruct(type, new_rows);
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_ADD: {
            auto a = operand(0);
            auto b = operand(1);
            auto a_type = inst->operand(0)->type();
            auto b_type = inst->operand(1)->type();
            auto rows = t->dimension();
            auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
            std::vector<spv::Id> new_rows;
            new_rows.reserve(rows);
            for (uint32_t i = 0u; i < rows; ++i) {
                spv::Id lhs, rhs;
                if (a_type->is_scalar()) {
                    lhs = _builder.smearScalar(spv::NoPrecision, a, row_type);
                } else {
                    lhs = _builder.createCompositeExtract(a, row_type, i);
                }
                if (b_type->is_scalar()) {
                    rhs = _builder.smearScalar(spv::NoPrecision, b, row_type);
                } else {
                    rhs = _builder.createCompositeExtract(b, row_type, i);
                }
                new_rows.push_back(mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpFAdd, row_type, lhs, rhs)));
            }
            id = _builder.createCompositeConstruct(type, new_rows);
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_SUB: {
            auto a = operand(0);
            auto b = operand(1);
            auto a_type = inst->operand(0)->type();
            auto b_type = inst->operand(1)->type();
            auto rows = t->dimension();
            auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
            std::vector<spv::Id> new_rows;
            new_rows.reserve(rows);
            for (uint32_t i = 0u; i < rows; ++i) {
                spv::Id lhs, rhs;
                if (a_type->is_scalar()) {
                    lhs = _builder.smearScalar(spv::NoPrecision, a, row_type);
                } else {
                    lhs = _builder.createCompositeExtract(a, row_type, i);
                }
                if (b_type->is_scalar()) {
                    rhs = _builder.smearScalar(spv::NoPrecision, b, row_type);
                } else {
                    rhs = _builder.createCompositeExtract(b, row_type, i);
                }
                new_rows.push_back(mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpFSub, row_type, lhs, rhs)));
            }
            id = _builder.createCompositeConstruct(type, new_rows);
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_MUL: {
            auto a = operand(0);
            auto b = operand(1);
            auto a_type = inst->operand(0)->type();
            auto b_type = inst->operand(1)->type();
            if (a_type->is_scalar()) {
                id = mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpMatrixTimesScalar, type, b, a));
            } else if (b_type->is_scalar()) {
                id = mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpMatrixTimesScalar, type, a, b));
            } else {
                auto rows = t->dimension();
                auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
                std::vector<spv::Id> new_rows;
                new_rows.reserve(rows);
                for (uint32_t i = 0u; i < rows; ++i) {
                    auto row_a = _builder.createCompositeExtract(a, row_type, i);
                    auto row_b = _builder.createCompositeExtract(b, row_type, i);
                    new_rows.push_back(mark_no_contraction(
                        _builder.createBinOp(
                            spv::Op::OpFMul, row_type, row_a, row_b)));
                }
                id = _builder.createCompositeConstruct(type, new_rows);
            }
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_DIV: {
            auto a = operand(0);
            auto b = operand(1);
            auto a_type = inst->operand(0)->type();
            auto b_type = inst->operand(1)->type();
            if (b_type->is_scalar()) {
                auto rows = t->dimension();
                auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
                std::vector<spv::Id> new_rows;
                new_rows.reserve(rows);
                for (uint32_t i = 0u; i < rows; ++i) {
                    auto row = _builder.createCompositeExtract(a, row_type, i);
                    auto smeared = _builder.smearScalar(spv::NoPrecision, b, row_type);
                    new_rows.push_back(_builder.createBinOp(spv::Op::OpFDiv, row_type, row, smeared));
                }
                id = _builder.createCompositeConstruct(type, new_rows);
            } else {
                auto rows = t->dimension();
                auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
                std::vector<spv::Id> new_rows;
                new_rows.reserve(rows);
                for (uint32_t i = 0u; i < rows; ++i) {
                    auto row_a = a_type->is_scalar() ?
                                     _builder.smearScalar(spv::NoPrecision, a, row_type) :
                                     _builder.createCompositeExtract(a, row_type, i);
                    auto row_b = _builder.createCompositeExtract(b, row_type, i);
                    new_rows.push_back(_builder.createBinOp(spv::Op::OpFDiv, row_type, row_a, row_b));
                }
                id = _builder.createCompositeConstruct(type, new_rows);
            }
            break;
        }
        case xir::ArithmeticOp::MATRIX_LINALG_MUL: {
            auto a_type = inst->operand(0)->type();
            auto b_type = inst->operand(1)->type();
            if (a_type->is_vector() && b_type->is_matrix()) {
                id = mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpVectorTimesMatrix, type,
                        operand(0), operand(1)));
            } else if (a_type->is_matrix() && b_type->is_vector()) {
                id = mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpMatrixTimesVector, type,
                        operand(0), operand(1)));
            } else {
                LUISA_ASSERT(a_type->is_matrix() && b_type->is_matrix(),
                             "SPIR-V dialect validation accepted invalid "
                             "matrix multiplication operands {} and {}.",
                             a_type->description(), b_type->description());
                id = mark_no_contraction(
                    _builder.createBinOp(
                        spv::Op::OpMatrixTimesMatrix, type,
                        operand(0), operand(1)));
            }
            break;
        }
        case xir::ArithmeticOp::MATRIX_DETERMINANT:
            id = glsl(GLSLstd450Determinant, operand(0));
            break;
        case xir::ArithmeticOp::MATRIX_TRANSPOSE:
            id = unary(spv::Op::OpTranspose);
            break;
        case xir::ArithmeticOp::MATRIX_INVERSE:
            id = glsl(GLSLstd450MatrixInverse, operand(0));
            break;

        // Composite operations
        case xir::ArithmeticOp::AGGREGATE: {
            std::vector<spv::Id> components;
            components.reserve(inst->operand_count());
            for (auto i = 0u; i < inst->operand_count(); ++i) {
                components.emplace_back(operand(i));
            }
            id = _builder.createCompositeConstruct(type, components);
            break;
        }
        case xir::ArithmeticOp::SHUFFLE: {
            auto v = operand(0);
            auto dim = t->dimension();
            // Peephole: if all indices are constants, emit OpVectorShuffle directly
            bool all_const = true;
            luisa::vector<uint32_t> shuffle_indices;
            shuffle_indices.reserve(dim);
            for (auto i = 1u; i <= dim; ++i) {
                if (inst->operand(i)->isa<xir::Constant>()) {
                    auto index = decode_constant_index(
                        static_cast<const xir::Constant *>(inst->operand(i)));
                    auto source_type = inst->operand(0)->type();
                    if (index >= source_type->dimension()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "Shuffle index {} is out of bounds for {}.",
                            index, source_type->description());
                    }
                    shuffle_indices.push_back(static_cast<uint32_t>(index));
                } else {
                    all_const = false;
                    break;
                }
            }
            if (all_const) {
                std::vector<uint32_t> std_indices(shuffle_indices.begin(), shuffle_indices.end());
                id = _builder.createRvalueSwizzle(spv::NoPrecision, type, v, std_indices);
            } else {
                std::vector<spv::Id> comps;
                comps.reserve(dim);
                for (uint32_t i = 0u; i < dim; ++i) {
                    auto idx = _emit_value(inst->operand(i + 1u));
                    comps.push_back(_builder.createVectorExtractDynamic(v, _convert_type(t->element(), Usage::READ), idx));
                }
                id = _builder.createCompositeConstruct(type, comps);
            }
            break;
        }
        case xir::ArithmeticOp::INSERT: {
            auto index_plan = make_aggregate_index_plan(
                inst->operand(0)->type(), 2u);
            LUISA_ASSERT(index_plan.indexed_type == inst->operand(1)->type(),
                         "SPIR-V INSERT index plan reaches {}, but the inserted value is {}.",
                         index_plan.indexed_type->description(),
                         inst->operand(1)->type()->description());
            auto v = operand(0);
            auto e = operand(1);
            std::vector<uint32_t> const_indices;
            if (index_plan.all_constant()) {
                const_indices.reserve(index_plan.steps.size());
                for (auto &&step : index_plan.steps) {
                    LUISA_ASSERT(step.constant_index <=
                                     std::numeric_limits<uint32_t>::max(),
                                 "SPIR-V composite-insert literal index is too wide.");
                    const_indices.emplace_back(
                        static_cast<uint32_t>(step.constant_index));
                }
                id = _builder.createCompositeInsert(e, v, type, const_indices);
            } else {
                // INSERT is a pure SSA operation. Never reuse and mutate an alloca
                // that happened to supply the base value: other loads may still
                // need to observe the original aggregate.
                auto temp_var = _builder.createVariable(
                    spv::NoPrecision, spv::StorageClass::Function, type, "insert_tmp");
                _builder.createStore(v, temp_var);
                auto access_indices =
                    _emit_aggregate_access_indices(index_plan);
                auto ptr = _create_access_chain(
                    spv::StorageClass::Function, temp_var, access_indices);
                _builder.createStore(e, ptr);
                id = _builder.createLoad(temp_var, spv::NoPrecision);
            }
            break;
        }
        case xir::ArithmeticOp::EXTRACT: {
            auto base_value = inst->operand(0);
            auto base_type = base_value->type();
            auto index_plan = make_aggregate_index_plan(base_type, 1u);
            LUISA_ASSERT(index_plan.indexed_type == inst->type(),
                         "SPIR-V EXTRACT index plan reaches {}, but the result is {}.",
                         index_plan.indexed_type->description(),
                         inst->type()->description());

            // Fast path for UBO-lowered constant arrays: emit a single indexed load
            // through the constant cache instead of materializing the array as an SSA value.
            if (base_value->isa<xir::Constant>()) {
                auto c = static_cast<const xir::Constant *>(base_value);
                if (auto ubo_it = _ubo_constant_member_indices.find(c);
                    ubo_it != _ubo_constant_member_indices.end()) {
                    std::vector<spv::Id> indices;
                    indices.reserve(inst->operand_count());
                    indices.push_back(_builder.makeUintConstant(ubo_it->second));
                    auto aggregate_indices =
                        _emit_aggregate_access_indices(index_plan);
                    indices.insert(indices.end(), aggregate_indices.begin(),
                                   aggregate_indices.end());
                    auto ptr = _create_access_chain(spv::StorageClass::Uniform, _constant_ubo_var, indices);
                    id = _builder.createLoad(ptr, spv::NoPrecision);
                    break;
                }
            }

            auto v = operand(0);
            std::vector<uint32_t> const_indices;
            if (index_plan.all_constant()) {
                const_indices.reserve(index_plan.steps.size());
                for (auto &&step : index_plan.steps) {
                    LUISA_ASSERT(step.constant_index <=
                                     std::numeric_limits<uint32_t>::max(),
                                 "SPIR-V composite-extract literal index is too wide.");
                    const_indices.emplace_back(
                        static_cast<uint32_t>(step.constant_index));
                }
            }
            if (index_plan.all_constant()) {
                id = _builder.createCompositeExtract(v, type, const_indices);
            } else {
                auto dynamic_indices =
                    _emit_aggregate_access_indices(index_plan);
                if (base_type->is_vector()) {
                    LUISA_ASSERT(
                        dynamic_indices.size() == 1u,
                        "SPIR-V vector extract should have only one index.");
                    id = _builder.createVectorExtractDynamic(
                        v, type, dynamic_indices[0]);
                } else if (base_type->is_array() || base_type->is_matrix()) {
                    // For small arrays/matrices, lower a single dynamic index
                    // to an OpSelect chain. This stays in registers and avoids
                    // a Function-scope memory round-trip on most drivers.
                    auto elem_count = base_type->dimension();
                    if (dynamic_indices.size() == 1u &&
                        elem_count <= 16u) {
                        auto idx = dynamic_indices[0];
                        auto result = _builder.createCompositeExtract(
                            v, type, {0u});
                        for (uint32_t i = 1u; i < elem_count; ++i) {
                            auto elem_i = _builder.createCompositeExtract(
                                v, type, {i});
                            auto cmp = _builder.createBinOp(
                                spv::Op::OpIEqual,
                                _builder.makeBoolType(), idx,
                                make_integer_constant(
                                    inst->operand(1)->type(), i));
                            result = _builder.createTriOp(
                                spv::Op::OpSelect, type, cmp, elem_i,
                                result);
                        }
                        id = result;
                    } else {
                        auto temp_var = _builder.createVariable(
                            spv::NoPrecision,
                            spv::StorageClass::Function,
                            _convert_type(base_type, Usage::READ),
                            "extract_tmp");
                        _builder.createStore(v, temp_var);
                        auto ptr = _create_access_chain(
                            spv::StorageClass::Function, temp_var,
                            dynamic_indices);
                        id = _builder.createLoad(ptr, spv::NoPrecision);
                    }
                } else if (base_type->is_structure()) {
                    auto temp_var = _builder.createVariable(
                        spv::NoPrecision, spv::StorageClass::Function,
                        _convert_type(base_type, Usage::READ),
                        "extract_tmp");
                    _builder.createStore(v, temp_var);
                    auto ptr = _create_access_chain(
                        spv::StorageClass::Function, temp_var,
                        dynamic_indices);
                    id = _builder.createLoad(ptr, spv::NoPrecision);
                } else if (base_type->tag() == Type::Tag::COOPERATIVE_VECTOR) {
                    // cooperative vectors support OpCompositeExtract but not
                    // OpVectorExtractDynamic; round-trip through Function memory
                    auto temp_var = _builder.createVariable(
                        spv::NoPrecision, spv::StorageClass::Function,
                        _convert_type(base_type, Usage::READ),
                        "extract_tmp");
                    _builder.createStore(v, temp_var);
                    auto ptr = _create_access_chain(
                        spv::StorageClass::Function, temp_var,
                        dynamic_indices);
                    id = _builder.createLoad(ptr, spv::NoPrecision);
                } else {
                    LUISA_NOT_IMPLEMENTED(
                        "SPIR-V dynamic extract for type {}.",
                        base_type->description());
                }
            }
            break;
        }

        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V arithmetic op {}.", xir::to_string(inst->op()));
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit arithmetic op.");
    if (!_enable_fast_math && is_float &&
        (inst->op() == xir::ArithmeticOp::BINARY_ADD ||
         inst->op() == xir::ArithmeticOp::BINARY_SUB ||
         inst->op() == xir::ArithmeticOp::BINARY_MUL)) {
        mark_no_contraction(id);
    }
    if (inst->type() != nullptr) {
        _value_map.emplace(inst, id);
    }
}

const SpirvCodegenEntry::KernelResourceBinding &
SpirvCodegenEntry::_kernel_resource_binding(
    const xir::Argument *argument) const noexcept {
    LUISA_ASSERT(_is_kernel_resource_argument(argument),
                 "SPIR-V resource binding lookup requires a resource argument.");
    auto *function = argument->parent_function();
    LUISA_ASSERT(function != nullptr &&
                     function->derived_function_tag() ==
                         xir::DerivedFunctionTag::KERNEL,
                 "SPIR-V global resource binding lookup is only valid for "
                 "kernel arguments. Callable resource accesses requiring a "
                 "second descriptor must be specialized at the call site.");
    auto resource_index = 0u;
    for (auto *candidate : function->arguments()) {
        if (!_is_kernel_resource_argument(candidate)) { continue; }
        if (candidate == argument) {
            LUISA_ASSERT(resource_index < _kernel_resource_bindings.size(),
                         "SPIR-V kernel resource binding {} is out of range ({} planned).",
                         resource_index, _kernel_resource_bindings.size());
            auto &binding = _kernel_resource_bindings[resource_index];
            LUISA_ASSERT(binding.type_tag == argument->type()->tag(),
                         "SPIR-V kernel resource binding {} has type tag {}, expected {}.",
                         resource_index,
                         static_cast<uint32_t>(binding.type_tag),
                         static_cast<uint32_t>(argument->type()->tag()));
            return binding;
        }
        ++resource_index;
    }
    LUISA_ERROR_WITH_LOCATION(
        "SPIR-V resource argument was not found in its kernel.");
}

spv::Id SpirvCodegenEntry::_kernel_resource_property_id(
    size_t property_index) const noexcept {
    LUISA_ASSERT(property_index != invalid_resource_property_index &&
                     property_index < _properties.size(),
                 "SPIR-V kernel resource property index {} is invalid ({} properties).",
                 property_index, _properties.size());
    // `_property_ids[0]` is the push constant, which is deliberately absent
    // from `_properties`. This is the only place where that representation
    // boundary is crossed.
    auto id_index = property_index + 1u;
    LUISA_ASSERT(id_index < _property_ids.size() &&
                     _property_ids[id_index] != spv::NoResult,
                 "SPIR-V kernel resource property {} has no emitted variable.",
                 property_index);
    return _property_ids[id_index];
}

spv::Id SpirvCodegenEntry::_resolve_resource_argument(
    const xir::Argument *arg) noexcept {
    if (auto it = _value_map.find(arg); it != _value_map.end()) {
        return it->second;
    }
    if (auto origin = _readonly_resource_origins.find(arg);
        origin != _readonly_resource_origins.end()) {
        auto id = _resolve_resource_argument(origin->second);
        _value_map.emplace(arg, id);
        return id;
    }
    auto &binding = _kernel_resource_binding(arg);
    auto property_index =
        binding.read_property_index != invalid_resource_property_index ?
            binding.read_property_index :
            binding.write_property_index;
    LUISA_ASSERT(property_index != invalid_resource_property_index,
                 "SPIR-V resource argument of type '{}' has no primary "
                 "read or write descriptor.",
                 arg->type()->description());
    auto id = _kernel_resource_property_id(property_index);
    auto property_type = _properties[property_index].type;
    if (property_type == ShaderVariableType::UAVTextureHeap) {
        _is_storage_image_map[id] = true;
    } else if (property_type == ShaderVariableType::SRVTextureHeap) {
        _is_storage_image_map[id] = false;
    }
    if (arg->type()->is_buffer()) {
        auto metadata_index = 0u;
        for (auto a : arg->parent_function()->arguments()) {
            if (a == arg) { break; }
            if (a->type()->is_buffer()) { ++metadata_index; }
        }
        _direct_buffer_metadata_indices.emplace(id, metadata_index);
    } else if (arg->type()->is_bindless_array() &&
               binding.bindless_buffer_metadata_property_index !=
                   invalid_resource_property_index) {
        auto metadata_id = _kernel_resource_property_id(
            binding.bindless_buffer_metadata_property_index);
        _bindless_buffer_metadata_ids.emplace(id, metadata_id);
    }
    _value_map.emplace(arg, id);
    return id;
}

size_t SpirvCodegenEntry::_direct_buffer_bias_alignment(
    const xir::Value *resource) const noexcept {
    if (resource == nullptr || !resource->isa<xir::Argument>()) {
        return 1u;
    }
    auto *argument = static_cast<const xir::Argument *>(resource);
    if (auto iter = _bound_direct_buffer_bias_alignments.find(argument);
        iter != _bound_direct_buffer_bias_alignments.end()) {
        return iter->second;
    }
    if (auto origin = _readonly_resource_origins.find(argument);
        origin != _readonly_resource_origins.end()) {
        return _direct_buffer_bias_alignment(origin->second);
    }
    return 1u;
}

spv::Id SpirvCodegenEntry::_resolve_writable_resource(
    const xir::Value *resource) noexcept {
    LUISA_ASSERT(resource != nullptr && resource->type() != nullptr &&
                     resource->type()->is_resource(),
                 "SPIR-V writable resource lookup requires a resource value.");
    LUISA_ASSERT(resource->isa<xir::Argument>(),
                 "SPIR-V writable resource was not an argument; resource "
                 "specialization must run before code generation.");
    auto *argument = static_cast<const xir::Argument *>(resource);
    auto *function = argument->parent_function();
    if (function != nullptr &&
        function->derived_function_tag() == xir::DerivedFunctionTag::KERNEL) {
        auto &binding = _kernel_resource_binding(argument);
        LUISA_ASSERT(binding.write_property_index !=
                         invalid_resource_property_index,
                     "SPIR-V resource argument of type '{}' has no writable descriptor.",
                     argument->type()->description());
        auto id = _kernel_resource_property_id(binding.write_property_index);
        if (_properties[binding.write_property_index].type ==
            ShaderVariableType::UAVTextureHeap) {
            _is_storage_image_map[id] = true;
        }
        return id;
    }
    auto id = _emit_value(resource);
    if (resource->type()->is_texture()) {
        auto iter = _is_storage_image_map.find(id);
        LUISA_ASSERT(iter != _is_storage_image_map.end() && iter->second,
                     "SPIR-V callable texture write requires a storage-image "
                     "parameter. A callable that both reads/samples and writes "
                     "one texture must be specialized at the call site.");
    }
    return id;
}

spv::Id SpirvCodegenEntry::_resolve_accel_instance_buffer(
    const xir::Value *accel) noexcept {
    LUISA_ASSERT(accel != nullptr && accel->type() != nullptr &&
                     accel->type()->is_accel(),
                 "SPIR-V acceleration-structure instance-buffer lookup "
                 "requires an accel value.");
    LUISA_ASSERT(accel->isa<xir::Argument>(),
                 "SPIR-V accel instance access was not specialized to a "
                 "resource argument before code generation.");
    auto *argument = static_cast<const xir::Argument *>(accel);
    auto &binding = _kernel_resource_binding(argument);
    LUISA_ASSERT(binding.accel_instance_property_index !=
                     invalid_resource_property_index,
                 "SPIR-V accel argument has no instance-buffer descriptor.");
    return _kernel_resource_property_id(
        binding.accel_instance_property_index);
}

spv::Id SpirvCodegenEntry::_emit_float_atomic_cas_loop(
    spv::Id ptr, spv::Id val, spv::Id float_type, xir::AtomicOp op,
    spv::Id scope, spv::Id load_semantics,
    spv::Id equal_semantics, spv::Id unequal_semantics) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto bool_type = _builder.makeBoolType();
    LUISA_ASSERT(float_type == _builder.makeFloatType(32),
                 "SPIR-V CAS loop only supports float32 for non-scalar buffer atomics.");

    auto *preheader = _builder.getBuildPoint();
    LUISA_ASSERT(preheader != nullptr && !preheader->isTerminated(),
                 "SPIR-V float atomic CAS loop requires an unterminated preheader.");
    auto initial_uint = _builder.createOp(
        spv::Op::OpAtomicLoad, uint_type,
        {ptr, scope, load_semantics});

    auto *loop_header = _create_physical_block();
    auto *loop_body = _create_physical_block();
    auto *loop_continue = _create_physical_block();
    auto *merge = _create_physical_block();

    _builder.createBranch(false, loop_header);

    _set_current_tail(loop_header);
    auto expected_phi = std::make_unique<spv::Instruction>(
        _builder.getUniqueId(), uint_type, spv::Op::OpPhi);
    auto *expected_phi_inst = expected_phi.get();
    auto expected_uint = expected_phi_inst->getResultId();
    expected_phi_inst->reserveOperands(4u);
    expected_phi_inst->addIdOperand(initial_uint);
    expected_phi_inst->addIdOperand(preheader->getId());
    loop_header->addInstruction(std::move(expected_phi));
    _builder.createLoopMerge(merge, loop_continue, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, loop_body);

    _set_current_tail(loop_body);
    auto old_float = _builder.createUnaryOp(
        spv::Op::OpBitcast, float_type, expected_uint);

    // Compute new_float
    spv::Id new_float;
    switch (op) {
        case xir::AtomicOp::FETCH_ADD:
            new_float = _builder.createBinOp(spv::Op::OpFAdd, float_type, old_float, val);
            break;
        case xir::AtomicOp::FETCH_SUB:
            new_float = _builder.createBinOp(spv::Op::OpFSub, float_type, old_float, val);
            break;
        case xir::AtomicOp::FETCH_MAX:
            new_float = _builder.createBuiltinCall(
                float_type, _glsl450, GLSLstd450NMax,
                {old_float, val});
            break;
        case xir::AtomicOp::FETCH_MIN:
            new_float = _builder.createBuiltinCall(
                float_type, _glsl450, GLSLstd450NMin,
                {old_float, val});
            break;
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V CAS loop for atomic op {}.", xir::to_string(op));
    }

    auto new_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, new_float);

    auto result = _builder.createOp(spv::Op::OpAtomicCompareExchange, uint_type,
                                    {ptr, scope, equal_semantics,
                                     unequal_semantics, new_uint,
                                     expected_uint});
    auto cmp = _builder.createBinOp(
        spv::Op::OpIEqual, bool_type, result, expected_uint);

    _builder.createConditionalBranch(cmp, merge, loop_continue);

    // Loop continue
    _set_current_tail(loop_continue);
    _builder.createBranch(false, loop_header);
    expected_phi_inst->addIdOperand(result);
    expected_phi_inst->addIdOperand(loop_continue->getId());

    // Merge
    _set_current_tail(merge);

    // On success, the compare-exchange result is bit-identical to the
    // expected value carried by the header Phi. Return that old float value.
    return _builder.createUnaryOp(
        spv::Op::OpBitcast, float_type, expected_uint);
}

void SpirvCodegenEntry::_emit_atomic_inst(const xir::AtomicInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    auto t = inst->type();
    if (t->is_float() && !t->is_float32()) {
        LUISA_ERROR_WITH_LOCATION(
            "Vulkan XIR-to-SPIR-V float atomics currently support float32 "
            "only; {} is unsupported. Sub-32-bit floats (including float16) "
            "require packed-word or exact native storage lowering, and "
            "float64 requires a 64-bit atomic representation.",
            t->description());
    }

    auto base = _emit_value(inst->base());
    spv::Id ptr = base;
    auto indices = inst->index_uses();
    auto base_xir_type = inst->base()->type();
    luisa::vector<const xir::Value *> index_values;
    index_values.reserve(indices.size());
    for (auto index_use : indices) {
        index_values.emplace_back(index_use->value());
    }
    auto index_plan = plan_spirv_aggregate_indices(
        base_xir_type, luisa::span{index_values});
    if (!index_plan) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid SPIR-V atomic aggregate indices: {}",
            index_plan.diagnostic);
    }
    LUISA_ASSERT(index_plan.indexed_type == t,
                 "SPIR-V atomic index plan reaches {}, but the atomic leaf is {}.",
                 index_plan.indexed_type->description(), t->description());
    bool uses_word_storage_buffer = false;
    if (base_xir_type != nullptr && base_xir_type->is_buffer() && base_xir_type->element() != nullptr) {
        uses_word_storage_buffer = _buffer_uses_word_storage(base_xir_type);
    }
    LUISA_ASSERT(!uses_word_storage_buffer || !indices.empty(),
                 "SPIR-V word-storage atomic requires a buffer element index.");
    if (!indices.empty()) {
        std::vector<spv::Id> idx_ids;
        // Buffer variables are pointers to structs containing a runtime array.
        // Prepend 0 to access the first (and only) struct member.
        auto base_type = _builder.getTypeId(base);
        auto pointee_type = _builder.getContainedTypeId(base_type);
        if (_builder.isStructType(pointee_type)) {
            idx_ids.push_back(_builder.makeUintConstant(0u));
        }

        if (uses_word_storage_buffer) {
            LUISA_ASSERT(base_xir_type != nullptr && base_xir_type->element() != nullptr,
                         "SPIR-V word-storage atomic requires a typed buffer base.");
            LUISA_ASSERT(!index_plan.steps.empty() &&
                             index_plan.steps.front().aggregate_type->is_buffer(),
                         "SPIR-V word-storage atomic index plan must begin at its buffer element.");
            auto uint_type = _builder.makeUintType(32);
            auto address_width = 32u;
            for (auto &&step : index_plan.steps) {
                if (step.kind ==
                    SpirvAggregateIndexKind::STRUCTURE_MEMBER) {
                    // Structure members contribute a compile-time byte offset;
                    // their source index IDs are never part of address math.
                    continue;
                }
                auto index_xir_type = step.index->type();
                address_width = std::max(
                    address_width,
                    static_cast<uint32_t>(index_xir_type->size() * 8u));
            }
            LUISA_ASSERT(address_width == 32u || address_width == 64u,
                         "SPIR-V word-storage atomic indices must be at most 64-bit.");
            auto address_type = address_width == 64u ?
                                    _builder.makeUintType(64) :
                                    uint_type;
            auto element_index = _emit_value(
                index_plan.steps.front().index);
            auto index_type = _builder.getTypeId(element_index);
            LUISA_ASSERT(_builder.isIntType(index_type) || _builder.isUintType(index_type),
                         "SPIR-V word-storage atomic index must be an integer scalar.");
            element_index = _ensure_type(element_index, address_type);
            auto make_address_constant = [&](size_t value) noexcept {
                return address_width == 64u ?
                           _builder.makeUint64Constant(static_cast<uint64_t>(value)) :
                           _builder.makeUintConstant(static_cast<uint32_t>(value));
            };
            auto add_static_offset = [&](spv::Id address, size_t offset) noexcept {
                return offset == 0u ? address :
                                      _builder.createBinOp(
                                          spv::Op::OpIAdd, address_type, address,
                                          make_address_constant(offset));
            };
            auto normalize_index = [&](spv::Id index) noexcept {
                return _ensure_type(index, address_type);
            };

            auto element_type = index_plan.steps.front().indexed_type;
            auto byte_offset = _builder.createBinOp(
                spv::Op::OpIMul, address_type, element_index,
                make_address_constant(element_type->size()));
            auto byte_alignment = std::gcd(element_type->size(), size_t{4u});
            for (size_t i = 1u; i < index_plan.steps.size(); ++i) {
                auto &&step = index_plan.steps[i];
                LUISA_ASSERT(step.aggregate_type == element_type,
                             "SPIR-V atomic index plan lost its aggregate type walk.");
                if (step.kind ==
                    SpirvAggregateIndexKind::STRUCTURE_MEMBER) {
                    auto member_index = static_cast<size_t>(
                        step.constant_index);
                    auto members = step.aggregate_type->members();
                    size_t member_offset = 0u;
                    for (auto j = 0u; j < member_index; ++j) {
                        member_offset = luisa::align(member_offset, members[j]->alignment());
                        member_offset += members[j]->size();
                    }
                    member_offset = luisa::align(member_offset, members[member_index]->alignment());
                    byte_offset = add_static_offset(byte_offset, member_offset);
                    byte_alignment = std::gcd(byte_alignment, member_offset);
                } else {
                    auto dynamic_index = normalize_index(
                        _emit_value(step.index));
                    auto relative_offset = _builder.createBinOp(
                        spv::Op::OpIMul, address_type, dynamic_index,
                        make_address_constant(step.indexed_type->size()));
                    byte_offset = _builder.createBinOp(
                        spv::Op::OpIAdd, address_type, byte_offset, relative_offset);
                    byte_alignment = std::gcd(
                        byte_alignment, step.indexed_type->size());
                }
                element_type = step.indexed_type;
            }
            LUISA_ASSERT(element_type == t,
                         "SPIR-V word-storage atomic target type {} does not match instruction type {}.",
                         element_type->description(), t->description());
            LUISA_ASSERT(t->size() == 4u && byte_alignment >= 4u,
                         "SPIR-V word-storage atomics require a naturally aligned 32-bit scalar target, got {} at byte alignment {}.",
                         t->description(), byte_alignment);
            byte_offset = _add_direct_buffer_bias(base, byte_offset);
            auto word_offset = _builder.createBinOp(
                spv::Op::OpUDiv, address_type, byte_offset,
                make_address_constant(4u));
            idx_ids.push_back(word_offset);
        } else {
            auto aggregate_indices =
                _emit_aggregate_access_indices(index_plan);
            idx_ids.insert(idx_ids.end(), aggregate_indices.begin(),
                           aggregate_indices.end());
            if (base_xir_type != nullptr && base_xir_type->is_buffer() &&
                base_xir_type->element() != nullptr &&
                _direct_buffer_metadata_indices.contains(base)) {
                LUISA_ASSERT(idx_ids.size() >= 2u,
                             "SPIR-V typed buffer atomic has no element index.");
                auto index_type = _builder.getTypeId(idx_ids[1]);
                LUISA_ASSERT(_builder.isIntType(index_type) ||
                                 _builder.isUintType(index_type),
                             "SPIR-V typed buffer atomic index must be integer.");
                auto address_type = _builder.getScalarTypeWidth(index_type) == 64u ?
                                        _builder.makeUintType(64) :
                                        _builder.makeUintType(32);
                auto bias = _load_direct_buffer_metadata(
                    base, StorageBufferMetadataField::DESCRIPTOR_BIAS_BYTES,
                    address_type);
                auto stride = base_xir_type->element()->size();
                auto stride_id = _builder.getScalarTypeWidth(address_type) == 64u ?
                                     _builder.makeUint64Constant(stride) :
                                     _builder.makeUintConstant(
                                         static_cast<uint32_t>(stride));
                auto element_bias = _builder.createBinOp(
                    spv::Op::OpUDiv, address_type, bias, stride_id);
                idx_ids[1] = _builder.createBinOp(
                    spv::Op::OpIAdd, address_type,
                    _ensure_type(idx_ids[1], address_type), element_bias);
            }
        }

        auto storage = _builder.getStorageClass(base);
        ptr = _create_access_chain(storage, base, idx_ids);
    }

    auto pointer_storage = _builder.getStorageClass(ptr);
    auto atomic_scope = pointer_storage == spv::StorageClass::Workgroup ?
                            spv::Scope::Workgroup :
                            spv::Scope::Device;
    auto scope = _builder.makeUintConstant(static_cast<uint32_t>(atomic_scope));
    // XIR atomic instructions guarantee atomicity but expose no memory-order
    // operand. Match the CUDA/HIP Monotonic and fallback Relaxed contract;
    // block synchronization and resource barriers own visibility ordering.
    // Adding AcquireRelease and broad memory-class bits here would silently
    // strengthen every RMW and can serialize otherwise independent atomics.
    auto semantics = _builder.makeUintConstant(
        static_cast<uint32_t>(spv::MemorySemanticsMask::MaskNone));
    auto semantics_equal = semantics;
    auto semantics_unequal = semantics;

    spv::Id id = spv::NoResult;
    auto values = inst->value_uses();
    auto uint_type = _builder.makeUintType(32);
    auto to_word_bits = [&](spv::Id value) noexcept {
        return _builder.getTypeId(value) == uint_type ?
                   value :
                   _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, value);
    };
    auto from_word_bits = [&](spv::Id value) noexcept {
        return type == uint_type ?
                   value :
                   _builder.createUnaryOp(spv::Op::OpBitcast, type, value);
    };
    auto emit_word_atomic = [&](spv::Op op, spv::Id value) noexcept {
        auto result = _builder.createOp(
            op, uint_type, {ptr, scope, semantics, to_word_bits(value)});
        return from_word_bits(result);
    };

    if (t->is_float()) {
        auto bit_width = static_cast<uint32_t>(t->size() * 8u);
        auto storage = [&]() noexcept {
            switch (pointer_storage) {
                case spv::StorageClass::StorageBuffer:
                    if (base_xir_type != nullptr &&
                        base_xir_type->is_buffer()) {
                        return SpirvFloatAtomicStorage::BUFFER;
                    }
                    break;
                case spv::StorageClass::Workgroup:
                    return SpirvFloatAtomicStorage::SHARED;
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V float atomic {} uses unsupported storage class {}. "
                "Only StorageBuffer and Workgroup are represented by the "
                "float-atomic lowering contract.",
                xir::to_string(inst->op()),
                static_cast<uint32_t>(pointer_storage));
        }();
        auto implementation = plan_spirv_float_atomic(
            inst->op(), bit_width, storage, _target_features);
        if (!uses_word_storage_buffer &&
            storage == SpirvFloatAtomicStorage::BUFFER &&
            implementation == SpirvFloatAtomicImplementation::WORD_CAS) {
            // A vendor preference may request word CAS, but the module-wide
            // buffer plan can retain typed storage when another leaf (notably
            // an int64 atomic) has no word representation. In that case the
            // exact enabled native operation is the only legal lowering.
            auto capability_implementation =
                plan_spirv_float_atomic_capability_driven(
                    inst->op(), bit_width, storage, _target_features);
            if (spirv_float_atomic_implementation_is_native(
                    capability_implementation)) {
                implementation = capability_implementation;
            }
        }
        if (uses_word_storage_buffer && bit_width == 32u) {
            switch (inst->op()) {
                case xir::AtomicOp::EXCHANGE:
                    implementation =
                        SpirvFloatAtomicImplementation::WORD_EXCHANGE;
                    break;
                case xir::AtomicOp::COMPARE_EXCHANGE:
                    implementation = SpirvFloatAtomicImplementation::
                        WORD_COMPARE_EXCHANGE;
                    break;
                case xir::AtomicOp::FETCH_ADD:
                case xir::AtomicOp::FETCH_SUB:
                case xir::AtomicOp::FETCH_MIN:
                case xir::AtomicOp::FETCH_MAX:
                    implementation = SpirvFloatAtomicImplementation::WORD_CAS;
                    break;
                default:
                    implementation = SpirvFloatAtomicImplementation::
                        UNSUPPORTED_OPERATION;
                    break;
            }
        }
        switch (implementation) {
            case SpirvFloatAtomicImplementation::WORD_EXCHANGE: {
                LUISA_ASSERT(uses_word_storage_buffer,
                             "SPIR-V float atomic representation planner "
                             "selected a word exchange for typed storage.");
                id = emit_word_atomic(
                    spv::Op::OpAtomicExchange,
                    _emit_value(values[0]->value()));
                break;
            }
            case SpirvFloatAtomicImplementation::WORD_COMPARE_EXCHANGE: {
                LUISA_ASSERT(uses_word_storage_buffer,
                             "SPIR-V float atomic representation planner "
                             "selected a word compare-exchange for typed storage.");
                auto expected = _emit_value(values[0]->value());
                auto desired = _emit_value(values[1]->value());
                auto result = _builder.createOp(
                    spv::Op::OpAtomicCompareExchange, uint_type,
                    {ptr, scope, semantics_equal, semantics_unequal,
                     to_word_bits(desired), to_word_bits(expected)});
                id = from_word_bits(result);
                break;
            }
            case SpirvFloatAtomicImplementation::WORD_CAS: {
                LUISA_ASSERT(uses_word_storage_buffer,
                             "SPIR-V float atomic representation planner "
                             "selected a word CAS loop for typed storage.");
                id = _emit_float_atomic_cas_loop(
                    ptr, _emit_value(values[0]->value()), type, inst->op(),
                    scope, semantics, semantics, semantics);
                break;
            }
            case SpirvFloatAtomicImplementation::NATIVE_EXCHANGE:
                LUISA_ASSERT(!uses_word_storage_buffer,
                             "SPIR-V native float exchange requires typed storage.");
                _require_target_feature(
                    storage == SpirvFloatAtomicStorage::BUFFER ?
                        target_feature::shader_buffer_float32_atomics :
                        target_feature::shader_shared_float32_atomics,
                    storage == SpirvFloatAtomicStorage::BUFFER ?
                        _target_features.shader_buffer_float32_atomics :
                        _target_features.shader_shared_float32_atomics);
                id = _builder.createOp(
                    spv::Op::OpAtomicExchange, type,
                    {ptr, scope, semantics,
                     _emit_value(values[0]->value())});
                break;
            case SpirvFloatAtomicImplementation::NATIVE_ADD: {
                LUISA_ASSERT(!uses_word_storage_buffer,
                             "SPIR-V native float add requires typed storage.");
                _require_target_feature(
                    storage == SpirvFloatAtomicStorage::BUFFER ?
                        target_feature::shader_buffer_float32_atomic_add :
                        target_feature::shader_shared_float32_atomic_add,
                    storage == SpirvFloatAtomicStorage::BUFFER ?
                        _target_features.shader_buffer_float32_atomic_add :
                        _target_features.shader_shared_float32_atomic_add);
                auto value = _emit_value(values[0]->value());
                if (inst->op() == xir::AtomicOp::FETCH_SUB) {
                    value = _builder.createUnaryOp(
                        spv::Op::OpFNegate, type, value);
                }
                _builder.addExtension(
                    spv::E_SPV_EXT_shader_atomic_float_add);
                _builder.addCapability(
                    spv::Capability::AtomicFloat32AddEXT);
                id = _builder.createOp(
                    spv::Op::OpAtomicFAddEXT, type,
                    {ptr, scope, semantics, value});
                break;
            }
            case SpirvFloatAtomicImplementation::NATIVE_MIN_MAX: {
                LUISA_ASSERT(!uses_word_storage_buffer,
                             "SPIR-V native float min/max requires typed storage.");
                _require_target_feature(
                    storage == SpirvFloatAtomicStorage::BUFFER ?
                        target_feature::shader_buffer_float32_atomic_min_max :
                        target_feature::shader_shared_float32_atomic_min_max,
                    storage == SpirvFloatAtomicStorage::BUFFER ?
                        _target_features.shader_buffer_float32_atomic_min_max :
                        _target_features.shader_shared_float32_atomic_min_max);
                _builder.addExtension(
                    spv::E_SPV_EXT_shader_atomic_float_min_max);
                _builder.addCapability(
                    spv::Capability::AtomicFloat32MinMaxEXT);
                auto op = inst->op() == xir::AtomicOp::FETCH_MIN ?
                              spv::Op::OpAtomicFMinEXT :
                              spv::Op::OpAtomicFMaxEXT;
                id = _builder.createOp(
                    op, type,
                    {ptr, scope, semantics,
                     _emit_value(values[0]->value())});
                break;
            }
            case SpirvFloatAtomicImplementation::UNSUPPORTED_WIDTH:
                LUISA_ERROR_WITH_LOCATION(
                    "Vulkan XIR-to-SPIR-V float atomics currently support "
                    "float32 only; {}-bit {} is unsupported.",
                    bit_width, xir::to_string(inst->op()));
            case SpirvFloatAtomicImplementation::UNSUPPORTED_OPERATION:
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported SPIR-V float atomic operation {}.",
                    xir::to_string(inst->op()));
            case SpirvFloatAtomicImplementation::UNSUPPORTED_REPRESENTATION:
                LUISA_ERROR_WITH_LOCATION(
                    "Vulkan XIR-to-SPIR-V cannot implement shared float32 "
                    "compare-exchange: OpAtomicCompareExchange accepts "
                    "integer values, and this shared allocation is not "
                    "represented as integer words.");
            case SpirvFloatAtomicImplementation::UNSUPPORTED_FEATURE: {
                auto feature = [&]() noexcept -> luisa::string_view {
                    switch (inst->op()) {
                        case xir::AtomicOp::EXCHANGE:
                            return "shaderSharedFloat32Atomics";
                        case xir::AtomicOp::FETCH_ADD:
                        case xir::AtomicOp::FETCH_SUB:
                            return "shaderSharedFloat32AtomicAdd";
                        case xir::AtomicOp::FETCH_MIN:
                        case xir::AtomicOp::FETCH_MAX:
                            return "shaderSharedFloat32AtomicMinMax";
                        default: return "unknown";
                    }
                }();
                LUISA_ERROR_WITH_LOCATION(
                    "Vulkan shared float32 atomic {} requires the enabled "
                    "{} feature.",
                    xir::to_string(inst->op()), feature);
            }
        }
        LUISA_ASSERT(id != spv::NoResult,
                     "Failed to emit float atomic op.");
        _value_map.emplace(inst, id);
        return;
    }

    if (t->is_int64() || t->is_uint64()) {
        switch (pointer_storage) {
            case spv::StorageClass::StorageBuffer:
                _require_target_feature(
                    target_feature::shader_buffer_int64_atomics,
                    _target_features.shader_buffer_int64_atomics);
                break;
            case spv::StorageClass::Workgroup:
                _require_target_feature(
                    target_feature::shader_shared_int64_atomics,
                    _target_features.shader_shared_int64_atomics);
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "SPIR-V int64 atomic {} uses unsupported storage class {}. "
                    "Only StorageBuffer and Workgroup have explicit Vulkan "
                    "feature contracts.",
                    xir::to_string(inst->op()),
                    static_cast<uint32_t>(pointer_storage));
        }
        _builder.addCapability(spv::Capability::Int64Atomics);
    }

    switch (inst->op()) {
        case xir::AtomicOp::EXCHANGE: {
            auto val = _emit_value(values[0]->value());
            if (uses_word_storage_buffer) {
                id = emit_word_atomic(spv::Op::OpAtomicExchange, val);
            } else {
                id = _builder.createOp(spv::Op::OpAtomicExchange, type, {ptr, scope, semantics, val});
            }
            break;
        }
        case xir::AtomicOp::COMPARE_EXCHANGE: {
            auto expected = _emit_value(values[0]->value());
            auto desired = _emit_value(values[1]->value());
            if (uses_word_storage_buffer) {
                auto result = _builder.createOp(
                    spv::Op::OpAtomicCompareExchange, uint_type,
                    {ptr, scope, semantics_equal, semantics_unequal,
                     to_word_bits(desired), to_word_bits(expected)});
                id = from_word_bits(result);
            } else {
                id = _builder.createOp(spv::Op::OpAtomicCompareExchange, type,
                                       {ptr, scope, semantics_equal, semantics_unequal, desired, expected});
            }
            break;
        }
        case xir::AtomicOp::FETCH_ADD: {
            auto val = _emit_value(values[0]->value());
            if (uses_word_storage_buffer) {
                id = emit_word_atomic(spv::Op::OpAtomicIAdd, val);
            } else {
                id = _builder.createOp(spv::Op::OpAtomicIAdd, type, {ptr, scope, semantics, val});
            }
            break;
        }
        case xir::AtomicOp::FETCH_SUB: {
            auto val = _emit_value(values[0]->value());
            if (uses_word_storage_buffer) {
                id = emit_word_atomic(spv::Op::OpAtomicISub, val);
            } else {
                id = _builder.createOp(spv::Op::OpAtomicISub, type, {ptr, scope, semantics, val});
            }
            break;
        }
        case xir::AtomicOp::FETCH_AND: {
            auto val = _emit_value(values[0]->value());
            id = uses_word_storage_buffer ?
                     emit_word_atomic(spv::Op::OpAtomicAnd, val) :
                     _builder.createOp(spv::Op::OpAtomicAnd, type, {ptr, scope, semantics, val});
            break;
        }
        case xir::AtomicOp::FETCH_OR: {
            auto val = _emit_value(values[0]->value());
            id = uses_word_storage_buffer ?
                     emit_word_atomic(spv::Op::OpAtomicOr, val) :
                     _builder.createOp(spv::Op::OpAtomicOr, type, {ptr, scope, semantics, val});
            break;
        }
        case xir::AtomicOp::FETCH_XOR: {
            auto val = _emit_value(values[0]->value());
            id = uses_word_storage_buffer ?
                     emit_word_atomic(spv::Op::OpAtomicXor, val) :
                     _builder.createOp(spv::Op::OpAtomicXor, type, {ptr, scope, semantics, val});
            break;
        }
        case xir::AtomicOp::FETCH_MIN: {
            auto val = _emit_value(values[0]->value());
            if (t->is_int()) {
                id = uses_word_storage_buffer ?
                         emit_word_atomic(spv::Op::OpAtomicSMin, val) :
                         _builder.createOp(spv::Op::OpAtomicSMin, type, {ptr, scope, semantics, val});
            } else if (t->is_uint()) {
                id = uses_word_storage_buffer ?
                         emit_word_atomic(spv::Op::OpAtomicUMin, val) :
                         _builder.createOp(spv::Op::OpAtomicUMin, type, {ptr, scope, semantics, val});
            } else {
                LUISA_NOT_IMPLEMENTED("SPIR-V atomic min for type {}.", t->description());
            }
            break;
        }
        case xir::AtomicOp::FETCH_MAX: {
            auto val = _emit_value(values[0]->value());
            if (t->is_int()) {
                id = uses_word_storage_buffer ?
                         emit_word_atomic(spv::Op::OpAtomicSMax, val) :
                         _builder.createOp(spv::Op::OpAtomicSMax, type, {ptr, scope, semantics, val});
            } else if (t->is_uint()) {
                id = uses_word_storage_buffer ?
                         emit_word_atomic(spv::Op::OpAtomicUMax, val) :
                         _builder.createOp(spv::Op::OpAtomicUMax, type, {ptr, scope, semantics, val});
            } else {
                LUISA_NOT_IMPLEMENTED("SPIR-V atomic max for type {}.", t->description());
            }
            break;
        }
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit atomic op.");
    _value_map.emplace(inst, id);
}

spv::Id SpirvCodegenEntry::_load_texture(spv::Id tex_var) noexcept {
    auto tex_type = _builder.getTypeId(tex_var);
    auto pointee_type = _builder.getContainedTypeId(tex_type);
    if (_builder.isArrayType(pointee_type)) {
        auto image_ptr = _create_access_chain(spv::StorageClass::UniformConstant, tex_var, {_builder.makeUintConstant(0u)});
        return _builder.createLoad(image_ptr, spv::NoPrecision);
    }
    return _builder.createLoad(tex_var, spv::NoPrecision);
}

void SpirvCodegenEntry::_emit_ray_query_traversal_to_completion(
    spv::Id ray_query) noexcept {
    // OpRayQueryProceedKHR returns false only after traversal completes. A
    // committed intersection observed while it still returns true is merely
    // the closest hit recorded so far, not necessarily the final closest hit.
    // Direct surface tracing forces triangles opaque and skips AABBs, so no
    // candidate-side action is required; repeatedly proceeding is sufficient.
    auto *header = _create_physical_block();
    auto *body = _create_physical_block();
    auto *continue_block = _create_physical_block();
    auto *merge = _create_physical_block();
    _builder.createBranch(false, header);

    _set_current_tail(header);
    _builder.createLoopMerge(
        merge, continue_block,
        spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, body);

    _set_current_tail(body);
    auto incomplete = _builder.createOp(
        spv::Op::OpRayQueryProceedKHR,
        _builder.makeBoolType(),
        std::vector<spv::Id>{ray_query});
    _builder.createConditionalBranch(
        incomplete, continue_block, merge);

    _set_current_tail(continue_block);
    _builder.createBranch(false, header);
    _set_current_tail(merge);
}

bool SpirvCodegenEntry::_bindless_index_is_nonuniform(
    const xir::Value *index,
    xir::BindlessResourceAccess access) const noexcept {
    // `uniform` is an explicit frontend promise. When it is absent, the
    // analysis may still prove uniformity and avoid descriptor-indexing
    // decorations without changing semantics.
    return !access.uniform && !_uniformity.is_uniform(index);
}

spv::Id SpirvCodegenEntry::_load_bindless_slot_word(
    spv::Id bindless_array, spv::Id slot_index,
    uint32_t stride_words, uint32_t field_word,
    bool nonuniform) noexcept {
    LUISA_ASSERT(stride_words != 0u && field_word < stride_words,
                 "Invalid SPIR-V bindless slot word layout ({}, {}).",
                 stride_words, field_word);
    auto uint_type = _builder.makeUintType(32);
    slot_index = _ensure_type(slot_index, uint_type);
    auto word = slot_index;
    if (stride_words != 1u) {
        word = _builder.createBinOp(
            spv::Op::OpIMul, uint_type, word,
            _builder.makeUintConstant(stride_words));
    }
    if (field_word != 0u) {
        word = _builder.createBinOp(
            spv::Op::OpIAdd, uint_type, word,
            _builder.makeUintConstant(field_word));
    }
    auto ptr = _create_access_chain(
        _builder.getStorageClass(bindless_array), bindless_array,
        {_builder.makeUintConstant(0u), word}, nonuniform);
    auto value = _builder.createLoad(ptr, spv::NoPrecision);
    if (nonuniform) {
        _builder.addDecoration(value, spv::Decoration::NonUniformEXT);
    }
    return value;
}

SpirvCodegenEntry::BindlessBufferBinding
SpirvCodegenEntry::_load_bindless_buffer_binding(
    spv::Id bindless_array, const xir::Value *slot,
    xir::BindlessResourceAccess access) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto slot_index = _ensure_type(_emit_value(slot), uint_type);
    auto nonuniform = _bindless_index_is_nonuniform(slot, access);
    _require_storage_buffer_array_indexing(nonuniform);
    auto descriptor_index = _load_bindless_slot_word(
        bindless_array, slot_index,
        access.typed ? 4u : 3u, 0u, nonuniform);
    LUISA_ASSERT(_buffer_heap_id != spv::NoResult,
                 "SPIR-V buffer heap not bound.");
    auto buffer = _create_access_chain(
        _builder.getStorageClass(_buffer_heap_id), _buffer_heap_id,
        {descriptor_index}, nonuniform);
    return {buffer, slot_index};
}

SpirvCodegenEntry::BindlessTextureBinding
SpirvCodegenEntry::_load_bindless_texture_binding(
    spv::Id bindless_array, const xir::Value *slot,
    bool is_2d, xir::BindlessResourceAccess access) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto slot_index = _ensure_type(_emit_value(slot), uint_type);
    auto nonuniform = _bindless_index_is_nonuniform(slot, access);
    _require_sampled_image_array_indexing(nonuniform);
    auto packed = _load_bindless_slot_word(
        bindless_array, slot_index,
        access.typed ? 1u : 3u,
        access.typed ? 0u : (is_2d ? 1u : 2u),
        nonuniform);
    auto texture_index = _builder.createBinOp(
        spv::Op::OpBitwiseAnd, uint_type, packed,
        _builder.makeUintConstant(0x0fffffffu));
    auto heap = is_2d ? _tex2d_heap_id : _tex3d_heap_id;
    LUISA_ASSERT(heap != spv::NoResult,
                 "SPIR-V {} texture heap not bound.",
                 is_2d ? "2D" : "3D");
    auto texture_ptr = _create_access_chain(
        spv::StorageClass::UniformConstant, heap,
        {texture_index}, nonuniform);
    auto image = _builder.createLoad(texture_ptr, spv::NoPrecision);
    if (nonuniform) {
        _builder.addDecoration(image, spv::Decoration::NonUniformEXT);
    }
    return {image, packed, nonuniform};
}

void SpirvCodegenEntry::_emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    spv::Id id = spv::NoResult;

    struct LoadedSampler {
        spv::Id id;
        bool nonuniform;
    };
    auto load_sampler = [&](spv::Id index, bool dynamically_indexed, bool nonuniform) noexcept -> LoadedSampler {
        LUISA_ASSERT(!_properties.empty() &&
                         _properties.front().type == ShaderVariableType::SamplerHeap &&
                         _property_ids.size() > 1u && _property_ids[1] != spv::NoResult,
                     "SPIR-V fixed 16-entry sampler heap is not bound.");
        if (dynamically_indexed) {
            _require_sampled_image_array_indexing(nonuniform);
        }
        auto sampler_ptr = _create_access_chain(
            spv::StorageClass::UniformConstant, _property_ids[1], {index}, nonuniform);
        auto sampler = _builder.createLoad(sampler_ptr, spv::NoPrecision);
        if (nonuniform) { _builder.addDecoration(sampler, spv::Decoration::NonUniformEXT); }
        return {sampler, nonuniform};
    };
    auto load_configured_sampler = [&](const xir::Value *filter_value,
                                       const xir::Value *address_value) noexcept -> LoadedSampler {
        LUISA_ASSERT(
            filter_value != nullptr &&
                spirv_sampler_selector_type_supported(
                    filter_value->type()),
            "SPIR-V dialect validation failed to reject an invalid texture "
            "sampler filter selector type.");
        LUISA_ASSERT(
            address_value != nullptr &&
                spirv_sampler_selector_type_supported(
                    address_value->type()),
            "SPIR-V dialect validation failed to reject an invalid texture "
            "sampler address selector type.");
        auto filter_decode =
            decode_spirv_sampler_selector_constant(filter_value);
        auto address_decode =
            decode_spirv_sampler_selector_constant(address_value);
        LUISA_ASSERT(filter_decode.succeeded(),
                     "SPIR-V dialect validation failed to reject an invalid "
                     "texture sampler filter selector: {}",
                     filter_decode.diagnostic);
        LUISA_ASSERT(address_decode.succeeded(),
                     "SPIR-V dialect validation failed to reject an invalid "
                     "texture sampler address selector: {}",
                     address_decode.diagnostic);
        auto filter_constant = filter_decode.value;
        auto address_constant = address_decode.value;
        switch (plan_spirv_sampler_filter(
            filter_constant.has_value(), filter_constant.value_or(0u),
            _target_features.sampler_anisotropy)) {
            case SpirvSamplerFilterPlan::SUPPORTED:
                break;
            case SpirvSamplerFilterPlan::INVALID_SELECTOR:
                LUISA_ASSERT(
                    false,
                    "SPIR-V dialect validation failed to reject texture "
                    "sampler filter selector {} outside [0, {}).",
                    *filter_constant,
                    spirv_configured_sampler_selector_count);
            case SpirvSamplerFilterPlan::REQUIRES_ANISOTROPY:
                LUISA_ASSERT(
                    false,
                    "SPIR-V target preflight failed to reject a texture "
                    "sampler filter that may select ANISOTROPIC without "
                    "samplerAnisotropy enabled.");
        }
        if (!filter_constant.has_value() ||
            *filter_constant ==
                spirv_configured_sampler_selector_max) {
            _require_target_feature(
                target_feature::sampler_anisotropy,
                _target_features.sampler_anisotropy);
        }
        if (address_constant &&
            *address_constant >=
                spirv_configured_sampler_selector_count) {
            LUISA_ASSERT(
                false,
                "SPIR-V dialect validation failed to reject texture sampler "
                "address selector {} outside [0, {}).",
                *address_constant,
                spirv_configured_sampler_selector_count);
        }
        if (filter_constant && address_constant) {
            auto index = *address_constant *
                             spirv_configured_sampler_selector_count +
                         *filter_constant;
            LUISA_ASSERT(
                index < spirv_configured_sampler_heap_size,
                "SPIR-V sampler selector planning produced index {} outside "
                "the fixed {}-entry heap.",
                index, spirv_configured_sampler_heap_size);
            return load_sampler(_builder.makeUintConstant(index), false, false);
        }
        auto uint_type = _builder.makeUintType(32);
        auto emit_bounded_selector =
            [&](const xir::Value *value,
                std::optional<uint32_t> constant,
                luisa::string_view name) noexcept -> spv::Id {
            if (constant) {
                LUISA_ASSERT(
                    *constant < spirv_configured_sampler_selector_count,
                    "SPIR-V dialect validation failed to reject texture "
                    "sampler {} selector {} outside [0, {}).",
                    name, *constant,
                    spirv_configured_sampler_selector_count);
                return _builder.makeUintConstant(*constant);
            }

            LUISA_ASSERT(
                spirv_sampler_selector_type_supported(value->type()),
                "SPIR-V dialect validation failed to reject an invalid "
                "dynamic texture sampler {} selector type.",
                name);
            auto selector = _emit_value(value);
            auto selector_type = _builder.getTypeId(selector);
            LUISA_ASSERT(
                selector_type == uint_type,
                "SPIR-V sampler {} selector emission is not uint32.",
                name);
            auto maximum = _builder.makeUintConstant(
                spirv_configured_sampler_selector_max);
            auto bool_type = _builder.makeBoolType();
            auto above_maximum = _builder.createBinOp(
                spv::Op::OpUGreaterThan,
                bool_type, selector, maximum);
            return _builder.createTriOp(
                spv::Op::OpSelect, uint_type,
                above_maximum, maximum, selector);
        };
        auto filter = emit_bounded_selector(
            filter_value, filter_constant, "filter");
        auto address = emit_bounded_selector(
            address_value, address_constant, "address");
        // The fixed heap is address-major and filter-minor: 4 x 4 entries.
        auto address_base = _builder.createBinOp(
            spv::Op::OpIMul, uint_type, address,
            _builder.makeUintConstant(
                spirv_configured_sampler_selector_count));
        auto index = _builder.createBinOp(spv::Op::OpIAdd, uint_type, address_base, filter);
        auto nonuniform = !_uniformity.is_uniform(filter_value) ||
                          !_uniformity.is_uniform(address_value);
        return load_sampler(index, true, nonuniform);
    };
    auto emit_texture_sample = [&](spv::Id image, LoadedSampler sampler,
                                   bool image_nonuniform, size_t uv_operand,
                                   SpirvTextureSampleOpInfo info) noexcept -> spv::Id {
        auto image_type = _builder.getTypeId(image);
        auto sampled_image_type = _builder.makeSampledImageType(image_type, "sampled_image");
        auto sampled_image = _builder.createOp(
            spv::Op::OpSampledImage, sampled_image_type, {image, sampler.id});
        if (image_nonuniform || sampler.nonuniform) {
            _builder.addDecoration(sampled_image, spv::Decoration::NonUniformEXT);
        }
        spv::Builder::TextureParameters params{};
        params.sampler = sampled_image;
        params.coords = _emit_value(inst->operand(uv_operand));
        if (info.explicit_lod) {
            params.lod = _emit_value(inst->operand(uv_operand + 1u));
        } else if (info.gradients) {
            params.gradX = _emit_value(inst->operand(uv_operand + 1u));
            params.gradY = _emit_value(inst->operand(uv_operand + 2u));
            if (info.lod_clamp) {
                _require_target_feature(
                    target_feature::shader_resource_min_lod,
                    _target_features.shader_resource_min_lod);
                params.lodClamp = _emit_value(inst->operand(uv_operand + 3u));
            }
        }
        // Compute shaders have no implicit derivatives. The plain sample form
        // therefore uses an explicit LOD of zero, relative to the bound image view.
        return _builder.createTextureCall(
            spv::NoPrecision, type, false, false, false, false, true,
            params, spv::ImageOperandsMask::MaskNone);
    };

    switch (inst->op()) {
        case xir::ResourceQueryOp::BUFFER_SIZE: {
            auto buffer = _emit_value(inst->operand(0));
            if (_direct_buffer_metadata_indices.contains(buffer)) {
                auto bytes = _load_direct_buffer_metadata(
                    buffer, StorageBufferMetadataField::LOGICAL_SIZE_BYTES,
                    type);
                auto element_type = inst->operand(0)->type()->element();
                LUISA_ASSERT(element_type != nullptr && element_type->size() != 0u,
                             "SPIR-V direct buffer size query requires a sized element type.");
                auto stride = inst->type()->is_uint64() ?
                                  _builder.makeUint64Constant(element_type->size()) :
                                  _builder.makeUintConstant(
                                      static_cast<uint32_t>(element_type->size()));
                id = _builder.createBinOp(
                    spv::Op::OpUDiv, type, bytes, stride);
                break;
            }
            auto len = _builder.createArrayLength(buffer, 0u, 32u);
            if (inst->type()->is_uint64()) {
                len = _builder.createUnaryOp(spv::Op::OpUConvert, type, len);
            }
            auto buffer_type = inst->operand(0)->type();
            if (_buffer_uses_word_storage(buffer_type)) {
                auto element_type = buffer_type->element();
                LUISA_ASSERT(element_type != nullptr && element_type->size() != 0u,
                             "SPIR-V word-storage buffer size query requires a sized element type.");
                auto four = inst->type()->is_uint64() ?
                                _builder.makeUint64Constant(4u) :
                                _builder.makeUintConstant(4u);
                auto stride = inst->type()->is_uint64() ?
                                  _builder.makeUint64Constant(element_type->size()) :
                                  _builder.makeUintConstant(static_cast<uint32_t>(element_type->size()));
                len = _builder.createBinOp(spv::Op::OpIMul, type, len, four);
                len = _builder.createBinOp(spv::Op::OpUDiv, type, len, stride);
            }
            id = len;
            break;
        }
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: {
            auto buffer = _emit_value(inst->operand(0));
            if (_direct_buffer_metadata_indices.contains(buffer)) {
                id = _load_direct_buffer_metadata(
                    buffer, StorageBufferMetadataField::LOGICAL_SIZE_BYTES,
                    type);
                break;
            }
            auto len = _builder.createArrayLength(buffer, 0u, 32u);
            if (inst->type()->is_uint64()) {
                len = _builder.createUnaryOp(spv::Op::OpUConvert, type, len);
            }
            auto four = inst->type()->is_uint64() ?
                            _builder.makeUint64Constant(4u) :
                            _builder.makeUintConstant(4u);
            id = _builder.createBinOp(spv::Op::OpIMul, type, len, four);
            break;
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _emit_value(inst->operand(1));
            if (inst->bindless_access().typed) {
                id = _load_bindless_slot_word(
                    bindless_array, slot_index, 4u, 2u, false);
                id = _ensure_type(id, type);
            } else {
                id = _load_bindless_buffer_metadata(
                    bindless_array, slot_index,
                    StorageBufferMetadataField::LOGICAL_SIZE_BYTES, type);
            }
            if (inst->op() == xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE) {
                auto stride = _ensure_type(_emit_value(inst->operand(2)), type);
                id = _builder.createBinOp(
                    spv::Op::OpUDiv, type, id, stride);
            }
            break;
        }
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: {
            LUISA_ASSERT(
                _runtime_target_plan_installed &&
                    _runtime_target_plan.uses_buffer_device_address,
                "SPIR-V buffer device-address query escaped runtime target preflight.");
            _require_target_feature(
                target_feature::buffer_device_address,
                _target_features.buffer_device_address);
            auto buffer = _emit_value(inst->operand(0));
            id = _load_direct_buffer_metadata(
                buffer, StorageBufferMetadataField::DEVICE_ADDRESS, type);
            break;
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: {
            LUISA_ASSERT(
                _runtime_target_plan_installed &&
                    _runtime_target_plan.uses_buffer_device_address,
                "SPIR-V bindless buffer device-address query escaped runtime target preflight.");
            _require_target_feature(
                target_feature::buffer_device_address,
                _target_features.buffer_device_address);
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _emit_value(inst->operand(1));
            id = _load_bindless_buffer_metadata(
                bindless_array, slot_index,
                StorageBufferMetadataField::DEVICE_ADDRESS, type);
            break;
        }
        case xir::ResourceQueryOp::TEXTURE2D_SIZE:
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: {
            auto tex_array = _emit_value(inst->operand(0));
            auto tex = _load_texture(tex_array);
            _builder.addCapability(spv::Capability::ImageQuery);
            if (_is_storage_image_map.at(tex_array)) {
                id = _builder.createOp(spv::Op::OpImageQuerySize, type, std::vector<spv::Id>{tex});
            } else {
                id = _builder.createOp(spv::Op::OpImageQuerySizeLod, type, {tex, _builder.makeUintConstant(0u)});
            }
            break;
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: {
            auto is_2d = inst->op() == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE ||
                         inst->op() == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL;
            auto has_level = inst->op() == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
                             inst->op() == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL;
            auto bindless_array = _emit_value(inst->operand(0));
            auto binding = _load_bindless_texture_binding(
                bindless_array, inst->operand(1), is_2d,
                inst->bindless_access());
            _builder.addCapability(spv::Capability::ImageQuery);
            auto uint_type = _builder.makeUintType(32);
            spv::Id lod = has_level ? _ensure_type(_emit_value(inst->operand(2)), uint_type) : _builder.makeUintConstant(0u);
            id = _builder.createOp(
                spv::Op::OpImageQuerySizeLod, type,
                {binding.image, lod});
            break;
        }
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: {
            auto info = spirv_texture_sample_op_info(inst->op());
            LUISA_ASSERT(info.valid,
                         "Invalid SPIR-V texture-sampling operation {}.",
                         xir::to_string(inst->op()));
            auto uv_operand = info.direct ? 1u : 2u;
            spv::Id image = spv::NoResult;
            bool image_nonuniform = false;
            LoadedSampler sampler{};
            if (info.direct) {
                auto texture_type = inst->operand(0)->type();
                LUISA_ASSERT(texture_type != nullptr && texture_type->is_texture() &&
                                 texture_type->dimension() == (info.is_2d ? 2u : 3u),
                             "SPIR-V {} texture sample received resource type {}.",
                             info.is_2d ? "2D" : "3D",
                             texture_type == nullptr ? "<null>" : texture_type->description());
                auto texture = _emit_value(inst->operand(0));
                auto storage = _is_storage_image_map.find(texture);
                LUISA_ASSERT(storage != _is_storage_image_map.end(),
                             "SPIR-V texture binding kind was not recorded before sampling.");
                if (storage->second) {
                    LUISA_ERROR_WITH_LOCATION(
                        "SPIR-V direct texture sampling requires a sampled-image (read-only/SRV) "
                        "binding, but resource specialization selected a storage-image parameter.");
                }
                image = _load_texture(texture);
                sampler = load_configured_sampler(
                    inst->operand(inst->operand_count() - 2u),
                    inst->operand(inst->operand_count() - 1u));
            } else {
                auto uint_type = _builder.makeUintType(32);
                auto bindless_array = _emit_value(inst->operand(0));
                auto binding = _load_bindless_texture_binding(
                    bindless_array, inst->operand(1), info.is_2d,
                    inst->bindless_access());
                image = binding.image;
                image_nonuniform = binding.nonuniform;
                if (info.sampler_operands) {
                    sampler = load_configured_sampler(
                        inst->operand(inst->operand_count() - 2u),
                        inst->operand(inst->operand_count() - 1u));
                } else {
                    auto sampler_index = _builder.createBinOp(
                        spv::Op::OpShiftRightLogical, uint_type,
                        binding.packed,
                        _builder.makeUintConstant(28u));
                    sampler = load_sampler(sampler_index, true, image_nonuniform);
                }
            }
            id = emit_texture_sample(image, sampler, image_nonuniform, uv_operand, info);
            break;
        }
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V motion-blur ray queries are unsupported: "
                "OpRayQueryInitializeKHR cannot represent the XIR time operand.");
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: {
            _require_target_feature(target_feature::ray_query,
                                    _target_features.ray_query);
            _builder.addExtension(spv::E_SPV_KHR_ray_query);
            _builder.addCapability(spv::Capability::RayQueryKHR);
            auto accel_ptr = _emit_value(inst->operand(0));
            auto accel = _builder.createLoad(accel_ptr, spv::NoPrecision);
            auto ray = _emit_value(inst->operand(1));
            auto mask = _emit_value(inst->operand(2));
            auto float_type = _builder.makeFloatType(32);
            auto uint_type = _builder.makeUintType(32);
            auto vec3_type = _builder.makeVectorType(float_type, 3);
            auto extract_vec3_from_array_member = [&](uint32_t member_idx) {
                std::vector<spv::Id> comps;
                comps.reserve(3);
                for (uint32_t i = 0; i < 3; ++i) {
                    comps.push_back(_builder.createCompositeExtract(ray, float_type, std::vector<uint32_t>{member_idx, i}));
                }
                return _builder.createCompositeConstruct(vec3_type, comps);
            };
            auto ray_origin = extract_vec3_from_array_member(0);
            auto ray_t_min = _builder.createCompositeExtract(ray, float_type, 1);
            auto ray_dir = extract_vec3_from_array_member(2);
            auto ray_t_max = _builder.createCompositeExtract(ray, float_type, 3);
            auto rq_type = _builder.makeRayQueryType();
            auto rq_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, rq_type, "rq");
            auto proceed_state = _builder.createVariable(
                spv::NoPrecision, spv::StorageClass::Function,
                _builder.makeBoolType(), "rq_proceed_state");
            _builder.createStore(_builder.makeBoolConstant(false), proceed_state);
            auto is_query_all = inst->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL;
            auto ray_flags = _builder.makeUintConstant(
                is_query_all ?
                    static_cast<uint32_t>(spv::RayFlagsMask::MaskNone) :
                    static_cast<uint32_t>(
                        spv::RayFlagsMask::TerminateOnFirstHitKHR));
            _builder.createNoResultOp(spv::Op::OpRayQueryInitializeKHR, std::vector<spv::Id>{
                                                                            rq_var, accel, ray_flags, mask, ray_origin, ray_t_min, ray_dir, ray_t_max});
            auto [_, inserted] = _ray_query_states.emplace(
                rq_var, RayQueryState{ray, proceed_state});
            LUISA_ASSERT(inserted,
                         "SPIR-V ray-query initialization state was registered twice.");
            id = rq_var;
            break;
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V motion-blur ray tracing is unsupported: "
                "OpRayQueryInitializeKHR cannot represent the XIR time operand.");
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: {
            _require_target_feature(target_feature::ray_query,
                                    _target_features.ray_query);
            _builder.addExtension(spv::E_SPV_KHR_ray_query);
            _builder.addCapability(spv::Capability::RayQueryKHR);
            // Surface-only direct tracing uses SkipAABBsKHR. The SPIR-V
            // grammar requires RayTraversalPrimitiveCullingKHR for that flag;
            // in the Vulkan environment, VkPhysicalDeviceRayQueryFeaturesKHR::
            // rayQuery authorizes this capability for ray-query instructions.
            // Keep it local to the direct-trace path: generic traversal does
            // not use a primitive-culling ray flag and must not declare it.
            _builder.addCapability(
                spv::Capability::RayTraversalPrimitiveCullingKHR);
            auto accel_ptr = _emit_value(inst->operand(0));
            auto accel = _builder.createLoad(accel_ptr, spv::NoPrecision);
            auto ray = _emit_value(inst->operand(1));
            auto mask = _emit_value(inst->operand(2));
            auto float_type = _builder.makeFloatType(32);
            auto uint_type = _builder.makeUintType(32);
            auto vec3_type = _builder.makeVectorType(float_type, 3);
            auto extract_vec3_from_array_member = [&](uint32_t member_idx) {
                std::vector<spv::Id> comps;
                comps.reserve(3);
                for (uint32_t i = 0; i < 3; ++i) {
                    comps.push_back(_builder.createCompositeExtract(ray, float_type, std::vector<uint32_t>{member_idx, i}));
                }
                return _builder.createCompositeConstruct(vec3_type, comps);
            };
            auto ray_origin = extract_vec3_from_array_member(0);
            auto ray_t_min = _builder.createCompositeExtract(ray, float_type, 1);
            auto ray_dir = extract_vec3_from_array_member(2);
            auto ray_t_max = _builder.createCompositeExtract(ray, float_type, 3);
            auto rq_type = _builder.makeRayQueryType();
            auto rq_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, rq_type, "rq");
            auto is_closest = inst->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST;
            auto surface_flags =
                spv::RayFlagsMask::OpaqueKHR |
                spv::RayFlagsMask::SkipAABBsKHR;
            auto any_hit_flags =
                surface_flags |
                spv::RayFlagsMask::TerminateOnFirstHitKHR |
                spv::RayFlagsMask::SkipClosestHitShaderKHR;
            auto ray_flags = _builder.makeUintConstant(
                static_cast<uint32_t>(is_closest ?
                                          surface_flags :
                                          any_hit_flags));
            _builder.createNoResultOp(spv::Op::OpRayQueryInitializeKHR, std::vector<spv::Id>{
                                                                            rq_var, accel, ray_flags, mask, ray_origin, ray_t_min, ray_dir, ray_t_max});
            if (is_closest) {
                _emit_ray_query_traversal_to_completion(rq_var);
            } else {
                // TerminateOnFirstHitKHR makes one proceed sufficient for the
                // direct any-hit query.
                _builder.createOp(
                    spv::Op::OpRayQueryProceedKHR,
                    _builder.makeBoolType(),
                    std::vector<spv::Id>{rq_var});
            }
            auto committed_intersection = _builder.makeIntConstant(1);
            auto committed_type = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTypeKHR, uint_type,
                                                    std::vector<spv::IdImmediate>{
                                                        {true, rq_var},
                                                        {true, committed_intersection}});
            if (is_closest) {
                auto is_triangle_hit = _builder.createBinOp(spv::Op::OpIEqual, _builder.makeBoolType(),
                                                            committed_type, _builder.makeUintConstant(1u));
                auto result_type = type;
                auto result_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, result_type, "trace_result");
                auto zero_result = _builder.makeNullConstant(result_type);
                _builder.createStore(zero_result, result_var);
                auto *true_block = _create_physical_block();
                auto *false_block = _create_physical_block();
                auto *merge_block = _create_physical_block();
                auto selection_merge = std::make_unique<spv::Instruction>(spv::Op::OpSelectionMerge);
                selection_merge->reserveOperands(2);
                selection_merge->addIdOperand(merge_block->getId());
                selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
                _builder.getBuildPoint()->addInstruction(std::move(selection_merge));
                _builder.createConditionalBranch(is_triangle_hit, true_block, false_block);
                _set_current_tail(true_block);
                auto inst_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionInstanceIdKHR, uint_type,
                                                  std::vector<spv::IdImmediate>{{true, rq_var}, {true, committed_intersection}});
                auto prim_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR, uint_type,
                                                  std::vector<spv::IdImmediate>{{true, rq_var}, {true, committed_intersection}});
                auto bary = _builder.createOp(spv::Op::OpRayQueryGetIntersectionBarycentricsKHR, _builder.makeVectorType(float_type, 2),
                                              std::vector<spv::IdImmediate>{{true, rq_var}, {true, committed_intersection}});
                auto ray_t = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTKHR, float_type,
                                               std::vector<spv::IdImmediate>{{true, rq_var}, {true, committed_intersection}});
                auto hit_result = _builder.createCompositeConstruct(result_type, {inst_idx, prim_idx, bary, ray_t});
                _builder.createStore(hit_result, result_var);
                _builder.createBranch(false, merge_block);
                _set_current_tail(false_block);
                auto no_hit_inst = _builder.makeUintConstant(0xFFFFFFFFu);
                auto vec2_type = _builder.makeVectorType(float_type, 2);
                auto no_hit_bary = _builder.createCompositeConstruct(vec2_type, {_builder.makeFloatConstant(0.0f), _builder.makeFloatConstant(0.0f)});
                auto no_hit_result = _builder.createCompositeConstruct(result_type, {no_hit_inst,
                                                                                     _builder.makeUintConstant(0u),
                                                                                     no_hit_bary,
                                                                                     _builder.makeFloatConstant(0.0f)});
                _builder.createStore(no_hit_result, result_var);
                _builder.createBranch(false, merge_block);
                _set_current_tail(merge_block);
                id = _builder.createLoad(result_var, spv::NoPrecision);
            } else {
                id = _builder.createBinOp(spv::Op::OpINotEqual, _builder.makeBoolType(),
                                          committed_type, _builder.makeUintConstant(0u));
            }
            break;
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: {
            auto instance_buffer =
                _resolve_accel_instance_buffer(inst->operand(0));
            auto uint_type = _builder.makeUintType(32);
            auto instance_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto float_type = _builder.makeFloatType(32);
            auto float4_type = _builder.makeVectorType(float_type, 4);
            // Runtime instance records use 16 uint32 words (64 bytes) per instance.
            auto byte_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, instance_index, _builder.makeUintConstant(64u));
            auto float4 = Type::vector(Type::of<float>(), 4);
            auto p0 = _emit_buffer_read_impl(instance_buffer, byte_offset, float4, 4u);
            auto p1_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, byte_offset, _builder.makeUintConstant(16u));
            auto p1 = _emit_buffer_read_impl(instance_buffer, p1_byte_offset, float4, 4u);
            auto p2_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, byte_offset, _builder.makeUintConstant(32u));
            auto p2 = _emit_buffer_read_impl(instance_buffer, p2_byte_offset, float4, 4u);
            auto p0_x = _builder.createCompositeExtract(p0, float_type, 0);
            auto p0_y = _builder.createCompositeExtract(p0, float_type, 1);
            auto p0_z = _builder.createCompositeExtract(p0, float_type, 2);
            auto p0_w = _builder.createCompositeExtract(p0, float_type, 3);
            auto p1_x = _builder.createCompositeExtract(p1, float_type, 0);
            auto p1_y = _builder.createCompositeExtract(p1, float_type, 1);
            auto p1_z = _builder.createCompositeExtract(p1, float_type, 2);
            auto p1_w = _builder.createCompositeExtract(p1, float_type, 3);
            auto p2_x = _builder.createCompositeExtract(p2, float_type, 0);
            auto p2_y = _builder.createCompositeExtract(p2, float_type, 1);
            auto p2_z = _builder.createCompositeExtract(p2, float_type, 2);
            auto p2_w = _builder.createCompositeExtract(p2, float_type, 3);
            auto zero = _builder.makeFloatConstant(0.0f);
            auto one = _builder.makeFloatConstant(1.0f);
            auto col0 = _builder.createCompositeConstruct(float4_type, {p0_x, p1_x, p2_x, zero});
            auto col1 = _builder.createCompositeConstruct(float4_type, {p0_y, p1_y, p2_y, zero});
            auto col2 = _builder.createCompositeConstruct(float4_type, {p0_z, p1_z, p2_z, zero});
            auto col3 = _builder.createCompositeConstruct(float4_type, {p0_w, p1_w, p2_w, one});
            auto mat_type = _convert_type(inst->type(), Usage::READ);
            id = _builder.createCompositeConstruct(mat_type, {col0, col1, col2, col3});
            break;
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: {
            auto instance_buffer =
                _resolve_accel_instance_buffer(inst->operand(0));
            auto uint_type = _builder.makeUintType(32);
            auto instance_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto byte_offset = _builder.createBinOp(
                spv::Op::OpIMul, uint_type, instance_index,
                _builder.makeUintConstant(64u));
            byte_offset = _builder.createBinOp(
                spv::Op::OpIAdd, uint_type, byte_offset,
                _builder.makeUintConstant(48u));
            auto packed = _emit_buffer_read_impl(
                instance_buffer, byte_offset, Type::of<uint32_t>(), 4u);
            if (inst->op() == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID) {
                id = _builder.createBinOp(
                    spv::Op::OpBitwiseAnd, uint_type, packed,
                    _builder.makeUintConstant(0x00ffffffu));
            } else {
                id = _builder.createBinOp(
                    spv::Op::OpShiftRightLogical, uint_type, packed,
                    _builder.makeUintConstant(24u));
            }
            break;
        }
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V resource query op {}.", xir::to_string(inst->op()));
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit resource query.");
    _value_map.emplace(inst, id);
}

spv::Id SpirvCodegenEntry::_load_direct_buffer_metadata(
    spv::Id buffer, StorageBufferMetadataField field,
    spv::Id target_type) noexcept {
    auto metadata = _direct_buffer_metadata_indices.find(buffer);
    LUISA_ASSERT(metadata != _direct_buffer_metadata_indices.end(),
                 "SPIR-V direct buffer {} has no view metadata.", buffer);
    LUISA_ASSERT(_argument_buffer_id != spv::NoResult,
                 "SPIR-V direct buffer metadata requires the argument buffer.");
    LUISA_ASSERT(_builder.isUintType(target_type),
                 "SPIR-V direct buffer metadata target must be unsigned integer.");
    auto target_width = _builder.getScalarTypeWidth(target_type);
    LUISA_ASSERT(target_width == 32u || target_width == 64u,
                 "SPIR-V direct buffer metadata target must be uint32 or uint64.");

    constexpr auto words_per_record =
        sizeof(StorageBufferMetadata) / sizeof(uint32_t);
    auto word = _buffer_metadata_offset / sizeof(uint32_t) +
                static_cast<size_t>(metadata->second) * words_per_record +
                storage_buffer_metadata_field_offset(field) /
                    sizeof(uint32_t);
    LUISA_ASSERT(word <= std::numeric_limits<uint32_t>::max() - 1u,
                 "SPIR-V direct buffer metadata word offset {} exceeds uint32.", word);
    auto load_word = [&](size_t index) noexcept {
        auto ptr = _create_access_chain(
            spv::StorageClass::StorageBuffer, _argument_buffer_id,
            {_builder.makeUintConstant(0u),
             _builder.makeUintConstant(static_cast<uint32_t>(index))});
        return _builder.createLoad(ptr, spv::NoPrecision);
    };
    auto low = load_word(word);
    if (target_width == 32u) { return low; }
    auto high = load_word(word + 1u);
    low = _builder.createUnaryOp(spv::Op::OpUConvert, target_type, low);
    high = _builder.createUnaryOp(spv::Op::OpUConvert, target_type, high);
    high = _builder.createBinOp(
        spv::Op::OpShiftLeftLogical, target_type, high,
        _builder.makeUint64Constant(32u));
    return _builder.createBinOp(spv::Op::OpBitwiseOr, target_type, low, high);
}

spv::Id SpirvCodegenEntry::_load_bindless_buffer_metadata(
    spv::Id bindless_array, spv::Id slot_index,
    StorageBufferMetadataField field,
    spv::Id target_type) noexcept {
    auto metadata = _bindless_buffer_metadata_ids.find(bindless_array);
    LUISA_ASSERT(metadata != _bindless_buffer_metadata_ids.end(),
                 "SPIR-V bindless array {} has no buffer-view metadata binding.",
                 bindless_array);
    LUISA_ASSERT(_builder.isUintType(target_type),
                 "SPIR-V bindless buffer metadata target must be unsigned integer.");
    auto target_width = _builder.getScalarTypeWidth(target_type);
    LUISA_ASSERT(target_width == 32u || target_width == 64u,
                 "SPIR-V bindless buffer metadata target must be uint32 or uint64.");

    auto uint_type = _builder.makeUintType(32);
    slot_index = _ensure_type(slot_index, uint_type);
    auto word = _builder.createBinOp(
        spv::Op::OpIMul, uint_type, slot_index,
        _builder.makeUintConstant(static_cast<uint32_t>(
            sizeof(StorageBufferMetadata) / sizeof(uint32_t))));
    auto field_word = storage_buffer_metadata_field_offset(field) /
                      sizeof(uint32_t);
    if (field_word != 0u) {
        word = _builder.createBinOp(
            spv::Op::OpIAdd, uint_type, word,
            _builder.makeUintConstant(
                static_cast<uint32_t>(field_word)));
    }
    auto load_word = [&](spv::Id index) noexcept {
        auto ptr = _create_access_chain(
            spv::StorageClass::StorageBuffer, metadata->second,
            {_builder.makeUintConstant(0u), index});
        return _builder.createLoad(ptr, spv::NoPrecision);
    };
    auto low = load_word(word);
    if (target_width == 32u) { return low; }
    auto high_word = _builder.createBinOp(
        spv::Op::OpIAdd, uint_type, word,
        _builder.makeUintConstant(1u));
    auto high = load_word(high_word);
    low = _builder.createUnaryOp(spv::Op::OpUConvert, target_type, low);
    high = _builder.createUnaryOp(spv::Op::OpUConvert, target_type, high);
    high = _builder.createBinOp(
        spv::Op::OpShiftLeftLogical, target_type, high,
        _builder.makeUint64Constant(32u));
    return _builder.createBinOp(
        spv::Op::OpBitwiseOr, target_type, low, high);
}

spv::Id SpirvCodegenEntry::_add_direct_buffer_bias(
    spv::Id buffer, spv::Id byte_offset) noexcept {
    if (!_direct_buffer_metadata_indices.contains(buffer)) { return byte_offset; }
    auto offset_type = _builder.getTypeId(byte_offset);
    LUISA_ASSERT(_builder.isUintType(offset_type),
                 "SPIR-V direct buffer byte offset must be unsigned integer.");
    auto address_type = _builder.getScalarTypeWidth(offset_type) == 64u ?
                            _builder.makeUintType(64) :
                            _builder.makeUintType(32);
    byte_offset = _ensure_type(byte_offset, address_type);
    auto bias = _load_direct_buffer_metadata(
        buffer, StorageBufferMetadataField::DESCRIPTOR_BIAS_BYTES,
        address_type);
    return _builder.createBinOp(
        spv::Op::OpIAdd, address_type, byte_offset, bias);
}

spv::Id SpirvCodegenEntry::_add_bindless_buffer_bias(
    spv::Id bindless_array, spv::Id slot_index,
    spv::Id byte_offset,
    xir::BindlessResourceAccess access) noexcept {
    auto offset_type = _builder.getTypeId(byte_offset);
    LUISA_ASSERT(_builder.isIntType(offset_type) ||
                     _builder.isUintType(offset_type),
                 "SPIR-V bindless buffer byte offset must be an integer scalar.");
    auto address_type = _builder.getScalarTypeWidth(offset_type) > 32u ?
                            _builder.makeUintType(64) :
                            _builder.makeUintType(32);
    byte_offset = _ensure_type(byte_offset, address_type);
    auto bias = access.typed ?
                    _ensure_type(
                        _load_bindless_slot_word(
                            bindless_array, slot_index,
                            4u, 1u, false),
                        address_type) :
                    _load_bindless_buffer_metadata(
                        bindless_array, slot_index,
                        StorageBufferMetadataField::DESCRIPTOR_BIAS_BYTES,
                        address_type);
    return _builder.createBinOp(
        spv::Op::OpIAdd, address_type, byte_offset, bias);
}

spv::Id SpirvCodegenEntry::_emit_buffer_read_impl(spv::Id buffer, spv::Id byte_offset, const Type *elem_type, size_t byte_alignment, spv::MemoryAccessMask memory_access) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto address_type = _builder.getTypeId(byte_offset);
    LUISA_ASSERT(_builder.isUintType(address_type),
                 "SPIR-V word-storage byte offsets must use an unsigned integer type.");
    auto address_width = _builder.getScalarTypeWidth(address_type);
    LUISA_ASSERT(address_width == 32 || address_width == 64,
                 "SPIR-V word-storage byte offsets must be 32- or 64-bit, got {} bits.",
                 address_width);
    auto make_address_constant = [&](size_t value) noexcept {
        return address_width == 64 ?
                   _builder.makeUint64Constant(static_cast<uint64_t>(value)) :
                   _builder.makeUintConstant(static_cast<uint32_t>(value));
    };
    auto spv_type = _convert_type(elem_type, Usage::READ);
    auto add_byte_offset = [&](spv::Id base, size_t offset) noexcept {
        return offset == 0u ? base :
                              _builder.createBinOp(spv::Op::OpIAdd, address_type, base,
                                                   make_address_constant(offset));
    };
    auto offset_alignment = [](size_t base_alignment, size_t offset) noexcept {
        return std::gcd(base_alignment, offset);
    };

    // Composite values are always decomposed first. Their ABI is defined by
    // Type::size()/alignment(), not by the number of 32-bit backing words.
    if (elem_type->is_vector()) {
        auto component_type = elem_type->element();
        std::vector<spv::Id> components;
        components.reserve(elem_type->dimension());
        for (auto i = 0u; i < elem_type->dimension(); ++i) {
            auto relative_offset = static_cast<size_t>(i) * component_type->size();
            components.emplace_back(_emit_buffer_read_impl(
                buffer, add_byte_offset(byte_offset, relative_offset), component_type,
                offset_alignment(byte_alignment, relative_offset), memory_access));
        }
        return _builder.createCompositeConstruct(spv_type, components);
    }
    if (elem_type->is_matrix()) {
        auto column_type = Type::vector(elem_type->element(), elem_type->dimension());
        std::vector<spv::Id> columns;
        columns.reserve(elem_type->dimension());
        for (auto i = 0u; i < elem_type->dimension(); ++i) {
            auto relative_offset = static_cast<size_t>(i) * column_type->size();
            columns.emplace_back(_emit_buffer_read_impl(
                buffer, add_byte_offset(byte_offset, relative_offset), column_type,
                offset_alignment(byte_alignment, relative_offset), memory_access));
        }
        return _builder.createCompositeConstruct(spv_type, columns);
    }
    if (elem_type->is_array()) {
        auto element_type = elem_type->element();
        std::vector<spv::Id> elements;
        elements.reserve(elem_type->dimension());
        for (auto i = 0u; i < elem_type->dimension(); ++i) {
            auto relative_offset = static_cast<size_t>(i) * element_type->size();
            elements.emplace_back(_emit_buffer_read_impl(
                buffer, add_byte_offset(byte_offset, relative_offset), element_type,
                offset_alignment(byte_alignment, relative_offset), memory_access));
        }
        return _builder.createCompositeConstruct(spv_type, elements);
    }
    if (elem_type->is_structure()) {
        std::vector<spv::Id> fields;
        fields.reserve(elem_type->members().size());
        size_t relative_offset = 0u;
        for (auto member : elem_type->members()) {
            relative_offset = luisa::align(relative_offset, member->alignment());
            fields.emplace_back(_emit_buffer_read_impl(
                buffer, add_byte_offset(byte_offset, relative_offset), member,
                offset_alignment(byte_alignment, relative_offset), memory_access));
            relative_offset += member->size();
        }
        return _builder.createCompositeConstruct(spv_type, fields);
    }

    LUISA_ASSERT(elem_type->is_scalar(),
                 "SPIR-V word-storage buffer read does not support type {}.",
                 elem_type->description());
    auto scalar_size = elem_type->size();
    LUISA_ASSERT(scalar_size == 1u || scalar_size == 2u || scalar_size == 4u || scalar_size == 8u,
                 "SPIR-V word-storage scalar read requires a 1-, 2-, 4-, or 8-byte type, got {} bytes for {}.",
                 scalar_size, elem_type->description());

    auto load_word = [&](spv::Id word_index) noexcept {
        auto ptr = _create_access_chain(
            _builder.getStorageClass(buffer), buffer,
            {_builder.makeUintConstant(0u), word_index});
        return _builder.createLoad(ptr, spv::NoPrecision, memory_access);
    };
    auto load_word_window = [&](spv::Id address, size_t byte_count,
                                bool can_cross_word) noexcept {
        LUISA_ASSERT(byte_count > 0u && byte_count <= 4u,
                     "SPIR-V word window must contain between one and four bytes.");
        auto word_index = _builder.createBinOp(
            spv::Op::OpUDiv, address_type, address, make_address_constant(4u));
        auto byte_in_word = _builder.createBinOp(
            spv::Op::OpUMod, address_type, address, make_address_constant(4u));
        byte_in_word = _ensure_type(byte_in_word, uint_type);
        auto shift = _builder.createBinOp(
            spv::Op::OpIMul, uint_type, byte_in_word, _builder.makeUintConstant(8u));
        auto first = _builder.createBinOp(
            spv::Op::OpShiftRightLogical, uint_type, load_word(word_index), shift);
        auto raw = first;
        if (can_cross_word) {
            // Select the first word again when the requested bytes do not cross
            // the boundary. This keeps the speculative second load in bounds
            // without relying on descriptor padding.
            auto crosses = _builder.createBinOp(
                spv::Op::OpUGreaterThan, _builder.makeBoolType(), byte_in_word,
                _builder.makeUintConstant(static_cast<uint32_t>(4u - byte_count)));
            auto increment = _builder.createTriOp(
                spv::Op::OpSelect, address_type, crosses,
                make_address_constant(1u), make_address_constant(0u));
            auto next_word_index = _builder.createBinOp(
                spv::Op::OpIAdd, address_type, word_index, increment);
            auto inverse_bytes = _builder.createBinOp(
                spv::Op::OpISub, uint_type, _builder.makeUintConstant(4u), byte_in_word);
            inverse_bytes = _builder.createBinOp(
                spv::Op::OpUMod, uint_type, inverse_bytes, _builder.makeUintConstant(4u));
            auto inverse_shift = _builder.createBinOp(
                spv::Op::OpIMul, uint_type, inverse_bytes, _builder.makeUintConstant(8u));
            auto second = _builder.createBinOp(
                spv::Op::OpShiftLeftLogical, uint_type,
                load_word(next_word_index), inverse_shift);
            second = _builder.createTriOp(
                spv::Op::OpSelect, uint_type, crosses, second,
                _builder.makeUintConstant(0u));
            raw = _builder.createBinOp(
                spv::Op::OpBitwiseOr, uint_type, first, second);
        }
        if (byte_count < 4u) {
            auto mask = static_cast<uint32_t>((uint64_t{1u} << (byte_count * 8u)) - 1u);
            raw = _builder.createBinOp(
                spv::Op::OpBitwiseAnd, uint_type, raw, _builder.makeUintConstant(mask));
        }
        return raw;
    };
    auto convert_low_word = [&](spv::Id raw) noexcept {
        if (elem_type->is_bool()) {
            return _builder.createBinOp(
                spv::Op::OpINotEqual, spv_type, raw, _builder.makeUintConstant(0u));
        }
        if (scalar_size == 4u) {
            return spv_type == uint_type ? raw :
                                           _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, raw);
        }
        auto bit_width = static_cast<int32_t>(scalar_size * 8u);
        auto raw_type = _builder.makeUintType(bit_width);
        auto narrowed = _builder.createUnaryOp(spv::Op::OpUConvert, raw_type, raw);
        return raw_type == spv_type ? narrowed :
                                      _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, narrowed);
    };

    if (scalar_size <= 4u) {
        spv::Id raw;
        if (scalar_size == 4u && byte_alignment >= 4u) {
            auto word_index = _builder.createBinOp(
                spv::Op::OpUDiv, address_type, byte_offset, make_address_constant(4u));
            raw = load_word(word_index);
        } else {
            raw = load_word_window(
                byte_offset, scalar_size, byte_alignment < scalar_size);
        }
        return convert_low_word(raw);
    }

    spv::Id low_word;
    spv::Id high_word;
    if (byte_alignment >= 4u) {
        auto word_index = _builder.createBinOp(
            spv::Op::OpUDiv, address_type, byte_offset, make_address_constant(4u));
        low_word = load_word(word_index);
        high_word = load_word(_builder.createBinOp(
            spv::Op::OpIAdd, address_type, word_index, make_address_constant(1u)));
    } else {
        low_word = load_word_window(byte_offset, 4u, true);
        high_word = load_word_window(add_byte_offset(byte_offset, 4u), 4u, true);
    }
    auto raw_type = _builder.makeVectorType(uint_type, 2);
    auto raw = _builder.createCompositeConstruct(raw_type, {low_word, high_word});
    return _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, raw);
}

spv::Id SpirvCodegenEntry::_emit_buffer_read(spv::Id buffer, spv::Id index, const Type *read_type, const Type *buffer_type, BufferIndexUnit index_unit, spv::MemoryAccessMask memory_access, size_t byte_index_alignment) noexcept {
    auto typed_buffer = buffer_type != nullptr && buffer_type->is_buffer() &&
                        buffer_type->element() != nullptr && !_buffer_uses_word_storage(buffer_type);
    if (typed_buffer) {
        LUISA_ASSERT(index_unit == BufferIndexUnit::ELEMENT,
                     "SPIR-V byte-addressed access cannot use a typed buffer binding.");
        LUISA_ASSERT(read_type == buffer_type->element(),
                     "SPIR-V typed buffer read type {} does not match buffer element type {}.",
                     read_type->description(), buffer_type->element()->description());
        if (_direct_buffer_metadata_indices.contains(buffer)) {
            auto index_type = _builder.getTypeId(index);
            LUISA_ASSERT(_builder.isIntType(index_type) ||
                             _builder.isUintType(index_type),
                         "SPIR-V typed buffer index must be an integer scalar.");
            auto address_type = _builder.getScalarTypeWidth(index_type) == 64u ?
                                    _builder.makeUintType(64) :
                                    _builder.makeUintType(32);
            auto bias = _load_direct_buffer_metadata(
                buffer, StorageBufferMetadataField::DESCRIPTOR_BIAS_BYTES,
                address_type);
            auto element_bias = _builder.createBinOp(
                spv::Op::OpUDiv, address_type, bias,
                _builder.getScalarTypeWidth(address_type) == 64u ?
                    _builder.makeUint64Constant(buffer_type->element()->size()) :
                    _builder.makeUintConstant(
                        static_cast<uint32_t>(buffer_type->element()->size())));
            index = _builder.createBinOp(
                spv::Op::OpIAdd, address_type,
                _ensure_type(index, address_type), element_bias);
        }
        auto ptr = _create_access_chain(
            _builder.getStorageClass(buffer), buffer,
            {_builder.makeUintConstant(0u), index});
        auto loaded = _builder.createLoad(ptr, spv::NoPrecision, memory_access);
        auto plain_type = _convert_type(read_type, Usage::READ);
        return _builder.getTypeId(loaded) == plain_type ? loaded :
                                                          _builder.createUnaryOp(spv::Op::OpCopyLogical, plain_type, loaded);
    }
    if (buffer_type != nullptr && buffer_type->is_buffer() && buffer_type->element() != nullptr) {
        LUISA_ASSERT(read_type == buffer_type->element(),
                     "SPIR-V word-storage buffer read type {} does not match buffer element type {}.",
                     read_type->description(), buffer_type->element()->description());
    }
    auto index_type = _builder.getTypeId(index);
    LUISA_ASSERT(_builder.isIntType(index_type) || _builder.isUintType(index_type),
                 "SPIR-V word-storage buffer index must be an integer scalar.");
    auto index_width = _builder.getScalarTypeWidth(index_type);
    auto address_type = index_width > 32 ? _builder.makeUintType(64) : _builder.makeUintType(32);
    auto byte_offset = _ensure_type(index, address_type);
    auto byte_alignment = index_unit == BufferIndexUnit::BYTE ?
                              byte_index_alignment :
                              size_t{1u};
    if (index_unit == BufferIndexUnit::ELEMENT) {
        byte_offset = _builder.createBinOp(
            spv::Op::OpIMul, address_type, byte_offset,
            index_width > 32 ?
                _builder.makeUint64Constant(static_cast<uint64_t>(read_type->size())) :
                _builder.makeUintConstant(static_cast<uint32_t>(read_type->size())));
        byte_alignment = std::gcd(read_type->size(), size_t{4u});
    }
    if (_direct_buffer_metadata_indices.contains(buffer)) {
        byte_offset = _add_direct_buffer_bias(buffer, byte_offset);
        // The Vulkan argument preprocessor validates direct typed-buffer
        // offsets and sizes as exact multiples of the logical element stride.
        // The runtime bias is therefore at least as aligned as the element
        // access planned above. Byte-buffer callers combine their XIR offset
        // proof with an independently established descriptor-bias proof; an
        // unbound view contributes alignment one. The host separately proves
        // the padded descriptor tail needed by a cross-word access.
    }
    return _emit_buffer_read_impl(buffer, byte_offset, read_type, byte_alignment, memory_access);
}

void SpirvCodegenEntry::_emit_buffer_write_word_masked(spv::Id buffer, spv::Id word_index, spv::Id value, spv::Id mask) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto bool_type = _builder.makeBoolType();
    auto ptr = _create_access_chain(
        _builder.getStorageClass(buffer), buffer,
        {_builder.makeUintConstant(0u), word_index});
    auto scope = _builder.makeUintConstant(static_cast<uint32_t>(spv::Scope::Device));
    auto semantics = _builder.makeUintConstant(static_cast<uint32_t>(spv::MemorySemanticsMask::MaskNone));
    auto expected_var = _builder.createVariable(
        spv::NoPrecision, spv::StorageClass::Function, uint_type, "masked_write_expected");
    auto initial = _builder.createOp(spv::Op::OpAtomicLoad, uint_type, {ptr, scope, semantics});
    _builder.createStore(initial, expected_var);

    auto *header = _create_physical_block();
    auto *body = _create_physical_block();
    auto *continue_block = _create_physical_block();
    auto *merge = _create_physical_block();
    _builder.createBranch(false, header);

    _set_current_tail(header);
    _builder.createLoopMerge(merge, continue_block, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, body);

    _set_current_tail(body);
    auto expected = _builder.createLoad(expected_var, spv::NoPrecision);
    auto clear_mask = _builder.createUnaryOp(spv::Op::OpNot, uint_type, mask);
    auto preserved = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, expected, clear_mask);
    auto inserted = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, value, mask);
    auto desired = _builder.createBinOp(spv::Op::OpBitwiseOr, uint_type, preserved, inserted);
    auto observed = _builder.createOp(
        spv::Op::OpAtomicCompareExchange, uint_type,
        {ptr, scope, semantics, semantics, desired, expected});
    _builder.createStore(observed, expected_var);
    auto succeeded = _builder.createBinOp(spv::Op::OpIEqual, bool_type, observed, expected);
    _builder.createConditionalBranch(succeeded, merge, continue_block);

    _set_current_tail(continue_block);
    _builder.createBranch(false, header);
    _set_current_tail(merge);
}

void SpirvCodegenEntry::_emit_buffer_write_impl(spv::Id buffer, spv::Id byte_offset, spv::Id value, const Type *elem_type, size_t byte_alignment, spv::MemoryAccessMask memory_access) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto address_type = _builder.getTypeId(byte_offset);
    LUISA_ASSERT(_builder.isUintType(address_type),
                 "SPIR-V word-storage byte offsets must use an unsigned integer type.");
    auto address_width = _builder.getScalarTypeWidth(address_type);
    LUISA_ASSERT(address_width == 32 || address_width == 64,
                 "SPIR-V word-storage byte offsets must be 32- or 64-bit, got {} bits.",
                 address_width);
    auto make_address_constant = [&](size_t offset) noexcept {
        return address_width == 64 ?
                   _builder.makeUint64Constant(static_cast<uint64_t>(offset)) :
                   _builder.makeUintConstant(static_cast<uint32_t>(offset));
    };
    auto add_byte_offset = [&](spv::Id base, size_t offset) noexcept {
        return offset == 0u ? base :
                              _builder.createBinOp(spv::Op::OpIAdd, address_type, base,
                                                   make_address_constant(offset));
    };
    auto offset_alignment = [](size_t base_alignment, size_t offset) noexcept {
        return std::gcd(base_alignment, offset);
    };

    if (elem_type->is_vector()) {
        auto component_type = elem_type->element();
        auto component_spv_type = _convert_type(component_type, Usage::READ);
        for (auto i = 0u; i < elem_type->dimension(); ++i) {
            auto relative_offset = static_cast<size_t>(i) * component_type->size();
            auto component = _builder.createCompositeExtract(value, component_spv_type, i);
            _emit_buffer_write_impl(
                buffer, add_byte_offset(byte_offset, relative_offset), component, component_type,
                offset_alignment(byte_alignment, relative_offset), memory_access);
        }
        return;
    }
    if (elem_type->is_matrix()) {
        auto column_type = Type::vector(elem_type->element(), elem_type->dimension());
        auto column_spv_type = _convert_type(column_type, Usage::READ);
        for (auto i = 0u; i < elem_type->dimension(); ++i) {
            auto relative_offset = static_cast<size_t>(i) * column_type->size();
            auto column = _builder.createCompositeExtract(value, column_spv_type, i);
            _emit_buffer_write_impl(
                buffer, add_byte_offset(byte_offset, relative_offset), column, column_type,
                offset_alignment(byte_alignment, relative_offset), memory_access);
        }
        return;
    }
    if (elem_type->is_array()) {
        auto element_type = elem_type->element();
        auto element_spv_type = _convert_type(element_type, Usage::READ);
        for (auto i = 0u; i < elem_type->dimension(); ++i) {
            auto relative_offset = static_cast<size_t>(i) * element_type->size();
            auto element = _builder.createCompositeExtract(value, element_spv_type, i);
            _emit_buffer_write_impl(
                buffer, add_byte_offset(byte_offset, relative_offset), element, element_type,
                offset_alignment(byte_alignment, relative_offset), memory_access);
        }
        return;
    }
    if (elem_type->is_structure()) {
        auto members = elem_type->members();
        size_t relative_offset = 0u;
        for (auto i = 0u; i < members.size(); ++i) {
            auto member = members[i];
            relative_offset = luisa::align(relative_offset, member->alignment());
            auto field = _builder.createCompositeExtract(
                value, _convert_type(member, Usage::READ), static_cast<uint32_t>(i));
            _emit_buffer_write_impl(
                buffer, add_byte_offset(byte_offset, relative_offset), field, member,
                offset_alignment(byte_alignment, relative_offset), memory_access);
            relative_offset += member->size();
        }
        return;
    }

    LUISA_ASSERT(elem_type->is_scalar(),
                 "SPIR-V word-storage buffer write does not support type {}.",
                 elem_type->description());
    auto scalar_size = elem_type->size();
    LUISA_ASSERT(scalar_size == 1u || scalar_size == 2u || scalar_size == 4u || scalar_size == 8u,
                 "SPIR-V word-storage scalar write requires a 1-, 2-, 4-, or 8-byte type, got {} bytes for {}.",
                 scalar_size, elem_type->description());

    auto store_word = [&](spv::Id word_index, spv::Id word) noexcept {
        auto ptr = _create_access_chain(
            _builder.getStorageClass(buffer), buffer,
            {_builder.makeUintConstant(0u), word_index});
        _builder.createStore(word, ptr, memory_access);
    };
    auto scalar_word = [&]() noexcept {
        if (elem_type->is_bool()) {
            return _builder.createOp(
                spv::Op::OpSelect, uint_type,
                {value, _builder.makeUintConstant(1u), _builder.makeUintConstant(0u)});
        }
        auto value_type = _builder.getTypeId(value);
        if (scalar_size == 4u) {
            return value_type == uint_type ? value :
                                             _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, value);
        }
        auto bit_width = static_cast<int32_t>(scalar_size * 8u);
        auto raw_type = _builder.makeUintType(bit_width);
        auto raw = value_type == raw_type ? value :
                                            _builder.createUnaryOp(spv::Op::OpBitcast, raw_type, value);
        return _builder.createUnaryOp(spv::Op::OpUConvert, uint_type, raw);
    };

    if (scalar_size == 4u && byte_alignment >= 4u) {
        auto word_index = _builder.createBinOp(
            spv::Op::OpUDiv, address_type, byte_offset, make_address_constant(4u));
        store_word(word_index, scalar_word());
        return;
    }
    if (scalar_size == 8u && byte_alignment >= 4u) {
        auto raw_type = _builder.makeVectorType(uint_type, 2);
        auto raw = _builder.createUnaryOp(spv::Op::OpBitcast, raw_type, value);
        auto word_index = _builder.createBinOp(
            spv::Op::OpUDiv, address_type, byte_offset, make_address_constant(4u));
        store_word(word_index, _builder.createCompositeExtract(raw, uint_type, 0u));
        store_word(_builder.createBinOp(
                       spv::Op::OpIAdd, address_type, word_index, make_address_constant(1u)),
                   _builder.createCompositeExtract(raw, uint_type, 1u));
        return;
    }

    spv::Id low_word = spv::NoResult;
    spv::Id high_word = spv::NoResult;
    if (scalar_size <= 4u) {
        low_word = scalar_word();
    } else {
        auto raw_type = _builder.makeVectorType(uint_type, 2);
        auto raw = _builder.createUnaryOp(spv::Op::OpBitcast, raw_type, value);
        low_word = _builder.createCompositeExtract(raw, uint_type, 0u);
        high_word = _builder.createCompositeExtract(raw, uint_type, 1u);
    }
    auto write_word_window = [&](spv::Id address, spv::Id raw, size_t byte_count, bool can_cross_word) noexcept {
        LUISA_ASSERT(byte_count > 0u && byte_count <= 4u,
                     "SPIR-V masked word window must contain between one and four bytes.");
        auto word_index = _builder.createBinOp(
            spv::Op::OpUDiv, address_type, address, make_address_constant(4u));
        auto byte_in_word = _builder.createBinOp(
            spv::Op::OpUMod, address_type, address, make_address_constant(4u));
        byte_in_word = _ensure_type(byte_in_word, uint_type);
        auto shift = _builder.createBinOp(
            spv::Op::OpIMul, uint_type, byte_in_word, _builder.makeUintConstant(8u));
        auto raw_mask_value = byte_count == 4u ?
                                  0xffffffffu :
                                  static_cast<uint32_t>((uint64_t{1u} << (byte_count * 8u)) - 1u);
        auto raw_mask = _builder.makeUintConstant(raw_mask_value);
        auto first_mask = _builder.createBinOp(
            spv::Op::OpShiftLeftLogical, uint_type, raw_mask, shift);
        auto first_value = _builder.createBinOp(
            spv::Op::OpShiftLeftLogical, uint_type, raw, shift);
        _emit_buffer_write_word_masked(buffer, word_index, first_value, first_mask);
        if (!can_cross_word) { return; }

        auto crosses = _builder.createBinOp(
            spv::Op::OpUGreaterThan, _builder.makeBoolType(), byte_in_word,
            _builder.makeUintConstant(static_cast<uint32_t>(4u - byte_count)));
        auto increment = _builder.createTriOp(
            spv::Op::OpSelect, address_type, crosses,
            make_address_constant(1u), make_address_constant(0u));
        auto next_word_index = _builder.createBinOp(
            spv::Op::OpIAdd, address_type, word_index, increment);
        auto inverse_bytes = _builder.createBinOp(
            spv::Op::OpISub, uint_type, _builder.makeUintConstant(4u), byte_in_word);
        inverse_bytes = _builder.createBinOp(
            spv::Op::OpUMod, uint_type, inverse_bytes, _builder.makeUintConstant(4u));
        auto inverse_shift = _builder.createBinOp(
            spv::Op::OpIMul, uint_type, inverse_bytes, _builder.makeUintConstant(8u));
        auto second_mask = _builder.createBinOp(
            spv::Op::OpShiftRightLogical, uint_type, raw_mask, inverse_shift);
        second_mask = _builder.createTriOp(
            spv::Op::OpSelect, uint_type, crosses, second_mask, _builder.makeUintConstant(0u));
        auto second_value = _builder.createBinOp(
            spv::Op::OpShiftRightLogical, uint_type, raw, inverse_shift);
        second_value = _builder.createTriOp(
            spv::Op::OpSelect, uint_type, crosses, second_value, _builder.makeUintConstant(0u));
        _emit_buffer_write_word_masked(buffer, next_word_index, second_value, second_mask);
    };
    if (scalar_size <= 4u) {
        auto can_cross_word = scalar_size > 1u && byte_alignment < scalar_size;
        write_word_window(byte_offset, low_word, scalar_size, can_cross_word);
    } else {
        write_word_window(byte_offset, low_word, 4u, true);
        write_word_window(add_byte_offset(byte_offset, 4u), high_word, 4u, true);
    }
}

void SpirvCodegenEntry::_emit_buffer_write(spv::Id buffer, spv::Id index, spv::Id value, const Type *value_type, const Type *buffer_type, BufferIndexUnit index_unit, spv::MemoryAccessMask memory_access, size_t byte_index_alignment) noexcept {
    auto typed_buffer = buffer_type != nullptr && buffer_type->is_buffer() &&
                        buffer_type->element() != nullptr && !_buffer_uses_word_storage(buffer_type);
    if (typed_buffer) {
        LUISA_ASSERT(index_unit == BufferIndexUnit::ELEMENT,
                     "SPIR-V byte-addressed access cannot use a typed buffer binding.");
        LUISA_ASSERT(value_type == buffer_type->element(),
                     "SPIR-V typed buffer write type {} does not match buffer element type {}.",
                     value_type->description(), buffer_type->element()->description());
        if (_direct_buffer_metadata_indices.contains(buffer)) {
            auto index_type = _builder.getTypeId(index);
            LUISA_ASSERT(_builder.isIntType(index_type) ||
                             _builder.isUintType(index_type),
                         "SPIR-V typed buffer index must be an integer scalar.");
            auto address_type = _builder.getScalarTypeWidth(index_type) == 64u ?
                                    _builder.makeUintType(64) :
                                    _builder.makeUintType(32);
            auto bias = _load_direct_buffer_metadata(
                buffer, StorageBufferMetadataField::DESCRIPTOR_BIAS_BYTES,
                address_type);
            auto element_bias = _builder.createBinOp(
                spv::Op::OpUDiv, address_type, bias,
                _builder.getScalarTypeWidth(address_type) == 64u ?
                    _builder.makeUint64Constant(buffer_type->element()->size()) :
                    _builder.makeUintConstant(
                        static_cast<uint32_t>(buffer_type->element()->size())));
            index = _builder.createBinOp(
                spv::Op::OpIAdd, address_type,
                _ensure_type(index, address_type), element_bias);
        }
        auto ptr = _create_access_chain(
            _builder.getStorageClass(buffer), buffer,
            {_builder.makeUintConstant(0u), index});
        auto pointee_type = _builder.getContainedTypeId(_builder.getTypeId(ptr));
        if (_builder.getTypeId(value) != pointee_type) {
            value = _builder.createUnaryOp(spv::Op::OpCopyLogical, pointee_type, value);
        }
        _builder.createStore(value, ptr, memory_access);
        return;
    }
    if (buffer_type != nullptr && buffer_type->is_buffer() && buffer_type->element() != nullptr) {
        LUISA_ASSERT(value_type == buffer_type->element(),
                     "SPIR-V word-storage buffer write type {} does not match buffer element type {}.",
                     value_type->description(), buffer_type->element()->description());
    }
    auto index_type = _builder.getTypeId(index);
    LUISA_ASSERT(_builder.isIntType(index_type) || _builder.isUintType(index_type),
                 "SPIR-V word-storage buffer index must be an integer scalar.");
    auto index_width = _builder.getScalarTypeWidth(index_type);
    auto address_type = index_width > 32 ? _builder.makeUintType(64) : _builder.makeUintType(32);
    auto byte_offset = _ensure_type(index, address_type);
    auto byte_alignment = index_unit == BufferIndexUnit::BYTE ?
                              byte_index_alignment :
                              size_t{1u};
    if (index_unit == BufferIndexUnit::ELEMENT) {
        byte_offset = _builder.createBinOp(
            spv::Op::OpIMul, address_type, byte_offset,
            index_width > 32 ?
                _builder.makeUint64Constant(static_cast<uint64_t>(value_type->size())) :
                _builder.makeUintConstant(static_cast<uint32_t>(value_type->size())));
        byte_alignment = std::gcd(value_type->size(), size_t{4u});
    }
    if (_direct_buffer_metadata_indices.contains(buffer)) {
        byte_offset = _add_direct_buffer_bias(buffer, byte_offset);
        // Direct typed-buffer metadata preserves the logical element-stride
        // alignment proved by storage_buffer_descriptor_range(). Byte-buffer
        // callers have already intersected the index and runtime-bias facts.
    }
    _emit_buffer_write_impl(buffer, byte_offset, value, value_type, byte_alignment, memory_access);
}

namespace {

[[nodiscard]] spv::Id spirv_cooperative_component_type_constant(
    spv::Builder &builder, const xir::Value *interp) noexcept {
    uint64_t raw = 0u;
    LUISA_ASSERT(
        interp != nullptr &&
            xir::try_decode_constant_nonnegative_integer(interp, raw),
        "SPIR-V cooperative-vector interpretation must be a constant.");
    uint32_t component = 0u;
    switch (static_cast<CoopRefVecType>(raw)) {
        case CoopRefVecType::UINT8: component = 7u; break;      // UnsignedInt8NV
        case CoopRefVecType::INT8: component = 3u; break;       // SignedInt8NV
        case CoopRefVecType::UINT32: component = 9u; break;     // UnsignedInt32NV
        case CoopRefVecType::INT32: component = 5u; break;      // SignedInt32NV
        case CoopRefVecType::FLOAT16: component = 0u; break;    // Float16NV
        case CoopRefVecType::FLOAT32: component = 1u; break;    // Float32NV
        case CoopRefVecType::FLOAT8_E4M3: component = 1000491002u; break;
        case CoopRefVecType::FLOAT8_E5M2: component = 1000491003u; break;
    }
    return builder.makeIntConstant(static_cast<int32_t>(component));
}

[[nodiscard]] spv::Id spirv_cooperative_scalar_component_constant(
    spv::Builder &builder, const Type *type) noexcept {
    uint32_t component = 0u;
    switch (type == nullptr ? Type::Tag::CUSTOM : type->tag()) {
        case Type::Tag::FLOAT16: component = 0u; break;
        case Type::Tag::FLOAT32: component = 1u; break;
        case Type::Tag::FLOAT64: component = 2u; break;
        case Type::Tag::INT8: component = 3u; break;
        case Type::Tag::INT16: component = 4u; break;
        case Type::Tag::INT32: component = 5u; break;
        case Type::Tag::INT64: component = 6u; break;
        case Type::Tag::UINT8: component = 7u; break;
        case Type::Tag::UINT16: component = 8u; break;
        case Type::Tag::UINT32: component = 9u; break;
        case Type::Tag::UINT64: component = 10u; break;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V cooperative-vector input component type {} is not "
                "representable.",
                type == nullptr ? "<null>" : type->description());
    }
    return builder.makeIntConstant(static_cast<int32_t>(component));
}

}// namespace

void SpirvCodegenEntry::_emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    spv::Id id = spv::NoResult;
    auto uint_type = _builder.makeUintType(32);

    switch (inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            id = _emit_buffer_read(buffer, index, inst->type(), inst->operand(0)->type(), BufferIndexUnit::ELEMENT);
            break;
        }
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            _builder.createMemoryBarrier(spv::Scope::Device,
                                         spv::MemorySemanticsAllMemory |
                                             spv::MemorySemanticsMask::AcquireRelease);
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            id = _emit_buffer_read(buffer, index, inst->type(), inst->operand(0)->type(), BufferIndexUnit::ELEMENT, spv::MemoryAccessMask::Volatile);
            break;
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _emit_value(inst->operand(1));
            auto byte_alignment = std::gcd(
                xir::integer_value_guaranteed_alignment(
                    inst->operand(1), 4u),
                _direct_buffer_bias_alignment(inst->operand(0)));
            id = _emit_buffer_read(
                buffer, byte_index, inst->type(), nullptr,
                BufferIndexUnit::BYTE,
                spv::MemoryAccessMask::MaskNone, byte_alignment);
            break;
        }
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            _builder.createMemoryBarrier(spv::Scope::Device,
                                         spv::MemorySemanticsAllMemory |
                                             spv::MemorySemanticsMask::AcquireRelease);
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _emit_value(inst->operand(1));
            auto byte_alignment = std::gcd(
                xir::integer_value_guaranteed_alignment(
                    inst->operand(1), 4u),
                _direct_buffer_bias_alignment(inst->operand(0)));
            id = _emit_buffer_read(
                buffer, byte_index, inst->type(), nullptr,
                BufferIndexUnit::BYTE,
                spv::MemoryAccessMask::Volatile, byte_alignment);
            break;
        }
        case xir::ResourceReadOp::TEXTURE2D_READ:
        case xir::ResourceReadOp::TEXTURE3D_READ: {
            auto tex_array = _emit_value(inst->operand(0));
            auto coord = _emit_value(inst->operand(1));
            auto tex = _load_texture(tex_array);
            if (_is_storage_image_map.at(tex_array)) {
                auto image_type = _builder.getImageType(tex);
                if (_builder.getImageTypeFormat(image_type) == spv::ImageFormat::Unknown) {
                    _require_target_feature(
                        target_feature::storage_image_read_without_format,
                        _target_features.storage_image_read_without_format);
                    _builder.addCapability(spv::Capability::StorageImageReadWithoutFormat);
                }
                id = _builder.createOp(spv::Op::OpImageRead, type, {tex, coord});
            } else {
                std::vector<spv::IdImmediate> operands;
                operands.emplace_back(true, tex);
                operands.emplace_back(true, coord);
                operands.emplace_back(false, spv::ImageOperandsMask::Lod);
                operands.emplace_back(true, _builder.makeUintConstant(0u));
                id = _builder.createOp(spv::Op::OpImageFetch, type, operands);
            }
            break;
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ:
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ:
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: {
            auto op = inst->op();
            auto is_2d = op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ ||
                         op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL;
            auto has_level = op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL ||
                             op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL;
            auto bindless_array = _emit_value(inst->operand(0));
            auto binding = _load_bindless_texture_binding(
                bindless_array, inst->operand(1), is_2d,
                inst->bindless_access());
            auto coord = _emit_value(inst->operand(2));
            auto lod = has_level ?
                           _ensure_type(_emit_value(inst->operand(3)), uint_type) :
                           _builder.makeUintConstant(0u);
            std::vector<spv::IdImmediate> operands;
            operands.emplace_back(true, binding.image);
            operands.emplace_back(true, coord);
            operands.emplace_back(false, spv::ImageOperandsMask::Lod);
            operands.emplace_back(true, lod);
            id = _builder.createOp(
                spv::Op::OpImageFetch, type, operands);
            break;
        }
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto binding = _load_bindless_buffer_binding(
                bindless_array, inst->operand(1),
                inst->bindless_access());
            auto elem_index = _emit_value(inst->operand(2));
            auto index_type = _builder.getTypeId(elem_index);
            LUISA_ASSERT(_builder.isIntType(index_type) ||
                             _builder.isUintType(index_type),
                         "SPIR-V bindless buffer element index must be an integer scalar.");
            auto address_type = _builder.getScalarTypeWidth(index_type) > 32u ?
                                    _builder.makeUintType(64) :
                                    uint_type;
            auto byte_index = _builder.createBinOp(
                spv::Op::OpIMul, address_type,
                _ensure_type(elem_index, address_type),
                _builder.getScalarTypeWidth(address_type) == 64u ?
                    _builder.makeUint64Constant(inst->type()->size()) :
                    _builder.makeUintConstant(
                        static_cast<uint32_t>(inst->type()->size())));
            byte_index = _add_bindless_buffer_bias(
                bindless_array, binding.slot_index, byte_index,
                inst->bindless_access());
            id = _emit_buffer_read(
                binding.buffer, byte_index, inst->type(), nullptr,
                BufferIndexUnit::BYTE);
            break;
        }
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto binding = _load_bindless_buffer_binding(
                bindless_array, inst->operand(1),
                inst->bindless_access());
            auto byte_index = _emit_value(inst->operand(2));
            byte_index = _add_bindless_buffer_bias(
                bindless_array, binding.slot_index, byte_index,
                inst->bindless_access());
            id = _emit_buffer_read(
                binding.buffer, byte_index, inst->type(), nullptr,
                BufferIndexUnit::BYTE);
            break;
        }
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LOAD:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_VECTOR_LOAD: {
            auto bindless = inst->op() == xir::ResourceReadOp::BINDLESS_COOPERATIVE_VECTOR_LOAD;
            spv::Id pointer;
            spv::Id offset;
            if (bindless) {
                auto bindless_array = _emit_value(inst->operand(0));
                auto binding = _load_bindless_buffer_binding(
                    bindless_array, inst->operand(1), inst->bindless_access());
                pointer = _emit_cooperative_array_pointer(binding.buffer);
                offset = _ensure_type(_emit_value(inst->operand(2)), uint_type);
                offset = _add_bindless_buffer_bias(
                    bindless_array, binding.slot_index, offset,
                    inst->bindless_access());
            } else {
                pointer = _emit_cooperative_array_pointer(_emit_value(inst->operand(0)));
                offset = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            }
            std::vector<spv::IdImmediate> operands;
            operands.emplace_back(true, pointer);
            operands.emplace_back(true, offset);
            id = _builder.createOp(spv::Op::OpCooperativeVectorLoadNV, type, operands);
            break;
        }
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SPLAT: {
            auto scalar = _emit_value(inst->operand(0));
            auto component = _convert_type(inst->type()->element(), Usage::READ);
            scalar = _ensure_type(scalar, component);
            std::vector<spv::Id> constituents(
                inst->type()->dimension(), scalar);
            id = _builder.createCompositeConstruct(type, constituents);
            break;
        }
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_CAST: {
            auto value = _emit_value(inst->operand(0));
            auto component = _convert_type(inst->type()->element(), Usage::READ);
            auto dst_elem = inst->type()->element();
            auto src_elem = inst->operand(0)->type()->element();
            spv::Op op;
            if (dst_elem->is_float32() || dst_elem->is_float16() || dst_elem->is_float64()) {
                op = (src_elem->is_int() || src_elem->is_uint()) ?
                         (src_elem->is_int() ? spv::Op::OpConvertSToF : spv::Op::OpConvertUToF) :
                         spv::Op::OpFConvert;
            } else if (dst_elem->is_int()) {
                op = (src_elem->is_float32() || src_elem->is_float16() || src_elem->is_float64()) ?
                         spv::Op::OpConvertFToS : spv::Op::OpSConvert;
            } else if (dst_elem->is_uint()) {
                op = (src_elem->is_float32() || src_elem->is_float16() || src_elem->is_float64()) ?
                         spv::Op::OpConvertFToU : spv::Op::OpUConvert;
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "SPIR-V cooperative-vector cast to {} is not supported.",
                    dst_elem->description());
            }
            // build the result component-wise through composite extract/construct
            auto n = inst->type()->dimension();
            std::vector<spv::Id> constituents;
            constituents.reserve(n);
            for (auto i = 0u; i < n; i++) {
                auto elem = _builder.createCompositeExtract(
                    value, _convert_type(src_elem, Usage::READ), {i});
                constituents.emplace_back(
                    _builder.createUnaryOp(op, component, elem));
            }
            id = _builder.createCompositeConstruct(type, constituents);
            break;
        }
        case xir::ResourceReadOp::COOPERATIVE_MUL:
        case xir::ResourceReadOp::COOPERATIVE_MUL_ADD:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL_ADD: {
            auto bindless = inst->op() == xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL ||
                            inst->op() == xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL_ADD;
            auto mul_add = inst->op() == xir::ResourceReadOp::COOPERATIVE_MUL_ADD ||
                           inst->op() == xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL_ADD;
            // canonical operand layout (see ast2xir):
            //   plain:   (matrix_buffer, matrix_offset, [bias_buffer, bias_offset,] input, matrix_interp[, bias_interp])
            //   bindless: (bindless, matrix_slot, matrix_offset, [bias_slot, bias_offset,] input, matrix_interp[, bias_interp])
            auto input_index = bindless ? (mul_add ? 5u : 3u) : (mul_add ? 4u : 2u);
            auto interp_index = input_index + 1u;
            auto input = _emit_value(inst->operand(input_index));
            std::vector<spv::IdImmediate> operands;
            operands.emplace_back(true, input);
            operands.emplace_back(
                true, spirv_cooperative_scalar_component_constant(
                          _builder, inst->operand(input_index)->type()->element()));
            spv::Id matrix_pointer;
            spv::Id matrix_offset;
            if (bindless) {
                auto bindless_array = _emit_value(inst->operand(0));
                auto binding = _load_bindless_buffer_binding(
                    bindless_array, inst->operand(1), inst->bindless_access());
                matrix_pointer = _emit_cooperative_array_pointer(binding.buffer);
                matrix_offset = _ensure_type(_emit_value(inst->operand(2)), uint_type);
                matrix_offset = _add_bindless_buffer_bias(
                    bindless_array, binding.slot_index, matrix_offset,
                    inst->bindless_access());
            } else {
                matrix_pointer = _emit_cooperative_array_pointer(_emit_value(inst->operand(0)));
                matrix_offset = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            }
            operands.emplace_back(true, matrix_pointer);
            operands.emplace_back(true, matrix_offset);
            operands.emplace_back(
                true, spirv_cooperative_component_type_constant(
                          _builder, inst->operand(interp_index)));
            if (mul_add) {
                spv::Id bias_pointer;
                spv::Id bias_offset;
                if (bindless) {
                    auto bindless_array = _emit_value(inst->operand(0));
                    auto binding = _load_bindless_buffer_binding(
                        bindless_array, inst->operand(3), inst->bindless_access());
                    bias_pointer = _emit_cooperative_array_pointer(binding.buffer);
                    bias_offset = _ensure_type(_emit_value(inst->operand(4)), uint_type);
                    bias_offset = _add_bindless_buffer_bias(
                        bindless_array, binding.slot_index, bias_offset,
                        inst->bindless_access());
                } else {
                    bias_pointer = _emit_cooperative_array_pointer(_emit_value(inst->operand(2)));
                    bias_offset = _ensure_type(_emit_value(inst->operand(3)), uint_type);
                }
                operands.emplace_back(true, bias_pointer);
                operands.emplace_back(true, bias_offset);
                operands.emplace_back(
                    true, spirv_cooperative_component_type_constant(
                              _builder, inst->operand(interp_index + 1u)));
            }
            operands.emplace_back(
                true, _builder.makeIntConstant(
                          static_cast<int32_t>(inst->type()->dimension())));// M
            operands.emplace_back(
                true, _builder.makeIntConstant(
                          static_cast<int32_t>(
                              inst->operand(input_index)->type()->dimension())));// K
            operands.emplace_back(
                true, _builder.makeIntConstant(2));// InferencingOptimalNV
            operands.emplace_back(true, _builder.makeBoolConstant(false));
            id = _builder.createOp(
                mul_add ? spv::Op::OpCooperativeVectorMatrixMulAddNV :
                          spv::Op::OpCooperativeVectorMatrixMulNV,
                type, operands);
            break;
        }
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD:
            LUISA_NOT_IMPLEMENTED();
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit resource read.");
    _value_map.emplace(inst, id);
}

spv::Id SpirvCodegenEntry::_emit_cooperative_array_pointer(spv::Id buffer) noexcept {
    // Cooperative-vector memory instructions take a pointer to the scalar
    // array inside the buffer block struct (member 0).
    return _create_access_chain(_builder.getStorageClass(buffer), buffer,
                                {_builder.makeUintConstant(0u)});
}

void SpirvCodegenEntry::_emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept {
    auto uint_type = _builder.makeUintType(32);

    switch (inst->op()) {
        case xir::ResourceWriteOp::BUFFER_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            _emit_buffer_write(buffer, index, value, inst->operand(2)->type(), inst->operand(0)->type(), BufferIndexUnit::ELEMENT);
            break;
        }
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            _emit_buffer_write(buffer, index, value, inst->operand(2)->type(), inst->operand(0)->type(), BufferIndexUnit::ELEMENT, spv::MemoryAccessMask::Volatile);
            _builder.createMemoryBarrier(spv::Scope::Device,
                                         spv::MemorySemanticsAllMemory |
                                             spv::MemorySemanticsMask::AcquireRelease);
            break;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            auto value_type = inst->operand(2)->type();
            auto byte_alignment = std::gcd(
                xir::integer_value_guaranteed_alignment(
                    inst->operand(1), 4u),
                _direct_buffer_bias_alignment(inst->operand(0)));
            _emit_buffer_write(
                buffer, byte_index, value, value_type, nullptr,
                BufferIndexUnit::BYTE,
                spv::MemoryAccessMask::MaskNone, byte_alignment);
            break;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            auto value_type = inst->operand(2)->type();
            auto byte_alignment = std::gcd(
                xir::integer_value_guaranteed_alignment(
                    inst->operand(1), 4u),
                _direct_buffer_bias_alignment(inst->operand(0)));
            _emit_buffer_write(
                buffer, byte_index, value, value_type, nullptr,
                BufferIndexUnit::BYTE,
                spv::MemoryAccessMask::Volatile, byte_alignment);
            _builder.createMemoryBarrier(spv::Scope::Device,
                                         spv::MemorySemanticsAllMemory |
                                             spv::MemorySemanticsMask::AcquireRelease);
            break;
        }
        case xir::ResourceWriteOp::TEXTURE2D_WRITE:
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: {
            auto tex_array = _resolve_writable_resource(inst->operand(0));
            auto coord = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            auto value_type = _builder.getTypeId(value);
            auto value_component_count = _builder.getNumTypeComponents(value_type);
            LUISA_ASSERT(
                value_component_count == 4u,
                "SPIR-V texture writes require the verified four-component "
                "texel value, got {} component(s).",
                value_component_count);
            auto tex = _load_texture(tex_array);
            auto image_type = _builder.getImageType(tex);
            if (_builder.getImageTypeFormat(image_type) == spv::ImageFormat::Unknown) {
                _require_target_feature(
                    target_feature::storage_image_write_without_format,
                    _target_features.storage_image_write_without_format);
                _builder.addCapability(spv::Capability::StorageImageWriteWithoutFormat);
            }
            _builder.createNoResultOp(spv::Op::OpImageWrite, {tex, coord, value});
            break;
        }
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto binding = _load_bindless_buffer_binding(
                bindless_array, inst->operand(1),
                inst->bindless_access());
            auto elem_index = _emit_value(inst->operand(2));
            auto value = _emit_value(inst->operand(3));
            auto elem_type = inst->operand(3)->type();
            auto index_type = _builder.getTypeId(elem_index);
            LUISA_ASSERT(_builder.isIntType(index_type) ||
                             _builder.isUintType(index_type),
                         "SPIR-V bindless buffer element index must be an integer scalar.");
            auto address_type = _builder.getScalarTypeWidth(index_type) > 32u ?
                                    _builder.makeUintType(64) :
                                    uint_type;
            auto byte_index = _builder.createBinOp(
                spv::Op::OpIMul, address_type,
                _ensure_type(elem_index, address_type),
                _builder.getScalarTypeWidth(address_type) == 64u ?
                    _builder.makeUint64Constant(elem_type->size()) :
                    _builder.makeUintConstant(
                        static_cast<uint32_t>(elem_type->size())));
            byte_index = _add_bindless_buffer_bias(
                bindless_array, binding.slot_index, byte_index,
                inst->bindless_access());
            _emit_buffer_write(
                binding.buffer, byte_index, value, elem_type, nullptr,
                BufferIndexUnit::BYTE);
            break;
        }
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto binding = _load_bindless_buffer_binding(
                bindless_array, inst->operand(1),
                inst->bindless_access());
            auto byte_index = _emit_value(inst->operand(2));
            auto value = _emit_value(inst->operand(3));
            auto value_type = inst->operand(3)->type();
            byte_index = _add_bindless_buffer_bias(
                bindless_array, binding.slot_index, byte_index,
                inst->bindless_access());
            _emit_buffer_write(
                binding.buffer, byte_index, value, value_type, nullptr,
                BufferIndexUnit::BYTE);
            break;
        }
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_STORE:
        case xir::ResourceWriteOp::BINDLESS_COOPERATIVE_VECTOR_STORE: {
            auto bindless = inst->op() == xir::ResourceWriteOp::BINDLESS_COOPERATIVE_VECTOR_STORE;
            spv::Id pointer;
            spv::Id offset;
            spv::Id object;
            if (bindless) {
                auto bindless_array = _emit_value(inst->operand(0));
                auto binding = _load_bindless_buffer_binding(
                    bindless_array, inst->operand(1), inst->bindless_access());
                pointer = _emit_cooperative_array_pointer(binding.buffer);
                offset = _ensure_type(_emit_value(inst->operand(2)), uint_type);
                offset = _add_bindless_buffer_bias(
                    bindless_array, binding.slot_index, offset,
                    inst->bindless_access());
                object = _emit_value(inst->operand(3));
            } else {
                pointer = _emit_cooperative_array_pointer(_emit_value(inst->operand(0)));
                offset = _ensure_type(_emit_value(inst->operand(1)), uint_type);
                object = _emit_value(inst->operand(2));
            }
            std::vector<spv::IdImmediate> operands;
            operands.emplace_back(true, pointer);
            operands.emplace_back(true, offset);
            operands.emplace_back(true, object);
            _builder.createNoResultOp(spv::Op::OpCooperativeVectorStoreNV, operands);
            break;
        }
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_ACCUMULATE: {
            _builder.addCapability(spv::Capability::CooperativeVectorTrainingNV);
            auto pointer = _emit_cooperative_array_pointer(_emit_value(inst->operand(0)));
            auto offset = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto value = _emit_value(inst->operand(2));
            std::vector<spv::IdImmediate> operands;
            operands.emplace_back(true, pointer);
            operands.emplace_back(true, offset);
            operands.emplace_back(true, value);
            _builder.createNoResultOp(
                spv::Op::OpCooperativeVectorReduceSumAccumulateNV, operands);
            break;
        }
        case xir::ResourceWriteOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE:
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_WORKGROUP_STORE:
            LUISA_NOT_IMPLEMENTED();
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: {
            LUISA_ASSERT(
                inst->operand_count() == 2u &&
                    _is_indirect_dispatch_type(inst->operand(0)->type()) &&
                    inst->operand(1)->type() == Type::of<uint32_t>(),
                "SPIR-V indirect-dispatch count write expects "
                "(LC_IndirectDispatchBuffer, uint), got {} operands.",
                inst->operand_count());
            auto buffer = _emit_value(inst->operand(0));
            auto count = _ensure_type(
                _emit_value(inst->operand(1)), uint_type);
            auto word_count = _builder.createArrayLength(buffer, 0u, 32u);
            auto capacity_words = _builder.createBinOp(
                spv::Op::OpISub, uint_type, word_count,
                _builder.makeUintConstant(
                    IndirectDispatchLayout::header_word_count));
            auto capacity = _builder.createBinOp(
                spv::Op::OpUDiv, uint_type, capacity_words,
                _builder.makeUintConstant(
                    IndirectDispatchLayout::record_word_count));
            auto exceeds_capacity = _builder.createBinOp(
                spv::Op::OpUGreaterThan, _builder.makeBoolType(),
                count, capacity);
            count = _builder.createOp(
                spv::Op::OpSelect, uint_type,
                {exceeds_capacity, capacity, count});
            auto count_ptr = _create_access_chain(
                spv::StorageClass::StorageBuffer, buffer,
                {_builder.makeUintConstant(0u),
                 _builder.makeUintConstant(0u)});
            _builder.createStore(count, count_ptr);
            break;
        }
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: {
            LUISA_ASSERT(
                inst->operand_count() == 5u &&
                    _is_indirect_dispatch_type(inst->operand(0)->type()) &&
                    inst->operand(1)->type() == Type::of<uint32_t>() &&
                    inst->operand(2)->type() == Type::of<uint3>() &&
                    inst->operand(3)->type() == Type::of<uint3>() &&
                    inst->operand(4)->type() == Type::of<uint32_t>(),
                "SPIR-V indirect-dispatch record write expects "
                "(LC_IndirectDispatchBuffer, uint, uint3, uint3, uint), "
                "got {} operands.",
                inst->operand_count());
            auto buffer = _emit_value(inst->operand(0));
            auto index = _ensure_type(
                _emit_value(inst->operand(1)), uint_type);
            auto block_size = _emit_value(inst->operand(2));
            auto logical_size = _emit_value(inst->operand(3));
            auto kernel_id = _ensure_type(
                _emit_value(inst->operand(4)), uint_type);
            auto bool_type = _builder.makeBoolType();
            auto word_count = _builder.createArrayLength(buffer, 0u, 32u);
            auto capacity_words = _builder.createBinOp(
                spv::Op::OpISub, uint_type, word_count,
                _builder.makeUintConstant(
                    IndirectDispatchLayout::header_word_count));
            auto capacity = _builder.createBinOp(
                spv::Op::OpUDiv, uint_type, capacity_words,
                _builder.makeUintConstant(
                    IndirectDispatchLayout::record_word_count));
            auto valid_index = _builder.createBinOp(
                spv::Op::OpULessThan, bool_type, index, capacity);
            auto valid_block = _builder.makeBoolConstant(true);
            for (auto component = 0u; component < 3u; ++component) {
                auto block_component = _builder.createCompositeExtract(
                    block_size, uint_type, component);
                auto nonzero = _builder.createBinOp(
                    spv::Op::OpINotEqual, bool_type, block_component,
                    _builder.makeUintConstant(0u));
                valid_block = _builder.createBinOp(
                    spv::Op::OpLogicalAnd, bool_type,
                    valid_block, nonzero);
            }

            auto *write_block = _create_physical_block();
            auto *merge_block = _create_physical_block();
            auto selection_merge = std::make_unique<spv::Instruction>(
                spv::Op::OpSelectionMerge);
            selection_merge->reserveOperands(2u);
            selection_merge->addIdOperand(merge_block->getId());
            selection_merge->addImmediateOperand(
                spv::SelectionControlMask::MaskNone);
            _builder.getBuildPoint()->addInstruction(
                std::move(selection_merge));
            _builder.createConditionalBranch(
                valid_index, write_block, merge_block);

            _set_current_tail(write_block);
            auto record_word = _builder.createBinOp(
                spv::Op::OpIMul, uint_type, index,
                _builder.makeUintConstant(
                    IndirectDispatchLayout::record_word_count));
            record_word = _builder.createBinOp(
                spv::Op::OpIAdd, uint_type, record_word,
                _builder.makeUintConstant(
                    IndirectDispatchLayout::header_word_count));
            auto store_word = [&](uint32_t word_offset,
                                  spv::Id value) noexcept {
                auto word = word_offset == 0u ? record_word :
                                                _builder.createBinOp(
                                                    spv::Op::OpIAdd,
                                                    uint_type, record_word,
                                                    _builder.makeUintConstant(
                                                        word_offset));
                auto ptr = _create_access_chain(
                    spv::StorageClass::StorageBuffer, buffer,
                    {_builder.makeUintConstant(0u), word});
                _builder.createStore(value, ptr);
            };
            for (auto component = 0u; component < 3u; ++component) {
                auto logical_component = _builder.createCompositeExtract(
                    logical_size, uint_type, component);
                auto block_component = _builder.createCompositeExtract(
                    block_size, uint_type, component);
                auto safe_block_component = _builder.createOp(
                    spv::Op::OpSelect, uint_type,
                    {valid_block, block_component,
                     _builder.makeUintConstant(1u)});
                store_word(
                    IndirectDispatchLayout::logical_size_word + component,
                    logical_component);
                auto quotient = _builder.createBinOp(
                    spv::Op::OpUDiv, uint_type, logical_component,
                    safe_block_component);
                auto remainder = _builder.createBinOp(
                    spv::Op::OpUMod, uint_type, logical_component,
                    safe_block_component);
                auto has_remainder = _builder.createBinOp(
                    spv::Op::OpINotEqual, bool_type, remainder,
                    _builder.makeUintConstant(0u));
                auto round_up = _builder.createOp(
                    spv::Op::OpSelect, uint_type,
                    {has_remainder,
                     _builder.makeUintConstant(1u),
                     _builder.makeUintConstant(0u)});
                auto group_count = _builder.createBinOp(
                    spv::Op::OpIAdd, uint_type, quotient, round_up);
                group_count = _builder.createOp(
                    spv::Op::OpSelect, uint_type,
                    {valid_block, group_count,
                     _builder.makeUintConstant(0u)});
                store_word(
                    IndirectDispatchLayout::group_count_word + component,
                    group_count);
            }
            store_word(
                IndirectDispatchLayout::kernel_id_word, kernel_id);
            _builder.createBranch(false, merge_block);
            _set_current_tail(merge_block);
            break;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: {
            auto buffer = _resolve_accel_instance_buffer(inst->operand(0));
            auto index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto base_word_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, index, _builder.makeUintConstant(16u));
            auto base_byte_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, index, _builder.makeUintConstant(64u));
            switch (inst->op()) {
                case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: {
                    auto mat = _emit_value(inst->operand(2));
                    auto float_type = _builder.makeFloatType(32);
                    auto vec4_type = _builder.makeVectorType(float_type, 4);
                    auto f32v4 = Type::vector(Type::of<float>(), 4);
                    for (auto row = 0u; row < 3u; ++row) {
                        std::vector<spv::Id> comps;
                        comps.reserve(4);
                        for (auto col = 0u; col < 4u; ++col) {
                            auto col_vec = _builder.createCompositeExtract(mat, vec4_type, col);
                            auto comp = _builder.createCompositeExtract(col_vec, float_type, row);
                            comps.push_back(comp);
                        }
                        auto row_vec = _builder.createCompositeConstruct(vec4_type, comps);
                        auto byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset,
                                                                _builder.makeUintConstant(row * 16u));
                        _emit_buffer_write_impl(buffer, byte_offset, row_vec, f32v4, 4u);
                    }
                    break;
                }
                case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK: {
                    auto value = _ensure_type(_emit_value(inst->operand(2)), uint_type);
                    value = _builder.createBinOp(
                        spv::Op::OpBitwiseAnd, uint_type, value,
                        _builder.makeUintConstant(0xffu));
                    value = _builder.createBinOp(
                        spv::Op::OpShiftLeftLogical, uint_type, value,
                        _builder.makeUintConstant(24u));
                    auto word_offset = _builder.createBinOp(
                        spv::Op::OpIAdd, uint_type, base_word_offset,
                        _builder.makeUintConstant(12u));
                    _emit_buffer_write_word_masked(
                        buffer, word_offset, value,
                        _builder.makeUintConstant(0xff000000u));
                    break;
                }
                case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: {
                    auto value = _ensure_type(_emit_value(inst->operand(2)), uint_type);
                    value = _builder.createBinOp(
                        spv::Op::OpBitwiseAnd, uint_type, value,
                        _builder.makeUintConstant(0x00ffffffu));
                    auto word_offset = _builder.createBinOp(
                        spv::Op::OpIAdd, uint_type, base_word_offset,
                        _builder.makeUintConstant(12u));
                    _emit_buffer_write_word_masked(
                        buffer, word_offset, value,
                        _builder.makeUintConstant(0x00ffffffu));
                    break;
                }
                case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY: {
                    auto opaque = _emit_value(inst->operand(2));
                    auto value = _builder.createOp(spv::Op::OpSelect, uint_type, {opaque, _builder.makeUintConstant(4u), _builder.makeUintConstant(8u)});
                    value = _builder.createBinOp(
                        spv::Op::OpShiftLeftLogical, uint_type, value,
                        _builder.makeUintConstant(24u));
                    auto word_offset = _builder.createBinOp(
                        spv::Op::OpIAdd, uint_type, base_word_offset,
                        _builder.makeUintConstant(13u));
                    _emit_buffer_write_word_masked(
                        buffer, word_offset, value,
                        _builder.makeUintConstant(0xff000000u));
                    break;
                }
                default: break;
            }
            break;
        }
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V resource write op {}.", xir::to_string(inst->op()));
    }
}

void SpirvCodegenEntry::_emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept {
    spv::Id id = spv::NoResult;
    auto subgroup_scope = _builder.makeUintConstant(static_cast<uint32_t>(spv::Scope::Subgroup));
    auto group_op_reduce = static_cast<uint32_t>(spv::GroupOperation::Reduce);
    auto group_op_exclusive_scan = static_cast<uint32_t>(spv::GroupOperation::ExclusiveScan);
    auto enable_subgroup_basic = [this]() noexcept {
        _require_target_feature(
            target_feature::subgroup_basic,
            _target_features.subgroup_basic);
        _builder.addCapability(spv::Capability::GroupNonUniform);
    };
    auto enable_subgroup_vote = [this, &enable_subgroup_basic]() noexcept {
        enable_subgroup_basic();
        _require_target_feature(
            target_feature::subgroup_vote,
            _target_features.subgroup_vote);
        _builder.addCapability(spv::Capability::GroupNonUniformVote);
    };
    auto enable_subgroup_arithmetic = [this, &enable_subgroup_basic]() noexcept {
        enable_subgroup_basic();
        _require_target_feature(
            target_feature::subgroup_arithmetic,
            _target_features.subgroup_arithmetic);
        _builder.addCapability(spv::Capability::GroupNonUniformArithmetic);
    };
    auto enable_subgroup_ballot = [this, &enable_subgroup_basic]() noexcept {
        enable_subgroup_basic();
        _require_target_feature(
            target_feature::subgroup_ballot,
            _target_features.subgroup_ballot);
        _builder.addCapability(spv::Capability::GroupNonUniformBallot);
    };
    auto enable_subgroup_shuffle = [this, &enable_subgroup_basic]() noexcept {
        enable_subgroup_basic();
        _require_target_feature(
            target_feature::subgroup_shuffle,
            _target_features.subgroup_shuffle);
        _builder.addCapability(spv::Capability::GroupNonUniformShuffle);
    };
    auto require_value_type = [this, inst]() noexcept {
        _require_subgroup_type(
            inst->operand(0)->type(), xir::to_string(inst->op()));
    };
    auto emit_shuffle = [this](const Type *type, spv::Id value,
                               spv::Id scope, spv::Id lane) noexcept {
        auto recurse = [&](auto &&self, const Type *component_type,
                           spv::Id component) noexcept -> spv::Id {
            if (component_type->is_scalar()) {
                return _builder.createOp(
                    spv::Op::OpGroupNonUniformShuffle,
                    _convert_type(component_type, Usage::READ),
                    {scope, component, lane});
            }
            if (component_type->is_vector() || component_type->is_matrix()) {
                auto element_type = component_type->is_matrix() ?
                                        Type::vector(component_type->element(),
                                                     component_type->dimension()) :
                                        component_type->element();
                auto element_spv_type =
                    _convert_type(element_type, Usage::READ);
                std::vector<spv::Id> elements;
                elements.reserve(component_type->dimension());
                for (auto i = 0u; i < component_type->dimension(); ++i) {
                    auto element = _builder.createCompositeExtract(
                        component, element_spv_type, i);
                    elements.emplace_back(
                        self(self, element_type, element));
                }
                return _builder.createCompositeConstruct(
                    _convert_type(component_type, Usage::READ), elements);
            }
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V subgroup shuffle does not support XIR type {}.",
                component_type->description());
        };
        return recurse(recurse, type, value);
    };
    switch (inst->op()) {
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: {
            _builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Workgroup,
                                          spv::MemorySemanticsMask::WorkgroupMemory |
                                              spv::MemorySemanticsMask::AcquireRelease);
            break;
        }
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: {
            enable_subgroup_basic();
            id = _builder.createOp(spv::Op::OpGroupNonUniformElect, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::Id>{subgroup_scope});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: {
            enable_subgroup_vote();
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformAll, _convert_type(inst->type(), Usage::READ),
                                   {subgroup_scope, val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: {
            enable_subgroup_vote();
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformAny, _convert_type(inst->type(), Usage::READ),
                                   {subgroup_scope, val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseAnd, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseOr, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseXor, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            auto elem_type = inst->operand(0)->type();
            auto scalar_elem = elem_type->is_vector() || elem_type->is_matrix() ? elem_type->element() : elem_type;
            spv::Op op;
            if (scalar_elem->is_float()) {
                op = spv::Op::OpGroupNonUniformFAdd;
            } else {
                op = spv::Op::OpGroupNonUniformIAdd;
            }
            id = _builder.createOp(op, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            auto elem_type = inst->operand(0)->type();
            auto scalar_elem = elem_type->is_vector() || elem_type->is_matrix() ? elem_type->element() : elem_type;
            spv::Op op;
            if (scalar_elem->is_float()) {
                op = spv::Op::OpGroupNonUniformFMul;
            } else {
                op = spv::Op::OpGroupNonUniformIMul;
            }
            id = _builder.createOp(op, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            auto elem_type = inst->operand(0)->type();
            auto scalar_elem = elem_type->is_vector() || elem_type->is_matrix() ? elem_type->element() : elem_type;
            spv::Op op;
            if (scalar_elem->is_float()) {
                op = spv::Op::OpGroupNonUniformFMin;
            } else if (scalar_elem->is_uint()) {
                op = spv::Op::OpGroupNonUniformUMin;
            } else {
                op = spv::Op::OpGroupNonUniformSMin;
            }
            id = _builder.createOp(op, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            auto elem_type = inst->operand(0)->type();
            auto scalar_elem = elem_type->is_vector() || elem_type->is_matrix() ? elem_type->element() : elem_type;
            spv::Op op;
            if (scalar_elem->is_float()) {
                op = spv::Op::OpGroupNonUniformFMax;
            } else if (scalar_elem->is_uint()) {
                op = spv::Op::OpGroupNonUniformUMax;
            } else {
                op = spv::Op::OpGroupNonUniformSMax;
            }
            id = _builder.createOp(op, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: {
            enable_subgroup_vote();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            auto value_type = inst->operand(0)->type();
            auto bool_type = _builder.makeBoolType();
            if (value_type->is_scalar()) {
                LUISA_ASSERT(inst->type()->is_bool(),
                             "Scalar XIR warp_active_all_equal must return bool.");
                id = _builder.createOp(
                    spv::Op::OpGroupNonUniformAllEqual, bool_type,
                    {subgroup_scope, val});
            } else {
                LUISA_ASSERT(value_type->is_vector() &&
                                 inst->type()->is_vector() &&
                                 inst->type()->element()->is_bool() &&
                                 value_type->dimension() == inst->type()->dimension(),
                             "Vector XIR warp_active_all_equal must return a "
                             "same-width bool vector.");
                auto element_type = value_type->element();
                auto element_spv_type =
                    _convert_type(element_type, Usage::READ);
                std::vector<spv::Id> equal_components;
                equal_components.reserve(value_type->dimension());
                for (auto i = 0u; i < value_type->dimension(); ++i) {
                    auto component = _builder.createCompositeExtract(
                        val, element_spv_type, i);
                    equal_components.emplace_back(_builder.createOp(
                        spv::Op::OpGroupNonUniformAllEqual, bool_type,
                        {subgroup_scope, component}));
                }
                id = _builder.createCompositeConstruct(
                    _convert_type(inst->type(), Usage::READ),
                    equal_components);
            }
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: {
            enable_subgroup_ballot();
            auto val = _emit_value(inst->operand(0));
            auto uint_type = _builder.makeUintType(32);
            // Ballot returns a uvec4 (4 x uint32) in SPIR-V
            auto ballot_type = _builder.makeVectorType(uint_type, 4);
            auto ballot = _builder.createOp(spv::Op::OpGroupNonUniformBallot, ballot_type,
                                            {subgroup_scope, val});
            id = _builder.createOp(spv::Op::OpGroupNonUniformBallotBitCount, uint_type,
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, ballot}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: {
            enable_subgroup_ballot();
            auto val = _emit_value(inst->operand(0));
            auto uint_type = _builder.makeUintType(32);
            auto ballot_type = _builder.makeVectorType(uint_type, 4);
            id = _builder.createOp(spv::Op::OpGroupNonUniformBallot, ballot_type,
                                   {subgroup_scope, val});
            break;
        }
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: {
            enable_subgroup_ballot();
            auto uint_type = _builder.makeUintType(32);
            auto ballot_type = _builder.makeVectorType(uint_type, 4);
            auto true_val = _builder.makeBoolConstant(true);
            auto ballot = _builder.createOp(spv::Op::OpGroupNonUniformBallot, ballot_type,
                                            {subgroup_scope, true_val});
            id = _builder.createOp(spv::Op::OpGroupNonUniformBallotFindLSB, _convert_type(inst->type(), Usage::READ),
                                   {subgroup_scope, ballot});
            break;
        }
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS: {
            enable_subgroup_ballot();
            auto val = _emit_value(inst->operand(0));
            auto uint_type = _builder.makeUintType(32);
            auto ballot_type = _builder.makeVectorType(uint_type, 4);
            auto ballot = _builder.createOp(spv::Op::OpGroupNonUniformBallot, ballot_type,
                                            {subgroup_scope, val});
            id = _builder.createOp(spv::Op::OpGroupNonUniformBallotBitCount, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_exclusive_scan},
                                       {true, ballot}});
            break;
        }
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            auto elem_type = inst->operand(0)->type();
            auto scalar_elem = elem_type->is_vector() || elem_type->is_matrix() ? elem_type->element() : elem_type;
            spv::Op op = scalar_elem->is_float() ? spv::Op::OpGroupNonUniformFAdd : spv::Op::OpGroupNonUniformIAdd;
            id = _builder.createOp(op, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_exclusive_scan},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: {
            enable_subgroup_arithmetic();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            auto elem_type = inst->operand(0)->type();
            auto scalar_elem = elem_type->is_vector() || elem_type->is_matrix() ? elem_type->element() : elem_type;
            spv::Op op = scalar_elem->is_float() ? spv::Op::OpGroupNonUniformFMul : spv::Op::OpGroupNonUniformIMul;
            id = _builder.createOp(op, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_exclusive_scan},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_READ_LANE: {
            enable_subgroup_shuffle();
            require_value_type();
            auto val = _emit_value(inst->operand(0));
            auto lane = _emit_value(inst->operand(1));
            id = emit_shuffle(inst->type(), val, subgroup_scope, lane);
            break;
        }
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: {
            enable_subgroup_ballot();
            enable_subgroup_shuffle();
            require_value_type();
            auto uint_type = _builder.makeUintType(32);
            auto ballot_type = _builder.makeVectorType(uint_type, 4);
            auto true_val = _builder.makeBoolConstant(true);
            auto ballot = _builder.createOp(spv::Op::OpGroupNonUniformBallot, ballot_type,
                                            {subgroup_scope, true_val});
            auto first_lane = _builder.createOp(spv::Op::OpGroupNonUniformBallotFindLSB, uint_type,
                                                {subgroup_scope, ballot});
            auto val = _emit_value(inst->operand(0));
            id = emit_shuffle(inst->type(), val, subgroup_scope, first_lane);
            break;
        }
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER:
            // This operation is an optimization-only scheduling hint. Ignoring
            // it preserves all defined shader results, consistent with the
            // fallback, plain-CUDA, HLSL, and Metal no-op implementations.
            break;
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V thread group op {}.", xir::to_string(inst->op()));
    }
    if (id != spv::NoResult && inst->type() != nullptr) {
        _value_map.emplace(inst, id);
    }
}
void SpirvCodegenEntry::_emit_instruction(const xir::Instruction *inst) noexcept {
    auto set_result = [&](spv::Id id) noexcept {
        if (inst->type() != nullptr) {
            _value_map.emplace(inst, id);
        }
    };
    switch (inst->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::ALLOCA: {
            static_cast<void>(_emit_alloca(static_cast<const xir::AllocaInst *>(inst)));
            break;
        }
        case xir::DerivedInstructionTag::LOAD: {
            auto load = static_cast<const xir::LoadInst *>(inst);
            auto ptr = _emit_value(load->variable());
            auto ptr_type = _builder.getTypeId(ptr);
            auto pointee_type = _builder.getContainedTypeId(ptr_type);
            if (_builder.getTypeClass(pointee_type) == spv::Op::OpTypeRayQueryKHR) {
                set_result(ptr);
            } else {
                auto id = _builder.createLoad(ptr, spv::NoPrecision);
                set_result(id);
            }
            break;
        }
        case xir::DerivedInstructionTag::STORE: {
            auto store = static_cast<const xir::StoreInst *>(inst);
            auto ptr = _emit_value(store->variable());
            auto val = _emit_value(store->value());
            auto ptr_type = _builder.getTypeId(ptr);
            auto pointee_type = _builder.getContainedTypeId(ptr_type);
            auto val_type = _builder.getTypeId(val);
            if (_builder.getTypeClass(pointee_type) == spv::Op::OpTypeRayQueryKHR) {
                // OpCopyMemory on OpTypeRayQueryKHR is forbidden since SPIR-V Rev 15.
                // Lifetime preflight guarantees this is the alloca's single,
                // dominating initializer. Remap it to the initialized opaque
                // object; actual query copies/reassignments are rejected.
                static_cast<void>(_ray_query_state(val));
                _value_map[store->variable()] = val;
            } else {
                LUISA_ASSERT(
                    pointee_type == val_type,
                    "SPIR-V store type mismatch escaped XIR dialect validation.");
                _builder.createStore(val, ptr);
            }
            break;
        }
        case xir::DerivedInstructionTag::GEP: {
            auto gep = static_cast<const xir::GEPInst *>(inst);
            auto base = _emit_value(gep->base());
            luisa::vector<const xir::Value *> index_values;
            index_values.reserve(gep->index_count());
            for (auto index_use : gep->index_uses()) {
                index_values.emplace_back(index_use->value());
            }
            auto index_plan = plan_spirv_aggregate_indices(
                gep->base()->type(), luisa::span{index_values});
            if (!index_plan) {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid SPIR-V GEP aggregate indices: {}",
                    index_plan.diagnostic);
            }
            LUISA_ASSERT(index_plan.indexed_type == gep->type(),
                         "SPIR-V GEP index plan reaches {}, but the instruction result is {}.",
                         index_plan.indexed_type->description(),
                         gep->type()->description());
            auto indices = _emit_aggregate_access_indices(index_plan);
            auto storage = _builder.getStorageClass(base);
            auto id = _create_access_chain(storage, base, indices);
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::ARITHMETIC: {
            auto *arithmetic =
                static_cast<const xir::ArithmeticInst *>(inst);
            if (_enable_fast_math &&
                is_deferred_fast_math_fma_multiply(arithmetic)) {
                break;
            }
            _emit_arithmetic_inst(arithmetic);
            break;
        }
        case xir::DerivedInstructionTag::CALL: {
            auto call = static_cast<const xir::CallInst *>(inst);
            auto callee = static_cast<const xir::Function *>(call->callee());
            auto callee_func = _function_map.at(callee);
            std::vector<spv::Id> args;
            luisa::vector<const xir::Argument *> callable_arg_list;
            for (auto arg : callee->arguments()) {
                callable_arg_list.emplace_back(arg);
            }
            LUISA_ASSERT(call->argument_count() == callable_arg_list.size(),
                         "SPIR-V call argument count {} does not match callee argument count {}.",
                         call->argument_count(), callable_arg_list.size());
            const luisa::vector<bool> *used_mask = nullptr;
            if (auto it = _callable_arg_used.find(callee);
                it != _callable_arg_used.end()) {
                used_mask = &it->second;
                LUISA_ASSERT(used_mask->size() == callable_arg_list.size(),
                             "SPIR-V callable used-argument mask has {} entries for {} arguments.",
                             used_mask->size(), callable_arg_list.size());
            }
            auto expected_parameter_count = callable_arg_list.size();
            for (auto i = 0u; i < callable_arg_list.size(); ++i) {
                auto callable_arg = callable_arg_list[i];
                if (used_mask != nullptr && !(*used_mask)[i] &&
                    callable_arg->is_resource()) {
                    --expected_parameter_count;
                } else if (callable_arg->is_reference() &&
                           _is_ray_query_type(callable_arg->type())) {
                    expected_parameter_count += 2u;
                }
            }
            auto callee_requires_dispatch_metadata =
                _functions_requiring_dispatch_metadata.contains(callee);
            if (callee_requires_dispatch_metadata) {
                expected_parameter_count++;
            }
            size_t idx = 0u;
            for (auto arg_use : call->argument_uses()) {
                auto callable_arg = callable_arg_list[idx];
                auto skip_unused_resource =
                    used_mask != nullptr && !(*used_mask)[idx] &&
                    callable_arg->is_resource();
                if (!skip_unused_resource) {
                    auto arg_val = [&] {
                        if (callable_arg->is_resource() &&
                            callable_arg->type()->is_texture()) {
                            auto usage = _function_argument_usage_of(
                                callee, callable_arg);
                            if ((luisa::to_underlying(usage) &
                                 luisa::to_underlying(Usage::WRITE)) != 0u) {
                                return _resolve_writable_resource(
                                    arg_use->value());
                            }
                        }
                        return _emit_value(arg_use->value());
                    }();
                    if (callable_arg->is_reference()) {
                        auto opcode = _builder.getOpCode(arg_val);
                        LUISA_ASSERT(
                            opcode == spv::Op::OpVariable ||
                                opcode == spv::Op::OpFunctionParameter,
                            "SPIR-V reference actual {} for callable '{}' was not legalized to "
                            "OpVariable or OpFunctionParameter (got {}).",
                            idx, callee->name().value_or("<unnamed>"),
                            static_cast<uint32_t>(opcode));
                    }
                    args.emplace_back(arg_val);
                    if (callable_arg->is_reference() &&
                        _is_ray_query_type(callable_arg->type())) {
                        auto &state = _ray_query_state(arg_val);
                        args.emplace_back(state.initial_ray);
                        args.emplace_back(state.proceed_state);
                    }
                }
                ++idx;
            }
            if (callee_requires_dispatch_metadata) {
                LUISA_ASSERT(
                    _dispatch_metadata.packed != spv::NoResult,
                    "SPIR-V caller '{}' has no dispatch metadata to pass to "
                    "callable '{}'.",
                    inst->parent_function()->name().value_or("<unnamed>"),
                    callee->name().value_or("<unnamed>"));
                args.emplace_back(_dispatch_metadata.packed);
            }
            LUISA_ASSERT(idx == callable_arg_list.size() &&
                             args.size() == expected_parameter_count,
                         "SPIR-V callable argument planning produced an inconsistent parameter count.");
            auto id = _builder.createFunctionCall(callee_func, args);
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::CAST: {
            auto cast = static_cast<const xir::CastInst *>(inst);
            auto val = _emit_value(cast->value());
            auto from = cast->value()->type();
            auto to = cast->type();
            auto spv_to = _convert_type(to, Usage::READ);
            spv::Id id = spv::NoResult;
            if (cast->op() == xir::CastOp::BITWISE_CAST) {
                auto from_is_bool = from->is_bool() || from->is_bool_vector();
                auto to_is_bool = to->is_bool() || to->is_bool_vector();
                if (from_is_bool || to_is_bool) {
                    LUISA_NOT_IMPLEMENTED(
                        "SPIR-V bitwise cast involving boolean type: {} -> {}.",
                        from->description(), to->description());
                }
                auto logical_size = [](const Type *type) noexcept {
                    return type->is_vector() ?
                               type->element()->size() * type->dimension() :
                               type->size();
                };
                LUISA_ASSERT(logical_size(from) == logical_size(to),
                             "SPIR-V bitwise cast width mismatch: {} -> {}.",
                             from->description(), to->description());
                id = _builder.createUnaryOp(spv::Op::OpBitcast, spv_to, val);
            } else {
                auto make_scalar_zero_one = [&](const Type *type) noexcept {
                    spv::Id zero = spv::NoResult;
                    spv::Id one = spv::NoResult;
                    if (type->is_int() || type->is_uint()) {
                        auto bit_width = static_cast<int32_t>(type->size() * 8);
                        auto int_type = _builder.makeIntegerType(
                            bit_width, type->is_int());
                        if (bit_width == 64) {
                            zero = _builder.makeInt64Constant(
                                int_type, 0u, false);
                            one = _builder.makeInt64Constant(
                                int_type, 1u, false);
                        } else {
                            zero = _builder.makeIntConstant(
                                int_type, 0u, false);
                            one = _builder.makeIntConstant(
                                int_type, 1u, false);
                        }
                    } else if (type->is_float()) {
                        auto bit_width = static_cast<int32_t>(type->size() * 8);
                        if (bit_width == 8) {
                            if (type->is_float8_e5m2()) {
                                zero = _builder.makeFloatE5M2Constant(0.0f);
                                one = _builder.makeFloatE5M2Constant(1.0f);
                            } else if (type->is_float8_e4m3()) {
                                zero = _builder.makeFloatE4M3Constant(0.0f);
                                one = _builder.makeFloatE4M3Constant(1.0f);
                            }
                        } else if (bit_width == 16) {
                            zero = _builder.makeFloat16Constant(0.0f);
                            one = _builder.makeFloat16Constant(1.0f);
                        } else if (bit_width == 32) {
                            zero = _builder.makeFloatConstant(0.0f);
                            one = _builder.makeFloatConstant(1.0f);
                        } else if (bit_width == 64) {
                            zero = _builder.makeDoubleConstant(0.0);
                            one = _builder.makeDoubleConstant(1.0);
                        }
                    }
                    if (zero == spv::NoResult || one == spv::NoResult) {
                        LUISA_NOT_IMPLEMENTED("SPIR-V scalar cast constant for {}.", type->description());
                    }
                    return std::pair{zero, one};
                };
                auto static_cast_scalar = [&](spv::Id scalar, const Type *source,
                                              const Type *target) noexcept {
                    auto spv_target = _convert_type(target, Usage::READ);
                    if (source == target) { return scalar; }
                    if (source->is_bool()) {
                        auto [zero, one] = make_scalar_zero_one(target);
                        return _builder.createTriOp(spv::Op::OpSelect, spv_target,
                                                    scalar, one, zero);
                    }
                    if (target->is_bool()) {
                        // SPV_EXT_float8 permits FP8 values in conversions and
                        // OpSelect, but not in floating-point comparisons.
                        // Preserve C-style truthiness by widening before the
                        // unordered comparison (so NaN remains true).
                        if (source->is_float8()) {
                            source = Type::of<float>();
                            scalar = _builder.createUnaryOp(
                                spv::Op::OpFConvert,
                                _convert_type(source, Usage::READ), scalar);
                        }
                        auto [zero, one] = make_scalar_zero_one(source);
                        static_cast<void>(one);
                        return _builder.createBinOp(
                            source->is_float() ? spv::Op::OpFUnordNotEqual :
                                                 spv::Op::OpINotEqual,
                            spv_target, scalar, zero);
                    }
                    if (source->is_float() && target->is_int()) {
                        return _builder.createUnaryOp(spv::Op::OpConvertFToS, spv_target, scalar);
                    }
                    if (source->is_float() && target->is_uint()) {
                        return _builder.createUnaryOp(spv::Op::OpConvertFToU, spv_target, scalar);
                    }
                    if (source->is_int() && target->is_float()) {
                        return _builder.createUnaryOp(spv::Op::OpConvertSToF, spv_target, scalar);
                    }
                    if (source->is_uint() && target->is_float()) {
                        return _builder.createUnaryOp(spv::Op::OpConvertUToF, spv_target, scalar);
                    }
                    if (source->is_float() && target->is_float()) {
                        return _builder.createUnaryOp(spv::Op::OpFConvert, spv_target, scalar);
                    }
                    if ((source->is_int() || source->is_uint()) &&
                        (target->is_int() || target->is_uint())) {
                        if (source->size() == target->size()) {
                            return _builder.createUnaryOp(spv::Op::OpBitcast,
                                                          spv_target, scalar);
                        }
                        if (source->is_int() == target->is_int()) {
                            return _builder.createUnaryOp(
                                source->is_int() ? spv::Op::OpSConvert :
                                                   spv::Op::OpUConvert,
                                spv_target, scalar);
                        }
                        auto temporary_type = _builder.makeIntegerType(
                            static_cast<int32_t>(target->size() * 8),
                            source->is_int());
                        auto converted = _builder.createUnaryOp(
                            source->is_int() ? spv::Op::OpSConvert :
                                               spv::Op::OpUConvert,
                            temporary_type, scalar);
                        return _builder.createUnaryOp(spv::Op::OpBitcast,
                                                      spv_target, converted);
                    }
                    LUISA_NOT_IMPLEMENTED("SPIR-V static cast from {} to {}.",
                                          source->description(), target->description());
                };
                if (from->is_scalar() && to->is_scalar()) {
                    id = static_cast_scalar(val, from, to);
                } else if (from->is_vector() && to->is_vector() &&
                           from->dimension() == to->dimension()) {
                    auto source_element = from->element();
                    auto target_element = to->element();
                    auto spv_source_element = _convert_type(source_element, Usage::READ);
                    std::vector<spv::Id> elements;
                    elements.reserve(from->dimension());
                    for (auto i = 0u; i < from->dimension(); i++) {
                        auto source_value = _builder.createCompositeExtract(
                            val, spv_source_element, i);
                        elements.emplace_back(static_cast_scalar(
                            source_value, source_element, target_element));
                    }
                    id = _builder.createCompositeConstruct(spv_to, elements);
                } else {
                    LUISA_NOT_IMPLEMENTED("SPIR-V static cast from {} to {}.",
                                          from->description(), to->description());
                }
            }
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::IF: _emit_if_inst(static_cast<const xir::IfInst *>(inst)); break;
        case xir::DerivedInstructionTag::LOOP: _emit_loop_inst(static_cast<const xir::LoopInst *>(inst)); break;
        case xir::DerivedInstructionTag::SIMPLE_LOOP: _emit_simple_loop_inst(static_cast<const xir::SimpleLoopInst *>(inst)); break;
        case xir::DerivedInstructionTag::SWITCH: _emit_switch_inst(static_cast<const xir::SwitchInst *>(inst)); break;
        case xir::DerivedInstructionTag::INDEXED_BRANCH:
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V codegen received raw IndexedBranchInst after dialect "
                "validation; run restructure_cfg before codegen.");
        case xir::DerivedInstructionTag::BRANCH: _emit_branch_inst(static_cast<const xir::BranchInst *>(inst)); break;
        case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: _emit_conditional_branch_inst(static_cast<const xir::ConditionalBranchInst *>(inst)); break;
        case xir::DerivedInstructionTag::BREAK: {
            auto br = static_cast<const xir::BreakInst *>(inst);
            auto tgt = br->target_block();
            auto spv_tgt = _resolve_branch_target(tgt);
            _builder.createBranch(false, spv_tgt);
            break;
        }
        case xir::DerivedInstructionTag::CONTINUE: {
            auto cont = static_cast<const xir::ContinueInst *>(inst);
            auto target = cont->target_block();
            _builder.createBranch(false, _resolve_branch_target(target));
            break;
        }
        case xir::DerivedInstructionTag::RETURN: {
            auto ret = static_cast<const xir::ReturnInst *>(inst);
            if (ret->return_value()) {
                _builder.makeReturn(false, _emit_value(ret->return_value()));
            } else {
                _builder.makeReturn(false);
            }
            break;
        }
        case xir::DerivedInstructionTag::UNREACHABLE:
            _builder.createNoResultOp(spv::Op::OpUnreachable);
            break;
        case xir::DerivedInstructionTag::ATOMIC: _emit_atomic_inst(static_cast<const xir::AtomicInst *>(inst)); break;
        case xir::DerivedInstructionTag::RESOURCE_QUERY: _emit_resource_query_inst(static_cast<const xir::ResourceQueryInst *>(inst)); break;
        case xir::DerivedInstructionTag::RESOURCE_READ: _emit_resource_read_inst(static_cast<const xir::ResourceReadInst *>(inst)); break;
        case xir::DerivedInstructionTag::RESOURCE_WRITE: _emit_resource_write_inst(static_cast<const xir::ResourceWriteInst *>(inst)); break;
        case xir::DerivedInstructionTag::THREAD_GROUP: _emit_thread_group_inst(static_cast<const xir::ThreadGroupInst *>(inst)); break;
        case xir::DerivedInstructionTag::PHI:
            _emit_phi_inst(static_cast<const xir::PhiInst *>(inst));
            break;
        case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
            _emit_ray_query_loop_inst(static_cast<const xir::RayQueryLoopInst *>(inst));
            break;
        case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
            _emit_ray_query_dispatch_inst(static_cast<const xir::RayQueryDispatchInst *>(inst));
            break;
        case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
        case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC:
            LUISA_ERROR_WITH_LOCATION("Instruction {} should be eliminated before SPIR-V codegen.",
                                      xir::to_string(inst->derived_instruction_tag()));
        case xir::DerivedInstructionTag::PRINT:
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V codegen received PRINT after dialect validation; "
                "silently dropping its side effect is forbidden.");
        case xir::DerivedInstructionTag::CLOCK: {
            _require_target_feature(
                target_feature::shader_device_clock,
                _target_features.shader_device_clock);
            _builder.addExtension(spv::E_SPV_KHR_shader_clock);
            _builder.addCapability(spv::Capability::ShaderClockKHR);
            auto scope = _builder.makeUintConstant(
                static_cast<uint32_t>(spv::Scope::Device));
            set_result(_builder.createOp(
                spv::Op::OpReadClockKHR,
                _convert_type(inst->type(), Usage::READ),
                std::vector<spv::Id>{scope}));
            break;
        }
        case xir::DerivedInstructionTag::ASSUME:
            // XIR assumptions are optimization-only hints. A false condition
            // already gives the shader undefined behavior, so emitting no
            // instruction preserves every defined result without requiring
            // SPV_KHR_expect_assume support from the Vulkan device.
            break;
        case xir::DerivedInstructionTag::ASSERT:
            if (!_enable_debug_info) {
                // Match the release-mode contract of the CUDA, HIP, Metal,
                // and HLSL codegens: device assertions are disabled when
                // debug information is not requested.
                break;
            }
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V codegen received a debug assertion after dialect "
                "validation; native Vulkan has no device-side assertion "
                "reporting contract.");
        case xir::DerivedInstructionTag::DEBUG_BREAK:
        case xir::DerivedInstructionTag::OUTLINE:
        case xir::DerivedInstructionTag::RASTER_DISCARD:
            LUISA_NOT_IMPLEMENTED("SPIR-V codegen for instruction {}.", xir::to_string(inst->derived_instruction_tag()));
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            _emit_ray_query_object_read_inst(static_cast<const xir::RayQueryObjectReadInst *>(inst));
            break;
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            _emit_ray_query_object_write_inst(static_cast<const xir::RayQueryObjectWriteInst *>(inst));
            break;
        case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
            LUISA_NOT_IMPLEMENTED("SPIR-V codegen for instruction {}.", xir::to_string(inst->derived_instruction_tag()));
    }
}

void SpirvCodegenEntry::_emit_ray_query_loop_inst(const xir::RayQueryLoopInst *) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "SPIR-V codegen received RayQueryLoopInst after control-flow planning. "
        "lower_ray_query_loop_to_loop must run before codegen.");
}

void SpirvCodegenEntry::_emit_ray_query_dispatch_inst(const xir::RayQueryDispatchInst *) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "SPIR-V codegen received RayQueryDispatchInst after control-flow planning. "
        "lower_ray_query_loop_to_loop must run before codegen.");
}

void SpirvCodegenEntry::_emit_ray_query_object_read_inst(const xir::RayQueryObjectReadInst *inst) noexcept {
    auto rq_obj = _emit_value(inst->operand(0));
    auto type = _convert_type(inst->type(), Usage::READ);
    auto float_type = _builder.makeFloatType(32);
    auto uint_type = _builder.makeUintType(32);
    auto int_type = _builder.makeIntType(32);
    auto bool_type = _builder.makeBoolType();
    auto vec2_type = _builder.makeVectorType(float_type, 2);
    auto vec3_type = _builder.makeVectorType(float_type, 3);
    spv::Id id = spv::NoResult;
    switch (inst->op()) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY: {
            auto &state = _ray_query_state(rq_obj);
            auto origin = _builder.createOp(spv::Op::OpRayQueryGetWorldRayOriginKHR, vec3_type, std::vector<spv::Id>{rq_obj});
            auto dir = _builder.createOp(spv::Op::OpRayQueryGetWorldRayDirectionKHR, vec3_type, std::vector<spv::Id>{rq_obj});
            auto t_min = _builder.createOp(spv::Op::OpRayQueryGetRayTMinKHR, float_type, std::vector<spv::Id>{rq_obj});
            auto committed = _builder.makeIntConstant(1);// RayQueryCommittedIntersectionKHR
            auto committed_type = _builder.createOp(
                spv::Op::OpRayQueryGetIntersectionTypeKHR, uint_type,
                std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
            auto has_committed = _builder.createBinOp(
                spv::Op::OpINotEqual, bool_type, committed_type,
                _builder.makeUintConstant(0u));
            auto initial_t_max = _builder.createCompositeExtract(
                state.initial_ray, float_type, 3u);
            auto t_max_var = _builder.createVariable(
                spv::NoPrecision, spv::StorageClass::Function,
                float_type, "rq_world_ray_tmax");
            _builder.createStore(initial_t_max, t_max_var);

            // SPV_KHR_ray_query makes committed IntersectionT undefined while
            // the committed type is None. Keep it out of that dynamic path;
            // OpSelect would still evaluate the invalid instruction eagerly.
            auto *committed_block = _create_physical_block();
            auto *merge_block = _create_physical_block();
            auto selection_merge = std::make_unique<spv::Instruction>(
                spv::Op::OpSelectionMerge);
            selection_merge->reserveOperands(2u);
            selection_merge->addIdOperand(merge_block->getId());
            selection_merge->addImmediateOperand(
                spv::SelectionControlMask::MaskNone);
            _builder.getBuildPoint()->addInstruction(std::move(selection_merge));
            _builder.createConditionalBranch(
                has_committed, committed_block, merge_block);

            _set_current_tail(committed_block);
            auto committed_t_max = _builder.createOp(
                spv::Op::OpRayQueryGetIntersectionTKHR, float_type,
                std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
            _builder.createStore(committed_t_max, t_max_var);
            _builder.createBranch(false, merge_block);

            _set_current_tail(merge_block);
            auto t_max = _builder.createLoad(t_max_var, spv::NoPrecision);
            auto ray_type = inst->type();
            auto origin_array_type = _convert_type(ray_type->members()[0], Usage::READ);
            auto dir_array_type = _convert_type(ray_type->members()[2], Usage::READ);
            auto origin_arr = _builder.createCompositeConstruct(origin_array_type, {_builder.createCompositeExtract(origin, float_type, 0),
                                                                                    _builder.createCompositeExtract(origin, float_type, 1),
                                                                                    _builder.createCompositeExtract(origin, float_type, 2)});
            auto dir_arr = _builder.createCompositeConstruct(dir_array_type, {_builder.createCompositeExtract(dir, float_type, 0),
                                                                              _builder.createCompositeExtract(dir, float_type, 1),
                                                                              _builder.createCompositeExtract(dir, float_type, 2)});
            id = _builder.createCompositeConstruct(type, {origin_arr, t_min, dir_arr, t_max});
            break;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY: {
            auto &state = _ray_query_state(rq_obj);
            auto candidate = _builder.makeIntConstant(0);// RayQueryCandidateIntersectionKHR
            auto origin = _builder.createOp(
                spv::Op::OpRayQueryGetIntersectionObjectRayOriginKHR,
                vec3_type,
                std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            auto dir = _builder.createOp(
                spv::Op::OpRayQueryGetIntersectionObjectRayDirectionKHR,
                vec3_type,
                std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            auto t_min = _builder.createOp(
                spv::Op::OpRayQueryGetRayTMinKHR, float_type,
                std::vector<spv::Id>{rq_obj});
            auto committed = _builder.makeIntConstant(1);// RayQueryCommittedIntersectionKHR
            auto committed_type = _builder.createOp(
                spv::Op::OpRayQueryGetIntersectionTypeKHR, uint_type,
                std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
            auto has_committed = _builder.createBinOp(
                spv::Op::OpINotEqual, bool_type, committed_type,
                _builder.makeUintConstant(0u));
            auto initial_t_max = _builder.createCompositeExtract(
                state.initial_ray, float_type, 3u);
            auto t_max_var = _builder.createVariable(
                spv::NoPrecision, spv::StorageClass::Function,
                float_type, "rq_object_ray_tmax");
            _builder.createStore(initial_t_max, t_max_var);

            // IntersectionT is undefined while the committed type is None.
            // Branch before reading it; OpSelect would evaluate both arms.
            auto *committed_block = _create_physical_block();
            auto *merge_block = _create_physical_block();
            auto selection_merge = std::make_unique<spv::Instruction>(
                spv::Op::OpSelectionMerge);
            selection_merge->reserveOperands(2u);
            selection_merge->addIdOperand(merge_block->getId());
            selection_merge->addImmediateOperand(
                spv::SelectionControlMask::MaskNone);
            _builder.getBuildPoint()->addInstruction(
                std::move(selection_merge));
            _builder.createConditionalBranch(
                has_committed, committed_block, merge_block);

            _set_current_tail(committed_block);
            auto committed_t_max = _builder.createOp(
                spv::Op::OpRayQueryGetIntersectionTKHR, float_type,
                std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
            _builder.createStore(committed_t_max, t_max_var);
            _builder.createBranch(false, merge_block);

            _set_current_tail(merge_block);
            auto t_max = _builder.createLoad(t_max_var, spv::NoPrecision);
            auto ray_type = inst->type();
            auto origin_array_type = _convert_type(
                ray_type->members()[0], Usage::READ);
            auto dir_array_type = _convert_type(
                ray_type->members()[2], Usage::READ);
            auto origin_arr = _builder.createCompositeConstruct(
                origin_array_type,
                {_builder.createCompositeExtract(origin, float_type, 0),
                 _builder.createCompositeExtract(origin, float_type, 1),
                 _builder.createCompositeExtract(origin, float_type, 2)});
            auto dir_arr = _builder.createCompositeConstruct(
                dir_array_type,
                {_builder.createCompositeExtract(dir, float_type, 0),
                 _builder.createCompositeExtract(dir, float_type, 1),
                 _builder.createCompositeExtract(dir, float_type, 2)});
            id = _builder.createCompositeConstruct(
                type, {origin_arr, t_min, dir_arr, t_max});
            break;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: {
            auto candidate = _builder.makeIntConstant(0);// RayQueryCandidateIntersectionKHR
            auto inst_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionInstanceIdKHR, uint_type,
                                              std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            auto prim_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR, uint_type,
                                              std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            auto bary = _builder.createOp(spv::Op::OpRayQueryGetIntersectionBarycentricsKHR, vec2_type,
                                          std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            auto ray_t = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTKHR, float_type,
                                           std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            id = _builder.createCompositeConstruct(type, {inst_idx, prim_idx, bary, ray_t});
            break;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: {
            auto candidate = _builder.makeIntConstant(0);// RayQueryCandidateIntersectionKHR
            auto inst_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionInstanceIdKHR, uint_type,
                                              std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            auto prim_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR, uint_type,
                                              std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            id = _builder.createCompositeConstruct(type, {inst_idx, prim_idx});
            break;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT: {
            auto committed = _builder.makeIntConstant(1);// RayQueryCommittedIntersectionKHR
            auto committed_type = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTypeKHR, uint_type,
                                                    std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
            auto *tri_block = _create_physical_block();
            auto *proc_block = _create_physical_block();
            auto *merge_block = _create_physical_block();
            auto result_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, type, "committed_hit");
            auto zero_result = _builder.makeNullConstant(type);
            _builder.createStore(zero_result, result_var);
            auto selection_merge = std::make_unique<spv::Instruction>(spv::Op::OpSelectionMerge);
            selection_merge->reserveOperands(2);
            selection_merge->addIdOperand(merge_block->getId());
            selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
            _builder.getBuildPoint()->addInstruction(std::move(selection_merge));
            auto switch_inst = std::make_unique<spv::Instruction>(spv::Op::OpSwitch);
            switch_inst->reserveOperands(6);
            switch_inst->addIdOperand(committed_type);
            switch_inst->addIdOperand(merge_block->getId());// default (none) -> merge
            switch_inst->addImmediateOperand(1u);           // triangle
            switch_inst->addIdOperand(tri_block->getId());
            switch_inst->addImmediateOperand(2u);// procedural
            switch_inst->addIdOperand(proc_block->getId());
            _builder.getBuildPoint()->addInstruction(std::move(switch_inst));
            merge_block->addPredecessor(_builder.getBuildPoint());
            tri_block->addPredecessor(_builder.getBuildPoint());
            proc_block->addPredecessor(_builder.getBuildPoint());
            _set_current_tail(tri_block);
            {
                auto inst_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionInstanceIdKHR, uint_type,
                                                  std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
                auto prim_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR, uint_type,
                                                  std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
                auto bary = _builder.createOp(spv::Op::OpRayQueryGetIntersectionBarycentricsKHR, vec2_type,
                                              std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
                auto ray_t = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTKHR, float_type,
                                               std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
                auto hit = _builder.createCompositeConstruct(type, {inst_idx, prim_idx, bary, _builder.makeUintConstant(1u), ray_t});
                _builder.createStore(hit, result_var);
            }
            _builder.createBranch(false, merge_block);
            _set_current_tail(proc_block);
            {
                auto inst_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionInstanceIdKHR, uint_type,
                                                  std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
                auto prim_idx = _builder.createOp(spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR, uint_type,
                                                  std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
                auto ray_t = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTKHR, float_type,
                                               std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
                auto zero_bary = _builder.createCompositeConstruct(vec2_type, {_builder.makeFloatConstant(0.0f), _builder.makeFloatConstant(0.0f)});
                auto hit = _builder.createCompositeConstruct(type, {inst_idx, prim_idx, zero_bary, _builder.makeUintConstant(2u), ray_t});
                _builder.createStore(hit, result_var);
            }
            _builder.createBranch(false, merge_block);
            _set_current_tail(merge_block);
            id = _builder.createLoad(result_var, spv::NoPrecision);
            break;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE: {
            auto candidate = _builder.makeIntConstant(0);// RayQueryCandidateIntersectionKHR
            auto candidate_type = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTypeKHR, uint_type,
                                                    std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            id = _builder.createBinOp(spv::Op::OpIEqual, bool_type, candidate_type, _builder.makeUintConstant(0u));
            break;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE: {
            auto candidate = _builder.makeIntConstant(0);// RayQueryCandidateIntersectionKHR
            auto candidate_type = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTypeKHR, uint_type,
                                                    std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate}});
            id = _builder.createBinOp(spv::Op::OpIEqual, bool_type, candidate_type, _builder.makeUintConstant(1u));
            break;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED: {
            auto &state = _ray_query_state(rq_obj);
            auto proceed_state = _builder.createLoad(
                state.proceed_state, spv::NoPrecision);
            id = _builder.createUnaryOp(
                spv::Op::OpLogicalNot, bool_type, proceed_state);
            break;
        }
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit ray query object read.");
    _value_map.emplace(inst, id);
}

void SpirvCodegenEntry::_emit_ray_query_object_write_inst(const xir::RayQueryObjectWriteInst *inst) noexcept {
    auto rq_obj = _emit_value(inst->operand(0));
    switch (inst->op()) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE:
            _builder.createNoResultOp(spv::Op::OpRayQueryConfirmIntersectionKHR, {rq_obj});
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == 2);
            auto dist = _emit_value(inst->operand(1));
            _builder.createNoResultOp(spv::Op::OpRayQueryGenerateIntersectionKHR, {rq_obj, dist});
            break;
        }
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            _builder.createNoResultOp(spv::Op::OpRayQueryTerminateKHR, {rq_obj});
            _builder.createStore(
                _builder.makeBoolConstant(false),
                _ray_query_state(rq_obj).proceed_state);
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED: {
            auto proceed_result = _builder.createOp(spv::Op::OpRayQueryProceedKHR, _builder.makeBoolType(), std::vector<spv::Id>{rq_obj});
            _builder.createStore(
                proceed_result, _ray_query_state(rq_obj).proceed_state);
            break;
        }
    }
}

spv::Id SpirvCodegenEntry::_ensure_type(spv::Id value, spv::Id target_type) noexcept {
    auto value_type = _builder.getTypeId(value);
    if (value_type == target_type) { return value; }
    auto value_is_vector = _builder.isVectorType(value_type);
    auto target_is_vector = _builder.isVectorType(target_type);
    LUISA_ASSERT(
        value_is_vector == target_is_vector &&
            (!value_is_vector ||
             _builder.getNumTypeComponents(value_type) ==
                 _builder.getNumTypeComponents(target_type)),
        "SPIR-V numeric type conversion requires equal scalar/vector shapes.");
    auto val_scalar = _builder.getScalarTypeId(value_type);
    auto tgt_scalar = _builder.getScalarTypeId(target_type);
    auto val_class = _builder.getTypeClass(val_scalar);
    auto tgt_class = _builder.getTypeClass(tgt_scalar);
    if (val_class == tgt_class) {
        if (val_class == spv::Op::OpTypeFloat) {
            return _builder.createUnaryOp(spv::Op::OpFConvert, target_type, value);
        }
        if (val_class == spv::Op::OpTypeInt) {
            auto val_signed = _builder.isIntType(val_scalar);
            auto tgt_signed = _builder.isIntType(tgt_scalar);
            auto val_width = _builder.getScalarTypeWidth(val_scalar);
            auto tgt_width = _builder.getScalarTypeWidth(tgt_scalar);
            if (val_width == tgt_width && val_signed == tgt_signed) {
                return value;// No conversion needed
            }
            if (val_signed == tgt_signed) {
                return _builder.createUnaryOp(val_signed ? spv::Op::OpSConvert : spv::Op::OpUConvert, target_type, value);
            }
            if (val_width == tgt_width) {
                return _builder.createUnaryOp(spv::Op::OpBitcast, target_type, value);
            }
            // Cross-signedness with different sizes
            auto tmp_type = _builder.makeIntegerType(tgt_width, val_signed);
            value = _builder.createUnaryOp(val_signed ? spv::Op::OpSConvert : spv::Op::OpUConvert, tmp_type, value);
            return _builder.createUnaryOp(spv::Op::OpBitcast, target_type, value);
        }
    }
    if (val_class == spv::Op::OpTypeFloat && tgt_class == spv::Op::OpTypeInt) {
        return _builder.createUnaryOp(_builder.isIntType(tgt_scalar) ? spv::Op::OpConvertFToS : spv::Op::OpConvertFToU, target_type, value);
    }
    if (val_class == spv::Op::OpTypeInt && tgt_class == spv::Op::OpTypeFloat) {
        return _builder.createUnaryOp(_builder.isIntType(val_scalar) ? spv::Op::OpConvertSToF : spv::Op::OpConvertUToF, target_type, value);
    }
    LUISA_ERROR_WITH_LOCATION(
        "Unsupported implicit SPIR-V numeric conversion from type ID {} to {}.",
        value_type, target_type);
}

}// namespace lc::spirv
