#include "entry.h"
#include <luisa/core/logging.h>
#include <SPIRV/GLSL.std.450.h>

namespace lc::spirv {

void SpirvCodegenEntry::_emit_arithmetic_inst(const xir::ArithmeticInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    auto t = inst->type();
    auto elem = t->is_vector() || t->is_matrix() ? t->element() : t;
    auto is_float = elem->is_float();
    auto is_signed_int = elem->is_int();
    auto is_bool = elem->is_bool();
    auto is_scalar = t->is_scalar();

    auto operand = [&](size_t i) noexcept { return _emit_value(inst->operand(i)); };
    spv::Id id = spv::NoResult;

    auto glsl = [&](int builtin, auto... args) noexcept -> spv::Id {
        std::vector<spv::Id> ops = {args...};
        return _builder.createBuiltinCall(type, _glsl450, builtin, ops);
    };

    auto glsl_typed = [&](int f_builtin, int s_builtin, int u_builtin, auto... args) noexcept -> spv::Id {
        std::vector<spv::Id> ops = {args...};
        int builtin = f_builtin;
        if (is_signed_int)
            builtin = s_builtin;
        else if (is_bool || elem->is_uint())
            builtin = u_builtin;
        return _builder.createBuiltinCall(type, _glsl450, builtin, ops);
    };

    auto unary = [&](spv::Op op) noexcept -> spv::Id {
        return _builder.createUnaryOp(op, type, operand(0));
    };
    auto binary = [&](spv::Op op) noexcept -> spv::Id {
        return _builder.createBinOp(op, type, operand(0), operand(1));
    };

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
        case xir::ArithmeticOp::BINARY_ADD:
            if (is_float)
                id = binary(spv::Op::OpFAdd);
            else
                id = binary(spv::Op::OpIAdd);
            break;
        case xir::ArithmeticOp::BINARY_SUB:
            if (is_float)
                id = binary(spv::Op::OpFSub);
            else
                id = binary(spv::Op::OpISub);
            break;
        case xir::ArithmeticOp::BINARY_MUL:
            if (is_float)
                id = binary(spv::Op::OpFMul);
            else
                id = binary(spv::Op::OpIMul);
            break;
        case xir::ArithmeticOp::BINARY_DIV:
            if (is_float)
                id = binary(spv::Op::OpFDiv);
            else if (is_signed_int)
                id = binary(spv::Op::OpSDiv);
            else
                id = binary(spv::Op::OpUDiv);
            break;
        case xir::ArithmeticOp::BINARY_MOD:
            if (is_float)
                id = binary(spv::Op::OpFMod);
            else if (is_signed_int)
                id = binary(spv::Op::OpSMod);
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
            auto width_id = _builder.makeUintConstant(static_cast<unsigned>(width));
            auto b_mod = _builder.createBinOp(spv::Op::OpUMod, _builder.makeUintType(32), b, width_id);
            auto left = _builder.createBinOp(spv::Op::OpShiftLeftLogical, type, a, b_mod);
            auto right = _builder.createBinOp(spv::Op::OpShiftRightLogical, type, a,
                                              _builder.createBinOp(spv::Op::OpISub, _builder.makeUintType(32), width_id, b_mod));
            id = _builder.createBinOp(spv::Op::OpBitwiseOr, type, left, right);
            break;
        }
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT: {
            auto a = operand(0);
            auto b = operand(1);
            auto width = t->is_scalar() ? t->size() * 8 : t->element()->size() * 8;
            auto width_id = _builder.makeUintConstant(static_cast<unsigned>(width));
            auto b_mod = _builder.createBinOp(spv::Op::OpUMod, _builder.makeUintType(32), b, width_id);
            auto right = _builder.createBinOp(spv::Op::OpShiftRightLogical, type, a, b_mod);
            auto left = _builder.createBinOp(spv::Op::OpShiftLeftLogical, type, a,
                                             _builder.createBinOp(spv::Op::OpISub, _builder.makeUintType(32), width_id, b_mod));
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
            else
                id = binary(spv::Op::OpIEqual);
            break;
        }
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: {
            auto op_elem = inst->operand(0)->type();
            op_elem = op_elem->is_vector() ? op_elem->element() : op_elem;
            if (op_elem->is_float())
                id = binary(spv::Op::OpFOrdNotEqual);
            else
                id = binary(spv::Op::OpINotEqual);
            break;
        }
        case xir::ArithmeticOp::SELECT: {
            // XIR SELECT operands are (false_value, true_value, condition)
            auto cond = operand(2);
            auto cond_type = _builder.getTypeId(cond);
            bool is_bool_cond = _builder.isBoolType(cond_type);
            if (_builder.isVectorType(cond_type) && _builder.isBoolType(_builder.getContainedTypeId(cond_type))) {
                is_bool_cond = true;
            }
            if (!is_bool_cond) {
                spv::Id zero = spv::NoResult;
                spv::Id bool_type = _builder.makeBoolType();
                if (_builder.isIntType(cond_type) || _builder.isUintType(cond_type)) {
                    zero = _builder.makeIntConstant(0);
                } else if (_builder.isFloatType(cond_type)) {
                    zero = _builder.makeFloatConstant(0.0f);
                }
                if (zero != spv::NoResult) {
                    if (_builder.isVectorType(cond_type)) {
                        auto dim = static_cast<int>(_builder.getNumTypeComponents(cond_type));
                        zero = _builder.smearScalar(spv::NoPrecision, zero, cond_type);
                        bool_type = _builder.makeVectorType(bool_type, dim);
                    }
                    cond = _builder.createBinOp(spv::Op::OpINotEqual, bool_type, cond, zero);
                }
            }
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
            id = glsl_typed(GLSLstd450FClamp, GLSLstd450SClamp, GLSLstd450UClamp,
                            operand(0), operand(1), operand(2));
            break;
        case xir::ArithmeticOp::SATURATE:
            if (is_float) {
                auto zero = _builder.makeFloatConstant(0.0f);
                auto one = _builder.makeFloatConstant(1.0f);
                if (!is_scalar) {
                    auto scalar_type = _convert_type(t->element(), Usage::READ);
                    zero = _builder.smearScalar(spv::NoPrecision, zero, type);
                    one = _builder.smearScalar(spv::NoPrecision, one, type);
                }
                id = glsl(GLSLstd450FClamp, operand(0), zero, one);
            } else {
                LUISA_NOT_IMPLEMENTED("SPIR-V saturate for integer types.");
            }
            break;
        case xir::ArithmeticOp::LERP:
            id = glsl(GLSLstd450FMix, operand(0), operand(1), operand(2));
            break;
        case xir::ArithmeticOp::STEP:
            id = glsl(GLSLstd450Step, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::SMOOTHSTEP:
            id = glsl(GLSLstd450SmoothStep, operand(0), operand(1), operand(2));
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
            id = glsl_typed(GLSLstd450FMin, GLSLstd450SMin, GLSLstd450UMin,
                            operand(0), operand(1));
            break;
        case xir::ArithmeticOp::MAX:
            id = glsl_typed(GLSLstd450FMax, GLSLstd450SMax, GLSLstd450UMax,
                            operand(0), operand(1));
            break;
        case xir::ArithmeticOp::CLZ: {
            auto find_msb = glsl_typed(GLSLstd450FindSMsb, GLSLstd450FindSMsb, GLSLstd450FindUMsb, operand(0));
            auto bit_width = static_cast<int>(t->is_scalar() ? t->size() * 8 : t->element()->size() * 8);
            auto bit_width_id = elem->is_uint() ? _builder.makeUintConstant(bit_width) : _builder.makeIntConstant(bit_width);
            auto minus_one = elem->is_uint() ? _builder.makeUintConstant(0xFFFFFFFFu) : _builder.makeIntConstant(-1);
            if (!is_scalar) {
                bit_width_id = _builder.smearScalar(spv::NoPrecision, bit_width_id, type);
                minus_one = _builder.smearScalar(spv::NoPrecision, minus_one, type);
            }
            auto is_zero = _builder.createBinOp(spv::Op::OpIEqual, _builder.makeBoolType(), find_msb, minus_one);
            auto one = elem->is_uint() ? _builder.makeUintConstant(1u) : _builder.makeIntConstant(1);
            auto clz_val = _builder.createBinOp(spv::Op::OpISub, type, _builder.createBinOp(spv::Op::OpISub, type, bit_width_id, one), find_msb);
            id = _builder.createTriOp(spv::Op::OpSelect, type, is_zero, bit_width_id, clz_val);
            break;
        }
        case xir::ArithmeticOp::CTZ: {
            auto find_lsb = glsl(GLSLstd450FindILsb, operand(0));
            auto bit_width = static_cast<int>(t->is_scalar() ? t->size() * 8 : t->element()->size() * 8);
            auto bit_width_id = elem->is_uint() ? _builder.makeUintConstant(bit_width) : _builder.makeIntConstant(bit_width);
            auto minus_one = elem->is_uint() ? _builder.makeUintConstant(0xFFFFFFFFu) : _builder.makeIntConstant(-1);
            if (!is_scalar) {
                bit_width_id = _builder.smearScalar(spv::NoPrecision, bit_width_id, type);
                minus_one = _builder.smearScalar(spv::NoPrecision, minus_one, type);
            }
            auto is_zero = _builder.createBinOp(spv::Op::OpIEqual, _builder.makeBoolType(), find_lsb, minus_one);
            id = _builder.createTriOp(spv::Op::OpSelect, type, is_zero, bit_width_id, find_lsb);
            break;
        }
        case xir::ArithmeticOp::POPCOUNT:
            id = unary(spv::Op::OpBitCount);
            break;
        case xir::ArithmeticOp::REVERSE:
            id = unary(spv::Op::OpBitReverse);
            break;
        case xir::ArithmeticOp::ISINF:
            id = unary(spv::Op::OpIsInf);
            break;
        case xir::ArithmeticOp::ISNAN:
            id = unary(spv::Op::OpIsNan);
            break;
        case xir::ArithmeticOp::ACOS:
            id = glsl(GLSLstd450Acos, operand(0));
            break;
        case xir::ArithmeticOp::ACOSH:
            id = glsl(GLSLstd450Acosh, operand(0));
            break;
        case xir::ArithmeticOp::ASIN:
            id = glsl(GLSLstd450Asin, operand(0));
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
            auto log2_10 = _builder.makeFloatConstant(3.321928094887362f);
            if (!is_scalar) log2_10 = _builder.smearScalar(spv::NoPrecision, log2_10, type);
            auto scaled = _builder.createBinOp(spv::Op::OpFMul, type, operand(0), log2_10);
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
            auto inv_log2_10 = _builder.makeFloatConstant(0.3010299956639812f);
            if (!is_scalar) inv_log2_10 = _builder.smearScalar(spv::NoPrecision, inv_log2_10, type);
            auto log2_val = glsl(GLSLstd450Log2, operand(0));
            id = _builder.createBinOp(spv::Op::OpFMul, type, log2_val, inv_log2_10);
            break;
        }
        case xir::ArithmeticOp::POW:
            id = glsl(GLSLstd450Pow, operand(0), operand(1));
            break;
        case xir::ArithmeticOp::POW_INT: {
            auto exp_float = _builder.createUnaryOp(spv::Op::OpConvertSToF, type, operand(1));
            id = glsl(GLSLstd450Pow, operand(0), exp_float);
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
        case xir::ArithmeticOp::ROUND:
            id = glsl(GLSLstd450Round, operand(0));
            break;
        case xir::ArithmeticOp::RINT:
            id = glsl(GLSLstd450RoundEven, operand(0));
            break;
        case xir::ArithmeticOp::FMA:
            id = glsl(GLSLstd450Fma, operand(0), operand(1), operand(2));
            break;
        case xir::ArithmeticOp::COPYSIGN: {
            auto a = operand(0);
            auto b = operand(1);
            auto abs_a = glsl(GLSLstd450FAbs, a);
            auto sign_b = glsl(GLSLstd450FSign, b);
            id = _builder.createBinOp(spv::Op::OpFMul, type, abs_a, sign_b);
            break;
        }
        case xir::ArithmeticOp::CROSS: {
            auto u = operand(0);
            auto v = operand(1);
            auto elem_type = _convert_type(t->element(), Usage::READ);
            auto ux = _builder.createCompositeExtract(u, elem_type, 0);
            auto uy = _builder.createCompositeExtract(u, elem_type, 1);
            auto uz = _builder.createCompositeExtract(u, elem_type, 2);
            auto vx = _builder.createCompositeExtract(v, elem_type, 0);
            auto vy = _builder.createCompositeExtract(v, elem_type, 1);
            auto vz = _builder.createCompositeExtract(v, elem_type, 2);
            auto rx = _builder.createBinOp(spv::Op::OpFMul, elem_type, uy, vz);
            auto ry = _builder.createBinOp(spv::Op::OpFMul, elem_type, uz, vx);
            auto rz = _builder.createBinOp(spv::Op::OpFMul, elem_type, ux, vy);
            auto lx = _builder.createBinOp(spv::Op::OpFMul, elem_type, vy, uz);
            auto ly = _builder.createBinOp(spv::Op::OpFMul, elem_type, vz, ux);
            auto lz = _builder.createBinOp(spv::Op::OpFMul, elem_type, vx, uy);
            auto cx = _builder.createBinOp(spv::Op::OpFSub, elem_type, rx, lx);
            auto cy = _builder.createBinOp(spv::Op::OpFSub, elem_type, ry, ly);
            auto cz = _builder.createBinOp(spv::Op::OpFSub, elem_type, rz, lz);
            id = _builder.createCompositeConstruct(type, {cx, cy, cz});
            break;
        }
        case xir::ArithmeticOp::DOT:
            id = binary(spv::Op::OpDot);
            break;
        case xir::ArithmeticOp::LENGTH:
            id = glsl(GLSLstd450Length, operand(0));
            break;
        case xir::ArithmeticOp::LENGTH_SQUARED: {
            auto a = operand(0);
            id = _builder.createBinOp(spv::Op::OpDot, type, a, a);
            break;
        }
        case xir::ArithmeticOp::NORMALIZE:
            id = glsl(GLSLstd450Normalize, operand(0));
            break;
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
            auto extract = [&](unsigned i) {
                return _builder.createCompositeExtract(v, elem_spv_type, i);
            };
            id = extract(0);
            for (auto i = 1u; i < dim; ++i) {
                auto comp = extract(i);
                switch (inst->op()) {
                    case xir::ArithmeticOp::REDUCE_SUM:
                        if (elem_type->is_float())
                            id = _builder.createBinOp(spv::Op::OpFAdd, elem_spv_type, id, comp);
                        else
                            id = _builder.createBinOp(spv::Op::OpIAdd, elem_spv_type, id, comp);
                        break;
                    case xir::ArithmeticOp::REDUCE_PRODUCT:
                        if (elem_type->is_float())
                            id = _builder.createBinOp(spv::Op::OpFMul, elem_spv_type, id, comp);
                        else
                            id = _builder.createBinOp(spv::Op::OpIMul, elem_spv_type, id, comp);
                        break;
                    case xir::ArithmeticOp::REDUCE_MIN:
                        if (elem_type->is_float())
                            id = _builder.createBuiltinCall(elem_spv_type, _glsl450, GLSLstd450FMin, {id, comp});
                        else if (elem_type->is_int())
                            id = _builder.createBuiltinCall(elem_spv_type, _glsl450, GLSLstd450SMin, {id, comp});
                        else
                            id = _builder.createBuiltinCall(elem_spv_type, _glsl450, GLSLstd450UMin, {id, comp});
                        break;
                    case xir::ArithmeticOp::REDUCE_MAX:
                        if (elem_type->is_float())
                            id = _builder.createBuiltinCall(elem_spv_type, _glsl450, GLSLstd450FMax, {id, comp});
                        else if (elem_type->is_int())
                            id = _builder.createBuiltinCall(elem_spv_type, _glsl450, GLSLstd450SMax, {id, comp});
                        else
                            id = _builder.createBuiltinCall(elem_spv_type, _glsl450, GLSLstd450UMax, {id, comp});
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
            auto a_dim = a_type->dimension();
            auto b_dim = b_type->dimension();
            auto elem_type = a_type->element();
            auto elem_spv_type = _convert_type(elem_type, Usage::READ);
            auto vec_type = _builder.makeVectorType(elem_spv_type, static_cast<int>(b_dim));
            std::vector<spv::Id> rows;
            rows.reserve(a_dim);
            for (auto i = 0u; i < a_dim; ++i) {
                auto ai = _builder.createCompositeExtract(a, elem_spv_type, i);
                auto smeared = _builder.smearScalar(spv::NoPrecision, ai, vec_type);
                auto row = _builder.createBinOp(spv::Op::OpFMul, vec_type, smeared, b);
                rows.push_back(row);
            }
            id = _builder.createCompositeConstruct(type, rows);
            break;
        }

        // Matrix operations
        case xir::ArithmeticOp::MATRIX_COMP_NEG: {
            auto mat = operand(0);
            auto rows = t->dimension();
            auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int>(rows));
            std::vector<spv::Id> new_rows;
            new_rows.reserve(rows);
            for (auto i = 0u; i < rows; ++i) {
                auto row = _builder.createCompositeExtract(mat, row_type, i);
                new_rows.push_back(_builder.createUnaryOp(spv::Op::OpFNegate, row_type, row));
            }
            id = _builder.createCompositeConstruct(type, new_rows);
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_ADD: {
            auto a = operand(0);
            auto b = operand(1);
            auto b_type = inst->operand(1)->type();
            auto rows = t->dimension();
            auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int>(rows));
            std::vector<spv::Id> new_rows;
            new_rows.reserve(rows);
            for (auto i = 0u; i < rows; ++i) {
                auto row_a = _builder.createCompositeExtract(a, row_type, i);
                if (b_type->is_scalar()) {
                    auto smeared = _builder.smearScalar(spv::NoPrecision, b, row_type);
                    new_rows.push_back(_builder.createBinOp(spv::Op::OpFAdd, row_type, row_a, smeared));
                } else {
                    auto row_b = _builder.createCompositeExtract(b, row_type, i);
                    new_rows.push_back(_builder.createBinOp(spv::Op::OpFAdd, row_type, row_a, row_b));
                }
            }
            id = _builder.createCompositeConstruct(type, new_rows);
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_SUB: {
            auto a = operand(0);
            auto b = operand(1);
            auto b_type = inst->operand(1)->type();
            auto rows = t->dimension();
            auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int>(rows));
            std::vector<spv::Id> new_rows;
            new_rows.reserve(rows);
            for (auto i = 0u; i < rows; ++i) {
                auto row_a = _builder.createCompositeExtract(a, row_type, i);
                if (b_type->is_scalar()) {
                    auto smeared = _builder.smearScalar(spv::NoPrecision, b, row_type);
                    new_rows.push_back(_builder.createBinOp(spv::Op::OpFSub, row_type, row_a, smeared));
                } else {
                    auto row_b = _builder.createCompositeExtract(b, row_type, i);
                    new_rows.push_back(_builder.createBinOp(spv::Op::OpFSub, row_type, row_a, row_b));
                }
            }
            id = _builder.createCompositeConstruct(type, new_rows);
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_MUL: {
            auto a = operand(0);
            auto b = operand(1);
            auto b_type = inst->operand(1)->type();
            if (b_type->is_scalar()) {
                id = _builder.createBinOp(spv::Op::OpMatrixTimesScalar, type, a, b);
            } else {
                auto rows = t->dimension();
                auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int>(rows));
                std::vector<spv::Id> new_rows;
                new_rows.reserve(rows);
                for (auto i = 0u; i < rows; ++i) {
                    auto row_a = _builder.createCompositeExtract(a, row_type, i);
                    auto row_b = _builder.createCompositeExtract(b, row_type, i);
                    new_rows.push_back(_builder.createBinOp(spv::Op::OpFMul, row_type, row_a, row_b));
                }
                id = _builder.createCompositeConstruct(type, new_rows);
            }
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_DIV: {
            auto a = operand(0);
            auto b = operand(1);
            auto b_type = inst->operand(1)->type();
            if (b_type->is_scalar()) {
                auto rows = t->dimension();
                auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int>(rows));
                std::vector<spv::Id> new_rows;
                new_rows.reserve(rows);
                for (auto i = 0u; i < rows; ++i) {
                    auto row = _builder.createCompositeExtract(a, row_type, i);
                    auto smeared = _builder.smearScalar(spv::NoPrecision, b, row_type);
                    new_rows.push_back(_builder.createBinOp(spv::Op::OpFDiv, row_type, row, smeared));
                }
                id = _builder.createCompositeConstruct(type, new_rows);
            } else {
                auto rows = t->dimension();
                auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int>(rows));
                std::vector<spv::Id> new_rows;
                new_rows.reserve(rows);
                for (auto i = 0u; i < rows; ++i) {
                    auto row_a = _builder.createCompositeExtract(a, row_type, i);
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
            if (a_type->is_scalar() || b_type->is_scalar()) {
                id = _builder.createBinOp(spv::Op::OpFMul, type, operand(0), operand(1));
            } else if (a_type->is_vector() && b_type->is_matrix()) {
                id = _builder.createBinOp(spv::Op::OpVectorTimesMatrix, type, operand(0), operand(1));
            } else if (a_type->is_matrix() && b_type->is_vector()) {
                id = _builder.createBinOp(spv::Op::OpMatrixTimesVector, type, operand(0), operand(1));
            } else {
                id = _builder.createBinOp(spv::Op::OpMatrixTimesMatrix, type, operand(0), operand(1));
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
            std::vector<spv::Id> comps;
            comps.reserve(inst->operand_count());
            for (auto i = 0u; i < inst->operand_count(); ++i) {
                comps.push_back(operand(i));
            }
            id = _builder.createCompositeConstruct(type, comps);
            break;
        }
        case xir::ArithmeticOp::SHUFFLE: {
            auto v = operand(0);
            auto dim = t->dimension();
            std::vector<spv::Id> comps;
            comps.reserve(dim);
            for (auto i = 1u; i <= dim; ++i) {
                auto idx = _emit_value(inst->operand(i));
                comps.push_back(_builder.createVectorExtractDynamic(v, _convert_type(t->element(), Usage::READ), idx));
            }
            id = _builder.createCompositeConstruct(type, comps);
            break;
        }
        case xir::ArithmeticOp::INSERT: {
            auto v = operand(0);
            auto e = operand(1);
            std::vector<unsigned> indices;
            for (auto i = 2u; i < inst->operand_count(); ++i) {
                if (auto op = inst->operand(i); op->isa<xir::Constant>()) {
                    auto c = static_cast<const xir::Constant *>(op);
                    auto idx = *static_cast<const uint32_t *>(c->data());
                    indices.push_back(idx);
                } else {
                    LUISA_ERROR_WITH_LOCATION("SPIR-V insert requires constant indices.");
                }
            }
            id = _builder.createCompositeInsert(e, v, type, indices);
            break;
        }
        case xir::ArithmeticOp::EXTRACT: {
            auto v = operand(0);
            auto base_type = inst->operand(0)->type();
            bool all_constant = true;
            std::vector<unsigned> const_indices;
            std::vector<spv::Id> dynamic_indices;
            for (auto i = 1u; i < inst->operand_count(); ++i) {
                if (auto op = inst->operand(i); op->isa<xir::Constant>()) {
                    auto c = static_cast<const xir::Constant *>(op);
                    auto idx = *static_cast<const uint32_t *>(c->data());
                    const_indices.push_back(idx);
                    dynamic_indices.push_back(_builder.makeUintConstant(idx));
                } else {
                    all_constant = false;
                    dynamic_indices.push_back(_emit_value(inst->operand(i)));
                }
            }
            if (all_constant) {
                id = _builder.createCompositeExtract(v, type, const_indices);
            } else if (base_type->is_vector()) {
                LUISA_ASSERT(dynamic_indices.size() == 1u, "SPIR-V vector extract should have only one index.");
                id = _builder.createVectorExtractDynamic(v, type, dynamic_indices[0]);
            } else if (base_type->is_array() || base_type->is_matrix()) {
                auto temp_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function,
                                                        _convert_type(base_type, Usage::READ), "extract_tmp");
                _builder.createStore(v, temp_var);
                auto ptr = _create_access_chain(spv::StorageClass::Function, temp_var, dynamic_indices);
                id = _builder.createLoad(ptr, spv::NoPrecision);
            } else {
                LUISA_NOT_IMPLEMENTED("SPIR-V dynamic extract for type {}.", base_type->description());
            }
            break;
        }

        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V arithmetic op {}.", xir::to_string(inst->op()));
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit arithmetic op.");
    if (inst->type() != nullptr) {
        _value_map.emplace(inst, id);
    }
}

spv::Id SpirvCodegenEntry::_resolve_resource_argument(const xir::Argument *arg) noexcept {
    if (auto it = _value_map.find(arg); it != _value_map.end()) {
        return it->second;
    }
    auto func = arg->parent_function();
    LUISA_ASSERT(func != nullptr, "Resource argument has no parent function.");
    size_t resource_index = 0;
    bool found = false;
    for (auto a : func->arguments()) {
        if (a == arg) {
            found = true;
            break;
        }
        if (a->is_resource()) {
            ++resource_index;
        }
    }
    LUISA_ASSERT(found, "Resource argument not found in parent function.");
    size_t base = 2;// ConstantValue + SamplerHeap
    bool cbuffer_non_empty = false;
    for (auto a : func->arguments()) {
        if (!a->is_resource()) {
            cbuffer_non_empty = true;
            break;
        }
    }
    if (cbuffer_non_empty) { ++base; }
    if (_use_buffer_bindless) { ++base; }
    if (_use_tex2d_bindless) { ++base; }
    if (_use_tex3d_bindless) { ++base; }
    auto prop_index = base + resource_index;
    LUISA_ASSERT(prop_index < _property_ids.size(), "Resource argument property out of range.");
    auto id = _property_ids[prop_index];
    if (prop_index < _properties.size()) {
        auto &prop = _properties[prop_index];
        if (prop.type == ShaderVariableType::UAVTextureHeap) {
            _is_storage_image_map[id] = true;
        } else if (prop.type == ShaderVariableType::SRVTextureHeap) {
            _is_storage_image_map[id] = false;
        }
    }
    _value_map.emplace(arg, id);
    return id;
}

void SpirvCodegenEntry::_emit_atomic_inst(const xir::AtomicInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    auto t = inst->type();

    auto base = _emit_value(inst->base());
    spv::Id ptr = base;
    auto indices = inst->index_uses();
    if (!indices.empty()) {
        std::vector<spv::Id> idx_ids;
        // Buffer variables are pointers to structs containing a runtime array.
        // Prepend 0 to access the first (and only) struct member.
        auto base_type = _builder.getTypeId(base);
        auto pointee_type = _builder.getContainedTypeId(base_type);
        if (_builder.isStructType(pointee_type)) {
            idx_ids.push_back(_builder.makeUintConstant(0u));
        }
        for (auto index_use : indices) {
            idx_ids.emplace_back(_emit_value(index_use->value()));
        }
        auto storage = _builder.getStorageClass(base);
        ptr = _create_access_chain(storage, base, idx_ids);
    }

    auto scope = _builder.makeUintConstant(static_cast<unsigned>(spv::Scope::Device));
    auto semantics = _builder.makeUintConstant(static_cast<unsigned>(spv::MemorySemanticsMask::MaskNone));
    auto semantics_equal = _builder.makeUintConstant(static_cast<unsigned>(spv::MemorySemanticsMask::MaskNone));

    spv::Id id = spv::NoResult;
    auto values = inst->value_uses();

    switch (inst->op()) {
        case xir::AtomicOp::EXCHANGE: {
            auto val = _emit_value(values[0]->value());
            id = _builder.createOp(spv::Op::OpAtomicExchange, type, {ptr, scope, semantics, val});
            break;
        }
        case xir::AtomicOp::COMPARE_EXCHANGE: {
            auto expected = _emit_value(values[0]->value());
            auto desired = _emit_value(values[1]->value());
            id = _builder.createOp(spv::Op::OpAtomicCompareExchange, type, {ptr, scope, semantics_equal, semantics, desired, expected});
            break;
        }
        case xir::AtomicOp::FETCH_ADD: {
            auto val = _emit_value(values[0]->value());
            if (t->is_float()) {
                if (t->is_float16()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float16_add);
                    _builder.addCapability(spv::Capability::AtomicFloat16AddEXT);
                } else if (t->is_float32()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_add);
                    _builder.addCapability(spv::Capability::AtomicFloat32AddEXT);
                } else if (t->is_float64()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_add);
                    _builder.addCapability(spv::Capability::AtomicFloat64AddEXT);
                }
                id = _builder.createOp(spv::Op::OpAtomicFAddEXT, type, {ptr, scope, semantics, val});
            } else {
                id = _builder.createOp(spv::Op::OpAtomicIAdd, type, {ptr, scope, semantics, val});
            }
            break;
        }
        case xir::AtomicOp::FETCH_SUB: {
            auto val = _emit_value(values[0]->value());
            if (t->is_float()) {
                auto neg_val = _builder.createUnaryOp(spv::Op::OpFNegate, type, val);
                if (t->is_float16()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float16_add);
                    _builder.addCapability(spv::Capability::AtomicFloat16AddEXT);
                } else if (t->is_float32()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_add);
                    _builder.addCapability(spv::Capability::AtomicFloat32AddEXT);
                } else if (t->is_float64()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_add);
                    _builder.addCapability(spv::Capability::AtomicFloat64AddEXT);
                }
                id = _builder.createOp(spv::Op::OpAtomicFAddEXT, type, {ptr, scope, semantics, neg_val});
            } else {
                id = _builder.createOp(spv::Op::OpAtomicISub, type, {ptr, scope, semantics, val});
            }
            break;
        }
        case xir::AtomicOp::FETCH_AND: {
            auto val = _emit_value(values[0]->value());
            id = _builder.createOp(spv::Op::OpAtomicAnd, type, {ptr, scope, semantics, val});
            break;
        }
        case xir::AtomicOp::FETCH_OR: {
            auto val = _emit_value(values[0]->value());
            id = _builder.createOp(spv::Op::OpAtomicOr, type, {ptr, scope, semantics, val});
            break;
        }
        case xir::AtomicOp::FETCH_XOR: {
            auto val = _emit_value(values[0]->value());
            id = _builder.createOp(spv::Op::OpAtomicXor, type, {ptr, scope, semantics, val});
            break;
        }
        case xir::AtomicOp::FETCH_MIN: {
            auto val = _emit_value(values[0]->value());
            if (t->is_float()) {
                if (t->is_float16()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_min_max);
                    _builder.addCapability(spv::Capability::AtomicFloat16MinMaxEXT);
                } else if (t->is_float32()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_min_max);
                    _builder.addCapability(spv::Capability::AtomicFloat32MinMaxEXT);
                } else if (t->is_float64()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_min_max);
                    _builder.addCapability(spv::Capability::AtomicFloat64MinMaxEXT);
                }
                id = _builder.createOp(spv::Op::OpAtomicFMinEXT, type, {ptr, scope, semantics, val});
            } else if (t->is_int()) {
                id = _builder.createOp(spv::Op::OpAtomicSMin, type, {ptr, scope, semantics, val});
            } else if (t->is_uint()) {
                id = _builder.createOp(spv::Op::OpAtomicUMin, type, {ptr, scope, semantics, val});
            } else {
                LUISA_NOT_IMPLEMENTED("SPIR-V atomic min for type {}.", t->description());
            }
            break;
        }
        case xir::AtomicOp::FETCH_MAX: {
            auto val = _emit_value(values[0]->value());
            if (t->is_float()) {
                if (t->is_float16()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_min_max);
                    _builder.addCapability(spv::Capability::AtomicFloat16MinMaxEXT);
                } else if (t->is_float32()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_min_max);
                    _builder.addCapability(spv::Capability::AtomicFloat32MinMaxEXT);
                } else if (t->is_float64()) {
                    _builder.addExtension(spv::E_SPV_EXT_shader_atomic_float_min_max);
                    _builder.addCapability(spv::Capability::AtomicFloat64MinMaxEXT);
                }
                id = _builder.createOp(spv::Op::OpAtomicFMaxEXT, type, {ptr, scope, semantics, val});
            } else if (t->is_int()) {
                id = _builder.createOp(spv::Op::OpAtomicSMax, type, {ptr, scope, semantics, val});
            } else if (t->is_uint()) {
                id = _builder.createOp(spv::Op::OpAtomicUMax, type, {ptr, scope, semantics, val});
            } else {
                LUISA_NOT_IMPLEMENTED("SPIR-V atomic max for type {}.", t->description());
            }
            break;
        }
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit atomic op.");
    _value_map.emplace(inst, id);
}

void SpirvCodegenEntry::_emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    spv::Id id = spv::NoResult;

    switch (inst->op()) {
        case xir::ResourceQueryOp::BUFFER_SIZE: {
            auto buffer = _emit_value(inst->operand(0));
            auto len = _builder.createArrayLength(buffer, 0u, 32u);
            if (inst->type()->is_uint64()) {
                id = _builder.createUnaryOp(spv::Op::OpUConvert, type, len);
            } else {
                id = len;
            }
            break;
        }
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: {
            auto buffer = _emit_value(inst->operand(0));
            auto len = _builder.createArrayLength(buffer, 0u, 32u);
            auto bytes = _builder.createBinOp(spv::Op::OpIMul, _builder.makeUintType(32), len, _builder.makeUintConstant(4u));
            if (inst->type()->is_uint64()) {
                id = _builder.createUnaryOp(spv::Op::OpUConvert, type, bytes);
            } else {
                id = bytes;
            }
            break;
        }
        case xir::ResourceQueryOp::TEXTURE2D_SIZE:
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: {
            auto tex_array = _emit_value(inst->operand(0));
            auto image_ptr = _create_access_chain(spv::StorageClass::UniformConstant, tex_array, {_builder.makeUintConstant(0u)});
            auto tex = _builder.createLoad(image_ptr, spv::NoPrecision);
            _builder.addCapability(spv::Capability::ImageQuery);
            if (_is_storage_image_map.at(tex_array)) {
                id = _builder.createOp(spv::Op::OpImageQuerySize, type, {tex});
            } else {
                id = _builder.createOp(spv::Op::OpImageQuerySizeLod, type, {tex, _builder.makeUintConstant(0u)});
            }
            break;
        }
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V resource query op {}.", xir::to_string(inst->op()));
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit resource query.");
    _value_map.emplace(inst, id);
}

spv::Id SpirvCodegenEntry::_emit_buffer_read_impl(spv::Id buffer, spv::Id word_offset, const Type *elem_type) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto spv_type = _convert_type(elem_type, Usage::READ);
    auto word_count = elem_type->size() / 4u;
    LUISA_ASSERT(word_count > 0u, "SPIR-V buffer read element size is zero.");
    if (word_count == 1u) {
        auto ptr = _create_access_chain(spv::StorageClass::StorageBuffer, buffer, {_builder.makeUintConstant(0u), word_offset});
        auto raw = _builder.createLoad(ptr, spv::NoPrecision);
        if (spv_type != uint_type) {
            return _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, raw);
        }
        return raw;
    }
    if (elem_type->is_vector()) {
        auto comp_type = _convert_type(elem_type->element(), Usage::READ);
        auto dim = elem_type->dimension();
        std::vector<spv::Id> comps;
        comps.reserve(dim);
        for (auto i = 0u; i < dim; ++i) {
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i));
            comps.push_back(_emit_buffer_read_impl(buffer, idx, elem_type->element()));
        }
        return _builder.createCompositeConstruct(spv_type, comps);
    }
    if (elem_type->is_matrix()) {
        auto elem = elem_type->element();
        auto dim = elem_type->dimension();
        auto col_type = Type::vector(elem, dim);
        auto col_word_count = col_type->size() / 4u;
        std::vector<spv::Id> cols;
        cols.reserve(dim);
        for (auto i = 0u; i < dim; ++i) {
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i * col_word_count));
            cols.push_back(_emit_buffer_read_impl(buffer, idx, col_type));
        }
        return _builder.createCompositeConstruct(spv_type, cols);
    }
    LUISA_NOT_IMPLEMENTED("SPIR-V buffer read for type {}.", elem_type->description());
}

spv::Id SpirvCodegenEntry::_emit_buffer_read(spv::Id buffer, spv::Id index, const Type *read_type, const Type *buffer_type) noexcept {
    if (buffer_type != nullptr && buffer_type->is_buffer() && buffer_type->element() != nullptr && buffer_type->element()->is_scalar()) {
        // Typed scalar buffer: direct element access
        auto ptr = _create_access_chain(spv::StorageClass::StorageBuffer, buffer, {_builder.makeUintConstant(0u), index});
        return _builder.createLoad(ptr, spv::NoPrecision);
    }
    // Byte buffer or bindless: word-level access
    auto uint_type = _builder.makeUintType(32);
    auto word_count = read_type->size() / 4u;
    LUISA_ASSERT(word_count > 0u, "SPIR-V buffer read element size is zero.");
    auto word_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, index, _builder.makeUintConstant(word_count));
    return _emit_buffer_read_impl(buffer, word_offset, read_type);
}

void SpirvCodegenEntry::_emit_buffer_write_impl(spv::Id buffer, spv::Id word_offset, spv::Id value, const Type *elem_type) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto spv_type = _convert_type(elem_type, Usage::READ);
    auto word_count = elem_type->size() / 4u;
    LUISA_ASSERT(word_count > 0u, "SPIR-V buffer write element size is zero.");
    if (word_count == 1u) {
        auto ptr = _create_access_chain(spv::StorageClass::StorageBuffer, buffer, {_builder.makeUintConstant(0u), word_offset});
        auto store_val = value;
        if (spv_type != uint_type) {
            store_val = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, value);
        }
        _builder.createStore(store_val, ptr);
        return;
    }
    if (elem_type->is_vector()) {
        auto comp_type = _convert_type(elem_type->element(), Usage::READ);
        auto dim = elem_type->dimension();
        for (auto i = 0u; i < dim; ++i) {
            auto comp = _builder.createCompositeExtract(value, comp_type, i);
            if (comp_type != uint_type) {
                comp = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, comp);
            }
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i));
            _emit_buffer_write_impl(buffer, idx, comp, elem_type->element());
        }
        return;
    }
    if (elem_type->is_matrix()) {
        auto elem = elem_type->element();
        auto dim = elem_type->dimension();
        auto col_type = Type::vector(elem, dim);
        auto col_word_count = col_type->size() / 4u;
        for (auto i = 0u; i < dim; ++i) {
            auto col = _builder.createCompositeExtract(value, _convert_type(col_type, Usage::READ), i);
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i * col_word_count));
            _emit_buffer_write_impl(buffer, idx, col, col_type);
        }
        return;
    }
    LUISA_NOT_IMPLEMENTED("SPIR-V buffer write for type {}.", elem_type->description());
}

void SpirvCodegenEntry::_emit_buffer_write(spv::Id buffer, spv::Id index, spv::Id value, const Type *value_type, const Type *buffer_type) noexcept {
    if (buffer_type != nullptr && buffer_type->is_buffer() && buffer_type->element() != nullptr && buffer_type->element()->is_scalar()) {
        // Typed scalar buffer: direct element access
        auto ptr = _create_access_chain(spv::StorageClass::StorageBuffer, buffer, {_builder.makeUintConstant(0u), index});
        _builder.createStore(value, ptr);
        return;
    }
    // Byte buffer or bindless: word-level access
    auto uint_type = _builder.makeUintType(32);
    auto word_count = value_type->size() / 4u;
    LUISA_ASSERT(word_count > 0u, "SPIR-V buffer write element size is zero.");
    auto word_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, index, _builder.makeUintConstant(word_count));
    _emit_buffer_write_impl(buffer, word_offset, value, value_type);
}

void SpirvCodegenEntry::_emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    spv::Id id = spv::NoResult;
    auto uint_type = _builder.makeUintType(32);

    switch (inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ:
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            id = _emit_buffer_read(buffer, index, inst->type(), inst->operand(0)->type());
            break;
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ:
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
            id = _emit_buffer_read(buffer, word_index, inst->type(), inst->operand(0)->type());
            break;
        }
        case xir::ResourceReadOp::TEXTURE2D_READ:
        case xir::ResourceReadOp::TEXTURE3D_READ: {
            auto tex_array = _emit_value(inst->operand(0));
            auto coord = _emit_value(inst->operand(1));
            auto image_ptr = _create_access_chain(spv::StorageClass::UniformConstant, tex_array, {_builder.makeUintConstant(0u)});
            auto tex = _builder.createLoad(image_ptr, spv::NoPrecision);
            if (_is_storage_image_map.at(tex_array)) {
                _builder.addCapability(spv::Capability::StorageImageReadWithoutFormat);
                id = _builder.createOp(spv::Op::OpImageRead, type, {tex, coord});
            } else {
                id = _builder.createOp(spv::Op::OpImageFetch, type, {tex, coord});
            }
            break;
        }
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: {
            // TODO: properly implement bindless buffer read
            id = _builder.makeFloatConstant(0.0f);
            break;
        }
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto byte_index = _ensure_type(_emit_value(inst->operand(2)), uint_type);
            auto slot_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto bdls_ptr = _create_access_chain(spv::StorageClass::StorageBuffer, bindless_array, {_builder.makeUintConstant(0u), slot_offset});
            auto buffer_idx = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            LUISA_ASSERT(_buffer_heap_id != spv::NoResult, "SPIR-V buffer heap not bound.");
            auto buffer_base = _create_access_chain(spv::StorageClass::StorageBuffer, _buffer_heap_id, {buffer_idx});
            auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
            id = _emit_buffer_read(buffer_base, word_index, inst->type(), nullptr);
            break;
        }
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V resource read op {}.", xir::to_string(inst->op()));
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit resource read.");
    _value_map.emplace(inst, id);
}

void SpirvCodegenEntry::_emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept {
    auto uint_type = _builder.makeUintType(32);

    switch (inst->op()) {
        case xir::ResourceWriteOp::BUFFER_WRITE:
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            _emit_buffer_write(buffer, index, value, inst->operand(2)->type(), inst->operand(0)->type());
            break;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE:
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto value = _emit_value(inst->operand(2));
            auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
            _emit_buffer_write(buffer, word_index, value, inst->operand(2)->type(), inst->operand(0)->type());
            break;
        }
        case xir::ResourceWriteOp::TEXTURE2D_WRITE:
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: {
            auto tex_array = _emit_value(inst->operand(0));
            auto coord = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            auto image_ptr = _create_access_chain(spv::StorageClass::UniformConstant, tex_array, {_builder.makeUintConstant(0u)});
            auto tex = _builder.createLoad(image_ptr, spv::NoPrecision);
            _builder.addCapability(spv::Capability::StorageImageWriteWithoutFormat);
            _builder.createNoResultOp(spv::Op::OpImageWrite, {tex, coord, value});
            break;
        }
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto elem_index = _emit_value(inst->operand(2));
            auto value = _emit_value(inst->operand(3));
            auto elem_type = inst->operand(3)->type();
            auto slot_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto bdls_ptr = _create_access_chain(spv::StorageClass::StorageBuffer, bindless_array, {_builder.makeUintConstant(0u), slot_offset});
            auto buffer_idx = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            LUISA_ASSERT(_buffer_heap_id != spv::NoResult, "SPIR-V buffer heap not bound.");
            auto buffer_base = _create_access_chain(spv::StorageClass::StorageBuffer, _buffer_heap_id, {buffer_idx});
            _emit_buffer_write(buffer_base, elem_index, value, elem_type, nullptr);
            break;
        }
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto byte_index = _ensure_type(_emit_value(inst->operand(2)), uint_type);
            auto value = _emit_value(inst->operand(3));
            auto slot_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto bdls_ptr = _create_access_chain(spv::StorageClass::StorageBuffer, bindless_array, {_builder.makeUintConstant(0u), slot_offset});
            auto buffer_idx = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            LUISA_ASSERT(_buffer_heap_id != spv::NoResult, "SPIR-V buffer heap not bound.");
            auto buffer_base = _create_access_chain(spv::StorageClass::StorageBuffer, _buffer_heap_id, {buffer_idx});
            auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
            _emit_buffer_write(buffer_base, word_index, value, inst->operand(3)->type(), nullptr);
            break;
        }
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V resource write op {}.", xir::to_string(inst->op()));
    }
}

void SpirvCodegenEntry::_emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept {
    spv::Id id = spv::NoResult;
    switch (inst->op()) {
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: {
            _builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Workgroup, spv::MemorySemanticsMask::AcquireRelease);
            break;
        }
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            id = _builder.createOp(spv::Op::OpGroupNonUniformElect, _convert_type(inst->type(), Usage::READ),
                                   {_builder.makeUintConstant(static_cast<unsigned>(spv::Scope::Subgroup))});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformAll, _convert_type(inst->type(), Usage::READ),
                                   {_builder.makeUintConstant(static_cast<unsigned>(spv::Scope::Subgroup)), val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformAny, _convert_type(inst->type(), Usage::READ),
                                   {_builder.makeUintConstant(static_cast<unsigned>(spv::Scope::Subgroup)), val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseAnd, _convert_type(inst->type(), Usage::READ),
                                   {_builder.makeUintConstant(static_cast<unsigned>(spv::Scope::Subgroup)),
                                    _builder.makeUintConstant(static_cast<unsigned>(spv::GroupOperation::Reduce)), val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseOr, _convert_type(inst->type(), Usage::READ),
                                   {_builder.makeUintConstant(static_cast<unsigned>(spv::Scope::Subgroup)),
                                    _builder.makeUintConstant(static_cast<unsigned>(spv::GroupOperation::Reduce)), val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseXor, _convert_type(inst->type(), Usage::READ),
                                   {_builder.makeUintConstant(static_cast<unsigned>(spv::Scope::Subgroup)),
                                    _builder.makeUintConstant(static_cast<unsigned>(spv::GroupOperation::Reduce)), val});
            break;
        }
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
            auto alloca = static_cast<const xir::AllocaInst *>(inst);
            auto type = _convert_type(alloca->type(), Usage::READ);
            auto storage = alloca->is_shared() ? spv::StorageClass::Workgroup : spv::StorageClass::Function;
            auto var = _builder.createVariable(spv::NoPrecision, storage, type, "alloca");
            set_result(var);
            break;
        }
        case xir::DerivedInstructionTag::LOAD: {
            auto load = static_cast<const xir::LoadInst *>(inst);
            auto ptr = _emit_value(load->variable());
            auto id = _builder.createLoad(ptr, spv::NoPrecision);
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::STORE: {
            auto store = static_cast<const xir::StoreInst *>(inst);
            auto ptr = _emit_value(store->variable());
            auto val = _emit_value(store->value());
            auto ptr_type = _builder.getTypeId(ptr);
            auto pointee_type = _builder.getContainedTypeId(ptr_type);
            auto val_type = _builder.getTypeId(val);
            if (pointee_type != val_type) {
                if (_builder.isScalarType(val_type) && _builder.isVectorType(pointee_type)) {
                    val = _builder.smearScalar(spv::NoPrecision, val, pointee_type);
                } else if (_builder.getTypeClass(pointee_type) == _builder.getTypeClass(val_type) &&
                           _builder.getNumTypeComponents(pointee_type) == _builder.getNumTypeComponents(val_type)) {
                    val = _builder.createUnaryOp(spv::Op::OpBitcast, pointee_type, val);
                }
            }
            _builder.createStore(val, ptr);
            break;
        }
        case xir::DerivedInstructionTag::GEP: {
            auto gep = static_cast<const xir::GEPInst *>(inst);
            auto base = _emit_value(gep->base());
            std::vector<spv::Id> indices;
            for (auto index_use : gep->index_uses()) {
                indices.emplace_back(_emit_value(index_use->value()));
            }
            auto storage = _builder.getStorageClass(base);
            auto id = _create_access_chain(storage, base, indices);
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::ARITHMETIC:
            _emit_arithmetic_inst(static_cast<const xir::ArithmeticInst *>(inst));
            break;
        case xir::DerivedInstructionTag::CALL: {
            auto call = static_cast<const xir::CallInst *>(inst);
            auto callee_func = _function_map.at(call->callee());
            std::vector<spv::Id> args;
            for (auto arg_use : call->argument_uses()) {
                auto arg = arg_use->value();
                if (arg->derived_value_tag() == xir::DerivedValueTag::ARGUMENT &&
                    static_cast<const xir::Argument *>(arg)->is_resource()) {
                    continue;
                }
                args.emplace_back(_emit_value(arg));
            }
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
                id = _builder.createUnaryOp(spv::Op::OpBitcast, spv_to, val);
            } else {
                if (from == to) {
                    id = val;
                } else if (from->is_bool() && to->is_scalar()) {
                    spv::Id zero = spv::NoResult;
                    spv::Id one = spv::NoResult;
                    if (to->is_int32()) {
                        zero = _builder.makeIntConstant(0);
                        one = _builder.makeIntConstant(1);
                    } else if (to->is_uint32()) {
                        zero = _builder.makeUintConstant(0);
                        one = _builder.makeUintConstant(1);
                    } else if (to->is_float32()) {
                        zero = _builder.makeFloatConstant(0.0f);
                        one = _builder.makeFloatConstant(1.0f);
                    } else {
                        LUISA_NOT_IMPLEMENTED("SPIR-V bool-to-scalar cast for {}.", to->description());
                    }
                    id = _builder.createTriOp(spv::Op::OpSelect, spv_to, val, one, zero);
                } else if (to->is_bool() && from->is_scalar()) {
                    spv::Id zero = spv::NoResult;
                    if (from->is_int32()) {
                        zero = _builder.makeIntConstant(0);
                    } else if (from->is_uint32()) {
                        zero = _builder.makeUintConstant(0);
                    } else if (from->is_float32()) {
                        zero = _builder.makeFloatConstant(0.0f);
                    } else {
                        LUISA_NOT_IMPLEMENTED("SPIR-V scalar-to-bool cast for {}.", from->description());
                    }
                    if (from->is_float()) {
                        id = _builder.createBinOp(spv::Op::OpFOrdNotEqual, spv_to, val, zero);
                    } else {
                        id = _builder.createBinOp(spv::Op::OpINotEqual, spv_to, val, zero);
                    }
                } else if (from->is_float() && to->is_int()) {
                    id = _builder.createUnaryOp(spv::Op::OpConvertFToS, spv_to, val);
                } else if (from->is_float() && to->is_uint()) {
                    id = _builder.createUnaryOp(spv::Op::OpConvertFToU, spv_to, val);
                } else if (from->is_int() && to->is_float()) {
                    id = _builder.createUnaryOp(spv::Op::OpConvertSToF, spv_to, val);
                } else if (from->is_uint() && to->is_float()) {
                    id = _builder.createUnaryOp(spv::Op::OpConvertUToF, spv_to, val);
                } else if (from->is_float() && to->is_float()) {
                    id = _builder.createUnaryOp(spv::Op::OpFConvert, spv_to, val);
                } else if ((from->is_int() || from->is_uint()) && (to->is_int() || to->is_uint())) {
                    if (from->size() == to->size()) {
                        id = val;
                    } else if (from->is_int() && to->is_int()) {
                        id = _builder.createUnaryOp(spv::Op::OpSConvert, spv_to, val);
                    } else if (from->is_uint() && to->is_uint()) {
                        id = _builder.createUnaryOp(spv::Op::OpUConvert, spv_to, val);
                    } else {
                        // Cross-signedness with different sizes: convert first, then bitcast
                        auto tmp_type = _builder.makeIntegerType(to->size() * 8, from->is_int());
                        id = _builder.createUnaryOp(from->is_int() ? spv::Op::OpSConvert : spv::Op::OpUConvert, tmp_type, val);
                        id = _builder.createUnaryOp(spv::Op::OpBitcast, spv_to, id);
                    }
                } else {
                    LUISA_NOT_IMPLEMENTED("SPIR-V static cast from {} to {}.", from->description(), to->description());
                }
            }
            set_result(id);
            break;
        }
        case xir::DerivedInstructionTag::IF: _emit_if_inst(static_cast<const xir::IfInst *>(inst)); break;
        case xir::DerivedInstructionTag::LOOP: _emit_loop_inst(static_cast<const xir::LoopInst *>(inst)); break;
        case xir::DerivedInstructionTag::SIMPLE_LOOP: _emit_simple_loop_inst(static_cast<const xir::SimpleLoopInst *>(inst)); break;
        case xir::DerivedInstructionTag::SWITCH: _emit_switch_inst(static_cast<const xir::SwitchInst *>(inst)); break;
        case xir::DerivedInstructionTag::BRANCH: _emit_branch_inst(static_cast<const xir::BranchInst *>(inst)); break;
        case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: _emit_conditional_branch_inst(static_cast<const xir::ConditionalBranchInst *>(inst)); break;
        case xir::DerivedInstructionTag::BREAK: {
            auto br = static_cast<const xir::BreakInst *>(inst);
            _builder.createBranch(false, _get_or_create_block(br->target_block()));
            break;
        }
        case xir::DerivedInstructionTag::CONTINUE: {
            auto cont = static_cast<const xir::ContinueInst *>(inst);
            _builder.createBranch(false, _get_or_create_block(cont->target_block()));
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
            LUISA_ERROR_WITH_LOCATION("Phi instructions should be eliminated before SPIR-V codegen.");
        case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
        case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
        case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
        case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC:
            LUISA_ERROR_WITH_LOCATION("Instruction {} should be eliminated before SPIR-V codegen.",
                                      xir::to_string(inst->derived_instruction_tag()));
        case xir::DerivedInstructionTag::PRINT:
            // Print is not supported in SPIR-V; emit as no-op
            break;
        case xir::DerivedInstructionTag::CLOCK:
        case xir::DerivedInstructionTag::ASSERT:
        case xir::DerivedInstructionTag::ASSUME:
        case xir::DerivedInstructionTag::DEBUG_BREAK:
        case xir::DerivedInstructionTag::OUTLINE:
        case xir::DerivedInstructionTag::RASTER_DISCARD:
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
        case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
            LUISA_NOT_IMPLEMENTED("SPIR-V codegen for instruction {}.", xir::to_string(inst->derived_instruction_tag()));
    }
}

spv::Id SpirvCodegenEntry::_ensure_type(spv::Id value, spv::Id target_type) noexcept {
    auto value_type = _builder.getTypeId(value);
    if (value_type == target_type) { return value; }
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
            if (val_signed == tgt_signed || val_width == tgt_width) {
                if (val_width == tgt_width) {
                    return _builder.createUnaryOp(spv::Op::OpBitcast, target_type, value);
                }
                return _builder.createUnaryOp(val_signed ? spv::Op::OpSConvert : spv::Op::OpUConvert, target_type, value);
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
    return _builder.createUnaryOp(spv::Op::OpBitcast, target_type, value);
}

}// namespace lc::spirv
