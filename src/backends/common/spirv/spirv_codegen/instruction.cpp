#include "entry.h"
#include <luisa/core/logging.h>
#include <SPIRV/GLSL.std.450.h>

namespace lc::spirv {

static const xir::Value *try_find_scalar_smear_source(const xir::Value *v) noexcept {
    if (!v->isa<xir::LoadInst>()) return nullptr;
    auto load = static_cast<const xir::LoadInst *>(v);
    auto ptr = load->variable();
    if (!ptr->isa<xir::Instruction>()) return nullptr;
    auto ptr_inst = static_cast<const xir::Instruction *>(ptr);
    if (!ptr_inst->isa<xir::AllocaInst>()) return nullptr;
    auto func = load->parent_function();
    if (!func) return nullptr;
    auto def = func->definition();
    if (!def) return nullptr;
    const xir::Value *stored = nullptr;
    size_t store_count = 0;
    def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
        if (inst->isa<xir::StoreInst>()) {
            auto store = static_cast<const xir::StoreInst *>(inst);
            if (store->variable() == ptr) {
                stored = store->value();
                store_count++;
            }
        }
    });
    if (store_count != 1 || stored == nullptr) return nullptr;
    if (!stored->isa<xir::ArithmeticInst>()) return nullptr;
    auto arith = static_cast<const xir::ArithmeticInst *>(stored);
    if (arith->op() != xir::ArithmeticOp::AGGREGATE) return nullptr;
    if (arith->operand_count() == 0) return nullptr;
    for (size_t i = 1; i < arith->operand_count(); ++i) {
        if (arith->operand(i) != arith->operand(0)) return nullptr;
    }
    return arith->operand(0);
}

void SpirvCodegenEntry::_emit_arithmetic_inst(const xir::ArithmeticInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    auto t = inst->type();
    auto elem = t->is_vector() || t->is_matrix() ? t->element() : t;
    if (elem->is_float8()) {
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
            if (is_float) {
                id = binary(spv::Op::OpFAdd);
            } else {
                auto a = operand(0);
                auto b = operand(1);
                // Constant-fold at emission time
                auto op_a = _builder.getOpCode(a);
                auto op_b = _builder.getOpCode(b);
                if ((op_a == spv::Op::OpConstant || op_a == spv::Op::OpSpecConstant) &&
                    (op_b == spv::Op::OpConstant || op_b == spv::Op::OpSpecConstant)) {
                    auto type_a = _builder.getTypeId(a);
                    if (_builder.isIntType(type_a) || _builder.isUintType(type_a)) {
                        // Attempt to compute folded constant
                        std::vector<spv::Id> ops = {a, b};
                        id = _builder.createSpecConstantOp(spv::Op::OpIAdd, type, ops, {});
                        break;
                    }
                }
                id = binary(spv::Op::OpIAdd);
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
            if (is_float) {
                // Peephole: if one operand is a smeared scalar, use VectorTimesScalar
                if (!is_scalar) {
                    auto op0 = inst->operand(0);
                    auto op1 = inst->operand(1);
                    auto try_vts = [&](const xir::Value *vec, const xir::Value *smeared) -> spv::Id {
                        if (smeared->isa<xir::Instruction>()) {
                            auto *smear_inst = static_cast<const xir::Instruction *>(smeared);
                            if (smear_inst->derived_instruction_tag() == xir::DerivedInstructionTag::ARITHMETIC) {
                                auto *arith = static_cast<const xir::ArithmeticInst *>(smear_inst);
                                if (arith->op() == xir::ArithmeticOp::AGGREGATE) {
                                    bool all_same = arith->operand_count() > 1;
                                    for (size_t j = 1; j < arith->operand_count(); ++j) {
                                        if (arith->operand(j) != arith->operand(0)) {
                                            all_same = false;
                                            break;
                                        }
                                    }
                                    if (all_same) {
                                        auto scalar = _emit_value(arith->operand(0));
                                        return _builder.createBinOp(spv::Op::OpVectorTimesScalar, type, _emit_value(vec), scalar);
                                    }
                                }
                            }
                        }
                        return spv::NoResult;
                    };
                    id = try_vts(op0, op1);
                    if (id == spv::NoResult) id = try_vts(op1, op0);
                    if (id == spv::NoResult) {
                        auto try_vts_load = [&](const xir::Value *vec, const xir::Value *loaded) -> spv::Id {
                            if (auto scalar_src = try_find_scalar_smear_source(loaded)) {
                                auto scalar = _emit_value(scalar_src);
                                return _builder.createBinOp(spv::Op::OpVectorTimesScalar, type, _emit_value(vec), scalar);
                            }
                            return spv::NoResult;
                        };
                        id = try_vts_load(op0, op1);
                        if (id == spv::NoResult) id = try_vts_load(op1, op0);
                    }
                }
                if (id == spv::NoResult) id = binary(spv::Op::OpFMul);
            } else {
                auto a = operand(0);
                auto b = operand(1);
                // Strength reduction: IMul(x, 2^n) → ShiftLeftLogical(x, n)
                if (inst->operand(1)->isa<xir::Constant>()) {
                    auto c = static_cast<const xir::Constant *>(inst->operand(1));
                    uint32_t val = 0;
                    bool is_pow2 = false;
                    if (c->type()->is_uint32()) {
                        val = c->as<uint32_t>();
                        is_pow2 = (val & (val - 1u)) == 0u && val != 0u;
                    } else if (c->type()->is_int32()) {
                        int32_t sval = c->as<int32_t>();
                        if (sval > 0) {
                            val = static_cast<uint32_t>(sval);
                            is_pow2 = (val & (val - 1u)) == 0u;
                        }
                    }
                    if (is_pow2) {
                        uint32_t shift = 0;
                        while (val >>= 1u) ++shift;
                        auto shift_id = _builder.makeUintConstant(shift);
                        id = _builder.createBinOp(spv::Op::OpShiftLeftLogical, type, a, shift_id);
                        break;
                    }
                }
                // Constant-fold at emission time
                auto op_a = _builder.getOpCode(a);
                auto op_b = _builder.getOpCode(b);
                if ((op_a == spv::Op::OpConstant || op_a == spv::Op::OpSpecConstant) &&
                    (op_b == spv::Op::OpConstant || op_b == spv::Op::OpSpecConstant)) {
                    auto type_a = _builder.getTypeId(a);
                    if (_builder.isIntType(type_a) || _builder.isUintType(type_a)) {
                        std::vector<spv::Id> ops = {a, b};
                        id = _builder.createSpecConstantOp(spv::Op::OpIMul, type, ops, {});
                        break;
                    }
                }
                id = binary(spv::Op::OpIMul);
            }
            break;
        }
        case xir::ArithmeticOp::BINARY_DIV:
            if (is_float) {
                id = binary(spv::Op::OpFDiv);
            } else if (is_signed_int) {
                auto a = operand(0);
                auto b = operand(1);
                // Strength reduction: IDiv(x, 2^n) → ShiftRightArithmetic(x, n)
                if (inst->operand(1)->isa<xir::Constant>()) {
                    auto c = static_cast<const xir::Constant *>(inst->operand(1));
                    int32_t sval = 0;
                    if (c->type()->is_int32()) {
                        sval = c->as<int32_t>();
                    }
                    if (sval > 0 && (static_cast<uint32_t>(sval) & (static_cast<uint32_t>(sval) - 1u)) == 0u) {
                        uint32_t shift = 0;
                        uint32_t v = static_cast<uint32_t>(sval);
                        while (v >>= 1u) ++shift;
                        auto shift_id = _builder.makeUintConstant(shift);
                        id = _builder.createBinOp(spv::Op::OpShiftRightArithmetic, type, a, shift_id);
                        break;
                    }
                }
                id = binary(spv::Op::OpSDiv);
            } else
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
            auto width_id = _builder.makeUintConstant(static_cast<uint32_t>(width));
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
            auto width_id = _builder.makeUintConstant(static_cast<uint32_t>(width));
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
            // Constant-fold at emission time
            if (inst->operand(0)->isa<xir::Constant>() && inst->operand(1)->isa<xir::Constant>()) {
                auto a = operand(0);
                auto b = operand(1);
                auto op_a = _builder.getOpCode(a);
                auto op_b = _builder.getOpCode(b);
                if ((op_a == spv::Op::OpConstant || op_a == spv::Op::OpSpecConstant) &&
                    (op_b == spv::Op::OpConstant || op_b == spv::Op::OpSpecConstant)) {
                    if (_builder.isIntType(_builder.getTypeId(a)) || _builder.isUintType(_builder.getTypeId(a))) {
                        id = _builder.createSpecConstantOp(spv::Op::OpIEqual, type, {a, b}, {});
                        break;
                    }
                }
            }
            if (op_elem->is_float())
                id = binary(spv::Op::OpFOrdEqual);
            else
                id = binary(spv::Op::OpIEqual);
            break;
        }
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: {
            auto op_elem = inst->operand(0)->type();
            op_elem = op_elem->is_vector() ? op_elem->element() : op_elem;
            // Constant-fold at emission time
            if (inst->operand(0)->isa<xir::Constant>() && inst->operand(1)->isa<xir::Constant>()) {
                auto a = operand(0);
                auto b = operand(1);
                auto op_a = _builder.getOpCode(a);
                auto op_b = _builder.getOpCode(b);
                if ((op_a == spv::Op::OpConstant || op_a == spv::Op::OpSpecConstant) &&
                    (op_b == spv::Op::OpConstant || op_b == spv::Op::OpSpecConstant)) {
                    if (_builder.isIntType(_builder.getTypeId(a)) || _builder.isUintType(_builder.getTypeId(a))) {
                        id = _builder.createSpecConstantOp(spv::Op::OpINotEqual, type, {a, b}, {});
                        break;
                    }
                }
            }
            if (op_elem->is_float())
                id = binary(spv::Op::OpFOrdNotEqual);
            else
                id = binary(spv::Op::OpINotEqual);
            break;
        }
        case xir::ArithmeticOp::SELECT: {
            // XIR SELECT operands are (false_value, true_value, condition)
            // Constant-fold: if condition is a constant bool, pick the correct operand
            if (inst->operand(2)->isa<xir::Constant>()) {
                auto cond_const = static_cast<const xir::Constant *>(inst->operand(2));
                if (cond_const->type()->is_bool()) {
                    bool cond_val = cond_const->as<bool>();
                    id = cond_val ? operand(1) : operand(0);
                    break;
                }
            }
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
                    auto bit_width = static_cast<int32_t>(_builder.getScalarTypeWidth(cond_type));
                    zero = _builder.makeIntConstant(_builder.makeIntType(bit_width), 0u, false);
                } else if (_builder.isFloatType(cond_type)) {
                    auto bit_width = static_cast<int32_t>(_builder.getScalarTypeWidth(cond_type));
                    if (bit_width == 16) {
                        zero = _builder.makeFloat16Constant(0.0f);
                    } else if (bit_width == 32) {
                        zero = _builder.makeFloatConstant(0.0f);
                    } else if (bit_width == 64) {
                        zero = _builder.makeDoubleConstant(0.0);
                    } else if (bit_width == 8) {
                        // FP8: use the appropriate constant constructor
                        // We don't know the exact encoding here, but float8 values
                        // should not appear as SELECT conditions. Fall through.
                    }
                }
                if (zero != spv::NoResult) {
                    if (_builder.isVectorType(cond_type)) {
                        auto dim = static_cast<int32_t>(_builder.getNumTypeComponents(cond_type));
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
            auto bit_width = static_cast<int32_t>(t->is_scalar() ? t->size() * 8 : t->element()->size() * 8);
            auto bit_width_id = elem->is_uint()
                ? _builder.makeIntConstant(_builder.makeUintType(bit_width), static_cast<unsigned>(bit_width), false)
                : _builder.makeIntConstant(_builder.makeIntType(bit_width), static_cast<unsigned>(bit_width), false);
            auto minus_one = elem->is_uint() ? _builder.makeIntConstant(_builder.makeUintType(bit_width), 0xFFFFFFFFu, false) : _builder.makeIntConstant(_builder.makeIntType(bit_width), 0xFFFFFFFFu, false);
            if (!is_scalar) {
                bit_width_id = _builder.smearScalar(spv::NoPrecision, bit_width_id, type);
                minus_one = _builder.smearScalar(spv::NoPrecision, minus_one, type);
            }
            auto is_zero = _builder.createBinOp(spv::Op::OpIEqual, _builder.makeBoolType(), find_msb, minus_one);
            auto one = elem->is_uint() ? _builder.makeIntConstant(_builder.makeUintType(bit_width), 1u, false) : _builder.makeIntConstant(_builder.makeIntType(bit_width), 1u, false);
            auto clz_val = _builder.createBinOp(spv::Op::OpISub, type, _builder.createBinOp(spv::Op::OpISub, type, bit_width_id, one), find_msb);
            id = _builder.createTriOp(spv::Op::OpSelect, type, is_zero, bit_width_id, clz_val);
            break;
        }
        case xir::ArithmeticOp::CTZ: {
            auto find_lsb = glsl(GLSLstd450FindILsb, operand(0));
            auto bit_width = static_cast<int32_t>(t->is_scalar() ? t->size() * 8 : t->element()->size() * 8);
            auto bit_width_id = elem->is_uint()
                ? _builder.makeIntConstant(_builder.makeUintType(bit_width), static_cast<unsigned>(bit_width), false)
                : _builder.makeIntConstant(_builder.makeIntType(bit_width), static_cast<unsigned>(bit_width), false);
            auto minus_one = elem->is_uint() ? _builder.makeIntConstant(_builder.makeUintType(bit_width), 0xFFFFFFFFu, false) : _builder.makeIntConstant(_builder.makeIntType(bit_width), 0xFFFFFFFFu, false);
            if (!is_scalar) {
                bit_width_id = _builder.smearScalar(spv::NoPrecision, bit_width_id, type);
                minus_one = _builder.smearScalar(spv::NoPrecision, minus_one, type);
            }
            auto is_zero = _builder.createBinOp(spv::Op::OpIEqual, _builder.makeBoolType(), find_lsb, minus_one);
            id = _builder.createTriOp(spv::Op::OpSelect, type, is_zero, bit_width_id, find_lsb);
            break;
        }
        case xir::ArithmeticOp::POPCOUNT: {
            auto op = operand(0);
            auto op_scalar = _builder.getScalarTypeId(_builder.getTypeId(op));
            if (_builder.getTypeClass(op_scalar) == spv::Op::OpTypeInt && _builder.getScalarTypeWidth(op_scalar) == 8) {
                op = _ensure_type(op, type);
            }
            id = _builder.createUnaryOp(spv::Op::OpBitCount, type, op);
            break;
        }
        case xir::ArithmeticOp::REVERSE: {
            auto op = operand(0);
            auto op_scalar = _builder.getScalarTypeId(_builder.getTypeId(op));
            if (_builder.getTypeClass(op_scalar) == spv::Op::OpTypeInt && _builder.getScalarTypeWidth(op_scalar) == 8) {
                op = _ensure_type(op, type);
            }
            id = _builder.createUnaryOp(spv::Op::OpBitReverse, type, op);
            break;
        }
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
            auto inv_log2_10 = _builder.makeFloatConstant(0.3010299956639812f);
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
        case xir::ArithmeticOp::ROUND: {
            auto x = operand(0);
            auto half = _builder.makeFloatConstant(0.5f);
            auto sign = glsl(GLSLstd450FSign, x);
            spv::Id signed_half;
            if (!is_scalar)
                signed_half = _builder.createBinOp(spv::Op::OpVectorTimesScalar, type, sign, half);
            else
                signed_half = _builder.createBinOp(spv::Op::OpFMul, type, half, sign);
            auto sum = _builder.createBinOp(spv::Op::OpFAdd, type, x, signed_half);
            id = glsl(GLSLstd450Trunc, sum);
            break;
        }
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
        case xir::ArithmeticOp::CROSS:
            id = glsl(GLSLstd450Cross, operand(0), operand(1));
            break;
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
            auto extract = [&](uint32_t i) {
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
                            id = make_glsl_call(GLSLstd450FMin, elem_spv_type, {id, comp});
                        else if (elem_type->is_int())
                            id = make_glsl_call(GLSLstd450SMin, elem_spv_type, {id, comp});
                        else
                            id = make_glsl_call(GLSLstd450UMin, elem_spv_type, {id, comp});
                        break;
                    case xir::ArithmeticOp::REDUCE_MAX:
                        if (elem_type->is_float())
                            id = make_glsl_call(GLSLstd450FMax, elem_spv_type, {id, comp});
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
            auto a_dim = a_type->dimension();
            auto b_dim = b_type->dimension();
            auto elem_type = a_type->element();
            auto elem_spv_type = _convert_type(elem_type, Usage::READ);
            auto vec_type = _builder.makeVectorType(elem_spv_type, static_cast<int32_t>(b_dim));
            std::vector<spv::Id> rows;
            rows.reserve(a_dim);
            for (uint i = 0u; i < a_dim; ++i) {
                auto ai = _builder.createCompositeExtract(a, elem_spv_type, i);
                auto row = _builder.createBinOp(spv::Op::OpVectorTimesScalar, vec_type, b, ai);
                rows.push_back(row);
            }
            id = _builder.createCompositeConstruct(type, rows);
            break;
        }

        // Matrix operations
        case xir::ArithmeticOp::MATRIX_COMP_NEG: {
            auto mat = operand(0);
            auto rows = t->dimension();
            auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
            std::vector<spv::Id> new_rows;
            new_rows.reserve(rows);
            for (uint i = 0u; i < rows; ++i) {
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
            for (uint i = 0u; i < rows; ++i) {
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
                new_rows.push_back(_builder.createBinOp(spv::Op::OpFAdd, row_type, lhs, rhs));
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
            for (uint i = 0u; i < rows; ++i) {
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
                new_rows.push_back(_builder.createBinOp(spv::Op::OpFSub, row_type, lhs, rhs));
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
                id = _builder.createBinOp(spv::Op::OpMatrixTimesScalar, type, b, a);
            } else if (b_type->is_scalar()) {
                id = _builder.createBinOp(spv::Op::OpMatrixTimesScalar, type, a, b);
            } else {
                auto rows = t->dimension();
                auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
                std::vector<spv::Id> new_rows;
                new_rows.reserve(rows);
                for (uint i = 0u; i < rows; ++i) {
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
                auto row_type = _builder.makeVectorType(_convert_type(t->element(), Usage::READ), static_cast<int32_t>(rows));
                std::vector<spv::Id> new_rows;
                new_rows.reserve(rows);
                for (uint i = 0u; i < rows; ++i) {
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
                for (uint i = 0u; i < rows; ++i) {
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
            // Peephole: detect Extract-from-same-vector pattern and emit OpVectorShuffle
            if (t->is_vector() && inst->operand_count() >= 2u) {
                const xir::Value *common_base = nullptr;
                bool all_extract = true;
                luisa::vector<uint32_t> indices;
                indices.reserve(inst->operand_count());
                for (uint i = 0u; i < inst->operand_count(); ++i) {
                    auto op = inst->operand(i);
                    if (!op->isa<xir::ArithmeticInst>()) { all_extract = false; break; }
                    auto ari = static_cast<const xir::ArithmeticInst *>(op);
                    if (ari->op() != xir::ArithmeticOp::EXTRACT) { all_extract = false; break; }
                    auto base = ari->operand(0);
                    if (!base->type()->is_vector()) { all_extract = false; break; }
                    if (common_base == nullptr) {
                        common_base = base;
                    } else if (common_base != base) {
                        all_extract = false;
                        break;
                    }
                    // Index must be constant
                    if (ari->operand_count() < 2u || !ari->operand(1)->isa<xir::Constant>()) {
                        all_extract = false;
                        break;
                    }
                    auto idx_val = static_cast<const xir::Constant *>(ari->operand(1))->as<uint32_t>();
                    indices.push_back(static_cast<uint32_t>(idx_val));
                }
                if (all_extract && common_base != nullptr) {
                    auto base_spv = _emit_value(const_cast<xir::Value *>(common_base));
                    std::vector<uint32_t> std_indices(indices.begin(), indices.end());
                    id = _builder.createRvalueSwizzle(spv::NoPrecision, type, base_spv, std_indices);
                    break;
                }
            }
            // Fallthrough: normal CompositeConstruct
            {
                std::vector<spv::Id> comps;
                comps.reserve(inst->operand_count());
                for (uint i = 0u; i < inst->operand_count(); ++i) {
                    comps.push_back(operand(i));
                }
                id = _builder.createCompositeConstruct(type, comps);
            }
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
                    auto idx = static_cast<const xir::Constant *>(inst->operand(i))->as<uint32_t>();
                    shuffle_indices.push_back(static_cast<uint32_t>(idx));
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
                for (auto i = 1u; i <= dim; ++i) {
                    auto idx = _emit_value(inst->operand(i));
                    comps.push_back(_builder.createVectorExtractDynamic(v, _convert_type(t->element(), Usage::READ), idx));
                }
                id = _builder.createCompositeConstruct(type, comps);
            }
            break;
        }
        case xir::ArithmeticOp::INSERT: {
            auto v = operand(0);
            auto e = operand(1);
            std::vector<uint32_t> const_indices;
            std::vector<spv::Id> dynamic_indices;
            bool all_constant = true;
            for (auto i = 2u; i < inst->operand_count(); ++i) {
                if (auto op = inst->operand(i); op->isa<xir::Constant>()) {
                    auto c = static_cast<const xir::Constant *>(op);
                    auto idx = *static_cast<const uint32_t *>(c->data());
                    const_indices.push_back(idx);
                } else {
                    all_constant = false;
                    dynamic_indices.push_back(_emit_value(inst->operand(i)));
                    const_indices.push_back(0u);// placeholder
                }
            }
            if (all_constant) {
                id = _builder.createCompositeInsert(e, v, type, const_indices);
            } else {
                // Fallback: use alloca + access chain + store + load
                auto temp_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, type, "insert_tmp");
                _builder.createStore(v, temp_var);
                std::vector<spv::Id> access_indices;
                for (auto i = 2u; i < inst->operand_count(); ++i) {
                    access_indices.push_back(_emit_value(inst->operand(i)));
                }
                auto ptr = _create_access_chain(spv::StorageClass::Function, temp_var, access_indices);
                _builder.createStore(e, ptr);
                id = _builder.createLoad(temp_var, spv::NoPrecision);
            }
            break;
        }
        case xir::ArithmeticOp::EXTRACT: {
            auto base_value = inst->operand(0);
            auto base_type = base_value->type();

            // Fast path for UBO-lowered constant arrays: emit a single indexed load
            // through the constant cache instead of materializing the array as an SSA value.
            if (base_value->isa<xir::Constant>()) {
                auto c = static_cast<const xir::Constant *>(base_value);
                if (auto ubo_it = _ubo_constant_member_by_hash.find(c->hash());
                    ubo_it != _ubo_constant_member_by_hash.end()) {
                    std::vector<spv::Id> indices;
                    indices.reserve(inst->operand_count());
                    indices.push_back(_builder.makeUintConstant(ubo_it->second));
                    for (auto i = 1u; i < inst->operand_count(); ++i) {
                        indices.push_back(_emit_value(inst->operand(i)));
                    }
                    auto ptr = _create_access_chain(spv::StorageClass::Uniform, _constant_ubo_var, indices);
                    id = _builder.createLoad(ptr, spv::NoPrecision);
                    break;
                }
            }

            auto v = operand(0);
            bool all_constant = true;
            std::vector<uint32_t> const_indices;
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
                // For small arrays/matrices, lower a single dynamic index to a chain of
                // OpSelect over OpCompositeExtract. This stays in registers and avoids the
                // memory round-trip that an OpAccessChain through a Function-scope variable
                // would introduce on most drivers.
                auto elem_count = base_type->dimension();
                if (dynamic_indices.size() == 1u && elem_count <= 16u) {
                    auto idx = dynamic_indices[0];
                    auto result = _builder.createCompositeExtract(v, type, {0u});
                    for (uint32_t i = 1u; i < elem_count; ++i) {
                        auto elem_i = _builder.createCompositeExtract(v, type, {i});
                        auto cmp = _builder.createBinOp(spv::Op::OpIEqual, _builder.makeBoolType(),
                                                        idx, _builder.makeUintConstant(i));
                        result = _builder.createTriOp(spv::Op::OpSelect, type, cmp, elem_i, result);
                    }
                    id = result;
                } else {
                    auto temp_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function,
                                                            _convert_type(base_type, Usage::READ), "extract_tmp");
                    _builder.createStore(v, temp_var);
                    auto ptr = _create_access_chain(spv::StorageClass::Function, temp_var, dynamic_indices);
                    id = _builder.createLoad(ptr, spv::NoPrecision);
                }
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

size_t SpirvCodegenEntry::_get_resource_property_base(const xir::Function *func) const noexcept {
    size_t base = 2;// ConstantValue + SamplerHeap
    bool cbuffer_non_empty = false;
    for (auto a : func->arguments()) {
        if (!a->is_resource()) {
            cbuffer_non_empty = true;
            break;
        }
    }
    if (cbuffer_non_empty) { ++base; }
    if (_has_constant_ubo) { ++base; }
    if (_use_buffer_bindless) { ++base; }
    if (_use_tex2d_bindless) { ++base; }
    if (_use_tex3d_bindless) { ++base; }
    return base;
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
    size_t base = _get_resource_property_base(func);
    auto prop_index = base;
    for (auto a : func->arguments()) {
        if (a == arg) { break; }
        if (a->is_resource()) {
            ++prop_index;
            if (a->type()->tag() == Type::Tag::ACCEL) {
                ++prop_index;
            }
        }
    }
    LUISA_ASSERT(prop_index < _property_ids.size(), "Resource argument property out of range.");
    auto id = _property_ids[prop_index];
    // _property_ids includes a push constant at index 0 not present in _properties,
    // so we need to shift by 1 when indexing into _properties.
    auto prop_idx_in_properties = prop_index - 1;
    if (prop_idx_in_properties < _properties.size()) {
        auto &prop = _properties[prop_idx_in_properties];
        if (prop.type == ShaderVariableType::UAVTextureHeap) {
            _is_storage_image_map[id] = true;
        } else if (prop.type == ShaderVariableType::SRVTextureHeap) {
            _is_storage_image_map[id] = false;
        }
    }
    if (arg->type()->tag() == Type::Tag::ACCEL && prop_index + 1 < _property_ids.size()) {
        _accel_instance_buffer_map.emplace(id, _property_ids[prop_index + 1]);
    }
    _value_map.emplace(arg, id);
    return id;
}

spv::Id SpirvCodegenEntry::_resolve_accel_instance_buffer(const xir::Argument *arg) noexcept {
    auto func = arg->parent_function();
    LUISA_ASSERT(func != nullptr, "Resource argument has no parent function.");
    size_t base = _get_resource_property_base(func);
    auto prop_index = base;
    for (auto a : func->arguments()) {
        if (a == arg) { break; }
        if (a->is_resource()) {
            ++prop_index;
            if (a->type()->tag() == Type::Tag::ACCEL) {
                ++prop_index;
            }
        }
    }
    LUISA_ASSERT(prop_index + 1 < _property_ids.size(), "Resource argument property out of range.");
    return _property_ids[prop_index + 1];
}

spv::Id SpirvCodegenEntry::_emit_float_atomic_cas_loop(spv::Id ptr, spv::Id val, spv::Id float_type, xir::AtomicOp op) noexcept {
    auto &function = _builder.getBuildPoint()->getParent();
    auto uint_type = _builder.makeUintType(32);
    auto bool_type = _builder.makeBoolType();
    LUISA_ASSERT(float_type == _builder.makeFloatType(32),
                 "SPIR-V CAS loop only supports float32 for non-scalar buffer atomics.");

    auto result_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, float_type, "atomic_result");

    auto loop_header = &_builder.makeNewBlock();
    auto loop_body = &_builder.makeNewBlock();
    auto loop_continue = &_builder.makeNewBlock();
    auto merge = &_builder.makeNewBlock();

    _builder.createBranch(false, loop_header);

    _builder.setBuildPoint(loop_header);
    _used_merge_blocks.emplace((merge)->getId());
        _builder.createLoopMerge(merge, loop_continue, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, loop_body);

    _builder.setBuildPoint(loop_body);
    auto scope = _builder.makeUintConstant(static_cast<uint32_t>(spv::Scope::Device));
    auto semantics = _builder.makeUintConstant(static_cast<uint32_t>(spv::MemorySemanticsMask::MaskNone));

    spv::Id old_uint;
    spv::Id old_float;
    old_uint = _builder.createOp(spv::Op::OpAtomicLoad, uint_type, {ptr, scope, semantics});
    old_float = _builder.createUnaryOp(spv::Op::OpBitcast, float_type, old_uint);
    _builder.createStore(old_float, result_var);

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
            new_float = _builder.createBuiltinCall(float_type, _glsl450, GLSLstd450FMax, {old_float, val});
            break;
        case xir::AtomicOp::FETCH_MIN:
            new_float = _builder.createBuiltinCall(float_type, _glsl450, GLSLstd450FMin, {old_float, val});
            break;
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V CAS loop for atomic op {}.", xir::to_string(op));
    }

    auto new_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, new_float);

    auto result = _builder.createOp(spv::Op::OpAtomicCompareExchange, uint_type,
                                    {ptr, scope, semantics, semantics, new_uint, old_uint});
    auto cmp = _builder.createBinOp(spv::Op::OpIEqual, bool_type, result, old_uint);

    _builder.createConditionalBranch(cmp, merge, loop_continue);

    // Loop continue
    _builder.setBuildPoint(loop_continue);
    _builder.createBranch(false, loop_header);

    // Merge
    _builder.setBuildPoint(merge);

    // Load and return the result
    return _builder.createLoad(result_var, spv::NoPrecision);
}

spv::Id SpirvCodegenEntry::_emit_float_compare_exchange_cas_loop(spv::Id ptr, spv::Id expected, spv::Id desired, spv::Id float_type) noexcept {
    auto &function = _builder.getBuildPoint()->getParent();
    auto bool_type = _builder.makeBoolType();
    auto uint_type = _builder.makeUintType(32);

    // Local variable to hold the result (old float value)
    auto result_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, float_type, "cas_result");

    // Create blocks
    auto loop_header = &_builder.makeNewBlock();
    auto loop_body = &_builder.makeNewBlock();
    auto loop_continue = &_builder.makeNewBlock();
    auto loop_merge = &_builder.makeNewBlock();

    // Branch to loop header
    _builder.createBranch(false, loop_header);

    // Loop header
    _builder.setBuildPoint(loop_header);
    _used_merge_blocks.emplace((loop_merge)->getId());
        _builder.createLoopMerge(loop_merge, loop_continue, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, loop_body);

    // Loop body
    _builder.setBuildPoint(loop_body);
    auto scope = _builder.makeUintConstant(static_cast<uint32_t>(spv::Scope::Device));
    auto semantics = _builder.makeUintConstant(static_cast<uint32_t>(spv::MemorySemanticsMask::MaskNone));

    // old = AtomicLoad(ptr)
    auto old_float = _builder.createOp(spv::Op::OpAtomicLoad, float_type, {ptr, scope, semantics});
    _builder.createStore(old_float, result_var);

    // Bitwise comparison via uint bitcast
    auto old_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, old_float);
    auto expected_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, expected);
    auto cmp = _builder.createBinOp(spv::Op::OpIEqual, bool_type, old_uint, expected_uint);
    _builder.createConditionalBranch(cmp, loop_continue, loop_merge);

    // Loop continue (try exchange)
    _builder.setBuildPoint(loop_continue);
    auto swapped = _builder.createOp(spv::Op::OpAtomicExchange, float_type, {ptr, scope, semantics, desired});
    auto swapped_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, swapped);
    auto stored_old = _builder.createLoad(result_var, spv::NoPrecision);
    auto stored_old_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, stored_old);
    auto success = _builder.createBinOp(spv::Op::OpIEqual, bool_type, swapped_uint, stored_old_uint);
    _builder.createConditionalBranch(success, loop_merge, loop_header);

    // Merge
    _builder.setBuildPoint(loop_merge);
    return _builder.createLoad(result_var, spv::NoPrecision);
}

void SpirvCodegenEntry::_emit_atomic_inst(const xir::AtomicInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    auto t = inst->type();

    auto base = _emit_value(inst->base());
    spv::Id ptr = base;
    auto indices = inst->index_uses();
    auto base_xir_type = inst->base()->type();
    size_t buffer_elem_word_count = 1;
    bool is_non_scalar_buffer = false;
    if (base_xir_type != nullptr && base_xir_type->is_buffer() && base_xir_type->element() != nullptr) {
        buffer_elem_word_count = std::max(size_t{1}, base_xir_type->element()->size() / 4u);
        is_non_scalar_buffer = buffer_elem_word_count > 1 && indices.size() > 1;
    }
    if (!indices.empty()) {
        std::vector<spv::Id> idx_ids;
        // Buffer variables are pointers to structs containing a runtime array.
        // Prepend 0 to access the first (and only) struct member.
        auto base_type = _builder.getTypeId(base);
        auto pointee_type = _builder.getContainedTypeId(base_type);
        if (_builder.isStructType(pointee_type)) {
            idx_ids.push_back(_builder.makeUintConstant(0u));
        }

        if (is_non_scalar_buffer) {
            auto uint_type = _builder.makeUintType(32);
            spv::Id word_offset;
            {
                auto elem_index = _emit_value(indices[0]->value());
                if (buffer_elem_word_count > 1) {
                    word_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, elem_index,
                                                       _builder.makeUintConstant(static_cast<uint32_t>(buffer_elem_word_count)));
                } else {
                    word_offset = elem_index;
                }
            }
            if (indices.size() > 1) {
                auto elem_type = base_xir_type->element();
                size_t byte_offset = 0;
                bool can_compute = true;
                for (size_t i = 1; i < indices.size(); ++i) {
                    auto idx_val = indices[i]->value();
                    if (idx_val->isa<xir::Constant>()) {
                        auto c = static_cast<const xir::Constant *>(idx_val);
                        auto idx = *static_cast<const uint32_t *>(c->data());
                        if (elem_type->is_structure()) {
                            auto members = elem_type->members();
                            for (auto j = 0u; j < idx; ++j) {
                                byte_offset = luisa::align(byte_offset, members[j]->alignment());
                                byte_offset += members[j]->size();
                            }
                            byte_offset = luisa::align(byte_offset, members[idx]->alignment());
                            elem_type = members[idx];
                        } else if (elem_type->is_array()) {
                            auto arr_elem = elem_type->element();
                            byte_offset += idx * arr_elem->size();
                            elem_type = arr_elem;
                        } else if (elem_type->is_vector()) {
                            auto vec_elem = elem_type->element();
                            byte_offset += idx * vec_elem->size();
                            elem_type = vec_elem;
                        } else if (elem_type->is_matrix()) {
                            auto mat_elem = elem_type->element();
                            auto col_type = Type::vector(mat_elem, elem_type->dimension());
                            byte_offset += idx * col_type->size();
                            elem_type = col_type;
                        } else {
                            can_compute = false;
                            break;
                        }
                    } else {
                        can_compute = false;
                        break;
                    }
                }
                if (can_compute) {
                    auto sub_word_offset = byte_offset / 4u;
                    if (sub_word_offset > 0) {
                        word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset,
                                                           _builder.makeUintConstant(static_cast<uint32_t>(sub_word_offset)));
                    }
                } else {
                    LUISA_NOT_IMPLEMENTED("SPIR-V dynamic sub-element atomic indices for non-scalar buffers.");
                }
            }
            idx_ids.push_back(word_offset);
        } else {
            for (auto index_use : indices) {
                idx_ids.emplace_back(_emit_value(index_use->value()));
            }
        }

        auto storage = _builder.getStorageClass(base);
        ptr = _create_access_chain(storage, base, idx_ids);
    }

    auto scope = _builder.makeUintConstant(static_cast<uint32_t>(spv::Scope::Device));
    auto semantics = _builder.makeUintConstant(static_cast<uint32_t>(spv::MemorySemanticsMask::MaskNone));
    auto semantics_equal = _builder.makeUintConstant(static_cast<uint32_t>(spv::MemorySemanticsMask::MaskNone));

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
            if (t->is_float()) {
                if (is_non_scalar_buffer) {
                    // Non-scalar buffer: ptr is uint32*, bitcast values to uint32
                    auto uint_type = _builder.makeUintType(32);
                    auto expected_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, expected);
                    auto desired_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, desired);
                    auto result = _builder.createOp(spv::Op::OpAtomicCompareExchange, uint_type,
                                                    {ptr, scope, semantics_equal, semantics, desired_uint, expected_uint});
                    id = _builder.createUnaryOp(spv::Op::OpBitcast, type, result);
                } else {
                    // Scalar buffer or shared memory: ptr is float*, use CAS loop
                    // with OpAtomicLoad + OpAtomicExchange to avoid pointer bitcast
                    // which crashes some Vulkan drivers (e.g., NVIDIA).
                    id = _emit_float_compare_exchange_cas_loop(ptr, expected, desired, type);
                }
            } else {
                id = _builder.createOp(spv::Op::OpAtomicCompareExchange, type,
                                       {ptr, scope, semantics_equal, semantics, desired, expected});
            }
            break;
        }
        case xir::AtomicOp::FETCH_ADD: {
            auto val = _emit_value(values[0]->value());
            if (t->is_float()) {
                if (!_use_native_float_atomics || (is_non_scalar_buffer && t->is_float32())) {
                    id = _emit_float_atomic_cas_loop(ptr, val, type, xir::AtomicOp::FETCH_ADD);
                } else {
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
                }
            } else {
                id = _builder.createOp(spv::Op::OpAtomicIAdd, type, {ptr, scope, semantics, val});
            }
            break;
        }
        case xir::AtomicOp::FETCH_SUB: {
            auto val = _emit_value(values[0]->value());
            if (t->is_float()) {
                if (!_use_native_float_atomics || (is_non_scalar_buffer && t->is_float32())) {
                    id = _emit_float_atomic_cas_loop(ptr, val, type, xir::AtomicOp::FETCH_SUB);
                } else {
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
                }
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
                if (!_use_native_float_atomics || (is_non_scalar_buffer && t->is_float32())) {
                    id = _emit_float_atomic_cas_loop(ptr, val, type, xir::AtomicOp::FETCH_MIN);
                } else {
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
                }
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
                if (!_use_native_float_atomics || (is_non_scalar_buffer && t->is_float32())) {
                    id = _emit_float_atomic_cas_loop(ptr, val, type, xir::AtomicOp::FETCH_MAX);
                } else {
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
                }
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

spv::Id SpirvCodegenEntry::_load_texture(spv::Id tex_var) noexcept {
    auto tex_type = _builder.getTypeId(tex_var);
    auto pointee_type = _builder.getContainedTypeId(tex_type);
    if (_builder.isArrayType(pointee_type)) {
        auto image_ptr = _create_access_chain(spv::StorageClass::UniformConstant, tex_var, {_builder.makeUintConstant(0u)});
        return _builder.createLoad(image_ptr, spv::NoPrecision);
    }
    return _builder.createLoad(tex_var, spv::NoPrecision);
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
            auto uint_type = _builder.makeUintType(32);
            auto bindless_array = _emit_value(inst->operand(0));
            auto nonuniform = !_uniformity.is_uniform(inst->operand(1));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto base_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto field_offset = _builder.makeUintConstant(is_2d ? 1u : 2u);
            auto slot_word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_offset, field_offset);
            auto bdls_ptr = _create_access_chain(_builder.getStorageClass(bindless_array), bindless_array, {_builder.makeUintConstant(0u), slot_word_offset}, nonuniform);
            auto packed = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            if (nonuniform) { _builder.addDecoration(packed, spv::Decoration::NonUniformEXT); }
            auto tex_idx = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, packed, _builder.makeUintConstant(0x0FFFFFFFu));
            auto heap_id = is_2d ? _tex2d_heap_id : _tex3d_heap_id;
            LUISA_ASSERT(heap_id != spv::NoResult, "SPIR-V {} texture heap not bound.", is_2d ? "2D" : "3D");
            auto tex_ptr = _create_access_chain(spv::StorageClass::UniformConstant, heap_id, {tex_idx}, nonuniform);
            auto tex = _builder.createLoad(tex_ptr, spv::NoPrecision);
            if (nonuniform) { _builder.addDecoration(tex, spv::Decoration::NonUniformEXT); }
            _builder.addCapability(spv::Capability::ImageQuery);
            spv::Id lod = has_level ? _ensure_type(_emit_value(inst->operand(2)), uint_type) : _builder.makeUintConstant(0u);
            id = _builder.createOp(spv::Op::OpImageQuerySizeLod, type, {tex, lod});
            break;
        }
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
            auto op = inst->op();
            auto is_2d = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto has_level = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER;
            auto has_grad = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                            op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER;
            auto has_grad_level = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                                  op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
                                  op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                                  op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto is_sampler_variant = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
                                      op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto uint_type = _builder.makeUintType(32);
            auto bindless_array = _emit_value(inst->operand(0));
            auto nonuniform = !_uniformity.is_uniform(inst->operand(1));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto base_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto field_offset = _builder.makeUintConstant(is_2d ? 1u : 2u);
            auto slot_word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_offset, field_offset);
            auto bdls_ptr = _create_access_chain(_builder.getStorageClass(bindless_array), bindless_array, {_builder.makeUintConstant(0u), slot_word_offset}, nonuniform);
            auto packed = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            if (nonuniform) { _builder.addDecoration(packed, spv::Decoration::NonUniformEXT); }
            auto tex_idx = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, packed, _builder.makeUintConstant(0x0FFFFFFFu));
            spv::Id samp_idx;
            if (is_sampler_variant) {
                auto filter = _emit_value(inst->operand(inst->operand_count() - 2));
                auto address = _emit_value(inst->operand(inst->operand_count() - 1));
                auto addr_mul = _builder.createBinOp(spv::Op::OpIMul, uint_type, address, _builder.makeUintConstant(4u));
                samp_idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, addr_mul, filter);
            } else {
                samp_idx = _builder.createBinOp(spv::Op::OpShiftRightLogical, uint_type, packed, _builder.makeUintConstant(28u));
            }
            auto heap_id = is_2d ? _tex2d_heap_id : _tex3d_heap_id;
            LUISA_ASSERT(heap_id != spv::NoResult, "SPIR-V {} texture heap not bound.", is_2d ? "2D" : "3D");
            auto tex_ptr = _create_access_chain(spv::StorageClass::UniformConstant, heap_id, {tex_idx}, nonuniform);
            auto image = _builder.createLoad(tex_ptr, spv::NoPrecision);
            if (nonuniform) { _builder.addDecoration(image, spv::Decoration::NonUniformEXT); }
            LUISA_ASSERT(!_properties.empty() && _properties[0].type == ShaderVariableType::SamplerHeap,
                         "SPIR-V sampler heap not bound.");
            auto sampler_heap = _property_ids[1];
            auto samp_nonuniform = nonuniform || (is_sampler_variant && (!_uniformity.is_uniform(inst->operand(inst->operand_count() - 2)) || !_uniformity.is_uniform(inst->operand(inst->operand_count() - 1))));
            auto samp_ptr = _create_access_chain(spv::StorageClass::UniformConstant, sampler_heap, {samp_idx}, samp_nonuniform);
            auto sampler = _builder.createLoad(samp_ptr, spv::NoPrecision);
            if (samp_nonuniform) { _builder.addDecoration(sampler, spv::Decoration::NonUniformEXT); }
            auto image_type = _builder.getTypeId(image);
            auto sampled_image_type = _builder.makeSampledImageType(image_type, "sampled_image");
            auto sampled_image = _builder.createOp(spv::Op::OpSampledImage, sampled_image_type, {image, sampler});
            if (nonuniform || samp_nonuniform) { _builder.addDecoration(sampled_image, spv::Decoration::NonUniformEXT); }
            spv::Builder::TextureParameters params{};
            params.sampler = sampled_image;
            size_t uv_op_idx = 2;
            params.coords = _emit_value(inst->operand(uv_op_idx));
            if (has_level) {
                params.lod = _emit_value(inst->operand(uv_op_idx + 1));
            } else if (has_grad) {
                params.gradX = _emit_value(inst->operand(uv_op_idx + 1));
                params.gradY = _emit_value(inst->operand(uv_op_idx + 2));
            } else if (has_grad_level) {
                params.gradX = _emit_value(inst->operand(uv_op_idx + 1));
                params.gradY = _emit_value(inst->operand(uv_op_idx + 2));
                params.lodClamp = _emit_value(inst->operand(uv_op_idx + 3));
            }
            // SPIR-V compute shaders cannot use implicit LOD; always use explicit LOD.
            // For non-level variants, createTextureCall will automatically add Lod=0 when noImplicitLod is true.
            auto no_implicit = true;
            id = _builder.createTextureCall(spv::NoPrecision, type, false, false, false, false, no_implicit, params, spv::ImageOperandsMask::MaskNone);
            break;
        }
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
            _builder.addExtension(spv::E_SPV_KHR_ray_query);
            _builder.addCapability(spv::Capability::RayQueryKHR);
            auto accel_ptr = _emit_value(inst->operand(0));
            auto accel = _builder.createLoad(accel_ptr, spv::NoPrecision);
            auto ray = _emit_value(inst->operand(1));
            spv::Id mask;
            size_t ray_operand_idx;
            if (inst->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
                inst->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR) {
                ray_operand_idx = 3;// operands: accel, ray, time, mask
            } else {
                ray_operand_idx = 2;// operands: accel, ray, mask
            }
            mask = _emit_value(inst->operand(ray_operand_idx));
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
            auto is_query_all = inst->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL || inst->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR;
            auto ray_flags = _builder.makeUintConstant(is_query_all ? 0u : 0x4u);
            _builder.createNoResultOp(spv::Op::OpRayQueryInitializeKHR, std::vector<spv::Id>{
                rq_var, accel, ray_flags, mask, ray_origin, ray_t_min, ray_dir, ray_t_max
            });
            id = rq_var;
            break;
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
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
            auto is_closest = inst->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST || inst->op() == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR;
            auto ray_flags = _builder.makeUintConstant(is_closest ? 0x201u : 0x20Du);
            _builder.createNoResultOp(spv::Op::OpRayQueryInitializeKHR, std::vector<spv::Id>{
                rq_var, accel, ray_flags, mask, ray_origin, ray_t_min, ray_dir, ray_t_max
            });
            _builder.createOp(spv::Op::OpRayQueryProceedKHR, _builder.makeBoolType(), std::vector<spv::Id>{rq_var});
            auto committed_intersection = _builder.makeIntConstant(1);
            auto committed_type = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTypeKHR, uint_type,
                std::vector<spv::IdImmediate>{
                    {true, rq_var},
                    {true, committed_intersection}
                });
            if (is_closest) {
                auto is_triangle_hit = _builder.createBinOp(spv::Op::OpIEqual, _builder.makeBoolType(),
                    committed_type, _builder.makeUintConstant(1u));
                auto result_type = type;
                auto result_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, result_type, "trace_result");
                auto zero_result = _builder.makeNullConstant(result_type);
                _builder.createStore(zero_result, result_var);
                auto &function = _builder.getBuildPoint()->getParent();
                auto true_block = new spv::Block(_builder.getUniqueId(), function);
                auto false_block = new spv::Block(_builder.getUniqueId(), function);
                auto merge_block = new spv::Block(_builder.getUniqueId(), function);
                auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
                selection_merge->reserveOperands(2);
                selection_merge->addIdOperand(merge_block->getId());
                selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
                _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));
                _builder.createConditionalBranch(is_triangle_hit, true_block, false_block);
                function.addBlock(true_block);
                _builder.setBuildPoint(true_block);
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
                function.addBlock(false_block);
                _builder.setBuildPoint(false_block);
                auto no_hit_inst = _builder.makeUintConstant(0xFFFFFFFFu);
                auto vec2_type = _builder.makeVectorType(float_type, 2);
                auto no_hit_bary = _builder.createCompositeConstruct(vec2_type, {_builder.makeFloatConstant(0.0f), _builder.makeFloatConstant(0.0f)});
                auto no_hit_result = _builder.createCompositeConstruct(result_type, {
                    no_hit_inst,
                    _builder.makeUintConstant(0u),
                    no_hit_bary,
                    _builder.makeFloatConstant(0.0f)
                });
                _builder.createStore(no_hit_result, result_var);
                _builder.createBranch(false, merge_block);
                function.addBlock(merge_block);
                _builder.setBuildPoint(merge_block);
                id = _builder.createLoad(result_var, spv::NoPrecision);
            } else {
                id = _builder.createBinOp(spv::Op::OpINotEqual, _builder.makeBoolType(),
                    committed_type, _builder.makeUintConstant(0u));
            }
            break;
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: {
            auto accel_ptr = _emit_value(inst->operand(0));
            auto it = _accel_instance_buffer_map.find(accel_ptr);
            LUISA_ASSERT(it != _accel_instance_buffer_map.end(), "SPIR-V ray_tracing_instance_transform: accel instance buffer not found.");
            auto instance_buffer = it->second;
            auto instance_index = _emit_value(inst->operand(1));
            if (!_uniformity.is_uniform(inst->operand(1))) { _builder.addDecoration(instance_index, spv::Decoration::NonUniformEXT); }
            auto uint_type = _builder.makeUintType(32);
            auto float_type = _builder.makeFloatType(32);
            auto float4_type = _builder.makeVectorType(float_type, 4);
            auto word_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, instance_index, _builder.makeUintConstant(16u));
            auto p0 = _emit_buffer_read_impl(instance_buffer, word_offset, Type::vector(Type::of<float>(), 4));
            auto p1_word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(4u));
            auto p1 = _emit_buffer_read_impl(instance_buffer, p1_word_offset, Type::vector(Type::of<float>(), 4));
            auto p2_word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(8u));
            auto p2 = _emit_buffer_read_impl(instance_buffer, p2_word_offset, Type::vector(Type::of<float>(), 4));
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
        default:
            LUISA_NOT_IMPLEMENTED("SPIR-V resource query op {}.", xir::to_string(inst->op()));
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit resource query.");
    _value_map.emplace(inst, id);
}

spv::Id SpirvCodegenEntry::_emit_buffer_read_impl(spv::Id buffer, spv::Id word_offset, const Type *elem_type, spv::MemoryAccessMask memory_access,
                                                   spv::Id byte_in_word) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto spv_type = _convert_type(elem_type, Usage::READ);
    auto word_count = elem_type->size() / 4u;
    auto is_subword_vector = elem_type->is_vector() && elem_type->element()->size() < 4u;
    if (is_subword_vector) {
        // Vector of sub-word elements: read each component individually so that
        // arbitrary byte alignment (byte_in_word) is honored correctly.
        auto comp_elem = elem_type->element();
        auto comp_type = _convert_type(comp_elem, Usage::READ);
        auto dim = elem_type->dimension();
        auto comp_size = comp_elem->size();
        auto biw = byte_in_word;
        if (biw == spv::NoResult) { biw = _builder.makeUintConstant(0u); }
        auto base_byte_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, word_offset, _builder.makeUintConstant(4u));
        base_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset, biw);
        std::vector<spv::Id> comps;
        comps.reserve(dim);
        for (uint i = 0u; i < dim; ++i) {
            auto comp_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset,
                                                         _builder.makeUintConstant(static_cast<uint32_t>(i * comp_size)));
            auto comp_word_offset = _builder.createBinOp(spv::Op::OpUDiv, uint_type, comp_byte_offset, _builder.makeUintConstant(4u));
            auto comp_biw = _builder.createBinOp(spv::Op::OpUMod, uint_type, comp_byte_offset, _builder.makeUintConstant(4u));
            comps.push_back(_emit_buffer_read_impl(buffer, comp_word_offset, comp_elem, memory_access, comp_biw));
        }
        return _builder.createCompositeConstruct(spv_type, comps);
    }
    if (elem_type->is_structure() && elem_type->size() < 4u) {
        // Sub-word structure (e.g. FP16Quantized): read each member with per-member
        // byte alignment so the recursive scalar sub-word path handles shifts/masks.
        auto biw = byte_in_word;
        if (biw == spv::NoResult) { biw = _builder.makeUintConstant(0u); }
        auto base_byte_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, word_offset, _builder.makeUintConstant(4u));
        base_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset, biw);
        std::vector<spv::Id> fields;
        size_t struct_offset = 0;
        for (auto member : elem_type->members()) {
            auto align = member->alignment();
            struct_offset = (struct_offset + align - 1) & ~(align - 1);
            auto member_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset,
                                                           _builder.makeUintConstant(static_cast<uint32_t>(struct_offset)));
            auto member_word_offset = _builder.createBinOp(spv::Op::OpUDiv, uint_type, member_byte_offset, _builder.makeUintConstant(4u));
            auto member_biw = _builder.createBinOp(spv::Op::OpUMod, uint_type, member_byte_offset, _builder.makeUintConstant(4u));
            fields.push_back(_emit_buffer_read_impl(buffer, member_word_offset, member, memory_access, member_biw));
            struct_offset += member->size();
        }
        return _builder.createCompositeConstruct(spv_type, fields);
    }
    if (word_count == 0u) {
        // Sub-word scalar type (e.g., bool, half, fp8): read a full word and extract
        auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), word_offset});
        auto raw = _builder.createLoad(ptr, spv::NoPrecision, memory_access);
        if (byte_in_word != spv::NoResult) {
            auto bit_shift = _builder.createBinOp(spv::Op::OpIMul, uint_type, byte_in_word, _builder.makeUintConstant(8u));
            raw = _builder.createBinOp(spv::Op::OpShiftRightLogical, uint_type, raw, bit_shift);
        }
        if (elem_type->is_bool()) {
            // bool: compare low bit with 0
            return _builder.createBinOp(spv::Op::OpINotEqual, spv_type, raw, _builder.makeUintConstant(0u));
        }
        if (elem_type->is_float8()) {
            // FP8: truncate to uint8, then bitcast to fp8
            auto uint8_type = _builder.makeUintType(8);
            auto u8 = _builder.createUnaryOp(spv::Op::OpUConvert, uint8_type, raw);
            return _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, u8);
        }
        // Other sub-word integer types: truncate
        auto bit_width = static_cast<int32_t>(elem_type->size() * 8);
        auto is_signed = elem_type->is_int();
        auto trunc_type = _builder.makeIntegerType(bit_width, is_signed);
        auto truncated = _builder.createUnaryOp(is_signed ? spv::Op::OpSConvert : spv::Op::OpUConvert, trunc_type, raw);
        if (trunc_type != spv_type) {
            return _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, truncated);
        }
        return truncated;
    }
    if (word_count == 1u) {
        auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), word_offset});
        auto raw = _builder.createLoad(ptr, spv::NoPrecision, memory_access);
        if (spv_type != uint_type) {
            if (_builder.isBoolType(spv_type) || (_builder.isVectorType(spv_type) && _builder.isBoolType(_builder.getScalarTypeId(spv_type)))) {
                // Convert uint to bool or bool vector
                if (_builder.isBoolType(spv_type)) {
                    return _builder.createBinOp(spv::Op::OpINotEqual, spv_type, raw, _builder.makeUintConstant(0u));
                }
                // Bool vector: extract each bit
                auto dim = _builder.getNumTypeComponents(spv_type);
                std::vector<spv::Id> comps;
                comps.reserve(dim);
                for (uint i = 0u; i < dim; ++i) {
                    auto bit = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, raw, _builder.makeUintConstant(1u << i));
                    auto cmp = _builder.createBinOp(spv::Op::OpINotEqual, _builder.makeBoolType(), bit, _builder.makeUintConstant(0u));
                    comps.push_back(cmp);
                }
                return _builder.createCompositeConstruct(spv_type, comps);
            }
            return _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, raw);
        }
        return raw;
    }
    if (elem_type->is_vector()) {
        auto comp_type = _convert_type(elem_type->element(), Usage::READ);
        auto dim = elem_type->dimension();
        auto comp_size = elem_type->element()->size();
        uint comp_word_count = std::max<uint>(static_cast<uint>(comp_size / 4u), 1u);
        if (comp_word_count == 1u && comp_size < 4u) {
            // Sub-word component vector (e.g., half3): load all words and extract via shift+mask
            auto comp_bit_width = static_cast<uint32_t>(comp_size * 8);
            auto total_bit_width = comp_bit_width * dim;
            auto total_words = (total_bit_width + 31u) / 32u;
            std::vector<spv::Id> words;
            words.reserve(total_words);
            for (uint w = 0u; w < total_words; ++w) {
                auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(w));
                auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), idx});
                words.push_back(_builder.createLoad(ptr, spv::NoPrecision, memory_access));
            }
            std::vector<spv::Id> comps;
            comps.reserve(dim);
            for (uint i = 0u; i < dim; ++i) {
                auto bit_offset = i * comp_bit_width;
                auto word_idx = bit_offset / 32u;
                auto bit_in_word = bit_offset % 32u;
                spv::Id comp_raw = words[word_idx];
                if (bit_in_word > 0u) {
                    comp_raw = _builder.createBinOp(spv::Op::OpShiftRightLogical, uint_type, comp_raw, _builder.makeUintConstant(bit_in_word));
                }
                auto comp_mask = (1ull << comp_bit_width) - 1ull;
                comp_raw = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, comp_raw, _builder.makeUintConstant(static_cast<uint32_t>(comp_mask)));
                // Convert raw uint to component type
                auto comp_elem = elem_type->element();
                if (comp_elem->is_bool()) {
                    comps.push_back(_builder.createBinOp(spv::Op::OpINotEqual, comp_type, comp_raw, _builder.makeUintConstant(0u)));
                } else {
                    auto trunc_type = _builder.makeIntegerType(static_cast<int32_t>(comp_bit_width), comp_elem->is_int());
                    auto truncated = _builder.createUnaryOp(comp_elem->is_int() ? spv::Op::OpSConvert : spv::Op::OpUConvert, trunc_type, comp_raw);
                    comps.push_back(trunc_type == comp_type ? truncated : _builder.createUnaryOp(spv::Op::OpBitcast, comp_type, truncated));
                }
            }
            return _builder.createCompositeConstruct(spv_type, comps);
        }
        std::vector<spv::Id> comps;
        comps.reserve(dim);
        for (uint i = 0u; i < dim; ++i) {
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i * comp_word_count));
            comps.push_back(_emit_buffer_read_impl(buffer, idx, elem_type->element(), memory_access));
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
        for (uint i = 0u; i < dim; ++i) {
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i * col_word_count));
            cols.push_back(_emit_buffer_read_impl(buffer, idx, col_type, memory_access));
        }
        return _builder.createCompositeConstruct(spv_type, cols);
    }
    if (elem_type->is_structure()) {
        std::vector<spv::Id> fields;
        size_t struct_offset = 0;
        for (auto member : elem_type->members()) {
            auto align = member->alignment();
            struct_offset = (struct_offset + align - 1) & ~(align - 1);
            auto member_size = member->size();
            auto member_word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset,
                                                            _builder.makeUintConstant(static_cast<uint32_t>(struct_offset / 4)));
            if (member_size < 4u) {
                // Sub-word member (scalar or vector): read the containing word, shift and mask
                auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), member_word_offset});
                auto raw = _builder.createLoad(ptr, spv::NoPrecision, memory_access);
                auto byte_shift = struct_offset % 4u;
                if (byte_shift > 0u) {
                    raw = _builder.createBinOp(spv::Op::OpShiftRightLogical, uint_type, raw, _builder.makeUintConstant(static_cast<uint32_t>(byte_shift * 8)));
                }
                auto bit_width = static_cast<uint32_t>(member_size * 8);
                if (bit_width < 32u) {
                    auto mask = (1ull << bit_width) - 1ull;
                    raw = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, raw, _builder.makeUintConstant(static_cast<uint32_t>(mask)));
                }
                auto spv_member_type = _convert_type(member, Usage::READ);
                if (member->is_vector()) {
                    // Decompose sub-word vector from extracted raw value
                    auto comp_type = member->element();
                    auto dim = member->dimension();
                    auto comp_size = member_size / dim;
                    auto spv_comp_type = _convert_type(comp_type, Usage::READ);
                    std::vector<spv::Id> comps;
                    comps.reserve(dim);
                    for (uint i = 0u; i < dim; ++i) {
                        auto comp_byte_shift = i * comp_size;
                        auto comp_bit_shift = comp_byte_shift * 8;
                        spv::Id comp_raw = raw;
                        if (comp_bit_shift > 0u) {
                            comp_raw = _builder.createBinOp(spv::Op::OpShiftRightLogical, uint_type, comp_raw, _builder.makeUintConstant(comp_bit_shift));
                        }
                        auto comp_bit_width = static_cast<uint32_t>(comp_size * 8);
                        if (comp_bit_width < 32u) {
                            auto comp_mask = (1ull << comp_bit_width) - 1ull;
                            comp_raw = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, comp_raw, _builder.makeUintConstant(static_cast<uint32_t>(comp_mask)));
                        }
                        if (comp_type->is_bool()) {
                            comps.push_back(_builder.createBinOp(spv::Op::OpINotEqual, spv_comp_type, comp_raw, _builder.makeUintConstant(0u)));
                        } else {
                            auto is_signed = comp_type->is_int();
                            auto trunc_type = _builder.makeIntegerType(static_cast<int32_t>(comp_bit_width), is_signed);
                            auto truncated = _builder.createUnaryOp(is_signed ? spv::Op::OpSConvert : spv::Op::OpUConvert, trunc_type, comp_raw);
                            comps.push_back(trunc_type == spv_comp_type ? truncated : _builder.createUnaryOp(spv::Op::OpBitcast, spv_comp_type, truncated));
                        }
                    }
                    fields.push_back(_builder.createCompositeConstruct(spv_member_type, comps));
                } else {
                    // Sub-word scalar
                    if (member->is_bool()) {
                        fields.push_back(_builder.createBinOp(spv::Op::OpINotEqual, spv_member_type, raw, _builder.makeUintConstant(0u)));
                    } else if (member->is_float8()) {
                        auto uint8_type = _builder.makeUintType(8);
                        auto u8 = _builder.createUnaryOp(spv::Op::OpUConvert, uint8_type, raw);
                        fields.push_back(_builder.createUnaryOp(spv::Op::OpBitcast, spv_member_type, u8));
                    } else {
                        auto bit_width = static_cast<int32_t>(member_size * 8u);
                        auto is_signed = member->is_int();
                        auto trunc_type = _builder.makeIntegerType(bit_width, is_signed);
                        auto truncated = _builder.createUnaryOp(is_signed ? spv::Op::OpSConvert : spv::Op::OpUConvert, trunc_type, raw);
                        fields.push_back(trunc_type == spv_member_type ? truncated : _builder.createUnaryOp(spv::Op::OpBitcast, spv_member_type, truncated));
                    }
                }
            } else {
                fields.push_back(_emit_buffer_read_impl(buffer, member_word_offset, member, memory_access));
            }
            struct_offset += member_size;
        }
        return _builder.createCompositeConstruct(spv_type, fields);
    }
    if (elem_type->is_array()) {
        auto elem = elem_type->element();
        auto dim = elem_type->dimension();
        auto elem_word_count = elem->size() / 4u;
        std::vector<spv::Id> elems;
        elems.reserve(dim);
        for (uint i = 0u; i < dim; ++i) {
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset,
                                            _builder.makeUintConstant(i * elem_word_count));
            elems.push_back(_emit_buffer_read_impl(buffer, idx, elem, memory_access));
        }
        return _builder.createCompositeConstruct(spv_type, elems);
    }
    if (word_count > 1u) {
        // Multi-word scalar (slong, ulong, double): load N uint32 words and bitcast
        auto uvec_type = _builder.makeVectorType(uint_type, static_cast<int>(word_count));
        std::vector<spv::Id> words;
        words.reserve(word_count);
        for (uint i = 0u; i < word_count; ++i) {
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i));
            auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), idx});
            words.push_back(_builder.createLoad(ptr, spv::NoPrecision, memory_access));
        }
        auto vec = _builder.createCompositeConstruct(uvec_type, words);
        return _builder.createUnaryOp(spv::Op::OpBitcast, spv_type, vec);
    }
    LUISA_NOT_IMPLEMENTED("SPIR-V buffer read for type {}.", elem_type->description());
}

spv::Id SpirvCodegenEntry::_emit_buffer_read(spv::Id buffer, spv::Id index, const Type *read_type, const Type *buffer_type, bool index_is_word_offset, spv::MemoryAccessMask memory_access) noexcept {
    auto uint_type = _builder.makeUintType(32);
    if (buffer_type != nullptr && buffer_type->is_buffer() && buffer_type->element() != nullptr && !_needs_atomic_buffer_types.contains(buffer_type) && !_type_contains_bool(buffer_type->element())) {
        // Typed buffer: direct element access via SPIR-V type system.
        // Works for scalar, vector, and matrix element types.
        auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), index});
        auto loaded = _builder.createLoad(ptr, spv::NoPrecision, memory_access);
        auto plain_type = _convert_type(read_type, Usage::READ);
        auto loaded_type = _builder.getTypeId(loaded);
        if (loaded_type != plain_type) {
            loaded = _builder.createUnaryOp(spv::Op::OpCopyLogical, plain_type, loaded);
        }
        return loaded;
    }
    // Byte buffer or bindless: word-level access
    auto word_count = read_type->size() / 4u;
    auto is_subword_vector = read_type->is_vector() && read_type->element()->size() < 4u;
    spv::Id word_offset;
    spv::Id byte_in_word = _builder.makeUintConstant(0u);
    if (index_is_word_offset) {
        // index is already a word offset (e.g., from BYTE_BUFFER_READ)
        word_offset = index;
    } else if (word_count == 0u || is_subword_vector) {
        // Sub-word scalar or vector of sub-word elements: index is a byte offset
        word_offset = _builder.createBinOp(spv::Op::OpUDiv, uint_type, index, _builder.makeUintConstant(4u));
        byte_in_word = _builder.createBinOp(spv::Op::OpUMod, uint_type, index, _builder.makeUintConstant(4u));
    } else {
        word_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, index, _builder.makeUintConstant(word_count));
    }
    return _emit_buffer_read_impl(buffer, word_offset, read_type, memory_access, byte_in_word);
}

void SpirvCodegenEntry::_emit_buffer_write_impl(spv::Id buffer, spv::Id word_offset, spv::Id value, const Type *elem_type, spv::MemoryAccessMask memory_access,
                                                 spv::Id byte_in_word) noexcept {
    auto uint_type = _builder.makeUintType(32);
    auto spv_type = _convert_type(elem_type, Usage::READ);
    auto word_count = elem_type->size() / 4u;
    auto is_subword_vector = elem_type->is_vector() && elem_type->element()->size() < 4u;
    if (is_subword_vector) {
        // Vector of sub-word elements: write each component individually so that
        // arbitrary byte alignment (byte_in_word) is honored correctly.
        auto comp_elem = elem_type->element();
        auto comp_type = _convert_type(comp_elem, Usage::READ);
        auto dim = elem_type->dimension();
        auto comp_size = comp_elem->size();
        auto biw = byte_in_word;
        if (biw == spv::NoResult) { biw = _builder.makeUintConstant(0u); }
        auto base_byte_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, word_offset, _builder.makeUintConstant(4u));
        base_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset, biw);
        for (uint i = 0u; i < dim; ++i) {
            auto comp = _builder.createCompositeExtract(value, comp_type, i);
            auto comp_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset,
                                                         _builder.makeUintConstant(static_cast<uint32_t>(i * comp_size)));
            auto comp_word_offset = _builder.createBinOp(spv::Op::OpUDiv, uint_type, comp_byte_offset, _builder.makeUintConstant(4u));
            auto comp_biw = _builder.createBinOp(spv::Op::OpUMod, uint_type, comp_byte_offset, _builder.makeUintConstant(4u));
            _emit_buffer_write_impl(buffer, comp_word_offset, comp, comp_elem, memory_access, comp_biw);
        }
        return;
    }
    if (elem_type->is_structure() && elem_type->size() < 4u) {
        // Sub-word structure (e.g. FP16Quantized): write each member with per-member
        // byte alignment so the recursive scalar sub-word path handles shifts/masks.
        auto biw = byte_in_word;
        if (biw == spv::NoResult) { biw = _builder.makeUintConstant(0u); }
        auto base_byte_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, word_offset, _builder.makeUintConstant(4u));
        base_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset, biw);
        size_t struct_offset = 0;
        auto members = elem_type->members();
        for (auto j = 0u; j < members.size(); ++j) {
            auto member = members[j];
            auto align = member->alignment();
            struct_offset = (struct_offset + align - 1) & ~(align - 1);
            auto member_spv_type = _convert_type(member, Usage::READ);
            auto field_val = _builder.createCompositeExtract(value, member_spv_type, j);
            auto member_byte_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_byte_offset,
                                                           _builder.makeUintConstant(static_cast<uint32_t>(struct_offset)));
            auto member_word_offset = _builder.createBinOp(spv::Op::OpUDiv, uint_type, member_byte_offset, _builder.makeUintConstant(4u));
            auto member_biw = _builder.createBinOp(spv::Op::OpUMod, uint_type, member_byte_offset, _builder.makeUintConstant(4u));
            _emit_buffer_write_impl(buffer, member_word_offset, field_val, member, memory_access, member_biw);
            struct_offset += member->size();
        }
        return;
    }
    if (word_count == 0u) {
        // Sub-word scalar type (e.g., bool, half, fp8): read-modify-write a full word
        auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), word_offset});
        spv::Id store_val = value;
        if (elem_type->is_bool()) {
            // bool -> uint: select 1u or 0u
            store_val = _builder.createOp(spv::Op::OpSelect, uint_type, {value, _builder.makeUintConstant(1u), _builder.makeUintConstant(0u)});
        } else if (elem_type->is_float8()) {
            // FP8: bitcast to uint8, then zero-extend to uint32
            auto uint8_type = _builder.makeUintType(8);
            store_val = _builder.createUnaryOp(spv::Op::OpBitcast, uint8_type, value);
            store_val = _builder.createUnaryOp(spv::Op::OpUConvert, uint_type, store_val);
        } else if (elem_type->is_float()) {
            // Float16: bitcast to same-width uint, then extend to uint32
            auto bit_width = static_cast<int32_t>(elem_type->size() * 8u);
            auto bit_type = _builder.makeIntegerType(bit_width, false);
            store_val = _builder.createUnaryOp(spv::Op::OpBitcast, bit_type, value);
            store_val = _builder.createUnaryOp(spv::Op::OpUConvert, uint_type, store_val);
        } else {
            // Extend sub-word integer to 32 bits
            store_val = _builder.createUnaryOp(elem_type->is_int() ? spv::Op::OpSConvert : spv::Op::OpUConvert, uint_type, value);
        }
        auto bit_width = static_cast<uint32_t>(elem_type->size() * 8u);
        spv::Id bit_shift;
        if (byte_in_word == spv::NoResult) {
            bit_shift = _builder.makeUintConstant(0u);
        } else {
            bit_shift = _builder.createBinOp(spv::Op::OpIMul, uint_type, byte_in_word, _builder.makeUintConstant(8u));
        }
        auto raw = _builder.createLoad(ptr, spv::NoPrecision, memory_access);
        auto shifted_mask = _builder.createBinOp(spv::Op::OpShiftLeftLogical, uint_type, _builder.makeUintConstant(static_cast<uint32_t>((1ull << bit_width) - 1ull)), bit_shift);
        auto clear_mask = _builder.createUnaryOp(spv::Op::OpNot, uint_type, shifted_mask);
        raw = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, raw, clear_mask);
        store_val = _builder.createBinOp(spv::Op::OpShiftLeftLogical, uint_type, store_val, bit_shift);
        raw = _builder.createBinOp(spv::Op::OpBitwiseOr, uint_type, raw, store_val);
        _builder.createStore(raw, ptr, memory_access);
        return;
    }
    if (word_count == 1u) {
        auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), word_offset});
        auto ptr_type = _builder.getTypeId(ptr);
        auto pointee_type = _builder.getContainedTypeId(ptr_type);
        auto store_val = value;
        auto val_type = _builder.getTypeId(value);
        if (val_type != pointee_type) {
            if (_builder.isBoolType(spv_type) || (_builder.isVectorType(spv_type) && _builder.isBoolType(_builder.getScalarTypeId(spv_type)))) {
                // Convert bool or bool vector to uint via select/shift/or
                if (_builder.isBoolType(spv_type)) {
                    store_val = _builder.createOp(spv::Op::OpSelect, pointee_type, {value, _builder.makeUintConstant(1u), _builder.makeUintConstant(0u)});
                } else {
                    // Bool vector: convert each component to a bit and pack
                    auto dim = _builder.getNumTypeComponents(spv_type);
                    store_val = _builder.makeUintConstant(0u);
                    for (uint i = 0u; i < dim; ++i) {
                        auto comp = _builder.createCompositeExtract(value, _builder.makeBoolType(), i);
                        auto bit = _builder.createOp(spv::Op::OpSelect, pointee_type, {comp, _builder.makeUintConstant(1u << i), _builder.makeUintConstant(0u)});
                        store_val = _builder.createBinOp(spv::Op::OpBitwiseOr, pointee_type, store_val, bit);
                    }
                }
            } else {
                store_val = _builder.createUnaryOp(spv::Op::OpBitcast, pointee_type, value);
            }
        }
        _builder.createStore(store_val, ptr, memory_access);
        return;
    }
    if (elem_type->is_vector()) {
        auto comp_type = _convert_type(elem_type->element(), Usage::READ);
        auto dim = elem_type->dimension();
        auto comp_size = elem_type->element()->size();
        uint comp_word_count = std::max<uint>(static_cast<uint>(comp_size / 4u), 1u);
        if (comp_word_count == 1u && comp_size < 4u) {
            // Sub-word component vector (e.g., half3): read-modify-write each word
            auto comp_bit_width = static_cast<uint32_t>(comp_size * 8);
            auto total_bit_width = comp_bit_width * dim;
            auto total_words = (total_bit_width + 31u) / 32u;
            for (uint w = 0u; w < total_words; ++w) {
                auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(w));
                auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), idx});
                auto raw = _builder.createLoad(ptr, spv::NoPrecision, memory_access);
                // Build the new value for this word by overlaying components
                spv::Id word_val = raw;
                for (uint i = 0u; i < dim; ++i) {
                    auto bit_offset = i * comp_bit_width;
                    auto word_idx = bit_offset / 32u;
                    if (word_idx != w) continue;
                    auto bit_in_word = bit_offset % 32u;
                    auto comp = _builder.createCompositeExtract(value, comp_type, i);
                    // Convert component to uint
                    spv::Id comp_uint;
                    auto comp_elem = elem_type->element();
                    if (comp_elem->is_bool()) {
                        comp_uint = _builder.createOp(spv::Op::OpSelect, uint_type, {comp, _builder.makeUintConstant(1u), _builder.makeUintConstant(0u)});
                    } else if (comp_elem->is_float()) {
                        auto bit_type = _builder.makeIntegerType(static_cast<int32_t>(comp_bit_width), false);
                        comp_uint = _builder.createUnaryOp(spv::Op::OpBitcast, bit_type, comp);
                        comp_uint = _builder.createUnaryOp(spv::Op::OpUConvert, uint_type, comp_uint);
                    } else {
                        comp_uint = _builder.createUnaryOp(comp_elem->is_int() ? spv::Op::OpSConvert : spv::Op::OpUConvert, uint_type, comp);
                    }
                    if (bit_in_word > 0u) {
                        comp_uint = _builder.createBinOp(spv::Op::OpShiftLeftLogical, uint_type, comp_uint, _builder.makeUintConstant(bit_in_word));
                    }
                    // Clear old bits and OR in new bits
                    auto comp_mask = ((1ull << comp_bit_width) - 1ull) << bit_in_word;
                    auto clear_mask = _builder.makeUintConstant(static_cast<uint32_t>(~comp_mask));
                    word_val = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, word_val, clear_mask);
                    word_val = _builder.createBinOp(spv::Op::OpBitwiseOr, uint_type, word_val, comp_uint);
                }
                _builder.createStore(word_val, ptr, memory_access);
            }
            return;
        }
        for (uint i = 0u; i < dim; ++i) {
            auto comp = _builder.createCompositeExtract(value, comp_type, i);
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i * comp_word_count));
            _emit_buffer_write_impl(buffer, idx, comp, elem_type->element(), memory_access);
        }
        return;
    }
    if (elem_type->is_matrix()) {
        auto elem = elem_type->element();
        auto dim = elem_type->dimension();
        auto col_type = Type::vector(elem, dim);
        auto col_word_count = col_type->size() / 4u;
        for (uint i = 0u; i < dim; ++i) {
            auto col = _builder.createCompositeExtract(value, _convert_type(col_type, Usage::READ), i);
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i * col_word_count));
            _emit_buffer_write_impl(buffer, idx, col, col_type, memory_access);
        }
        return;
    }
    if (elem_type->is_structure()) {
        size_t struct_offset = 0;
        auto members = elem_type->members();
        for (auto j = 0u; j < members.size(); ++j) {
            auto member = members[j];
            auto align = member->alignment();
            struct_offset = (struct_offset + align - 1) & ~(align - 1);
            auto member_size = member->size();
            auto member_word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset,
                                                            _builder.makeUintConstant(static_cast<uint32_t>(struct_offset / 4)));
            auto field_val = _builder.createCompositeExtract(value, _convert_type(member, Usage::READ), j);
            if (member_size < 4u) {
                // Sub-word member: read-modify-write to avoid corrupting neighboring fields
                auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), member_word_offset});
                auto raw = _builder.createLoad(ptr, spv::NoPrecision, memory_access);
                auto byte_shift = struct_offset % 4u;
                auto total_bit_shift = byte_shift * 8;
                auto total_bit_width = static_cast<uint32_t>(member_size * 8);
                // Convert field_val to a uint with the member's bytes in the low bits
                spv::Id field_uint = spv::NoResult;
                if (member->is_vector()) {
                    auto comp_type = member->element();
                    auto dim = member->dimension();
                    auto comp_size = member_size / dim;
                    field_uint = _builder.makeUintConstant(0u);
                    for (uint i = 0u; i < dim; ++i) {
                        auto comp = _builder.createCompositeExtract(field_val, _convert_type(comp_type, Usage::READ), i);
                        spv::Id comp_uint = spv::NoResult;
                        if (comp_type->is_bool()) {
                            comp_uint = _builder.createOp(spv::Op::OpSelect, uint_type, {comp, _builder.makeUintConstant(1u), _builder.makeUintConstant(0u)});
                        } else if (comp_type->is_float8()) {
                            auto uint8_type = _builder.makeUintType(8);
                            comp_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint8_type, comp);
                        } else if (comp_type->is_float()) {
                           auto bit_width = static_cast<int32_t>(comp_size * 8u);
                            auto bit_type = _builder.makeIntegerType(bit_width, false);
                            comp_uint = _builder.createUnaryOp(spv::Op::OpBitcast, bit_type, comp);
                            comp_uint = _builder.createUnaryOp(spv::Op::OpUConvert, uint_type, comp_uint);
                        } else {
                            comp_uint = _builder.createUnaryOp(comp_type->is_int() ? spv::Op::OpSConvert : spv::Op::OpUConvert, uint_type, comp);
                        }
                        auto comp_bit_shift = i * comp_size * 8;
                        if (comp_bit_shift > 0u) {
                            comp_uint = _builder.createBinOp(spv::Op::OpShiftLeftLogical, uint_type, comp_uint, _builder.makeUintConstant(comp_bit_shift));
                        }
                        field_uint = _builder.createBinOp(spv::Op::OpBitwiseOr, uint_type, field_uint, comp_uint);
                    }
                } else {
                    if (member->is_bool()) {
                        field_uint = _builder.createOp(spv::Op::OpSelect, uint_type, {field_val, _builder.makeUintConstant(1u), _builder.makeUintConstant(0u)});
                    } else if (member->is_float8()) {
                        auto uint8_type = _builder.makeUintType(8);
                        field_uint = _builder.createUnaryOp(spv::Op::OpBitcast, uint8_type, field_val);
                    } else if (member->is_float()) {
                        auto bit_width = static_cast<int32_t>(member_size * 8u);
                        auto bit_type = _builder.makeIntegerType(bit_width, false);
                        field_uint = _builder.createUnaryOp(spv::Op::OpBitcast, bit_type, field_val);
                        field_uint = _builder.createUnaryOp(spv::Op::OpUConvert, uint_type, field_uint);
                    } else {
                        field_uint = _builder.createUnaryOp(member->is_int() ? spv::Op::OpSConvert : spv::Op::OpUConvert, uint_type, field_val);
                    }
                }
                // Shift to position
                if (total_bit_shift > 0u) {
                    field_uint = _builder.createBinOp(spv::Op::OpShiftLeftLogical, uint_type, field_uint, _builder.makeUintConstant(total_bit_shift));
                }
                // Clear old bits: mask keeps everything except the member's bits
                uint64_t member_mask = total_bit_width < 64u ? ((uint64_t{1} << total_bit_width) - 1u) : ~0ull;
                member_mask <<= total_bit_shift;
                auto clear_mask_val = static_cast<uint32_t>(~member_mask);
                auto clear_mask = _builder.makeUintConstant(clear_mask_val);
                raw = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, raw, clear_mask);
                raw = _builder.createBinOp(spv::Op::OpBitwiseOr, uint_type, raw, field_uint);
                _builder.createStore(raw, ptr, memory_access);
            } else {
                _emit_buffer_write_impl(buffer, member_word_offset, field_val, member, memory_access);
            }
            struct_offset += member_size;
        }
        return;
    }
    if (elem_type->is_array()) {
        auto elem = elem_type->element();
        auto dim = elem_type->dimension();
        auto elem_word_count = elem->size() / 4u;
        for (uint i = 0u; i < dim; ++i) {
            auto elem_val = _builder.createCompositeExtract(value, _convert_type(elem, Usage::READ), i);
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset,
                                            _builder.makeUintConstant(i * elem_word_count));
            _emit_buffer_write_impl(buffer, idx, elem_val, elem, memory_access);
        }
        return;
    }
    if (word_count > 1u) {
        // Multi-word scalar (slong, ulong, double): bitcast to uvec and store each word
        auto uvec_type = _builder.makeVectorType(uint_type, static_cast<int>(word_count));
        auto vec = _builder.createUnaryOp(spv::Op::OpBitcast, uvec_type, value);
        for (uint i = 0u; i < word_count; ++i) {
            auto word = _builder.createCompositeExtract(vec, uint_type, i);
            auto idx = _builder.createBinOp(spv::Op::OpIAdd, uint_type, word_offset, _builder.makeUintConstant(i));
            auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), idx});
            _builder.createStore(word, ptr, memory_access);
        }
        return;
    }
    LUISA_NOT_IMPLEMENTED("SPIR-V buffer write for type {}.", elem_type->description());
}

void SpirvCodegenEntry::_emit_buffer_write(spv::Id buffer, spv::Id index, spv::Id value, const Type *value_type, const Type *buffer_type, bool index_is_word_offset, spv::MemoryAccessMask memory_access) noexcept {
    auto uint_type = _builder.makeUintType(32);
    if (buffer_type != nullptr && buffer_type->is_buffer() && buffer_type->element() != nullptr && !_needs_atomic_buffer_types.contains(buffer_type) && !_type_contains_bool(buffer_type->element())) {
        // Typed buffer: direct element access via SPIR-V type system.
        // Works for scalar, vector, and matrix element types.
        auto ptr = _create_access_chain(_builder.getStorageClass(buffer), buffer, {_builder.makeUintConstant(0u), index});
        auto ptr_type = _builder.getTypeId(ptr);
        auto pointee_type = _builder.getContainedTypeId(ptr_type);
        auto val_type = _builder.getTypeId(value);
        if (pointee_type != val_type) {
            value = _builder.createUnaryOp(spv::Op::OpCopyLogical, pointee_type, value);
        }
        _builder.createStore(value, ptr, memory_access);
        return;
    }
    // Byte buffer or bindless: word-level access
    auto word_count = value_type->size() / 4u;
    auto is_subword_vector = value_type->is_vector() && value_type->element()->size() < 4u;
    spv::Id word_offset;
    spv::Id byte_in_word = _builder.makeUintConstant(0u);
    if (index_is_word_offset) {
        // index is already a word offset (e.g., from BYTE_BUFFER_WRITE)
        word_offset = index;
    } else if (word_count == 0u || is_subword_vector) {
        // Sub-word scalar or vector of sub-word elements: index is a byte offset
        word_offset = _builder.createBinOp(spv::Op::OpUDiv, uint_type, index, _builder.makeUintConstant(4u));
        byte_in_word = _builder.createBinOp(spv::Op::OpUMod, uint_type, index, _builder.makeUintConstant(4u));
    } else {
        word_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, index, _builder.makeUintConstant(word_count));
    }
    _emit_buffer_write_impl(buffer, word_offset, value, value_type, memory_access, byte_in_word);
}

void SpirvCodegenEntry::_emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept {
    auto type = _convert_type(inst->type(), Usage::READ);
    spv::Id id = spv::NoResult;
    auto uint_type = _builder.makeUintType(32);

    switch (inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            id = _emit_buffer_read(buffer, index, inst->type(), inst->operand(0)->type());
            break;
        }
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            id = _emit_buffer_read(buffer, index, inst->type(), inst->operand(0)->type(), false, spv::MemoryAccessMask::Volatile);
            break;
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto read_type = inst->type();
            auto is_subword_vector = read_type->is_vector() && read_type->element()->size() < 4u;
            if (read_type->size() < 4u || is_subword_vector) {
                id = _emit_buffer_read(buffer, byte_index, read_type, nullptr, false);
            } else {
                auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
                id = _emit_buffer_read(buffer, word_index, read_type, nullptr, true);
            }
            break;
        }
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto read_type = inst->type();
            auto is_subword_vector = read_type->is_vector() && read_type->element()->size() < 4u;
            if (read_type->size() < 4u || is_subword_vector) {
                id = _emit_buffer_read(buffer, byte_index, read_type, nullptr, false, spv::MemoryAccessMask::Volatile);
            } else {
                auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
                id = _emit_buffer_read(buffer, word_index, read_type, nullptr, true, spv::MemoryAccessMask::Volatile);
            }
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
                    _builder.addCapability(spv::Capability::StorageImageReadWithoutFormat);
                }
                id = _builder.createOp(spv::Op::OpImageRead, type, {tex, coord});
            } else {
                id = _builder.createOp(spv::Op::OpImageFetch, type, {tex, coord});
            }
            break;
        }
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto elem_index = _ensure_type(_emit_value(inst->operand(2)), uint_type);
            auto nonuniform = !_uniformity.is_uniform(inst->operand(1));
            auto slot_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto bdls_ptr = _create_access_chain(_builder.getStorageClass(bindless_array), bindless_array, {_builder.makeUintConstant(0u), slot_offset}, nonuniform);
            auto buffer_idx = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            if (nonuniform) { _builder.addDecoration(buffer_idx, spv::Decoration::NonUniformEXT); }
            LUISA_ASSERT(_buffer_heap_id != spv::NoResult, "SPIR-V buffer heap not bound.");
            auto buffer_base = _create_access_chain(_builder.getStorageClass(_buffer_heap_id), _buffer_heap_id, {buffer_idx}, nonuniform);
            id = _emit_buffer_read(buffer_base, elem_index, inst->type(), nullptr);
            break;
        }
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto byte_index = _ensure_type(_emit_value(inst->operand(2)), uint_type);
            auto nonuniform = !_uniformity.is_uniform(inst->operand(1));
            auto slot_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto bdls_ptr = _create_access_chain(_builder.getStorageClass(bindless_array), bindless_array, {_builder.makeUintConstant(0u), slot_offset}, nonuniform);
            auto buffer_idx = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            if (nonuniform) { _builder.addDecoration(buffer_idx, spv::Decoration::NonUniformEXT); }
            LUISA_ASSERT(_buffer_heap_id != spv::NoResult, "SPIR-V buffer heap not bound.");
            auto buffer_base = _create_access_chain(_builder.getStorageClass(_buffer_heap_id), _buffer_heap_id, {buffer_idx}, nonuniform);
            auto read_type = inst->type();
            auto is_subword_vector = read_type->is_vector() && read_type->element()->size() < 4u;
            if (read_type->size() < 4u || is_subword_vector) {
                id = _emit_buffer_read(buffer_base, byte_index, read_type, nullptr, false);
            } else {
                auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
                id = _emit_buffer_read(buffer_base, word_index, read_type, nullptr, true);
            }
            break;
        }
    }
    LUISA_ASSERT(id != spv::NoResult, "Failed to emit resource read.");
    _value_map.emplace(inst, id);
}

void SpirvCodegenEntry::_emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept {
    auto uint_type = _builder.makeUintType(32);

    switch (inst->op()) {
        case xir::ResourceWriteOp::BUFFER_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            _emit_buffer_write(buffer, index, value, inst->operand(2)->type(), inst->operand(0)->type());
            break;
        }
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            _emit_buffer_write(buffer, index, value, inst->operand(2)->type(), inst->operand(0)->type(), false, spv::MemoryAccessMask::Volatile);
            _builder.createMemoryBarrier(spv::Scope::Device,
                                         spv::MemorySemanticsAllMemory |
                                             spv::MemorySemanticsMask::AcquireRelease);
            break;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto value = _emit_value(inst->operand(2));
            auto value_type = inst->operand(2)->type();
            auto is_subword_vector = value_type->is_vector() && value_type->element()->size() < 4u;
            if (value_type->size() < 4u || is_subword_vector) {
                _emit_buffer_write(buffer, byte_index, value, value_type, nullptr, false);
            } else {
                auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
                _emit_buffer_write(buffer, word_index, value, value_type, nullptr, true);
            }
            break;
        }
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto buffer = _emit_value(inst->operand(0));
            auto byte_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto value = _emit_value(inst->operand(2));
            auto value_type = inst->operand(2)->type();
            auto is_subword_vector = value_type->is_vector() && value_type->element()->size() < 4u;
            if (value_type->size() < 4u || is_subword_vector) {
                _emit_buffer_write(buffer, byte_index, value, value_type, nullptr, false, spv::MemoryAccessMask::Volatile);
            } else {
                auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
                _emit_buffer_write(buffer, word_index, value, value_type, nullptr, true, spv::MemoryAccessMask::Volatile);
            }
            _builder.createMemoryBarrier(spv::Scope::Device,
                                         spv::MemorySemanticsAllMemory |
                                             spv::MemorySemanticsMask::AcquireRelease);
            break;
        }
        case xir::ResourceWriteOp::TEXTURE2D_WRITE:
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: {
            auto tex_array = _emit_value(inst->operand(0));
            auto coord = _emit_value(inst->operand(1));
            auto value = _emit_value(inst->operand(2));
            auto tex = _load_texture(tex_array);
            auto image_type = _builder.getImageType(tex);
            if (_builder.getImageTypeFormat(image_type) == spv::ImageFormat::Unknown) {
                _builder.addCapability(spv::Capability::StorageImageWriteWithoutFormat);
            }
            _builder.createNoResultOp(spv::Op::OpImageWrite, {tex, coord, value});
            break;
        }
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto elem_index = _emit_value(inst->operand(2));
            auto value = _emit_value(inst->operand(3));
            auto elem_type = inst->operand(3)->type();
            auto nonuniform = !_uniformity.is_uniform(inst->operand(1));
            auto slot_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto bdls_ptr = _create_access_chain(_builder.getStorageClass(bindless_array), bindless_array, {_builder.makeUintConstant(0u), slot_offset}, nonuniform);
            auto buffer_idx = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            if (nonuniform) { _builder.addDecoration(buffer_idx, spv::Decoration::NonUniformEXT); }
            LUISA_ASSERT(_buffer_heap_id != spv::NoResult, "SPIR-V buffer heap not bound.");
            auto buffer_base = _create_access_chain(_builder.getStorageClass(_buffer_heap_id), _buffer_heap_id, {buffer_idx}, nonuniform);
            _emit_buffer_write(buffer_base, elem_index, value, elem_type, nullptr);
            break;
        }
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: {
            auto bindless_array = _emit_value(inst->operand(0));
            auto slot_index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto byte_index = _ensure_type(_emit_value(inst->operand(2)), uint_type);
            auto value = _emit_value(inst->operand(3));
            auto value_type = inst->operand(3)->type();
            auto nonuniform = !_uniformity.is_uniform(inst->operand(1));
            auto slot_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, slot_index, _builder.makeUintConstant(3u));
            auto bdls_ptr = _create_access_chain(_builder.getStorageClass(bindless_array), bindless_array, {_builder.makeUintConstant(0u), slot_offset}, nonuniform);
            auto buffer_idx = _builder.createLoad(bdls_ptr, spv::NoPrecision);
            if (nonuniform) { _builder.addDecoration(buffer_idx, spv::Decoration::NonUniformEXT); }
            LUISA_ASSERT(_buffer_heap_id != spv::NoResult, "SPIR-V buffer heap not bound.");
            auto buffer_base = _create_access_chain(_builder.getStorageClass(_buffer_heap_id), _buffer_heap_id, {buffer_idx}, nonuniform);
            auto is_subword_vector = value_type->is_vector() && value_type->element()->size() < 4u;
            if (value_type->size() < 4u || is_subword_vector) {
                _emit_buffer_write(buffer_base, byte_index, value, value_type, nullptr, false);
            } else {
                auto word_index = _builder.createBinOp(spv::Op::OpUDiv, uint_type, byte_index, _builder.makeUintConstant(4u));
                _emit_buffer_write(buffer_base, word_index, value, value_type, nullptr, true);
            }
            break;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: {
            auto buffer = _emit_value(inst->operand(0));
            auto index = _ensure_type(_emit_value(inst->operand(1)), uint_type);
            auto base_offset = _builder.createBinOp(spv::Op::OpIMul, uint_type, index, _builder.makeUintConstant(16u));
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
                        auto word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_offset,
                                                                _builder.makeUintConstant(row * 4u));
                        _emit_buffer_write_impl(buffer, word_offset, row_vec, f32v4);
                    }
                    break;
                }
                case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK: {
                    auto value = _emit_value(inst->operand(2));
                    auto word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_offset, _builder.makeUintConstant(13u));
                    _emit_buffer_write_impl(buffer, word_offset, value, Type::of<uint>());
                    break;
                }
                case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: {
                    auto value = _emit_value(inst->operand(2));
                    auto word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_offset, _builder.makeUintConstant(12u));
                    _emit_buffer_write_impl(buffer, word_offset, value, Type::of<uint>());
                    break;
                }
                case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY: {
                    auto opaque = _emit_value(inst->operand(2));
                    auto value = _builder.createOp(spv::Op::OpSelect, uint_type, {
                        opaque, _builder.makeUintConstant(4u), _builder.makeUintConstant(8u)
                    });
                    auto word_offset = _builder.createBinOp(spv::Op::OpIAdd, uint_type, base_offset, _builder.makeUintConstant(14u));
                    _emit_buffer_write_impl(buffer, word_offset, value, Type::of<uint>());
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
    switch (inst->op()) {
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: {
            _builder.createControlBarrier(spv::Scope::Workgroup, spv::Scope::Device,
                                          spv::MemorySemanticsAllMemory |
                                              spv::MemorySemanticsMask::AcquireRelease);
            break;
        }
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            id = _builder.createOp(spv::Op::OpGroupNonUniformElect, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::Id>{subgroup_scope});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformVote);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformAll, _convert_type(inst->type(), Usage::READ),
                                   {subgroup_scope, val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformVote);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformAny, _convert_type(inst->type(), Usage::READ),
                                   {subgroup_scope, val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseAnd, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseOr, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformBitwiseXor, _convert_type(inst->type(), Usage::READ),
                                   std::vector<spv::IdImmediate>{
                                       {true, subgroup_scope},
                                       {false, group_op_reduce},
                                       {true, val}});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformArithmetic);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformArithmetic);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformArithmetic);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformArithmetic);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformVote);
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformAllEqual, _convert_type(inst->type(), Usage::READ),
                                   {subgroup_scope, val});
            break;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformBallot);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformBallot);
            auto val = _emit_value(inst->operand(0));
            auto uint_type = _builder.makeUintType(32);
            auto ballot_type = _builder.makeVectorType(uint_type, 4);
            id = _builder.createOp(spv::Op::OpGroupNonUniformBallot, ballot_type,
                                   {subgroup_scope, val});
            break;
        }
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformBallot);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformBallot);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformArithmetic);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformArithmetic);
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
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformShuffle);
            auto val = _emit_value(inst->operand(0));
            auto lane = _emit_value(inst->operand(1));
            id = _builder.createOp(spv::Op::OpGroupNonUniformShuffle, _convert_type(inst->type(), Usage::READ),
                                   {subgroup_scope, val, lane});
            break;
        }
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: {
            _builder.addCapability(spv::Capability::GroupNonUniform);
            _builder.addCapability(spv::Capability::GroupNonUniformBallot);
            _builder.addCapability(spv::Capability::GroupNonUniformShuffle);
            auto uint_type = _builder.makeUintType(32);
            auto ballot_type = _builder.makeVectorType(uint_type, 4);
            auto true_val = _builder.makeBoolConstant(true);
            auto ballot = _builder.createOp(spv::Op::OpGroupNonUniformBallot, ballot_type,
                                            {subgroup_scope, true_val});
            auto first_lane = _builder.createOp(spv::Op::OpGroupNonUniformBallotFindLSB, uint_type,
                                                {subgroup_scope, ballot});
            auto val = _emit_value(inst->operand(0));
            id = _builder.createOp(spv::Op::OpGroupNonUniformShuffle, _convert_type(inst->type(), Usage::READ),
                                   {subgroup_scope, val, first_lane});
            break;
        }
        // ignored
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER:{

        } break;
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
            if (storage == spv::StorageClass::Workgroup && _entry_point_inst != nullptr) {
                _entry_point_inst->addIdOperand(var);
            }
            set_result(var);
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
                // Remap the alloca so subsequent uses resolve to the source variable.
                _value_map[store->variable()] = val;
            } else {
                if (pointee_type != val_type) {
                    if (_builder.isScalarType(val_type) && _builder.isVectorType(pointee_type)) {
                        val = _builder.smearScalar(spv::NoPrecision, val, pointee_type);
                    } else if (!_builder.isStructType(pointee_type) && !_builder.isStructType(val_type) &&
                               _builder.getTypeClass(pointee_type) == _builder.getTypeClass(val_type) &&
                               _builder.getNumTypeComponents(pointee_type) == _builder.getNumTypeComponents(val_type)) {
                        val = _builder.createUnaryOp(spv::Op::OpBitcast, pointee_type, val);
                    }
                }
                _builder.createStore(val, ptr);
            }
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
            auto callee = static_cast<const xir::Function *>(call->callee());
            auto callee_func = _function_map.at(callee);
            std::vector<spv::Id> args;
            luisa::vector<std::pair<spv::Id, spv::Id>> temp_copies;
            luisa::vector<const xir::Argument *> callable_arg_list;
            for (auto arg : callee->arguments()) {
                callable_arg_list.emplace_back(arg);
            }
            if (auto it = _callable_arg_used.find(callee); it != _callable_arg_used.end()) {
                const auto &used_mask = it->second;
                size_t idx = 0;
                for (auto arg_use : call->argument_uses()) {
                    if (idx < used_mask.size() && !used_mask[idx] && callable_arg_list[idx]->is_resource()) {
                        ++idx;
                        continue;
                    }
                    auto arg_val = _emit_value(arg_use->value());
                    if (idx < callable_arg_list.size() && callable_arg_list[idx]->is_reference()) {
                        auto opcode = _builder.getOpCode(arg_val);
                        if (opcode != spv::Op::OpVariable && opcode != spv::Op::OpFunctionParameter) {
                            auto pointee_type = _builder.getContainedTypeId(_builder.getTypeId(arg_val));
                            auto temp_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, pointee_type, "call_tmp");
                            auto loaded = _builder.createLoad(arg_val, spv::NoPrecision);
                            _builder.createStore(loaded, temp_var);
                            auto storage = _builder.getStorageClass(arg_val);
                            if (storage != spv::StorageClass::UniformConstant &&
                                storage != spv::StorageClass::Input &&
                                storage != spv::StorageClass::PushConstant) {
                                temp_copies.emplace_back(temp_var, arg_val);
                            }
                            arg_val = temp_var;
                        }
                    }
                    args.emplace_back(arg_val);
                    ++idx;
                }
            } else {
                size_t idx = 0;
                for (auto arg_use : call->argument_uses()) {
                    auto arg_val = _emit_value(arg_use->value());
                    if (idx < callable_arg_list.size() && callable_arg_list[idx]->is_reference()) {
                        auto opcode = _builder.getOpCode(arg_val);
                        if (opcode != spv::Op::OpVariable && opcode != spv::Op::OpFunctionParameter) {
                            auto pointee_type = _builder.getContainedTypeId(_builder.getTypeId(arg_val));
                            auto temp_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, pointee_type, "call_tmp");
                            auto loaded = _builder.createLoad(arg_val, spv::NoPrecision);
                            _builder.createStore(loaded, temp_var);
                            auto storage = _builder.getStorageClass(arg_val);
                            if (storage != spv::StorageClass::UniformConstant &&
                                storage != spv::StorageClass::Input &&
                                storage != spv::StorageClass::PushConstant) {
                                temp_copies.emplace_back(temp_var, arg_val);
                            }
                            arg_val = temp_var;
                        }
                    }
                    args.emplace_back(arg_val);
                    ++idx;
                }
            }
            auto id = _builder.createFunctionCall(callee_func, args);
            for (auto [temp_var, original_ptr] : temp_copies) {
                auto loaded = _builder.createLoad(temp_var, spv::NoPrecision);
                _builder.createStore(loaded, original_ptr);
            }
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
                if (from_is_bool != to_is_bool || (from_is_bool && from->size() != to->size())) {
                    // Cannot bitcast between bool and non-bool, or between different-sized bools
                    // Convert bool to/from uint first
                    auto uint_type = _builder.makeUintType(32);
                    spv::Id uint_val = val;
                    if (from_is_bool && !to_is_bool) {
                        // bool → non-bool: convert to uint first
                        if (spv::Id bool_type = _convert_type(from, Usage::READ);
                            _builder.isBoolType(bool_type)) {
                            uint_val = _builder.createOp(spv::Op::OpSelect, uint_type, {val, _builder.makeUintConstant(1u), _builder.makeUintConstant(0u)});
                        } else {
                            // Bool vector: decompose and pack
                            auto dim = _builder.getNumTypeComponents(bool_type);
                            uint_val = _builder.makeUintConstant(0u);
                            for (uint i = 0u; i < dim; ++i) {
                                auto comp = _builder.createCompositeExtract(val, _builder.makeBoolType(), i);
                                auto bit = _builder.createOp(spv::Op::OpSelect, uint_type, {comp, _builder.makeUintConstant(1u << i), _builder.makeUintConstant(0u)});
                                uint_val = _builder.createBinOp(spv::Op::OpBitwiseOr, uint_type, uint_val, bit);
                            }
                        }
                        id = _builder.createUnaryOp(spv::Op::OpBitcast, spv_to, uint_val);
                    } else if (!from_is_bool && to_is_bool) {
                        // non-bool → bool: bitcast to uint, then convert
                        uint_val = _builder.createUnaryOp(spv::Op::OpBitcast, uint_type, val);
                        if (_builder.isBoolType(spv_to)) {
                            id = _builder.createBinOp(spv::Op::OpINotEqual, spv_to, uint_val, _builder.makeUintConstant(0u));
                        } else {
                            // Bool vector: unpack bits
                            auto dim = _builder.getNumTypeComponents(spv_to);
                            std::vector<spv::Id> comps;
                            comps.reserve(dim);
                            for (uint i = 0u; i < dim; ++i) {
                                auto bit = _builder.createBinOp(spv::Op::OpBitwiseAnd, uint_type, uint_val, _builder.makeUintConstant(1u << i));
                                comps.push_back(_builder.createBinOp(spv::Op::OpINotEqual, _builder.makeBoolType(), bit, _builder.makeUintConstant(0u)));
                            }
                            id = _builder.createCompositeConstruct(spv_to, comps);
                        }
                    } else {
                        // bool → bool with different sizes: compare with 0
                        id = _builder.createBinOp(spv::Op::OpINotEqual, spv_to, val, _builder.makeUintConstant(0u));
                    }
                } else {
                    id = _builder.createUnaryOp(spv::Op::OpBitcast, spv_to, val);
                }
            } else {
                if (from == to) {
                    id = val;
                } else if (from->is_bool() && to->is_scalar()) {
                    spv::Id zero = spv::NoResult;
                    spv::Id one = spv::NoResult;
                    if (to->is_int()) {
                        auto bit_width = static_cast<int32_t>(to->size() * 8);
                        zero = _builder.makeIntConstant(_builder.makeIntType(bit_width), 0u, false);
                        one = _builder.makeIntConstant(_builder.makeIntType(bit_width), 1u, false);
                    } else if (to->is_uint()) {
                        auto bit_width = static_cast<int32_t>(to->size() * 8);
                        zero = _builder.makeIntConstant(_builder.makeUintType(bit_width), 0u, false);
                        one = _builder.makeIntConstant(_builder.makeUintType(bit_width), 1u, false);
                    } else if (to->is_float()) {
                        auto bit_width = static_cast<int32_t>(to->size() * 8);
                        if (bit_width == 8) {
                            if (to->is_float8_e5m2()) {
                                zero = _builder.makeFloatE5M2Constant(0.0f);
                                one = _builder.makeFloatE5M2Constant(1.0f);
                            } else if (to->is_float8_e4m3()) {
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
                        LUISA_NOT_IMPLEMENTED("SPIR-V bool-to-scalar cast for {}.", to->description());
                    }
                    id = _builder.createTriOp(spv::Op::OpSelect, spv_to, val, one, zero);
                } else if (to->is_bool() && from->is_scalar()) {
                    spv::Id zero = spv::NoResult;
                    if (from->is_int()) {
                        auto bit_width = static_cast<int32_t>(from->size() * 8);
                        zero = _builder.makeIntConstant(_builder.makeIntType(bit_width), 0u, false);
                    } else if (from->is_uint()) {
                        auto bit_width = static_cast<int32_t>(from->size() * 8);
                        zero = _builder.makeIntConstant(_builder.makeUintType(bit_width), 0u, false);
                    } else if (from->is_float()) {
                        auto bit_width = static_cast<int32_t>(from->size() * 8);
                        if (bit_width == 8) {
                            if (from->is_float8_e5m2()) {
                                zero = _builder.makeFloatE5M2Constant(0.0f);
                            } else if (from->is_float8_e4m3()) {
                                zero = _builder.makeFloatE4M3Constant(0.0f);
                            }
                        } else if (bit_width == 16) {
                            zero = _builder.makeFloat16Constant(0.0f);
                        } else if (bit_width == 32) {
                            zero = _builder.makeFloatConstant(0.0f);
                        } else if (bit_width == 64) {
                            zero = _builder.makeDoubleConstant(0.0);
                        }
                    }
                    if (zero == spv::NoResult) {
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
                        if (from->is_int() != to->is_int()) {
                            id = _builder.createUnaryOp(spv::Op::OpBitcast, spv_to, val);
                        } else {
                            id = val;
                        }
                    } else if (from->is_int() && to->is_int()) {
                        id = _builder.createUnaryOp(spv::Op::OpSConvert, spv_to, val);
                    } else if (from->is_uint() && to->is_uint()) {
                        id = _builder.createUnaryOp(spv::Op::OpUConvert, spv_to, val);
                    } else {
                        // Cross-signedness with different sizes: convert first, then bitcast
                        auto tmp_type = _builder.makeIntegerType(static_cast<int32_t>(to->size() * 8), from->is_int());
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
        case xir::DerivedInstructionTag::PHI: {
            auto phi = static_cast<const xir::PhiInst *>(inst);
            auto spv_type = _convert_type(phi->type(), Usage::READ);
            std::vector<spv::IdImmediate> operands;
            for (size_t i = 0; i < phi->incoming_count(); ++i) {
                auto [value, block] = phi->incoming(i);
                operands.push_back({true, _emit_value(value)});
                operands.push_back({true, _block_map.at(block)->getId()});
            }
            _value_map.emplace(inst, _builder.createOp(spv::Op::OpPhi, spv_type, operands));
            break;
        }
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
            // Print is not supported in SPIR-V; emit as no-op
            break;
        case xir::DerivedInstructionTag::CLOCK:
        case xir::DerivedInstructionTag::ASSERT:
        case xir::DerivedInstructionTag::ASSUME:
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

void SpirvCodegenEntry::_emit_ray_query_loop_inst(const xir::RayQueryLoopInst *inst) noexcept {
    auto &function = _builder.getBuildPoint()->getParent();
    auto dispatch_spv = _get_or_create_block(inst->dispatch_block());
    auto merge_spv = _get_or_create_block(inst->merge_block());
    auto header = &_builder.makeNewBlock();
    auto continue_block = &_builder.makeNewBlock();
    _loop_header_redirect.emplace(inst->dispatch_block(), continue_block);
    _loop_header_info.emplace(inst->dispatch_block(), std::make_pair(header, continue_block));
    _builder.createBranch(false, header);
    _builder.setBuildPoint(header);
    _used_merge_blocks.emplace((merge_spv)->getId());
        _builder.createLoopMerge(merge_spv, continue_block, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, dispatch_spv);
    _emit_block(inst->dispatch_block());
    while (!_pending_blocks.empty()) {
        auto *bb = _pending_blocks.back();
        _pending_blocks.pop_back();
        if (bb == inst->merge_block()) {
            continue;
        }
        _emit_block(bb);
    }
    _builder.setBuildPoint(continue_block);
    _builder.createBranch(false, header);
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_ray_query_dispatch_inst(const xir::RayQueryDispatchInst *inst) noexcept {
    auto rq_obj = _emit_value(inst->query_object());
    auto bool_type = _builder.makeBoolType();
    auto proceed = _builder.createOp(spv::Op::OpRayQueryProceedKHR, bool_type, std::vector<spv::Id>{rq_obj});
    auto &function = _builder.getBuildPoint()->getParent();
    auto exit_block = _get_or_create_block(inst->exit_block());
    auto check_block = new spv::Block(_builder.getUniqueId(), function);
    _builder.createConditionalBranch(proceed, check_block, exit_block);
    function.addBlock(check_block);
    _builder.setBuildPoint(check_block);
    auto uint_type = _builder.makeUintType(32);
    auto candidate_const = _builder.makeIntConstant(0);// RayQueryCandidateIntersectionKHR
    auto candidate_type = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTypeKHR, uint_type,
                                            std::vector<spv::IdImmediate>{{true, rq_obj}, {true, candidate_const}});
    auto bind_or_get = [&](const xir::BasicBlock *xb, bool &is_fresh) -> spv::Block * {
        if (auto it = _block_map.find(xb); it != _block_map.end()) {
            is_fresh = false;
            return it->second;
        }
        is_fresh = true;
        auto blk = new spv::Block(_builder.getUniqueId(), function);
        _block_map.emplace(xb, blk);
        return blk;
    };
    bool surface_fresh = false, procedural_fresh = false;
    auto surface_block = bind_or_get(inst->on_surface_candidate_block(), surface_fresh);
    auto procedural_block = bind_or_get(inst->on_procedural_candidate_block(), procedural_fresh);
    auto dispatch_merge_block = new spv::Block(_builder.getUniqueId(), function);
    auto dispatch_xir_block = inst->parent_block();
    spv::Block *continue_block = nullptr;
    if (auto it = _loop_header_info.find(dispatch_xir_block); it != _loop_header_info.end()) {
        continue_block = it->second.second;
    } else {
        LUISA_ERROR_WITH_LOCATION("RayQueryDispatchInst not inside a RayQueryLoop.");
    }
    auto is_triangle = _builder.createBinOp(spv::Op::OpIEqual, bool_type, candidate_type, _builder.makeUintConstant(0u));
    _used_merge_blocks.emplace(dispatch_merge_block->getId());
    auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2);
    selection_merge->addIdOperand(dispatch_merge_block->getId());
    selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));
    _builder.createConditionalBranch(is_triangle, surface_block, procedural_block);
    // Temporarily redirect branches to the dispatch block so they target the dispatch merge block
    // instead of the continue block, satisfying SPIR-V structured control flow rules.
    auto saved_redirect = _loop_header_redirect[dispatch_xir_block];
    _loop_header_redirect[dispatch_xir_block] = dispatch_merge_block;
    if (surface_fresh) { function.addBlock(surface_block); }
    _builder.setBuildPoint(surface_block);
    _emit_block(inst->on_surface_candidate_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, dispatch_merge_block);
    }
    if (procedural_fresh) { function.addBlock(procedural_block); }
    _builder.setBuildPoint(procedural_block);
    _emit_block(inst->on_procedural_candidate_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, dispatch_merge_block);
    }
    _loop_header_redirect[dispatch_xir_block] = saved_redirect;
    function.addBlock(dispatch_merge_block);
    _builder.setBuildPoint(dispatch_merge_block);
    _builder.createBranch(false, continue_block);
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
            auto origin = _builder.createOp(spv::Op::OpRayQueryGetWorldRayOriginKHR, vec3_type, std::vector<spv::Id>{rq_obj});
            auto dir = _builder.createOp(spv::Op::OpRayQueryGetWorldRayDirectionKHR, vec3_type, std::vector<spv::Id>{rq_obj});
            auto t_min = _builder.createOp(spv::Op::OpRayQueryGetRayTMinKHR, float_type, std::vector<spv::Id>{rq_obj});
            auto committed = _builder.makeIntConstant(1);// RayQueryCommittedIntersectionKHR
            auto t_max = _builder.createOp(spv::Op::OpRayQueryGetIntersectionTKHR, float_type,
                                           std::vector<spv::IdImmediate>{{true, rq_obj}, {true, committed}});
            auto ray_type = inst->type();
            auto origin_array_type = _convert_type(ray_type->members()[0], Usage::READ);
            auto dir_array_type = _convert_type(ray_type->members()[2], Usage::READ);
            auto origin_arr = _builder.createCompositeConstruct(origin_array_type, {
                _builder.createCompositeExtract(origin, float_type, 0),
                _builder.createCompositeExtract(origin, float_type, 1),
                _builder.createCompositeExtract(origin, float_type, 2)});
            auto dir_arr = _builder.createCompositeConstruct(dir_array_type, {
                _builder.createCompositeExtract(dir, float_type, 0),
                _builder.createCompositeExtract(dir, float_type, 1),
                _builder.createCompositeExtract(dir, float_type, 2)});
            id = _builder.createCompositeConstruct(type, {origin_arr, t_min, dir_arr, t_max});
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
            auto &function = _builder.getBuildPoint()->getParent();
            auto tri_block = new spv::Block(_builder.getUniqueId(), function);
            auto proc_block = new spv::Block(_builder.getUniqueId(), function);
            auto merge_block = new spv::Block(_builder.getUniqueId(), function);
            auto result_var = _builder.createVariable(spv::NoPrecision, spv::StorageClass::Function, type, "committed_hit");
            auto zero_result = _builder.makeNullConstant(type);
            _builder.createStore(zero_result, result_var);
            auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
            selection_merge->reserveOperands(2);
            selection_merge->addIdOperand(merge_block->getId());
            selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
            _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));
            auto switch_inst = new spv::Instruction(spv::Op::OpSwitch);
            switch_inst->reserveOperands(6);
            switch_inst->addIdOperand(committed_type);
            switch_inst->addIdOperand(merge_block->getId());// default (none) -> merge
            switch_inst->addImmediateOperand(1u);            // triangle
            switch_inst->addIdOperand(tri_block->getId());
            switch_inst->addImmediateOperand(2u);            // procedural
            switch_inst->addIdOperand(proc_block->getId());
            _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(switch_inst));
            merge_block->addPredecessor(_builder.getBuildPoint());
            tri_block->addPredecessor(_builder.getBuildPoint());
            proc_block->addPredecessor(_builder.getBuildPoint());
            function.addBlock(tri_block);
            _builder.setBuildPoint(tri_block);
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
            function.addBlock(proc_block);
            _builder.setBuildPoint(proc_block);
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
            function.addBlock(merge_block);
            _builder.setBuildPoint(merge_block);
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
            auto iter = _rq_proceed_result.find(rq_obj);
            if (iter == _rq_proceed_result.end()) {
                id = _builder.makeBoolConstant(true);
            } else {
                id = _builder.createUnaryOp(spv::Op::OpLogicalNot, bool_type, iter->second);
            }
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
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED: {
            auto proceed_result = _builder.createOp(spv::Op::OpRayQueryProceedKHR, _builder.makeBoolType(), std::vector<spv::Id>{rq_obj});
            _rq_proceed_result[rq_obj] = proceed_result;
            break;
        }
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
            if (val_width == tgt_width && val_signed == tgt_signed) {
                return value;  // No conversion needed
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
    if (val_class == spv::Op::OpTypeBool && tgt_class == spv::Op::OpTypeInt) {
        // bool → int: select 1 or 0
        auto bit_width = static_cast<int32_t>(_builder.getScalarTypeWidth(tgt_scalar));
        auto one = _builder.isIntType(tgt_scalar) ? _builder.makeIntConstant(_builder.makeIntType(bit_width), 1u, false) : _builder.makeIntConstant(_builder.makeUintType(bit_width), 1u, false);
        auto zero = _builder.isIntType(tgt_scalar) ? _builder.makeIntConstant(_builder.makeIntType(bit_width), 0u, false) : _builder.makeIntConstant(_builder.makeUintType(bit_width), 0u, false);
        return _builder.createOp(spv::Op::OpSelect, target_type, {value, one, zero});
    }
    if (val_class == spv::Op::OpTypeInt && tgt_class == spv::Op::OpTypeBool) {
        // int → bool: compare with 0
        auto bit_width = static_cast<int32_t>(_builder.getScalarTypeWidth(val_scalar));
        auto zero = _builder.isIntType(val_scalar) ? _builder.makeIntConstant(_builder.makeIntType(bit_width), 0u, false) : _builder.makeIntConstant(_builder.makeUintType(bit_width), 0u, false);
        return _builder.createBinOp(spv::Op::OpINotEqual, target_type, value, zero);
    }
    return _builder.createUnaryOp(spv::Op::OpBitcast, target_type, value);
}

}// namespace lc::spirv
