//
// Created by mike on 3/18/26.
//

#include <span>
#include <numbers>

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

namespace detail {
[[nodiscard]] llvm::Value *hip_codegen_llvm_determinant2x2(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1) noexcept;
[[nodiscard]] llvm::Value *hip_codegen_llvm_determinant3x3(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1, llvm::Value *m2) noexcept;
[[nodiscard]] llvm::Value *hip_codegen_llvm_determinant4x4(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1, llvm::Value *m2, llvm::Value *m3) noexcept;
[[nodiscard]] llvm::Value *hip_codegen_llvm_inverse2x2(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1) noexcept;
[[nodiscard]] llvm::Value *hip_codegen_llvm_inverse3x3(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1, llvm::Value *m2) noexcept;
[[nodiscard]] llvm::Value *hip_codegen_llvm_inverse4x4(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1, llvm::Value *m2, llvm::Value *m3) noexcept;
}// namespace detail

llvm::Value *HIPCodegenLLVMImpl::_translate_arithmetic_inst(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    auto translate_unary = [&](auto op) noexcept {
        LUISA_DEBUG_ASSERT(inst->operand_count() == 1 &&
                           inst->operand(0)->type() == inst->type());
        auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
        return op(llvm_value);
    };
    auto translate_binary = [&](auto op) noexcept {
        LUISA_DEBUG_ASSERT(inst->operand_count() == 2 &&
                           inst->operand(0)->type() == inst->type() &&
                           inst->operand(1)->type() == inst->type());
        auto llvm_lhs = _get_llvm_value(b, func_ctx, inst->operand(0));
        auto llvm_rhs = _get_llvm_value(b, func_ctx, inst->operand(1));
        return op(llvm_lhs, llvm_rhs);
    };
    auto translate_relational = [&](auto op) noexcept {
        LUISA_DEBUG_ASSERT(inst->operand_count() == 2 &&
                           inst->operand(0)->type() == inst->operand(1)->type() &&
                           inst->type()->is_bool_or_bool_vector());
        auto llvm_lhs = _get_llvm_value(b, func_ctx, inst->operand(0));
        auto llvm_rhs = _get_llvm_value(b, func_ctx, inst->operand(1));
        return op(llvm_lhs, llvm_rhs);
    };
    auto translate_ternary = [&](auto op) noexcept {
        LUISA_DEBUG_ASSERT(inst->operand_count() == 3 &&
                           inst->operand(0)->type() == inst->type() &&
                           inst->operand(1)->type() == inst->type() &&
                           inst->operand(2)->type() == inst->type());
        auto llvm_a = _get_llvm_value(b, func_ctx, inst->operand(0));
        auto llvm_b = _get_llvm_value(b, func_ctx, inst->operand(1));
        auto llvm_c = _get_llvm_value(b, func_ctx, inst->operand(2));
        return op(llvm_a, llvm_b, llvm_c);
    };
    auto broadcast_like = [&](llvm::Value *value, llvm::Value *shape) noexcept {
        if (value->getType() == shape->getType()) { return value; }
        auto shape_type = llvm::dyn_cast<llvm::FixedVectorType>(shape->getType());
        LUISA_DEBUG_ASSERT(shape_type != nullptr &&
                           value->getType() == shape_type->getElementType());
        return b.CreateVectorSplat(shape_type->getNumElements(), value);
    };
    auto translate_matrix_comp_unary_op = [&](auto op) noexcept {
        LUISA_DEBUG_ASSERT(inst->operand_count() == 1 &&
                           inst->operand(0)->type() == inst->type() &&
                           inst->type()->is_matrix());
        auto llvm_m = _get_llvm_value(b, func_ctx, inst->operand(0));
        auto llvm_result = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_m->getType()));
        auto dim = inst->type()->dimension();
        for (auto c = 0; c < dim; c++) {
            auto llvm_col = op(b.CreateExtractValue(llvm_m, c));
            llvm_result = b.CreateInsertValue(llvm_result, llvm_col, c);
        }
        return llvm_result;
    };
    auto translate_matrix_comp_binary_op = [&](auto op) noexcept {
        LUISA_DEBUG_ASSERT(inst->operand_count() == 2 && inst->type()->is_matrix());
        auto llvm_lhs = _get_llvm_value(b, func_ctx, inst->operand(0));
        auto llvm_rhs = _get_llvm_value(b, func_ctx, inst->operand(1));
        auto dim = inst->type()->dimension();
        if (auto lhs_type = llvm_lhs->getType(); !lhs_type->isArrayTy()) {
            LUISA_DEBUG_ASSERT(lhs_type->isFloatingPointTy());
            llvm_lhs = b.CreateVectorSplat(dim, llvm_lhs);
        }
        if (auto rhs_type = llvm_rhs->getType(); !rhs_type->isArrayTy()) {
            LUISA_DEBUG_ASSERT(rhs_type->isFloatingPointTy());
            llvm_rhs = b.CreateVectorSplat(dim, llvm_rhs);
        }
        auto llvm_result_type = _get_llvm_type(inst->type())->reg_type;
        LUISA_DEBUG_ASSERT(llvm_lhs->getType() == llvm_result_type || llvm_lhs->getType() == llvm_result_type->getArrayElementType());
        LUISA_DEBUG_ASSERT(llvm_rhs->getType() == llvm_result_type || llvm_rhs->getType() == llvm_result_type->getArrayElementType());
        auto llvm_result = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_result_type));
        for (auto c = 0; c < dim; c++) {
            auto llvm_lhs_col = llvm_lhs->getType()->isVectorTy() ? llvm_lhs : b.CreateExtractValue(llvm_lhs, c);
            auto llvm_rhs_col = llvm_rhs->getType()->isVectorTy() ? llvm_rhs : b.CreateExtractValue(llvm_rhs, c);
            auto llvm_col_result = op(llvm_lhs_col, llvm_rhs_col);
            llvm_result = b.CreateInsertValue(llvm_result, llvm_col_result, c);
        }
        return llvm_result;
    };
    auto dot_product_fp = [&](llvm::Value *u, llvm::Value *v) noexcept {
        LUISA_DEBUG_ASSERT(u->getType()->isVectorTy() && v->getType()->isFPOrFPVectorTy() && u->getType() == v->getType());
        auto zero = llvm::ConstantFP::getNegativeZero(u->getType()->getScalarType());
        return b.CreateFAddReduce(zero, b.CreateFMul(u, v));
    };
    auto inf_nan_mask_and_test = [&](llvm::Type *t) noexcept -> std::pair<llvm::Constant *, llvm::Constant *> {
        LUISA_DEBUG_ASSERT(t->isFPOrFPVectorTy());
        auto elem_t = t->getScalarType();
        auto int_t = static_cast<llvm::Type *>(llvm::Type::getIntNTy(b.getContext(), elem_t->getPrimitiveSizeInBits()));
        if (auto vt = llvm::dyn_cast<llvm::VectorType>(t)) {
            int_t = llvm::FixedVectorType::get(int_t, vt->getElementCount().getFixedValue());
        }
        return elem_t->isHalfTy()  ? std::make_pair(llvm::ConstantInt::get(int_t, 0x7fffull), llvm::ConstantInt::get(int_t, 0x7c00ull)) :
               elem_t->isFloatTy() ? std::make_pair(llvm::ConstantInt::get(int_t, 0x7fff'ffffull), llvm::ConstantInt::get(int_t, 0x7f80'0000ull)) :
                                     std::make_pair(llvm::ConstantInt::get(int_t, 0x7fff'ffff'ffff'ffffull), llvm::ConstantInt::get(int_t, 0x7ff0'0000'0000'0000ull));
    };
    auto call_unary_fp_intrinsic = [&](llvm::Value *v, llvm::Intrinsic::ID id) noexcept -> llvm::Value * {
        if (auto vt = llvm::dyn_cast<llvm::VectorType>(v->getType())) {
            auto dim = vt->getElementCount().getFixedValue();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(vt));
            for (auto i = 0; i < dim; i++) {
                auto elem = b.CreateExtractElement(v, i);
                auto res = b.CreateUnaryIntrinsic(id, elem);
                result = b.CreateInsertElement(result, res, i);
            }
            return result;
        }
        return b.CreateUnaryIntrinsic(id, v);
    };
    auto call_binary_fp_intrinsic = [&](llvm::Value *a, llvm::Value *b_arg, llvm::Intrinsic::ID id) noexcept -> llvm::Value * {
        if (auto vt = llvm::dyn_cast<llvm::VectorType>(a->getType())) {
            auto dim = vt->getElementCount().getFixedValue();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(vt));
            for (auto i = 0; i < dim; i++) {
                auto a_elem = b.CreateExtractElement(a, i);
                auto b_elem = b.CreateExtractElement(b_arg, i);
                auto res = b.CreateBinaryIntrinsic(id, a_elem, b_elem);
                result = b.CreateInsertElement(result, res, i);
            }
            return result;
        }
        return b.CreateBinaryIntrinsic(id, a, b_arg);
    };
    auto call_rsqrt = [&](llvm::Value *v) noexcept -> llvm::Value * {
        auto sqrt_v = call_unary_fp_intrinsic(v, llvm::Intrinsic::sqrt);
        auto one = llvm::ConstantFP::get(v->getType(), 1.0);
        return b.CreateFDiv(one, sqrt_v);
    };
    auto call_acosh = [&](llvm::Value *v) noexcept -> llvm::Value * {
        auto x_squared = b.CreateFMul(v, v);
        auto x_squared_minus_1 = b.CreateFSub(x_squared, llvm::ConstantFP::get(v->getType(), 1.0));
        auto sqrt_term = call_unary_fp_intrinsic(x_squared_minus_1, llvm::Intrinsic::sqrt);
        auto sum = b.CreateFAdd(v, sqrt_term);
        return call_unary_fp_intrinsic(sum, llvm::Intrinsic::log);
    };
    auto call_asinh = [&](llvm::Value *v) noexcept -> llvm::Value * {
        auto x_squared = b.CreateFMul(v, v);
        auto x_squared_plus_1 = b.CreateFAdd(x_squared, llvm::ConstantFP::get(v->getType(), 1.0));
        auto sqrt_term = call_unary_fp_intrinsic(x_squared_plus_1, llvm::Intrinsic::sqrt);
        auto sum = b.CreateFAdd(v, sqrt_term);
        return call_unary_fp_intrinsic(sum, llvm::Intrinsic::log);
    };
    auto call_atanh = [&](llvm::Value *v) noexcept -> llvm::Value * {
        auto one = llvm::ConstantFP::get(v->getType(), 1.0);
        auto one_plus_x = b.CreateFAdd(one, v);
        auto one_minus_x = b.CreateFSub(one, v);
        auto ratio = b.CreateFDiv(one_plus_x, one_minus_x);
        auto log_ratio = call_unary_fp_intrinsic(ratio, llvm::Intrinsic::log);
        auto half = llvm::ConstantFP::get(v->getType(), 0.5);
        return b.CreateFMul(half, log_ratio);
    };
    switch (inst->op()) {
        case xir::ArithmeticOp::UNARY_MINUS: return translate_unary([&](auto v) noexcept {
            return v->getType()->isIntOrIntVectorTy() ? b.CreateNeg(v) : b.CreateFNeg(v);
        });
        case xir::ArithmeticOp::UNARY_BIT_NOT: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy());
            return b.CreateNot(v);
        });
        case xir::ArithmeticOp::BINARY_ADD: return translate_binary([&](auto lhs, auto rhs) noexcept {
            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWAdd(lhs, rhs) :
                   inst->type()->is_uint_or_uint_vector() ? b.CreateAdd(lhs, rhs) :
                                                            b.CreateFAdd(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_SUB: return translate_binary([&](auto lhs, auto rhs) noexcept {
            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWSub(lhs, rhs) :
                   inst->type()->is_uint_or_uint_vector() ? b.CreateSub(lhs, rhs) :
                                                            b.CreateFSub(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_MUL: return translate_binary([&](auto lhs, auto rhs) noexcept {
            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWMul(lhs, rhs) :
                   inst->type()->is_uint_or_uint_vector() ? b.CreateMul(lhs, rhs) :
                                                            b.CreateFMul(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_DIV: return translate_binary([&](auto lhs, auto rhs) noexcept {
            return inst->type()->is_int_or_int_vector()   ? b.CreateSDiv(lhs, rhs) :
                   inst->type()->is_uint_or_uint_vector() ? b.CreateUDiv(lhs, rhs) :
                                                            b.CreateFDiv(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_MOD: return translate_binary([&](auto lhs, auto rhs) noexcept {
            return inst->type()->is_int_or_int_vector()   ? b.CreateSRem(lhs, rhs) :
                   inst->type()->is_uint_or_uint_vector() ? b.CreateURem(lhs, rhs) :
                                                            b.CreateFRem(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_BIT_AND: return translate_binary([&](auto lhs, auto rhs) noexcept {
            LUISA_DEBUG_ASSERT(lhs->getType()->isIntOrIntVectorTy());
            return b.CreateAnd(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_BIT_OR: return translate_binary([&](auto lhs, auto rhs) noexcept {
            LUISA_DEBUG_ASSERT(lhs->getType()->isIntOrIntVectorTy());
            return b.CreateOr(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_BIT_XOR: return translate_binary([&](auto lhs, auto rhs) noexcept {
            LUISA_DEBUG_ASSERT(lhs->getType()->isIntOrIntVectorTy());
            return b.CreateXor(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT: return translate_binary([&](auto lhs, auto rhs) noexcept {
            LUISA_DEBUG_ASSERT(lhs->getType()->isIntOrIntVectorTy());
            return b.CreateShl(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT: return translate_binary([&](auto lhs, auto rhs) noexcept {
            LUISA_DEBUG_ASSERT(lhs->getType()->isIntOrIntVectorTy());
            return inst->type()->is_int_or_int_vector() ? b.CreateAShr(lhs, rhs) : b.CreateLShr(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_ROTATE_LEFT: return translate_binary([&](auto lhs, auto rhs) noexcept {
            LUISA_DEBUG_ASSERT(lhs->getType()->isIntOrIntVectorTy());
            return b.CreateIntrinsic(llvm::Intrinsic::fshl, lhs->getType(), {lhs, lhs, rhs});
        });
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT: return translate_binary([&](auto lhs, auto rhs) noexcept {
            LUISA_DEBUG_ASSERT(lhs->getType()->isIntOrIntVectorTy());
            return b.CreateIntrinsic(llvm::Intrinsic::fshr, lhs->getType(), {lhs, lhs, rhs});
        });
        case xir::ArithmeticOp::BINARY_LESS: return translate_relational([&](auto lhs, auto rhs) noexcept {
            auto op_type = inst->operand(0)->type();
            return op_type->is_int_or_int_vector()   ? b.CreateICmpSLT(lhs, rhs) :
                   op_type->is_uint_or_uint_vector() ? b.CreateICmpULT(lhs, rhs) :
                                                       b.CreateFCmpOLT(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_GREATER: return translate_relational([&](auto lhs, auto rhs) noexcept {
            auto op_type = inst->operand(0)->type();
            return op_type->is_int_or_int_vector()   ? b.CreateICmpSGT(lhs, rhs) :
                   op_type->is_uint_or_uint_vector() ? b.CreateICmpUGT(lhs, rhs) :
                                                       b.CreateFCmpOGT(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_LESS_EQUAL: return translate_relational([&](auto lhs, auto rhs) noexcept {
            auto op_type = inst->operand(0)->type();
            return op_type->is_int_or_int_vector()   ? b.CreateICmpSLE(lhs, rhs) :
                   op_type->is_uint_or_uint_vector() ? b.CreateICmpULE(lhs, rhs) :
                                                       b.CreateFCmpOLE(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL: return translate_relational([&](auto lhs, auto rhs) noexcept {
            auto op_type = inst->operand(0)->type();
            return op_type->is_int_or_int_vector()   ? b.CreateICmpSGE(lhs, rhs) :
                   op_type->is_uint_or_uint_vector() ? b.CreateICmpUGE(lhs, rhs) :
                                                       b.CreateFCmpOGE(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_EQUAL: return translate_relational([&](auto lhs, auto rhs) noexcept {
            auto op_type = inst->operand(0)->type();
            return op_type->is_float_or_float_vector() ? b.CreateFCmpOEQ(lhs, rhs) : b.CreateICmpEQ(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: return translate_relational([&](auto lhs, auto rhs) noexcept {
            auto op_type = inst->operand(0)->type();
            return op_type->is_float_or_float_vector() ? b.CreateFCmpONE(lhs, rhs) : b.CreateICmpNE(lhs, rhs);
        });
        case xir::ArithmeticOp::ALL: {
            auto llvm_v = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_v->getType()->isIntOrIntVectorTy(1));
            return b.CreateAndReduce(llvm_v);
        }
        case xir::ArithmeticOp::ANY: {
            auto llvm_v = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_v->getType()->isIntOrIntVectorTy(1));
            return b.CreateOrReduce(llvm_v);
        }
        case xir::ArithmeticOp::SELECT: {
            auto llvm_false_v = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_true_v = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto llvm_cond_v = _get_llvm_value(b, func_ctx, inst->operand(2));
            LUISA_DEBUG_ASSERT(llvm_cond_v->getType()->isIntOrIntVectorTy(1));
            return b.CreateSelect(llvm_cond_v, llvm_true_v, llvm_false_v);
        }
        case xir::ArithmeticOp::CLAMP: return translate_ternary([&](auto v, auto lo, auto hi) noexcept {
            auto [max_op, min_op] = inst->type()->is_int_or_int_vector()   ? std::make_pair(llvm::Intrinsic::smax, llvm::Intrinsic::smin) :
                                    inst->type()->is_uint_or_uint_vector() ? std::make_pair(llvm::Intrinsic::umax, llvm::Intrinsic::umin) :
                                                                             std::make_pair(llvm::Intrinsic::maxnum, llvm::Intrinsic::minnum);
            auto clamp_lo = b.CreateBinaryIntrinsic(max_op, v, lo);
            return b.CreateBinaryIntrinsic(min_op, clamp_lo, hi);
        });
        case xir::ArithmeticOp::SATURATE: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(v->getType()->isFPOrFPVectorTy());
            auto zero = llvm::Constant::getNullValue(v->getType());
            auto one = llvm::ConstantFP::get(v->getType(), 1.);
            return b.CreateMinNum(b.CreateMaxNum(v, zero), one);
        });
        case xir::ArithmeticOp::LERP: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == 3 &&
                               inst->operand(0)->type() == inst->type() &&
                               inst->operand(1)->type() == inst->type());
            auto x = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto y = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto t = broadcast_like(_get_llvm_value(b, func_ctx, inst->operand(2)), x);
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            auto d = b.CreateFSub(y, x);
            return b.CreateFMA(t, d, x);
        }
        case xir::ArithmeticOp::SMOOTHSTEP: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == 3 &&
                               inst->operand(2)->type() == inst->type());
            auto x = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto e0 = broadcast_like(_get_llvm_value(b, func_ctx, inst->operand(0)), x);
            auto e1 = broadcast_like(_get_llvm_value(b, func_ctx, inst->operand(1)), x);
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            auto zero = llvm::Constant::getNullValue(x->getType());
            auto one = llvm::ConstantFP::get(x->getType(), 1.);
            auto t = b.CreateFDiv(b.CreateFSub(x, e0), b.CreateFSub(e1, e0));
            t = b.CreateMinNum(b.CreateMaxNum(t, zero), one);
            auto t_squared = b.CreateFMul(t, t);
            auto three = llvm::ConstantFP::get(x->getType(), 3.);
            auto minus_two = llvm::ConstantFP::get(x->getType(), -2.);
            return b.CreateFMul(t_squared, b.CreateFMA(minus_two, t, three));
        }
        case xir::ArithmeticOp::STEP: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == 2 &&
                               inst->operand(1)->type() == inst->type());
            auto x = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto edge = broadcast_like(_get_llvm_value(b, func_ctx, inst->operand(0)), x);
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            auto zero = llvm::Constant::getNullValue(x->getType());
            auto one = llvm::ConstantFP::get(x->getType(), 1.);
            auto cmp = b.CreateFCmpOLT(x, edge);
            return b.CreateSelect(cmp, zero, one);
        }
        case xir::ArithmeticOp::ABS: return translate_unary([&](auto v) noexcept {
            return inst->type()->is_float_or_float_vector() ? b.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, v) :
                   inst->type()->is_int_or_int_vector()     ? b.CreateIntrinsic(llvm::Intrinsic::abs, v->getType(), {v, b.getInt1(false)}) :
                                                              v;
        });
        case xir::ArithmeticOp::MIN: return translate_binary([&](auto lhs, auto rhs) noexcept {
            auto min_op = inst->type()->is_int_or_int_vector()   ? llvm::Intrinsic::smin :
                          inst->type()->is_uint_or_uint_vector() ? llvm::Intrinsic::umin :
                                                                   llvm::Intrinsic::minnum;
            return b.CreateBinaryIntrinsic(min_op, lhs, rhs);
        });
        case xir::ArithmeticOp::MAX: return translate_binary([&](auto lhs, auto rhs) noexcept {
            auto max_op = inst->type()->is_int_or_int_vector()   ? llvm::Intrinsic::smax :
                          inst->type()->is_uint_or_uint_vector() ? llvm::Intrinsic::umax :
                                                                   llvm::Intrinsic::maxnum;
            return b.CreateBinaryIntrinsic(max_op, lhs, rhs);
        });
        case xir::ArithmeticOp::CLZ: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy());
            return b.CreateBinaryIntrinsic(llvm::Intrinsic::ctlz, v, b.getInt1(false));
        });
        case xir::ArithmeticOp::CTZ: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy());
            return b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, v, b.getInt1(false));
        });
        case xir::ArithmeticOp::POPCOUNT: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy());
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, v);
        });
        case xir::ArithmeticOp::REVERSE: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy());
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::bitreverse, v);
        });
        case xir::ArithmeticOp::ISINF: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto [mask, test] = inf_nan_mask_and_test(v->getType());
            auto v_as_int = b.CreateBitCast(v, mask->getType());
            return b.CreateICmpEQ(b.CreateAnd(v_as_int, mask), test);
        }
        case xir::ArithmeticOp::ISNAN: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto [mask, test] = inf_nan_mask_and_test(v->getType());
            auto v_as_int = b.CreateBitCast(v, mask->getType());
            return b.CreateICmpUGT(b.CreateAnd(v_as_int, mask), test);
        }
        case xir::ArithmeticOp::ACOS: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "acos", v);
        });
        case xir::ArithmeticOp::ACOSH: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "acosh", v);
        });
        case xir::ArithmeticOp::ASIN: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "asin", v);
        });
        case xir::ArithmeticOp::ASINH: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "asinh", v);
        });
        case xir::ArithmeticOp::ATAN: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "atan", v);
        });
        case xir::ArithmeticOp::ATAN2: return translate_binary([&](auto y, auto x) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_binary_op(b, "atan2", y, x);
        });
        case xir::ArithmeticOp::ATANH: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "atanh", v);
        });
        case xir::ArithmeticOp::COS: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::cos);
        });
        case xir::ArithmeticOp::COSH: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "cosh", v);
        });
        case xir::ArithmeticOp::SIN: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::sin);
        });
        case xir::ArithmeticOp::SINH: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "sinh", v);
        });
        case xir::ArithmeticOp::TAN: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "tan", v);
        });
        case xir::ArithmeticOp::TANH: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return _call_ocml_unary_op(b, "tanh", v);
        });
        case xir::ArithmeticOp::EXP: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::exp);
        });
        case xir::ArithmeticOp::EXP2: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::exp2);
        });
        case xir::ArithmeticOp::EXP10: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::exp10);
        });
        case xir::ArithmeticOp::LOG: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::log);
        });
        case xir::ArithmeticOp::LOG2: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::log2);
        });
        case xir::ArithmeticOp::LOG10: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::log10);
        });
        case xir::ArithmeticOp::POW: return translate_binary([&](auto base, auto exponent) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_binary_fp_intrinsic(base, exponent, llvm::Intrinsic::pow);
        });
        case xir::ArithmeticOp::POW_INT: {
            auto base = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto exponent = _get_llvm_value(b, func_ctx, inst->operand(1));
            LUISA_DEBUG_ASSERT(base->getType()->isFPOrFPVectorTy() &&
                               exponent->getType()->isIntOrIntVectorTy() &&
                               inst->type()->is_float_or_float_vector());
            auto call_scalar_power_magnitude = [&](llvm::Value *scalar_base,
                                                   llvm::Value *magnitude,
                                                   llvm::Value *negative) noexcept {
                auto base_type = scalar_base->getType();
                auto exponent_type = llvm::cast<llvm::IntegerType>(magnitude->getType());
                auto name = fmt::format("luisa.pow_int.{}.u{}",
                                        _to_string(base_type),
                                        exponent_type->getBitWidth());
                auto helper = _llvm_module->getFunction(name);
                if (helper == nullptr) {
                    auto helper_type = llvm::FunctionType::get(
                        base_type,
                        {base_type, exponent_type, b.getInt1Ty()}, false);
                    helper = llvm::Function::Create(
                        helper_type, llvm::Function::PrivateLinkage,
                        name, *_llvm_module);
                    helper->addFnAttr(llvm::Attribute::AlwaysInline);

                    auto entry = llvm::BasicBlock::Create(
                        _llvm_context, "entry", helper);
                    auto positive = llvm::BasicBlock::Create(
                        _llvm_context, "positive", helper);
                    auto negative_block = llvm::BasicBlock::Create(
                        _llvm_context, "negative", helper);
                    auto loop = llvm::BasicBlock::Create(
                        _llvm_context, "loop", helper);
                    auto body = llvm::BasicBlock::Create(
                        _llvm_context, "body", helper);
                    auto exit = llvm::BasicBlock::Create(
                        _llvm_context, "exit", helper);
                    IB helper_b{entry};
                    helper_b.setFastMathFlags(b.getFastMathFlags());
                    helper_b.CreateCondBr(
                        helper->getArg(2), negative_block, positive);

                    auto one = llvm::ConstantFP::get(base_type, 1.0);
                    helper_b.SetInsertPoint(positive);
                    helper_b.CreateBr(loop);

                    helper_b.SetInsertPoint(negative_block);
                    auto reciprocal_base = helper_b.CreateFDiv(
                        one, helper->getArg(0));
                    helper_b.CreateBr(loop);

                    helper_b.SetInsertPoint(loop);
                    auto current_magnitude = helper_b.CreatePHI(
                        exponent_type, 3u, "magnitude");
                    auto current_result = helper_b.CreatePHI(
                        base_type, 3u, "result");
                    auto current_factor = helper_b.CreatePHI(
                        base_type, 3u, "factor");
                    auto zero = llvm::ConstantInt::get(exponent_type, 0u);
                    current_magnitude->addIncoming(helper->getArg(1), positive);
                    current_magnitude->addIncoming(helper->getArg(1), negative_block);
                    current_result->addIncoming(one, positive);
                    current_result->addIncoming(one, negative_block);
                    current_factor->addIncoming(helper->getArg(0), positive);
                    current_factor->addIncoming(reciprocal_base, negative_block);
                    auto done = helper_b.CreateICmpEQ(
                        current_magnitude, zero);
                    helper_b.CreateCondBr(done, exit, body);

                    helper_b.SetInsertPoint(body);
                    auto odd = helper_b.CreateTrunc(
                        current_magnitude, helper_b.getInt1Ty());
                    auto multiplied = helper_b.CreateFMul(
                        current_result, current_factor);
                    auto next_result = helper_b.CreateSelect(
                        odd, multiplied, current_result);
                    auto next_magnitude = helper_b.CreateLShr(
                        current_magnitude, 1u);
                    auto next_factor = helper_b.CreateFMul(
                        current_factor, current_factor);
                    helper_b.CreateBr(loop);
                    current_magnitude->addIncoming(next_magnitude, body);
                    current_result->addIncoming(next_result, body);
                    current_factor->addIncoming(next_factor, body);

                    helper_b.SetInsertPoint(exit);
                    helper_b.CreateRet(current_result);
                }
                return b.CreateCall(
                    helper, {scalar_base, magnitude, negative});
            };
            auto exponent_xir_type = inst->operand(1)->type();
            auto exponent_element_type = exponent_xir_type->is_vector() ?
                                             exponent_xir_type->element() :
                                             exponent_xir_type;
            auto exponent_is_signed = exponent_element_type->is_int();
            auto call_scalar_power = [&](llvm::Value *scalar_base,
                                         llvm::Value *scalar_exponent) noexcept {
                if (!exponent_is_signed) {
                    return call_scalar_power_magnitude(
                        scalar_base, scalar_exponent, b.getFalse());
                }
                auto exponent_type = llvm::cast<llvm::IntegerType>(
                    scalar_exponent->getType());
                auto zero = llvm::ConstantInt::get(exponent_type, 0u);
                auto negative = b.CreateICmpSLT(scalar_exponent, zero);
                auto magnitude = b.CreateSelect(
                    negative,
                    b.CreateSub(zero, scalar_exponent),
                    scalar_exponent);
                return call_scalar_power_magnitude(
                    scalar_base, magnitude, negative);
            };
            if (auto base_type = llvm::dyn_cast<llvm::FixedVectorType>(
                    base->getType())) {
                auto result = static_cast<llvm::Value *>(
                    llvm::PoisonValue::get(base_type));
                auto exponent_is_vector = exponent->getType()->isVectorTy();
                for (auto i = 0u; i < base_type->getNumElements(); i++) {
                    auto base_element = b.CreateExtractElement(base, i);
                    auto exponent_element = exponent_is_vector ?
                                                b.CreateExtractElement(exponent, i) :
                                                exponent;
                    auto element_result = call_scalar_power(
                        base_element, exponent_element);
                    result = b.CreateInsertElement(
                        result, element_result, i);
                }
                return result;
            }
            return call_scalar_power(base, exponent);
        }
        case xir::ArithmeticOp::SQRT: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_unary_fp_intrinsic(v, llvm::Intrinsic::sqrt);
        });
        case xir::ArithmeticOp::RSQRT: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return call_rsqrt(v);
        });
        case xir::ArithmeticOp::CEIL: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::ceil, v);
        });
        case xir::ArithmeticOp::FLOOR: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::floor, v);
        });
        case xir::ArithmeticOp::FRACT: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            auto floor_v = b.CreateUnaryIntrinsic(llvm::Intrinsic::floor, v);
            return b.CreateFSub(v, floor_v);
        });
        case xir::ArithmeticOp::TRUNC: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::trunc, v);
        });
        case xir::ArithmeticOp::ROUND: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::round, v);
        });
        case xir::ArithmeticOp::RINT: return translate_unary([&](auto v) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::rint, v);
        });
        case xir::ArithmeticOp::FMA: return translate_ternary([&](auto x, auto y, auto z) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return b.CreateFMA(x, y, z);
        });
        case xir::ArithmeticOp::COPYSIGN: return translate_binary([&](auto magnitude, auto sign) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_float_or_float_vector());
            return b.CreateBinaryIntrinsic(llvm::Intrinsic::copysign, magnitude, sign);
        });
        case xir::ArithmeticOp::CROSS: return translate_binary([&](auto u, auto v) noexcept {
            LUISA_DEBUG_ASSERT(u->getType()->isVectorTy() && u->getType()->isFPOrFPVectorTy() && u->getType() == v->getType());
            auto poison = llvm::PoisonValue::get(u->getType());
            auto s0 = b.CreateShuffleVector(u, poison, {1, 2, 0});
            auto s1 = b.CreateShuffleVector(v, poison, {2, 0, 1});
            auto s2 = b.CreateShuffleVector(v, poison, {1, 2, 0});
            auto s3 = b.CreateShuffleVector(u, poison, {2, 0, 1});
            auto s01 = b.CreateFMul(s0, s1);
            auto s23 = b.CreateFMul(s2, s3);
            return b.CreateFSub(s01, s23);
        });
        case xir::ArithmeticOp::DOT: {
            auto u = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto v = _get_llvm_value(b, func_ctx, inst->operand(1));
            if (inst->type()->is_float_or_float_vector()) { return dot_product_fp(u, v); }
            LUISA_DEBUG_ASSERT(u->getType()->isVectorTy() && u->getType()->isIntOrIntVectorTy() && u->getType() == v->getType());
            return inst->type()->is_int_or_int_vector() ?
                       b.CreateAddReduce(b.CreateNSWMul(u, v)) :
                       b.CreateAddReduce(b.CreateMul(u, v));
        }
        case xir::ArithmeticOp::LENGTH: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto length_sq = dot_product_fp(v, v);
            return call_unary_fp_intrinsic(length_sq, llvm::Intrinsic::sqrt);
        }
        case xir::ArithmeticOp::LENGTH_SQUARED: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            return dot_product_fp(v, v);
        }
        case xir::ArithmeticOp::NORMALIZE: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto length_sq = dot_product_fp(v, v);
            auto length_rsqrt = call_rsqrt(length_sq);
            auto length_rsqrt_splat = b.CreateVectorSplat(inst->type()->dimension(), length_rsqrt);
            return b.CreateFMul(v, length_rsqrt_splat);
        }
        case xir::ArithmeticOp::FACEFORWARD: {
            auto n = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto i = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto n_ref = _get_llvm_value(b, func_ctx, inst->operand(2));
            auto dot_product = dot_product_fp(i, n_ref);
            auto zero = llvm::Constant::getNullValue(dot_product->getType());
            auto cond = b.CreateFCmpOLT(dot_product, zero);
            auto neg_n = b.CreateFNeg(n);
            return b.CreateSelect(cond, n, neg_n);
        }
        case xir::ArithmeticOp::REFLECT: {
            auto i = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto n = _get_llvm_value(b, func_ctx, inst->operand(1));
            auto dot_product = dot_product_fp(n, i);
            auto neg_two = llvm::ConstantFP::get(dot_product->getType(), -2.);
            auto scale = b.CreateFMul(neg_two, dot_product);
            auto scale_splat = b.CreateVectorSplat(inst->type()->dimension(), scale);
            return b.CreateFMA(scale_splat, n, i);
        }
        case xir::ArithmeticOp::REDUCE_SUM: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(v->getType()->isVectorTy());
            return v->getType()->isFPOrFPVectorTy() ?
                       b.CreateFAddReduce(llvm::ConstantFP::getNegativeZero(v->getType()->getScalarType()), v) :
                       b.CreateAddReduce(v);
        }
        case xir::ArithmeticOp::REDUCE_PRODUCT: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(v->getType()->isVectorTy());
            auto elem_type = v->getType()->getScalarType();
            return v->getType()->isFPOrFPVectorTy() ?
                       b.CreateFMulReduce(llvm::ConstantFP::get(elem_type, 1.), v) :
                       b.CreateMulReduce(v);
        }
        case xir::ArithmeticOp::REDUCE_MIN: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(v->getType()->isVectorTy());
            return inst->type()->is_int()  ? b.CreateIntMinReduce(v, true) :
                   inst->type()->is_uint() ? b.CreateIntMinReduce(v, false) :
                                             b.CreateFPMinReduce(v);
        }
        case xir::ArithmeticOp::REDUCE_MAX: {
            auto v = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(v->getType()->isVectorTy());
            return inst->type()->is_int()  ? b.CreateIntMaxReduce(v, true) :
                   inst->type()->is_uint() ? b.CreateIntMaxReduce(v, false) :
                                             b.CreateFPMaxReduce(v);
        }
        case xir::ArithmeticOp::OUTER_PRODUCT: {
            LUISA_DEBUG_ASSERT(inst->type()->is_matrix() && inst->operand_count() == 2);
            auto lhs = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto rhs = _get_llvm_value(b, func_ctx, inst->operand(1));
            return _translate_outer_product(b, lhs, rhs);
        }
        case xir::ArithmeticOp::MATRIX_COMP_NEG: return translate_matrix_comp_unary_op([&](auto col) noexcept {
            return b.CreateFNeg(col);
        });
        case xir::ArithmeticOp::MATRIX_COMP_ADD: return translate_matrix_comp_binary_op([&](auto lhs_col, auto rhs_col) noexcept {
            return b.CreateFAdd(lhs_col, rhs_col);
        });
        case xir::ArithmeticOp::MATRIX_COMP_SUB: return translate_matrix_comp_binary_op([&](auto lhs_col, auto rhs_col) noexcept {
            return b.CreateFSub(lhs_col, rhs_col);
        });
        case xir::ArithmeticOp::MATRIX_COMP_MUL: return translate_matrix_comp_binary_op([&](auto lhs_col, auto rhs_col) noexcept {
            return b.CreateFMul(lhs_col, rhs_col);
        });
        case xir::ArithmeticOp::MATRIX_COMP_DIV: return translate_matrix_comp_binary_op([&](auto lhs_col, auto rhs_col) noexcept {
            return b.CreateFDiv(lhs_col, rhs_col);
        });
        case xir::ArithmeticOp::MATRIX_LINALG_MUL: {
            auto lhs_type = inst->operand(0)->type();
            auto rhs_type = inst->operand(1)->type();
            LUISA_DEBUG_ASSERT(inst->operand_count() == 2 &&
                                   (lhs_type->is_matrix() || lhs_type->is_float_vector()) &&
                                   (rhs_type->is_matrix() || rhs_type->is_float_vector()) &&
                                   (lhs_type->is_matrix() || rhs_type->is_matrix()) &&
                                   lhs_type->dimension() == rhs_type->dimension() &&
                                   lhs_type->element() == rhs_type->element(),
                               "Invalid types in matrix multiply instruction: {} x {}",
                               lhs_type->description(), rhs_type->description());
            auto lhs = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto rhs = _get_llvm_value(b, func_ctx, inst->operand(1));
            return _translate_matrix_multiply(b, lhs, rhs);
        }
        case xir::ArithmeticOp::MATRIX_DETERMINANT: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == 1 &&
                               inst->operand(0)->type()->is_matrix() &&
                               inst->type() == inst->operand(0)->type()->element());
            auto m = _get_llvm_value(b, func_ctx, inst->operand(0));
            return _translate_matrix_determinant(b, m);
        }
        case xir::ArithmeticOp::MATRIX_TRANSPOSE: return translate_unary([&](auto m) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_matrix());
            return _translate_matrix_transpose(b, m);
        });
        case xir::ArithmeticOp::MATRIX_INVERSE: return translate_unary([&](auto m) noexcept {
            LUISA_DEBUG_ASSERT(inst->type()->is_matrix());
            return _translate_matrix_inverse(b, m);
        });
        case xir::ArithmeticOp::AGGREGATE: return _translate_aggregate(b, func_ctx, inst);
        case xir::ArithmeticOp::SHUFFLE: return _translate_shuffle(b, func_ctx, inst);
        case xir::ArithmeticOp::INSERT: return _translate_insert(b, func_ctx, inst);
        case xir::ArithmeticOp::EXTRACT: return _translate_extract(b, func_ctx, inst);
    }
    LUISA_NOT_IMPLEMENTED();
}

llvm::Value *HIPCodegenLLVMImpl::_translate_outer_product(IB &b, llvm::Value *lhs, llvm::Value *rhs) noexcept {
    LUISA_DEBUG_ASSERT(lhs->getType() == rhs->getType());
    if (auto col_type = llvm::dyn_cast<llvm::VectorType>(lhs->getType())) {
        auto dim = col_type->getElementCount().getFixedValue();
        auto result_type = llvm::ArrayType::get(col_type, dim);
        auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
        for (auto i = 0; i < dim; i++) {
            auto s = b.CreateVectorSplat(dim, b.CreateExtractElement(rhs, i));
            result = b.CreateInsertValue(result, b.CreateFMul(lhs, s), i);
        }
        return result;
    }
    auto rhs_transposed = _translate_matrix_transpose(b, rhs);
    return _translate_matrix_multiply(b, lhs, rhs_transposed);
}

llvm::Value *HIPCodegenLLVMImpl::_translate_matrix_multiply(IB &b, llvm::Value *lhs, llvm::Value *rhs) noexcept {
    if (auto lhs_type = llvm::dyn_cast<llvm::FixedVectorType>(lhs->getType())) {
        auto rhs_type = llvm::cast<llvm::ArrayType>(rhs->getType());
        auto dim = rhs_type->getNumElements();
        LUISA_DEBUG_ASSERT(lhs_type == rhs_type->getElementType() &&
                           lhs_type->getNumElements() == dim);
        auto result = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(lhs_type));
        for (auto i = 0u; i < dim; i++) {
            auto rhs_col = b.CreateExtractValue(rhs, i);
            auto product = b.CreateFMul(lhs, rhs_col);
            auto component = b.CreateFAddReduce(
                llvm::ConstantFP::getNegativeZero(
                    lhs_type->getElementType()),
                product);
            result = b.CreateInsertElement(result, component, i);
        }
        return result;
    }
    LUISA_DEBUG_ASSERT(lhs->getType()->isArrayTy());
    auto dim = lhs->getType()->getArrayNumElements();
    if (rhs->getType()->isVectorTy()) {
        auto result = static_cast<llvm::Value *>(llvm::ConstantFP::getNegativeZero(rhs->getType()));
        for (auto i = 0; i < dim; i++) {
            auto lhs_col = b.CreateExtractValue(lhs, i);
            auto rhs_elem = b.CreateExtractElement(rhs, i);
            auto rhs_elem_splat = b.CreateVectorSplat(dim, rhs_elem);
            result = b.CreateFMA(lhs_col, rhs_elem_splat, result);
        }
        return result;
    }
    LUISA_DEBUG_ASSERT(lhs->getType() == rhs->getType());
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(lhs->getType()));
    for (auto i = 0; i < dim; i++) {
        auto rhs_col = b.CreateExtractValue(rhs, i);
        auto result_col = _translate_matrix_multiply(b, lhs, rhs_col);
        result = b.CreateInsertValue(result, result_col, i);
    }
    return result;
}

llvm::Value *HIPCodegenLLVMImpl::_translate_matrix_determinant(IB &b, llvm::Value *m) noexcept {
    auto scalar_t = m->getType()->getArrayElementType()->getScalarType();
    auto dim = m->getType()->getArrayNumElements();
    auto name = fmt::format("luisa.determinant.{}.{}x{}", _to_string(scalar_t), dim, dim);
    auto func = _llvm_module->getFunction(name);
    if (func == nullptr) {
        auto func_type = llvm::FunctionType::get(scalar_t, {m->getType()}, false);
        func = llvm::Function::Create(func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        auto entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", func);
        IB func_b{entry_bb};
        auto matrix = func->getArg(0);
        auto det = static_cast<llvm::Value *>(nullptr);
        switch (dim) {
            case 2: {
                auto m0 = func_b.CreateExtractValue(matrix, 0);
                auto m1 = func_b.CreateExtractValue(matrix, 1);
                det = detail::hip_codegen_llvm_determinant2x2(func_b, m0, m1);
                break;
            }
            case 3: {
                auto m0 = func_b.CreateExtractValue(matrix, 0);
                auto m1 = func_b.CreateExtractValue(matrix, 1);
                auto m2 = func_b.CreateExtractValue(matrix, 2);
                det = detail::hip_codegen_llvm_determinant3x3(func_b, m0, m1, m2);
                break;
            }
            case 4: {
                auto m0 = func_b.CreateExtractValue(matrix, 0);
                auto m1 = func_b.CreateExtractValue(matrix, 1);
                auto m2 = func_b.CreateExtractValue(matrix, 2);
                auto m3 = func_b.CreateExtractValue(matrix, 3);
                det = detail::hip_codegen_llvm_determinant4x4(func_b, m0, m1, m2, m3);
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Unsupported matrix dimension {} for determinant.", dim);
        }
        func_b.CreateRet(det);
    }
    return b.CreateCall(func, {m});
}

llvm::Value *HIPCodegenLLVMImpl::_translate_matrix_transpose(IB &b, llvm::Value *m) noexcept {
    auto dim = m->getType()->getArrayNumElements();
    auto col_type = llvm::FixedVectorType::get(b.getFloatTy(), dim);
    auto result_type = llvm::ArrayType::get(col_type, dim);
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
    for (auto c = 0u; c < dim; c++) {
        auto transposed_col = static_cast<llvm::Value *>(llvm::PoisonValue::get(col_type));
        for (auto r = 0u; r < dim; r++) {
            auto src_col = b.CreateExtractValue(m, r);
            auto elem = b.CreateExtractElement(src_col, c);
            transposed_col = b.CreateInsertElement(transposed_col, elem, r);
        }
        // build identity shuffle mask for the actual dimension
        llvm::SmallVector<int, 4> mask(dim);
        std::iota(mask.begin(), mask.end(), 0);
        auto transposed_row = b.CreateShuffleVector(transposed_col, mask);
        result = b.CreateInsertValue(result, transposed_row, c);
    }
    return result;
}

llvm::Value *HIPCodegenLLVMImpl::_translate_matrix_inverse(IB &b, llvm::Value *m) noexcept {
    auto scalar_t = m->getType()->getArrayElementType()->getScalarType();
    auto dim = m->getType()->getArrayNumElements();
    auto name = fmt::format("luisa.inverse.{}.{}x{}", _to_string(scalar_t), dim, dim);
    auto func = _llvm_module->getFunction(name);
    if (func == nullptr) {
        auto func_type = llvm::FunctionType::get(m->getType(), {m->getType()}, false);
        func = llvm::Function::Create(func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
        func->addFnAttr(llvm::Attribute::AlwaysInline);
        auto entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", func);
        IB func_b{entry_bb};
        auto matrix = func->getArg(0);
        auto inv = static_cast<llvm::Value *>(nullptr);
        switch (dim) {
            case 2: {
                auto m0 = func_b.CreateExtractValue(matrix, 0);
                auto m1 = func_b.CreateExtractValue(matrix, 1);
                inv = detail::hip_codegen_llvm_inverse2x2(func_b, m0, m1);
                break;
            }
            case 3: {
                auto m0 = func_b.CreateExtractValue(matrix, 0);
                auto m1 = func_b.CreateExtractValue(matrix, 1);
                auto m2 = func_b.CreateExtractValue(matrix, 2);
                inv = detail::hip_codegen_llvm_inverse3x3(func_b, m0, m1, m2);
                break;
            }
            case 4: {
                auto m0 = func_b.CreateExtractValue(matrix, 0);
                auto m1 = func_b.CreateExtractValue(matrix, 1);
                auto m2 = func_b.CreateExtractValue(matrix, 2);
                auto m3 = func_b.CreateExtractValue(matrix, 3);
                inv = detail::hip_codegen_llvm_inverse4x4(func_b, m0, m1, m2, m3);
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Unsupported matrix dimension {} for inverse.", dim);
        }
        func_b.CreateRet(inv);
    }
    return b.CreateCall(func, {m});
}

llvm::Value *HIPCodegenLLVMImpl::_translate_aggregate(IB &b, const FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    auto llvm_result_type = _get_llvm_type(inst->type());
    auto llvm_result = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_result_type->reg_type));
    switch (inst->type()->tag()) {
        case Type::Tag::VECTOR: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == inst->type()->dimension());
            auto llvm_elem_type = llvm::cast<llvm::VectorType>(llvm_result_type->reg_type)->getElementType();
            for (auto i = 0; i < inst->operand_count(); i++) {
                auto llvm_elem = _get_llvm_value(b, func_ctx, inst->operand(i));
                LUISA_DEBUG_ASSERT(llvm_elem->getType() == llvm_elem_type);
                llvm_result = b.CreateInsertElement(llvm_result, llvm_elem, i);
            }
            return llvm_result;
        }
        case Type::Tag::MATRIX: [[fallthrough]];
        case Type::Tag::ARRAY: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == inst->type()->dimension());
            auto llvm_elem_type = llvm_result_type->reg_type->getArrayElementType();
            for (auto i = 0; i < inst->operand_count(); i++) {
                auto llvm_elem = _get_llvm_value(b, func_ctx, inst->operand(i));
                LUISA_DEBUG_ASSERT(llvm_elem->getType() == llvm_elem_type);
                llvm_result = b.CreateInsertValue(llvm_result, llvm_elem, i);
            }
            return llvm_result;
        }
        case Type::Tag::STRUCTURE: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == inst->type()->members().size());
            for (auto i = 0; i < inst->operand_count(); i++) {
                auto llvm_elem = _get_llvm_value(b, func_ctx, inst->operand(i));
                LUISA_DEBUG_ASSERT(llvm_elem->getType() == llvm_result_type->reg_type->getStructElementType(i));
                llvm_result = b.CreateInsertValue(llvm_result, llvm_elem, i);
            }
            return llvm_result;
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported aggregate type {} in LLVM codegen.", inst->type()->description());
}

llvm::Value *HIPCodegenLLVMImpl::_translate_shuffle(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    LUISA_DEBUG_ASSERT(inst->type()->is_vector());
    auto llvm_src = _get_llvm_value(b, func_ctx, inst->operand(0));
    auto index_uses = inst->operand_uses().subspan(1);
    LUISA_DEBUG_ASSERT(index_uses.size() == inst->type()->dimension());
    auto evaluate_constant = [](const xir::Constant *c) noexcept -> uint64_t {
        switch (c->type()->tag()) {
            case Type::Tag::INT8: return c->as<int8_t>();
            case Type::Tag::UINT8: return c->as<uint8_t>();
            case Type::Tag::INT16: return c->as<int16_t>();
            case Type::Tag::UINT16: return c->as<uint16_t>();
            case Type::Tag::INT32: return c->as<int32_t>();
            case Type::Tag::UINT32: return c->as<uint32_t>();
            case Type::Tag::INT64: return c->as<int64_t>();
            case Type::Tag::UINT64: return c->as<uint64_t>();
            default: LUISA_ERROR_WITH_LOCATION("Unsupported constant type for shuffle evaluation.");
        }
    };
    if (std::all_of(index_uses.begin(), index_uses.end(), [](const xir::Use *index_use) noexcept {
            return index_use->value()->isa<xir::Constant>();
        })) {
        llvm::SmallVector<int> llvm_indices;
        llvm_indices.reserve(index_uses.size());
        for (auto index_use : index_uses) {
            auto static_index = static_cast<const xir::Constant *>(index_use->value());
            llvm_indices.emplace_back(static_cast<int>(evaluate_constant(static_index)));
        }
        return b.CreateShuffleVector(llvm_src, llvm_indices);
    }
    auto llvm_dst_type = _get_llvm_type(inst->type())->reg_type;
    LUISA_DEBUG_ASSERT(llvm_src->getType()->getScalarType() == llvm_dst_type->getScalarType());
    auto llvm_dst = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_dst_type));
    for (auto [i, index_use] : llvm::enumerate(std::span{index_uses.data(), index_uses.size()})) {
        auto llvm_index = _get_llvm_value(b, func_ctx, index_use->value());
        auto llvm_src_elem = b.CreateExtractElement(llvm_src, llvm_index);
        llvm_dst = b.CreateInsertElement(llvm_dst, llvm_src_elem, i);
    }
    return llvm_dst;
}

llvm::Value *HIPCodegenLLVMImpl::_translate_insert(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    auto llvm_src = _get_llvm_value(b, func_ctx, inst->operand(0));
    LUISA_DEBUG_ASSERT(inst->type() == inst->operand(0)->type());
    auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(1));
    auto index_uses = inst->operand_uses().subspan(2);
    if (llvm_src->getType()->isVectorTy()) {
        LUISA_DEBUG_ASSERT(index_uses.size() == 1);
        auto llvm_index = _get_llvm_value(b, func_ctx, index_uses.front()->value());
        return b.CreateInsertElement(llvm_src, llvm_value, llvm_index);
    }
    LUISA_DEBUG_ASSERT(llvm_src->getType()->isAggregateType());
    if (index_uses.size() == 1 && index_uses.front()->value()->isa<xir::Constant>()) {
        auto index = static_cast<const xir::Constant *>(index_uses.front()->value());
        auto evaluate_constant = [](const xir::Constant *c) noexcept -> uint64_t {
            switch (c->type()->tag()) {
                case Type::Tag::INT8: return c->as<int8_t>();
                case Type::Tag::UINT8: return c->as<uint8_t>();
                case Type::Tag::INT16: return c->as<int16_t>();
                case Type::Tag::UINT16: return c->as<uint16_t>();
                case Type::Tag::INT32: return c->as<int32_t>();
                case Type::Tag::UINT32: return c->as<uint32_t>();
                case Type::Tag::INT64: return c->as<int64_t>();
                case Type::Tag::UINT64: return c->as<uint64_t>();
                default: LUISA_ERROR_WITH_LOCATION("Unsupported constant type for insert evaluation.");
            }
        };
        auto static_index = static_cast<unsigned>(evaluate_constant(index));
        return b.CreateInsertValue(llvm_src, llvm_value, static_index);
    }
    auto llvm_src_mem = _convert_llvm_reg_value_to_mem(b, llvm_src, inst->type());
    auto llvm_temp = _create_temp_in_alloca_block(func_ctx, llvm_src_mem->getType(), inst->type()->alignment());
    b.CreateStore(llvm_src_mem, llvm_temp);
    auto [llvm_ptr, elem_type] = _lower_access_chain_address(b, func_ctx, llvm_temp, inst->type(), index_uses);
    LUISA_DEBUG_ASSERT(elem_type == inst->operand(1)->type());
    _store_llvm_value(b, llvm_ptr, llvm_value, elem_type);
    return _load_llvm_value(b, llvm_temp, inst->type());
}

llvm::Value *HIPCodegenLLVMImpl::_translate_extract(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept {
    auto evaluate_constant = [](const xir::Constant *c) noexcept -> uint64_t {
        switch (c->type()->tag()) {
            case Type::Tag::INT8: return c->as<int8_t>();
            case Type::Tag::UINT8: return c->as<uint8_t>();
            case Type::Tag::INT16: return c->as<int16_t>();
            case Type::Tag::UINT16: return c->as<uint16_t>();
            case Type::Tag::INT32: return c->as<int32_t>();
            case Type::Tag::UINT32: return c->as<uint32_t>();
            case Type::Tag::INT64: return c->as<int64_t>();
            case Type::Tag::UINT64: return c->as<uint64_t>();
            default: LUISA_ERROR_WITH_LOCATION("Unsupported constant type for extract evaluation.");
        }
    };
    if (auto src = inst->operand(0); src->isa<xir::Constant>() && src->type()->is_array()) {
        auto llvm_global = _get_llvm_constant(b, static_cast<const xir::Constant *>(src), false);
        LUISA_DEBUG_ASSERT(llvm_global->getType()->isPointerTy());
        auto index_uses = inst->operand_uses().subspan(1);
        auto [llvm_ptr, elem_type] = _lower_access_chain_address(b, func_ctx, llvm_global, src->type(), index_uses);
        LUISA_DEBUG_ASSERT(elem_type == inst->type());
        return _load_llvm_value(b, llvm_ptr, elem_type);
    }
    auto llvm_src = _get_llvm_value(b, func_ctx, inst->operand(0));
    auto index_uses = inst->operand_uses().subspan(1);
    if (llvm_src->getType()->isVectorTy()) {
        LUISA_DEBUG_ASSERT(index_uses.size() == 1);
        auto llvm_index = _get_llvm_value(b, func_ctx, index_uses.front()->value());
        return b.CreateExtractElement(llvm_src, llvm_index);
    }
    LUISA_DEBUG_ASSERT(llvm_src->getType()->isAggregateType());
    if (index_uses.size() == 1 && index_uses.front()->value()->isa<xir::Constant>()) {
        auto index = static_cast<const xir::Constant *>(index_uses.front()->value());
        auto static_index = static_cast<unsigned>(evaluate_constant(index));
        return b.CreateExtractValue(llvm_src, static_index);
    }
    auto llvm_src_mem = _convert_llvm_reg_value_to_mem(b, llvm_src, inst->operand(0)->type());
    auto llvm_temp = _create_temp_in_alloca_block(func_ctx, llvm_src_mem->getType(), inst->operand(0)->type()->alignment());
    b.CreateStore(llvm_src_mem, llvm_temp);
    auto [llvm_ptr, elem_type] = _lower_access_chain_address(b, func_ctx, llvm_temp, inst->operand(0)->type(), index_uses);
    LUISA_DEBUG_ASSERT(elem_type == inst->type());
    return _load_llvm_value(b, llvm_ptr, elem_type);
}

namespace detail {

[[nodiscard]] llvm::Function *find_ocml_function(llvm::Module &m, llvm::StringRef op_name,
                                                  llvm::Type *t, bool enable_fast_math) noexcept {
    auto suffix = t->isDoubleTy() ? "_f64" : t->isHalfTy() ? "_f16" :
                                                             "_f32";
    auto ocml_name = fmt::format("__ocml_{}{}", std::string_view{op_name}, suffix);
    auto op = m.getFunction(ocml_name);
    LUISA_ASSERT(op != nullptr, "OCML function {} not found.", ocml_name);
    return op;
}

[[nodiscard]] llvm::Value *hip_codegen_llvm_determinant2x2(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1) noexcept {
    auto m00 = b.CreateExtractElement(m0, b.getInt32(0));
    auto m01 = b.CreateExtractElement(m0, b.getInt32(1));
    auto m10 = b.CreateExtractElement(m1, b.getInt32(0));
    auto m11 = b.CreateExtractElement(m1, b.getInt32(1));
    return b.CreateFSub(b.CreateFMul(m00, m11), b.CreateFMul(m10, m01));
}

[[nodiscard]] llvm::Value *hip_codegen_llvm_determinant3x3(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1, llvm::Value *m2) noexcept {
    auto m00 = b.CreateExtractElement(m0, b.getInt32(0));
    auto m01 = b.CreateExtractElement(m0, b.getInt32(1));
    auto m02 = b.CreateExtractElement(m0, b.getInt32(2));
    auto m10 = b.CreateExtractElement(m1, b.getInt32(0));
    auto m11 = b.CreateExtractElement(m1, b.getInt32(1));
    auto m12 = b.CreateExtractElement(m1, b.getInt32(2));
    auto m20 = b.CreateExtractElement(m2, b.getInt32(0));
    auto m21 = b.CreateExtractElement(m2, b.getInt32(1));
    auto m22 = b.CreateExtractElement(m2, b.getInt32(2));
    auto term0 = b.CreateFMul(m00, b.CreateFSub(b.CreateFMul(m11, m22), b.CreateFMul(m21, m12)));
    auto term1 = b.CreateFMul(m10, b.CreateFSub(b.CreateFMul(m21, m02), b.CreateFMul(m01, m22)));
    auto term2 = b.CreateFMul(m20, b.CreateFSub(b.CreateFMul(m01, m12), b.CreateFMul(m11, m02)));
    return b.CreateFAdd(b.CreateFAdd(term0, term1), term2);
}

[[nodiscard]] llvm::Value *hip_codegen_llvm_determinant4x4(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1, llvm::Value *m2, llvm::Value *m3) noexcept {
    auto make_vec4 = [&](auto e0, auto e1, auto e2, auto e3) noexcept {
        auto v = static_cast<llvm::Value *>(llvm::PoisonValue::get(m0->getType()));
        v = b.CreateInsertElement(v, e0, b.getInt32(0));
        v = b.CreateInsertElement(v, e1, b.getInt32(1));
        v = b.CreateInsertElement(v, e2, b.getInt32(2));
        v = b.CreateInsertElement(v, e3, b.getInt32(3));
        return v;
    };
    auto m00 = b.CreateExtractElement(m0, b.getInt32(0));
    auto m01 = b.CreateExtractElement(m0, b.getInt32(1));
    auto m02 = b.CreateExtractElement(m0, b.getInt32(2));
    auto m03 = b.CreateExtractElement(m0, b.getInt32(3));
    auto m10 = b.CreateExtractElement(m1, b.getInt32(0));
    auto m11 = b.CreateExtractElement(m1, b.getInt32(1));
    auto m12 = b.CreateExtractElement(m1, b.getInt32(2));
    auto m13 = b.CreateExtractElement(m1, b.getInt32(3));
    auto m20 = b.CreateExtractElement(m2, b.getInt32(0));
    auto m21 = b.CreateExtractElement(m2, b.getInt32(1));
    auto m22 = b.CreateExtractElement(m2, b.getInt32(2));
    auto m23 = b.CreateExtractElement(m2, b.getInt32(3));
    auto m30 = b.CreateExtractElement(m3, b.getInt32(0));
    auto m31 = b.CreateExtractElement(m3, b.getInt32(1));
    auto m32 = b.CreateExtractElement(m3, b.getInt32(2));
    auto m33 = b.CreateExtractElement(m3, b.getInt32(3));
    auto coef00 = b.CreateFSub(b.CreateFMul(m22, m33), b.CreateFMul(m32, m23));
    auto coef02 = b.CreateFSub(b.CreateFMul(m12, m33), b.CreateFMul(m32, m13));
    auto coef03 = b.CreateFSub(b.CreateFMul(m12, m23), b.CreateFMul(m22, m13));
    auto coef04 = b.CreateFSub(b.CreateFMul(m21, m33), b.CreateFMul(m31, m23));
    auto coef06 = b.CreateFSub(b.CreateFMul(m11, m33), b.CreateFMul(m31, m13));
    auto coef07 = b.CreateFSub(b.CreateFMul(m11, m23), b.CreateFMul(m21, m13));
    auto coef08 = b.CreateFSub(b.CreateFMul(m21, m32), b.CreateFMul(m31, m22));
    auto coef10 = b.CreateFSub(b.CreateFMul(m11, m32), b.CreateFMul(m31, m12));
    auto coef11 = b.CreateFSub(b.CreateFMul(m11, m22), b.CreateFMul(m21, m12));
    auto coef12 = b.CreateFSub(b.CreateFMul(m20, m33), b.CreateFMul(m30, m23));
    auto coef14 = b.CreateFSub(b.CreateFMul(m10, m33), b.CreateFMul(m30, m13));
    auto coef15 = b.CreateFSub(b.CreateFMul(m10, m23), b.CreateFMul(m20, m13));
    auto coef16 = b.CreateFSub(b.CreateFMul(m20, m32), b.CreateFMul(m30, m22));
    auto coef18 = b.CreateFSub(b.CreateFMul(m10, m32), b.CreateFMul(m30, m12));
    auto coef19 = b.CreateFSub(b.CreateFMul(m10, m22), b.CreateFMul(m20, m12));
    auto coef20 = b.CreateFSub(b.CreateFMul(m20, m31), b.CreateFMul(m30, m21));
    auto coef22 = b.CreateFSub(b.CreateFMul(m10, m31), b.CreateFMul(m30, m11));
    auto coef23 = b.CreateFSub(b.CreateFMul(m10, m21), b.CreateFMul(m20, m11));
    auto fac0 = make_vec4(coef00, coef00, coef02, coef03);
    auto fac1 = make_vec4(coef04, coef04, coef06, coef07);
    auto fac2 = make_vec4(coef08, coef08, coef10, coef11);
    auto fac3 = make_vec4(coef12, coef12, coef14, coef15);
    auto fac4 = make_vec4(coef16, coef16, coef18, coef19);
    auto fac5 = make_vec4(coef20, coef20, coef22, coef23);
    auto vec0 = make_vec4(m10, m00, m00, m00);
    auto vec1 = make_vec4(m11, m01, m01, m01);
    auto vec2 = make_vec4(m12, m02, m02, m02);
    auto vec3 = make_vec4(m13, m03, m03, m03);
    auto inv0 = b.CreateFAdd(b.CreateFSub(b.CreateFMul(vec1, fac0), b.CreateFMul(vec2, fac1)), b.CreateFMul(vec3, fac2));
    auto inv1 = b.CreateFAdd(b.CreateFSub(b.CreateFMul(vec0, fac0), b.CreateFMul(vec2, fac3)), b.CreateFMul(vec3, fac4));
    auto inv2 = b.CreateFAdd(b.CreateFSub(b.CreateFMul(vec0, fac1), b.CreateFMul(vec1, fac3)), b.CreateFMul(vec3, fac5));
    auto inv3 = b.CreateFAdd(b.CreateFSub(b.CreateFMul(vec0, fac2), b.CreateFMul(vec1, fac4)), b.CreateFMul(vec2, fac5));
    auto minus_one = llvm::ConstantFP::get(m00->getType(), -1.);
    auto one = llvm::ConstantFP::get(m00->getType(), 1.);
    auto sign_a = llvm::ConstantVector::get({one, minus_one, one, minus_one});
    auto sign_b = llvm::ConstantVector::get({minus_one, one, minus_one, one});
    auto inv_0 = b.CreateFMul(inv0, sign_a);
    auto inv_1 = b.CreateFMul(inv1, sign_b);
    auto inv_2 = b.CreateFMul(inv2, sign_a);
    auto inv_3 = b.CreateFMul(inv3, sign_b);
    auto inv_0_x = b.CreateExtractElement(inv_0, b.getInt32(0));
    auto inv_1_x = b.CreateExtractElement(inv_1, b.getInt32(0));
    auto inv_2_x = b.CreateExtractElement(inv_2, b.getInt32(0));
    auto inv_3_x = b.CreateExtractElement(inv_3, b.getInt32(0));
    auto dot0 = b.CreateFMul(m0, make_vec4(inv_0_x, inv_1_x, inv_2_x, inv_3_x));
    return b.CreateFAddReduce(llvm::ConstantFP::getNegativeZero(m00->getType()), dot0);
}

[[nodiscard]] llvm::Value *hip_codegen_llvm_inverse2x2(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1) noexcept {
    auto m00 = b.CreateExtractElement(m0, b.getInt32(0));
    auto m01 = b.CreateExtractElement(m0, b.getInt32(1));
    auto m10 = b.CreateExtractElement(m1, b.getInt32(0));
    auto m11 = b.CreateExtractElement(m1, b.getInt32(1));
    auto det = b.CreateFSub(b.CreateFMul(m00, m11), b.CreateFMul(m10, m01));
    auto one = llvm::ConstantFP::get(m00->getType(), 1.);
    auto one_over_det = b.CreateFDiv(one, det);
    auto minus_one_over_det = b.CreateFNeg(one_over_det);
    auto im0 = static_cast<llvm::Value *>(llvm::PoisonValue::get(m0->getType()));
    im0 = b.CreateInsertElement(im0, b.CreateFMul(m11, one_over_det), b.getInt32(0));
    im0 = b.CreateInsertElement(im0, b.CreateFMul(m01, minus_one_over_det), b.getInt32(1));
    auto im1 = static_cast<llvm::Value *>(llvm::PoisonValue::get(m1->getType()));
    im1 = b.CreateInsertElement(im1, b.CreateFMul(m10, minus_one_over_det), b.getInt32(0));
    im1 = b.CreateInsertElement(im1, b.CreateFMul(m00, one_over_det), b.getInt32(1));
    auto im = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm::ArrayType::get(m0->getType(), 2)));
    im = b.CreateInsertValue(im, im0, 0);
    im = b.CreateInsertValue(im, im1, 1);
    return im;
}

[[nodiscard]] llvm::Value *hip_codegen_llvm_inverse3x3(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1, llvm::Value *m2) noexcept {
    auto m00 = b.CreateExtractElement(m0, b.getInt32(0));
    auto m01 = b.CreateExtractElement(m0, b.getInt32(1));
    auto m02 = b.CreateExtractElement(m0, b.getInt32(2));
    auto m10 = b.CreateExtractElement(m1, b.getInt32(0));
    auto m11 = b.CreateExtractElement(m1, b.getInt32(1));
    auto m12 = b.CreateExtractElement(m1, b.getInt32(2));
    auto m20 = b.CreateExtractElement(m2, b.getInt32(0));
    auto m21 = b.CreateExtractElement(m2, b.getInt32(1));
    auto m22 = b.CreateExtractElement(m2, b.getInt32(2));
    auto term0 = b.CreateFMul(m00, b.CreateFSub(b.CreateFMul(m11, m22), b.CreateFMul(m21, m12)));
    auto term1 = b.CreateFMul(m10, b.CreateFSub(b.CreateFMul(m21, m02), b.CreateFMul(m01, m22)));
    auto term2 = b.CreateFMul(m20, b.CreateFSub(b.CreateFMul(m01, m12), b.CreateFMul(m11, m02)));
    auto det = b.CreateFAdd(b.CreateFAdd(term0, term1), term2);
    auto one = llvm::ConstantFP::get(m00->getType(), 1.);
    auto one_over_det = b.CreateFDiv(one, det);
    auto im00 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m11, m22), b.CreateFMul(m21, m12)), one_over_det);
    auto im01 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m21, m02), b.CreateFMul(m01, m22)), one_over_det);
    auto im02 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m01, m12), b.CreateFMul(m11, m02)), one_over_det);
    auto im10 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m20, m12), b.CreateFMul(m10, m22)), one_over_det);
    auto im11 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m00, m22), b.CreateFMul(m20, m02)), one_over_det);
    auto im12 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m10, m02), b.CreateFMul(m00, m12)), one_over_det);
    auto im20 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m10, m21), b.CreateFMul(m20, m11)), one_over_det);
    auto im21 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m20, m01), b.CreateFMul(m00, m21)), one_over_det);
    auto im22 = b.CreateFMul(b.CreateFSub(b.CreateFMul(m00, m11), b.CreateFMul(m10, m01)), one_over_det);
    auto im0 = static_cast<llvm::Value *>(llvm::PoisonValue::get(m0->getType()));
    im0 = b.CreateInsertElement(im0, im00, b.getInt32(0));
    im0 = b.CreateInsertElement(im0, im01, b.getInt32(1));
    im0 = b.CreateInsertElement(im0, im02, b.getInt32(2));
    auto im1 = static_cast<llvm::Value *>(llvm::PoisonValue::get(m1->getType()));
    im1 = b.CreateInsertElement(im1, im10, b.getInt32(0));
    im1 = b.CreateInsertElement(im1, im11, b.getInt32(1));
    im1 = b.CreateInsertElement(im1, im12, b.getInt32(2));
    auto im2 = static_cast<llvm::Value *>(llvm::PoisonValue::get(m2->getType()));
    im2 = b.CreateInsertElement(im2, im20, b.getInt32(0));
    im2 = b.CreateInsertElement(im2, im21, b.getInt32(1));
    im2 = b.CreateInsertElement(im2, im22, b.getInt32(2));
    auto im = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm::ArrayType::get(m0->getType(), 3)));
    im = b.CreateInsertValue(im, im0, 0);
    im = b.CreateInsertValue(im, im1, 1);
    im = b.CreateInsertValue(im, im2, 2);
    return im;
}

[[nodiscard]] llvm::Value *hip_codegen_llvm_inverse4x4(HIPCodegenLLVMImpl::IB &b, llvm::Value *m0, llvm::Value *m1, llvm::Value *m2, llvm::Value *m3) noexcept {
    auto m00 = b.CreateExtractElement(m0, b.getInt32(0));
    auto m01 = b.CreateExtractElement(m0, b.getInt32(1));
    auto m02 = b.CreateExtractElement(m0, b.getInt32(2));
    auto m03 = b.CreateExtractElement(m0, b.getInt32(3));
    auto m10 = b.CreateExtractElement(m1, b.getInt32(0));
    auto m11 = b.CreateExtractElement(m1, b.getInt32(1));
    auto m12 = b.CreateExtractElement(m1, b.getInt32(2));
    auto m13 = b.CreateExtractElement(m1, b.getInt32(3));
    auto m20 = b.CreateExtractElement(m2, b.getInt32(0));
    auto m21 = b.CreateExtractElement(m2, b.getInt32(1));
    auto m22 = b.CreateExtractElement(m2, b.getInt32(2));
    auto m23 = b.CreateExtractElement(m2, b.getInt32(3));
    auto m30 = b.CreateExtractElement(m3, b.getInt32(0));
    auto m31 = b.CreateExtractElement(m3, b.getInt32(1));
    auto m32 = b.CreateExtractElement(m3, b.getInt32(2));
    auto m33 = b.CreateExtractElement(m3, b.getInt32(3));
    auto coef00 = b.CreateFSub(b.CreateFMul(m22, m33), b.CreateFMul(m32, m23));
    auto coef02 = b.CreateFSub(b.CreateFMul(m12, m33), b.CreateFMul(m32, m13));
    auto coef03 = b.CreateFSub(b.CreateFMul(m12, m23), b.CreateFMul(m22, m13));
    auto coef04 = b.CreateFSub(b.CreateFMul(m21, m33), b.CreateFMul(m31, m23));
    auto coef06 = b.CreateFSub(b.CreateFMul(m11, m33), b.CreateFMul(m31, m13));
    auto coef07 = b.CreateFSub(b.CreateFMul(m11, m23), b.CreateFMul(m21, m13));
    auto coef08 = b.CreateFSub(b.CreateFMul(m21, m32), b.CreateFMul(m31, m22));
    auto coef10 = b.CreateFSub(b.CreateFMul(m11, m32), b.CreateFMul(m31, m12));
    auto coef11 = b.CreateFSub(b.CreateFMul(m11, m22), b.CreateFMul(m21, m12));
    auto coef12 = b.CreateFSub(b.CreateFMul(m20, m33), b.CreateFMul(m30, m23));
    auto coef14 = b.CreateFSub(b.CreateFMul(m10, m33), b.CreateFMul(m30, m13));
    auto coef15 = b.CreateFSub(b.CreateFMul(m10, m23), b.CreateFMul(m20, m13));
    auto coef16 = b.CreateFSub(b.CreateFMul(m20, m32), b.CreateFMul(m30, m22));
    auto coef18 = b.CreateFSub(b.CreateFMul(m10, m32), b.CreateFMul(m30, m12));
    auto coef19 = b.CreateFSub(b.CreateFMul(m10, m22), b.CreateFMul(m20, m12));
    auto coef20 = b.CreateFSub(b.CreateFMul(m20, m31), b.CreateFMul(m30, m21));
    auto coef22 = b.CreateFSub(b.CreateFMul(m10, m31), b.CreateFMul(m30, m11));
    auto coef23 = b.CreateFSub(b.CreateFMul(m10, m21), b.CreateFMul(m20, m11));
    auto make_vec4 = [&](auto e0, auto e1, auto e2, auto e3) noexcept {
        auto v = static_cast<llvm::Value *>(llvm::PoisonValue::get(m0->getType()));
        v = b.CreateInsertElement(v, e0, b.getInt32(0));
        v = b.CreateInsertElement(v, e1, b.getInt32(1));
        v = b.CreateInsertElement(v, e2, b.getInt32(2));
        v = b.CreateInsertElement(v, e3, b.getInt32(3));
        return v;
    };
    auto fac0 = make_vec4(coef00, coef00, coef02, coef03);
    auto fac1 = make_vec4(coef04, coef04, coef06, coef07);
    auto fac2 = make_vec4(coef08, coef08, coef10, coef11);
    auto fac3 = make_vec4(coef12, coef12, coef14, coef15);
    auto fac4 = make_vec4(coef16, coef16, coef18, coef19);
    auto fac5 = make_vec4(coef20, coef20, coef22, coef23);
    auto vec0 = make_vec4(m10, m00, m00, m00);
    auto vec1 = make_vec4(m11, m01, m01, m01);
    auto vec2 = make_vec4(m12, m02, m02, m02);
    auto vec3 = make_vec4(m13, m03, m03, m03);
    auto inv0 = b.CreateFAdd(b.CreateFSub(b.CreateFMul(vec1, fac0), b.CreateFMul(vec2, fac1)), b.CreateFMul(vec3, fac2));
    auto inv1 = b.CreateFAdd(b.CreateFSub(b.CreateFMul(vec0, fac0), b.CreateFMul(vec2, fac3)), b.CreateFMul(vec3, fac4));
    auto inv2 = b.CreateFAdd(b.CreateFSub(b.CreateFMul(vec0, fac1), b.CreateFMul(vec1, fac3)), b.CreateFMul(vec3, fac5));
    auto inv3 = b.CreateFAdd(b.CreateFSub(b.CreateFMul(vec0, fac2), b.CreateFMul(vec1, fac4)), b.CreateFMul(vec2, fac5));
    auto minus_one = llvm::ConstantFP::get(m00->getType(), -1.);
    auto one = llvm::ConstantFP::get(m00->getType(), 1.);
    auto sign_a = llvm::ConstantVector::get({one, minus_one, one, minus_one});
    auto sign_b = llvm::ConstantVector::get({minus_one, one, minus_one, one});
    auto inv_0 = b.CreateFMul(inv0, sign_a);
    auto inv_1 = b.CreateFMul(inv1, sign_b);
    auto inv_2 = b.CreateFMul(inv2, sign_a);
    auto inv_3 = b.CreateFMul(inv3, sign_b);
    auto inv_0_x = b.CreateExtractElement(inv_0, b.getInt32(0));
    auto inv_1_x = b.CreateExtractElement(inv_1, b.getInt32(0));
    auto inv_2_x = b.CreateExtractElement(inv_2, b.getInt32(0));
    auto inv_3_x = b.CreateExtractElement(inv_3, b.getInt32(0));
    auto dot0 = b.CreateFMul(m0, make_vec4(inv_0_x, inv_1_x, inv_2_x, inv_3_x));
    auto dot1 = b.CreateFAddReduce(llvm::ConstantFP::getNegativeZero(m00->getType()), dot0);
    auto one_over_det = b.CreateFDiv(one, dot1);
    one_over_det = b.CreateVectorSplat(4, one_over_det);
    auto im0 = b.CreateFMul(inv_0, one_over_det);
    auto im1 = b.CreateFMul(inv_1, one_over_det);
    auto im2 = b.CreateFMul(inv_2, one_over_det);
    auto im3 = b.CreateFMul(inv_3, one_over_det);
    auto im = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm::ArrayType::get(m0->getType(), 4)));
    im = b.CreateInsertValue(im, im0, 0);
    im = b.CreateInsertValue(im, im1, 1);
    im = b.CreateInsertValue(im, im2, 2);
    im = b.CreateInsertValue(im, im3, 3);
    return im;
}

}// namespace detail

llvm::Value *HIPCodegenLLVMImpl::_call_ocml_unary_op(IB &b, llvm::StringRef op_name, llvm::Value *llvm_value) const noexcept {
    auto llvm_scalar_t = llvm_value->getType()->getScalarType();
    auto op = detail::find_ocml_function(*_llvm_module, op_name, llvm_scalar_t, _config.enable_fast_math);
    auto should_cast_to_float = !llvm_scalar_t->isFloatTy() && !llvm_scalar_t->isDoubleTy();
    auto call_scalar = [&](llvm::Value *llvm_elem) noexcept {
        if (should_cast_to_float) { llvm_elem = _safe_fp_cast(b, llvm_elem, llvm::Type::getFloatTy(b.getContext())); }
        auto llvm_res_elem = static_cast<llvm::Value *>(b.CreateCall(op, {llvm_elem}));
        if (should_cast_to_float) { llvm_res_elem = _safe_fp_cast(b, llvm_res_elem, llvm_scalar_t); }
        return llvm_res_elem;
    };
    if (auto vt = llvm::dyn_cast<llvm::VectorType>(llvm_value->getType())) {
        auto dim = vt->getElementCount().getFixedValue();
        auto llvm_result = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_value->getType()));
        for (auto i = 0; i < dim; i++) {
            auto llvm_elem = call_scalar(b.CreateExtractElement(llvm_value, i));
            llvm_result = b.CreateInsertElement(llvm_result, llvm_elem, i);
        }
        return llvm_result;
    }
    return call_scalar(llvm_value);
}

llvm::Value *HIPCodegenLLVMImpl::_call_ocml_binary_op(IB &b, llvm::StringRef op_name, llvm::Value *llvm_lhs, llvm::Value *llvm_rhs) noexcept {
    auto llvm_lhs_scalar_t = llvm_lhs->getType()->getScalarType();
    auto llvm_rhs_scalar_t = llvm_rhs->getType()->getScalarType();
    auto op = detail::find_ocml_function(*_llvm_module, op_name, llvm_lhs_scalar_t, _config.enable_fast_math);
    auto lhs_should_cast_to_float = llvm_lhs_scalar_t->isFloatingPointTy() && !llvm_lhs_scalar_t->isFloatTy() && !llvm_lhs_scalar_t->isDoubleTy();
    auto rhs_should_cast_to_float = llvm_rhs_scalar_t->isFloatingPointTy() && !llvm_rhs_scalar_t->isFloatTy() && !llvm_rhs_scalar_t->isDoubleTy();
    auto call_scalar = [&](llvm::Value *llvm_lhs_elem, llvm::Value *llvm_rhs_elem) noexcept {
        if (lhs_should_cast_to_float) { llvm_lhs_elem = _safe_fp_cast(b, llvm_lhs_elem, llvm::Type::getFloatTy(b.getContext())); }
        if (rhs_should_cast_to_float) { llvm_rhs_elem = _safe_fp_cast(b, llvm_rhs_elem, llvm::Type::getFloatTy(b.getContext())); }
        auto llvm_res_elem = static_cast<llvm::Value *>(b.CreateCall(op, {llvm_lhs_elem, llvm_rhs_elem}));
        if (lhs_should_cast_to_float) { llvm_res_elem = _safe_fp_cast(b, llvm_res_elem, llvm_lhs_scalar_t); }
        return llvm_res_elem;
    };
    if (auto vt = llvm::dyn_cast<llvm::VectorType>(llvm_lhs->getType())) {
        auto dim = vt->getElementCount().getFixedValue();
        auto llvm_result = static_cast<llvm::Value *>(llvm::PoisonValue::get(vt));
        for (auto i = 0; i < dim; i++) {
            auto llvm_lhs_elem = b.CreateExtractElement(llvm_lhs, i);
            auto llvm_rhs_elem = b.CreateExtractElement(llvm_rhs, i);
            auto llvm_elem = call_scalar(llvm_lhs_elem, llvm_rhs_elem);
            llvm_result = b.CreateInsertElement(llvm_result, llvm_elem, i);
        }
        return llvm_result;
    }
    return call_scalar(llvm_lhs, llvm_rhs);
}

}// namespace luisa::compute::hip
