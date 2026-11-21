#include "llvm_schedule_emitter.h"

#include "../../common/llvm_native_math.h"
#include <llvm/IR/MDBuilder.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::simd::detail {

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_select_data(
    ::llvm::Value *condition, ::llvm::Value *true_value,
    ::llvm::Value *false_value, const Type *type, bool varying) {
    if (_is_scalar_data(type)) {
        return _builder.CreateSelect(
            condition, true_value, false_value);
    }
    return _assemble(type, varying, [&](uint32_t i) {
        return _select_data(
            condition,
            _extract_child(true_value, type, i, varying),
            _extract_child(false_value, type, i, varying),
            _child_type(type, i), varying);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_componentwise_unary(
    const Type *result_type, ::llvm::Value *operand,
    const Type *operand_type, bool varying, const UnaryLeaf &leaf) {
    if (_is_scalar_data(result_type)) {
        return leaf(operand, operand_type);
    }
    return _assemble(result_type, varying, [&](uint32_t i) {
        auto scalar_operand = _is_scalar_data(operand_type);
        auto *child_operand_type = scalar_operand ? operand_type :
                                                    _child_type(operand_type, i);
        auto *child_operand = scalar_operand ? operand :
                                               _extract_child(operand, operand_type, i, varying);
        return _componentwise_unary(
            _child_type(result_type, i), child_operand,
            child_operand_type, varying, leaf);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_componentwise_binary(
    const Type *result_type, ::llvm::Value *lhs, const Type *lhs_type,
    ::llvm::Value *rhs, const Type *rhs_type, bool varying,
    const BinaryLeaf &leaf) {
    if (_is_scalar_data(result_type)) {
        return leaf(lhs, rhs, lhs_type, rhs_type);
    }
    return _assemble(result_type, varying, [&](uint32_t i) {
        auto lhs_scalar = _is_scalar_data(lhs_type);
        auto rhs_scalar = _is_scalar_data(rhs_type);
        auto *lhs_child_type = lhs_scalar ? lhs_type :
                                            _child_type(lhs_type, i);
        auto *rhs_child_type = rhs_scalar ? rhs_type :
                                            _child_type(rhs_type, i);
        auto *lhs_child = lhs_scalar ? lhs :
                                       _extract_child(lhs, lhs_type, i, varying);
        auto *rhs_child = rhs_scalar ? rhs :
                                       _extract_child(rhs, rhs_type, i, varying);
        return _componentwise_binary(
            _child_type(result_type, i), lhs_child, lhs_child_type,
            rhs_child, rhs_child_type, varying, leaf);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_componentwise_ternary(
    const Type *result_type,
    ::llvm::Value *a, const Type *a_type,
    ::llvm::Value *b, const Type *b_type,
    ::llvm::Value *c, const Type *c_type, bool varying,
    const TernaryLeaf &leaf) {
    if (_is_scalar_data(result_type)) {
        return leaf(a, b, c, a_type, b_type, c_type);
    }
    return _assemble(result_type, varying, [&](uint32_t i) {
        auto a_scalar = _is_scalar_data(a_type);
        auto b_scalar = _is_scalar_data(b_type);
        auto c_scalar = _is_scalar_data(c_type);
        auto *a_child_type = a_scalar ? a_type :
                                        _child_type(a_type, i);
        auto *b_child_type = b_scalar ? b_type :
                                        _child_type(b_type, i);
        auto *c_child_type = c_scalar ? c_type :
                                        _child_type(c_type, i);
        auto *a_child = a_scalar ? a :
                                   _extract_child(a, a_type, i, varying);
        auto *b_child = b_scalar ? b :
                                   _extract_child(b, b_type, i, varying);
        auto *c_child = c_scalar ? c :
                                   _extract_child(c, c_type, i, varying);
        return _componentwise_ternary(
            _child_type(result_type, i),
            a_child, a_child_type, b_child, b_child_type,
            c_child, c_child_type, varying, leaf);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_componentwise_varying_to_uniform(
    const Type *result_type, ::llvm::Value *operand,
    const Type *operand_type, const UnaryLeaf &leaf) {
    if (_is_scalar_data(result_type)) {
        if (!_is_scalar_data(operand_type)) {
            _fail("aggregate warp collective result shape does not match its operand");
            return nullptr;
        }
        return leaf(operand, operand_type);
    }
    if (_is_scalar_data(operand_type) ||
        _child_count(result_type) != _child_count(operand_type)) {
        _fail("aggregate warp collective result shape does not match its operand");
        return nullptr;
    }
    return _assemble(result_type, false, [&](uint32_t i) {
        return _componentwise_varying_to_uniform(
            _child_type(result_type, i),
            _extract_child(operand, operand_type, i, true),
            _child_type(operand_type, i), leaf);
    });
}

[[nodiscard]] luisa::optional<uint64_t> ScheduleEmitter::_constant_index(
    ::llvm::Value *value) noexcept {
    if (auto *integer = ::llvm::dyn_cast<::llvm::ConstantInt>(value)) {
        return integer->getZExtValue();
    }
    if (auto *constant = ::llvm::dyn_cast<::llvm::Constant>(value)) {
        if (auto *splat = constant->getSplatValue()) {
            if (auto *integer = ::llvm::dyn_cast<::llvm::ConstantInt>(splat)) {
                return integer->getZExtValue();
            }
        }
    }
    return luisa::nullopt;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_index_constant_like(
    ::llvm::Value *index, uint64_t value) {
    auto *type = index->getType();
    if (auto *vector = ::llvm::dyn_cast<::llvm::VectorType>(type)) {
        auto *element = ::llvm::cast<::llvm::IntegerType>(
            vector->getElementType());
        return _builder.CreateVectorSplat(
            vector->getElementCount(),
            ::llvm::ConstantInt::get(element, value));
    }
    return ::llvm::ConstantInt::get(
        ::llvm::cast<::llvm::IntegerType>(type), value);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_extract_indexed(
    ::llvm::Value *aggregate, const Type *type,
    const std::vector<::llvm::Value *> &indices, size_t depth,
    bool varying) {
    if (depth == indices.size()) { return aggregate; }
    auto *index = indices[depth];
    auto count = _child_count(type);
    if (count == 0u) {
        _fail("aggregate extract has too many indices");
        return nullptr;
    }
    ::llvm::Value *selected = nullptr;
    auto *child_type = _child_type(type, 0u);
    if (auto constant = _constant_index(index)) {
        if (*constant >= count) {
            _fail("aggregate extract index is out of range");
            return nullptr;
        }
        selected = _extract_child(
            aggregate, type, static_cast<uint32_t>(*constant), varying);
        child_type = _child_type(
            type, static_cast<uint32_t>(*constant));
    } else {
        if (type->is_structure()) {
            _fail("dynamic structure member extraction is invalid");
            return nullptr;
        }
        selected = _extract_child(aggregate, type, 0u, varying);
        for (auto i = uint32_t{1u}; i < count; i++) {
            auto *candidate = _extract_child(
                aggregate, type, i, varying);
            auto *condition = _builder.CreateICmpEQ(
                index, _index_constant_like(index, i));
            selected = _select_data(
                condition, candidate, selected, child_type, varying);
        }
    }
    return _extract_indexed(
        selected, child_type, indices, depth + 1u, varying);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_insert_indexed(
    ::llvm::Value *aggregate, const Type *type,
    ::llvm::Value *replacement,
    const std::vector<::llvm::Value *> &indices, size_t depth,
    bool varying) {
    if (depth == indices.size()) { return replacement; }
    auto *index = indices[depth];
    auto count = _child_count(type);
    if (count == 0u) {
        _fail("aggregate insert has too many indices");
        return nullptr;
    }
    if (auto constant = _constant_index(index)) {
        if (*constant >= count) {
            _fail("aggregate insert index is out of range");
            return nullptr;
        }
        auto i = static_cast<uint32_t>(*constant);
        auto *child_type = _child_type(type, i);
        auto *old_child = _extract_child(
            aggregate, type, i, varying);
        auto *new_child = _insert_indexed(
            old_child, child_type, replacement,
            indices, depth + 1u, varying);
        return new_child == nullptr ? nullptr :
                                      _insert_child(aggregate, new_child, type, i, varying);
    }
    if (type->is_structure()) {
        _fail("dynamic structure member insertion is invalid");
        return nullptr;
    }
    auto *result = aggregate;
    for (auto i = uint32_t{0u}; i < count; i++) {
        auto *child_type = _child_type(type, i);
        auto *old_child = _extract_child(
            aggregate, type, i, varying);
        auto *updated = _insert_indexed(
            old_child, child_type, replacement,
            indices, depth + 1u, varying);
        if (updated == nullptr) { return nullptr; }
        auto *condition = _builder.CreateICmpEQ(
            index, _index_constant_like(index, i));
        auto *selected = _select_data(
            condition, updated, old_child, child_type, varying);
        result = _insert_child(
            result, selected, type, i, varying);
    }
    return result;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_aggregate_operation(
    const schedule::Value &result,
    const schedule::Instruction &instruction,
    const std::vector<::llvm::Value *> &operands, bool varying) {
    if (operands.size() != _child_count(result.type)) {
        _fail("aggregate construction operand count mismatch");
        return nullptr;
    }
    return _assemble(result.type, varying, [&](uint32_t i) {
        return operands[i];
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_emit_float_fmod(
    ::llvm::Value *lhs, ::llvm::Value *rhs, bool varying) {
    auto *vec_ty = ::llvm::dyn_cast<::llvm::FixedVectorType>(lhs->getType());
    if (varying && vec_ty != nullptr &&
        vec_ty->getElementType()->isFloatTy() &&
        vec_ty->getNumElements() >= 1u) {
        const auto width = vec_ty->getNumElements();
        auto &context = _module.getContext();
        auto slow_name = "luisa.simd.frem.slow.v" + std::to_string(width) + "f32";
        auto *slow_helper = _module.getFunction(slow_name);
        if (slow_helper == nullptr) {
            auto *slow_ft = ::llvm::FunctionType::get(vec_ty, {vec_ty, vec_ty}, false);
            slow_helper = ::llvm::Function::Create(
                slow_ft, ::llvm::GlobalValue::PrivateLinkage, slow_name, &_module);
            slow_helper->setCallingConv(::llvm::CallingConv::Fast);
            slow_helper->addFnAttr(::llvm::Attribute::NoInline);
            slow_helper->addFnAttr(::llvm::Attribute::Cold);
            slow_helper->setDoesNotAccessMemory();
            slow_helper->setDoesNotThrow();
            slow_helper->setWillReturn();
            auto *sb_bb = ::llvm::BasicBlock::Create(context, "entry", slow_helper);
            ::llvm::IRBuilder<> sb{sb_bb};
            sb.CreateRet(sb.CreateFRem(slow_helper->getArg(0u), slow_helper->getArg(1u)));
        }
        auto fast_name = "luisa.simd.fmod.v" + std::to_string(width) + "f32";
        auto *fast_helper = _module.getFunction(fast_name);
        if (fast_helper == nullptr) {
            auto *mask_ty = ::llvm::FixedVectorType::get(::llvm::Type::getInt1Ty(context), width);
            auto *f64_vec_ty = ::llvm::FixedVectorType::get(::llvm::Type::getDoubleTy(context), width);
            auto *i64_vec_ty = ::llvm::FixedVectorType::get(::llvm::Type::getInt64Ty(context), width);
            auto *fast_ft = ::llvm::FunctionType::get(vec_ty, {vec_ty, vec_ty, mask_ty}, false);
            fast_helper = ::llvm::Function::Create(
                fast_ft, ::llvm::GlobalValue::PrivateLinkage, fast_name, &_module);
            fast_helper->setCallingConv(::llvm::CallingConv::Fast);
            fast_helper->addFnAttr(::llvm::Attribute::NoInline);
            fast_helper->addFnAttr(::llvm::Attribute::Cold);
            fast_helper->setDoesNotAccessMemory();
            fast_helper->setDoesNotThrow();
            fast_helper->setWillReturn();
            auto *entry_bb = ::llvm::BasicBlock::Create(context, "entry", fast_helper);
            auto *trivial_bb = ::llvm::BasicBlock::Create(context, "trivial", fast_helper);
            auto *compute_bb = ::llvm::BasicBlock::Create(context, "compute", fast_helper);
            auto *fast_bb = ::llvm::BasicBlock::Create(context, "fast", fast_helper);
            auto *slow_bb = ::llvm::BasicBlock::Create(context, "slow", fast_helper);
            // Keep strict IEEE semantics (no fast-math poison) for exact fmodf equivalence.
            ::llvm::IRBuilder<> hb{entry_bb};
            auto *zero_f32 = ::llvm::ConstantFP::get(vec_ty, 0.0);
            auto *one_f32 = ::llvm::ConstantFP::get(vec_ty, 1.0);
            auto *x_s = hb.CreateSelect(fast_helper->getArg(2u), fast_helper->getArg(0u), zero_f32);
            auto *y_s = hb.CreateSelect(fast_helper->getArg(2u), fast_helper->getArg(1u), one_f32);
            auto *abs_x_f32 = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, x_s);
            auto *abs_y_f32 = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, y_s);
            auto *all_trivial = hb.CreateAndReduce(hb.CreateFCmpOLT(abs_x_f32, abs_y_f32));
            auto *triv_br = hb.CreateCondBr(all_trivial, trivial_bb, compute_bb);
            triv_br->setMetadata(
                ::llvm::LLVMContext::MD_prof,
                ::llvm::MDBuilder{context}.createBranchWeights(1000u, 100u));

            hb.SetInsertPoint(trivial_bb);
            hb.CreateRet(x_s);

            hb.SetInsertPoint(compute_bb);
            auto *dx = hb.CreateFPExt(x_s, f64_vec_ty);
            auto *dy = hb.CreateFPExt(y_s, f64_vec_ty);
            auto *q_raw = hb.CreateFDiv(dx, dy);
            auto *abs_dy = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, dy);
            auto *abs_q = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, q_raw);
            auto *inf_f64 = ::llvm::ConstantFP::getInfinity(f64_vec_ty);
            auto *limit_f64 = ::llvm::ConstantFP::get(f64_vec_ty, 268435456.0);
            auto *dy_fin = hb.CreateFCmpOLT(abs_dy, inf_f64);
            auto *q_ok = hb.CreateFCmpOLT(abs_q, limit_f64);
            auto *all_ok = hb.CreateAndReduce(hb.CreateAnd(dy_fin, q_ok));
            auto *br = hb.CreateCondBr(all_ok, fast_bb, slow_bb);
            br->setMetadata(
                ::llvm::LLVMContext::MD_prof,
                ::llvm::MDBuilder{context}.createBranchWeights(1000000u, 1u));

            hb.SetInsertPoint(fast_bb);
            auto *q = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::trunc, q_raw);
            auto *neg_q = hb.CreateFNeg(q);
            auto *r0 = hb.CreateIntrinsic(::llvm::Intrinsic::fma, {f64_vec_ty}, {neg_q, dy, dx});
            auto *signed_dy = hb.CreateIntrinsic(::llvm::Intrinsic::copysign, {f64_vec_ty}, {abs_dy, dx});
            auto *zero_f64 = ::llvm::ConstantFP::get(f64_vec_ty, 0.0);
            auto *r0_nz = hb.CreateFCmpONE(r0, zero_f64);
            auto *r0_bits = hb.CreateBitCast(r0, i64_vec_ty);
            auto *dx_bits = hb.CreateBitCast(dx, i64_vec_ty);
            auto *sign_diff = hb.CreateICmpSLT(
                hb.CreateXor(r0_bits, dx_bits),
                ::llvm::Constant::getNullValue(i64_vec_ty));
            auto *r1 = hb.CreateSelect(
                hb.CreateAnd(r0_nz, sign_diff),
                hb.CreateFAdd(r0, signed_dy), r0);
            auto *abs_r1 = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, r1);
            auto *too_big = hb.CreateFCmpOGE(abs_r1, abs_dy);
            auto *r2 = hb.CreateSelect(too_big, hb.CreateFSub(r1, signed_dy), r1);
            auto *rf0 = hb.CreateFPTrunc(r2, vec_ty);
            auto *signed_zero = hb.CreateIntrinsic(::llvm::Intrinsic::copysign, {vec_ty}, {zero_f32, x_s});
            auto *rf = hb.CreateSelect(hb.CreateFCmpOEQ(rf0, zero_f32), signed_zero, rf0);
            hb.CreateRet(rf);

            hb.SetInsertPoint(slow_bb);
            auto *slow_call = hb.CreateCall(slow_helper, {x_s, y_s});
            slow_call->setCallingConv(::llvm::CallingConv::Fast);
            hb.CreateRet(slow_call);
        }
        // Inline the exact |x| < |y| identity fast path so common
        // finite range-guarded calls (e.g. Cycles noise fmod(p, 1e5))
        // avoid a function call and register spills altogether.
        auto *cur_bb = _builder.GetInsertBlock();
        auto *fn = cur_bb->getParent();
        auto *slow_call_bb = ::llvm::BasicBlock::Create(context, "fmod.call", fn);
        auto *done_bb = ::llvm::BasicBlock::Create(context, "fmod.done", fn);
        ::llvm::Value *all_trivial = nullptr;
        {
            ::llvm::IRBuilder<>::FastMathFlagGuard strict_guard{_builder};
            _builder.clearFastMathFlags();
            auto *abs_x = _builder.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, lhs);
            auto *abs_y = _builder.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, rhs);
            auto *lt = _builder.CreateFCmpOLT(abs_x, abs_y);
            if (width == 1u) {
                all_trivial = _builder.CreateExtractElement(lt, _builder.getInt32(0u));
            } else {
                auto *lane_ok = _builder.CreateOr(_builder.CreateNot(_active_mask), lt);
                all_trivial = _builder.CreateAndReduce(lane_ok);
            }
        }
        auto *guard_br = _builder.CreateCondBr(all_trivial, done_bb, slow_call_bb);
        guard_br->setMetadata(
            ::llvm::LLVMContext::MD_prof,
            ::llvm::MDBuilder{context}.createBranchWeights(1000u, 10u));
        _builder.SetInsertPoint(slow_call_bb);
        ::llvm::CallInst *call = nullptr;
        {
            ::llvm::IRBuilder<>::FastMathFlagGuard strict_guard{_builder};
            _builder.clearFastMathFlags();
            call = _builder.CreateCall(fast_helper, {lhs, rhs, _active_mask});
        }
        call->setCallingConv(::llvm::CallingConv::Fast);
        _builder.CreateBr(done_bb);
        _builder.SetInsertPoint(done_bb);
        auto *phi = _builder.CreatePHI(vec_ty, 2u, "fmod.res");
        phi->addIncoming(lhs, cur_bb);
        phi->addIncoming(call, slow_call_bb);
        return phi;
    }
    if (!varying && lhs->getType()->isFloatTy()) {
        auto &context = _module.getContext();
        auto *f32_ty = lhs->getType();
        auto *slow_helper = _module.getFunction("luisa.simd.frem.slow.f32");
        if (slow_helper == nullptr) {
            auto *slow_ft = ::llvm::FunctionType::get(f32_ty, {f32_ty, f32_ty}, false);
            slow_helper = ::llvm::Function::Create(
                slow_ft, ::llvm::GlobalValue::PrivateLinkage, "luisa.simd.frem.slow.f32", &_module);
            slow_helper->setCallingConv(::llvm::CallingConv::Fast);
            slow_helper->addFnAttr(::llvm::Attribute::NoInline);
            slow_helper->addFnAttr(::llvm::Attribute::Cold);
            slow_helper->setDoesNotAccessMemory();
            slow_helper->setDoesNotThrow();
            slow_helper->setWillReturn();
            auto *sb_bb = ::llvm::BasicBlock::Create(context, "entry", slow_helper);
            ::llvm::IRBuilder<> sb{sb_bb};
            sb.CreateRet(sb.CreateFRem(slow_helper->getArg(0u), slow_helper->getArg(1u)));
        }
        auto *fast_helper = _module.getFunction("luisa.simd.fmod.f32");
        if (fast_helper == nullptr) {
            auto *f64_ty = ::llvm::Type::getDoubleTy(context);
            auto *i64_ty = ::llvm::Type::getInt64Ty(context);
            auto *fast_ft = ::llvm::FunctionType::get(f32_ty, {f32_ty, f32_ty}, false);
            fast_helper = ::llvm::Function::Create(
                fast_ft, ::llvm::GlobalValue::PrivateLinkage, "luisa.simd.fmod.f32", &_module);
            fast_helper->setCallingConv(::llvm::CallingConv::Fast);
            fast_helper->addFnAttr(::llvm::Attribute::NoInline);
            fast_helper->addFnAttr(::llvm::Attribute::Cold);
            fast_helper->setDoesNotAccessMemory();
            fast_helper->setDoesNotThrow();
            fast_helper->setWillReturn();
            auto *entry_bb = ::llvm::BasicBlock::Create(context, "entry", fast_helper);
            auto *fast_bb = ::llvm::BasicBlock::Create(context, "fast", fast_helper);
            auto *slow_bb = ::llvm::BasicBlock::Create(context, "slow", fast_helper);
            ::llvm::IRBuilder<> hb{entry_bb};
            auto *dx = hb.CreateFPExt(fast_helper->getArg(0u), f64_ty);
            auto *dy = hb.CreateFPExt(fast_helper->getArg(1u), f64_ty);
            auto *q_raw = hb.CreateFDiv(dx, dy);
            auto *abs_dy = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, dy);
            auto *abs_q = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, q_raw);
            auto *all_ok = hb.CreateAnd(
                hb.CreateFCmpOLT(abs_dy, ::llvm::ConstantFP::getInfinity(f64_ty)),
                hb.CreateFCmpOLT(abs_q, ::llvm::ConstantFP::get(f64_ty, 268435456.0)));
            hb.CreateCondBr(all_ok, fast_bb, slow_bb);
            hb.SetInsertPoint(fast_bb);
            auto *q = hb.CreateUnaryIntrinsic(::llvm::Intrinsic::trunc, q_raw);
            auto *r0 = hb.CreateIntrinsic(::llvm::Intrinsic::fma, {f64_ty}, {hb.CreateFNeg(q), dy, dx});
            auto *signed_dy = hb.CreateIntrinsic(::llvm::Intrinsic::copysign, {f64_ty}, {abs_dy, dx});
            auto *r0_nz = hb.CreateFCmpONE(r0, ::llvm::ConstantFP::get(f64_ty, 0.0));
            auto *sign_diff = hb.CreateICmpSLT(
                hb.CreateXor(hb.CreateBitCast(r0, i64_ty), hb.CreateBitCast(dx, i64_ty)),
                ::llvm::ConstantInt::get(i64_ty, 0));
            auto *r1 = hb.CreateSelect(hb.CreateAnd(r0_nz, sign_diff), hb.CreateFAdd(r0, signed_dy), r0);
            auto *r2 = hb.CreateSelect(
                hb.CreateFCmpOGE(hb.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, r1), abs_dy),
                hb.CreateFSub(r1, signed_dy), r1);
            auto *rf0 = hb.CreateFPTrunc(r2, f32_ty);
            auto *zero_f32 = ::llvm::ConstantFP::get(f32_ty, 0.0);
            auto *signed_zero = hb.CreateIntrinsic(::llvm::Intrinsic::copysign, {f32_ty}, {zero_f32, fast_helper->getArg(0u)});
            hb.CreateRet(hb.CreateSelect(hb.CreateFCmpOEQ(rf0, zero_f32), signed_zero, rf0));
            hb.SetInsertPoint(slow_bb);
            auto *sc = hb.CreateCall(slow_helper, {fast_helper->getArg(0u), fast_helper->getArg(1u)});
            sc->setCallingConv(::llvm::CallingConv::Fast);
            hb.CreateRet(sc);
        }
        auto *cur_bb = _builder.GetInsertBlock();
        auto *fn = cur_bb->getParent();
        auto *slow_call_bb = ::llvm::BasicBlock::Create(context, "fmod.s.call", fn);
        auto *done_bb = ::llvm::BasicBlock::Create(context, "fmod.s.done", fn);
        ::llvm::Value *trivial = nullptr;
        {
            ::llvm::IRBuilder<>::FastMathFlagGuard strict_guard{_builder};
            _builder.clearFastMathFlags();
            trivial = _builder.CreateFCmpOLT(
                _builder.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, lhs),
                _builder.CreateUnaryIntrinsic(::llvm::Intrinsic::fabs, rhs));
        }
        auto *guard_br = _builder.CreateCondBr(trivial, done_bb, slow_call_bb);
        guard_br->setMetadata(
            ::llvm::LLVMContext::MD_prof,
            ::llvm::MDBuilder{context}.createBranchWeights(1000u, 10u));
        _builder.SetInsertPoint(slow_call_bb);
        ::llvm::CallInst *call = nullptr;
        {
            ::llvm::IRBuilder<>::FastMathFlagGuard strict_guard{_builder};
            _builder.clearFastMathFlags();
            call = _builder.CreateCall(fast_helper, {lhs, rhs});
        }
        call->setCallingConv(::llvm::CallingConv::Fast);
        _builder.CreateBr(done_bb);
        _builder.SetInsertPoint(done_bb);
        auto *phi = _builder.CreatePHI(f32_ty, 2u, "fmod.s.res");
        phi->addIncoming(lhs, cur_bb);
        phi->addIncoming(call, slow_call_bb);
        return phi;
    }
    ::llvm::IRBuilder<>::FastMathFlagGuard strict_guard{_builder};
    _builder.clearFastMathFlags();
    return _builder.CreateFRem(lhs, rhs);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_cast(
    const schedule::Instruction &instruction,
    ::llvm::Value *operand_sanitization_mask) {
    if (!instruction.result || !instruction.source_op ||
        instruction.operands.size() != 1u) {
        _fail("cast instruction is malformed");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    auto *source = _source.value(instruction.operands.front());
    auto *value = _load_value(instruction.operands.front());
    if (result == nullptr || source == nullptr || value == nullptr ||
        !_is_data(result->type) || !_is_data(source->type)) {
        _fail("cast requires supported data types");
        return nullptr;
    }
    auto varying = result->value_class == schedule::ValueClass::varying;
    if (varying) { value = _as_lane_vector(value, *source); }
    auto *sanitization_mask = operand_sanitization_mask != nullptr ?
                                  operand_sanitization_mask :
                              varying ? _active_mask :
                                        nullptr;
    auto op = static_cast<xir::CastOp>(*instruction.source_op);
    if (op == xir::CastOp::BITWISE_CAST) {
        auto *uniform_result_type = _data_type(result->type, false);
        if (uniform_result_type == nullptr) { return nullptr; }
        if (!varying) {
            return _builder.CreateBitCast(value, uniform_result_type);
        }

        // Varying aggregates use a structure-of-arrays representation, while
        // an XIR bitwise cast reinterprets one complete logical register. Pack
        // and cast each logical lane independently so shape-changing casts
        // such as float2 <-> uint64 preserve their bits and still return the
        // expected SIMD lane-vector/SoA representation.
        auto *varying_result_type = _data_type(result->type, true);
        if (varying_result_type == nullptr) { return nullptr; }
        if (_is_scalar_data(source->type) &&
            _is_scalar_data(result->type)) {
            return _builder.CreateBitCast(value, varying_result_type);
        }
        auto *cast = static_cast<::llvm::Value *>(
            ::llvm::PoisonValue::get(varying_result_type));
        for (auto lane_index = uint32_t{0u};
             lane_index < _width; lane_index++) {
            auto *lane = _builder.getInt32(lane_index);
            auto *source_lane = _extract_lane(
                value, source->type, lane);
            auto *result_lane = _builder.CreateBitCast(
                source_lane, uniform_result_type);
            cast = _insert_lane(
                cast, result_lane, result->type, lane);
            if (cast == nullptr) { return nullptr; }
        }
        return cast;
    }
    return _componentwise_unary(
        result->type, value, source->type, varying,
        [&](::llvm::Value *scalar, const Type *source_type) {
            auto *destination_type = result->type;
            while (!_is_scalar_data(destination_type)) {
                destination_type = destination_type->element();
            }
            auto destination_is_float =
                destination_type->is_float16() ||
                destination_type->is_float32() ||
                destination_type->is_float64();
            auto source_is_float = source_type->is_float16() ||
                                   source_type->is_float32() ||
                                   source_type->is_float64();
            // LLVM floating-point-to-integer conversions produce poison for
            // NaN and out-of-range operands. SIMD control flow still forms
            // vector instructions for inactive lanes, so replace only those
            // leaf operands before the conversion. Explicitly predicated
            // regions pass their block mask; ordinary Schedule blocks use the
            // current cohort mask. Masking the result afterwards would be too
            // late because poison could already have escaped through a
            // following comparison or address expression.
            if (sanitization_mask != nullptr &&
                source_is_float && !destination_is_float &&
                !destination_type->is_bool() &&
                scalar->getType()->isVectorTy()) {
                scalar = _builder.CreateSelect(
                    sanitization_mask, scalar,
                    ::llvm::Constant::getNullValue(
                        scalar->getType()));
            }
            auto *destination = scalar->getType()->isVectorTy() ?
                                    static_cast<::llvm::Type *>(::llvm::FixedVectorType::get(
                                        _data_type(destination_type, false), _width)) :
                                    _data_type(destination_type, false);
            if (destination_type->is_bool()) {
                auto *zero = ::llvm::Constant::getNullValue(
                    scalar->getType());
                return source_is_float ?
                           _builder.CreateFCmpUNE(scalar, zero) :
                           _builder.CreateICmpNE(scalar, zero);
            }
            if (source_type->is_bool()) {
                return destination_is_float ?
                           _builder.CreateUIToFP(scalar, destination) :
                           _builder.CreateZExtOrTrunc(scalar, destination);
            }
            if (source_is_float && destination_is_float) {
                return _builder.CreateFPCast(scalar, destination);
            }
            if (source_is_float) {
                return destination_type->is_int() ?
                           _builder.CreateFPToSI(scalar, destination) :
                           _builder.CreateFPToUI(scalar, destination);
            }
            if (destination_is_float) {
                return source_type->is_int() ?
                           _builder.CreateSIToFP(scalar, destination) :
                           _builder.CreateUIToFP(scalar, destination);
            }
            return source_type->is_int() ?
                       _builder.CreateSExtOrTrunc(scalar, destination) :
                       _builder.CreateZExtOrTrunc(scalar, destination);
        });
}

}// namespace luisa::compute::simd::detail
