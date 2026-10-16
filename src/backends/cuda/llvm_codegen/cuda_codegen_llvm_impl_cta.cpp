//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"
#include <luisa/core/stl/memory.h>

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept {

    auto pack_into_i32_vector = [&](llvm::Value *v) noexcept {
        LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy() || v->getType()->isFPOrFPVectorTy());
        auto bitwidth = _data_layout->getTypeSizeInBits(v->getType());
        if (bitwidth < 32) {
            v = b.CreateZExt(b.CreateBitCast(v, b.getIntNTy(bitwidth)), b.getInt32Ty());
        }
        auto n = (bitwidth + 31) / 32;
        return std::make_pair(b.CreateBitCast(v, llvm::VectorType::get(b.getInt32Ty(), n, false)), n);
    };

    auto unpack_from_i32_vector = [&](llvm::Value *v, llvm::Type *target_type) noexcept {
        LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy(32));
        LUISA_DEBUG_ASSERT(target_type->isIntOrIntVectorTy() || target_type->isFPOrFPVectorTy());
        auto bitwidth = _data_layout->getTypeSizeInBits(target_type);
        if (bitwidth < 32) {
            v = b.CreateTrunc(b.CreateExtractElement(v, b.getInt32(0)), b.getIntNTy(bitwidth));
        }
        return b.CreateBitCast(v, target_type);
    };

    auto reduce_active = [&](llvm::Value *mask, llvm::Value *lane, llvm::Value *value, auto binary_op) noexcept -> llvm::Value * {
        LUISA_DEBUG_ASSERT(value->getType()->isIntOrIntVectorTy(32));
        auto shuffle = [&b, mask](llvm::Value *x, auto offset) noexcept {
            return b.CreateIntrinsic(llvm::Intrinsic::nvvm_shfl_sync_bfly_i32,
                                     {mask, x, b.getInt32(offset), b.getInt32(31)});
        };
        for (auto offset = 16u; offset >= 1u; offset /= 2u) {
            auto shuffled_value = static_cast<llvm::Value *>(nullptr);
            if (auto vt = llvm::dyn_cast<llvm::VectorType>(value->getType())) {
                llvm::SmallVector<llvm::Value *, 8> shuffled_values;
                auto dim = vt->getElementCount().getFixedValue();
                for (auto i = 0; i < dim; i++) {
                    auto elem = b.CreateExtractElement(value, i);
                    shuffled_values.emplace_back(shuffle(elem, offset));
                }
                shuffled_value = _create_llvm_vector(b, shuffled_values);
            } else {
                shuffled_value = shuffle(value, offset);
            }
            auto alive_mask = b.CreateShl(b.getInt32(1), b.CreateXor(lane, offset));
            auto is_alive = b.CreateICmpNE(b.CreateAnd(mask, alive_mask), b.getInt32(0));
            value = b.CreateSelect(is_alive, binary_op(value, shuffled_value), value);
        }
        return value;
    };

    // i1 luisa.warp.prefix.scan.cond(i32 lane, i32 mask, i32 offset)
    auto reduce_prefix_cond_func = [this] {
        using namespace std::string_view_literals;
        constexpr auto name = "luisa.warp.prefix.scan.cond"sv;
        auto cond = _llvm_module->getFunction(name);
        if (cond != nullptr) { return cond; }
        auto i1_type = llvm::Type::getInt1Ty(_llvm_module->getContext());
        auto i32_type = llvm::Type::getInt32Ty(_llvm_module->getContext());
        auto func_type = llvm::FunctionType::get(i1_type, {i32_type, i32_type, i32_type}, false);
        cond = llvm::Function::Create(func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
        cond->addFnAttr(llvm::Attribute::AlwaysInline);
        auto entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", cond);
        IB b{entry_bb};
        auto lane = cond->getArg(0);
        auto mask = cond->getArg(1);
        auto offset = cond->getArg(2);
        auto lane_ge_offset = b.CreateICmpUGE(lane, offset);
        auto lane_sub_offset = b.CreateSub(lane, offset);
        auto lane_shift = b.CreateShl(b.getInt32(1), lane_sub_offset);
        auto is_alive = b.CreateICmpNE(b.CreateAnd(mask, lane_shift), b.getInt32(0));
        b.CreateRet(b.CreateAnd(lane_ge_offset, is_alive));
        return cond;
        // auto true_bb = llvm::BasicBlock::Create(_llvm_context, "exit.true", cond);
        // auto false_bb = llvm::BasicBlock::Create(_llvm_context, "exit.false", cond);
        // auto active_bb = llvm::BasicBlock::Create(_llvm_context, "active", cond);
        // IB func_b{entry_bb};
        // auto lane = cond->getArg(0);
        // auto mask = cond->getArg(1);
        // auto offset = cond->getArg(2);
        // auto lane_ge_offset = func_b.CreateICmpUGE(lane, offset);
        // func_b.CreateCondBr(lane_ge_offset, active_bb, false_bb);
        // func_b.SetInsertPoint(active_bb);
        // auto relative_offset = func_b.CreateSub(lane, offset, "", true, true);
        // auto alive_mask = func_b.CreateShl(func_b.getInt32(1), relative_offset, "", true, true);
        // auto is_alive = func_b.CreateICmpNE(func_b.CreateAnd(mask, alive_mask), func_b.getInt32(0));
        // func_b.CreateCondBr(is_alive, true_bb, false_bb);
        // func_b.SetInsertPoint(true_bb);
        // func_b.CreateRet(func_b.getInt1(true));
        // func_b.SetInsertPoint(false_bb);
        // func_b.CreateRet(func_b.getInt1(false));
        // return cond;
    };

    auto reduce_prefix = [&](llvm::Value *mask, llvm::Value *lane, llvm::Value *unit, llvm::Value *value, auto binary_op) noexcept {
        LUISA_DEBUG_ASSERT(value->getType()->isIntOrIntVectorTy(32));
        auto shuffle = [&b, mask](llvm::Value *x, auto offset) noexcept {
            return b.CreateIntrinsic(llvm::Intrinsic::nvvm_shfl_sync_up_i32,
                                     {mask, x, b.getInt32(offset), b.getInt32(0) /* !!! NOTE: ONYL shfl_sync_up USES 0 !!! */});
        };
        auto prev_mask = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_lt, {});
        auto active_prev_mask = b.CreateAnd(prev_mask, mask);
        auto active_prev_lane = b.CreateSub(b.getInt32(31),
                                            b.CreateBinaryIntrinsic(llvm::Intrinsic::ctlz, active_prev_mask, b.getInt1(false)),
                                            "", true, true);
        // value = shfl_sync_idx(mask, x, active_prev_lane)
        if (auto vt = llvm::dyn_cast<llvm::VectorType>(value->getType())) {
            llvm::SmallVector<llvm::Value *, 8> shuffled_values;
            auto dim = vt->getElementCount().getFixedValue();
            for (auto i = 0; i < dim; i++) {
                auto elem = b.CreateExtractElement(value, i);
                shuffled_values.emplace_back(b.CreateIntrinsic(llvm::Intrinsic::nvvm_shfl_sync_idx_i32,
                                                               {mask, elem, active_prev_lane, b.getInt32(31)}));
            }
            value = _create_llvm_vector(b, shuffled_values);
        } else {
            value = b.CreateIntrinsic(llvm::Intrinsic::nvvm_shfl_sync_idx_i32,
                                      {mask, value, active_prev_lane, b.getInt32(31)});
        }
        // value = select(lane == first_active_lane, unit, value)
        auto first_active_lane = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, mask, b.getInt1(true));
        auto is_first_active = b.CreateICmpEQ(lane, first_active_lane);
        value = b.CreateSelect(is_first_active, unit, value);
        // perform shuffles
        auto cond_func = reduce_prefix_cond_func();
        for (auto offset = 1u; offset <= 16u; offset *= 2u) {
            auto shuffled_value = static_cast<llvm::Value *>(nullptr);
            if (auto vt = llvm::dyn_cast<llvm::VectorType>(value->getType())) {
                llvm::SmallVector<llvm::Value *, 8> shuffled_values;
                auto dim = vt->getElementCount().getFixedValue();
                for (auto i = 0; i < dim; i++) {
                    auto elem = b.CreateExtractElement(value, i);
                    shuffled_values.emplace_back(shuffle(elem, offset));
                }
                shuffled_value = _create_llvm_vector(b, shuffled_values);
            } else {
                shuffled_value = shuffle(value, offset);
            }
            auto cond = b.CreateCall(cond_func, {lane, mask, b.getInt32(offset)});
            auto new_value = binary_op(value, shuffled_value);
            value = b.CreateSelect(cond, new_value, value);
        }
        return value;
    };

    switch (auto op = inst->op()) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER: {
            // no-op in CUDA, while asm sideeffect "call (), _optix_hitobject_reorder, ($0,$1);", "r,r"(i32 %3, i32 %4) in ray tracing shaders
            if (_rt_analysis.uses_ray_tracing) {
                // we observe almost constant slowdown with SER, so warn our users...
                LUISA_WARNING_WITH_LOCATION("Shader execution reordering (SER) is on. This might actually slow down your rendering...");
                auto llvm_asm = _get_inline_asm("call (), _optix_hitobject_reorder, ($0,$1);", "r,r", true);
                auto arg_count = inst->operand_count();
                auto llvm_hint = arg_count >= 1 ? _get_llvm_value(b, func_ctx, inst->operand(0)) : b.getInt32(0);
                auto llvm_hint_bits = arg_count >= 2 ? _get_llvm_value(b, func_ctx, inst->operand(1)) : b.getInt32(0);
                LUISA_DEBUG_ASSERT(llvm_hint->getType()->isIntegerTy(32) && llvm_hint_bits->getType()->isIntegerTy(32));
                return b.CreateCall(llvm_asm, {llvm_hint, llvm_hint_bits});
            }
            return nullptr;
        }
        case xir::ThreadGroupOp::RASTER_QUAD_DDX: LUISA_ERROR_WITH_LOCATION("RASTER_QUAD_DDX is not supported in CUDA backend.");
        case xir::ThreadGroupOp::RASTER_QUAD_DDY: LUISA_ERROR_WITH_LOCATION("RASTER_QUAD_DDY is not supported in CUDA backend.");
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: {// lane_id == ctz(activemask)
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_first_active_lane_id = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            return b.CreateICmpEQ(llvm_lane_id, llvm_first_active_lane_id);
        }
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: {// ctz(activemask)
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            return b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool_or_bool_vector());
            LUISA_DEBUG_ASSERT(inst->operand_count() == 1 && inst->operand(0)->type()->is_scalar_or_vector());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            auto llvm_scalar_bitwidth = llvm_value_type->getScalarType()->getPrimitiveSizeInBits();
            // we may use nvvm_match_all_sync_i32p/i64p for scalars or 32b/64b vectors on supported archs
            if (_config.cuda_arch >= nvvm_required_arch_match_all) {
                auto handle_scalar = [&](llvm::Value *s) noexcept {
                    // bitcast to integer if not integer
                    if (!s->getType()->isIntegerTy()) { s = b.CreateBitCast(s, b.getIntNTy(llvm_scalar_bitwidth)); }
                    auto llvm_op = llvm_scalar_bitwidth <= 32 ?
                                       llvm::Intrinsic::nvvm_match_all_sync_i32p :
                                       llvm::Intrinsic::nvvm_match_all_sync_i64p;
                    auto x = b.CreateZExt(s, llvm_scalar_bitwidth <= 32 ? b.getInt32Ty() : b.getInt64Ty());
                    auto llvm_result = b.CreateIntrinsic(llvm_op, {llvm_active_mask, x});
                    return b.CreateExtractValue(llvm_result, 1);
                };
                if (inst->type()->is_scalar()) { return handle_scalar(llvm_value); }
                // for vectors, we iteratively call match_all_sync on each element
                llvm::SmallVector<llvm::Value *, 4> llvm_preds;
                for (auto i = 0u; i < inst->type()->dimension(); i++) {
                    auto llvm_elem = b.CreateExtractElement(llvm_value, b.getInt32(i));
                    llvm_preds.emplace_back(handle_scalar(llvm_elem));
                }
                return _create_llvm_vector(b, llvm_preds);
            }
            // otherwise, we fall back to shuffle, compare, and vote
            auto llvm_first_active_lane_id = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            auto [llvm_packed_value, packed_i32_count] = pack_into_i32_vector(llvm_value);
            auto llvm_packed_value_from_first = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_packed_value->getType()));
            for (auto i = 0; i < packed_i32_count; i++) {
                auto llvm_local_elem = b.CreateExtractElement(llvm_packed_value, i);
                auto llvm_elem_from_first = b.CreateIntrinsic(
                    llvm::Intrinsic::nvvm_shfl_sync_idx_i32,
                    {llvm_active_mask, llvm_local_elem, llvm_first_active_lane_id, b.getInt32(31)});
                llvm_packed_value_from_first = b.CreateInsertElement(llvm_packed_value_from_first, llvm_elem_from_first, i);
            }
            // decode the value from packed i32 vector
            auto llvm_value_from_first = unpack_from_i32_vector(llvm_packed_value_from_first, llvm_value_type);
            auto llvm_cmp = llvm_value_type->isFPOrFPVectorTy() ?
                                b.CreateFCmpOEQ(llvm_value, llvm_value_from_first) :
                                b.CreateICmpEQ(llvm_value, llvm_value_from_first);
            // vote the comparison result
            if (inst->type()->is_bool()) {// scalar
                return b.CreateIntrinsic(b.getInt1Ty(), llvm::Intrinsic::nvvm_vote_all_sync, {llvm_active_mask, llvm_cmp});
            }
            // vector
            auto llvm_result = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_cmp->getType()));
            for (auto i = 0u; i < inst->type()->dimension(); i++) {
                auto llvm_elem = b.CreateExtractElement(llvm_cmp, i);
                auto llvm_elem_voted = b.CreateIntrinsic(b.getInt1Ty(), llvm::Intrinsic::nvvm_vote_all_sync, {llvm_active_mask, llvm_elem});
                llvm_result = b.CreateInsertElement(llvm_result, llvm_elem_voted, i);
            }
            return llvm_result;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == 1 && inst->type() == inst->operand(0)->type());
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isIntOrIntVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            auto [llvm_packed_value, packed_i32_count] = pack_into_i32_vector(llvm_value);
            auto llvm_result_packed = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_packed_value->getType()));
            auto handle_one_i32 = [&](llvm::Value *llvm_local_i32) noexcept -> llvm::Value * {
                // we can use nvvm.redux.sync.and/or/xor on supported archs
                if (_config.cuda_arch >= nvvm_required_arch_redux_i32) {
                    auto llvm_op = op == xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND ? llvm::Intrinsic::nvvm_redux_sync_and :
                                   op == xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR  ? llvm::Intrinsic::nvvm_redux_sync_or :
                                                                                   llvm::Intrinsic::nvvm_redux_sync_xor;
                    return b.CreateIntrinsic(b.getInt32Ty(), llvm_op, {llvm_local_i32, llvm_active_mask});
                }
                // otherwise, we fall back to shuffle and manual reduction
                return reduce_active(llvm_active_mask, llvm_lane_id, llvm_local_i32, [&](auto x, auto y) noexcept {
                    switch (op) {
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: return b.CreateAnd(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: return b.CreateOr(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: return b.CreateXor(x, y);
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid bitwise warp reduction op.");
                });
            };
            for (auto i = 0; i < packed_i32_count; i++) {
                auto llvm_local_elem = b.CreateExtractElement(llvm_packed_value, i);
                auto llvm_reduced_elem = handle_one_i32(llvm_local_elem);
                llvm_result_packed = b.CreateInsertElement(llvm_result_packed, llvm_reduced_elem, i);
            }
            return unpack_from_i32_vector(llvm_result_packed, llvm_value->getType());
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: {// popc(ballot(activemask, pred))
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_vote_ballot_sync, {llvm_active_mask, llvm_pred});
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_ballot);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: {
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            LUISA_DEBUG_ASSERT(llvm_value_type->isIntOrIntVectorTy() || llvm_value_type->isFPOrFPVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            // we might use nvvm.redux.sync.i32/fp32 on supported archs
            if (op != xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT &&
                _config.cuda_arch >= nvvm_required_arch_redux_i32 &&
                llvm_value_type->isIntOrIntVectorTy() &&
                llvm_value_type->getScalarType()->getPrimitiveSizeInBits() <= 32) {
                auto reduce_scalar = [&](llvm::Value *v) noexcept -> llvm::Value * {
                    auto scalar_t = v->getType();
                    v = b.CreateZExt(v, b.getInt32Ty());
                    if (op == xir::ThreadGroupOp::WARP_ACTIVE_MAX) {
                        auto llvm_op = inst->type()->is_int_or_int_vector() ? llvm::Intrinsic::nvvm_redux_sync_max :
                                                                              llvm::Intrinsic::nvvm_redux_sync_umax;
                        v = b.CreateIntrinsic(b.getInt32Ty(), llvm_op, {v, llvm_active_mask});
                    } else if (op == xir::ThreadGroupOp::WARP_ACTIVE_MIN) {
                        auto llvm_op = inst->type()->is_int_or_int_vector() ? llvm::Intrinsic::nvvm_redux_sync_min :
                                                                              llvm::Intrinsic::nvvm_redux_sync_umin;
                        v = b.CreateIntrinsic(b.getInt32Ty(), llvm_op, {v, llvm_active_mask});
                    } else if (op == xir::ThreadGroupOp::WARP_ACTIVE_SUM) {
                        v = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_redux_sync_add, {v, llvm_active_mask});
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid integer warp reduction op.");
                    }
                    return b.CreateTrunc(v, scalar_t);
                };
                if (auto vt = llvm::dyn_cast<llvm::VectorType>(llvm_value_type)) {
                    llvm::SmallVector<llvm::Value *, 4> llvm_reduced_elems;
                    auto dim = vt->getElementCount().getFixedValue();
                    for (auto i = 0; i < dim; i++) {
                        auto llvm_elem = b.CreateExtractElement(llvm_value, i);
                        llvm_reduced_elems.emplace_back(reduce_scalar(llvm_elem));
                    }
                    return _create_llvm_vector(b, llvm_reduced_elems);
                }
                return reduce_scalar(llvm_value);
            }
            if (op != xir::ThreadGroupOp::WARP_ACTIVE_SUM &&
                op != xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT &&
                _config.cuda_arch >= nvvm_required_arch_redux_f32 &&
                llvm_value_type->isFPOrFPVectorTy() &&
                llvm_value_type->getScalarType()->getPrimitiveSizeInBits() <= 32) {
                auto reduce_scalar = [&](llvm::Value *v) noexcept -> llvm::Value * {
                    auto scalar_t = v->getType();
                    v = _safe_fp_cast(b, v, b.getFloatTy());
                    if (op == xir::ThreadGroupOp::WARP_ACTIVE_MAX) {
                        v = b.CreateIntrinsic(b.getFloatTy(), llvm::Intrinsic::nvvm_redux_sync_fmax, {v, llvm_active_mask});
                    } else if (op == xir::ThreadGroupOp::WARP_ACTIVE_MIN) {
                        v = b.CreateIntrinsic(b.getFloatTy(), llvm::Intrinsic::nvvm_redux_sync_fmin, {v, llvm_active_mask});
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid floating-point warp reduction op.");
                    }
                    return _safe_fp_cast(b, v, scalar_t);
                };
                if (auto vt = llvm::dyn_cast<llvm::VectorType>(llvm_value_type)) {
                    llvm::SmallVector<llvm::Value *, 4> llvm_reduced_elems;
                    auto dim = vt->getElementCount().getFixedValue();
                    for (auto i = 0; i < dim; i++) {
                        auto llvm_elem = b.CreateExtractElement(llvm_value, i);
                        llvm_reduced_elems.emplace_back(reduce_scalar(llvm_elem));
                    }
                    return _create_llvm_vector(b, llvm_reduced_elems);
                }
                return reduce_scalar(llvm_value);
            }
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            auto llvm_packed_value = pack_into_i32_vector(llvm_value).first;
            auto llvm_result_packed = reduce_active(llvm_active_mask, llvm_lane_id, llvm_packed_value, [&](auto x, auto y) noexcept {
                x = unpack_from_i32_vector(x, llvm_value_type);
                y = unpack_from_i32_vector(y, llvm_value_type);
                auto reduced = [&] {
                    switch (op) {
                        case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateBinaryIntrinsic(llvm::Intrinsic::smax, x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateBinaryIntrinsic(llvm::Intrinsic::umax, x, y) :
                                                                            b.CreateBinaryIntrinsic(llvm::Intrinsic::maxnum, x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateBinaryIntrinsic(llvm::Intrinsic::smin, x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, x, y) :
                                                                            b.CreateBinaryIntrinsic(llvm::Intrinsic::minnum, x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWMul(x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateMul(x, y) :
                                                                            b.CreateFMul(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWAdd(x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateAdd(x, y) :
                                                                            b.CreateFAdd(x, y);
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid warp reduction op.");
                }();
                return pack_into_i32_vector(reduced).first;
            });
            return unpack_from_i32_vector(llvm_result_packed, llvm_value_type);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            return b.CreateIntrinsic(b.getInt1Ty(), llvm::Intrinsic::nvvm_vote_all_sync, {llvm_active_mask, llvm_pred});
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            return b.CreateIntrinsic(b.getInt1Ty(), llvm::Intrinsic::nvvm_vote_any_sync, {llvm_active_mask, llvm_pred});
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: {// uint4(ballot(activemask, pred), 0, 0, 0)
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<int4>() || inst->type() == Type::of<uint4>());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_vote_ballot_sync, {llvm_active_mask, llvm_pred});
            auto llvm_zero = llvm::Constant::getNullValue(llvm::VectorType::get(b.getInt32Ty(), 4, false));
            return b.CreateInsertElement(llvm_zero, llvm_ballot, b.getInt64(0));
        }
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS: {// popc(ballot(activemask, pred) & lane_mask)
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_prefix_mask = _read_warp_prefix_lane_mask(b);
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_vote_ballot_sync, {llvm_active_mask, llvm_pred});
            auto llvm_ballot_and_prefix = b.CreateAnd(llvm_ballot, llvm_prefix_mask);
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_ballot_and_prefix);
        }
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: {
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            LUISA_DEBUG_ASSERT(llvm_value_type->isIntOrIntVectorTy() || llvm_value_type->isFPOrFPVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            auto llvm_packed_value = pack_into_i32_vector(llvm_value).first;
            auto llvm_result_packed = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ThreadGroupOp::WARP_PREFIX_SUM) {
                auto llvm_unit = pack_into_i32_vector(llvm::Constant::getNullValue(llvm_value_type)).first;
                llvm_result_packed = reduce_prefix(llvm_active_mask, llvm_lane_id, llvm_unit, llvm_packed_value, [&](auto x, auto y) noexcept {
                    x = unpack_from_i32_vector(x, llvm_value_type);
                    y = unpack_from_i32_vector(y, llvm_value_type);
                    auto result = inst->type()->is_int_or_int_vector()   ? b.CreateNSWAdd(x, y) :
                                  inst->type()->is_uint_or_uint_vector() ? b.CreateAdd(x, y) :
                                                                           b.CreateFAdd(x, y);
                    return pack_into_i32_vector(result).first;
                });
            } else if (op == xir::ThreadGroupOp::WARP_PREFIX_PRODUCT) {
                auto llvm_unit = pack_into_i32_vector(
                                     llvm_value_type->isIntOrIntVectorTy() ?
                                         llvm::ConstantInt::get(llvm_value_type, 1) :
                                         llvm::ConstantFP::get(llvm_value_type, 1.))
                                     .first;
                llvm_result_packed = reduce_prefix(llvm_active_mask, llvm_lane_id, llvm_unit, llvm_packed_value, [&](auto x, auto y) noexcept {
                    x = unpack_from_i32_vector(x, llvm_value_type);
                    y = unpack_from_i32_vector(y, llvm_value_type);
                    auto result = inst->type()->is_int_or_int_vector()   ? b.CreateNSWMul(x, y) :
                                  inst->type()->is_uint_or_uint_vector() ? b.CreateMul(x, y) :
                                                                           b.CreateFMul(x, y);
                    return pack_into_i32_vector(result).first;
                });
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid warp prefix op.");
            }
            return unpack_from_i32_vector(llvm_result_packed, llvm_value_type);
        }
        case xir::ThreadGroupOp::WARP_READ_LANE: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: {
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_lane_id = op == xir::ThreadGroupOp::WARP_READ_LANE ?
                                    _get_llvm_value(b, func_ctx, inst->operand(1)) :
                                    b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            LUISA_DEBUG_ASSERT(llvm_lane_id->getType()->isIntegerTy(32));
            auto [llvm_value_packed, llvm_packed_i32_count] = pack_into_i32_vector(llvm_value);
            auto llvm_result_packed = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_value_packed->getType()));
            for (auto i = 0; i < llvm_packed_i32_count; i++) {
                auto llvm_local_elem = b.CreateExtractElement(llvm_value_packed, i);
                auto llvm_elem_from_lane = b.CreateIntrinsic(
                    llvm::Intrinsic::nvvm_shfl_sync_idx_i32,
                    {llvm_active_mask, llvm_local_elem, llvm_lane_id, b.getInt32(31)});
                llvm_result_packed = b.CreateInsertElement(llvm_result_packed, llvm_elem_from_lane, i);
            }
            return unpack_from_i32_vector(llvm_result_packed, llvm_value->getType());
        }
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: {
#if LLVM_VERSION_MAJOR >= 21
            return b.CreateIntrinsic(llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all, {b.getInt32(0)});
#else
            return b.CreateIntrinsic(llvm::Intrinsic::nvvm_barrier0, {});
#endif
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::cuda
