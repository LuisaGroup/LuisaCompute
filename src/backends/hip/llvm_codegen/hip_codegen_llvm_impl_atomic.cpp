//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

// Build a <4 x i32> buffer resource descriptor (V#/SRD) from buffer struct {ptr addrspace(1), i64}.
// gfx11/gfx12 format: {addr_lo, addr_hi, size_bytes, 0x31004000}
static llvm::Value *make_buffer_resource_descriptor(llvm::IRBuilder<> &b, llvm::Value *buffer) {
    auto *i32_ty = b.getInt32Ty();
    auto *i64_ty = b.getInt64Ty();
    auto *v4i32_ty = llvm::FixedVectorType::get(i32_ty, 4);

    auto *buffer_ptr = b.CreateExtractValue(buffer, HIPCodegenLLVMImpl::llvm_buffer_type_ptr_index);
    auto *buffer_size = b.CreateExtractValue(buffer, HIPCodegenLLVMImpl::llvm_buffer_type_size_index);

    auto *addr = b.CreatePtrToInt(buffer_ptr, i64_ty, "buf.addr");
    auto *addr_lo = b.CreateTrunc(addr, i32_ty, "buf.addr.lo");
    auto *addr_hi = b.CreateTrunc(b.CreateLShr(addr, 32), i32_ty, "buf.addr.hi");

    auto *size_clamped = b.CreateTrunc(
        b.CreateSelect(
            b.CreateICmpUGT(buffer_size, b.getInt64(UINT32_MAX)),
            b.getInt64(UINT32_MAX), buffer_size),
        i32_ty, "buf.size");

    llvm::Value *rsrc = llvm::UndefValue::get(v4i32_ty);
    rsrc = b.CreateInsertElement(rsrc, addr_lo, b.getInt32(0));
    rsrc = b.CreateInsertElement(rsrc, addr_hi, b.getInt32(1));
    rsrc = b.CreateInsertElement(rsrc, size_clamped, b.getInt32(2));
    rsrc = b.CreateInsertElement(rsrc, b.getInt32(0x31004000), b.getInt32(3), "buf.rsrc");
    return rsrc;
}

// Emit a buffer atomic intrinsic. AMDGPU raw-buffer atomics are overloaded on data_ty.
static llvm::Value *emit_buffer_atomic(llvm::IRBuilder<> &b, llvm::Module *mod, llvm::LLVMContext &ctx,
                                       llvm::Intrinsic::ID intrinsic_id, llvm::Type *data_ty,
                                       llvm::Value *vdata, llvm::Value *rsrc,
                                       llvm::Value *voffset_bytes, llvm::Value *soffset = nullptr,
                                       int cachepolicy = 0) {
    if (!soffset) soffset = b.getInt32(0);
    auto *cachepolicy_val = b.getInt32(cachepolicy);
    auto *func = llvm::Intrinsic::getOrInsertDeclaration(mod, intrinsic_id, {data_ty});
    return b.CreateCall(func, {vdata, rsrc, voffset_bytes, soffset, cachepolicy_val});
}

// buffer_atomic_cmpswap: (src=desired, cmp=expected, rsrc, voffset, soffset, cachepolicy) → old
static llvm::Value *emit_buffer_atomic_cmpswap(llvm::IRBuilder<> &b, llvm::Module *mod, llvm::LLVMContext &ctx,
                                               llvm::Value *src, llvm::Value *cmp, llvm::Value *rsrc,
                                               llvm::Value *voffset_bytes) {
    auto *func = llvm::Intrinsic::getOrInsertDeclaration(mod, llvm::Intrinsic::amdgcn_raw_buffer_atomic_cmpswap, {b.getInt32Ty()});
    return b.CreateCall(func, {src, cmp, rsrc, voffset_bytes, b.getInt32(0), b.getInt32(0)});
}

llvm::Value *HIPCodegenLLVMImpl::_translate_atomic_inst(IB &b, FunctionContext &func_ctx, const xir::AtomicInst *inst) noexcept {
    auto index_uses = inst->index_uses();

    auto base = inst->base();
    auto is_buffer = base->type()->is_buffer();

    llvm::Value *llvm_buf_rsrc = nullptr;
    llvm::Value *llvm_buf_voffset = nullptr;
    llvm::Value *llvm_elem_ptr = nullptr;
    const Type *elem_type = nullptr;

    if (is_buffer) {
        auto llvm_base = _get_llvm_value(b, func_ctx, base);
        LUISA_DEBUG_ASSERT(!index_uses.empty());
        auto llvm_index = _get_llvm_value(b, func_ctx, index_uses.front()->value());
        index_uses = index_uses.subspan(1);
        elem_type = base->type()->element();
        LUISA_DEBUG_ASSERT(elem_type != nullptr);

        llvm_buf_rsrc = make_buffer_resource_descriptor(b, llvm_base);

        auto index_i64 = b.CreateZExt(llvm_index, b.getInt64Ty(), "", true);
        auto offset_bytes = b.CreateMul(index_i64, b.getInt64(elem_type->size()), "", true, true);

        if (!index_uses.empty()) {
            auto llvm_base_ptr = b.CreateExtractValue(llvm_base, llvm_buffer_type_ptr_index);
            auto flat_ptr = b.CreateInBoundsGEP(b.getInt8Ty(), llvm_base_ptr, offset_bytes);
            auto [chain_ptr, chain_type] = _lower_access_chain_address(b, func_ctx, flat_ptr, elem_type, index_uses);
            auto chain_ptr_int = b.CreatePtrToInt(chain_ptr, b.getInt64Ty());
            auto base_ptr_int = b.CreatePtrToInt(llvm_base_ptr, b.getInt64Ty());
            offset_bytes = b.CreateSub(chain_ptr_int, base_ptr_int);
            elem_type = chain_type;
        }

        llvm_buf_voffset = b.CreateTrunc(offset_bytes, b.getInt32Ty(), "buf.voffset");
        LUISA_DEBUG_ASSERT(elem_type == inst->type() && elem_type->is_scalar());
    } else {
        auto llvm_base = _get_llvm_value(b, func_ctx, base);
        auto [ptr, type] = _lower_access_chain_address(b, func_ctx, llvm_base, base->type(), index_uses);
        llvm_elem_ptr = ptr;
        elem_type = type;
        LUISA_DEBUG_ASSERT(elem_type == inst->type() && elem_type->is_scalar());
    }

    llvm::SmallVector<llvm::Value *, 2> llvm_values;
    for (auto value_use : inst->value_uses()) {
        llvm_values.emplace_back(_get_llvm_value(b, func_ctx, value_use->value()));
    }

    auto agent_scope = _llvm_context.getOrInsertSyncScopeID("agent");

    auto emit_atomicrmw = [&](llvm::AtomicRMWInst::BinOp op, llvm::Value *val) -> llvm::Value * {
        auto *rmw = b.CreateAtomicRMW(op, llvm_elem_ptr, val,
                                      llvm::MaybeAlign{elem_type->alignment()},
                                      llvm::AtomicOrdering::Monotonic);
        rmw->setSyncScopeID(agent_scope);
        return rmw;
    };

    switch (inst->op()) {
        case xir::AtomicOp::EXCHANGE: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            if (is_buffer) {
                auto *val = llvm_values[0];
                if (val->getType()->isFloatingPointTy()) {
                    val = b.CreateBitCast(val, b.getInt32Ty());
                }
                auto *result = emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                                  llvm::Intrinsic::amdgcn_raw_buffer_atomic_swap,
                                                  b.getInt32Ty(), val, llvm_buf_rsrc, llvm_buf_voffset);
                if (llvm_values[0]->getType()->isFloatingPointTy()) {
                    result = b.CreateBitCast(result, llvm_values[0]->getType());
                }
                return result;
            }
            return emit_atomicrmw(llvm::AtomicRMWInst::Xchg, llvm_values[0]);
        }
        case xir::AtomicOp::COMPARE_EXCHANGE: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 2);
            auto llvm_expected_value = llvm_values[0];
            auto llvm_desired_value = llvm_values[1];
            LUISA_DEBUG_ASSERT(llvm_expected_value->getType() == llvm_desired_value->getType());
            auto llvm_value_type = llvm_expected_value->getType();
            if (is_buffer) {
                if (llvm_value_type->isFloatingPointTy()) {
                    llvm_expected_value = b.CreateBitCast(llvm_expected_value, b.getInt32Ty());
                    llvm_desired_value = b.CreateBitCast(llvm_desired_value, b.getInt32Ty());
                }
                // cmpswap(src=desired, cmp=expected, rsrc, offset, soffset, cachepolicy)
                auto *result = emit_buffer_atomic_cmpswap(b, _llvm_module.get(), _llvm_context,
                                                          llvm_desired_value, llvm_expected_value,
                                                          llvm_buf_rsrc, llvm_buf_voffset);
                if (llvm_value_type->isFloatingPointTy()) {
                    result = b.CreateBitCast(result, llvm_value_type);
                }
                return result;
            }
            if (llvm_value_type->isFloatingPointTy()) {
                auto llvm_int_type = llvm::IntegerType::get(_llvm_context, llvm_value_type->getPrimitiveSizeInBits());
                llvm_expected_value = b.CreateBitCast(llvm_expected_value, llvm_int_type);
                llvm_desired_value = b.CreateBitCast(llvm_desired_value, llvm_int_type);
            }
            auto llvm_ret = b.CreateAtomicCmpXchg(llvm_elem_ptr,
                                                  llvm_expected_value, llvm_desired_value,
                                                  llvm::MaybeAlign{elem_type->alignment()},
                                                  llvm::AtomicOrdering::Monotonic,
                                                  llvm::AtomicOrdering::Monotonic);
            llvm_ret->setSyncScopeID(agent_scope);
            auto llvm_old_value = b.CreateExtractValue(llvm_ret, 0);
            if (llvm_value_type->isFloatingPointTy()) {
                llvm_old_value = b.CreateBitCast(llvm_old_value, llvm_value_type);
            }
            return llvm_old_value;
        }
        case xir::AtomicOp::FETCH_ADD: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            if (is_buffer) {
                if (elem_type->is_float()) {
                    // CAS loop instead of native fadd: fire-and-forget fadd floods L2 atomic units on RDNA 4
                    auto *func = b.GetInsertBlock()->getParent();
                    auto *loop_bb = llvm::BasicBlock::Create(_llvm_context, "cas.loop", func);
                    auto *exit_bb = llvm::BasicBlock::Create(_llvm_context, "cas.exit", func);

                    auto *load_func = llvm::Intrinsic::getOrInsertDeclaration(
                        _llvm_module.get(), llvm::Intrinsic::amdgcn_raw_buffer_load, {b.getFloatTy()});
                    auto *initial = b.CreateCall(load_func,
                                                 {llvm_buf_rsrc, llvm_buf_voffset, b.getInt32(0), b.getInt32(0)});
                    auto *initial_i32 = b.CreateBitCast(initial, b.getInt32Ty());
                    b.CreateBr(loop_bb);

                    b.SetInsertPoint(loop_bb);
                    auto *old_phi = b.CreatePHI(b.getInt32Ty(), 2, "cas.old");
                    old_phi->addIncoming(initial_i32, loop_bb->getSinglePredecessor());

                    auto *old_f = b.CreateBitCast(old_phi, b.getFloatTy());
                    auto *new_f = b.CreateFAdd(old_f, llvm_values[0]);
                    auto *new_i32 = b.CreateBitCast(new_f, b.getInt32Ty());

                    auto *prev = emit_buffer_atomic_cmpswap(b, _llvm_module.get(), _llvm_context,
                                                            new_i32, old_phi, llvm_buf_rsrc, llvm_buf_voffset);

                    auto *success = b.CreateICmpEQ(prev, old_phi);
                    old_phi->addIncoming(prev, loop_bb);
                    b.CreateCondBr(success, exit_bb, loop_bb);

                    b.SetInsertPoint(exit_bb);
                    return b.CreateBitCast(prev, b.getFloatTy());
                } else {
                    return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                              llvm::Intrinsic::amdgcn_raw_buffer_atomic_add,
                                              b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
                }
            }
            auto llvm_op = elem_type->is_float() ? llvm::AtomicRMWInst::FAdd : llvm::AtomicRMWInst::Add;
            return emit_atomicrmw(llvm_op, llvm_values[0]);
        }
        case xir::AtomicOp::FETCH_SUB: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            if (is_buffer) {
                if (elem_type->is_float()) {
                    auto *neg_val = b.CreateFNeg(llvm_values[0]);
                    // CAS loop instead of native fadd: fire-and-forget fadd floods L2 atomic units on RDNA 4
                    auto *func = b.GetInsertBlock()->getParent();
                    auto *loop_bb = llvm::BasicBlock::Create(_llvm_context, "cas.loop", func);
                    auto *exit_bb = llvm::BasicBlock::Create(_llvm_context, "cas.exit", func);

                    auto *load_func = llvm::Intrinsic::getOrInsertDeclaration(
                        _llvm_module.get(), llvm::Intrinsic::amdgcn_raw_buffer_load, {b.getFloatTy()});
                    auto *initial = b.CreateCall(load_func,
                                                 {llvm_buf_rsrc, llvm_buf_voffset, b.getInt32(0), b.getInt32(0)});
                    auto *initial_i32 = b.CreateBitCast(initial, b.getInt32Ty());
                    b.CreateBr(loop_bb);

                    b.SetInsertPoint(loop_bb);
                    auto *old_phi = b.CreatePHI(b.getInt32Ty(), 2, "cas.old");
                    old_phi->addIncoming(initial_i32, loop_bb->getSinglePredecessor());

                    auto *old_f = b.CreateBitCast(old_phi, b.getFloatTy());
                    auto *new_f = b.CreateFAdd(old_f, neg_val);
                    auto *new_i32 = b.CreateBitCast(new_f, b.getInt32Ty());

                    auto *prev = emit_buffer_atomic_cmpswap(b, _llvm_module.get(), _llvm_context,
                                                            new_i32, old_phi, llvm_buf_rsrc, llvm_buf_voffset);

                    auto *success = b.CreateICmpEQ(prev, old_phi);
                    old_phi->addIncoming(prev, loop_bb);
                    b.CreateCondBr(success, exit_bb, loop_bb);

                    b.SetInsertPoint(exit_bb);
                    return b.CreateBitCast(prev, b.getFloatTy());
                } else {
                    return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                              llvm::Intrinsic::amdgcn_raw_buffer_atomic_sub,
                                              b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
                }
            }
            auto llvm_op = elem_type->is_float() ? llvm::AtomicRMWInst::FSub : llvm::AtomicRMWInst::Sub;
            return emit_atomicrmw(llvm_op, llvm_values[0]);
        }
        case xir::AtomicOp::FETCH_AND: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            if (is_buffer) {
                return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                          llvm::Intrinsic::amdgcn_raw_buffer_atomic_and,
                                          b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
            }
            return emit_atomicrmw(llvm::AtomicRMWInst::And, llvm_values[0]);
        }
        case xir::AtomicOp::FETCH_OR: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            if (is_buffer) {
                return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                          llvm::Intrinsic::amdgcn_raw_buffer_atomic_or,
                                          b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
            }
            return emit_atomicrmw(llvm::AtomicRMWInst::Or, llvm_values[0]);
        }
        case xir::AtomicOp::FETCH_XOR: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            if (is_buffer) {
                return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                          llvm::Intrinsic::amdgcn_raw_buffer_atomic_xor,
                                          b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
            }
            return emit_atomicrmw(llvm::AtomicRMWInst::Xor, llvm_values[0]);
        }
        case xir::AtomicOp::FETCH_MIN: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            if (is_buffer) {
                if (elem_type->is_float()) {
                    return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                              llvm::Intrinsic::amdgcn_raw_buffer_atomic_fmin,
                                              b.getFloatTy(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
                } else if (elem_type->is_int()) {
                    return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                              llvm::Intrinsic::amdgcn_raw_buffer_atomic_smin,
                                              b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
                } else {
                    return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                              llvm::Intrinsic::amdgcn_raw_buffer_atomic_umin,
                                              b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
                }
            }
            auto llvm_op = elem_type->is_int()  ? llvm::AtomicRMWInst::Min :
                           elem_type->is_uint() ? llvm::AtomicRMWInst::UMin :
                                                  llvm::AtomicRMWInst::FMin;
            return emit_atomicrmw(llvm_op, llvm_values[0]);
        }
        case xir::AtomicOp::FETCH_MAX: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            if (is_buffer) {
                if (elem_type->is_float()) {
                    return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                              llvm::Intrinsic::amdgcn_raw_buffer_atomic_fmax,
                                              b.getFloatTy(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
                } else if (elem_type->is_int()) {
                    return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                              llvm::Intrinsic::amdgcn_raw_buffer_atomic_smax,
                                              b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
                } else {
                    return emit_buffer_atomic(b, _llvm_module.get(), _llvm_context,
                                              llvm::Intrinsic::amdgcn_raw_buffer_atomic_umax,
                                              b.getInt32Ty(), llvm_values[0], llvm_buf_rsrc, llvm_buf_voffset);
                }
            }
            auto llvm_op = elem_type->is_int()  ? llvm::AtomicRMWInst::Max :
                           elem_type->is_uint() ? llvm::AtomicRMWInst::UMax :
                                                  llvm::AtomicRMWInst::FMax;
            return emit_atomicrmw(llvm_op, llvm_values[0]);
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported atomic operation.");
}

}// namespace luisa::compute::hip
