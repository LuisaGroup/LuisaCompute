//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_translate_atomic_inst(IB &b, FunctionContext &func_ctx, const xir::AtomicInst *inst) noexcept {
    auto index_uses = inst->index_uses();
    auto [llvm_ptr, base_type] = [this, &b, &func_ctx, inst, &index_uses] {
        auto base = inst->base();
        auto llvm_base = _get_llvm_value(b, func_ctx, base);
        if (base->type()->is_buffer()) {
            LUISA_DEBUG_ASSERT(!index_uses.empty());
            auto llvm_index = _get_llvm_value(b, func_ctx, index_uses.front()->value());
            index_uses = index_uses.subspan(1);
            auto elem_type = base->type()->element();
            LUISA_DEBUG_ASSERT(elem_type != nullptr);
            auto llvm_elem_ptr = _get_buffer_element_pointer(b, llvm_base, llvm_index, elem_type->size(), elem_type->size());
            return std::make_pair(llvm_elem_ptr, elem_type);
        }
        return std::make_pair(llvm_base, base->type());
    }();
    auto [llvm_elem_ptr, elem_type] = _lower_access_chain_address(b, func_ctx, llvm_ptr, base_type, index_uses);
    LUISA_DEBUG_ASSERT(elem_type == inst->type() && elem_type->is_scalar());
    llvm::SmallVector<llvm::Value *, 2> llvm_values;
    for (auto value_use : inst->value_uses()) {
        llvm_values.emplace_back(_get_llvm_value(b, func_ctx, value_use->value()));
    }
    // Use "agent" (device) syncscope for all atomics on AMDGPU.
    // Default (no syncscope) maps to SCOPE_SYS which can be problematic on some targets.
    auto agent_scope = _llvm_context.getOrInsertSyncScopeID("agent");
    switch (inst->op()) {
        case xir::AtomicOp::EXCHANGE: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            auto *rmw = b.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                          llvm_elem_ptr, llvm_values[0],
                                          llvm::MaybeAlign{elem_type->alignment()},
                                          llvm::AtomicOrdering::Monotonic);
            rmw->setSyncScopeID(agent_scope);
            return static_cast<llvm::Value *>(rmw);
        }
        case xir::AtomicOp::COMPARE_EXCHANGE: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 2);
            auto llvm_expected_value = llvm_values[0];
            auto llvm_desired_value = llvm_values[1];
            LUISA_DEBUG_ASSERT(llvm_expected_value->getType() == llvm_desired_value->getType());
            auto llvm_value_type = llvm_expected_value->getType();
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
            auto llvm_op = elem_type->is_float() ? llvm::AtomicRMWInst::FAdd : llvm::AtomicRMWInst::Add;
            auto *rmw = b.CreateAtomicRMW(llvm_op, llvm_elem_ptr, llvm_values[0],
                                          llvm::MaybeAlign{elem_type->alignment()},
                                          llvm::AtomicOrdering::Monotonic);
            rmw->setSyncScopeID(agent_scope);
            return static_cast<llvm::Value *>(rmw);
        }
        case xir::AtomicOp::FETCH_SUB: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            auto llvm_op = elem_type->is_float() ? llvm::AtomicRMWInst::FSub : llvm::AtomicRMWInst::Sub;
            auto *rmw = b.CreateAtomicRMW(llvm_op, llvm_elem_ptr, llvm_values[0],
                                          llvm::MaybeAlign{elem_type->alignment()},
                                          llvm::AtomicOrdering::Monotonic);
            rmw->setSyncScopeID(agent_scope);
            return static_cast<llvm::Value *>(rmw);
        }
        case xir::AtomicOp::FETCH_AND: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            auto *rmw = b.CreateAtomicRMW(llvm::AtomicRMWInst::And,
                                          llvm_elem_ptr, llvm_values[0],
                                          llvm::MaybeAlign{elem_type->alignment()},
                                          llvm::AtomicOrdering::Monotonic);
            rmw->setSyncScopeID(agent_scope);
            return static_cast<llvm::Value *>(rmw);
        }
        case xir::AtomicOp::FETCH_OR: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            auto *rmw = b.CreateAtomicRMW(llvm::AtomicRMWInst::Or,
                                          llvm_elem_ptr, llvm_values[0],
                                          llvm::MaybeAlign{elem_type->alignment()},
                                          llvm::AtomicOrdering::Monotonic);
            rmw->setSyncScopeID(agent_scope);
            return static_cast<llvm::Value *>(rmw);
        }
        case xir::AtomicOp::FETCH_XOR: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            auto *rmw = b.CreateAtomicRMW(llvm::AtomicRMWInst::Xor,
                                          llvm_elem_ptr, llvm_values[0],
                                          llvm::MaybeAlign{elem_type->alignment()},
                                          llvm::AtomicOrdering::Monotonic);
            rmw->setSyncScopeID(agent_scope);
            return static_cast<llvm::Value *>(rmw);
        }
        case xir::AtomicOp::FETCH_MIN: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            auto llvm_op = elem_type->is_int()  ? llvm::AtomicRMWInst::Min :
                           elem_type->is_uint() ? llvm::AtomicRMWInst::UMin :
                                                  llvm::AtomicRMWInst::FMin;
            auto *rmw = b.CreateAtomicRMW(llvm_op, llvm_elem_ptr, llvm_values[0],
                                          llvm::MaybeAlign{elem_type->alignment()},
                                          llvm::AtomicOrdering::Monotonic);
            rmw->setSyncScopeID(agent_scope);
            return static_cast<llvm::Value *>(rmw);
        }
        case xir::AtomicOp::FETCH_MAX: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            auto llvm_op = elem_type->is_int()  ? llvm::AtomicRMWInst::Max :
                           elem_type->is_uint() ? llvm::AtomicRMWInst::UMax :
                                                  llvm::AtomicRMWInst::FMax;
            auto *rmw = b.CreateAtomicRMW(llvm_op, llvm_elem_ptr, llvm_values[0],
                                          llvm::MaybeAlign{elem_type->alignment()},
                                          llvm::AtomicOrdering::Monotonic);
            rmw->setSyncScopeID(agent_scope);
            return static_cast<llvm::Value *>(rmw);
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported atomic operation.");
}

}// namespace luisa::compute::hip
