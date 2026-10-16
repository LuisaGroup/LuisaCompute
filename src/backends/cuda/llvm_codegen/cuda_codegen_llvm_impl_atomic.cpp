//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"
#include <luisa/core/stl/memory.h>

namespace luisa::compute::cuda {

void CUDACodegenLLVMImpl::_legalize_optix_ir_atomics() noexcept {
    llvm::SmallVector<llvm::AtomicRMWInst *, 8u> atomics;
    for (auto &function : *_llvm_module) {
        for (auto &block : function) {
            for (auto &inst : block) {
                if (auto atomic = llvm::dyn_cast<llvm::AtomicRMWInst>(&inst);
                    atomic != nullptr && atomic->getType()->isFloatingPointTy()) {
                    atomics.emplace_back(atomic);
                }
            }
        }
    }
    for (auto atomic : atomics) {
        auto value_type = atomic->getType();
        auto bit_width = value_type->getPrimitiveSizeInBits().getFixedValue();
        LUISA_ASSERT(bit_width == 32u || bit_width == 64u,
                     "OptiX IR floating-point atomics require 32-bit or 64-bit values.");
        auto integer_type = llvm::IntegerType::get(_llvm_context, bit_width);
        auto pointer = atomic->getPointerOperand();
        auto operand = atomic->getValOperand();
        auto alignment = atomic->getAlign();
        auto ordering = atomic->getOrdering();
        auto scope = atomic->getSyncScopeID();
        auto is_volatile = atomic->isVolatile();
        if (atomic->getOperation() == llvm::AtomicRMWInst::Xchg) {
            llvm::IRBuilder<> b{atomic};
            auto bits = b.CreateBitCast(operand, integer_type);
            auto old_bits = b.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg, pointer, bits,
                                              alignment, ordering, scope);
            old_bits->setVolatile(is_volatile);
            atomic->replaceAllUsesWith(b.CreateBitCast(old_bits, value_type));
            atomic->eraseFromParent();
            continue;
        }
        // LLVM 7 cannot encode floating-point atomicrmw. Expand only after the
        // host optimization pipeline so its global fast-math policy cannot
        // weaken the arithmetic or NaN handling of the atomic operation.
        auto entry = atomic->getParent();
        auto function = entry->getParent();
        auto done = entry->splitBasicBlock(atomic->getIterator(), "atomic.fp.done");
        auto loop = llvm::BasicBlock::Create(_llvm_context, "atomic.fp.cas", function, done);
        auto failure_ordering = llvm::AtomicCmpXchgInst::getStrongestFailureOrdering(ordering);
        llvm::IRBuilder<> b{entry->getTerminator()};
        auto initial = b.CreateAlignedLoad(integer_type, pointer, alignment);
        initial->setAtomic(failure_ordering, scope);
        initial->setVolatile(is_volatile);
        entry->getTerminator()->setSuccessor(0u, loop);
        b.SetInsertPoint(loop);
        auto expected_bits = b.CreatePHI(integer_type, 2u, "atomic.fp.expected");
        expected_bits->addIncoming(initial, entry);
        auto old_value = b.CreateBitCast(expected_bits, value_type);
        auto new_value = [&]() -> llvm::Value * {
            switch (atomic->getOperation()) {
                case llvm::AtomicRMWInst::FAdd: return b.CreateFAdd(old_value, operand);
                case llvm::AtomicRMWInst::FSub: return b.CreateFSub(old_value, operand);
                case llvm::AtomicRMWInst::FMin: return b.CreateMinNum(old_value, operand);
                case llvm::AtomicRMWInst::FMax: return b.CreateMaxNum(old_value, operand);
                default: LUISA_ERROR_WITH_LOCATION("Unsupported floating-point atomic for OptiX IR.");
            }
        }();
        auto desired_bits = b.CreateBitCast(new_value, integer_type);
        auto result = b.CreateAtomicCmpXchg(pointer, expected_bits, desired_bits,
                                            alignment, ordering, failure_ordering, scope);
        result->setWeak(false);
        result->setVolatile(is_volatile);
        expected_bits->addIncoming(b.CreateExtractValue(result, 0u), loop);
        // Compare integer bits through cmpxchg's success flag. A floating-point
        // equality test would retry forever when the stored value is NaN.
        b.CreateCondBr(b.CreateExtractValue(result, 1u), done, loop);
        atomic->replaceAllUsesWith(old_value);
        atomic->eraseFromParent();
    }
}

llvm::Value *CUDACodegenLLVMImpl::_translate_atomic_inst(IB &b, FunctionContext &func_ctx, const xir::AtomicInst *inst) noexcept {
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
    switch (inst->op()) {
        case xir::AtomicOp::EXCHANGE: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            return b.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                     llvm_elem_ptr, llvm_values[0],
                                     llvm::MaybeAlign{elem_type->alignment()},
                                     llvm::AtomicOrdering::Monotonic);
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
            auto llvm_old_value = b.CreateExtractValue(llvm_ret, 0);
            if (llvm_value_type->isFloatingPointTy()) {
                llvm_old_value = b.CreateBitCast(llvm_old_value, llvm_value_type);
            }
            return llvm_old_value;
        }
        case xir::AtomicOp::FETCH_ADD: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            auto llvm_op = elem_type->is_float() ? llvm::AtomicRMWInst::FAdd : llvm::AtomicRMWInst::Add;
            return b.CreateAtomicRMW(llvm_op, llvm_elem_ptr, llvm_values[0],
                                     llvm::MaybeAlign{elem_type->alignment()},
                                     llvm::AtomicOrdering::Monotonic);
        }
        case xir::AtomicOp::FETCH_SUB: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            auto llvm_op = elem_type->is_float() ? llvm::AtomicRMWInst::FSub : llvm::AtomicRMWInst::Sub;
            return b.CreateAtomicRMW(llvm_op, llvm_elem_ptr, llvm_values[0],
                                     llvm::MaybeAlign{elem_type->alignment()},
                                     llvm::AtomicOrdering::Monotonic);
        }
        case xir::AtomicOp::FETCH_AND: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            return b.CreateAtomicRMW(llvm::AtomicRMWInst::And,
                                     llvm_elem_ptr, llvm_values[0],
                                     llvm::MaybeAlign{elem_type->alignment()},
                                     llvm::AtomicOrdering::Monotonic);
        }
        case xir::AtomicOp::FETCH_OR: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            return b.CreateAtomicRMW(llvm::AtomicRMWInst::Or,
                                     llvm_elem_ptr, llvm_values[0],
                                     llvm::MaybeAlign{elem_type->alignment()},
                                     llvm::AtomicOrdering::Monotonic);
        }
        case xir::AtomicOp::FETCH_XOR: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1 && llvm_values[0]->getType()->isIntegerTy());
            return b.CreateAtomicRMW(llvm::AtomicRMWInst::Xor,
                                     llvm_elem_ptr, llvm_values[0],
                                     llvm::MaybeAlign{elem_type->alignment()},
                                     llvm::AtomicOrdering::Monotonic);
        }
        case xir::AtomicOp::FETCH_MIN: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            auto llvm_op = elem_type->is_int()  ? llvm::AtomicRMWInst::Min :
                           elem_type->is_uint() ? llvm::AtomicRMWInst::UMin :
                                                  llvm::AtomicRMWInst::FMin;
            return b.CreateAtomicRMW(llvm_op, llvm_elem_ptr, llvm_values[0],
                                     llvm::MaybeAlign{elem_type->alignment()},
                                     llvm::AtomicOrdering::Monotonic);
        }
        case xir::AtomicOp::FETCH_MAX: {
            LUISA_DEBUG_ASSERT(llvm_values.size() == 1);
            auto llvm_op = elem_type->is_int()  ? llvm::AtomicRMWInst::Max :
                           elem_type->is_uint() ? llvm::AtomicRMWInst::UMax :
                                                  llvm::AtomicRMWInst::FMax;
            return b.CreateAtomicRMW(llvm_op, llvm_elem_ptr, llvm_values[0],
                                     llvm::MaybeAlign{elem_type->alignment()},
                                     llvm::AtomicOrdering::Monotonic);
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported atomic operation.");
}

}// namespace luisa::compute::cuda
