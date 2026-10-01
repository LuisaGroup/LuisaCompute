#pragma once

#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Transforms/Utils/FunctionComparator.h>

namespace luisa::compute::cuda {

// Share complete hardware-result dispatch bodies only when their callbacks
// cannot depend on per-query captures. Trace flags and query IDs stay at the
// original call sites; each invocation still receives fresh local query state.
class RayQueryCallbackSharing {

public:
    struct Target {
        llvm::BasicBlock *block;
        bool created;
    };

private:
    [[nodiscard]] static bool _same_instruction_flags(const llvm::Function *lhs, const llvm::Function *rhs) noexcept {
        // LLVM 22/23 cmpOperations is not virtual and its GEP fast path ignores
        // no-wrap flags. Walk matching successor indices from the entry blocks
        // to check every instruction, including unused results. Require a
        // bijection; physical block order and unreachable blocks do not matter.
        struct BlockPair {
            const llvm::BasicBlock *left;
            const llvm::BasicBlock *right;
        };
        llvm::SmallVector<BlockPair, 8u> pending{{&lhs->getEntryBlock(), &rhs->getEntryBlock()}};
        llvm::DenseMap<const llvm::BasicBlock *, const llvm::BasicBlock *> left_to_right;
        llvm::DenseMap<const llvm::BasicBlock *, const llvm::BasicBlock *> right_to_left;
        while (!pending.empty()) {
            auto [left, right] = pending.pop_back_val();
            auto [iter, inserted] = left_to_right.try_emplace(left, right);
            if (!inserted) {
                if (iter->second != right) { return false; }
                continue;
            }
            if (!right_to_left.try_emplace(right, left).second) { return false; }
            auto l = left->begin();
            auto r = right->begin();
            for (; l != left->end() && r != right->end(); ++l, ++r) {
                if (l->getOpcode() != r->getOpcode() ||
                    l->getRawSubclassOptionalData() != r->getRawSubclassOptionalData()) {
                    return false;
                }
            }
            if (l != left->end() || r != right->end()) { return false; }
            auto left_term = left->getTerminator();
            auto right_term = right->getTerminator();
            if (left_term == nullptr || right_term == nullptr ||
                left_term->getNumSuccessors() != right_term->getNumSuccessors()) { return false; }
            for (auto i = 0u; i < left_term->getNumSuccessors(); i++) {
                pending.emplace_back(BlockPair{left_term->getSuccessor(i), right_term->getSuccessor(i)});
            }
        }
        return true;
    }

    struct Leader {
        const llvm::Function *callback;
        llvm::BasicBlock *block;
    };
    llvm::Function *_dispatcher;
    // FunctionComparator requires stable global numbering across comparisons.
    // Callback bodies and globals must remain unchanged for this table's life.
    llvm::GlobalNumberState _global_numbers;
    llvm::SmallVector<Leader, 4u> _leaders;
    size_t _shared_count{0u};

    [[nodiscard]] static bool _eligible(const llvm::Function *callback) noexcept {
        if (callback == nullptr || callback->isDeclaration() ||
            !callback->hasPrivateLinkage() || callback->isVarArg() ||
            callback->getCallingConv() != llvm::CallingConv::PTX_Device ||
            !callback->getReturnType()->isVoidTy() || callback->arg_size() != 3u) {
            return false;
        }
        auto pointer = llvm::dyn_cast<llvm::PointerType>(callback->getArg(0u)->getType());
        auto dispatch_size = llvm::dyn_cast<llvm::FixedVectorType>(callback->getArg(1u)->getType());
        if (pointer == nullptr || pointer->getAddressSpace() != 0u ||
            dispatch_size == nullptr || dispatch_size->getNumElements() != 3u ||
            !dispatch_size->getElementType()->isIntegerTy(32u) ||
            !callback->getArg(2u)->getType()->isIntegerTy(32u)) {
            return false;
        }
        // LLVM 22/23 FunctionComparator does not compare every function property
        // or recursively compare arbitrary instruction metadata. Generated RQ
        // callbacks need none of these features: retain separate targets if they
        // appear, including observable function/block identities.
        if (callback->hasPersonalityFn() || callback->hasGC() ||
            callback->hasPrefixData() || callback->hasPrologueData() ||
            callback->hasComdat() || callback->hasSection() || callback->hasPartition() ||
            callback->hasMetadata() || callback->hasSanitizerMetadata() ||
            callback->hasAddressTaken(nullptr, false, false)) {
            return false;
        }
        for (auto &&block : *callback) {
            if (block.hasAddressTaken()) { return false; }
            for (auto &&inst : block) {
                if (inst.hasMetadataOtherThanDebugLoc()) { return false; }
                for (auto &&operand : inst.operands()) {
                    if (llvm::isa<llvm::MetadataAsValue>(operand.get())) { return false; }
                }
            }
        }
        return true;
    }

    [[nodiscard]] bool _equivalent(const llvm::Function *lhs, const llvm::Function *rhs) noexcept {
        // Preserve the exact ABI and optimization contract before asking LLVM
        // to compare CFG, constants, callees, memory effects and instruction flags.
        return lhs->getFunctionType() == rhs->getFunctionType() &&
               lhs->getAttributes() == rhs->getAttributes() &&
               lhs->getCallingConv() == rhs->getCallingConv() &&
               lhs->getAddressSpace() == rhs->getAddressSpace() &&
               lhs->getVisibility() == rhs->getVisibility() &&
               lhs->getDLLStorageClass() == rhs->getDLLStorageClass() &&
               lhs->isDSOLocal() == rhs->isDSOLocal() &&
               lhs->getUnnamedAddr() == rhs->getUnnamedAddr() &&
               lhs->getAlign() == rhs->getAlign() &&
               llvm::FunctionComparator{lhs, rhs, &_global_numbers}.compare() == 0 &&
               _same_instruction_flags(lhs, rhs);
    }

public:
    explicit RayQueryCallbackSharing(llvm::Function *dispatcher) noexcept
        : _dispatcher{dispatcher} {}

    [[nodiscard]] Target get_or_create(const llvm::Function *callback, bool hardware_result,
                                       size_t capture_count, const llvm::Twine &name) noexcept {
        auto eligible = hardware_result && capture_count == 0u && _eligible(callback);
        if (eligible) {
            for (auto &&leader : _leaders) {
                if (_equivalent(leader.callback, callback)) {
                    _shared_count++;
                    return {leader.block, false};
                }
            }
        }
        auto block = llvm::BasicBlock::Create(_dispatcher->getContext(), name, _dispatcher);
        if (eligible) { _leaders.emplace_back(Leader{callback, block}); }
        return {block, true};
    }

    [[nodiscard]] size_t shared_count() const noexcept { return _shared_count; }
};

}// namespace luisa::compute::cuda
