#include "hip_private_memory.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include <luisa/core/logging.h>

namespace luisa::compute::hip {

namespace {

struct ByteRange {
    int64_t begin;
    int64_t end;
};

[[nodiscard]] bool overlaps(ByteRange lhs, ByteRange rhs) noexcept {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

[[nodiscard]] bool is_lifetime_marker(const llvm::CallBase *call) noexcept {
    if (auto intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(call)) {
        return intrinsic->getIntrinsicID() == llvm::Intrinsic::lifetime_start ||
               intrinsic->getIntrinsicID() == llvm::Intrinsic::lifetime_end;
    }
    return false;
}

[[nodiscard]] bool type_store_size(const llvm::DataLayout &layout,
                                   llvm::Type *type,
                                   int64_t &size) noexcept {
    const auto bytes = layout.getTypeStoreSize(type);
    if (bytes.isScalable() ||
        bytes.getFixedValue() >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    size = static_cast<int64_t>(bytes.getFixedValue());
    return true;
}

[[nodiscard]] bool checked_range(int64_t offset, int64_t size,
                                 ByteRange &range) noexcept {
    if (size < 0 ||
        offset > std::numeric_limits<int64_t>::max() - size) {
        return false;
    }
    range = {.begin = offset, .end = offset + size};
    return true;
}

[[nodiscard]] bool constant_gep_offset(const llvm::DataLayout &layout,
                                       const llvm::GetElementPtrInst *gep,
                                       int64_t base_offset,
                                       int64_t &result) noexcept {
    auto bits = layout.getIndexTypeSizeInBits(gep->getType());
    llvm::APInt offset{bits, 0u, true};
    if (!gep->accumulateConstantOffset(layout, offset) ||
        !offset.isSignedIntN(64u)) {
        return false;
    }
    const auto delta = offset.getSExtValue();
    if ((delta > 0 &&
         base_offset > std::numeric_limits<int64_t>::max() - delta) ||
        (delta < 0 &&
         base_offset < std::numeric_limits<int64_t>::min() - delta)) {
        return false;
    }
    result = base_offset + delta;
    return true;
}

struct AllocaAccessProof {
    llvm::DenseMap<llvm::Value *, int64_t> pointer_offsets;
    llvm::SmallVector<llvm::PtrToIntInst *, 2u> pointer_integers;
    llvm::SmallVector<ByteRange, 8u> read_ranges;
    bool complete{true};
};

[[nodiscard]] AllocaAccessProof
analyze_alloca_accesses(llvm::AllocaInst *alloca,
                        const llvm::DataLayout &layout) noexcept {
    AllocaAccessProof proof;
    proof.pointer_offsets.try_emplace(alloca, 0);
    llvm::SmallVector<llvm::Value *, 16u> worklist{alloca};

    while (!worklist.empty() && proof.complete) {
        auto value = worklist.pop_back_val();
        const auto base_offset = proof.pointer_offsets.lookup(value);
        for (auto *user : value->users()) {
            if (auto gep = llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {
                if (gep->getPointerOperand() != value) {
                    proof.complete = false;
                    break;
                }
                auto offset = int64_t{};
                if (!constant_gep_offset(layout, gep, base_offset, offset)) {
                    proof.complete = false;
                    break;
                }
                auto [it, inserted] =
                    proof.pointer_offsets.try_emplace(gep, offset);
                if (!inserted && it->second != offset) {
                    proof.complete = false;
                    break;
                }
                if (inserted) { worklist.emplace_back(gep); }
                continue;
            }
            if (auto cast = llvm::dyn_cast<llvm::CastInst>(user);
                cast != nullptr && cast->getOperand(0u) == value &&
                (cast->getOpcode() == llvm::Instruction::BitCast ||
                 cast->getOpcode() == llvm::Instruction::AddrSpaceCast)) {
                auto [it, inserted] =
                    proof.pointer_offsets.try_emplace(cast, base_offset);
                if (!inserted && it->second != base_offset) {
                    proof.complete = false;
                    break;
                }
                if (inserted) { worklist.emplace_back(cast); }
                continue;
            }
            if (auto pointer_integer = llvm::dyn_cast<llvm::PtrToIntInst>(user);
                pointer_integer != nullptr &&
                pointer_integer->getPointerOperand() == value) {
                proof.pointer_integers.emplace_back(pointer_integer);
                continue;
            }
            if (auto load = llvm::dyn_cast<llvm::LoadInst>(user)) {
                if (load->getPointerOperand() != value || load->isVolatile() ||
                    load->isAtomic()) {
                    proof.complete = false;
                    break;
                }
                auto size = int64_t{};
                auto range = ByteRange{};
                if (!type_store_size(layout, load->getType(), size) ||
                    !checked_range(base_offset, size, range)) {
                    proof.complete = false;
                    break;
                }
                proof.read_ranges.emplace_back(range);
                continue;
            }
            if (auto store = llvm::dyn_cast<llvm::StoreInst>(user)) {
                if (store->getPointerOperand() == value) {
                    if (store->isVolatile() || store->isAtomic()) {
                        proof.complete = false;
                        break;
                    }
                    continue;
                }
                // Storing a derived pointer exposes the aggregate through an
                // unmodeled alias. A pointer used as both value and address is
                // rejected by the same condition.
                proof.complete = false;
                break;
            }
            if (auto call = llvm::dyn_cast<llvm::CallBase>(user)) {
                if (is_lifetime_marker(call)) { continue; }
                proof.complete = false;
                break;
            }
            // PHI/select, comparisons, returns, atomics and every other use
            // make pointer identity or an unknown memory region observable.
            proof.complete = false;
            break;
        }
    }
    return proof;
}

[[nodiscard]] size_t eliminate_dead_self_references(
    llvm::AllocaInst *alloca, const llvm::DataLayout &layout) noexcept {
    auto proof = analyze_alloca_accesses(alloca, layout);
    if (!proof.complete || proof.pointer_integers.empty()) { return 0u; }

    llvm::SmallVector<llvm::StoreInst *, 4u> removable_stores;
    llvm::DenseSet<llvm::StoreInst *> unique_stores;
    for (auto pointer_integer : proof.pointer_integers) {
        if (pointer_integer->use_empty()) { continue; }
        llvm::SmallVector<llvm::StoreInst *, 2u> stores;
        for (auto *user : pointer_integer->users()) {
            auto store = llvm::dyn_cast<llvm::StoreInst>(user);
            if (store == nullptr ||
                store->getValueOperand() != pointer_integer ||
                store->isVolatile() || store->isAtomic()) {
                return 0u;
            }
            auto found = proof.pointer_offsets.find(
                store->getPointerOperand());
            if (found == proof.pointer_offsets.end()) { return 0u; }
            auto size = int64_t{};
            auto written = ByteRange{};
            if (!type_store_size(layout, pointer_integer->getType(), size) ||
                !checked_range(found->second, size, written) ||
                std::any_of(proof.read_ranges.begin(),
                            proof.read_ranges.end(),
                            [written](auto read) noexcept {
                                return overlaps(written, read);
                            })) {
                return 0u;
            }
            stores.emplace_back(store);
        }
        for (auto store : stores) {
            if (unique_stores.insert(store).second) {
                removable_stores.emplace_back(store);
            }
        }
    }
    if (removable_stores.empty()) { return 0u; }

    for (auto store : removable_stores) { store->eraseFromParent(); }
    for (auto pointer_integer : proof.pointer_integers) {
        if (pointer_integer->use_empty()) {
            pointer_integer->eraseFromParent();
        }
    }
    return removable_stores.size();
}

void run_scalar_cleanup(llvm::Module &module,
                        llvm::TargetMachine *target_machine) noexcept {
    llvm::LoopAnalysisManager loop_analyses;
    llvm::FunctionAnalysisManager function_analyses;
    llvm::CGSCCAnalysisManager cgscc_analyses;
    llvm::ModuleAnalysisManager module_analyses;
    llvm::PassInstrumentationCallbacks instrumentation;
    llvm::PassBuilder builder{
        target_machine, llvm::PipelineTuningOptions{}, std::nullopt,
        &instrumentation};
    builder.registerModuleAnalyses(module_analyses);
    builder.registerCGSCCAnalyses(cgscc_analyses);
    builder.registerFunctionAnalyses(function_analyses);
    builder.registerLoopAnalyses(loop_analyses);
    builder.crossRegisterProxies(loop_analyses, function_analyses,
                                 cgscc_analyses, module_analyses);
    if (target_machine != nullptr) {
#if LLVM_VERSION_MAJOR >= 19
        target_machine->registerPassBuilderCallbacks(builder);
#else
        target_machine->registerPassBuilderCallbacks(builder, true);
#endif
    }

    llvm::ModulePassManager cleanup;
    if (auto error = builder.parsePassPipeline(
            cleanup,
            "function(sroa,instcombine,simplifycfg,dse,adce)")) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to build HIP private-memory cleanup pipeline: {}.",
            llvm::toString(std::move(error)));
    }
    cleanup.run(module, module_analyses);
}

}// namespace

HIPPrivateMemoryOptimizationStats
optimize_hip_private_memory(llvm::Module &module,
                            llvm::TargetMachine *target_machine) noexcept {
    HIPPrivateMemoryOptimizationStats stats;
    llvm::SmallVector<llvm::AllocaInst *, 16u> allocas;
    for (auto &function : module) {
        if (function.isDeclaration()) { continue; }
        for (auto &block : function) {
            for (auto &instruction : block) {
                if (auto alloca = llvm::dyn_cast<llvm::AllocaInst>(
                        &instruction)) {
                    allocas.emplace_back(alloca);
                }
            }
        }
    }
    stats.analyzed_allocas = allocas.size();
    const auto &layout = module.getDataLayout();
    for (auto alloca : allocas) {
        stats.eliminated_self_reference_stores +=
            eliminate_dead_self_references(alloca, layout);
    }
    if (stats.eliminated_self_reference_stores != 0u) {
        run_scalar_cleanup(module, target_machine);
    }
    return stats;
}

}// namespace luisa::compute::hip
