#include "hip_switch_table.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::hip {

namespace {

using namespace llvm;

[[nodiscard]] bool all_predecessors_are(
    BasicBlock *block, BasicBlock *expected) noexcept {
    for (auto it = pred_begin(block), end = pred_end(block); it != end; ++it) {
        if (*it != expected) { return false; }
    }
    return true;
}

[[nodiscard]] BasicBlock *unique_successor(
    const SwitchInst *switch_inst, SmallVectorImpl<BasicBlock *> &arms) noexcept {
    BasicBlock *merge = nullptr;
    for (auto i = 0u; i < switch_inst->getNumSuccessors(); i++) {
        auto *arm = switch_inst->getSuccessor(i);
        if (std::find(arms.begin(), arms.end(), arm) == arms.end()) {
            arms.emplace_back(arm);
        }
        if (arm->size() == 1u &&
            isa<BranchInst>(arm->getTerminator()) &&
            cast<BranchInst>(arm->getTerminator())->isUnconditional()) {
            auto *target = arm->getTerminator()->getSuccessor(0u);
            if (merge == nullptr) {
                merge = target;
            } else if (merge != target) {
                return nullptr;
            }
        }
    }
    return merge;
}

}// namespace

HIPSwitchTableLoweringStats lower_hip_constant_switch_tables(
    llvm::Module &module) noexcept {
    using namespace llvm;
    HIPSwitchTableLoweringStats stats;
    for (auto &function : module) {
        if (function.isDeclaration()) { continue; }
        SmallVector<SwitchInst *, 16u> switches;
        for (auto &block : function) {
            for (auto &instruction : block) {
                if (auto *switch_inst = dyn_cast<SwitchInst>(&instruction)) {
                    switches.emplace_back(switch_inst);
                }
            }
        }
        for (auto *switch_inst : switches) {
            auto *head = switch_inst->getParent();
            SmallVector<BasicBlock *, 32u> arms;
            auto *merge = unique_successor(switch_inst, arms);
            if (merge == nullptr || merge == head) { continue; }

            auto forwarding = true;
            for (auto *arm : arms) {
                if (arm == merge) { continue; }
                auto *terminator = arm->getTerminator();
                if (arm->size() != 1u || !isa<BranchInst>(terminator) ||
                    !cast<BranchInst>(terminator)->isUnconditional() ||
                    terminator->getSuccessor(0u) != merge ||
                    !all_predecessors_are(arm, head)) {
                    forwarding = false;
                    break;
                }
            }
            if (!forwarding) { continue; }

            SmallVector<PHINode *, 8u> phis;
            auto valid_phis = true;
            for (auto &instruction : *merge) {
                auto *phi = dyn_cast<PHINode>(&instruction);
                if (phi == nullptr) { break; }
                if (phi->getNumIncomingValues() != arms.size() ||
                    !phi->getType()->isIntegerTy()) {
                    valid_phis = false;
                    break;
                }
                for (auto *arm : arms) {
                    auto *incoming = arm == merge ? head : arm;
                    auto index = phi->getBasicBlockIndex(incoming);
                    if (index < 0 ||
                        !isa<ConstantInt>(phi->getIncomingValue(index))) {
                        valid_phis = false;
                        break;
                    }
                }
                if (!valid_phis) { break; }
                phis.emplace_back(phi);
            }
            // Keep the first implementation deliberately narrow: one scalar
            // payload is enough for the SVM cursor offset and avoids creating
            // partially committed tables for heterogeneous merge PHIs.
            if (!valid_phis || phis.size() != 1u) { continue; }

            auto selector_type = switch_inst->getCondition()->getType();
            auto selector_bits = selector_type->getIntegerBitWidth();
            uint64_t maximum_case = 0u;
            for (auto &case_value : switch_inst->cases()) {
                maximum_case = std::max(
                    maximum_case, case_value.getCaseValue()->getZExtValue());
            }
            // The table is intended for compact dispatch metadata. A bounded
            // 32-bit selector still handles sparse SVM IDs while rejecting
            // arbitrary user switches that would inflate constant storage.
            if (selector_bits == 0u || selector_bits > 32u ||
                maximum_case > 255u ||
                maximum_case + 1u >
                    4u * static_cast<uint64_t>(switch_inst->getNumCases()) + 1u) {
                continue;
            }
            // The range check and the out-of-range sentinel are materialized
            // in the selector's integer type below.  Reject a full-width
            // maximum for narrow selectors so max+1 cannot truncate to zero
            // (for example, an i8 switch containing case 255).
            if (selector_bits < 32u &&
                maximum_case >= ((uint64_t{1u} << selector_bits) - 1u)) {
                continue;
            }

            auto *default_destination = switch_inst->getDefaultDest();
            auto *default_incoming =
                default_destination == merge ? head : default_destination;
            auto *first_phi = phis.front();
            auto default_index = first_phi->getBasicBlockIndex(default_incoming);
            if (default_index < 0) { continue; }

            auto table_size = static_cast<unsigned>(maximum_case + 2u);
            auto *element_type = first_phi->getType();
            SmallVector<Constant *, 256u> values;
            values.resize(table_size,
                          cast<Constant>(first_phi->getIncomingValue(default_index)));
            for (auto &case_value : switch_inst->cases()) {
                auto *destination = case_value.getCaseSuccessor();
                auto *incoming = destination == merge ? head : destination;
                auto index = first_phi->getBasicBlockIndex(incoming);
                if (index < 0) {
                    values.clear();
                    break;
                }
                values[case_value.getCaseValue()->getZExtValue()] =
                    cast<Constant>(first_phi->getIncomingValue(index));
            }
            if (values.empty()) { continue; }
            values[maximum_case + 1u] =
                cast<Constant>(first_phi->getIncomingValue(default_index));

            IRBuilder<> builder{switch_inst};
            auto *zero = ConstantInt::get(selector_type, 0u);
            auto *limit = ConstantInt::get(selector_type, maximum_case + 1u);
            auto *in_range = builder.CreateICmpULT(
                switch_inst->getCondition(), limit, "switch.table.inrange");
            auto *index = builder.CreateSelect(
                in_range, switch_inst->getCondition(),
                ConstantInt::get(selector_type, maximum_case + 1u),
                "switch.table.index");

            auto *array_type = ArrayType::get(element_type, table_size);
            auto table_name =
                (Twine{"luisa.switch.table."} + Twine{stats.rewritten_switch_count})
                    .str();
            auto *table = new GlobalVariable(
                module, array_type, true, GlobalValue::PrivateLinkage,
                ConstantArray::get(array_type, values), table_name, nullptr,
                GlobalValue::NotThreadLocal, 4u);
            auto *pointer = builder.CreateInBoundsGEP(
                array_type, table, {zero, index}, "switch.table.ptr");

            auto *load = builder.CreateLoad(element_type, pointer,
                                            "switch.table.value");
            first_phi->replaceAllUsesWith(load);
            first_phi->eraseFromParent();
            stats.rewritten_phi_count++;
            switch_inst->eraseFromParent();
            BranchInst::Create(merge, head);
            for (auto *arm : arms) {
                if (arm != merge) { merge->removePredecessor(arm, false); }
            }
            stats.rewritten_switch_count++;
        }
    }
    return stats;
}

}// namespace luisa::compute::hip
