//
// LLVM adapter for generated-callable inlining decisions.
//

#include "hip_callable_inline_graph.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::hip {

GeneratedCallableInlineGraph
build_generated_callable_inline_graph(
    llvm::Module &module,
    const char *generated_attribute) noexcept {
    auto graph = GeneratedCallableInlineGraph{};
    auto indices = llvm::DenseMap<const llvm::Function *, size_t>{};
    for (auto &function : module) {
        if (!function.isDeclaration() &&
            function.hasFnAttribute(generated_attribute)) {
            auto index = graph.functions.size();
            graph.functions.emplace_back(&function);
            indices.try_emplace(&function, index);
        }
    }
    graph.nodes.resize(graph.functions.size());
    for (auto node_index = size_t{0u};
         node_index < graph.functions.size(); node_index++) {
        auto *function = graph.functions[node_index];
        auto &node = graph.nodes[node_index];
        node.instruction_count = function->getInstructionCount();
        auto call_site_indices =
            llvm::DenseMap<const llvm::CallBase *, size_t>{};
        for (auto &block : *function) {
            for (auto &instruction : block) {
                auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
                if (call == nullptr) { continue; }
                auto *callee = llvm::dyn_cast<llvm::Function>(
                    call->getCalledOperand()->stripPointerCasts());
                if (callee == nullptr) { continue; }
                auto iter = indices.find(callee);
                if (iter == indices.end()) { continue; }
                auto call_site = node.callees.size();
                node.callees.emplace_back(iter->second);
                call_site_indices.try_emplace(call, call_site);
            }
        }
        if (node.callees.size() < 2u) { continue; }
        auto post_dominator_tree = llvm::PostDominatorTree{*function};
        for (auto &switch_block : *function) {
            auto *switch_instruction =
                llvm::dyn_cast<llvm::SwitchInst>(
                    switch_block.getTerminator());
            if (switch_instruction == nullptr ||
                switch_instruction->getNumSuccessors() < 2u) {
                continue;
            }

            // A generated polymorphic dispatch normally sends its default to
            // `unreachable`. Such an error path has no common post-dominator
            // with valid cases, so only explicit cases define the alternative
            // region when at least two of them exist.
            const auto default_is_unreachable =
                llvm::isa<llvm::UnreachableInst>(
                    switch_instruction->getDefaultDest()->getTerminator());
            auto starts = llvm::SmallVector<llvm::BasicBlock *, 16u>{};
            if (switch_instruction->getNumCases() >= 2u &&
                default_is_unreachable) {
                for (auto &switch_case : switch_instruction->cases()) {
                    starts.emplace_back(switch_case.getCaseSuccessor());
                }
            } else {
                for (auto successor_index = 0u;
                     successor_index <
                     switch_instruction->getNumSuccessors();
                     successor_index++) {
                    starts.emplace_back(
                        switch_instruction->getSuccessor(successor_index));
                }
            }
            auto *merge = starts.front();
            for (auto start_index = 1u;
                 merge != nullptr && start_index < starts.size();
                 start_index++) {
                merge = post_dominator_tree.findNearestCommonDominator(
                    merge, starts[start_index]);
            }
            if (merge == nullptr) { continue; }

            auto alternative_sites = std::vector<size_t>{};
            auto seen_sites = llvm::SmallDenseSet<size_t, 16u>{};
            auto invalid_group = false;
            for (auto successor_index = 0u;
                 successor_index < switch_instruction->getNumSuccessors();
                 successor_index++) {
                auto *successor = switch_instruction->getSuccessor(
                    successor_index);
                auto successor_sites =
                    llvm::SmallDenseSet<size_t, 4u>{};
                auto worklist =
                    llvm::SmallVector<llvm::BasicBlock *, 16u>{successor};
                auto visited =
                    llvm::SmallDenseSet<llvm::BasicBlock *, 16u>{};
                while (!worklist.empty()) {
                    auto *block = worklist.pop_back_val();
                    if (block == merge || !visited.insert(block).second) {
                        continue;
                    }
                    for (auto &instruction : *block) {
                        auto *call = llvm::dyn_cast<llvm::CallBase>(
                            &instruction);
                        if (call == nullptr) { continue; }
                        if (auto iter = call_site_indices.find(call);
                            iter != call_site_indices.end()) {
                            successor_sites.insert(iter->second);
                        }
                    }
                    for (auto *next : llvm::successors(block)) {
                        worklist.emplace_back(next);
                    }
                }
                // Exactly one direct generated call is a proven outline
                // frontier for this successor. Multiple calls have no unique
                // boundary and are deliberately left to inner dispatchers or
                // LLVM's normal inliner. Duplicate call sites arise when CFG
                // successors share a region and are counted only once.
                if (successor_sites.size() == 1u) {
                    auto call_site = *successor_sites.begin();
                    if (seen_sites.insert(call_site).second) {
                        alternative_sites.emplace_back(call_site);
                    }
                } else if (successor_sites.size() > 1u) {
                    invalid_group = true;
                    break;
                }
            }
            if (!invalid_group && alternative_sites.size() >= 2u) {
                node.alternative_call_groups.emplace_back(
                    std::move(alternative_sites));
            }
        }
    }
    return graph;
}

}// namespace luisa::compute::hip
