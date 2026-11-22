#include <utility>

#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

using LastStoreMap = luisa::unordered_map<AllocaInst *, luisa::vector<std::pair<Value *, StoreInst *>>>;

// Erase all tracked stores whose pointer traces to the given base alloca.
static void erase_stores_with_base_alloca(
    LastStoreMap &last_store,
    AllocaInst *base) noexcept {
    last_store.erase(base);
}

// Process a single block for dead stores. Mutates last_store in place.
// Returns dead stores found in this block alone.
static luisa::vector<StoreInst *> process_block_dse(
    BasicBlock *block,
    LastStoreMap &last_store) noexcept {

    luisa::vector<StoreInst *> dead_stores;

    for (auto inst : block->instructions()) {
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::STORE: {
                auto store = static_cast<StoreInst *>(inst);
                auto ptr = store->variable();
                auto *base = trace_pointer_base_local_alloca_inst(ptr);
                if (!base) break;

                auto &entries = last_store[base];
                StoreInst *prev_store = nullptr;
                auto updated = false;
                for (auto &[tracked_ptr, tracked_store] : entries) {
                    if (tracked_ptr == ptr) {
                        prev_store = tracked_store;
                        tracked_store = store;
                        updated = true;
                        break;
                    }
                }
                // The newer store proves the previous value dead, but it is
                // not a unique replacement owner for instruction-local
                // metadata: several annotated stores may be overwritten by
                // the same write, and merging their metadata can itself be
                // verifier-invalid. Retain annotated stores conservatively.
                if (prev_store != nullptr &&
                    prev_store->metadata_list().empty()) {
                    dead_stores.push_back(prev_store);
                }
                if (!updated) {
                    entries.emplace_back(ptr, store);
                }
                break;
            }
            case DerivedInstructionTag::LOAD: {
                auto load = static_cast<LoadInst *>(inst);
                if (auto base = trace_pointer_base_local_alloca_inst(load->variable())) {
                    erase_stores_with_base_alloca(last_store, base);
                }
                break;
            }
            case DerivedInstructionTag::GEP: {
                break;
            }
            default: {
                for (auto op_use : inst->operand_uses()) {
                    if (auto base = trace_pointer_base_local_alloca_inst(op_use->value())) {
                        erase_stores_with_base_alloca(last_store, base);
                    }
                }
                break;
            }
        }
    }

    return dead_stores;
}

// Run intra-block DSE on each block independently.
static size_t run_intra_block_dse(FunctionDefinition *function) noexcept {
    size_t count = 0u;
    function->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        LastStoreMap last_store;
        auto dead = process_block_dse(block, last_store);
        for (auto store : dead) {
            store->remove_self();
            count++;
        }
    });
    return count;
}

// Run cross-block DSE along straight-line chains (single-successor, single-predecessor).
// Carries the last_store map forward across blocks in a chain.
static size_t run_straight_line_dse(FunctionDefinition *function) noexcept {
    size_t count = 0u;
    luisa::unordered_set<BasicBlock *> visited;
    function->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
        if (visited.contains(block)) return;
        LastStoreMap last_store;
        luisa::vector<StoreInst *> dead_stores;
        BasicBlock *current = block;
        while (visited.emplace(current).second) {
            auto block_dead = process_block_dse(current, last_store);
            dead_stores.insert(dead_stores.end(), block_dead.begin(), block_dead.end());
            BasicBlock *next = nullptr;
            size_t successor_count = 0;
            current->traverse_successors(true, [&](BasicBlock *succ) noexcept {
                successor_count++;
                next = succ;
            });
            if (successor_count != 1) break;
            size_t pred_count = 0;
            next->traverse_predecessors(false, [&](BasicBlock *) noexcept { pred_count++; });
            if (pred_count != 1) break;
            current = next;
        }
        for (auto store : dead_stores) {
            store->remove_self();
            count++;
        }
    });
    return count;
}

// Run dead store elimination on a function.
static void run_dead_store_elimination_on_function(Function *function,
                                                   DeadStoreEliminationInfo &info) noexcept {
    if (function == nullptr) { return; }
    if (auto def = function->definition()) {
        if (def->body_block() == nullptr) { return; }
        info.eliminated_store_count += run_intra_block_dse(def);
        info.eliminated_store_count += run_straight_line_dse(def);
    }
}

}// namespace detail

DeadStoreEliminationInfo dead_store_elimination_pass_run_on_function(Function *function) noexcept {
    DeadStoreEliminationInfo info;
    detail::run_dead_store_elimination_on_function(function, info);
    return info;
}

DeadStoreEliminationInfo dead_store_elimination_pass_run_on_module(Module *module, PassReport *report) noexcept {
    DeadStoreEliminationInfo info;
    if (module == nullptr) {
        if (report != nullptr) { report->set("eliminated_store", 0u); }
        return info;
    }
    for (auto f : module->function_list()) {
        detail::run_dead_store_elimination_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("eliminated_store", info.eliminated_store_count);
    }
    return info;
}

}// namespace luisa::compute::xir
