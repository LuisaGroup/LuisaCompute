#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>

#include "helpers.h"
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

// Straight-line load elimination: within a chain of single-successor/single-predecessor
// blocks, eliminate a load if an earlier load from the same pointer already dominates.
static void run_local_load_elimination_on_basic_block(luisa::unordered_set<BasicBlock *> &visited,
                                                      BasicBlock *block,
                                                      LocalLoadEliminationInfo &info) noexcept {

    luisa::unordered_map<AllocaInst *, luisa::vector<Value *>> variable_pointers;// maps variables to pointers
    luisa::unordered_map<Value *, LoadInst *> already_loaded;                    // maps pointers to the earliest load instructions
    luisa::unordered_map<LoadInst *, LoadInst *> removable_loads;                // maps loads to the load that can be forwarded

    auto invalidate_interfering_loads = [&](Value *ptr) noexcept -> AllocaInst * {
        if (auto alloca_inst = trace_pointer_base_local_alloca_inst(ptr)) {
            auto &interfering_ptrs = variable_pointers[alloca_inst];
            interfering_ptrs.emplace_back(ptr);
            for (auto interfering_ptr : interfering_ptrs) {
                already_loaded.erase(interfering_ptr);
            }
            return alloca_inst;
        }
        // Also invalidate for non-alloca pointers (e.g., function arguments),
        // since stores through GEPs of reference args must invalidate
        // previously cached loads from the same base pointer.
        already_loaded.erase(ptr);
        if (auto base = trace_pointer_base_value(ptr); base != ptr) {
            already_loaded.erase(base);
        }
        return nullptr;
    };

    // we visit the block and all of its single straight-line successors to find the earliest loads
    while (visited.emplace(block).second) {

        // process the instructions in the block
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::LOAD: {
                    auto load = static_cast<LoadInst *>(inst);
                    if (auto iter = already_loaded.find(load->variable()); iter != already_loaded.end()) {
                        removable_loads.emplace(load, iter->second);
                    } else {
                        if (auto alloca_inst = trace_pointer_base_local_alloca_inst(load->variable())) {
                            variable_pointers[alloca_inst].emplace_back(load->variable());
                        }
                        already_loaded[load->variable()] = load;
                    }
                    break;
                }
                case DerivedInstructionTag::GEP: {
                    // users of GEPs will handle the forwarding, so we don't need to do anything here
                    break;
                }
                default: {
                    for (auto op_use : inst->operand_uses()) {
                        invalidate_interfering_loads(op_use->value());
                    }
                    break;
                }
            }
        }

        // move to the next block if it is the only successor and only has a single predecessor
        BasicBlock *next = nullptr;
        size_t successor_count = 0;
        block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            successor_count++;
            next = succ;
        });
        if (successor_count != 1) { break; }
        // check if the next block has a single predecessor
        size_t pred_count = 0;
        next->traverse_predecessors(false, [&](BasicBlock *) noexcept { pred_count++; });
        if (pred_count != 1) { break; }
        block = next;
    }

    // process the instructions
    for (auto [current_load, earlier_load] : removable_loads) {
        current_load->replace_all_uses_with(earlier_load);
        current_load->remove_self();
        info.removed_load_count++;
    }
}

// Dominator-based cross-block load elimination.
// Propagates reaching loads through the CFG; at merge points, a reaching load
// is kept only if it agrees across all predecessors.
static void run_dominator_load_elimination_on_function(FunctionDefinition *function, LocalLoadEliminationInfo &info) noexcept {
    luisa::vector<BasicBlock *> rpo;
    function->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
        rpo.push_back(block);
    });
    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> predecessors;
    for (auto block : rpo) {
        block->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
            predecessors[block].push_back(pred);
        });
    }
    using ReachingMap = luisa::unordered_map<Value *, LoadInst *>;
    luisa::unordered_map<BasicBlock *, ReachingMap> reaching_load;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto block : rpo) {
            ReachingMap in;
            auto &preds = predecessors[block];
            if (!preds.empty()) {
                in = reaching_load[preds.front()];
                for (size_t i = 1; i < preds.size(); ++i) {
                    auto &pred_map = reaching_load[preds[i]];
                    for (auto it = in.begin(); it != in.end();) {
                        auto jt = pred_map.find(it->first);
                        if (jt == pred_map.end() || jt->second != it->second) {
                            it = in.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }
            auto &block_reaching = reaching_load[block];
            auto current = in;
            luisa::vector<std::pair<LoadInst *, LoadInst *>> to_remove;
            for (auto inst : block->instructions()) {
                if (inst->isa<LoadInst>()) {
                    auto load = static_cast<LoadInst *>(inst);
                    auto ptr = load->variable();
                    auto it = current.find(ptr);
                    if (it != current.end() && it->second != load) {
                        to_remove.emplace_back(load, it->second);
                    } else {
                        current[ptr] = load;
                    }
                } else if (inst->isa<StoreInst>()) {
                    auto store = static_cast<StoreInst *>(inst);
                    auto ptr = store->variable();
                    auto base = trace_pointer_base_local_alloca_inst(ptr);
                    if (base) {
                        for (auto it = current.begin(); it != current.end();) {
                            if (trace_pointer_base_local_alloca_inst(it->first) == base) {
                                it = current.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    } else {
                        current.erase(ptr);
                        auto value_base = trace_pointer_base_value(ptr);
                        if (value_base != ptr) { current.erase(value_base); }
                    }
                } else {
                    for (auto op_use : inst->operand_uses()) {
                        auto val = op_use->value();
                        auto base = trace_pointer_base_local_alloca_inst(val);
                        if (base) {
                            for (auto it = current.begin(); it != current.end();) {
                                if (trace_pointer_base_local_alloca_inst(it->first) == base) {
                                    it = current.erase(it);
                                } else {
                                    ++it;
                                }
                            }
                        } else {
                            current.erase(val);
                            auto value_base = trace_pointer_base_value(val);
                            if (value_base != val) { current.erase(value_base); }
                        }
                    }
                }
            }
            if (block_reaching != current) {
                block_reaching = std::move(current);
                changed = true;
            }
            for (auto [load, earlier] : to_remove) {
                load->replace_all_uses_with(earlier);
                load->remove_self();
                info.removed_load_count++;
            }
        }
    }
}

static void run_local_load_elimination_on_function(Function *function, LocalLoadEliminationInfo &info) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<BasicBlock *> visited;
        definition->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
            run_local_load_elimination_on_basic_block(visited, block, info);
        });
        run_dominator_load_elimination_on_function(definition, info);
    }
}

}// namespace detail

LocalLoadEliminationInfo local_load_elimination_pass_run_on_function(Function *function) noexcept {
    LocalLoadEliminationInfo info;
    detail::run_local_load_elimination_on_function(function, info);
    return info;
}

LocalLoadEliminationInfo local_load_elimination_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LocalLoadEliminationInfo info;
    for (auto f : module->function_list()) {
        detail::run_local_load_elimination_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("removed_load", info.removed_load_count);
    }
    return info;
}

}// namespace luisa::compute::xir
