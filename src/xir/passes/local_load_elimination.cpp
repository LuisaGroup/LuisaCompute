#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>

#include "helpers.h"
#include <luisa/xir/passes/local_load_elimination.h>

namespace luisa::compute::xir {

namespace detail {

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
                    } else if (auto alloca_inst = trace_pointer_base_local_alloca_inst(load->variable())) {
                        variable_pointers[alloca_inst].emplace_back(load->variable());
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

static void run_local_load_elimination_on_function(Function *function, LocalLoadEliminationInfo &info) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<BasicBlock *> visited;
        definition->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
            run_local_load_elimination_on_basic_block(visited, block, info);
        });
    }
}

}// namespace detail

LocalLoadEliminationInfo local_load_elimination_pass_run_on_function(Function *function) noexcept {
    LocalLoadEliminationInfo info;
    detail::run_local_load_elimination_on_function(function, info);
    return info;
}

LocalLoadEliminationInfo local_load_elimination_pass_run_on_module(Module *module) noexcept {
    LocalLoadEliminationInfo info;
    for (auto f : module->function_list()) {
        detail::run_local_load_elimination_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
