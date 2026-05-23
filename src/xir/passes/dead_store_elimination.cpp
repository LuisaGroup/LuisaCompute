#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/dead_store_elimination.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

// Eliminate stores that are overwritten before any load in the same basic block.
// Also eliminate stores whose value has no users (truly dead stores).
static void eliminate_dead_stores_in_block(BasicBlock *block, DeadStoreEliminationInfo &info) noexcept {

    // Track the last store to each pointer (must be exact same pointer for aliasing safety).
    luisa::unordered_map<Value *, StoreInst *> last_store;
    // Track which stores are dead (overwritten before load)
    luisa::vector<StoreInst *> dead_stores;

    // Erase all tracked stores whose pointer traces to the given base alloca.
    // Used when a load or call may access memory through an aliasing pointer.
    auto erase_stores_with_base_alloca = [&](AllocaInst *base) noexcept {
        for (auto it = last_store.begin(); it != last_store.end();) {
            if (trace_pointer_base_local_alloca_inst(it->first) == base) {
                it = last_store.erase(it);
            } else {
                ++it;
            }
        }
    };

    for (auto inst : block->instructions()) {
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::STORE: {
                auto store = static_cast<StoreInst *>(inst);
                auto ptr = store->variable();
                // Only track stores to local allocas (direct or via GEP)
                if (!trace_pointer_base_local_alloca_inst(ptr)) break;

                auto it = last_store.find(ptr);
                if (it != last_store.end()) {
                    dead_stores.push_back(it->second);
                }
                last_store[ptr] = store;
                break;
            }
            case DerivedInstructionTag::LOAD: {
                auto load = static_cast<LoadInst *>(inst);
                // A load from any pointer to the same alloca invalidates all
                // tracked stores to that alloca.
                if (auto base = trace_pointer_base_local_alloca_inst(load->variable())) {
                    erase_stores_with_base_alloca(base);
                }
                break;
            }
            case DerivedInstructionTag::GEP: {
                break;
            }
            default: {
                for (auto op_use : inst->operand_uses()) {
                    // Any non-load/non-store use of a pointer invalidates all
                    // tracked stores to the same alloca.
                    if (auto base = trace_pointer_base_local_alloca_inst(op_use->value())) {
                        erase_stores_with_base_alloca(base);
                    }
                }
                break;
            }
        }
    }

    // Actually remove the dead stores
    for (auto store : dead_stores) {
        store->remove_self();
        info.eliminated_store_count++;
    }
}

static void run_dead_store_elimination_on_function(Function *function, DeadStoreEliminationInfo &info) noexcept {
    if (auto def = function->definition()) {
        def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            eliminate_dead_stores_in_block(block, info);
        });
    }
}

}// namespace detail

DeadStoreEliminationInfo dead_store_elimination_pass_run_on_function(Function *function) noexcept {
    DeadStoreEliminationInfo info;
    detail::run_dead_store_elimination_on_function(function, info);
    return info;
}

DeadStoreEliminationInfo dead_store_elimination_pass_run_on_module(Module *module) noexcept {
    DeadStoreEliminationInfo info;
    for (auto f : module->function_list()) {
        detail::run_dead_store_elimination_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
