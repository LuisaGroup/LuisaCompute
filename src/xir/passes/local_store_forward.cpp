#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

// TODO: we only handle local alloca's in straight-line code for now
static void run_local_store_forward_on_basic_block(luisa::unordered_set<BasicBlock *> &visited,
                                                   BasicBlock *block,
                                                   LocalStoreForwardInfo &info) noexcept {

    luisa::unordered_map<AllocaInst *, luisa::vector<Value *>> variable_pointers;// maps variables to pointers
    luisa::unordered_map<Value *, StoreInst *> latest_stores;                    // maps pointers to the latest store instruction
    luisa::unordered_map<LoadInst *, StoreInst *> removable_loads;               // maps loads to the store that can be forwarded

    auto invalidate_interfering_stores = [&](Value *ptr) noexcept -> AllocaInst * {
        if (auto alloca_inst = trace_pointer_base_local_alloca_inst(ptr)) {
            auto &interfering_ptrs = variable_pointers[alloca_inst];
            interfering_ptrs.emplace_back(ptr);
            for (auto interfering_ptr : interfering_ptrs) {
                latest_stores.erase(interfering_ptr);
            }
            return alloca_inst;
        }
        // Also invalidate for non-alloca pointers (e.g., function arguments)
        latest_stores.erase(ptr);
        if (auto base = trace_pointer_base_value(ptr); base != ptr) {
            latest_stores.erase(base);
        }
        return nullptr;
    };

    // we visit the block and all of its single straight-line successors
    while (visited.emplace(block).second) {

        // process the instructions in the block
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::LOAD: {
                    auto load = static_cast<LoadInst *>(inst);
                    if (auto iter = latest_stores.find(load->variable()); iter != latest_stores.end()) {
                        removable_loads.emplace(load, iter->second);
                    }
                    break;
                }
                case DerivedInstructionTag::STORE: {
                    auto store = static_cast<StoreInst *>(inst);
                    // if this is a store to (part of) a local alloca, we might be able to forward it
                    if (auto pointer = store->variable(); invalidate_interfering_stores(pointer)) {
                        latest_stores[pointer] = store;
                    }
                    break;
                }
                case DerivedInstructionTag::GEP: {
                    // users of GEPs will handle the forwarding, so we don't need to do anything here
                    break;
                }
                default: {// for other instructions, we invalidate possibly interfering stores
                    for (auto op_use : inst->operand_uses()) {
                        invalidate_interfering_stores(op_use->value());
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

    // perform the forwarding
    for (auto [load, store] : removable_loads) {
        load->replace_all_uses_with(store->value());
        load->remove_self();
        info.removed_load_count++;
    }
}

// forward stores to loads within straight-line code
static void forward_straight_line_stores_to_loads_on_function(FunctionDefinition *function, LocalStoreForwardInfo &info) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    function->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
        run_local_store_forward_on_basic_block(visited, block, info);
    });
}

// find and remove all loads from local variables that only have a single (or no) store
static void forward_single_store_to_loads_on_function(FunctionDefinition *function, LocalStoreForwardInfo &info) noexcept {

    luisa::unordered_map<AllocaInst *, StoreInst *> single_store;
    luisa::unordered_map<Instruction *, size_t> inst_indices;
    // search for local variables that only have a single store
    {
        luisa::unordered_map<AllocaInst *, size_t> store_count;
        function->traverse_instructions([&](Instruction *inst) noexcept {
            inst_indices.emplace(inst, inst_indices.size());
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::LOAD: [[fallthrough]];
                case DerivedInstructionTag::GEP: break;
                default: {
                    for (auto op_use : inst->operand_uses()) {
                        if (auto base_alloca = trace_pointer_base_local_alloca_inst(op_use->value())) {
                            store_count.try_emplace(base_alloca, 0).first->second++;
                        }
                    }
                    break;
                }
            }
        });
        for (auto [alloca_inst, count] : store_count) {
            if (count == 1) {
                for (auto &&use : alloca_inst->use_list()) {
                    if (auto user = use->user(); user->isa<StoreInst>()) {
                        auto store_inst = static_cast<StoreInst *>(user);
                        if (store_inst->variable() == alloca_inst) {// only consider stores to the entire alloca
                            single_store.emplace(alloca_inst, store_inst);
                        }
                        break;
                    }
                }
            }
        }
    }
    // early return if no chances
    if (single_store.empty()) { return; }

    // collect the loads that might be eliminated
    luisa::vector<LoadInst *> removable_loads;
    {
        // create a dom tree to check if the use is dominated by the def
        auto dom_tree = compute_dom_tree(function);
        auto dominates = [&](StoreInst *store, LoadInst *load) noexcept {
            auto store_block = store->parent_block();
            auto load_block = load->parent_block();
            if (!dom_tree.contains(store_block) || !dom_tree.contains(load_block)) {
                return false;
            }
            return store_block == load_block ?
                       inst_indices.at(store) < inst_indices.at(load) :
                       dom_tree.dominates(store_block, load_block);
        };
        // check if replacing 'load' with 'store->value()' would create a cycle
        auto store_value_depends_on_load = [&](StoreInst *store, LoadInst *load) noexcept {
            luisa::vector<Value *> worklist;
            luisa::unordered_set<Value *> visited;
            worklist.push_back(store->value());
            while (!worklist.empty()) {
                auto v = worklist.back();
                worklist.pop_back();
                if (!visited.emplace(v).second) { continue; }
                if (v == load) { return true; }
                if (v->isa<Instruction>()) {
                    auto inst = static_cast<Instruction *>(v);
                    for (auto &&op_use : inst->operand_uses()) {
                        worklist.push_back(op_use->value());
                    }
                }
            }
            return false;
        };
        // Composite insert stores depend on the current aggregate value;
        // forwarding them to loads creates self-referencing inserts.
        auto is_composite_insert_store = [&](StoreInst *store) noexcept -> bool {
            if (auto v = store->value(); v->isa<ArithmeticInst>()) {
                auto arith = static_cast<ArithmeticInst *>(v);
                return arith->op() == ArithmeticOp::INSERT;
            }
            return false;
        };
        function->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<LoadInst>()) {
                auto load = static_cast<LoadInst *>(inst);
                if (auto base_alloca = trace_pointer_base_local_alloca_inst(load->variable())) {
                    auto iter = single_store.find(base_alloca);
                    if (iter != single_store.end() && dominates(iter->second, load) &&
                        !store_value_depends_on_load(iter->second, load) &&
                        !is_composite_insert_store(iter->second)) {
                        removable_loads.emplace_back(load);
                    }
                }
            }
        });
    }

    // do the elimination
    for (auto load : removable_loads) {
        // convert load to extract
        luisa::fixed_vector<Value *, 8> extract_args;
        LUISA_DEBUG_ASSERT(load->variable()->isa<Instruction>(), "Load variable must be an instruction.");
        auto pointer = static_cast<Instruction *>(load->variable());
        for (;;) {
            if (pointer->isa<AllocaInst>()) { break; }
            if (pointer->isa<GEPInst>()) {
                auto gep = static_cast<GEPInst *>(pointer);
                LUISA_DEBUG_ASSERT(gep->base()->isa<Instruction>(), "GEP base must be an instruction.");
                auto sub_indices = gep->index_uses();
                // note: we emplace the indices in reverse order to avoid
                // expensive insertions at the beginning of the vector
                for (auto iter = sub_indices.rbegin(); iter != sub_indices.rend(); ++iter) {
                    extract_args.emplace_back((*iter)->value());
                }
                pointer = static_cast<Instruction *>(gep->base());
            } else {
                LUISA_ERROR_WITH_LOCATION("Unexpected instruction type.");
            }
        }
        // process the alloca pointer
        LUISA_DEBUG_ASSERT(pointer->isa<AllocaInst>(), "Pointer must be an alloca.");
        auto store = single_store[static_cast<AllocaInst *>(pointer)];
        LUISA_DEBUG_ASSERT(store != nullptr, "Store must not be null.");
        extract_args.emplace_back(store->value());
        auto value = [&]() noexcept -> Value * {
            // simple case: scalar load
            if (extract_args.size() == 1) { return extract_args.front(); }
            // reverse the indices to the correct order
            std::reverse(extract_args.begin(), extract_args.end());
            // create the extract instruction
            XIRBuilder builder;
            builder.set_insertion_point(load);
            return builder.call(load->type(), ArithmeticOp::EXTRACT, extract_args);
        }();
        load->replace_all_uses_with(value);
        load->remove_self();
        // record the elimination
        info.removed_load_count++;
    }
}

[[nodiscard]] static bool is_alloca_used_only_by_load_store_gep(Value *ptr) noexcept {
    for (auto &&use : ptr->use_list()) {
        if (auto user = use->user(); user != nullptr) {
            if (user->isa<LoadInst>() || user->isa<StoreInst>()) { continue; }
            if (user->isa<GEPInst>()) {
                if (!is_alloca_used_only_by_load_store_gep(static_cast<GEPInst *>(user))) { return false; }
                continue;
            }
            return false;
        }
    }
    return true;
}

// forward stores to loads from local variables where all stores store the same value
static void forward_uniform_store_to_loads_on_function(FunctionDefinition *function, LocalStoreForwardInfo &info) noexcept {
    // find allocas where all direct stores store the same value
    luisa::unordered_map<AllocaInst *, luisa::vector<StoreInst *>> alloca_stores;
    luisa::unordered_set<AllocaInst *> partially_stored_allocas;
    function->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<StoreInst>()) {
            auto store = static_cast<StoreInst *>(inst);
            if (auto base = trace_pointer_base_local_alloca_inst(store->variable())) {
                if (store->variable() == base) {
                    alloca_stores[base].push_back(store);
                } else {
                    // This includes arbitrarily nested GEP chains. Looking at
                    // only direct GEP users misses stores through gep(gep(A)).
                    partially_stored_allocas.emplace(base);
                }
            }
        }
    });
    luisa::unordered_map<AllocaInst *, Value *> uniform_value;
    for (auto &[alloca_inst, stores] : alloca_stores) {
        if (stores.empty()) continue;
        if (partially_stored_allocas.contains(alloca_inst)) continue;
        Value *common = stores.front()->value();
        bool all_same = true;
        for (size_t i = 1; i < stores.size(); ++i) {
            if (stores[i]->value() != common) {
                all_same = false;
                break;
            }
        }
        if (all_same && is_alloca_used_only_by_load_store_gep(alloca_inst)) {
            uniform_value.emplace(alloca_inst, common);
        }
    }
    if (uniform_value.empty()) return;
    // build dom tree and instruction indices for dominance checks
    auto dom_tree = compute_dom_tree(function);
    luisa::unordered_map<Instruction *, size_t> inst_indices;
    function->traverse_instructions([&](Instruction *inst) noexcept {
        inst_indices.emplace(inst, inst_indices.size());
    });
    auto dominates = [&](StoreInst *store, LoadInst *load) noexcept {
        auto store_block = store->parent_block();
        auto load_block = load->parent_block();
        if (!dom_tree.contains(store_block) || !dom_tree.contains(load_block)) {
            return false;
        }
        if (store_block == load_block) {
            return inst_indices.at(store) < inst_indices.at(load);
        }
        return dom_tree.dominates(store_block, load_block);
    };
    luisa::vector<LoadInst *> removable_loads;
    function->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<LoadInst>()) {
            auto load = static_cast<LoadInst *>(inst);
            if (auto base = trace_pointer_base_local_alloca_inst(load->variable())) {
                auto it = uniform_value.find(base);
                if (it != uniform_value.end()) {
                    auto &stores = alloca_stores[base];
                    bool any_dominates = false;
                    for (auto store : stores) {
                        if (dominates(store, load)) {
                            any_dominates = true;
                            break;
                        }
                    }
                    if (any_dominates) {
                        removable_loads.push_back(load);
                    }
                }
            }
        }
    });
    // replace loads with the uniform value (handling GEPs like single_store pass)
    for (auto load : removable_loads) {
        luisa::fixed_vector<Value *, 8> extract_args;
        LUISA_DEBUG_ASSERT(load->variable()->isa<Instruction>(), "Load variable must be an instruction.");
        auto pointer = static_cast<Instruction *>(load->variable());
        for (;;) {
            if (pointer->isa<AllocaInst>()) { break; }
            if (pointer->isa<GEPInst>()) {
                auto gep = static_cast<GEPInst *>(pointer);
                LUISA_DEBUG_ASSERT(gep->base()->isa<Instruction>(), "GEP base must be an instruction.");
                auto sub_indices = gep->index_uses();
                for (auto iter = sub_indices.rbegin(); iter != sub_indices.rend(); ++iter) {
                    extract_args.emplace_back((*iter)->value());
                }
                pointer = static_cast<Instruction *>(gep->base());
            } else {
                LUISA_ERROR_WITH_LOCATION("Unexpected instruction type.");
            }
        }
        LUISA_DEBUG_ASSERT(pointer->isa<AllocaInst>(), "Pointer must be an alloca.");
        auto it = uniform_value.find(static_cast<AllocaInst *>(pointer));
        LUISA_DEBUG_ASSERT(it != uniform_value.end(), "Uniform value must exist.");
        auto value = it->second;
        LUISA_DEBUG_ASSERT(value != nullptr, "Uniform value must not be null.");
        extract_args.emplace_back(value);
        auto replacement = [&]() noexcept -> Value * {
            if (extract_args.size() == 1) { return extract_args.front(); }
            std::reverse(extract_args.begin(), extract_args.end());
            XIRBuilder builder;
            builder.set_insertion_point(load);
            return builder.call(load->type(), ArithmeticOp::EXTRACT, extract_args);
        }();
        load->replace_all_uses_with(replacement);
        load->remove_self();
        info.removed_load_count++;
    }
}

static void run_local_store_forward_on_function(Function *function, LocalStoreForwardInfo &info) noexcept {
    if (auto definition = function->definition()) {
        // first pass: forward stores to loads within straight-line code
        forward_straight_line_stores_to_loads_on_function(definition, info);
        // second pass: forward stores to loads from local variables that only have a single (or no) store
        forward_single_store_to_loads_on_function(definition, info);
        // third pass: forward uniform stores to loads
        forward_uniform_store_to_loads_on_function(definition, info);
    }
}

}// namespace detail

LocalStoreForwardInfo local_store_forward_pass_run_on_function(Function *function) noexcept {
    LocalStoreForwardInfo info;
    detail::run_local_store_forward_on_function(function, info);
    return info;
}

LocalStoreForwardInfo local_store_forward_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LocalStoreForwardInfo info;
    for (auto f : module->function_list()) {
        detail::run_local_store_forward_on_function(f, info);
    }
    if (report != nullptr) {
        report->set("removed_load", info.removed_load_count);
    }
    return info;
}

}// namespace luisa::compute::xir
