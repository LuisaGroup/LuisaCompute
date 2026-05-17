#include <luisa/core/logging.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/transpose_gep.h>
#include <luisa/xir/passes/mem2reg.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool is_alloca_promotable(AllocaInst *inst) noexcept {
    // check if it's a local variable
    if (inst->op() != AllocaOp::LOCAL) { return false; }
    // check if it's used as reference in other instructions than load/store/gep
    for (auto &&use : inst->use_list()) {
        LUISA_DEBUG_ASSERT(use->user() != nullptr && use->user()->isa<Instruction>(), "Invalid user.");
        if (auto user_inst = static_cast<Instruction *>(use->user());
            !user_inst->isa<LoadInst>() && !user_inst->isa<StoreInst>() && !user_inst->isa<GEPInst>()) {
            return false;
        }
    }
    // Also check that GEP users are only used by load/store
    // (trace through GEP chain to ensure the alloca is only loaded/stored)
    luisa::vector<const Instruction *> work_list;
    luisa::unordered_set<const Instruction *> visited;
    for (auto &&use : inst->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<Instruction>()) {
            work_list.push_back(static_cast<const Instruction *>(user));
        }
    }
    while (!work_list.empty()) {
        auto u = work_list.back();
        work_list.pop_back();
        if (!visited.emplace(u).second) { continue; }
        if (u->isa<LoadInst>() || u->isa<StoreInst>()) { continue; }
        if (u->isa<GEPInst>()) {
            // GEP is fine, trace its users
            for (auto &&gep_use : u->use_list()) {
                if (auto gep_user = gep_use->user(); gep_user != nullptr && gep_user->isa<Instruction>()) {
                    work_list.push_back(static_cast<const Instruction *>(gep_user));
                }
            }
        } else {
            return false;// non-load/store/gep user found through GEP chain
        }
    }
    return true;
}

struct AllocaAnalysis {

    const DomTree &dom;
    const luisa::unordered_map<Instruction *, uint> &inst_indices;
    const luisa::unordered_map<BasicBlock *, uint> &block_indices;

    luisa::unordered_map<BasicBlock *, StoreInst *> def_blocks;
    luisa::unordered_map<BasicBlock *, LoadInst *> use_blocks;
    luisa::unordered_set<BasicBlock *> live_in_blocks;

    void analyze(AllocaInst *inst) noexcept {
        def_blocks.clear();
        use_blocks.clear();
        live_in_blocks.clear();
        // find def and use blocks
        for (auto &&use : inst->use_list()) {
            if (auto user = use->user()) {
                LUISA_DEBUG_ASSERT(user->isa<Instruction>(), "Invalid user.");
                switch (auto user_inst = static_cast<Instruction *>(user); user_inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::LOAD: {
                        LUISA_DEBUG_ASSERT(user_inst->parent_block() != nullptr, "Invalid parent.");
                        auto [_, success] = use_blocks.try_emplace(user_inst->parent_block(), static_cast<LoadInst *>(user_inst));
                        LUISA_DEBUG_ASSERT(success, "Invalid state.");
                        break;
                    }
                    case DerivedInstructionTag::STORE: {
                        LUISA_DEBUG_ASSERT(user_inst->parent_block() != nullptr, "Invalid parent.");
                        auto [_, success] = def_blocks.try_emplace(user_inst->parent_block(), static_cast<StoreInst *>(user_inst));
                        LUISA_DEBUG_ASSERT(success, "Invalid state.");
                        break;
                    }
                    default: break;
                }
            }
        }
        // compute live-in blocks
        luisa::fixed_vector<BasicBlock *, 64u> work_list;
        work_list.reserve(use_blocks.size());
        for (auto [use_block, load] : use_blocks) {
            if (auto def_iter = def_blocks.find(use_block); def_iter != def_blocks.end()) {
                // make sure the store is after the load
                LUISA_ASSERT(inst_indices.at(def_iter->second) > inst_indices.at(load), "Invalid state.");
            }
            work_list.emplace_back(use_block);
        }
        // extend the live-in block set by adding all non-defining predecessors of the known live-in blocks
        while (!work_list.empty()) {
            auto block = work_list.back();
            work_list.pop_back();
            if (live_in_blocks.emplace(block).second) {
                block->traverse_predecessors(true, [&](BasicBlock *pred) noexcept {
                    if (!def_blocks.contains(pred) && !live_in_blocks.contains(pred)) {
                        work_list.emplace_back(pred);
                    }
                });
            }
        }
    }
};

struct Mem2RegPassContext {

private:
    luisa::vector<ManagedPtr<Instruction>> _lifetime_holder;

public:
    luisa::unordered_set<Instruction *> removed;
    void mark_as_removed(ManagedPtr<Instruction> i) noexcept {
        removed.emplace(_lifetime_holder.emplace_back(std::move(i)).get());
    }
};

static void replace_load_with_value(LoadInst *load_inst, Value *value,
                                    Mem2RegPassContext &ctx, Mem2RegInfo &info) noexcept {
    load_inst->replace_all_uses_with(value);
    ctx.mark_as_removed(load_inst->remove_self());
    info.removed_load_count++;
}

static void remove_store(StoreInst *store_inst, Mem2RegPassContext &ctx, Mem2RegInfo &info) noexcept {
    ctx.mark_as_removed(store_inst->remove_self());
    info.removed_store_count++;
}

static void remove_alloca(AllocaInst *alloca_inst, Mem2RegPassContext &ctx, Mem2RegInfo &info) noexcept {
    ctx.mark_as_removed(alloca_inst->remove_self());
    info.promoted_alloca_count++;
}

struct PhiInsertionAndRenaming {

    Mem2RegPassContext &ctx;

    luisa::unordered_map<BasicBlock *, PhiInst *> block_to_phi;

    // the following fields are used across the processing of different alloca's
    luisa::vector<PhiInst *> inserted;

    [[nodiscard]] Value *find_dom_value_from_block(BasicBlock *block, const Type *type,
                                                   const AllocaAnalysis &analysis) noexcept {
        for (auto node = analysis.dom.node_or_null(block); node != nullptr; node = node->parent()) {
            // store must have higher priority than phi nodes as it's closer to the use block
            if (auto iter = analysis.def_blocks.find(node->block()); iter != analysis.def_blocks.end()) {
                return iter->second->value();
            }
            // check phi nodes if no store is found
            if (auto iter = block_to_phi.find(node->block()); iter != block_to_phi.end()) {
                return iter->second;
            }
        }
        // if no dominant value found, get an undef value
        return block->parent_module()->create_undefined(type);
    }

    void place_phi_nodes(AllocaInst *inst, const AllocaAnalysis &analysis, Mem2RegInfo &info) noexcept {
        // insert new phi nodes by traversing the closure of dominance frontiers of the def blocks
        block_to_phi.clear();
        auto type = inst->type();
        {
            luisa::fixed_vector<BasicBlock *, 64u> work_list;
            work_list.reserve(analysis.def_blocks.size());
            for (auto [def_block, _] : analysis.def_blocks) {
                work_list.emplace_back(def_block);
            }
            while (!work_list.empty()) {
                auto block = work_list.back();
                work_list.pop_back();
                for (auto frontier : analysis.dom.node(block)->frontiers()) {
                    if (auto fb = frontier->block(); analysis.live_in_blocks.contains(fb)) {
                        if (auto iter = block_to_phi.try_emplace(fb, nullptr).first; iter->second == nullptr) {
                            // insert the phi node
                            XIRBuilder b;
                            b.set_insertion_point(fb->instructions().head_sentinel());
                            auto phi = b.phi(type);
                            iter->second = phi;
                            inserted.emplace_back(phi);
                            // add the block to the work list to compute the closure
                            work_list.emplace_back(fb);
                        }
                    }
                }
            }
        }
        // other loads must be dominated by some def/phi block, or it must contain undefined value
        for (auto [use_block, load_inst] : analysis.use_blocks) {
            LUISA_DEBUG_ASSERT(!ctx.removed.contains(load_inst), "Invalid state.");
            if (auto phi_iter = block_to_phi.find(use_block); phi_iter != block_to_phi.end()) {
                // if we have a phi node in the use block, we can replace the load with it
                replace_load_with_value(load_inst, phi_iter->second, ctx, info);
            } else if (auto parent = analysis.dom.immediate_dominator(use_block)) {
                // otherwise, we walk the dom tree to find the value that dominates the use block
                auto dom_value = find_dom_value_from_block(parent, type, analysis);
                replace_load_with_value(load_inst, dom_value, ctx, info);
            } else {
                // otherwise we have to use an undefined value
                auto undef = use_block->parent_module()->create_undefined(type);
                replace_load_with_value(load_inst, undef, ctx, info);
            }
        }
        // now the alloca should have no load uses but only store uses, check it
        for (auto &&use : inst->use_list()) {
            if (auto user = use->user()) {
                LUISA_ASSERT(user->isa<StoreInst>(), "Invalid user.");
            }
        }
        // now we fill the incoming values of the phi nodes
        for (auto mapping : block_to_phi) {
            // earlier clang compilers have trouble with structural binding in lambda capture, so we manually unpack here
            auto phi_block = mapping.first;
            auto phi_inst = mapping.second;
            phi_block->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                auto dom_value = find_dom_value_from_block(pred, type, analysis);
                phi_inst->add_incoming(dom_value, pred);
            });
        }
        // remove the stores
        for (auto [def_block, store_inst] : analysis.def_blocks) {
            remove_store(store_inst, ctx, info);
        }
        // remove the local variable (which should have no uses now) and record the promotion
        LUISA_ASSERT(inst->use_list().empty(), "Invalid state.");
        remove_alloca(inst, ctx, info);
    }

    void simplify_phi_nodes(Mem2RegInfo &info) noexcept {
        for (;;) {
            auto prev_inserted_count = inserted.size();
            inserted.erase(std::remove_if(inserted.begin(), inserted.end(), remove_redundant_phi_instruction),
                           inserted.end());
            if (inserted.size() == prev_inserted_count) { break; }
        }
        info.inserted_phi_count += inserted.size();
    }
};

using AllocaStoreLoadSequence = luisa::unordered_map<BasicBlock *, std::vector<Instruction *>>;

// after this function, for each block, the must be at most one store and one load instruction for an
// alloca, and the load instruction must precede the store instruction if both exist
static void simplify_single_block_store_load(AllocaInst *inst, AllocaStoreLoadSequence &seq,
                                             const luisa::unordered_map<Instruction *, uint> &inst_indices,
                                             Mem2RegPassContext &ctx, Mem2RegInfo &info) noexcept {
    // collect load/store instructions concerning the alloca
    seq.clear();
    for (auto &&use : inst->use_list()) {
        if (auto user = use->user()) {
            if (user->isa<LoadInst>() || user->isa<StoreInst>()) {
                auto user_inst = static_cast<Instruction *>(user);
                auto parent_block = user_inst->parent_block();
                LUISA_DEBUG_ASSERT(parent_block != nullptr, "Invalid parent.");
                seq[parent_block].emplace_back(user_inst);
            }
        }
    }
    // sort the load/store instructions per block and eliminate them when possible
    for (auto &&[block, instructions] : seq) {
        std::sort(instructions.begin(), instructions.end(), [&](Instruction *lhs, Instruction *rhs) noexcept {
            return inst_indices.at(lhs) < inst_indices.at(rhs);
        });
        // eliminate redundant loads and overwritten stores
        auto last_store = static_cast<StoreInst *>(nullptr);
        auto last_value = static_cast<Value *>(nullptr);
        for (auto store_or_load : instructions) {
            switch (store_or_load->derived_instruction_tag()) {
                case DerivedInstructionTag::LOAD: {
                    if (last_value != nullptr) {// we can forward the last loaded/stored value to this load
                        replace_load_with_value(static_cast<LoadInst *>(store_or_load), last_value, ctx, info);
                    } else {// otherwise, record this load
                        last_value = store_or_load;
                    }
                    break;
                }
                case DerivedInstructionTag::STORE: {
                    // we have overwritten the last store so remove it if any
                    if (last_store != nullptr) {
                        remove_store(last_store, ctx, info);
                    }
                    // record this store
                    last_store = static_cast<StoreInst *>(store_or_load);
                    last_value = last_store->value();
                    LUISA_DEBUG_ASSERT(last_value != nullptr, "Invalid store.");
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION("Invalid instruction.");
            }
        }
    }
    // if we find the alloca now is stored to only, we can remove it
    auto all_store = true;
    for (auto &&use : inst->use_list()) {
        if (auto user = use->user(); user != nullptr && !user->isa<StoreInst>()) {
            all_store = false;
            break;
        }
    }
    if (all_store) {
        // remove all users
        while (!inst->use_list().empty()) {
            remove_store(static_cast<StoreInst *>(inst->use_list().front()->user()), ctx, info);
        }
        // remove self
        remove_alloca(inst, ctx, info);
    }
}

static void promote_alloca_instructions_in_function(Function *f, Mem2RegInfo &info) noexcept {
    if (auto def = f->definition()) {
        // run the transpose GEP pass first so we can possibly handle more aggregates
        if (auto transpose_gep_info = transpose_gep_pass_run_on_function(def);
            transpose_gep_info.transposed_load_count != 0u ||
            transpose_gep_info.transposed_store_count != 0u) {
            LUISA_VERBOSE("Transposed {} load instruction(s) and {} store instruction(s) in mem2reg pass.",
                          transpose_gep_info.transposed_load_count,
                          transpose_gep_info.transposed_store_count);
        }
        // collect local alloca instructions that can be promoted
        luisa::vector<AllocaInst *> promotable;
        luisa::unordered_map<Instruction *, uint> inst_indices;
        luisa::unordered_map<BasicBlock *, uint> block_indices;
        def->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
            block_indices.emplace(block, static_cast<uint>(block_indices.size()));
            block->traverse_instructions([&](Instruction *inst) noexcept {
                switch (inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::ALLOCA: {
                        if (auto alloca_inst = static_cast<AllocaInst *>(inst); is_alloca_promotable(alloca_inst)) {
                            promotable.emplace_back(alloca_inst);
                        }
                        break;
                    }
                    case DerivedInstructionTag::LOAD: [[fallthrough]];
                    case DerivedInstructionTag::STORE: {
                        inst_indices.emplace(inst, static_cast<uint>(inst_indices.size()));
                        break;
                    }
                    default: break;
                }
            });
        });
        // do some simplification first
        Mem2RegPassContext ctx;
        if (!promotable.empty()) {
            AllocaStoreLoadSequence seq;
            for (auto inst : promotable) {
                simplify_single_block_store_load(inst, seq, inst_indices, ctx, info);
            }
            // erase the alloca instructions that are already removed
            promotable.erase(
                std::remove_if(promotable.begin(), promotable.end(), [&](AllocaInst *inst) noexcept {
                    return ctx.removed.contains(inst);
                }),
                promotable.end());
        }
        // perform the SSA rewrite pass for the remaining alloca instructions
        if (!promotable.empty()) {
            auto dom = compute_dom_tree(def);
            AllocaAnalysis analysis{.dom = dom,
                                    .inst_indices = inst_indices,
                                    .block_indices = block_indices};
            PhiInsertionAndRenaming insertion{.ctx = ctx};
            for (auto inst : promotable) {
                // analyze and insert phi nodes
                analysis.analyze(inst);
                insertion.place_phi_nodes(inst, analysis, info);
            }
            insertion.simplify_phi_nodes(info);
        }
    }
}

}// namespace detail

Mem2RegInfo mem2reg_pass_run_on_function(Function *function) noexcept {
    Mem2RegInfo info;
    detail::promote_alloca_instructions_in_function(function, info);
    return info;
}

Mem2RegInfo mem2reg_pass_run_on_module(Module *module) noexcept {
    Mem2RegInfo info;
    for (auto f : module->function_list()) {
        detail::promote_alloca_instructions_in_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
