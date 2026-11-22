#include <algorithm>
#include <cstdint>
#include <limits>

#include <luisa/core/logging.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/transpose_gep.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/core/stl/algorithm.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool is_alloca_promotable(
    AllocaInst *inst,
    const luisa::unordered_set<BasicBlock *> &reachable_blocks) noexcept {
    if (inst->op() != AllocaOp::LOCAL) { return false; }
    // A reg2mem spill slot carries synthetic provenance metadata that mem2reg
    // is specifically responsible for consuming. Other annotated allocas need
    // a distinct storage owner and are therefore not promotable.
    auto is_reg2mem_spill =
        inst->find_metadata<Reg2MemSpillMD>() != nullptr;
    const NameMD *spill_reload_name = nullptr;
    if (!is_reg2mem_spill) {
        for (auto *metadata : inst->metadata_list()) {
            // AST locals carry a debug name by default. A name describes the
            // logical variable rather than the storage operation, so it can
            // be copied to every Phi created for that variable. Other alloca
            // metadata has no unique SSA owner and remains a hard boundary.
            if (!metadata->isa<NameMD>()) { return false; }
        }
    }
    for (auto &&use : inst->use_list()) {
        LUISA_DEBUG_ASSERT(use->user() != nullptr && use->user()->isa<Instruction>(), "Invalid user.");
        auto user_inst = static_cast<Instruction *>(use->user());
        // Disconnected blocks remain owned by the function. Removing a load or
        // store from such a block while promoting the reachable uses can leave
        // retained instructions there with dangling operands. Keep the alloca
        // intact unless every direct user participates in this CFG rewrite.
        if (auto *parent = user_inst->parent_block();
            parent == nullptr || !reachable_blocks.contains(parent)) {
            return false;
        }
        if (user_inst->isa<LoadInst>()) {
            // Replacing an annotated load with a shared SSA value has no
            // unique metadata owner. The one exception is a variable name on
            // a synthetic reg2mem reload: it came from the lowered Phi and is
            // copied back to every Phi reconstructed for the spill slot.
            if (!user_inst->metadata_list().empty()) {
                if (!is_reg2mem_spill) { return false; }
                for (auto *metadata :
                     user_inst->metadata_list()) {
                    if (!metadata->isa<NameMD>()) {
                        return false;
                    }
                    auto *name =
                        static_cast<const NameMD *>(metadata);
                    if (spill_reload_name != nullptr &&
                        spill_reload_name->name() !=
                            name->name()) {
                        return false;
                    }
                    spill_reload_name = name;
                }
            }
            continue;
        }
        if (!user_inst->isa<StoreInst>()) { return false; }
        if (!user_inst->metadata_list().empty()) { return false; }
        // Reject allocas whose pointer value escapes via being stored.
        if (static_cast<StoreInst *>(user_inst)->value() == inst) { return false; }
    }
    return true;
}

struct AllocaAnalysis {

    static constexpr uint32_t invalid_block_id = std::numeric_limits<uint32_t>::max();

    const DomTree &dom;
    const luisa::unordered_map<Instruction *, uint32_t> &inst_indices;
    const luisa::unordered_map<BasicBlock *, uint32_t> &block_indices;
    const luisa::unordered_set<BasicBlock *> &reachable_blocks;
    luisa::span<BasicBlock *const> rpo_blocks;
    luisa::span<const uint32_t> dom_parent;
    luisa::span<const uint32_t> df_offsets;
    luisa::span<const uint32_t> df_ids;
    luisa::span<const uint32_t> reach_pred_offsets;
    luisa::span<const uint32_t> reach_pred_ids;
    luisa::span<const uint32_t> all_pred_offsets;
    luisa::span<BasicBlock *const> all_pred_blocks;
    luisa::span<const uint32_t> all_pred_ids;

    luisa::vector<uint32_t> def_epoch;
    luisa::vector<StoreInst *> def_stores;
    luisa::vector<uint32_t> def_block_ids;

    luisa::vector<uint32_t> use_slot_epoch;
    luisa::vector<uint32_t> use_slot_index;
    luisa::vector<uint32_t> use_block_ids;
    luisa::vector<luisa::vector<LoadInst *>> use_loads;

    luisa::vector<uint32_t> cand_idf_epoch;
    luisa::vector<uint32_t> live_epoch;
    luisa::vector<uint32_t> phi_block_ids;
    luisa::vector<uint32_t> work_list;
    uint32_t current_epoch{0u};

    void ensure_initialized() noexcept {
        if (def_epoch.size() == rpo_blocks.size()) { return; }
        const auto n = rpo_blocks.size();
        def_epoch.assign(n, 0u);
        def_stores.assign(n, nullptr);
        use_slot_epoch.assign(n, 0u);
        use_slot_index.assign(n, 0u);
        cand_idf_epoch.assign(n, 0u);
        live_epoch.assign(n, 0u);
        current_epoch = 0u;
    }

    [[nodiscard]] size_t all_in_degree(uint32_t bid) const noexcept {
        return static_cast<size_t>(all_pred_offsets[bid + 1u] - all_pred_offsets[bid]);
    }

    [[nodiscard]] bool prepare_and_check_budget(
        AllocaInst *inst, bool is_reg2mem_spill) noexcept {
        ensure_initialized();
        def_block_ids.clear();
        use_block_ids.clear();
        phi_block_ids.clear();
        if (++current_epoch == 0u) {
            std::fill(def_epoch.begin(), def_epoch.end(), 0u);
            std::fill(use_slot_epoch.begin(), use_slot_epoch.end(), 0u);
            std::fill(cand_idf_epoch.begin(), cand_idf_epoch.end(), 0u);
            std::fill(live_epoch.begin(), live_epoch.end(), 0u);
            current_epoch = 1u;
        }

        size_t total_load_count = 0u;
        // Find def and use blocks over dense block indices.
        for (auto &&use : inst->use_list()) {
            if (auto *user = use->user()) {
                LUISA_DEBUG_ASSERT(user->isa<Instruction>(), "Invalid user.");
                auto *user_inst = static_cast<Instruction *>(user);
                switch (user_inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::LOAD: {
                        auto *parent = user_inst->parent_block();
                        LUISA_DEBUG_ASSERT(parent != nullptr, "Invalid parent.");
                        auto iter = block_indices.find(parent);
                        if (iter == block_indices.end()) { break; }
                        auto bid = iter->second;
                        if (use_slot_epoch[bid] != current_epoch) {
                            auto slot = static_cast<uint32_t>(use_block_ids.size());
                            use_block_ids.emplace_back(bid);
                            use_slot_epoch[bid] = current_epoch;
                            use_slot_index[bid] = slot;
                            if (slot >= use_loads.size()) {
                                use_loads.resize(static_cast<size_t>(slot) + 1u);
                            }
                            use_loads[slot].clear();
                        }
                        use_loads[use_slot_index[bid]].emplace_back(static_cast<LoadInst *>(user_inst));
                        ++total_load_count;
                        break;
                    }
                    case DerivedInstructionTag::STORE: {
                        auto *parent = user_inst->parent_block();
                        LUISA_DEBUG_ASSERT(parent != nullptr, "Invalid parent.");
                        auto iter = block_indices.find(parent);
                        if (iter == block_indices.end()) { break; }
                        auto bid = iter->second;
                        LUISA_DEBUG_ASSERT(def_epoch[bid] != current_epoch, "Invalid state.");
                        def_epoch[bid] = current_epoch;
                        def_stores[bid] = static_cast<StoreInst *>(user_inst);
                        def_block_ids.emplace_back(bid);
                        break;
                    }
                    default: break;
                }
            }
        }

        // Compute unpruned candidate IDF(Def) over precomputed CSR dominance frontiers.
        work_list.assign(def_block_ids.begin(), def_block_ids.end());
        bool any_cand_idf = false;
        while (!work_list.empty()) {
            auto bid = work_list.back();
            work_list.pop_back();
            for (uint32_t e = df_offsets[bid], end = df_offsets[bid + 1u]; e < end; ++e) {
                auto f = df_ids[e];
                if (cand_idf_epoch[f] != current_epoch) {
                    cand_idf_epoch[f] = current_epoch;
                    work_list.emplace_back(f);
                    any_cand_idf = true;
                }
            }
        }
        // Empty IDF guarantees zero Phi placements regardless of live-in blocks.
        if (!any_cand_idf) {
            return true;
        }

        // Ordinary allocas must not explode IR size by placing massive Phis on
        // flattened state-machine dispatchers; retain sparse high-fanin allocas
        // as explicit memory. Reg2MemSpillMD must always round-trip to SSA.
        constexpr size_t kMinPhiOperandAllowance = 256u;
        constexpr size_t kPhiOperandsPerAccess = 16u;
        constexpr size_t kMaxPhiOperandsPerOrdinaryAlloca = 4096u;
        const auto access_count = def_block_ids.size() + total_load_count;
        const auto phi_operand_budget = std::min(
            kMaxPhiOperandsPerOrdinaryAlloca,
            std::max(kMinPhiOperandAllowance,
                     kPhiOperandsPerAccess * access_count));

        // Backward live-in BFS with pruned-SSA IDF intersection and early abort.
        work_list.clear();
        size_t live_phi_operands = 0u;
        auto visit_live_in = [&](uint32_t bid) noexcept -> bool {
            live_epoch[bid] = current_epoch;
            if (cand_idf_epoch[bid] == current_epoch) {
                phi_block_ids.emplace_back(bid);
                live_phi_operands += all_in_degree(bid);
                if (!is_reg2mem_spill && live_phi_operands > phi_operand_budget) {
                    return false;
                }
            }
            return true;
        };
        for (uint32_t ub : use_block_ids) {
            if (live_epoch[ub] != current_epoch) {
                if (!visit_live_in(ub)) { return false; }
                work_list.emplace_back(ub);
            }
        }
        while (!work_list.empty()) {
            auto bid = work_list.back();
            work_list.pop_back();
            for (uint32_t e = reach_pred_offsets[bid], end = reach_pred_offsets[bid + 1u]; e < end; ++e) {
                auto pred = reach_pred_ids[e];
                if (def_epoch[pred] != current_epoch && live_epoch[pred] != current_epoch) {
                    if (!visit_live_in(pred)) { return false; }
                    work_list.emplace_back(pred);
                }
            }
        }
        return true;
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

    luisa::vector<uint32_t> phi_inst_epoch;
    luisa::vector<PhiInst *> phi_insts;
    uint32_t current_phi_epoch{0u};

    // the following fields are used across the processing of different alloca's
    luisa::vector<PhiInst *> inserted;

    void ensure_initialized(size_t block_count) noexcept {
        if (phi_inst_epoch.size() == block_count) { return; }
        phi_inst_epoch.assign(block_count, 0u);
        phi_insts.assign(block_count, nullptr);
        current_phi_epoch = 0u;
    }

    [[nodiscard]] Value *find_dom_value_from_block(uint32_t bid, const Type *type,
                                                   Module *module,
                                                   const AllocaAnalysis &analysis) const noexcept {
        for (uint32_t cur = bid; cur != AllocaAnalysis::invalid_block_id; cur = analysis.dom_parent[cur]) {
            // store must have higher priority than phi nodes as it's closer to the use block
            if (analysis.def_epoch[cur] == analysis.current_epoch) {
                return analysis.def_stores[cur]->value();
            }
            // check phi nodes if no store is found
            if (phi_inst_epoch[cur] == current_phi_epoch) {
                return phi_insts[cur];
            }
        }
        // if no dominant value found, get an undef value
        return module->create_undefined(type);
    }

    void place_phi_nodes(AllocaInst *inst, const AllocaAnalysis &analysis, Mem2RegInfo &info) noexcept {
        ensure_initialized(analysis.rpo_blocks.size());
        if (++current_phi_epoch == 0u) {
            std::fill(phi_inst_epoch.begin(), phi_inst_epoch.end(), 0u);
            current_phi_epoch = 1u;
        }
        auto *type = inst->type();
        auto *module = inst->parent_module();
        const NameMD *logical_name = nullptr;
        if (inst->find_metadata<Reg2MemSpillMD>() ==
            nullptr) {
            logical_name = inst->find_metadata<NameMD>();
        } else {
            for (uint32_t ub : analysis.use_block_ids) {
                const auto slot = analysis.use_slot_index[ub];
                for (auto *load : analysis.use_loads[slot]) {
                    if (auto *name =
                            load->find_metadata<NameMD>()) {
                        logical_name = name;
                        break;
                    }
                }
                if (logical_name != nullptr) { break; }
            }
        }
        // Insert new Phi nodes at precomputed IDF(Def) ∩ LiveIn blocks.
        {
            XIRBuilder b;
            for (uint32_t fb_id : analysis.phi_block_ids) {
                auto *fb = analysis.rpo_blocks[fb_id];
                b.set_insertion_point(fb->instructions().head_sentinel());
                auto *phi = b.phi(type);
                if (logical_name != nullptr) {
                    phi->metadata_list().push_front(
                        logical_name->clone());
                }
                phi_inst_epoch[fb_id] = current_phi_epoch;
                phi_insts[fb_id] = phi;
                inserted.emplace_back(phi);
            }
        }
        // other loads must be dominated by some def/phi block, or it must contain undefined value
        for (uint32_t ub : analysis.use_block_ids) {
            const auto slot = analysis.use_slot_index[ub];
            for (auto *load_inst : analysis.use_loads[slot]) {
                LUISA_DEBUG_ASSERT(!ctx.removed.contains(load_inst), "Invalid state.");
                Value *replacement = nullptr;
                // Check if there is a store in this block that comes before this load.
                // If so, this load reads the store's value directly.
                if (analysis.def_epoch[ub] == analysis.current_epoch &&
                    analysis.inst_indices.at(analysis.def_stores[ub]) < analysis.inst_indices.at(load_inst)) {
                    replacement = analysis.def_stores[ub]->value();
                } else if (phi_inst_epoch[ub] == current_phi_epoch) {
                    replacement = phi_insts[ub];
                } else if (auto parent_id = analysis.dom_parent[ub];
                           parent_id != AllocaAnalysis::invalid_block_id) {
                    replacement = find_dom_value_from_block(parent_id, type, module, analysis);
                } else {
                    replacement = module->create_undefined(type);
                }
                replace_load_with_value(load_inst, replacement, ctx, info);
            }
        }
        // now the alloca should have no load uses but only store uses, check it
        for (auto &&use : inst->use_list()) {
            if (auto user = use->user()) {
                if (user->isa<Instruction>() &&
                    analysis.reachable_blocks.contains(static_cast<Instruction *>(user)->parent_block()) &&
                    !static_cast<Instruction *>(user)->isa<StoreInst>()) {
                    LUISA_ERROR("Invalid user.");
                }
            }
        }
        // now we fill the incoming values of the phi nodes
        for (uint32_t phi_bid : analysis.phi_block_ids) {
            auto *phi_inst = phi_insts[phi_bid];
            for (uint32_t e = analysis.all_pred_offsets[phi_bid],
                          end = analysis.all_pred_offsets[phi_bid + 1u];
                 e < end; ++e) {
                auto *pred = analysis.all_pred_blocks[e];
                auto pred_bid = analysis.all_pred_ids[e];
                auto *dom_value = pred_bid != AllocaAnalysis::invalid_block_id ?
                                      find_dom_value_from_block(pred_bid, type, module, analysis) :
                                      module->create_undefined(type);
                phi_inst->add_incoming(dom_value, pred);
            }
        }
        // remove the stores
        for (uint32_t db : analysis.def_block_ids) {
            remove_store(analysis.def_stores[db], ctx, info);
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
                                             const luisa::unordered_map<Instruction *, uint32_t> &inst_indices,
                                             const luisa::unordered_set<BasicBlock *> &reachable_blocks,
                                             Mem2RegPassContext &ctx, Mem2RegInfo &info) noexcept {
    // collect load/store instructions concerning the alloca
    seq.clear();
    for (auto &&use : inst->use_list()) {
        if (auto user = use->user()) {
            if (user->isa<LoadInst>() || user->isa<StoreInst>()) {
                auto user_inst = static_cast<Instruction *>(user);
                auto parent_block = user_inst->parent_block();
                LUISA_DEBUG_ASSERT(parent_block != nullptr, "Invalid parent.");
                if (!reachable_blocks.contains(parent_block)) { continue; }
                seq[parent_block].emplace_back(user_inst);
            }
        }
    }
    // sort the load/store instructions per block and eliminate them when possible
    for (auto &&[block, instructions] : seq) {
        luisa::sort(instructions.begin(), instructions.end(), [&](Instruction *lhs, Instruction *rhs) noexcept {
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
                    // A direct StoreInst to this alloca always replaces the whole
                    // object, independently of whether its value is scalar or an
                    // aggregate. Partial field/element stores use a GEP pointer and
                    // make the alloca non-promotable unless transpose_gep first turns
                    // them into an explicit load/insert/full-store sequence. In both
                    // cases, forwarding the direct store value is exact.
                    if (last_store != nullptr) {
                        remove_store(last_store, ctx, info);
                    }
                    last_value = static_cast<StoreInst *>(store_or_load)->value();
                    LUISA_DEBUG_ASSERT(last_value != nullptr, "Invalid store.");
                    // record this store
                    last_store = static_cast<StoreInst *>(store_or_load);
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
    if (f == nullptr) { return; }
    if (auto def = f->definition()) {
        if (def->body_block() == nullptr) { return; }

        // run the transpose GEP pass first so we can possibly handle more aggregates (EDIT: disabled for better performance on CUDA)
        // if (auto transpose_gep_info = transpose_gep_pass_run_on_function(def);
        //     transpose_gep_info.transposed_load_count != 0 ||
        //     transpose_gep_info.transposed_store_count != 0) {
        //     LUISA_VERBOSE("Transposed {} load instruction(s) and {} store instruction(s) in mem2reg pass.",
        //                   transpose_gep_info.transposed_load_count,
        //                   transpose_gep_info.transposed_store_count);
        // }
        
        // collect local alloca instructions that can be promoted
        luisa::vector<AllocaInst *> local_allocas;
        luisa::vector<AllocaInst *> promotable;
        luisa::vector<BasicBlock *> rpo_blocks;
        luisa::unordered_map<Instruction *, uint32_t> inst_indices;
        luisa::unordered_map<BasicBlock *, uint32_t> block_indices;
        luisa::unordered_set<BasicBlock *> reachable_blocks;
        def->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
            reachable_blocks.emplace(block);
            block_indices.emplace(block, static_cast<uint32_t>(rpo_blocks.size()));
            rpo_blocks.emplace_back(block);
            block->traverse_instructions([&](Instruction *inst) noexcept {
                switch (inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::ALLOCA: {
                        local_allocas.emplace_back(static_cast<AllocaInst *>(inst));
                        break;
                    }
                    case DerivedInstructionTag::LOAD: [[fallthrough]];
                    case DerivedInstructionTag::STORE: {
                        inst_indices.emplace(inst, static_cast<uint32_t>(inst_indices.size()));
                        break;
                    }
                    default: break;
                }
            });
        });
        for (auto *alloca_inst : local_allocas) {
            if (is_alloca_promotable(alloca_inst, reachable_blocks)) {
                promotable.emplace_back(alloca_inst);
            }
        }
        // do some simplification first
        Mem2RegPassContext ctx;
        if (!promotable.empty()) {
            AllocaStoreLoadSequence seq;
            for (auto inst : promotable) {
                simplify_single_block_store_load(inst, seq, inst_indices, reachable_blocks, ctx, info);
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
            const auto block_count = rpo_blocks.size();
            luisa::vector<uint32_t> dom_parent(block_count, AllocaAnalysis::invalid_block_id);
            luisa::vector<uint32_t> df_offsets(block_count + 1u, 0u);
            luisa::vector<uint32_t> df_ids;
            luisa::vector<uint32_t> reach_pred_offsets(block_count + 1u, 0u);
            luisa::vector<uint32_t> reach_pred_ids;
            luisa::vector<uint32_t> all_pred_offsets(block_count + 1u, 0u);
            luisa::vector<BasicBlock *> all_pred_blocks;
            luisa::vector<uint32_t> all_pred_ids;
            for (size_t bid = 0u; bid < block_count; ++bid) {
                auto *bb = rpo_blocks[bid];
                if (auto *node = dom.node_or_null(bb)) {
                    if (auto *parent = node->parent()) {
                        if (auto it = block_indices.find(parent->block());
                            it != block_indices.end()) {
                            dom_parent[bid] = it->second;
                        }
                    }
                    for (auto *frontier : node->frontiers()) {
                        if (auto it = block_indices.find(frontier->block());
                            it != block_indices.end()) {
                            df_ids.emplace_back(it->second);
                        }
                    }
                }
                df_offsets[bid + 1u] = static_cast<uint32_t>(df_ids.size());
                bb->traverse_predecessors(true, [&](BasicBlock *pred) noexcept {
                    if (auto it = block_indices.find(pred); it != block_indices.end()) {
                        reach_pred_ids.emplace_back(it->second);
                    }
                });
                reach_pred_offsets[bid + 1u] = static_cast<uint32_t>(reach_pred_ids.size());
                bb->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                    all_pred_blocks.emplace_back(pred);
                    auto it = block_indices.find(pred);
                    all_pred_ids.emplace_back(it != block_indices.end() ?
                                                  it->second :
                                                  AllocaAnalysis::invalid_block_id);
                });
                all_pred_offsets[bid + 1u] = static_cast<uint32_t>(all_pred_blocks.size());
            }
            AllocaAnalysis analysis{.dom = dom,
                                    .inst_indices = inst_indices,
                                    .block_indices = block_indices,
                                    .reachable_blocks = reachable_blocks,
                                    .rpo_blocks = rpo_blocks,
                                    .dom_parent = dom_parent,
                                    .df_offsets = df_offsets,
                                    .df_ids = df_ids,
                                    .reach_pred_offsets = reach_pred_offsets,
                                    .reach_pred_ids = reach_pred_ids,
                                    .all_pred_offsets = all_pred_offsets,
                                    .all_pred_blocks = all_pred_blocks,
                                    .all_pred_ids = all_pred_ids};
            PhiInsertionAndRenaming insertion{.ctx = ctx};
            luisa::vector<LoadInst *> loads;
            luisa::unordered_set<BasicBlock *> seen_store_blocks;
            for (auto inst : promotable) {
                // The local simplifier above must leave at most one direct store
                // per block. Keep this defensive boundary check: the SSA rewrite's
                // per-block definition map cannot represent more than one.
                loads.clear();
                size_t store_count = 0u;
                StoreInst *single_store = nullptr;
                BasicBlock *first_store_block = nullptr;
                bool has_multiple_stores = false;
                for (auto &&use : inst->use_list()) {
                    auto *user = use->user();
                    if (user == nullptr || !user->isa<Instruction>()) { continue; }
                    auto *user_inst = static_cast<Instruction *>(user);
                    if (user_inst->isa<LoadInst>()) {
                        loads.emplace_back(static_cast<LoadInst *>(user_inst));
                    } else if (user_inst->isa<StoreInst>()) {
                        auto *store = static_cast<StoreInst *>(user_inst);
                        auto *block = store->parent_block();
                        if (store_count == 0u) {
                            single_store = store;
                            first_store_block = block;
                            store_count = 1u;
                        } else {
                            if (store_count == 1u) {
                                if (block == first_store_block) {
                                    has_multiple_stores = true;
                                    break;
                                }
                                seen_store_blocks.clear();
                                seen_store_blocks.emplace(first_store_block);
                            }
                            if (!seen_store_blocks.emplace(block).second) {
                                has_multiple_stores = true;
                                break;
                            }
                            ++store_count;
                        }
                    }
                }
                if (has_multiple_stores) {
                    continue;
                }
                if (store_count == 0u) {
                    auto *undef = loads.empty() ?
                                      nullptr :
                                      def->parent_module()->create_undefined(inst->type());
                    for (auto *load : loads) {
                        replace_load_with_value(load, undef, ctx, info);
                    }
                    LUISA_ASSERT(inst->use_list().empty(), "Invalid state.");
                    remove_alloca(inst, ctx, info);
                    continue;
                }
                if (store_count == 1u) {
                    auto *def_block = single_store->parent_block();
                    auto def_idx = inst_indices.at(single_store);
                    auto all_loads_dominated = true;
                    for (auto *load : loads) {
                        auto *load_block = load->parent_block();
                        if (load_block == def_block) {
                            if (def_idx >= inst_indices.at(load)) {
                                all_loads_dominated = false;
                                break;
                            }
                        } else if (!dom.dominates(def_block, load_block)) {
                            all_loads_dominated = false;
                            break;
                        }
                    }
                    if (all_loads_dominated) {
                        auto *stored_value = single_store->value();
                        for (auto *load : loads) {
                            replace_load_with_value(load, stored_value, ctx, info);
                        }
                        remove_store(single_store, ctx, info);
                        LUISA_ASSERT(inst->use_list().empty(), "Invalid state.");
                        remove_alloca(inst, ctx, info);
                        continue;
                    }
                }
                // Analyze pruned-SSA Phi placements and enforce ordinary-alloca
                // Phi-operand budget before mutating the IR.
                const auto is_reg2mem_spill =
                    inst->find_metadata<Reg2MemSpillMD>() != nullptr;
                if (!analysis.prepare_and_check_budget(inst, is_reg2mem_spill)) {
                    continue;
                }
                insertion.place_phi_nodes(inst, analysis, info);
            }
            insertion.simplify_phi_nodes(info);
        }
    }
}

}// namespace detail

Mem2RegInfo mem2reg_pass_run_on_function(Function *function) noexcept {
    Mem2RegInfo info;
    if (function != nullptr) {
        detail::promote_alloca_instructions_in_function(function, info);
    }
    return info;
}

Mem2RegInfo mem2reg_pass_run_on_module(Module *module, PassReport *report) noexcept {
    Mem2RegInfo info;
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            detail::promote_alloca_instructions_in_function(f, info);
        }
    }
    if (report != nullptr) {
        report->set("promoted_alloca", info.promoted_alloca_count);
        report->set("removed_store", info.removed_store_count);
        report->set("removed_load", info.removed_load_count);
        report->set("inserted_phi", info.inserted_phi_count);
    }
    return info;
}

}// namespace luisa::compute::xir
