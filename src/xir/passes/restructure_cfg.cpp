#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace {

static void check_phi_free(FunctionDefinition *def) noexcept {
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        for (auto *inst : bb->instructions()) {
            if (inst->isa<PhiInst>()) {
                LUISA_ERROR_WITH_LOCATION("restructure_cfg requires phi-free input; run reg2mem_pass first");
            }
        }
    });
}

[[nodiscard]] static bool is_sink(BasicBlock *bb) noexcept {
    if (!bb->is_terminated()) { return true; }
    auto *t = bb->terminator();
    if (t->isa<ReturnInst>() || t->isa<UnreachableInst>() || t->isa<RasterDiscardInst>()) { return true; }
    bool has_succ = false;
    bb->traverse_successors(false, [&](BasicBlock *) noexcept { has_succ = true; });
    return !has_succ;
}

struct PostDomInfo {
    luisa::unordered_map<BasicBlock *, BasicBlock *> ipostdom;
    BasicBlock *virtual_exit{nullptr};
};

[[nodiscard]] static PostDomInfo compute_post_dom(FunctionDefinition *def) noexcept {
    luisa::vector<BasicBlock *> all_blocks;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        all_blocks.emplace_back(bb);
    });

    luisa::vector<BasicBlock *> sinks;
    for (auto *bb : all_blocks) {
        if (is_sink(bb)) { sinks.emplace_back(bb); }
    }

    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> pred_map;
    for (auto *bb : all_blocks) {
        bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            pred_map[succ].emplace_back(bb);
        });
    }

    static int virtual_exit_sentinel = 0;
    BasicBlock *virt = reinterpret_cast<BasicBlock *>(&virtual_exit_sentinel);

    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> aug_pred_map = pred_map;
    for (auto *s : sinks) {
        aug_pred_map[virt].emplace_back(s);
    }

    luisa::vector<BasicBlock *> rpo;
    {
        luisa::unordered_set<BasicBlock *> visited;
        luisa::vector<std::pair<BasicBlock *, size_t>> stack;
        visited.emplace(virt);
        stack.emplace_back(virt, 0);
        while (!stack.empty()) {
            auto *cur = stack.back().first;
            auto &idx = stack.back().second;
            auto &preds = aug_pred_map[cur];
            if (idx < preds.size()) {
                auto *pred = preds[idx++];
                if (!visited.contains(pred)) {
                    visited.emplace(pred);
                    stack.emplace_back(pred, 0);
                }
            } else {
                rpo.emplace_back(cur);
                stack.pop_back();
            }
        }
    }

    luisa::unordered_map<BasicBlock *, size_t> rpo_index;
    for (size_t i = 0; i < rpo.size(); i++) { rpo_index[rpo[i]] = i; }
    rpo_index[nullptr] = SIZE_MAX;

    PostDomInfo result;
    result.virtual_exit = virt;
    auto &ipostdom = result.ipostdom;
    for (auto *bb : rpo) { ipostdom[bb] = nullptr; }
    ipostdom[virt] = virt;

    luisa::unordered_set<BasicBlock *> processed;
    processed.emplace(virt);

    auto intersect = [&](BasicBlock *b1, BasicBlock *b2) noexcept -> BasicBlock * {
        if (b1 == nullptr) { return b2; }
        if (b2 == nullptr) { return b1; }
        auto f1 = b1;
        auto f2 = b2;
        while (f1 != f2) {
            if (!rpo_index.count(f1) || !rpo_index.count(f2)) { return nullptr; }
            auto i1 = rpo_index[f1];
            auto i2 = rpo_index[f2];
            if (i1 == i2) { return nullptr; }
            while (rpo_index.count(f1) && rpo_index.count(f2) && rpo_index[f1] < rpo_index[f2]) {
                f1 = ipostdom[f1];
                if (f1 == nullptr) { return nullptr; }
            }
            while (rpo_index.count(f1) && rpo_index.count(f2) && rpo_index[f2] < rpo_index[f1]) {
                f2 = ipostdom[f2];
                if (f2 == nullptr) { return nullptr; }
            }
        }
        return f1;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = rpo.rbegin(); it != rpo.rend(); ++it) {
            auto *bb = *it;
            if (bb == virt) { continue; }

            luisa::vector<BasicBlock *> succs;
            if (is_sink(bb)) {
                succs.emplace_back(virt);
            } else {
                bb->traverse_successors(false, [&](BasicBlock *s) noexcept { succs.emplace_back(s); });
            }

            BasicBlock *new_ipostdom = nullptr;
            for (auto *s : succs) {
                if (processed.contains(s)) {
                    new_ipostdom = intersect(new_ipostdom, s);
                }
            }
            if (ipostdom[bb] != new_ipostdom) {
                ipostdom[bb] = new_ipostdom;
                changed = true;
            }
            if (new_ipostdom != nullptr) { processed.emplace(bb); }
        }
    }

    return result;
}

[[nodiscard]] static size_t dom_depth(const DomTree &dom, BasicBlock *bb) noexcept {
    size_t d = 0;
    auto *node = dom.node_or_null(bb);
    while (node != nullptr && node->parent() != nullptr) {
        ++d;
        node = node->parent();
    }
    return d;
}

[[nodiscard]] static BasicBlock *common_postdom(const PostDomInfo &pdom, luisa::span<BasicBlock *const> blocks) noexcept {
    if (blocks.empty()) { return nullptr; }
    auto ancestors_of = [&](BasicBlock *bb) noexcept {
        luisa::unordered_set<BasicBlock *> chain;
        auto *cur = bb;
        while (cur != nullptr && cur != pdom.virtual_exit) {
            if (!chain.emplace(cur).second) { return chain; }
            auto it = pdom.ipostdom.find(cur);
            if (it == pdom.ipostdom.end()) { return chain; }
            cur = it->second;
        }
        if (cur == pdom.virtual_exit) { chain.emplace(pdom.virtual_exit); }
        return chain;
    };
    auto common = ancestors_of(blocks[0]);
    for (size_t i = 1; i < blocks.size(); i++) {
        auto other = ancestors_of(blocks[i]);
        luisa::unordered_set<BasicBlock *> next;
        for (auto *bb : common) {
            if (other.contains(bb)) { next.emplace(bb); }
        }
        common = std::move(next);
        if (common.empty()) { return nullptr; }
    }
    auto *cur = blocks[0];
    while (cur != nullptr && cur != pdom.virtual_exit) {
        if (common.contains(cur)) { return cur == pdom.virtual_exit ? nullptr : cur; }
        auto it = pdom.ipostdom.find(cur);
        if (it == pdom.ipostdom.end()) { return nullptr; }
        cur = it->second;
    }
    return nullptr;
}

static bool retarget_terminator(Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
    if (term == nullptr) { return false; }
    auto changed = false;
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH: {
            auto *br = static_cast<BranchInst *>(term);
            if (br->target_block() == from) {
                br->set_target_block(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto *cb = static_cast<ConditionalBranchInst *>(term);
            if (cb->true_block() == from) {
                cb->set_true_target(to);
                changed = true;
            }
            if (cb->false_block() == from) {
                cb->set_false_target(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::SWITCH: {
            auto *sw = static_cast<SwitchInst *>(term);
            if (sw->default_block() == from) {
                sw->set_default_block(to);
                changed = true;
            }
            for (size_t i = 0; i < sw->case_count(); i++) {
                if (sw->case_block(i) == from) {
                    sw->set_case_block(i, to);
                    changed = true;
                }
            }
            break;
        }
        case DerivedInstructionTag::IF: {
            auto *ii = static_cast<IfInst *>(term);
            if (ii->true_block() == from) {
                ii->set_true_target(to);
                changed = true;
            }
            if (ii->false_block() == from) {
                ii->set_false_target(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::LOOP: {
            auto *lp = static_cast<LoopInst *>(term);
            if (lp->prepare_block() == from) {
                lp->set_prepare_block(to);
                changed = true;
            }
            if (lp->body_block() == from) {
                lp->set_body_block(to);
                changed = true;
            }
            if (lp->update_block() == from) {
                lp->set_update_block(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::SIMPLE_LOOP: {
            auto *sl = static_cast<SimpleLoopInst *>(term);
            if (sl->body_block() == from) {
                sl->set_body_block(to);
                changed = true;
            }
            break;
        }
        default: break;
    }
    return changed;
}

// Forward declaration.
[[nodiscard]] static bool clone_subgraph_to_target(FunctionDefinition *def,
                                                    BasicBlock *E, BasicBlock *P,
                                                    BasicBlock *target,
                                                    BasicBlock *new_target) noexcept;

[[nodiscard]] static bool try_restructure_loop(FunctionDefinition *def,
                                               const DomTree &dom,
                                               const PostDomInfo &pdom,
                                               RestructureCFGInfo &info) noexcept {
    luisa::vector<BasicBlock *> all_blocks;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept { all_blocks.emplace_back(bb); });

    luisa::unordered_set<BasicBlock *> already_loop_headers;
    for (auto *bb : all_blocks) {
        if (!bb->is_terminated()) { continue; }
        auto *term = bb->terminator();
        if (term->isa<LoopInst>()) {
            auto *li = static_cast<LoopInst *>(term);
            if (li->prepare_block()) { already_loop_headers.emplace(li->prepare_block()); }
        } else if (term->isa<SimpleLoopInst>()) {
            auto *sl = static_cast<SimpleLoopInst *>(term);
            if (sl->body_block()) { already_loop_headers.emplace(sl->body_block()); }
        }
    }

    struct LoopCandidate {
        BasicBlock *header{nullptr};
        luisa::vector<BasicBlock *> latches;
        size_t depth{0};
    };

    luisa::vector<LoopCandidate> candidates;

    for (auto *bb : all_blocks) {
        if (!bb->is_terminated()) { continue; }
        auto *term = bb->terminator();
        BasicBlock *back_target = nullptr;
        if (term->isa<BranchInst>()) {
            back_target = static_cast<BranchInst *>(term)->target_block();
        } else if (term->isa<ConditionalBranchInst>()) {
            auto *cb = static_cast<ConditionalBranchInst *>(term);
            if (dom.dominates(cb->true_block(), bb)) {
                back_target = cb->true_block();
            } else if (dom.dominates(cb->false_block(), bb)) {
                back_target = cb->false_block();
            }
        }
        if (back_target == nullptr) { continue; }
        if (!dom.dominates(back_target, bb)) { continue; }
        if (already_loop_headers.contains(back_target)) { continue; }

        bool found = false;
        for (auto &c : candidates) {
            if (c.header == back_target) {
                c.latches.emplace_back(bb);
                found = true;
                break;
            }
        }
        if (!found) {
            LoopCandidate c;
            c.header = back_target;
            c.latches.emplace_back(bb);
            c.depth = dom_depth(dom, back_target);
            candidates.emplace_back(std::move(c));
        }
    }

    if (candidates.empty()) { return false; }

    luisa::sort(candidates.begin(), candidates.end(), [](const LoopCandidate &a, const LoopCandidate &b) noexcept {
        return a.depth > b.depth;
    });

    auto &cand = candidates[0];
    auto *header = cand.header;
    auto &latches = cand.latches;

    for (auto *latch : latches) {
        if (!dom.dominates(header, latch)) {
            LUISA_WARNING_WITH_LOCATION("restructure_cfg: irreducible back-edge from block to non-dominating header; skipping region");
            info.irreducible_region_count++;
            return false;
        }
    }

    luisa::unordered_set<BasicBlock *> loop_blocks;
    {
        luisa::vector<BasicBlock *> worklist{latches.begin(), latches.end()};
        loop_blocks.emplace(header);
        for (auto *l : latches) { loop_blocks.emplace(l); }
        while (!worklist.empty()) {
            auto *cur = worklist.back();
            worklist.pop_back();
            cur->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                if (!dom.contains(pred)) { return; }
                if (!loop_blocks.contains(pred) && dom.dominates(header, pred)) {
                    loop_blocks.emplace(pred);
                    worklist.emplace_back(pred);
                }
            });
        }
    }

    luisa::vector<std::pair<BasicBlock *, BasicBlock *>> pre_exit_edges;
    for (auto *lb : loop_blocks) {
        if (!lb->is_terminated()) { continue; }
        lb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (!loop_blocks.contains(succ)) {
                pre_exit_edges.emplace_back(lb, succ);
            }
        });
    }
    luisa::unordered_set<BasicBlock *> pre_exit_targets_set;
    for (auto &[src, tgt] : pre_exit_edges) { pre_exit_targets_set.emplace(tgt); }
    luisa::vector<BasicBlock *> pre_exit_targets{pre_exit_targets_set.begin(), pre_exit_targets_set.end()};

    BasicBlock *dispatch_merge_or_null = nullptr;
    if (pre_exit_targets.size() > 1) {
        dispatch_merge_or_null = common_postdom(pdom, luisa::span<BasicBlock *const>{pre_exit_targets});
        if (dispatch_merge_or_null == pdom.virtual_exit) {
            dispatch_merge_or_null = nullptr;
        }
    }

    BasicBlock *canonical_latch = nullptr;
    if (latches.size() == 1) {
        canonical_latch = latches[0];
    } else {
        canonical_latch = def->create_basic_block();
        for (auto *latch : latches) {
            if (!latch->is_terminated()) { continue; }
            retarget_terminator(latch->terminator(), header, canonical_latch);
        }
        XIRBuilder b;
        b.set_insertion_point(canonical_latch);
        b.br(header);
        loop_blocks.emplace(canonical_latch);
    }

    luisa::vector<BasicBlock *> entry_preds;
    header->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
        if (!dom.contains(pred)) { return; }
        if (!loop_blocks.contains(pred)) { entry_preds.emplace_back(pred); }
    });

    auto *preheader = def->create_basic_block();
    if (def->body_block() == header) { def->set_body_block(preheader); }
    for (auto *pred : entry_preds) {
        if (!pred->is_terminated()) { continue; }
        retarget_terminator(pred->terminator(), header, preheader);
    }
    {
        XIRBuilder b;
        b.set_insertion_point(preheader);
        b.br(header);
    }

    luisa::vector<std::pair<BasicBlock *, BasicBlock *>> exit_edges;
    for (auto *lb : loop_blocks) {
        if (!lb->is_terminated()) { continue; }
        lb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (!loop_blocks.contains(succ)) {
                exit_edges.emplace_back(lb, succ);
            }
        });
    }

    luisa::unordered_set<BasicBlock *> exit_targets_set;
    for (auto &[src, tgt] : exit_edges) { exit_targets_set.emplace(tgt); }
    luisa::vector<BasicBlock *> exit_targets{exit_targets_set.begin(), exit_targets_set.end()};

    auto *loop_merge = def->create_basic_block();

    auto *mod = def->parent_module();

    if (exit_targets.size() <= 1) {
        for (auto &[src, tgt] : exit_edges) {
            retarget_terminator(src->terminator(), tgt, loop_merge);
        }
        {
            XIRBuilder b;
            b.set_insertion_point(loop_merge);
            if (!exit_targets.empty()) {
                b.br(exit_targets[0]);
            } else {
                b.unreachable_();
            }
        }
    } else {
        auto *dispatch_merge = dispatch_merge_or_null;
        if (dispatch_merge == nullptr) {
            dispatch_merge = def->create_basic_block();
            XIRBuilder mb;
            mb.set_insertion_point(dispatch_merge);
            mb.unreachable_();
        }

        XIRBuilder b;
        auto *entry_bb = def->body_block();
        b.set_insertion_point(entry_bb->instructions().front());
        auto *keep_going = b.alloca_local(Type::of<bool>());
        auto *exit_sel = b.alloca_local(Type::of<uint32_t>());
        auto *true_const = mod->create_constant_one(Type::of<bool>());
        b.set_insertion_point(preheader);
        auto *preheader_br = preheader->terminator();
        preheader_br->remove_self();
        b.set_insertion_point(preheader);
        b.store(keep_going, true_const);
        b.br(header);

        uint32_t sel_id = 0;
        luisa::unordered_map<BasicBlock *, uint32_t> exit_target_id;
        luisa::vector<BasicBlock *> used_exit_targets;

        for (auto &[src, tgt] : exit_edges) {
            auto *stub = def->create_basic_block();
            auto changed = retarget_terminator(src->terminator(), tgt, stub);
            if (!changed) {
                stub->remove_self();
                continue;
            }
            auto id_it = exit_target_id.find(tgt);
            uint32_t id;
            if (id_it == exit_target_id.end()) {
                id = sel_id++;
                exit_target_id.emplace(tgt, id);
                used_exit_targets.emplace_back(tgt);
            } else {
                id = id_it->second;
            }
            auto *false_const = mod->create_constant_zero(Type::of<bool>());
            auto *id_const = mod->create_constant(Type::of<uint32_t>(), &id);
            b.set_insertion_point(stub);
            b.store(keep_going, false_const);
            b.store(exit_sel, id_const);
            b.br(loop_merge);
        }

        b.set_insertion_point(loop_merge);
        if (used_exit_targets.empty()) {
            b.unreachable_();
        } else if (used_exit_targets.size() == 1) {
            b.br(used_exit_targets[0]);
        } else {
            auto *loaded_sel = b.load(Type::of<uint32_t>(), exit_sel);
            auto *dispatch_bb = def->create_basic_block();
            b.br(dispatch_bb);

            b.set_insertion_point(dispatch_bb);
            auto *sw = b.switch_(loaded_sel);
            sw->set_merge_block(dispatch_merge);
            sw->set_default_block(used_exit_targets[0]);
            for (size_t i = 1; i < used_exit_targets.size(); i++) {
                auto *tgt = used_exit_targets[i];
                auto id = static_cast<SwitchInst::case_value_type>(exit_target_id[tgt]);
                sw->add_case(id, tgt);
            }
        }
    }

    if (canonical_latch->is_terminated()) {
        canonical_latch->terminator()->remove_self();
    }
    {
        XIRBuilder b;
        b.set_insertion_point(canonical_latch);
        b.br(header);
    }

    if (preheader->is_terminated()) {
        preheader->terminator()->remove_self();
    }

    BasicBlock *loop_body_succ = nullptr;
    if (header->is_terminated()) {
        auto *ht = header->terminator();
        if (ht->isa<ConditionalBranchInst>()) {
            auto *cb = static_cast<ConditionalBranchInst *>(ht);
            auto *tb = cb->true_block();
            auto *fb = cb->false_block();
            bool true_in_loop = loop_blocks.contains(tb);
            bool false_in_loop = loop_blocks.contains(fb);
            if (true_in_loop && fb == loop_merge) {
                loop_body_succ = tb;
            } else if (false_in_loop && tb == loop_merge) {
                loop_body_succ = fb;
            }
        }
    }

    {
        XIRBuilder b;
        b.set_insertion_point(preheader);
        if (loop_body_succ != nullptr && loop_body_succ != canonical_latch) {
            auto *li = b.loop();
            li->set_prepare_block(header);
            li->set_body_block(loop_body_succ);
            li->set_update_block(canonical_latch);
            li->set_merge_block(loop_merge);
        } else {
            auto *sl = b.simple_loop();
            sl->set_body_block(header);
            sl->set_merge_block(loop_merge);
        }
    }

    info.restructured_loop_count++;
    return true;
}

[[nodiscard]] static bool try_restructure_if_batch(FunctionDefinition *def,
                                                     const DomTree &dom,
                                                     const PostDomInfo &pdom,
                                                     RestructureCFGInfo &info) noexcept {
    // Collect merge blocks of already-structured loops.
    luisa::unordered_map<BasicBlock *, BasicBlock *> loop_merge_to_header;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        BasicBlock *merge = nullptr;
        if (term->isa<LoopInst>()) {
            merge = static_cast<LoopInst *>(term)->merge_block();
        } else if (term->isa<SimpleLoopInst>()) {
            merge = static_cast<SimpleLoopInst *>(term)->merge_block();
        }
        if (merge != nullptr) { loop_merge_to_header.emplace(merge, bb); }
    });

    struct Candidate {
        BasicBlock *header;
        ConditionalBranchInst *cbr;
        BasicBlock *merge;
        BasicBlock *enclosing_loop_continue;
        size_t depth;
    };
    luisa::vector<Candidate> candidates;

    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<ConditionalBranchInst>()) { return; }
        auto *cbr = static_cast<ConditionalBranchInst *>(term);
        auto *true_bb = cbr->true_block();
        auto *false_bb = cbr->false_block();
        if (true_bb == nullptr || false_bb == nullptr) { return; }
        if (true_bb == false_bb) { return; }

        auto ipm_it = pdom.ipostdom.find(bb);
        if (ipm_it == pdom.ipostdom.end() || ipm_it->second == nullptr) { return; }
        auto *merge = ipm_it->second;
        if (merge == pdom.virtual_exit) { return; }
        if (merge == bb) { return; }

        if (!dom.strictly_dominates(bb, true_bb)) { return; }
        if (!dom.strictly_dominates(bb, false_bb)) { return; }

        BasicBlock *enclosing_loop_continue = nullptr;
        if (auto it = loop_merge_to_header.find(merge); it != loop_merge_to_header.end()) {
            if (dom.dominates(it->second, bb)) {
                auto *loop_term = it->second->terminator();
                if (loop_term->isa<LoopInst>()) {
                    auto *li = static_cast<LoopInst *>(loop_term);
                    enclosing_loop_continue = li->update_block();
                    if (enclosing_loop_continue == nullptr) { enclosing_loop_continue = li->prepare_block(); }
                } else if (loop_term->isa<SimpleLoopInst>()) {
                    auto *sl = static_cast<SimpleLoopInst *>(loop_term);
                    enclosing_loop_continue = sl->body_block();
                }
            }
        }

        candidates.push_back({bb, cbr, merge, enclosing_loop_continue, dom_depth(dom, bb)});
    });

    if (candidates.empty()) { return false; }

    // Sort by depth descending (innermost first)
    luisa::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
        return a.depth > b.depth;
    });

    bool any = false;
    for (auto &cand : candidates) {
        auto *found_header = cand.header;
        auto *found_cbr = cand.cbr;
        auto *found_merge = cand.merge;
        auto *enclosing_loop_continue = cand.enclosing_loop_continue;

        // Skip if already restructured in this batch
        if (!found_header->is_terminated()) { continue; }
        if (!found_header->terminator()->isa<ConditionalBranchInst>()) { continue; }
        if (found_header->terminator() != found_cbr) { continue; }

        auto *true_bb = found_cbr->true_block();
        auto *false_bb = found_cbr->false_block();
        auto *cond = found_cbr->condition();

        BasicBlock *structural_merge = nullptr;
        if (enclosing_loop_continue != nullptr &&
            (true_bb == found_merge || false_bb == found_merge)) {
            structural_merge = found_merge;
        } else {
            structural_merge = def->create_basic_block();
            {
                XIRBuilder mb;
                mb.set_insertion_point(structural_merge);
                mb.br(found_merge);
            }
            {
                // Walk the dominator-tree subtree rooted at found_header instead of
                // scanning the entire CFG. Every block dominated by found_header is
                // exactly the descendant subtree in the dom tree.
                auto header_node = dom.node_or_null(found_header);
                if (header_node != nullptr) {
                    luisa::vector<const DomTreeNode *> work;
                    work.push_back(header_node);
                    while (!work.empty()) {
                        auto *node = work.back();
                        work.pop_back();
                        auto *bb = node->block();
                        if (bb != structural_merge && bb->is_terminated()) {
                            retarget_terminator(bb->terminator(), found_merge, structural_merge);
                        }
                        for (auto *child : node->children()) {
                            work.push_back(child);
                        }
                    }
                }
            }
        }

        if (true_bb == found_merge) { true_bb = structural_merge; }
        if (false_bb == found_merge) { false_bb = structural_merge; }

        found_cbr->remove_self();

        XIRBuilder b;
        b.set_insertion_point(found_header);
        auto *if_inst = b.if_(cond);
        if_inst->set_true_target(true_bb);
        if_inst->set_false_target(false_bb);
        if_inst->set_merge_block(structural_merge);

        for (auto *arm_bb : {true_bb, false_bb}) {
            if (arm_bb == nullptr) { continue; }
            if (!arm_bb->is_terminated()) { continue; }
            luisa::vector<BasicBlock *> bad_succs;
            arm_bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (succ == structural_merge) { return; }
                if (!dom.contains(succ)) { return; }
                if (dom.dominates(found_header, succ)) { return; }
                bad_succs.emplace_back(succ);
            });
            luisa::sort(bad_succs.begin(), bad_succs.end());
            bad_succs.erase(std::unique(bad_succs.begin(), bad_succs.end()), bad_succs.end());
            for (auto *succ : bad_succs) {
                clone_subgraph_to_target(def, succ, arm_bb, found_merge, structural_merge);
            }
        }

        info.restructured_if_count++;
        any = true;
    }
    return any;
}

[[nodiscard]] static bool try_restructure_if(FunctionDefinition *def,
                                             const DomTree &dom,
                                             const PostDomInfo &pdom,
                                             RestructureCFGInfo &info) noexcept {
    // Collect merge blocks of already-structured loops. A cbr whose postdom is
    // a loop merge must NOT be restructured as an if-construct: doing so would
    // make the loop body flow to the if's structural_merge (which exits the
    // loop) instead of the loop's continue block.
    luisa::unordered_map<BasicBlock *, BasicBlock *> loop_merge_to_header;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        BasicBlock *merge = nullptr;
        if (term->isa<LoopInst>()) {
            merge = static_cast<LoopInst *>(term)->merge_block();
        } else if (term->isa<SimpleLoopInst>()) {
            merge = static_cast<SimpleLoopInst *>(term)->merge_block();
        }
        if (merge != nullptr) { loop_merge_to_header.emplace(merge, bb); }
    });

    BasicBlock *found_header = nullptr;
    ConditionalBranchInst *found_cbr = nullptr;
    BasicBlock *found_merge = nullptr;
    BasicBlock *enclosing_loop_continue = nullptr;// non-null when postdom is a loop merge

    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<ConditionalBranchInst>()) { return; }
        auto *cbr = static_cast<ConditionalBranchInst *>(term);
        auto *true_bb = cbr->true_block();
        auto *false_bb = cbr->false_block();
        if (true_bb == nullptr || false_bb == nullptr) { return; }
        if (true_bb == false_bb) { return; }

        auto ipm_it = pdom.ipostdom.find(bb);
        if (ipm_it == pdom.ipostdom.end() || ipm_it->second == nullptr) { return; }
        auto *merge = ipm_it->second;
        if (merge == pdom.virtual_exit) { return; }
        if (merge == bb) { return; }

        if (!dom.strictly_dominates(bb, true_bb)) { return; }
        if (!dom.strictly_dominates(bb, false_bb)) { return; }

        // Prefer innermost: choose the candidate dominated by all other candidates,
        // i.e., the deepest in the dominator tree. Picking innermost first ensures
        // that when we synthesize the structural merge and retarget in-construct
        // edges, no still-unstructured nested cbr is left with edges to the
        // newly-created merge (which would later force it to skip its own merge).
        if (found_header == nullptr || dom.strictly_dominates(found_header, bb)) {
            found_header = bb;
            found_cbr = cbr;
            found_merge = merge;
            // Detect if this cbr's postdom is an enclosing loop's merge.
            enclosing_loop_continue = nullptr;
            if (auto it = loop_merge_to_header.find(merge); it != loop_merge_to_header.end()) {
                if (dom.dominates(it->second, bb)) {
                    auto *loop_term = it->second->terminator();
                    if (loop_term->isa<LoopInst>()) {
                        auto *li = static_cast<LoopInst *>(loop_term);
                        enclosing_loop_continue = li->update_block();
                        if (enclosing_loop_continue == nullptr) { enclosing_loop_continue = li->prepare_block(); }
                    } else if (loop_term->isa<SimpleLoopInst>()) {
                        auto *sl = static_cast<SimpleLoopInst *>(loop_term);
                        enclosing_loop_continue = sl->body_block();
                    }
                }
            }
        }
    });

    if (found_header == nullptr) { return false; }

    auto *true_bb = found_cbr->true_block();
    auto *false_bb = found_cbr->false_block();
    auto *cond = found_cbr->condition();

    // Conservatively synthesize a fresh structural merge block for this if-construct.
    // This guarantees:
    //   1. Merge uniqueness across all constructs (each construct owns its merge).
    //   2. No collision with reserved loop blocks (continue/update/header).
    //   3. The original postdom (`found_merge`) keeps its semantics (phi-loads from
    //      reg2mem, etc.) and gets a single predecessor (the fresh merge).
    //
    // When the postdom is an enclosing loop's merge AND one arm directly targets
    // the loop merge (loop condition check), use the loop merge directly as the
    // if's merge. The body arm flows to continue naturally without reaching the
    // if merge, which is valid in SPIR-V (selection inside loop).
    // For other cbrs inside the loop (if-else with break), also use the loop merge
    // directly — the break arm reaches the merge, the non-break arm flows to continue.
    auto *structural_merge = def->create_basic_block();
    {
        XIRBuilder mb;
        mb.set_insertion_point(structural_merge);
        mb.br(found_merge);
    }
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb == structural_merge) { return; }
        if (!bb->is_terminated()) { return; }
        if (!dom.contains(bb)) { return; }
        if (!dom.dominates(found_header, bb)) { return; }
        retarget_terminator(bb->terminator(), found_merge, structural_merge);
    });

    // If an if-arm directly targeted found_merge (empty arm), it must now target
    // structural_merge instead. The retarget loop above already handled the cbr's
    // successors, but we captured true_bb/false_bb before retargeting.
    if (true_bb == found_merge) { true_bb = structural_merge; }
    if (false_bb == found_merge) { false_bb = structural_merge; }

    found_cbr->remove_self();

    XIRBuilder b;
    b.set_insertion_point(found_header);
    auto *if_inst = b.if_(cond);
    if_inst->set_true_target(true_bb);
    if_inst->set_false_target(false_bb);
    if_inst->set_merge_block(structural_merge);

    // If an arm branches to a block that is not dominated by the header and not the
    // structural merge, it is a shared tail from outside the construct. SPIR-V requires
    // every block inside a selection to branch only to other in-construct blocks or the
    // merge. Clone the shared tail so the arm has its own copy ending at the merge.
    for (auto *arm_bb : {true_bb, false_bb}) {
        if (arm_bb == nullptr) { continue; }
        if (!arm_bb->is_terminated()) { continue; }
        // Collect bad successors first to avoid UB from mutating the terminator
        // (which modifies operand_uses()) while traversing it.
        luisa::vector<BasicBlock *> bad_succs;
        arm_bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (succ == structural_merge) { return; }
            if (!dom.contains(succ)) { return; }
            if (dom.dominates(found_header, succ)) { return; }
            bad_succs.emplace_back(succ);
        });
        // Deduplicate: a terminator may have multiple edges to the same successor
        // (e.g. a conditional branch where both arms target the same block).
        luisa::sort(bad_succs.begin(), bad_succs.end());
        bad_succs.erase(std::unique(bad_succs.begin(), bad_succs.end()), bad_succs.end());
        for (auto *succ : bad_succs) {
            clone_subgraph_to_target(def, succ, arm_bb, found_merge, structural_merge);
        }
    }

    info.restructured_if_count++;
    return true;
}

// Collect the entry blocks of a structured construct C whose header is `header_bb`.
// "Entry blocks" are blocks that should only be reachable from the header (or from
// authorized internal back-edges, e.g. the update block of a loop), and NEVER from
// sibling arms. Returns nullptr-free, possibly-duplicate-free list.
static void collect_construct_entries(BasicBlock *header_bb,
                                      luisa::vector<BasicBlock *> &entries) noexcept {
    entries.clear();
    auto *term = header_bb->terminator();
    if (term == nullptr) { return; }
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::IF: {
            auto *ii = static_cast<IfInst *>(term);
            if (ii->true_block() != nullptr) { entries.emplace_back(ii->true_block()); }
            if (ii->false_block() != nullptr) { entries.emplace_back(ii->false_block()); }
            break;
        }
        case DerivedInstructionTag::SWITCH: {
            auto *sw = static_cast<SwitchInst *>(term);
            for (size_t i = 0; i < sw->case_count(); i++) {
                if (auto *cb = sw->case_block(i); cb != nullptr) { entries.emplace_back(cb); }
            }
            if (sw->default_block() != nullptr) { entries.emplace_back(sw->default_block()); }
            break;
        }
        case DerivedInstructionTag::LOOP: {
            auto *lp = static_cast<LoopInst *>(term);
            if (lp->prepare_block() != nullptr) { entries.emplace_back(lp->prepare_block()); }
            // body/update are loop-internal; they may legitimately have multiple preds.
            break;
        }
        case DerivedInstructionTag::SIMPLE_LOOP: {
            auto *sl = static_cast<SimpleLoopInst *>(term);
            if (sl->body_block() != nullptr) { entries.emplace_back(sl->body_block()); }
            break;
        }
        default: break;
    }
}

// Resolver for Instruction::clone: maps any value in our remap table to the cloned
// version; otherwise returns the original value (constants, args, globals, allocas,
// instructions defined outside the cloned region, frontier BBs).
struct CloneRemap final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> map;
    Value *resolve(const Value *v) noexcept override {
        if (auto it = map.find(v); it != map.end()) { return it->second; }
        return const_cast<Value *>(v);
    }
};

// For a construct C with header H and one of its entries E, decide whether predecessor
// P of E is "authorized" per the XIR invariant.
[[nodiscard]] static bool is_authorized_construct_pred(Instruction *header_term,
                                                       BasicBlock * /*entry*/,
                                                       BasicBlock *header_bb,
                                                       BasicBlock *pred) noexcept {
    if (pred == header_bb) { return true; }
    if (header_term == nullptr) { return false; }
    switch (header_term->derived_instruction_tag()) {
        case DerivedInstructionTag::LOOP: {
            auto *lp = static_cast<LoopInst *>(header_term);
            if (pred == lp->update_block()) { return true; }
            if (pred == lp->body_block()) { return true; }
            return false;
        }
        case DerivedInstructionTag::SIMPLE_LOOP: {
            auto *sl = static_cast<SimpleLoopInst *>(header_term);
            if (pred == sl->body_block()) { return true; }
            return false;
        }
        default: break;
    }
    return false;
}

// Decide if a block S is on the "frontier" of the clone region rooted at E within
// construct C (with header H). Frontier blocks are NOT cloned; edges into them from
// cloned blocks remain pointing at the original block.
[[nodiscard]] static bool is_clone_boundary(BasicBlock *S, BasicBlock *E,
                                            BasicBlock *header_bb,
                                            luisa::span<BasicBlock *const> entries,
                                            BasicBlock *merge_bb,
                                            const DomTree &dom) noexcept {
    if (S == nullptr) { return true; }
    if (S == header_bb) { return true; }
    if (S == merge_bb) { return true; }
    for (auto *en : entries) {
        if (en == S && en != E) { return true; }
    }
    // S must be dominated by E to belong to E's owned subgraph.
    if (!dom.dominates(E, S)) { return true; }
    return false;
}

// Walk forward from E, collecting all blocks owned by E that are not boundary.
// Blocks are recorded in deterministic DFS discovery order in `ordered`.
static void collect_owned_region(BasicBlock *E, BasicBlock *header_bb,
                                 luisa::span<BasicBlock *const> entries,
                                 BasicBlock *merge_bb, const DomTree &dom,
                                 luisa::unordered_set<BasicBlock *> &region,
                                 luisa::vector<BasicBlock *> &ordered) noexcept {
    region.clear();
    ordered.clear();
    luisa::vector<BasicBlock *> work;
    work.emplace_back(E);
    while (!work.empty()) {
        auto *B = work.back();
        work.pop_back();
        if (region.contains(B)) { continue; }
        if (is_clone_boundary(B, E, header_bb, entries, merge_bb, dom)) { continue; }
        region.emplace(B);
        ordered.emplace_back(B);
        B->traverse_successors(false, [&](BasicBlock *S) noexcept {
            if (!is_clone_boundary(S, E, header_bb, entries, merge_bb, dom) &&
                !region.contains(S)) {
                work.emplace_back(S);
            }
        });
    }
}

// Clone the owned subgraph rooted at E. P (with its terminator) is rerouted via a
// fresh relay block to the clone of E. Returns true on success.
[[nodiscard]] static bool clone_owned_subgraph_for_edge(FunctionDefinition *def,
                                                        BasicBlock *header_bb,
                                                        BasicBlock *E, BasicBlock *P,
                                                        luisa::span<BasicBlock *const> entries,
                                                        BasicBlock *merge_bb,
                                                        const DomTree &dom) noexcept {
    luisa::unordered_set<BasicBlock *> region;
    luisa::vector<BasicBlock *> ordered;
    collect_owned_region(E, header_bb, entries, merge_bb, dom, region, ordered);
    if (region.empty()) { return false; }

    // Pre-create cloned BBs in deterministic order.
    CloneRemap remap;
    for (auto *B : ordered) {
        auto *NB = def->create_basic_block();
        remap.map[B] = NB;
    }

    // Clone instructions of each region block into its counterpart.
    XIRBuilder builder;
    for (auto *old_bb : ordered) {
        auto *new_bb = static_cast<BasicBlock *>(remap.map[old_bb]);
        builder.set_insertion_point(new_bb);
        for (auto *old_inst : old_bb->instructions()) {
            auto *new_inst = old_inst->clone(builder, remap);
            if (old_inst->type() != nullptr) {
                remap.map[old_inst] = new_inst;
            }
        }
    }

    // Create a relay block: P -> relay -> clone(E). Branching through a relay (rather
    // than redirecting P directly to clone(E)) guarantees the clone's entry has a
    // single predecessor regardless of how many bad edges from P there are.
    auto *clone_E = static_cast<BasicBlock *>(remap.map[E]);
    auto *relay = def->create_basic_block();
    {
        XIRBuilder rb;
        rb.set_insertion_point(relay);
        rb.br(clone_E);
    }
    // Reroute every edge in P's terminator that targeted E to instead target relay.
    retarget_terminator(P->terminator(), E, relay);
    return true;
}

// Clone all blocks reachable from E until reaching target (exclusive).
// Any terminators in the cloned region that point to target are retargeted to new_target.
// P's terminator edge to E is rerouted through a fresh relay to clone(E).
[[nodiscard]] static bool clone_subgraph_to_target(FunctionDefinition *def,
                                                    BasicBlock *E, BasicBlock *P,
                                                    BasicBlock *target,
                                                    BasicBlock *new_target) noexcept {
    // Defensive: all parameters must be non-null.
    if (E == nullptr || P == nullptr || target == nullptr || new_target == nullptr) {
        return false;
    }

    // Collect region: all blocks reachable from E, stopping at target.
    // Every successor of a region block is either in region or == target (because
    // the BFS pushes every non-target successor). Cloned terminators therefore only
    // reference region blocks (resolved to their clones) or target (retargeted below).
    luisa::unordered_set<BasicBlock *> region;
    luisa::vector<BasicBlock *> ordered;
    luisa::vector<BasicBlock *> work{E};
    while (!work.empty()) {
        auto *B = work.back();
        work.pop_back();
        if (region.contains(B)) { continue; }
        if (B == target) { continue; }
        region.emplace(B);
        ordered.emplace_back(B);
        if (!B->is_terminated()) { continue; }
        B->traverse_successors(false, [&](BasicBlock *S) noexcept {
            if (!region.contains(S) && S != target) {
                work.emplace_back(S);
            }
        });
    }
    if (region.empty()) { return false; }
    LUISA_DEBUG_ASSERT(!region.contains(target), "target must not be inside the cloned region");

    // Pre-create cloned BBs in deterministic order.
    CloneRemap remap;
    for (auto *B : ordered) {
        auto *NB = def->create_basic_block();
        remap.map[B] = NB;
    }

    // Clone instructions of each region block into its counterpart.
    XIRBuilder builder;
    for (auto *old_bb : ordered) {
        auto *new_bb = static_cast<BasicBlock *>(remap.map[old_bb]);
        builder.set_insertion_point(new_bb);
        for (auto *old_inst : old_bb->instructions()) {
            auto *new_inst = old_inst->clone(builder, remap);
            if (old_inst->type() != nullptr) {
                remap.map[old_inst] = new_inst;
            }
        }
    }

    // Retarget cloned blocks that branch to target to new_target instead.
    for (auto *old_bb : ordered) {
        auto *new_bb = static_cast<BasicBlock *>(remap.map[old_bb]);
        if (!new_bb->is_terminated()) { continue; }
        retarget_terminator(new_bb->terminator(), target, new_target);
    }

    // Create a relay block: P -> relay -> clone(E).
    auto *clone_E = static_cast<BasicBlock *>(remap.map[E]);
    auto *relay = def->create_basic_block();
    {
        XIRBuilder rb;
        rb.set_insertion_point(relay);
        rb.br(clone_E);
    }
    retarget_terminator(P->terminator(), E, relay);
    return true;
}

// Per-construct entry-uniqueness fix. Returns true if any edges were rewritten.
[[nodiscard]] static bool enforce_construct_entries(FunctionDefinition *def,
                                                    BasicBlock *header_bb,
                                                    BasicBlock *merge_bb) noexcept {
    luisa::vector<BasicBlock *> entries;
    collect_construct_entries(header_bb, entries);
    if (entries.size() <= 1) { return false; }
    bool changed_any = false;
    // Iterate entries in their natural order; per Oracle's design, if the sibling-entry
    // graph is acyclic, fixing earlier entries does not create new bad edges into them.
    // We bound the inner loop to defend against malformed CFGs.
    for (auto *E : entries) {
        size_t guard = 64;
        // Defer dom-tree computation until we know there are offenders.
        // Recompute only after a successful clone invalidates the old tree.
        DomTree dom;
        bool dom_valid = false;
        while (guard-- > 0) {
            luisa::vector<BasicBlock *> offenders;
            E->traverse_predecessors(false, [&](BasicBlock *P) noexcept {
                if (!is_authorized_construct_pred(header_bb->terminator(), E, header_bb, P)) {
                    offenders.emplace_back(P);
                }
            });
            if (offenders.empty()) { break; }
            if (!dom_valid) {
                dom = compute_dom_tree(def);
                dom_valid = true;
            }
            bool local_change = false;
            for (auto *P : offenders) {
                if (clone_owned_subgraph_for_edge(def, header_bb, E, P,
                                                  luisa::span<BasicBlock *const>{entries},
                                                  merge_bb, dom)) {
                    local_change = true;
                }
            }
            if (!local_change) { break; }
            changed_any = true;
            // The CFG was modified; the dom tree is now stale.
            dom_valid = false;
        }
    }
    return changed_any;
}

// Visit each structured construct (If/Switch/Loop/SimpleLoop) and enforce the
// invariant. We rescan after each change because the BB list has grown.
static void enforce_unique_construct_entries(FunctionDefinition *def) noexcept {
    size_t outer_guard = 64;
    while (outer_guard-- > 0) {
        bool changed = false;
        luisa::vector<std::pair<BasicBlock *, BasicBlock *>> construct_sites;// header_bb, merge_bb
        def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
            auto *t = bb->terminator();
            if (t == nullptr) { return; }
            BasicBlock *merge_bb = nullptr;
            if (auto *cm = t->control_flow_merge(); cm != nullptr) {
                merge_bb = cm->merge_block();
            }
            switch (t->derived_instruction_tag()) {
                case DerivedInstructionTag::IF:
                case DerivedInstructionTag::SWITCH:
                case DerivedInstructionTag::LOOP:
                case DerivedInstructionTag::SIMPLE_LOOP:
                    construct_sites.emplace_back(bb, merge_bb);
                    break;
                default: break;
            }
        });
        for (auto &[hbb, mbb] : construct_sites) {
            if (enforce_construct_entries(def, hbb, mbb)) {
                changed = true;
                break;// restart outer loop: BB list and dominance changed
            }
        }
        if (!changed) { break; }
    }
}

[[nodiscard]] static RestructureCFGInfo restructure_cfg_on_definition(FunctionDefinition *def) noexcept {
    check_phi_free(def);
    RestructureCFGInfo info{};
    size_t max_iters = 10000;
    while (max_iters-- > 0) {
        auto dom = compute_dom_tree(def);
        auto pdom = compute_post_dom(def);
        if (try_restructure_loop(def, dom, pdom, info)) { continue; }
        if (try_restructure_if_batch(def, dom, pdom, info)) {
            // Fast path: if no conditional branches remain, we can skip the
            // expensive dom/pdom recomputation and break out early.
            bool has_cbr = false;
            def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
                if (has_cbr) { return; }
                if (bb->is_terminated()) {
                    if (bb->terminator()->isa<ConditionalBranchInst>()) {
                        has_cbr = true;
                    }
                }
            });
            if (!has_cbr) { break; }
            continue;
        }
        break;
    }
    enforce_unique_construct_entries(def);
    return info;
}

}// namespace

RestructureCFGInfo restructure_cfg_pass_run_on_function(Function *function) noexcept {
    if (function == nullptr) { return {}; }
    auto *def = function->definition();
    if (def == nullptr) { return {}; }
    return restructure_cfg_on_definition(def);
}

RestructureCFGInfo restructure_cfg_pass_run_on_module(Module *module, PassReport *report) noexcept {
    RestructureCFGInfo total{};
    for (auto *f : module->function_list()) {
        auto info = restructure_cfg_pass_run_on_function(f);
        total.restructured_loop_count += info.restructured_loop_count;
        total.restructured_if_count += info.restructured_if_count;
        total.irreducible_region_count += info.irreducible_region_count;
    }
    if (report != nullptr) {
        report->set("restructured_loop", total.restructured_loop_count);
        report->set("restructured_if", total.restructured_if_count);
        report->set("irreducible_region", total.irreducible_region_count);
    }
    return total;
}

}// namespace luisa::compute::xir
