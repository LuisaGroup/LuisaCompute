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
        stack.emplace_back(virt, 0u);
        while (!stack.empty()) {
            auto *cur = stack.back().first;
            auto &idx = stack.back().second;
            auto &preds = aug_pred_map[cur];
            if (idx < preds.size()) {
                auto *pred = preds[idx++];
                if (!visited.contains(pred)) {
                    visited.emplace(pred);
                    stack.emplace_back(pred, 0u);
                }
            } else {
                rpo.emplace_back(cur);
                stack.pop_back();
            }
        }
    }

    luisa::unordered_map<BasicBlock *, size_t> rpo_index;
    for (size_t i = 0u; i < rpo.size(); i++) { rpo_index[rpo[i]] = i; }
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
    size_t d = 0u;
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
    for (size_t i = 1u; i < blocks.size(); i++) {
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

static void retarget_terminator(Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
    if (term == nullptr) { return; }
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH: {
            auto *br = static_cast<BranchInst *>(term);
            if (br->target_block() == from) { br->set_target_block(to); }
            break;
        }
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto *cb = static_cast<ConditionalBranchInst *>(term);
            if (cb->true_block() == from) { cb->set_true_target(to); }
            if (cb->false_block() == from) { cb->set_false_target(to); }
            break;
        }
        default: break;
    }
}

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
        size_t depth{0u};
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
            if (dom.dominates(cb->true_block(), bb)) { back_target = cb->true_block(); }
            else if (dom.dominates(cb->false_block(), bb)) { back_target = cb->false_block(); }
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
    if (pre_exit_targets.size() > 1u) {
        dispatch_merge_or_null = common_postdom(pdom, luisa::span<BasicBlock *const>{pre_exit_targets});
        if (dispatch_merge_or_null == pdom.virtual_exit) {
            dispatch_merge_or_null = nullptr;
        }
    }

    BasicBlock *canonical_latch = nullptr;
    if (latches.size() == 1u) {
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

    if (exit_targets.size() <= 1u) {
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

        uint32_t sel_id = 0u;
        luisa::unordered_map<BasicBlock *, uint32_t> exit_target_id;
        for (auto *tgt : exit_targets) { exit_target_id[tgt] = sel_id++; }

        for (auto &[src, tgt] : exit_edges) {
            auto *stub = def->create_basic_block();
            retarget_terminator(src->terminator(), tgt, stub);
            auto *false_const = mod->create_constant_zero(Type::of<bool>());
            uint32_t id = exit_target_id[tgt];
            auto *id_const = mod->create_constant(Type::of<uint32_t>(), &id);
            b.set_insertion_point(stub);
            b.store(keep_going, false_const);
            b.store(exit_sel, id_const);
            b.br(loop_merge);
        }

        b.set_insertion_point(loop_merge);
        auto *loaded_sel = b.load(Type::of<uint32_t>(), exit_sel);
        auto *dispatch_bb = def->create_basic_block();
        b.br(dispatch_bb);

        b.set_insertion_point(dispatch_bb);
        auto *sw = b.switch_(loaded_sel);
        sw->set_merge_block(dispatch_merge);
        sw->set_default_block(exit_targets[0]);
        for (size_t i = 1u; i < exit_targets.size(); i++) {
            auto *tgt = exit_targets[i];
            auto id = static_cast<SwitchInst::case_value_type>(exit_target_id[tgt]);
            sw->add_case(id, tgt);
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

[[nodiscard]] static bool try_restructure_if(FunctionDefinition *def,
                                             const DomTree &dom,
                                             const PostDomInfo &pdom,
                                             RestructureCFGInfo &info) noexcept {
    BasicBlock *found_header = nullptr;
    ConditionalBranchInst *found_cbr = nullptr;
    BasicBlock *found_merge = nullptr;

    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (found_header != nullptr) { return; }
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

        found_header = bb;
        found_cbr = cbr;
        found_merge = merge;
    });

    if (found_header == nullptr) { return false; }

    auto *true_bb = found_cbr->true_block();
    auto *false_bb = found_cbr->false_block();
    auto *cond = found_cbr->condition();

    found_cbr->remove_self();

    XIRBuilder b;
    b.set_insertion_point(found_header);
    auto *if_inst = b.if_(cond);
    if_inst->set_true_target(true_bb);
    if_inst->set_false_target(false_bb);
    if_inst->set_merge_block(found_merge);

    info.restructured_if_count++;
    return true;
}

[[nodiscard]] static RestructureCFGInfo restructure_cfg_on_definition(FunctionDefinition *def) noexcept {
    check_phi_free(def);
    RestructureCFGInfo info{};
    size_t max_iters = 10000u;
    while (max_iters-- > 0u) {
        auto dom = compute_dom_tree(def);
        auto pdom = compute_post_dom(def);
        if (try_restructure_loop(def, dom, pdom, info)) { continue; }
        if (try_restructure_if(def, dom, pdom, info)) { continue; }
        break;
    }
    return info;
}

}// namespace

RestructureCFGInfo restructure_cfg_pass_run_on_function(Function *function) noexcept {
    if (function == nullptr) { return {}; }
    auto *def = function->definition();
    if (def == nullptr) { return {}; }
    return restructure_cfg_on_definition(def);
}

RestructureCFGInfo restructure_cfg_pass_run_on_module(Module *module) noexcept {
    RestructureCFGInfo total{};
    for (auto *f : module->function_list()) {
        auto info = restructure_cfg_pass_run_on_function(f);
        total.restructured_loop_count += info.restructured_loop_count;
        total.restructured_if_count += info.restructured_if_count;
        total.irreducible_region_count += info.irreducible_region_count;
    }
    return total;
}

}// namespace luisa::compute::xir
