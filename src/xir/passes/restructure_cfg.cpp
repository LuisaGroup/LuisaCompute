#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/restructure_cfg.h>

namespace luisa::compute::xir {

namespace {

struct PostDomTree {
    luisa::unordered_map<BasicBlock *, BasicBlock *> ipostdom;
};

[[nodiscard]] PostDomTree compute_post_dom_tree(FunctionDefinition *def) noexcept {
    luisa::vector<BasicBlock *> all_blocks;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        all_blocks.emplace_back(bb);
    });

    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> reverse_succs;
    for (auto bb : all_blocks) {
        bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            reverse_succs[succ].emplace_back(bb);
        });
    }

    luisa::vector<BasicBlock *> sinks;
    for (auto bb : all_blocks) {
        bool has_succ = false;
        bb->traverse_successors(false, [&](BasicBlock *) noexcept { has_succ = true; });
        if (!has_succ) { sinks.emplace_back(bb); }
    }

    luisa::unordered_map<BasicBlock *, size_t> rpo_index;
    luisa::vector<BasicBlock *> rpo;

    {
        luisa::unordered_set<BasicBlock *> visited;
        luisa::vector<std::pair<BasicBlock *, size_t>> dfs_stack;
        for (auto sink : sinks) {
            if (!visited.contains(sink)) {
                visited.emplace(sink);
                dfs_stack.emplace_back(sink, 0u);
            }
        }
        while (!dfs_stack.empty()) {
            auto bb = dfs_stack.back().first;
            auto &preds = reverse_succs[bb];
            auto cur_idx = dfs_stack.back().second;
            if (cur_idx < preds.size()) {
                dfs_stack.back().second++;
                auto pred = preds[cur_idx];
                if (!visited.contains(pred)) {
                    visited.emplace(pred);
                    dfs_stack.emplace_back(pred, 0u);
                }
            } else {
                rpo.emplace_back(bb);
                dfs_stack.pop_back();
            }
        }
    }

    for (size_t i = 0; i < rpo.size(); i++) {
        rpo_index[rpo[i]] = i;
    }
    rpo_index[nullptr] = SIZE_MAX;

    PostDomTree result;
    auto &ipostdom = result.ipostdom;
    for (auto bb : rpo) { ipostdom[bb] = nullptr; }

    luisa::unordered_set<BasicBlock *> processed;
    for (auto sink : sinks) { processed.emplace(sink); }

    auto intersect = [&](BasicBlock *b1, BasicBlock *b2) noexcept -> BasicBlock * {
        if (b1 == nullptr) { return b2; }
        if (b2 == nullptr) { return b1; }
        auto finger1 = b1;
        auto finger2 = b2;
        while (finger1 != finger2) {
            while (rpo_index[finger1] < rpo_index[finger2]) {
                finger1 = ipostdom[finger1];
                if (finger1 == nullptr) { return nullptr; }
            }
            while (rpo_index[finger2] < rpo_index[finger1]) {
                finger2 = ipostdom[finger2];
                if (finger2 == nullptr) { return nullptr; }
            }
        }
        return finger1;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = rpo.rbegin(); it != rpo.rend(); ++it) {
            auto bb = *it;
            bool is_sink = false;
            for (auto s : sinks) {
                if (s == bb) { is_sink = true; break; }
            }
            if (is_sink) { continue; }

            BasicBlock *new_ipostdom = nullptr;
            bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (processed.contains(succ)) {
                    new_ipostdom = intersect(new_ipostdom, succ);
                }
            });
            if (ipostdom[bb] != new_ipostdom) {
                ipostdom[bb] = new_ipostdom;
                changed = true;
            }
            if (new_ipostdom != nullptr) { processed.emplace(bb); }
        }
    }

    return result;
}

[[nodiscard]] bool try_restructure_if(FunctionDefinition *def,
                                      const DomTree &dom,
                                      const PostDomTree &pdom,
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

[[nodiscard]] bool try_restructure_loop(FunctionDefinition *def,
                                        const DomTree &dom,
                                        const PostDomTree &pdom,
                                        RestructureCFGInfo &info) noexcept {
    BasicBlock *found_header = nullptr;
    BasicBlock *found_latch = nullptr;
    BasicBlock *found_merge = nullptr;

    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (found_header != nullptr) { return; }
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<BranchInst>()) { return; }
        auto *br = static_cast<BranchInst *>(term);
        auto *target = br->target_block();
        if (target == nullptr) { return; }

        if (!dom.strictly_dominates(target, bb)) { return; }

        auto *header = target;

        bool already_structured = false;
        header->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
            if (!pred->is_terminated()) { return; }
            auto *t2 = pred->terminator();
            if (t2->isa<SimpleLoopInst>()) {
                auto *sl2 = static_cast<SimpleLoopInst *>(t2);
                if (sl2->body_block() == header) { already_structured = true; }
            }
        });
        if (already_structured) { return; }

        luisa::vector<BasicBlock *> latches;
        def->traverse_basic_blocks([&](BasicBlock *inner) noexcept {
            if (!dom.dominates(header, inner)) { return; }
            inner->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (succ == header) { latches.emplace_back(inner); }
            });
        });

        if (latches.size() != 1u) { return; }
        if (latches[0] != bb) { return; }

        auto pdom_it = pdom.ipostdom.find(header);
        if (pdom_it == pdom.ipostdom.end() || pdom_it->second == nullptr) { return; }
        auto *merge = pdom_it->second;
        if (merge == header) { return; }

        found_header = header;
        found_latch = bb;
        found_merge = merge;
    });

    if (found_header == nullptr) { return false; }

    luisa::vector<BasicBlock *> entry_preds;
    found_header->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
        if (pred != found_latch) { entry_preds.emplace_back(pred); }
    });

    auto *pre_header = def->create_basic_block();

    for (auto *pred : entry_preds) {
        if (!pred->is_terminated()) { continue; }
        auto *t = pred->terminator();
        if (t->isa<BranchInst>()) {
            static_cast<BranchInst *>(t)->set_target_block(pre_header);
        } else if (t->isa<ConditionalBranchInst>()) {
            auto *cbr = static_cast<ConditionalBranchInst *>(t);
            if (cbr->true_block() == found_header) { cbr->set_true_target(pre_header); }
            if (cbr->false_block() == found_header) { cbr->set_false_target(pre_header); }
        }
    }

    found_latch->terminator()->remove_self();

    XIRBuilder b;
    b.set_insertion_point(pre_header);
    auto *sl = b.simple_loop();
    sl->set_body_block(found_header);
    sl->set_merge_block(found_merge);

    b.set_insertion_point(found_latch);
    b.br(found_header);

    info.restructured_loop_count++;
    return true;
}

[[nodiscard]] RestructureCFGInfo restructure_cfg_on_definition(FunctionDefinition *def) noexcept {
    RestructureCFGInfo info{};
    size_t max_iters = 10000u;
    while (max_iters-- > 0u) {
        auto dom = compute_dom_tree(def);
        auto pdom = compute_post_dom_tree(def);
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
