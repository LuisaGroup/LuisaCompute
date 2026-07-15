#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
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
#include <luisa/xir/passes/convergence_region.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace {

struct ScopedTimer {
#ifndef NDEBUG
    Clock clock;
    const char *name;
#endif
    ScopedTimer(const char *n) noexcept
#ifndef NDEBUG
        : name(n)
#endif
    {
    }
    ~ScopedTimer() noexcept {
#ifndef NDEBUG
        auto ms = clock.toc();
        LUISA_VERBOSE_WITH_LOCATION("[restructure_cfg] {}: {:.3f} ms", name, ms);
#endif
    }
};

void check_phi_free(FunctionDefinition *def) noexcept {
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        for (auto *inst : bb->instructions()) {
            if (inst->isa<PhiInst>()) {
                LUISA_ERROR_WITH_LOCATION("restructure_cfg requires phi-free input; run reg2mem_pass first");
            }
        }
    });
}

// Return the number of cyclic SCCs with more than one entry block. Such a
// region cannot be represented by XIR's structured loop form without node
// splitting. Detect it before the restructuring pipeline mutates anything so
// failure is atomic and callers can choose a dedicated irreducible-CFG lowering.
[[nodiscard]] size_t count_irreducible_regions(FunctionDefinition *def) noexcept {
    luisa::vector<BasicBlock *> blocks;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept { blocks.emplace_back(bb); });
    if (blocks.empty()) { return 0u; }

    luisa::unordered_map<BasicBlock *, size_t> block_index;
    block_index.reserve(blocks.size());
    for (size_t i = 0u; i < blocks.size(); ++i) { block_index.emplace(blocks[i], i); }

    luisa::vector<luisa::vector<size_t>> successors(blocks.size());
    luisa::vector<luisa::vector<size_t>> predecessors(blocks.size());
    for (size_t i = 0u; i < blocks.size(); ++i) {
        auto *bb = blocks[i];
        if (!bb->is_terminated()) { continue; }
        bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (auto iter = block_index.find(succ); iter != block_index.end()) {
                successors[i].emplace_back(iter->second);
                predecessors[iter->second].emplace_back(i);
            }
        });
    }

    // Kosaraju's algorithm, written iteratively to avoid recursion depth limits
    // on generated kernels with large CFGs.
    luisa::vector<uint8_t> visited(blocks.size(), 0u);
    luisa::vector<size_t> finish_order;
    finish_order.reserve(blocks.size());
    for (size_t root = 0u; root < blocks.size(); ++root) {
        if (visited[root] != 0u) { continue; }
        visited[root] = 1u;
        luisa::vector<std::pair<size_t, size_t>> stack;
        stack.emplace_back(root, 0u);
        while (!stack.empty()) {
            auto &[node, next_index] = stack.back();
            if (next_index < successors[node].size()) {
                auto next = successors[node][next_index++];
                if (visited[next] == 0u) {
                    visited[next] = 1u;
                    stack.emplace_back(next, 0u);
                }
            } else {
                finish_order.emplace_back(node);
                stack.pop_back();
            }
        }
    }

    constexpr auto invalid_component = static_cast<size_t>(-1);
    luisa::vector<size_t> component(blocks.size(), invalid_component);
    luisa::vector<size_t> component_size;
    for (size_t i = finish_order.size(); i-- > 0u;) {
        auto root = finish_order[i];
        if (component[root] != invalid_component) { continue; }
        auto component_id = component_size.size();
        auto size = size_t{0u};
        luisa::vector<size_t> work{root};
        component[root] = component_id;
        while (!work.empty()) {
            auto node = work.back();
            work.pop_back();
            ++size;
            for (auto pred : predecessors[node]) {
                if (component[pred] == invalid_component) {
                    component[pred] = component_id;
                    work.emplace_back(pred);
                }
            }
        }
        component_size.emplace_back(size);
    }

    luisa::vector<uint8_t> cyclic(component_size.size(), 0u);
    for (size_t node = 0u; node < blocks.size(); ++node) {
        auto cid = component[node];
        if (component_size[cid] > 1u) {
            cyclic[cid] = 1u;
        } else {
            for (auto succ : successors[node]) {
                if (succ == node) {
                    cyclic[cid] = 1u;
                    break;
                }
            }
        }
    }

    // Count distinct entry *nodes*, not incoming edges. A natural loop may have
    // several external edges to its single header and is still reducible.
    luisa::vector<uint8_t> is_entry_node(blocks.size(), 0u);
    if (auto iter = block_index.find(def->body_block()); iter != block_index.end()) {
        is_entry_node[iter->second] = 1u;
    }
    for (size_t source = 0u; source < blocks.size(); ++source) {
        for (auto target : successors[source]) {
            if (component[source] != component[target]) { is_entry_node[target] = 1u; }
        }
    }
    luisa::vector<size_t> entry_count(component_size.size(), 0u);
    for (size_t node = 0u; node < blocks.size(); ++node) {
        if (is_entry_node[node] != 0u && cyclic[component[node]] != 0u) {
            ++entry_count[component[node]];
        }
    }
    size_t irreducible_count = 0u;
    for (size_t cid = 0u; cid < component_size.size(); ++cid) {
        if (cyclic[cid] != 0u && entry_count[cid] > 1u) { ++irreducible_count; }
    }
    return irreducible_count;
}

[[nodiscard]] bool is_sink(BasicBlock *bb) noexcept {
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

[[nodiscard]] PostDomInfo compute_post_dom(FunctionDefinition *def) noexcept {
    ScopedTimer _timer_pdom("compute_post_dom");
    luisa::vector<BasicBlock *> all_blocks;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        all_blocks.emplace_back(bb);
    });

    luisa::vector<BasicBlock *> sinks;
    for (auto *bb : all_blocks) {
        if (is_sink(bb)) { sinks.emplace_back(bb); }
    }

    // Assign dense block IDs for O(1) array indexing.
    luisa::unordered_map<BasicBlock *, size_t> block_id;
    for (size_t i = 0; i < all_blocks.size(); i++) {
        block_id[all_blocks[i]] = i;
    }
    size_t n = all_blocks.size();

    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> pred_map;
    for (auto *bb : all_blocks) {
        if (!bb->is_terminated()) { continue; }
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

    // Dense RPO index — array lookup instead of hash map.
    luisa::vector<size_t> rpo_idx_vec(n, SIZE_MAX);
    for (size_t i = 0; i < rpo.size(); i++) {
        if (rpo[i] != virt) {
            auto it = block_id.find(rpo[i]);
            if (it != block_id.end()) {
                rpo_idx_vec[it->second] = i;
            }
        }
    }
    // virt is always last in RPO; use rpo.size()-1 for its index.
    const size_t virt_rpo_idx = rpo.size() - 1;

    // Dense ipostdom — array lookup instead of hash map.
    luisa::vector<BasicBlock *> ipostdom_vec(n, nullptr);

    // Dense processed flag.
    luisa::vector<bool> processed_vec(n, false);

    // Helpers for dense lookups.
    auto get_rpo_idx = [&](BasicBlock *b) noexcept -> size_t {
        if (b == virt) { return virt_rpo_idx; }
        if (b == nullptr) { return SIZE_MAX; }
        auto it = block_id.find(b);
        if (it == block_id.end()) { return SIZE_MAX; }
        return rpo_idx_vec[it->second];
    };
    auto get_ipostdom = [&](BasicBlock *b) noexcept -> BasicBlock * {
        if (b == virt) { return virt; }
        if (b == nullptr) { return nullptr; }
        auto it = block_id.find(b);
        if (it == block_id.end()) { return nullptr; }
        return ipostdom_vec[it->second];
    };
    auto set_ipostdom = [&](BasicBlock *b, BasicBlock *val) noexcept {
        if (b == virt || b == nullptr) { return; }
        auto it = block_id.find(b);
        if (it != block_id.end()) {
            ipostdom_vec[it->second] = val;
        }
    };
    auto is_processed = [&](BasicBlock *b) noexcept -> bool {
        if (b == virt) { return true; }
        if (b == nullptr) { return false; }
        auto it = block_id.find(b);
        if (it == block_id.end()) { return false; }
        return processed_vec[it->second];
    };
    auto set_processed = [&](BasicBlock *b) noexcept {
        if (b == virt || b == nullptr) { return; }
        auto it = block_id.find(b);
        if (it != block_id.end()) {
            processed_vec[it->second] = true;
        }
    };

    auto intersect = [&](BasicBlock *b1, BasicBlock *b2) noexcept -> BasicBlock * {
        if (b1 == nullptr) { return b2; }
        if (b2 == nullptr) { return b1; }
        auto f1 = b1;
        auto f2 = b2;
        while (f1 != f2) {
            auto i1 = get_rpo_idx(f1);
            auto i2 = get_rpo_idx(f2);
            if (i1 == SIZE_MAX || i2 == SIZE_MAX) { return nullptr; }
            while (i1 < i2) {
                f1 = get_ipostdom(f1);
                if (f1 == nullptr) { return nullptr; }
                i1 = get_rpo_idx(f1);
            }
            while (i2 < i1) {
                f2 = get_ipostdom(f2);
                if (f2 == nullptr) { return nullptr; }
                i2 = get_rpo_idx(f2);
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
                if (is_processed(s)) {
                    new_ipostdom = intersect(new_ipostdom, s);
                }
            }
            if (get_ipostdom(bb) != new_ipostdom) {
                set_ipostdom(bb, new_ipostdom);
                changed = true;
            }
            if (new_ipostdom != nullptr) { set_processed(bb); }
        }
    }

    // Convert dense results back to hash map for API compatibility.
    PostDomInfo result;
    result.virtual_exit = virt;
    for (auto *bb : rpo) {
        if (bb != virt) {
            result.ipostdom[bb] = get_ipostdom(bb);
        }
    }
    result.ipostdom[virt] = virt;

    return result;
}

[[nodiscard]] size_t dom_depth(const DomTree &dom, BasicBlock *bb) noexcept {
    size_t d = 0;
    auto *node = dom.node_or_null(bb);
    while (node != nullptr && node->parent() != nullptr) {
        ++d;
        node = node->parent();
    }
    return d;
}

[[nodiscard]] BasicBlock *common_postdom(const PostDomInfo &pdom, luisa::span<BasicBlock *const> blocks) noexcept {
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

bool retarget_terminator(Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
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
            if (sw->merge_block() == from) {
                sw->set_merge_block(to);
                changed = true;
            }
            break;
        }
        default: break;
    }
    return changed;
}

[[nodiscard]] bool retarget_loop_exit_to(Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
    if (term == nullptr) { return false; }
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH:
        case DerivedInstructionTag::CONDITIONAL_BRANCH:
        case DerivedInstructionTag::SWITCH:
            return retarget_terminator(term, from, to);
        case DerivedInstructionTag::IF: {
            auto *if_inst = static_cast<IfInst *>(term);
            bool changed = false;
            if (if_inst->true_block() == from) {
                if_inst->set_true_target(to);
                changed = true;
            }
            if (if_inst->false_block() == from) {
                if_inst->set_false_target(to);
                changed = true;
            }
            return changed;
        }
        default: return false;
    }
}

// After retargeting, a conditional branch may have both targets equal.
// Replace it with an unconditional branch to avoid duplicate successors.
void fix_degenerate_terminator(BasicBlock *bb) noexcept {
    if (!bb->is_terminated()) { return; }
    auto *term = bb->terminator();
    if (term->isa<ConditionalBranchInst>()) {
        auto *cb = static_cast<ConditionalBranchInst *>(term);
        if (cb->true_block() == cb->false_block()) {
            auto *target = cb->true_block();
            cb->remove_self();
            XIRBuilder b;
            b.set_insertion_point(bb);
            b.br(target);
        }
    }
}

[[nodiscard]] bool has_only_terminator(BasicBlock *bb) noexcept {
    if (bb == nullptr || !bb->is_terminated()) { return false; }
    auto iter = bb->instructions().begin();
    return iter != bb->instructions().end() && *iter == bb->terminator();
}

[[nodiscard]] BasicBlock *trivial_branch_target(BasicBlock *bb) noexcept {
    if (!has_only_terminator(bb) || !bb->terminator()->isa<BranchInst>()) { return nullptr; }
    return static_cast<BranchInst *>(bb->terminator())->target_block();
}

[[nodiscard]] BasicBlock *trivial_branch_chain_target(BasicBlock *bb) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    auto *cur = bb;
    while (cur != nullptr && visited.emplace(cur).second) {
        auto *next = trivial_branch_target(cur);
        if (next == nullptr) { break; }
        cur = next;
    }
    return cur;
}

[[nodiscard]] bool replace_branch_with_continue(BasicBlock *bb, BasicBlock *from, BasicBlock *continue_target) noexcept {
    if (!bb->is_terminated()) { return false; }
    auto *term = bb->terminator();
    if (!term->isa<BranchInst>()) { return false; }
    if (static_cast<BranchInst *>(term)->target_block() != from) { return false; }
    term->remove_self();
    XIRBuilder b;
    b.set_insertion_point(bb);
    b.continue_(continue_target);
    return true;
}

[[nodiscard]] bool retarget_edges_to_continue(FunctionDefinition *def,
                                              BasicBlock *bb,
                                              BasicBlock *from,
                                              BasicBlock *continue_target) noexcept {
    if (!bb->is_terminated()) { return false; }
    auto *term = bb->terminator();
    if (term->isa<BranchInst>()) {
        return replace_branch_with_continue(bb, from, continue_target);
    }
    if (!term->isa<ConditionalBranchInst>() && !term->isa<SwitchInst>()) { return false; }
    auto *proxy = def->create_basic_block();
    XIRBuilder b;
    b.set_insertion_point(proxy);
    b.continue_(continue_target);
    if (!retarget_terminator(term, from, proxy)) {
        proxy->remove_self();
        return false;
    }
    fix_degenerate_terminator(bb);
    return true;
}

[[nodiscard]] bool replace_branch_with_break(BasicBlock *bb, BasicBlock *from, BasicBlock *break_target) noexcept {
    if (!bb->is_terminated()) { return false; }
    auto *term = bb->terminator();
    if (!term->isa<BranchInst>()) { return false; }
    if (static_cast<BranchInst *>(term)->target_block() != from) { return false; }
    term->remove_self();
    XIRBuilder b;
    b.set_insertion_point(bb);
    b.break_(break_target);
    return true;
}

[[nodiscard]] bool retarget_edges_to_break(FunctionDefinition *def,
                                           BasicBlock *bb,
                                           BasicBlock *from,
                                           BasicBlock *break_target) noexcept {
    if (!bb->is_terminated()) { return false; }
    auto *term = bb->terminator();
    if (term->isa<BranchInst>()) {
        return replace_branch_with_break(bb, from, break_target);
    }
    if (!term->isa<ConditionalBranchInst>() && !term->isa<SwitchInst>()) { return false; }
    auto *proxy = def->create_basic_block();
    XIRBuilder b;
    b.set_insertion_point(proxy);
    b.break_(break_target);
    if (!retarget_terminator(term, from, proxy)) {
        proxy->remove_self();
        return false;
    }
    fix_degenerate_terminator(bb);
    return true;
}

[[nodiscard]] luisa::unordered_set<BasicBlock *> collect_function_blocks(FunctionDefinition *def) noexcept {
    luisa::unordered_set<BasicBlock *> blocks;
    for (auto *bb : def->basic_blocks()) {
        blocks.emplace(bb);
    }
    return blocks;
}

template<typename Visitor>
void traverse_structured_successors(BasicBlock *bb, Visitor &&visit) noexcept {
    if (bb == nullptr || !bb->is_terminated()) { return; }
    auto *term = bb->terminator();
    for (auto *op_use : term->operand_uses()) {
        auto *op = op_use->value();
        if (op != nullptr && op->isa<BasicBlock>()) {
            visit(static_cast<BasicBlock *>(op));
        }
    }
    if (auto *cfm = term->control_flow_merge(); cfm != nullptr) {
        if (auto *merge = cfm->merge_block(); merge != nullptr) { visit(merge); }
    }
    if (term->isa<LoopInst>()) {
        auto *loop = static_cast<LoopInst *>(term);
        if (auto *body = loop->body_block(); body != nullptr) { visit(body); }
        if (auto *update = loop->update_block(); update != nullptr) { visit(update); }
    }
}

[[nodiscard]] bool retarget_loop_backedges_to_continue(FunctionDefinition *def,
                                                       BasicBlock *loop_entry,
                                                       BasicBlock *body,
                                                       BasicBlock *continue_target,
                                                       BasicBlock *merge) noexcept {
    auto function_blocks = collect_function_blocks(def);
    auto is_live_block = [&](BasicBlock *bb) noexcept {
        return bb != nullptr && function_blocks.contains(bb);
    };
    if (!is_live_block(loop_entry) || !is_live_block(body) ||
        !is_live_block(continue_target) || !is_live_block(merge)) {
        return false;
    }
    auto allow_loop_entry_in_region = loop_entry == body && body == continue_target;
    luisa::unordered_set<BasicBlock *> loop_region;
    luisa::vector<BasicBlock *> work;
    auto enqueue = [&](BasicBlock *bb) noexcept {
        if (!is_live_block(bb) || bb == merge) { return; }
        if (!allow_loop_entry_in_region && bb == loop_entry) { return; }
        if (loop_region.emplace(bb).second) { work.emplace_back(bb); }
    };
    enqueue(body);
    enqueue(continue_target);
    while (!work.empty()) {
        auto *bb = work.back();
        work.pop_back();
        if (!is_live_block(bb)) { continue; }
        if (!bb->is_terminated()) { continue; }
        traverse_structured_successors(bb, [&](BasicBlock *succ) noexcept {
            if (!is_live_block(succ) || succ == merge) { return; }
            if (!allow_loop_entry_in_region && succ == loop_entry) { return; }
            enqueue(succ);
        });
    }
    auto changed = false;
    auto *merge_successor = trivial_branch_target(merge);
    for (auto *bb : loop_region) {
        if (!is_live_block(bb)) { continue; }
        if (bb != continue_target) {
            changed |= retarget_edges_to_continue(def, bb, continue_target, continue_target);
        }
        if (bb != merge) {
            changed |= retarget_edges_to_break(def, bb, merge, merge);
            if (merge_successor != nullptr) {
                changed |= retarget_edges_to_break(def, bb, merge_successor, merge);
            }
        }
        if (continue_target == loop_entry) {
            continue;
        }
        if (bb != continue_target ||
            !bb->is_terminated() || !bb->terminator()->isa<BranchInst>() ||
            static_cast<BranchInst *>(bb->terminator())->target_block() != loop_entry) {
            changed |= retarget_edges_to_continue(def, bb, loop_entry, continue_target);
        }
    }
    return changed;
}

[[nodiscard]] bool is_loop_continue_target(BasicBlock *target,
                                           BasicBlock *continue_target,
                                           BasicBlock *loop_entry) noexcept {
    if (target == nullptr || continue_target == nullptr || loop_entry == nullptr) { return false; }
    auto *resolved = trivial_branch_chain_target(target);
    if (resolved == continue_target || resolved == loop_entry) {
        return true;
    }
    if (has_only_terminator(resolved) && resolved->terminator()->isa<ContinueInst>()) {
        auto *continue_inst = static_cast<ContinueInst *>(resolved->terminator());
        return continue_inst->target_block() == continue_target ||
               continue_inst->target_block() == loop_entry;
    }
    return false;
}

[[nodiscard]] bool is_loop_break_target(BasicBlock *target,
                                        BasicBlock *merge) noexcept {
    if (target == nullptr || merge == nullptr) { return false; }
    auto *resolved = trivial_branch_chain_target(target);
    if (resolved == merge) { return true; }
    if (has_only_terminator(resolved) && resolved->terminator()->isa<BreakInst>()) {
        return static_cast<BreakInst *>(resolved->terminator())->target_block() == merge;
    }
    return false;
}

enum struct LoopBoundaryTargetKind {
    NONE,
    BREAK,
    CONTINUE,
};

[[nodiscard]] LoopBoundaryTargetKind classify_loop_boundary_path(BasicBlock *target,
                                                                 BasicBlock *continue_target,
                                                                 BasicBlock *loop_entry,
                                                                 BasicBlock *merge) noexcept {
    if (target == nullptr || continue_target == nullptr || loop_entry == nullptr || merge == nullptr) {
        return LoopBoundaryTargetKind::NONE;
    }
    LoopBoundaryTargetKind kind = LoopBoundaryTargetKind::NONE;
    auto add_kind = [&](LoopBoundaryTargetKind k) noexcept {
        if (k == LoopBoundaryTargetKind::NONE) { return false; }
        if (kind == LoopBoundaryTargetKind::NONE) {
            kind = k;
            return true;
        }
        return kind == k;
    };
    luisa::unordered_set<BasicBlock *> visited;
    luisa::vector<BasicBlock *> work{target};
    while (!work.empty()) {
        auto *bb = work.back();
        work.pop_back();
        if (bb == nullptr || !visited.emplace(bb).second) { continue; }
        if (bb == merge) {
            if (!add_kind(LoopBoundaryTargetKind::BREAK)) { return LoopBoundaryTargetKind::NONE; }
            continue;
        }
        if (bb == continue_target || bb == loop_entry) {
            if (!add_kind(LoopBoundaryTargetKind::CONTINUE)) { return LoopBoundaryTargetKind::NONE; }
            continue;
        }
        if (!bb->is_terminated()) { return LoopBoundaryTargetKind::NONE; }
        auto *term = bb->terminator();
        if (term->isa<BreakInst>()) {
            auto *br = static_cast<BreakInst *>(term);
            if (br->target_block() != merge) { return LoopBoundaryTargetKind::NONE; }
            if (!add_kind(LoopBoundaryTargetKind::BREAK)) { return LoopBoundaryTargetKind::NONE; }
            continue;
        }
        if (term->isa<ContinueInst>()) {
            auto *cont = static_cast<ContinueInst *>(term);
            auto *cont_target = cont->target_block();
            if (cont_target != continue_target && cont_target != loop_entry) { return LoopBoundaryTargetKind::NONE; }
            if (!add_kind(LoopBoundaryTargetKind::CONTINUE)) { return LoopBoundaryTargetKind::NONE; }
            continue;
        }
        if (term->isa<ReturnInst>() || term->isa<UnreachableInst>() || term->isa<RasterDiscardInst>()) {
            return LoopBoundaryTargetKind::NONE;
        }
        traverse_structured_successors(bb, [&](BasicBlock *succ) noexcept {
            if (succ != nullptr) { work.emplace_back(succ); }
        });
    }
    return kind;
}

[[nodiscard]] bool normalize_loop_boundary_conditional_branches(FunctionDefinition *def) noexcept {
    struct LoopSite {
        BasicBlock *entry{nullptr};
        BasicBlock *body{nullptr};
        BasicBlock *continue_target{nullptr};
        BasicBlock *merge{nullptr};
        BasicBlock *selection_merge{nullptr};
    };
    luisa::vector<LoopSite> loops;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            loops.emplace_back(loop->prepare_block(), loop->body_block(), loop->update_block(), loop->merge_block(), loop->update_block());
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            loops.emplace_back(loop->body_block(), loop->body_block(), loop->body_block(), loop->merge_block(), loop->merge_block());
        }
    });

    struct Candidate {
        BasicBlock *branch_block{nullptr};
        BasicBlock *true_target{nullptr};
        BasicBlock *false_target{nullptr};
        BasicBlock *continue_target{nullptr};
        BasicBlock *merge{nullptr};
        BasicBlock *selection_merge{nullptr};
        Value *condition{nullptr};
    };
    luisa::vector<Candidate> candidates;
    for (auto site : loops) {
        if (site.entry == nullptr || site.body == nullptr ||
            site.continue_target == nullptr || site.merge == nullptr || site.selection_merge == nullptr) {
            continue;
        }
        luisa::unordered_set<BasicBlock *> visited;
        luisa::vector<BasicBlock *> work;
        auto enqueue = [&](BasicBlock *bb) noexcept {
            if (bb == nullptr || bb == site.merge) { return; }
            if (visited.emplace(bb).second) { work.emplace_back(bb); }
        };
        enqueue(site.body);
        while (!work.empty()) {
            auto *bb = work.back();
            work.pop_back();
            if (!bb->is_terminated()) { continue; }
            auto *term = bb->terminator();
            if (term->isa<ConditionalBranchInst>()) {
                auto *cbr = static_cast<ConditionalBranchInst *>(term);
                auto *t = cbr->true_block();
                auto *f = cbr->false_block();
                auto true_kind = classify_loop_boundary_path(t, site.continue_target, site.entry, site.merge);
                auto false_kind = classify_loop_boundary_path(f, site.continue_target, site.entry, site.merge);
                auto true_is_continue = true_kind == LoopBoundaryTargetKind::CONTINUE;
                auto false_is_continue = false_kind == LoopBoundaryTargetKind::CONTINUE;
                auto true_is_break = true_kind == LoopBoundaryTargetKind::BREAK;
                auto false_is_break = false_kind == LoopBoundaryTargetKind::BREAK;
                if (true_is_break && false_is_continue) {
                    candidates.emplace_back(Candidate{bb, t, f, site.continue_target, site.merge, site.selection_merge, cbr->condition()});
                    break;
                }
                if (true_is_continue && false_is_break) {
                    candidates.emplace_back(Candidate{bb, t, f, site.continue_target, site.merge, site.selection_merge, cbr->condition()});
                    break;
                }
            }
            traverse_structured_successors(bb, [&](BasicBlock *succ) noexcept {
                if (succ == site.entry || succ == site.merge) { return; }
                enqueue(succ);
            });
        }
        if (!candidates.empty()) { break; }
    }
    if (candidates.empty()) { return false; }

    auto cand = candidates.front();
    if (cand.branch_block == nullptr || !cand.branch_block->is_terminated()) { return false; }
    auto *old_term = cand.branch_block->terminator();
    if (!old_term->isa<ConditionalBranchInst>()) { return false; }
    auto true_kind = classify_loop_boundary_path(cand.true_target, cand.continue_target, cand.continue_target, cand.merge);
    auto false_kind = classify_loop_boundary_path(cand.false_target, cand.continue_target, cand.continue_target, cand.merge);
    auto true_is_break = true_kind == LoopBoundaryTargetKind::BREAK;
    auto false_is_break = false_kind == LoopBoundaryTargetKind::BREAK;
    if (true_is_break == false_is_break) { return false; }

    old_term->remove_self();
    XIRBuilder b;
    b.set_insertion_point(cand.branch_block);
    auto *if_inst = b.if_(cand.condition);
    auto create_boundary_block = [&](bool break_arm) noexcept {
        auto *block = def->create_basic_block();
        XIRBuilder bb;
        bb.set_insertion_point(block);
        if (break_arm) {
            bb.break_(cand.merge);
        } else {
            bb.continue_(cand.continue_target);
        }
        return block;
    };
    auto *true_block = true_is_break ?
                           (is_loop_break_target(cand.true_target, cand.merge) ? create_boundary_block(true) : cand.true_target) :
                           (is_loop_continue_target(cand.true_target, cand.continue_target, cand.continue_target) ? create_boundary_block(false) : cand.true_target);
    auto *false_block = false_is_break ?
                            (is_loop_break_target(cand.false_target, cand.merge) ? create_boundary_block(true) : cand.false_target) :
                            (is_loop_continue_target(cand.false_target, cand.continue_target, cand.continue_target) ? create_boundary_block(false) : cand.false_target);
    if_inst->set_true_target(true_block);
    if_inst->set_false_target(false_block);
    if_inst->set_merge_block(cand.selection_merge);
    return true;
}

[[nodiscard]] bool normalize_structured_loop_continues(FunctionDefinition *def) noexcept {
    struct LoopSite {
        BasicBlock *entry{nullptr};
        BasicBlock *body{nullptr};
        BasicBlock *continue_target{nullptr};
        BasicBlock *merge{nullptr};
    };
    bool changed = false;
    luisa::vector<LoopSite> loops;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            loops.emplace_back(loop->prepare_block(), loop->body_block(), loop->update_block(), loop->merge_block());
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            loops.emplace_back(loop->body_block(), loop->body_block(), loop->body_block(), loop->merge_block());
        }
    });
    for (auto site : loops) {
        changed |= retarget_loop_backedges_to_continue(def, site.entry, site.body, site.continue_target, site.merge);
    }
    return changed;
}

[[nodiscard]] bool block_terminates_with_loop_continue(BasicBlock *bb,
                                                       BasicBlock *continue_target,
                                                       BasicBlock *loop_entry) noexcept {
    auto *resolved = trivial_branch_chain_target(bb);
    if (!has_only_terminator(resolved) || !resolved->terminator()->isa<ContinueInst>()) { return false; }
    auto *target = static_cast<ContinueInst *>(resolved->terminator())->target_block();
    return target == continue_target || target == loop_entry;
}

[[nodiscard]] bool block_terminates_with_loop_break(BasicBlock *bb, BasicBlock *merge) noexcept {
    auto *resolved = trivial_branch_chain_target(bb);
    if (!has_only_terminator(resolved) || !resolved->terminator()->isa<BreakInst>()) { return false; }
    return static_cast<BreakInst *>(resolved->terminator())->target_block() == merge;
}

[[nodiscard]] bool is_loop_boundary_if(IfInst *if_inst,
                                       BasicBlock *continue_target,
                                       BasicBlock *loop_entry,
                                       BasicBlock *merge) noexcept {
    if (if_inst == nullptr) { return false; }
    if (continue_target == nullptr || loop_entry == nullptr || merge == nullptr) { return false; }
    auto true_is_continue = block_terminates_with_loop_continue(if_inst->true_block(), continue_target, loop_entry);
    auto false_is_continue = block_terminates_with_loop_continue(if_inst->false_block(), continue_target, loop_entry);
    auto true_is_break = block_terminates_with_loop_break(if_inst->true_block(), merge);
    auto false_is_break = block_terminates_with_loop_break(if_inst->false_block(), merge);
    return (true_is_continue && false_is_break) || (true_is_break && false_is_continue);
}

[[nodiscard]] bool is_loop_boundary_selection_entry(BasicBlock *entry, FunctionDefinition *def) noexcept {
    if (entry == nullptr || !entry->is_terminated() || !entry->terminator()->isa<IfInst>()) { return false; }
    auto *if_inst = static_cast<IfInst *>(entry->terminator());
    bool found = false;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (found || !bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        BasicBlock *body = nullptr;
        BasicBlock *continue_target = nullptr;
        BasicBlock *loop_entry = nullptr;
        BasicBlock *merge = nullptr;
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            body = loop->body_block();
            continue_target = loop->update_block();
            if (continue_target == nullptr) { continue_target = loop->prepare_block(); }
            loop_entry = loop->prepare_block();
            merge = loop->merge_block();
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            body = loop->body_block();
            continue_target = loop->body_block();
            loop_entry = loop->body_block();
            merge = loop->merge_block();
        } else {
            return;
        }
        if (body == nullptr || merge == nullptr) { return; }
        luisa::unordered_set<BasicBlock *> visited;
        luisa::vector<BasicBlock *> work;
        auto enqueue = [&](BasicBlock *candidate) noexcept {
            if (candidate == nullptr || candidate == merge) { return; }
            if (visited.emplace(candidate).second) { work.emplace_back(candidate); }
        };
        enqueue(body);
        while (!work.empty() && !found) {
            auto *cur = work.back();
            work.pop_back();
            if (cur == entry) {
                found = is_loop_boundary_if(if_inst, continue_target, loop_entry, merge);
                break;
            }
            traverse_structured_successors(cur, [&](BasicBlock *succ) noexcept {
                if (succ == loop_entry || succ == merge) { return; }
                enqueue(succ);
            });
        }
    });
    return found;
}

[[nodiscard]] bool canonicalize_loop_boundary_selection_merges(FunctionDefinition *def) noexcept {
    bool changed = false;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        BasicBlock *body = nullptr;
        BasicBlock *continue_target = nullptr;
        BasicBlock *loop_entry = nullptr;
        BasicBlock *merge = nullptr;
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            body = loop->body_block();
            continue_target = loop->update_block();
            if (continue_target == nullptr) { continue_target = loop->prepare_block(); }
            loop_entry = loop->prepare_block();
            merge = loop->merge_block();
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            body = loop->body_block();
            continue_target = loop->body_block();
            loop_entry = loop->body_block();
            merge = loop->merge_block();
        } else {
            return;
        }
        luisa::unordered_set<BasicBlock *> visited;
        luisa::vector<BasicBlock *> work;
        auto enqueue = [&](BasicBlock *candidate) noexcept {
            if (candidate == nullptr || candidate == merge) { return; }
            if (visited.emplace(candidate).second) { work.emplace_back(candidate); }
        };
        enqueue(body);
        while (!work.empty()) {
            auto *cur = work.back();
            work.pop_back();
            if (cur->is_terminated() && cur->terminator()->isa<IfInst>()) {
                auto *if_inst = static_cast<IfInst *>(cur->terminator());
                if (is_loop_boundary_if(if_inst, continue_target, loop_entry, merge) &&
                    if_inst->merge_block() != merge) {
                    if_inst->set_merge_block(merge);
                    changed = true;
                }
            }
            traverse_structured_successors(cur, [&](BasicBlock *succ) noexcept {
                if (succ == loop_entry || succ == merge) { return; }
                enqueue(succ);
            });
        }
    });
    return changed;
}

[[nodiscard]] bool canonicalize_loop_update_blocks(FunctionDefinition *def) noexcept {
    struct LoopSite {
        LoopInst *loop{nullptr};
        BasicBlock *old_update{nullptr};
        BasicBlock *prepare{nullptr};
    };
    luisa::vector<LoopSite> loops;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<LoopInst>()) { return; }
        auto *loop = static_cast<LoopInst *>(term);
        auto *prepare = loop->prepare_block();
        auto *update = loop->update_block();
        if (prepare == nullptr || update == nullptr) { return; }
        auto canonical = update->is_terminated() && update->terminator()->isa<BranchInst>() &&
                         static_cast<BranchInst *>(update->terminator())->target_block() == prepare;
        if (!canonical) { loops.emplace_back(LoopSite{loop, update, prepare}); }
    });
    if (loops.empty()) { return false; }
    for (auto site : loops) {
        auto *new_update = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(new_update);
        b.br(site.prepare);
        site.loop->set_update_block(new_update);
        def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
            if (!bb->is_terminated()) { return; }
            auto *term = bb->terminator();
            if (term->isa<ContinueInst>()) {
                auto *cont = static_cast<ContinueInst *>(term);
                if (cont->target_block() == site.old_update) { cont->set_target_block(new_update); }
            }
        });
    }
    return true;
}

[[nodiscard]] bool proxy_switch_targets_to_structural_boundaries(FunctionDefinition *def) noexcept {
    luisa::unordered_set<BasicBlock *> structural_boundaries;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (auto *merge = term->control_flow_merge(); merge != nullptr) {
            if (auto *merge_block = merge->merge_block(); merge_block != nullptr) {
                structural_boundaries.emplace(merge_block);
            }
        }
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            if (auto *prepare = loop->prepare_block(); prepare != nullptr) { structural_boundaries.emplace(prepare); }
            if (auto *update = loop->update_block(); update != nullptr) { structural_boundaries.emplace(update); }
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            if (auto *body = loop->body_block(); body != nullptr) { structural_boundaries.emplace(body); }
        }
    });
    if (structural_boundaries.empty()) { return false; }

    struct Target {
        SwitchInst *sw;
        size_t index;
        BasicBlock *target;
        bool is_default;
    };
    luisa::vector<Target> targets;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<SwitchInst>()) { return; }
        auto *sw = static_cast<SwitchInst *>(term);
        auto *own_merge = sw->merge_block();
        auto collect = [&](BasicBlock *target, size_t index, bool is_default) noexcept {
            if (target != nullptr && target != own_merge && structural_boundaries.contains(target)) {
                targets.emplace_back(Target{sw, index, target, is_default});
            }
        };
        collect(sw->default_block(), 0u, true);
        for (size_t i = 0u; i < sw->case_count(); i++) {
            collect(sw->case_block(i), i, false);
        }
    });

    auto changed = false;
    XIRBuilder b;
    for (auto target : targets) {
        auto *proxy = def->create_basic_block();
        b.set_insertion_point(proxy);
        b.br(target.target);
        if (target.is_default) {
            target.sw->set_default_block(proxy);
        } else {
            target.sw->set_case_block(target.index, proxy);
        }
        changed = true;
    }
    return changed;
}

[[nodiscard]] luisa::vector<BasicBlock *> selection_entries(Instruction *term) noexcept {
    luisa::vector<BasicBlock *> entries;
    if (term->isa<IfInst>()) {
        auto *inst = static_cast<IfInst *>(term);
        if (auto *true_block = inst->true_block(); true_block != nullptr) { entries.emplace_back(true_block); }
        if (auto *false_block = inst->false_block(); false_block != nullptr) { entries.emplace_back(false_block); }
    } else if (term->isa<SwitchInst>()) {
        auto *inst = static_cast<SwitchInst *>(term);
        for (size_t i = 0u; i < inst->case_count(); i++) {
            if (auto *case_block = inst->case_block(i); case_block != nullptr) { entries.emplace_back(case_block); }
        }
        if (auto *default_block = inst->default_block(); default_block != nullptr) { entries.emplace_back(default_block); }
    }
    return entries;
}

[[nodiscard]] BasicBlock *structured_statement_merge(Instruction *term) noexcept {
    if (term == nullptr) { return nullptr; }
    auto tag = term->derived_instruction_tag();
    if (tag != DerivedInstructionTag::IF &&
        tag != DerivedInstructionTag::SWITCH &&
        tag != DerivedInstructionTag::LOOP &&
        tag != DerivedInstructionTag::SIMPLE_LOOP) {
        return nullptr;
    }
    auto *cfm = term->control_flow_merge();
    return cfm == nullptr ? nullptr : cfm->merge_block();
}

[[nodiscard]] BasicBlock *canonical_exit_target(BasicBlock *target) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    auto *cur = target;
    while (cur != nullptr && visited.emplace(cur).second) {
        if (!has_only_terminator(cur) || !cur->terminator()->isa<BranchInst>()) { break; }
        auto *next = static_cast<BranchInst *>(cur->terminator())->target_block();
        if (next == nullptr) { break; }
        cur = next;
    }
    return cur;
}

[[nodiscard]] luisa::unordered_set<BasicBlock *> collect_enclosing_loop_exits(FunctionDefinition *def,
                                                                             BasicBlock *header,
                                                                             const DomTree &dom) noexcept {
    luisa::unordered_set<BasicBlock *> exits;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated() || !dom.contains(bb) || !dom.contains(header)) { return; }
        if (!dom.dominates(bb, header)) { return; }
        auto *term = bb->terminator();
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            if (auto *prepare = loop->prepare_block(); prepare != nullptr) { exits.emplace(prepare); }
            if (auto *update = loop->update_block(); update != nullptr) { exits.emplace(update); }
            if (auto *merge = loop->merge_block(); merge != nullptr) { exits.emplace(merge); }
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            if (auto *body = loop->body_block(); body != nullptr) { exits.emplace(body); }
            if (auto *merge = loop->merge_block(); merge != nullptr) { exits.emplace(merge); }
        }
    });
    return exits;
}

struct SelectionExitEdge {
    BasicBlock *src;
    BasicBlock *dst;
};

void append_unique_exit_edge(luisa::vector<SelectionExitEdge> &edges,
                             BasicBlock *src,
                             BasicBlock *dst) noexcept {
    for (auto edge : edges) {
        if (edge.src == src && edge.dst == dst) { return; }
    }
    edges.emplace_back(SelectionExitEdge{src, dst});
}

[[nodiscard]] bool canonicalize_selection_exits(FunctionDefinition *def,
                                                BasicBlock *header,
                                                Instruction *term,
                                                BasicBlock *merge,
                                                const DomTree &dom) noexcept {
    if (header == nullptr || term == nullptr || merge == nullptr) { return false; }
    auto entries = selection_entries(term);
    if (entries.empty()) { return false; }
    auto loop_exits = collect_enclosing_loop_exits(def, header, dom);

    luisa::vector<SelectionExitEdge> invalid_exits;
    luisa::vector<SelectionExitEdge> merge_exits;
    luisa::unordered_set<BasicBlock *> region;
    auto entry_is_valid = [&](BasicBlock *entry) noexcept {
        return entry != nullptr && dom.contains(entry) && dom.dominates(header, entry);
    };
    for (auto *entry : entries) {
        if (!entry_is_valid(entry)) { continue; }
        luisa::vector<BasicBlock *> work{entry};
        while (!work.empty()) {
            auto *bb = work.back();
            work.pop_back();
            if (bb == nullptr || bb == merge) { continue; }
            if (!dom.contains(bb) || !dom.dominates(header, bb)) { continue; }
            if (!region.emplace(bb).second) { continue; }
            if (!bb->is_terminated()) { continue; }
            if (bb != header) {
                if (auto *nested_merge = structured_statement_merge(bb->terminator());
                    nested_merge != nullptr && nested_merge != merge &&
                    dom.contains(nested_merge) && dom.dominates(entry, nested_merge)) {
                    work.emplace_back(nested_merge);
                    continue;
                }
            }
            bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (succ == nullptr) { return; }
                if (succ == merge) {
                    append_unique_exit_edge(merge_exits, bb, succ);
                    return;
                }
                if (loop_exits.contains(succ)) { return; }
                if (succ == header) { return; }
                if (is_sink(succ)) {
                    append_unique_exit_edge(invalid_exits, bb, succ);
                    return;
                }
                if (!dom.contains(succ) || !dom.dominates(entry, succ)) {
                    append_unique_exit_edge(invalid_exits, bb, succ);
                    return;
                }
                work.emplace_back(succ);
            });
        }
    }
    if (invalid_exits.empty()) { return false; }

    luisa::vector<SelectionExitEdge> reroute_edges;
    reroute_edges.reserve(invalid_exits.size() + merge_exits.size());
    for (auto edge : invalid_exits) { reroute_edges.emplace_back(edge); }
    for (auto edge : merge_exits) { reroute_edges.emplace_back(edge); }

    struct RerouteEdge {
        BasicBlock *src;
        BasicBlock *dst;
        BasicBlock *target;
    };
    luisa::vector<RerouteEdge> normalized_edges;
    normalized_edges.reserve(reroute_edges.size());

    luisa::unordered_map<BasicBlock *, uint32_t> target_ids;
    luisa::vector<BasicBlock *> targets;
    auto add_target = [&](BasicBlock *target) noexcept -> uint32_t {
        if (auto it = target_ids.find(target); it != target_ids.end()) { return it->second; }
        auto id = static_cast<uint32_t>(targets.size());
        target_ids.emplace(target, id);
        targets.emplace_back(target);
        return id;
    };
    for (auto edge : reroute_edges) {
        auto *target = canonical_exit_target(edge.dst);
        normalized_edges.emplace_back(RerouteEdge{edge.src, edge.dst, target});
        (void)add_target(target);
    }
    if (term->isa<IfInst>() && targets.size() > 1u) { return false; }

    auto *new_merge = def->create_basic_block();
    XIRBuilder b;
    if (targets.size() == 1u) {
        for (auto edge : normalized_edges) {
            (void)retarget_terminator(edge.src->terminator(), edge.dst, new_merge);
            fix_degenerate_terminator(edge.src);
        }
        b.set_insertion_point(new_merge);
        b.br(targets.front());
    } else {
        auto *entry_bb = def->body_block();
        b.set_insertion_point(entry_bb->instructions().head_sentinel());
        auto *selector = b.alloca_local(Type::of<uint32_t>());
        auto *mod = def->parent_module();
        for (auto edge : normalized_edges) {
            auto *stub = def->create_basic_block();
            if (!retarget_terminator(edge.src->terminator(), edge.dst, stub)) {
                stub->remove_self();
                continue;
            }
            fix_degenerate_terminator(edge.src);
            auto id = target_ids[edge.target];
            auto *id_const = mod->create_constant(Type::of<uint32_t>(), &id);
            b.set_insertion_point(stub);
            b.store(selector, id_const);
            b.br(new_merge);
        }
        b.set_insertion_point(new_merge);
        auto *loaded = b.load(Type::of<uint32_t>(), selector);
        auto *dispatch = def->create_basic_block();
        b.br(dispatch);
        b.set_insertion_point(dispatch);
        auto *sw = b.switch_(loaded);
        sw->set_default_block(targets.front());
        auto *dispatch_merge = def->create_basic_block();
        XIRBuilder mb;
        mb.set_insertion_point(dispatch_merge);
        mb.unreachable_();
        sw->set_merge_block(dispatch_merge);
        for (size_t i = 1u; i < targets.size(); i++) {
            sw->add_case(static_cast<SwitchInst::case_value_type>(i), targets[i]);
        }
    }

    if (term->isa<IfInst>()) {
        static_cast<IfInst *>(term)->set_merge_block(new_merge);
    } else if (term->isa<SwitchInst>()) {
        static_cast<SwitchInst *>(term)->set_merge_block(new_merge);
    }
    return true;
}

[[nodiscard]] bool canonicalize_selection_exits(FunctionDefinition *def, const DomTree &dom) noexcept {
    struct Site {
        BasicBlock *header;
        Instruction *term;
        BasicBlock *merge;
        size_t depth;
    };
    luisa::vector<Site> sites;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<IfInst>() && !term->isa<SwitchInst>()) { return; }
        auto *cfm = term->control_flow_merge();
        if (cfm == nullptr || cfm->merge_block() == nullptr) { return; }
        sites.emplace_back(Site{bb, term, cfm->merge_block(), dom_depth(dom, bb)});
    });
    luisa::sort(sites.begin(), sites.end(), [](auto lhs, auto rhs) noexcept {
        return lhs.depth > rhs.depth;
    });
    for (auto site : sites) {
        if (canonicalize_selection_exits(def, site.header, site.term, site.merge, dom)) {
            return true;
        }
    }
    return false;
}

// Forward declaration.
[[nodiscard]] bool clone_subgraph_to_target(FunctionDefinition *def,
                                            BasicBlock *E, BasicBlock *P,
                                            BasicBlock *target,
                                            BasicBlock *new_target) noexcept;

[[nodiscard]] bool try_restructure_loop(FunctionDefinition *def,
                                        const DomTree &dom,
                                        const PostDomInfo &pdom,
                                        RestructureCFGInfo &info) noexcept {
    ScopedTimer _timer_try_loop("try_restructure_loop");
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

    bool any = false;
    luisa::unordered_set<BasicBlock *> newly_restructured_headers;

    for (auto &cand : candidates) {
        auto *header = cand.header;
        auto &latches = cand.latches;

        // Re-validate: header may have been restructured by a previous candidate in this batch.
        if (!header->is_terminated()) { continue; }
        if (already_loop_headers.contains(header)) { continue; }
        if (newly_restructured_headers.contains(header)) { continue; }
        auto *header_term = header->terminator();
        if (header_term->isa<LoopInst>() || header_term->isa<SimpleLoopInst>()) { continue; }

        // Re-validate latches: they may have been modified by earlier restructuring.
        luisa::vector<BasicBlock *> valid_latches;
        bool latches_ok = true;
        for (auto *latch : latches) {
            if (!dom.dominates(header, latch)) {
                LUISA_WARNING_WITH_LOCATION("restructure_cfg: irreducible back-edge from block to non-dominating header; skipping region");
                info.irreducible_region_count++;
                latches_ok = false;
                break;
            }
            if (!latch->is_terminated()) {
                latches_ok = false;
                break;
            }
            auto *lt = latch->terminator();
            bool has_back_edge = false;
            if (lt->isa<BranchInst>()) {
                has_back_edge = (static_cast<BranchInst *>(lt)->target_block() == header);
            } else if (lt->isa<ConditionalBranchInst>()) {
                auto *cb = static_cast<ConditionalBranchInst *>(lt);
                has_back_edge = (cb->true_block() == header || cb->false_block() == header);
            }
            if (!has_back_edge) {
                latches_ok = false;
                break;
            }
            valid_latches.emplace_back(latch);
        }
        if (!latches_ok || valid_latches.empty()) { continue; }

        BasicBlock *loop_scope_boundary = nullptr;
        if (auto it = pdom.ipostdom.find(header);
            it != pdom.ipostdom.end() && it->second != pdom.virtual_exit) {
            loop_scope_boundary = it->second;
        }
        luisa::unordered_set<BasicBlock *> loop_blocks;
        auto loop_scope_boundary_reaches_latch = [&]() noexcept {
            if (loop_scope_boundary == nullptr || !dom.contains(loop_scope_boundary)) { return false; }
            luisa::unordered_set<BasicBlock *> visited;
            luisa::vector<BasicBlock *> work{loop_scope_boundary};
            while (!work.empty()) {
                auto *cur = work.back();
                work.pop_back();
                if (cur == nullptr || !visited.emplace(cur).second) { continue; }
                for (auto *latch : valid_latches) {
                    if (cur == latch) { return true; }
                }
                if (!cur->is_terminated()) { continue; }
                cur->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                    if (succ == header || (dom.contains(succ) && dom.dominates(header, succ))) {
                        work.emplace_back(succ);
                    }
                });
            }
            return false;
        };
        auto boundary_is_loop_internal = loop_scope_boundary_reaches_latch();
        auto collect_forward_loop_blocks = [&]() noexcept {
            loop_blocks.clear();
            loop_blocks.emplace(header);
            luisa::vector<BasicBlock *> fwd_work{header};
            while (!fwd_work.empty()) {
                auto *cur = fwd_work.back();
                fwd_work.pop_back();
                if (!cur->is_terminated()) { continue; }
                cur->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                    if (succ == loop_scope_boundary && !boundary_is_loop_internal) { return; }
                    if (!dom.contains(succ)) { return; }
                    if (!dom.strictly_dominates(header, succ)) { return; }
                    if (loop_blocks.emplace(succ).second) {
                        fwd_work.emplace_back(succ);
                    }
                });
            }
        };
        auto all_latches_in_loop = [&]() noexcept {
            for (auto *latch : valid_latches) {
                if (!loop_blocks.contains(latch)) { return false; }
            }
            return true;
        };
        auto reaches_latch_or_header = [&](BasicBlock *start) noexcept {
            if (start == nullptr || !dom.contains(start)) { return false; }
            luisa::unordered_set<BasicBlock *> visited;
            luisa::vector<BasicBlock *> work{start};
            while (!work.empty()) {
                auto *cur = work.back();
                work.pop_back();
                if (cur == nullptr || !visited.emplace(cur).second) { continue; }
                if (cur == header) { return true; }
                for (auto *latch : valid_latches) {
                    if (cur == latch) { return true; }
                }
                if (!cur->is_terminated()) { continue; }
                cur->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                    if (succ != nullptr && dom.contains(succ) && dom.dominates(header, succ)) {
                        work.emplace_back(succ);
                    }
                });
            }
            return false;
        };
        auto loop_has_internal_exit = [&]() noexcept {
            for (auto *lb : loop_blocks) {
                if (!lb->is_terminated()) { continue; }
                bool found = false;
                lb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                    if (found || succ == header || loop_blocks.contains(succ)) { return; }
                    found = reaches_latch_or_header(succ);
                });
                if (found) { return true; }
            }
            return false;
        };
        auto collect_natural_loop_blocks = [&]() noexcept {
            loop_blocks.clear();
            loop_blocks.emplace(header);
            luisa::vector<BasicBlock *> loop_work;
            for (auto *latch : valid_latches) {
                if (loop_blocks.emplace(latch).second) {
                    loop_work.emplace_back(latch);
                }
            }
            while (!loop_work.empty()) {
                auto *cur = loop_work.back();
                loop_work.pop_back();
                cur->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                    if (pred == nullptr || !dom.contains(pred)) { return; }
                    if (pred != header && !dom.dominates(header, pred)) { return; }
                    if (loop_blocks.emplace(pred).second) {
                        loop_work.emplace_back(pred);
                    }
                });
            }
        };
        collect_forward_loop_blocks();
        if (!all_latches_in_loop() || loop_has_internal_exit()) {
            collect_natural_loop_blocks();
        }
        for (auto *latch : valid_latches) {
            if (!loop_blocks.contains(latch)) {
                info.irreducible_region_count++;
                latches_ok = false;
                break;
            }
        }
        if (!latches_ok) { continue; }
        if (loop_has_internal_exit()) {
            info.irreducible_region_count++;
            continue;
        }

        luisa::vector<std::pair<BasicBlock *, BasicBlock *>> pre_exit_edges;
        for (auto *lb : loop_blocks) {
            if (!lb->is_terminated()) { continue; }
            lb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (succ == header) { return; }
                if (loop_blocks.contains(succ)) { return; }
                pre_exit_edges.emplace_back(lb, succ);
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
            auto *fresh_merge = def->create_basic_block();
            XIRBuilder mb;
            mb.set_insertion_point(fresh_merge);
            if (dispatch_merge_or_null) {
                mb.br(dispatch_merge_or_null);
            } else {
                mb.unreachable_();
            }
            dispatch_merge_or_null = fresh_merge;
        }

        BasicBlock *canonical_latch = nullptr;
        if (valid_latches.size() == 1) {
            canonical_latch = valid_latches[0];
        } else {
            canonical_latch = def->create_basic_block();
            for (auto *latch : valid_latches) {
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

        auto *loop_merge = def->create_basic_block();

        luisa::vector<std::pair<BasicBlock *, BasicBlock *>> exit_edges;
        for (auto *lb : loop_blocks) {
            if (!lb->is_terminated()) { continue; }
            lb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (succ == loop_merge) { return; }
                if (succ == header) { return; }
                if (loop_blocks.contains(succ)) { return; }
                exit_edges.emplace_back(lb, succ);
            });
        }

        luisa::unordered_set<BasicBlock *> exit_targets_set;
        for (auto &[src, tgt] : exit_edges) { exit_targets_set.emplace(tgt); }
        luisa::vector<BasicBlock *> exit_targets{exit_targets_set.begin(), exit_targets_set.end()};

        auto *mod = def->parent_module();

        if (exit_targets.size() <= 1) {
            for (auto &[src, tgt] : exit_edges) {
                (void)retarget_loop_exit_to(src->terminator(), tgt, loop_merge);
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
            auto *exit_sel = b.alloca_local(Type::of<uint32_t>());
            b.set_insertion_point(preheader);
            auto *preheader_br = preheader->terminator();
            preheader_br->remove_self();
            b.set_insertion_point(preheader);
            auto *zero_const = mod->create_constant_zero(Type::of<uint32_t>());
            b.store(exit_sel, zero_const);
            b.br(header);

            uint32_t sel_id = 0;
            luisa::unordered_map<BasicBlock *, uint32_t> exit_target_id;
            luisa::vector<BasicBlock *> used_exit_targets;

            BasicBlock *direct_header_exit_target = nullptr;
            for (auto &[src, tgt] : exit_edges) {
                if (src == header) {
                    direct_header_exit_target = tgt;
                    exit_target_id.emplace(tgt, sel_id++);
                    used_exit_targets.emplace_back(tgt);
                    break;
                }
            }

            for (auto &[src, tgt] : exit_edges) {
                if (src == header && tgt == direct_header_exit_target) {
                    (void)retarget_loop_exit_to(src->terminator(), tgt, loop_merge);
                    continue;
                }
                auto *stub = def->create_basic_block();
                auto changed = retarget_loop_exit_to(src->terminator(), tgt, stub);
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
                auto *id_const = mod->create_constant(Type::of<uint32_t>(), &id);
                b.set_insertion_point(stub);
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
        BasicBlock *loop_exit_succ = nullptr;
        if (header->is_terminated()) {
            auto *ht = header->terminator();
            if (ht->isa<ConditionalBranchInst>()) {
                auto *cb = static_cast<ConditionalBranchInst *>(ht);
                auto *tb = cb->true_block();
                auto *fb = cb->false_block();
                bool true_in_loop = loop_blocks.contains(tb);
                bool false_in_loop = loop_blocks.contains(fb);
                // The loop body successor is the target that remains in the loop.
                // This handles both single-exit and multi-exit cases.
                if (true_in_loop && !false_in_loop) {
                    loop_body_succ = tb;
                    loop_exit_succ = fb;
                } else if (!true_in_loop && false_in_loop) {
                    loop_body_succ = fb;
                    loop_exit_succ = tb;
                }
            } else if (ht->isa<BranchInst>()) {
                auto *target = static_cast<BranchInst *>(ht)->target_block();
                if (loop_blocks.contains(target)) {
                    loop_body_succ = target;
                }
            }
        }

        {
            XIRBuilder b;
            b.set_insertion_point(preheader);
            if (loop_body_succ != nullptr && loop_body_succ != canonical_latch) {
                if (header->terminator()->isa<ConditionalBranchInst>()) {
                    auto *cb = static_cast<ConditionalBranchInst *>(header->terminator());
                    if (cb->true_block() != loop_body_succ) {
                        XIRBuilder hb;
                        hb.set_insertion_point(cb->prev());
                        auto *not_cond = hb.call(Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {cb->condition()});
                        cb->set_condition(not_cond);
                        cb->set_true_target(loop_body_succ);
                        cb->set_false_target(loop_exit_succ);
                    }
                }
                for (auto *lb : loop_blocks) {
                    if (lb != canonical_latch) {
                        (void)retarget_edges_to_continue(def, lb, header, canonical_latch);
                    }
                }
                b.set_insertion_point(preheader);
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

        newly_restructured_headers.emplace(header);
        info.restructured_loop_count++;
        any = true;
        return true;
    }

    return any;
}

[[nodiscard]] bool try_restructure_if_batch(FunctionDefinition *def,
                                            const DomTree &dom,
                                            const PostDomInfo &pdom,
                                            RestructureCFGInfo &info,
                                            luisa::unordered_set<BasicBlock *> &all_created_structural_merges,
                                            luisa::unordered_map<BasicBlock *, BasicBlock *> &sm_to_header) noexcept {
    ScopedTimer _timer_try_if("try_restructure_if_batch");
    // Collect merge blocks and headers of already-structured loops.
    luisa::unordered_map<BasicBlock *, BasicBlock *> loop_merge_to_header;
    luisa::unordered_set<BasicBlock *> loop_headers;
    luisa::unordered_set<BasicBlock *> loop_prepare_blocks;
    luisa::unordered_map<BasicBlock *, BasicBlock *> loop_update_to_prepare;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        BasicBlock *merge = nullptr;
        if (term->isa<LoopInst>()) {
            auto *li = static_cast<LoopInst *>(term);
            merge = li->merge_block();
            if (li->prepare_block() != nullptr) { loop_prepare_blocks.emplace(li->prepare_block()); }
            if (li->update_block() != nullptr && li->prepare_block() != nullptr) {
                loop_update_to_prepare.emplace(li->update_block(), li->prepare_block());
            }
        } else if (term->isa<SimpleLoopInst>()) {
            merge = static_cast<SimpleLoopInst *>(term)->merge_block();
        }
        if (merge != nullptr) {
            loop_merge_to_header.emplace(merge, bb);
            loop_headers.emplace(bb);
        }
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
        if (loop_prepare_blocks.contains(bb)) { return; }
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
    auto &created_structural_merges = all_created_structural_merges;

    // Process all candidates from innermost to outermost.
    // Since we process innermost first, restructuring an inner if does not
    // invalidate the dom/pdom for outer if-candidates. We re-validate each
    // candidate before processing to guard against stale state.
    for (auto &cand : candidates) {
        auto *found_header = cand.header;
        auto *found_cbr = cand.cbr;
        auto *found_merge = cand.merge;
        auto *enclosing_loop_continue = cand.enclosing_loop_continue;

        // Re-validate: header may have been restructured by a previous candidate in this batch.
        if (!found_header->is_terminated()) { continue; }
        auto *check_term = found_header->terminator();
        if (!check_term->isa<ConditionalBranchInst>()) { continue; }
        if (static_cast<ConditionalBranchInst *>(check_term) != found_cbr) { continue; }

        auto *true_bb = found_cbr->true_block();
        auto *false_bb = found_cbr->false_block();
        auto *cond = found_cbr->condition();

        // If found_merge is a structural_merge created earlier,
        // follow its unique successor chain to find the real merge point.
        while (created_structural_merges.contains(found_merge)) {
            auto *term = found_merge->terminator();
            if (term != nullptr && term->isa<BranchInst>()) {
                auto *br = static_cast<BranchInst *>(term);
                if (auto *target = br->target_block(); target != nullptr) {
                    found_merge = target;
                    continue;
                }
            }
            break;
        }

        BasicBlock *structural_merge = nullptr;
        if (loop_headers.contains(found_merge) && found_header == found_merge) {
            structural_merge = found_merge;
        } else {
            structural_merge = def->create_basic_block();
            created_structural_merges.emplace(structural_merge);
            sm_to_header.emplace(structural_merge, found_header);
            {
                XIRBuilder mb;
                mb.set_insertion_point(structural_merge);
                mb.br(found_merge);
            }
        }

        luisa::unordered_set<BasicBlock *> allowed_outside_targets;
        for (auto &[loop_merge, loop_header] : loop_merge_to_header) {
            if (dom.dominates(loop_header, found_header)) {
                allowed_outside_targets.emplace(loop_merge);
                allowed_outside_targets.emplace(loop_header);
                auto *loop_term = loop_header->terminator();
                if (loop_term->isa<LoopInst>()) {
                    auto *li = static_cast<LoopInst *>(loop_term);
                    if (li->prepare_block() != nullptr) {
                        allowed_outside_targets.emplace(li->prepare_block());
                    }
                    if (li->update_block() != nullptr) {
                        allowed_outside_targets.emplace(li->update_block());
                    }
                } else if (loop_term->isa<SimpleLoopInst>()) {
                    auto *sl = static_cast<SimpleLoopInst *>(loop_term);
                    allowed_outside_targets.emplace(sl->body_block());
                }
            }
        }

        // Compute all blocks reachable from structural_merge.
        luisa::unordered_set<BasicBlock *> reachable_from_sm;
        {
            luisa::vector<BasicBlock *> sm_work;
            sm_work.push_back(structural_merge);
            while (!sm_work.empty()) {
                auto *bb = sm_work.back();
                sm_work.pop_back();
                if (!reachable_from_sm.emplace(bb).second) { continue; }
                if (!bb->is_terminated()) { continue; }
                bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                    sm_work.emplace_back(succ);
                });
            }
        }

        // Compute the set of blocks inside the current if's scope.
        luisa::unordered_set<BasicBlock *> if_scope_blocks;
        {
            luisa::vector<BasicBlock *> scope_work;
            if (true_bb != found_merge && true_bb != structural_merge) {
                scope_work.push_back(true_bb);
            }
            if (false_bb != found_merge && false_bb != structural_merge) {
                scope_work.push_back(false_bb);
            }
            while (!scope_work.empty()) {
                auto *bb = scope_work.back();
                scope_work.pop_back();
                if (bb == found_merge || bb == structural_merge) { continue; }
                if (!if_scope_blocks.emplace(bb).second) { continue; }
                if (!bb->is_terminated()) { continue; }
                bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                    scope_work.emplace_back(succ);
                });
            }
        }

        // Walk the dominator-tree subtree rooted at found_header.
        // Only retarget unstructured cbr/br blocks that are actually inside
        // the if's scope. Skip IfInst/SwitchInst/LoopInst terminators to avoid
        // corrupting already-structured inner constructs.
        auto header_node = dom.node_or_null(found_header);
        if (header_node != nullptr) {
            luisa::vector<const DomTreeNode *> work;
            work.push_back(header_node);
            while (!work.empty()) {
                auto *node = work.back();
                work.pop_back();
                auto *bb = node->block();
                if (bb != structural_merge && bb != found_header && bb != found_merge &&
                    bb->is_terminated() && if_scope_blocks.contains(bb) &&
                    !allowed_outside_targets.contains(bb)) {
                    auto *term = bb->terminator();
                    if (term->isa<ConditionalBranchInst>() || term->isa<BranchInst>()) {
                        bool is_loop_update_backedge = false;
                        if (auto it = loop_update_to_prepare.find(bb);
                            it != loop_update_to_prepare.end() && it->second == found_merge) {
                            is_loop_update_backedge = true;
                        }
                        if (!is_loop_update_backedge) {
                            if (dom.contains(found_merge) && !dom.strictly_dominates(found_merge, bb)) {
                                retarget_terminator(term, found_merge, structural_merge);
                            } else if (dom.contains(found_merge) && dom.strictly_dominates(found_merge, bb) && dom.strictly_dominates(found_header, bb)) {
                                retarget_terminator(term, found_merge, structural_merge);
                            }
                        }
                        fix_degenerate_terminator(bb);
                    }
                }
                for (auto *child : node->children()) {
                    work.push_back(child);
                }
            }
        }

        if (true_bb == found_merge) { true_bb = structural_merge; }
        if (false_bb == found_merge) { false_bb = structural_merge; }

        // Sanity check: retargeting must not have removed the header's terminator.
        if (found_header->is_terminated() &&
            found_header->terminator()->isa<ConditionalBranchInst>() &&
            found_header->terminator() == found_cbr) {

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
                    if (succ == found_merge) { return; }
                    if (succ == enclosing_loop_continue) { return; }
                    if (allowed_outside_targets.contains(succ)) { return; }
                    if (!dom.contains(succ)) { return; }
                    if (dom.dominates(found_header, succ)) { return; }
                    bad_succs.emplace_back(succ);
                });
                luisa::sort(bad_succs.begin(), bad_succs.end());
                bad_succs.erase(std::unique(bad_succs.begin(), bad_succs.end()), bad_succs.end());
                for (auto *succ : bad_succs) {
                    (void)clone_subgraph_to_target(def, succ, arm_bb, found_merge, structural_merge);
                }
            }

            info.restructured_if_count++;
            any = true;
        }
    }

    return any;
}

// Collect the entry blocks of a structured construct C whose header is `header_bb`.
// "Entry blocks" are blocks that should only be reachable from the header (or from
// authorized internal back-edges, e.g. the update block of a loop), and NEVER from
// sibling arms. Returns nullptr-free, possibly-duplicate-free list.
void collect_construct_entries(BasicBlock *header_bb,
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
[[nodiscard]] bool is_authorized_construct_pred(Instruction *header_term,
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
[[nodiscard]] bool is_clone_boundary(BasicBlock *S, BasicBlock *E,
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
void collect_owned_region(BasicBlock *E, BasicBlock *header_bb,
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
[[nodiscard]] bool clone_owned_subgraph_for_edge(FunctionDefinition *def,
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
[[nodiscard]] bool clone_subgraph_to_target(FunctionDefinition *def,
                                            BasicBlock *E, BasicBlock *P,
                                            BasicBlock *target,
                                            BasicBlock *new_target) noexcept {
    ScopedTimer _timer_clone_subgraph("clone_subgraph_to_target");
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
[[nodiscard]] bool enforce_construct_entries(FunctionDefinition *def,
                                             BasicBlock *header_bb,
                                             BasicBlock *merge_bb) noexcept {
    ScopedTimer _timer_enforce_entries("enforce_construct_entries");
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
void enforce_unique_construct_entries(FunctionDefinition *def) noexcept {
    ScopedTimer _timer_enforce_unique("enforce_unique_construct_entries");
    size_t outer_guard = 64;
    while (outer_guard-- > 0) {
        bool changed = false;
        luisa::vector<std::pair<BasicBlock *, BasicBlock *>> construct_sites;// header_bb, merge_bb
        def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
            if (!bb->is_terminated()) { return; }
            auto *t = bb->terminator();
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

// Ensure each case target of a SwitchInst is unique.
// If multiple cases branch to the same block, a proxy block is inserted.
// Ported from LLVM SPIRVStructurizer::splitSwitchCases.
[[nodiscard]] static bool split_switch_cases(FunctionDefinition *def) noexcept {
    ScopedTimer _timer_split_switch("split_switch_cases");
    bool modified = false;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<SwitchInst>()) { return; }
        auto *sw = static_cast<SwitchInst *>(term);

        luisa::unordered_set<BasicBlock *> seen;
        if (auto *db = sw->default_block(); db != nullptr) { seen.emplace(db); }

        for (size_t i = 0; i < sw->case_count();) {
            auto *target = sw->case_block(i);
            if (target == nullptr || !seen.contains(target)) {
                if (target != nullptr) { seen.emplace(target); }
                ++i;
                continue;
            }
            modified = true;
            auto *proxy = def->create_basic_block();
            XIRBuilder b;
            b.set_insertion_point(proxy);
            b.br(target);
            sw->set_case_block(i, proxy);
            ++i;
        }
    });
    return modified;
}

// Structurize remaining conditional branches that were missed by
// try_restructure_if_batch (e.g., when both arms eventually return). Uses the
// nearest common post-dominator of all successors as the merge block.
// Ported from LLVM SPIRVStructurizer::addHeaderToRemainingDivergentDAG.
[[nodiscard]] static bool add_header_to_remaining_divergent(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom,
    RestructureCFGInfo &info) noexcept {
    ScopedTimer _timer_add_header("add_header_to_remaining_divergent");

    // Recompute structured metadata fresh.
    luisa::unordered_set<BasicBlock *> merge_set;
    luisa::unordered_set<BasicBlock *> header_set;
    luisa::unordered_set<BasicBlock *> continue_set;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        auto tag = term->derived_instruction_tag();
        if (tag == DerivedInstructionTag::IF || tag == DerivedInstructionTag::SWITCH ||
            tag == DerivedInstructionTag::LOOP || tag == DerivedInstructionTag::SIMPLE_LOOP) {
            header_set.emplace(bb);
        }
        if (auto *cm = term->control_flow_merge(); cm != nullptr) {
            if (auto *mb = cm->merge_block(); mb != nullptr) { merge_set.emplace(mb); }
        }
        if (term->isa<LoopInst>()) {
            auto *lp = static_cast<LoopInst *>(term);
            if (lp->update_block()) { continue_set.emplace(lp->update_block()); }
            if (lp->prepare_block()) { continue_set.emplace(lp->prepare_block()); }
        } else if (term->isa<SimpleLoopInst>()) {
            auto *sl = static_cast<SimpleLoopInst *>(term);
            if (sl->body_block()) { continue_set.emplace(sl->body_block()); }
        }
    });

    luisa::vector<BasicBlock *> all_blocks;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept { all_blocks.emplace_back(bb); });

    // Find the first conditional branch that needs a header.
    BasicBlock *found_bb = nullptr;
    BasicBlock *found_t = nullptr;
    BasicBlock *found_f = nullptr;
    BasicBlock *found_merge = nullptr;
    bool found_is_synthetic = false;
    Value *found_cond = nullptr;

    for (auto *bb : all_blocks) {
        if (found_bb != nullptr) { break; }
        if (header_set.contains(bb)) { continue; }
        if (!bb->is_terminated()) { continue; }
        auto *term = bb->terminator();
        if (!term->isa<ConditionalBranchInst>()) { continue; }
        auto *cbr = static_cast<ConditionalBranchInst *>(term);

        auto *t = cbr->true_block();
        auto *f = cbr->false_block();
        if (t == nullptr || f == nullptr || t == f) { continue; }

        size_t candidate = 0;
        for (auto *s : {t, f}) {
            if (!merge_set.contains(s) && !continue_set.contains(s) && !header_set.contains(s)) {
                ++candidate;
            }
        }
        if (candidate <= 1) { continue; }

        luisa::vector<BasicBlock *> succs_vec;
        succs_vec.push_back(t);
        succs_vec.push_back(f);
        auto *merge = common_postdom(pdom, luisa::span<BasicBlock *const>{succs_vec.data(), succs_vec.size()});
        bool is_synthetic = (merge == nullptr || merge == pdom.virtual_exit || merge == bb);

        if (!is_synthetic) {
            bool has_bad = false;
            luisa::unordered_set<BasicBlock *> visited;
            luisa::vector<BasicBlock *> work;
            work.push_back(t);
            work.push_back(f);
            while (!work.empty() && !has_bad) {
                auto *cur = work.back();
                work.pop_back();
                if (cur == merge || cur == bb) { continue; }
                if (!visited.emplace(cur).second) { continue; }
                if (!dom.dominates(bb, cur)) { continue; }
                if (dom.dominates(merge, cur)) { continue; }
                if (merge_set.contains(cur) || continue_set.contains(cur) || header_set.contains(cur)) {
                    has_bad = true;
                    break;
                }
                if (!cur->is_terminated()) { continue; }
                cur->traverse_successors(false, [&](BasicBlock *s) noexcept { work.emplace_back(s); });
            }
            if (has_bad) { continue; }
        }

        found_bb = bb;
        found_t = t;
        found_f = f;
        found_merge = merge;
        found_is_synthetic = is_synthetic;
        found_cond = cbr->condition();
    }

    if (found_bb == nullptr) { return false; }

    // Apply one fixup, then recompute dom/pdom for the caller.
    auto *merge = found_merge;
    if (found_is_synthetic) {
        merge = def->create_basic_block();
        XIRBuilder ub;
        ub.set_insertion_point(merge);
        ub.unreachable_();
    }

    auto *bb = found_bb;
    auto *t = found_t;
    auto *f = found_f;
    auto *cond = found_cond;

    if (bb->is_terminated() && bb->terminator()->isa<ConditionalBranchInst>()) {
        bb->terminator()->remove_self();
    }

    auto *structural_merge = def->create_basic_block();
    {
        XIRBuilder mb;
        mb.set_insertion_point(structural_merge);
        mb.br(merge);
    }

    luisa::vector<BasicBlock *> fwd_work;
    fwd_work.push_back(t);
    fwd_work.push_back(f);
    luisa::unordered_set<BasicBlock *> fwd_visited;
    fwd_visited.emplace(t);
    fwd_visited.emplace(f);
    fwd_visited.emplace(merge);
    fwd_visited.emplace(structural_merge);
    while (!fwd_work.empty()) {
        auto *cur = fwd_work.back();
        fwd_work.pop_back();
        if (cur == bb || cur == merge) { continue; }
        if (!dom.dominates(bb, cur)) { continue; }
        if (cur->is_terminated()) {
            retarget_terminator(cur->terminator(), merge, structural_merge);
            fix_degenerate_terminator(cur);
        }
        cur->traverse_successors(false, [&](BasicBlock *s) noexcept {
            if (fwd_visited.emplace(s).second) { fwd_work.emplace_back(s); }
        });
    }
    if (t == merge) { t = structural_merge; }
    if (f == merge) { f = structural_merge; }

    XIRBuilder b;
    b.set_insertion_point(bb);
    auto *if_inst = b.if_(cond);
    if_inst->set_true_target(t);
    if_inst->set_false_target(f);
    if_inst->set_merge_block(structural_merge);
    info.restructured_if_count++;

    // Invalidate analyses after CFG mutation.
    dom = compute_dom_tree(def);
    pdom = compute_post_dom(def);
    return true;
}

// Ensure each structured construct's exit edges respect SPIR-V hierarchy:
// an exit from construct C must go through C's immediate parent's merge block.
// Fix up exit edges of structured constructs using convergence region analysis.
// Ported from LLVM SPIRVStructurizer::fixupConstruct with S.invalidate() pattern.
[[nodiscard]] static bool fixup_construct_exits(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom) noexcept {
    ScopedTimer _timer_fixup_exits("fixup_construct_exits");

    bool modified = false;
    size_t max_iters = 64;

    // Helper: check if a construct needs merge-equality fixup.
    auto needs_fixup = [](const ConvergenceRegion *cr, const ConvergenceRegion *parent) -> bool {
        if (cr->convergence_merge == parent->convergence_merge) { return true; }
        auto *ht = cr->entry->terminator();
        if (ht->isa<LoopInst>()) {
            auto *lp = static_cast<LoopInst *>(ht);
            auto *cont = lp->update_block();
            if (cont == nullptr) { cont = lp->prepare_block(); }
            if (cr->convergence_merge == cont) { return true; }
        } else if (ht->isa<SimpleLoopInst>()) {
            if (cr->convergence_merge == static_cast<SimpleLoopInst *>(ht)->body_block()) { return true; }
        }
        return false;
    };

    while (max_iters-- > 0) {
        // Compute fresh analysis before each fixup pass (LLVM's S.invalidate()).
        auto cri = compute_convergence_regions(def, dom);
        if (cri.top_level == nullptr || cri.top_level->children.empty()) { break; }

        // Walk post-order to find the first construct needing fixup.
        // Break after one fixup to recompute from scratch.
        bool local_mod = false;
        luisa::function<bool(ConvergenceRegion *, ConvergenceRegion *)> try_fixup;
        try_fixup = [&](ConvergenceRegion *cr, ConvergenceRegion *parent) -> bool {
            for (auto &child : cr->children) {
                if (try_fixup(child.get(), cr)) { return true; }
            }
            if (parent == nullptr || parent == cri.top_level.get()) { return false; }
            if (cr->entry == nullptr || cr->convergence_merge == nullptr) { return false; }
            if (is_loop_boundary_selection_entry(cr->entry, def)) { return false; }

            auto &blks = cr->blocks;
            luisa::vector<std::pair<BasicBlock *, BasicBlock *>> exits;
            for (auto *bb : blks) {
                if (!bb->is_terminated()) { continue; }
                bb->traverse_successors(false, [&](BasicBlock *s) noexcept {
                    if (!blks.contains(s)) {
                        exits.emplace_back(bb, s);
                    }
                });
            }
            if (exits.empty()) { return false; }
            if (!needs_fixup(cr, parent)) { return false; }

            local_mod = true;
            luisa::unordered_set<BasicBlock *> et_set;
            for (auto &[src, dst] : exits) { et_set.emplace(dst); }
            luisa::vector<BasicBlock *> et{et_set.begin(), et_set.end()};
            auto *new_exit = def->create_basic_block();
            bool retargeted_any = false;

            if (et.size() == 1) {
                for (auto &[src, dst] : exits) { retargeted_any |= retarget_terminator(src->terminator(), dst, new_exit); }
                XIRBuilder b;
                b.set_insertion_point(new_exit);
                b.br(et[0]);
            } else {
                auto *entry_bb = def->body_block();
                XIRBuilder b;
                b.set_insertion_point(entry_bb->instructions().head_sentinel());
                auto *sel = b.alloca_local(Type::of<uint32_t>());
                uint32_t id = 0;
                luisa::unordered_map<BasicBlock *, uint32_t> tid_map;
                luisa::vector<BasicBlock *> ord;
                for (auto &[src, dst] : exits) {
                    auto *stub = def->create_basic_block();
                    if (!retarget_terminator(src->terminator(), dst, stub)) {
                        stub->remove_self();
                        continue;
                    }
                    retargeted_any = true;
                    auto it = tid_map.find(dst);
                    uint32_t v;
                    if (it == tid_map.end()) {
                        v = id++;
                        tid_map.emplace(dst, v);
                        ord.emplace_back(dst);
                    } else {
                        v = it->second;
                    }
                    b.set_insertion_point(stub);
                    b.store(sel, def->parent_module()->create_constant(Type::of<uint32_t>(), &v));
                    b.br(new_exit);
                }
                b.set_insertion_point(new_exit);
                auto *ld = b.load(Type::of<uint32_t>(), sel);
                auto *disp = def->create_basic_block();
                b.br(disp);
                b.set_insertion_point(disp);
                if (ord.empty()) {
                    b.unreachable_();
                } else if (ord.size() == 1u) {
                    b.br(ord[0]);
                } else if (ord.size() == 2u) {
                    auto *zero = def->parent_module()->create_constant_zero(Type::of<uint32_t>());
                    auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {ld, zero});
                    b.cond_br(cond, ord[0], ord[1]);
                } else {
                    auto *sw = b.switch_(ld);
                    sw->set_default_block(ord[0]);
                    for (size_t i = 1; i < ord.size(); ++i) {
                        sw->add_case(static_cast<SwitchInst::case_value_type>(tid_map[ord[i]]), ord[i]);
                    }
                }
            }
            if (!retargeted_any) {
                new_exit->remove_self();
                return false;
            }

            auto *ht = cr->entry->terminator();
            if (auto *cm2 = ht->control_flow_merge(); cm2 != nullptr) {
                if (cm2->merge_block() == cr->convergence_merge) { cm2->set_merge_block(new_exit); }
            }
            cr->convergence_merge = new_exit;
            return true;
        };

        for (auto &child : cri.top_level->children) {
            if (try_fixup(child.get(), cri.top_level.get())) { break; }
        }
        if (!local_mod) { break; }
        modified = true;
        // Invalidate and recompute after CFG modification (LLVM's S.invalidate()).
        dom = compute_dom_tree(def);
        pdom = compute_post_dom(def);
    }
    return modified;
}

[[nodiscard]] RestructureCFGInfo restructure_cfg_on_definition(FunctionDefinition *def) noexcept {
    ScopedTimer _timer_overall("restructure_cfg_on_definition");
    check_phi_free(def);
    RestructureCFGInfo info{};
    if (auto count = count_irreducible_regions(def); count != 0u) {
        info.irreducible_region_count = count;
        LUISA_WARNING_WITH_LOCATION(
            "restructure_cfg rejected {} irreducible multi-entry cyclic region(s); "
            "the function was left unchanged.",
            count);
        return info;
    }
    luisa::unordered_set<BasicBlock *> all_created_structural_merges;
    luisa::unordered_map<BasicBlock *, BasicBlock *> sm_to_header;
    size_t max_iters = 10000;
    while (max_iters-- > 0) {
        ScopedTimer _timer_main_iter("main_loop_iteration");
        auto dom = compute_dom_tree(def);
        auto pdom = compute_post_dom(def);
        if (try_restructure_loop(def, dom, pdom, info)) {
            // Fast path: if no conditional branches remain after restructuring
            // all loops, there are no if-candidates either — break early.
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
        if (try_restructure_if_batch(def, dom, pdom, info, all_created_structural_merges, sm_to_header)) {
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
    (void)split_switch_cases(def);

    // Post-restructure fixed-point: passes may create new structured
    // constructs that need further normalization. Recompute dom/pdom
    // after each pass modifies the CFG.
    size_t post_iters = 16;
    {
        ScopedTimer _timer_post("post_restructure_fixed_point");
        auto dom = compute_dom_tree(def);
        auto pdom = compute_post_dom(def);
        while (post_iters-- > 0) {
            ScopedTimer _timer_post_iter("post_restructure_iteration");
            bool local = false;
            if (try_restructure_loop(def, dom, pdom, info)) {
                local = true;
                dom = compute_dom_tree(def);
                pdom = compute_post_dom(def);
            }
            if (add_header_to_remaining_divergent(def, dom, pdom, info)) {
                local = true;
                // dom/pdom already recomputed internally by add_header_to_remaining_divergent.
            }
            if (proxy_switch_targets_to_structural_boundaries(def)) {
                local = true;
                dom = compute_dom_tree(def);
                pdom = compute_post_dom(def);
            }
            if (canonicalize_selection_exits(def, dom)) {
                local = true;
                dom = compute_dom_tree(def);
                pdom = compute_post_dom(def);
            }
            if (canonicalize_loop_boundary_selection_merges(def)) {
                local = true;
                dom = compute_dom_tree(def);
                pdom = compute_post_dom(def);
            }
            if (normalize_loop_boundary_conditional_branches(def)) {
                local = true;
                dom = compute_dom_tree(def);
                pdom = compute_post_dom(def);
            }
            if (normalize_structured_loop_continues(def)) {
                local = true;
                dom = compute_dom_tree(def);
                pdom = compute_post_dom(def);
            }
            if (canonicalize_loop_update_blocks(def)) {
                local = true;
                dom = compute_dom_tree(def);
                pdom = compute_post_dom(def);
            }
            if (fixup_construct_exits(def, dom, pdom)) {
                local = true;
                dom = compute_dom_tree(def);
                pdom = compute_post_dom(def);
            }
            if (!local) { break; }
        }
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
