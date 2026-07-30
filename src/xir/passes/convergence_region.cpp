#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/passes/convergence_region.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::xir {

namespace {

// Collect all loop header blocks by detecting back-edges via dominance.
luisa::unordered_set<BasicBlock *> collect_loop_headers(FunctionDefinition *def, const DomTree &dom) noexcept {
    luisa::unordered_set<BasicBlock *> headers;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (dom.contains(succ) && dom.dominates(succ, bb)) {
                headers.emplace(succ);
            }
        });
    });
    return headers;
}

// Collect blocks that have successors outside the given region.
luisa::unordered_set<BasicBlock *> find_exit_nodes(
    const luisa::unordered_set<BasicBlock *> &region) noexcept {
    luisa::unordered_set<BasicBlock *> exits;
    for (auto *bb : region) {
        if (!bb->is_terminated()) { continue; }
        bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (!region.contains(succ)) { exits.emplace(bb); }
        });
    }
    return exits;
}

// Check if a block belongs to the convergence scope of a construct with
// the given header and merge. A block belongs if:
// 1. It is dominated by the construct's header, AND
// 2. It is NOT strictly dominated by the construct's merge.
bool block_shares_convergence(BasicBlock *bb, BasicBlock *header, BasicBlock *merge, const DomTree &dom) noexcept {
    if (bb == merge) { return false; }
    if (!dom.contains(bb) || !dom.contains(header) || !dom.contains(merge)) { return false; }
    if (!dom.dominates(header, bb)) { return false; }
    return !dom.strictly_dominates(merge, bb);
}

struct FlatConstruct {
    BasicBlock *header;
    BasicBlock *merge;
    luisa::unordered_set<BasicBlock *> blocks;
};

// Walk forward from exits to extend region blocks.
void extend_region_from_exits(
    FlatConstruct &fc,
    const DomTree &dom,
    const luisa::unordered_set<BasicBlock *> &loop_headers,
    luisa::unordered_set<BasicBlock *> &visited) noexcept {

    auto exits = find_exit_nodes(fc.blocks);
    luisa::vector<BasicBlock *> exit_list{exits.begin(), exits.end()};
    for (auto *exit_bb : exit_list) {
        if (!exit_bb->is_terminated()) { continue; }
        exit_bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (loop_headers.contains(succ) && dom.contains(exit_bb) && dom.dominates(succ, exit_bb)) {
                return;
            }
            if (visited.contains(succ)) { return; }
            visited.emplace(succ);
            if (block_shares_convergence(succ, fc.header, fc.merge, dom)) {
                fc.blocks.emplace(succ);
                extend_region_from_exits(fc, dom, loop_headers, visited);
            }
        });
    }
}

// Find the innermost region in the tree whose blocks set contains `bb`.
ConvergenceRegion *find_parent_region(ConvergenceRegion *start, BasicBlock *bb) noexcept {
    if (start == nullptr || !start->blocks.contains(bb)) { return nullptr; }
    ConvergenceRegion *candidate = nullptr;
    ConvergenceRegion *next = start;
    while (candidate != next && next != nullptr) {
        candidate = next;
        next = nullptr;
        if (candidate->children.empty()) { return candidate; }
        for (auto &child : candidate->children) {
            if (child->blocks.contains(bb)) {
                next = child.get();
                break;
            }
        }
    }
    return candidate;
}

// Collect all flat constructs (structured constructs with merge blocks).
luisa::vector<FlatConstruct> collect_flat_constructs(FunctionDefinition *def, const DomTree &dom) noexcept {
    luisa::vector<FlatConstruct> result;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        BasicBlock *merge = nullptr;
        if (auto *cm = term->control_flow_merge(); cm != nullptr) { merge = cm->merge_block(); }
        if (term->isa<IfInst>() || term->isa<SwitchInst>()) {
            if (merge == nullptr) { return; }
        } else if (term->isa<LoopInst>() || term->isa<SimpleLoopInst>()) {
            if (merge == nullptr) { return; }
        } else {
            return;
        }
        FlatConstruct fc;
        fc.header = bb;
        fc.merge = merge;
        // Compute initial blocks dominated by header, not by merge.
        luisa::vector<BasicBlock *> work{bb};
        while (!work.empty()) {
            auto *cur = work.back();
            work.pop_back();
            if (cur == merge) { continue; }
            if (!dom.contains(cur)) { continue; }
            if (!dom.contains(merge)) { continue; }
            if (dom.strictly_dominates(merge, cur)) { continue; }
            if (!dom.dominates(bb, cur)) { continue; }
            if (!fc.blocks.emplace(cur).second) { continue; }
            if (!cur->is_terminated()) { continue; }
            cur->traverse_successors(false, [&](BasicBlock *s) noexcept { work.emplace_back(s); });
        }
        result.push_back(std::move(fc));
    });
    return result;
}

} // namespace

const ConvergenceRegion *ConvergenceRegionInfo::find_region(BasicBlock *bb) const noexcept {
    if (top_level == nullptr) { return nullptr; }
    return find_parent_region(top_level.get(), bb);
}

ConvergenceRegionInfo compute_convergence_regions(
    Function *function, const DomTree &dom) noexcept {

    if (function == nullptr) { return {}; }
    auto *def = function->definition();
    if (def == nullptr || def->body_block() == nullptr ||
        !dom.contains(def->body_block())) {
        return {};
    }

    auto loop_headers = collect_loop_headers(def, dom);
    auto flat = collect_flat_constructs(def, dom);

    // Create top-level region with all blocks (even if no constructs exist).
    auto top = luisa::make_unique<ConvergenceRegion>();
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        top->blocks.emplace(bb);
    });

    if (flat.empty()) {
        ConvergenceRegionInfo info;
        info.top_level = std::move(top);
        return info;
    }

    // Extend each region by walking forward from exits.
    for (auto &fc : flat) {
        luisa::unordered_set<BasicBlock *> visited;
        extend_region_from_exits(fc, dom, loop_headers, visited);
    }

    // Sort constructs by dominance depth (outermost first) to ensure
    // find_parent_region sees parents before children during tree building.
    luisa::unordered_map<BasicBlock *, size_t> depth_cache;
    auto compute_depth = [&](BasicBlock *bb) noexcept -> size_t {
        auto it = depth_cache.find(bb);
        if (it != depth_cache.end()) { return it->second; }
        size_t d = 0;
        auto *node = dom.node_or_null(bb);
        while (node != nullptr && node->parent() != nullptr) {
            ++d;
            node = node->parent();
        }
        depth_cache.emplace(bb, d);
        return d;
    };
    luisa::sort(flat.begin(), flat.end(), [&](const FlatConstruct &a, const FlatConstruct &b) {
        return compute_depth(a.header) < compute_depth(b.header);
    });

    // Build tree: for each construct, find parent via entry containment.
    for (auto &fc : flat) {
        auto cr = luisa::make_unique<ConvergenceRegion>();
        cr->entry = fc.header;
        cr->convergence_merge = fc.merge;
        cr->blocks = std::move(fc.blocks);

        auto *parent = find_parent_region(top.get(), cr->entry);
        LUISA_ASSERT(parent != nullptr, "Convergence region must have a parent.");
        cr->parent = parent;
        parent->children.push_back(std::move(cr));
    }

    ConvergenceRegionInfo info;
    info.top_level = std::move(top);
    return info;
}

}// namespace luisa::compute::xir
