#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
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

// Walk forward from `from`, following successors (ignoring back-edges to loop headers),
// and collect all blocks on paths that eventually reach a block matching `is_match`.
luisa::unordered_set<BasicBlock *> find_paths_to_match(
    BasicBlock *from,
    const DomTree &dom,
    const luisa::unordered_set<BasicBlock *> &loop_headers,
    luisa::function<bool(BasicBlock *)> is_match) noexcept {

    luisa::unordered_set<BasicBlock *> result;
    if (is_match(from)) { result.emplace(from); }

    if (!from->is_terminated()) { return result; }
    from->traverse_successors(false, [&](BasicBlock *to) noexcept {
        if (loop_headers.contains(to) && dom.contains(from) && dom.dominates(to, from)) {
            return;
        }
        auto child = find_paths_to_match(to, dom, loop_headers, is_match);
        if (child.empty()) { return; }
        result.insert(child.begin(), child.end());
        result.emplace(from);
    });
    return result;
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

// Check if a block belongs to a structure with the given merge block.
// A block "belongs" if it is dominated by the merge's corresponding header
// and not strictly dominated by the merge. We use the merge itself as the token.
bool block_shares_convergence(BasicBlock *bb, BasicBlock *merge, const DomTree &dom) noexcept {
    if (!dom.contains(bb) || !dom.contains(merge)) { return false; }
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
            if (block_shares_convergence(succ, fc.merge, dom)) {
                fc.blocks.emplace(succ);
                extend_region_from_exits(fc, dom, loop_headers, visited);
            }
        });
    }
}

// Find the innermost region in the tree whose blocks set contains `bb`.
ConvergenceRegion *find_parent_region(ConvergenceRegion *start, BasicBlock *bb) noexcept {
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

    auto *def = function->definition();
    if (def == nullptr) { return {}; }

    auto loop_headers = collect_loop_headers(def, dom);
    auto flat = collect_flat_constructs(def, dom);
    if (flat.empty()) { return {}; }

    // Extend each region by walking forward from exits.
    for (auto &fc : flat) {
        luisa::unordered_set<BasicBlock *> visited;
        extend_region_from_exits(fc, dom, loop_headers, visited);
    }

    // Create top-level region with all blocks.
    auto top = luisa::make_unique<ConvergenceRegion>();
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        top->blocks.emplace(bb);
    });

    // Build tree: for each construct, find parent via entry containment.
    for (auto &fc : flat) {
        auto cr = luisa::make_unique<ConvergenceRegion>();
        cr->entry = fc.header;
        cr->convergence_merge = fc.merge;
        cr->blocks = std::move(fc.blocks);
        cr->exits = find_exit_nodes(cr->blocks);

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
