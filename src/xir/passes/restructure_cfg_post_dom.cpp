#include "restructure_cfg_post_dom.h"

#include <cstdint>
#include <limits>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/unreachable.h>

namespace luisa::compute::xir::detail {

namespace {

constexpr auto invalid_index = std::numeric_limits<size_t>::max();

struct NumberedEdge {
    size_t source;
    size_t target;
};

[[nodiscard]] bool has_explicit_sink_terminator(
    BasicBlock *block) noexcept {
    auto *terminator = block->terminator();
    return terminator->isa<ReturnInst>() ||
           terminator->isa<UnreachableInst>() ||
           terminator->isa<RasterDiscardInst>();
}

}// namespace

bool is_restructure_cfg_sink(BasicBlock *block) noexcept {
    if (!block->is_terminated()) { return true; }
    if (has_explicit_sink_terminator(block)) {
        return true;
    }
    auto has_successor = false;
    block->traverse_successors(
        false, [&](BasicBlock *) noexcept {
            has_successor = true;
        });
    return !has_successor;
}

RestructurePostDomInfo compute_restructure_post_dom(
    FunctionDefinition *definition,
    RestructurePostDomStats *stats) noexcept {
    RestructurePostDomStats local_stats;

    // Value-number the owned CFG once. Pointer hashing is confined to this
    // boundary; DFS and the immediate-dominator fixed point use only IDs.
    luisa::vector<BasicBlock *> blocks;
    definition->traverse_basic_blocks(
        [&](BasicBlock *block) noexcept {
            blocks.emplace_back(block);
        });
    const auto block_count = blocks.size();
    local_stats.numbered_block_count = block_count;
    luisa::unordered_map<BasicBlock *, size_t> block_indices;
    block_indices.reserve(block_count);
    for (auto index = size_t{0u};
         index < block_count; ++index) {
        block_indices.emplace(blocks[index], index);
    }

    // Number each local edge once, retaining definition/successor order. The
    // two CSR views are the original successor graph and its reverse.
    luisa::vector<NumberedEdge> edges;
    luisa::vector<uint8_t> sinks(block_count, 0u);
    for (auto source = size_t{0u};
         source < block_count; ++source) {
        auto *block = blocks[source];
        if (!block->is_terminated()) {
            sinks[source] = 1u;
            continue;
        }
        auto has_successor = false;
        block->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                has_successor = true;
                if (auto iter = block_indices.find(successor);
                    iter != block_indices.end()) {
                    edges.emplace_back(NumberedEdge{
                        source, iter->second});
                }
            });
        sinks[source] = static_cast<uint8_t>(
            has_explicit_sink_terminator(block) ||
            !has_successor);
    }
    local_stats.numbered_edge_count = edges.size();

    luisa::vector<size_t> successor_offsets(
        block_count + 1u, 0u);
    luisa::vector<size_t> predecessor_offsets(
        block_count + 1u, 0u);
    for (auto edge : edges) {
        ++successor_offsets[edge.source + 1u];
        ++predecessor_offsets[edge.target + 1u];
    }
    for (auto index = size_t{0u};
         index < block_count; ++index) {
        successor_offsets[index + 1u] +=
            successor_offsets[index];
        predecessor_offsets[index + 1u] +=
            predecessor_offsets[index];
    }
    luisa::vector<size_t> successor_indices(
        edges.size(), invalid_index);
    luisa::vector<size_t> predecessor_indices(
        edges.size(), invalid_index);
    auto successor_cursors = successor_offsets;
    auto predecessor_cursors = predecessor_offsets;
    for (auto edge : edges) {
        successor_indices[successor_cursors[edge.source]++] = edge.target;
        predecessor_indices[predecessor_cursors[edge.target]++] = edge.source;
    }

    // In the reversed CFG, a synthetic root has one outgoing edge to every
    // sink. Iterative DFS yields reverse postorder without host recursion.
    const auto virtual_exit_index = block_count;
    luisa::vector<size_t> sink_indices;
    for (auto index = size_t{0u};
         index < block_count; ++index) {
        if (sinks[index] != 0u) {
            sink_indices.emplace_back(index);
        }
    }
    struct DFSFrame {
        size_t node;
        size_t next_predecessor;
    };
    luisa::vector<uint8_t> visited(
        block_count + 1u, 0u);
    luisa::vector<DFSFrame> stack;
    luisa::vector<size_t> postorder;
    visited[virtual_exit_index] = 1u;
    stack.emplace_back(DFSFrame{
        virtual_exit_index, 0u});
    while (!stack.empty()) {
        auto &frame = stack.back();
        const auto predecessor_count =
            frame.node == virtual_exit_index ?
                sink_indices.size() :
                predecessor_offsets[frame.node + 1u] -
                    predecessor_offsets[frame.node];
        if (frame.next_predecessor < predecessor_count) {
            const auto predecessor =
                frame.node == virtual_exit_index ?
                    sink_indices[frame.next_predecessor++] :
                    predecessor_indices[predecessor_offsets[frame.node] +
                                        frame.next_predecessor++];
            if (visited[predecessor] == 0u) {
                visited[predecessor] = 1u;
                stack.emplace_back(DFSFrame{
                    predecessor, 0u});
            }
        } else {
            postorder.emplace_back(frame.node);
            stack.pop_back();
        }
    }
    luisa::vector<size_t> reverse_postorder{
        postorder.rbegin(), postorder.rend()};
    LUISA_DEBUG_ASSERT(
        !reverse_postorder.empty() &&
            reverse_postorder.front() == virtual_exit_index,
        "Post-dominator reverse CFG has no virtual root.");
    local_stats.active_block_count =
        reverse_postorder.size() - 1u;

    // Cooper-Harvey-Kennedy on the reversed CFG. Solver indices are reverse-
    // postorder IDs, so every resolved idom moves toward a smaller ID.
    luisa::vector<size_t> solver_indices(
        block_count + 1u, invalid_index);
    for (auto index = size_t{0u};
         index < reverse_postorder.size(); ++index) {
        solver_indices[reverse_postorder[index]] = index;
    }
    luisa::vector<size_t> immediate_dominators(
        reverse_postorder.size(), invalid_index);
    immediate_dominators.front() = 0u;
    const auto intersect = [&](size_t lhs,
                               size_t rhs) noexcept {
        while (lhs != rhs) {
            while (lhs > rhs) {
                lhs = immediate_dominators[lhs];
                ++local_stats.intersect_step_count;
            }
            while (rhs > lhs) {
                rhs = immediate_dominators[rhs];
                ++local_stats.intersect_step_count;
            }
        }
        return lhs;
    };
    for (;;) {
        ++local_stats.fixed_point_iteration_count;
        auto changed = false;
        for (auto solver_index = size_t{1u};
             solver_index < reverse_postorder.size();
             ++solver_index) {
            ++local_stats.fixed_point_block_visit_count;
            const auto block_index =
                reverse_postorder[solver_index];
            auto new_idom = invalid_index;
            if (sinks[block_index] != 0u) {
                ++local_stats.fixed_point_edge_visit_count;
                new_idom = 0u;
            } else {
                for (auto edge_index =
                         successor_offsets[block_index];
                     edge_index <
                     successor_offsets[block_index + 1u];
                     ++edge_index) {
                    ++local_stats.fixed_point_edge_visit_count;
                    const auto successor_solver_index =
                        solver_indices[successor_indices[edge_index]];
                    if (successor_solver_index == invalid_index ||
                        immediate_dominators[successor_solver_index] ==
                            invalid_index) {
                        continue;
                    }
                    new_idom = new_idom == invalid_index ?
                                   successor_solver_index :
                                   intersect(
                                       successor_solver_index,
                                       new_idom);
                }
            }
            if (new_idom != invalid_index &&
                immediate_dominators[solver_index] !=
                    new_idom) {
                immediate_dominators[solver_index] =
                    new_idom;
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    for (auto idom : immediate_dominators) {
        LUISA_ASSERT(
            idom != invalid_index,
            "Sink-reachable restructure CFG block has no "
            "immediate post-dominator.");
    }

    // Convert back to the historical pointer API only after the dense solve.
    static int virtual_exit_sentinel = 0;
    auto *virtual_exit = reinterpret_cast<BasicBlock *>(
        &virtual_exit_sentinel);
    RestructurePostDomInfo result;
    result.virtual_exit = virtual_exit;
    result.ipostdom.reserve(reverse_postorder.size());
    result.ipostdom.emplace(virtual_exit, virtual_exit);
    for (auto solver_index = size_t{1u};
         solver_index < reverse_postorder.size();
         ++solver_index) {
        const auto block_index =
            reverse_postorder[solver_index];
        const auto parent_solver_index =
            immediate_dominators[solver_index];
        auto *parent = parent_solver_index == 0u ?
                           virtual_exit :
                           blocks[reverse_postorder[parent_solver_index]];
        result.ipostdom.emplace(
            blocks[block_index], parent);
    }
    if (stats != nullptr) { *stats = local_stats; }
    return result;
}

}// namespace luisa::compute::xir::detail
