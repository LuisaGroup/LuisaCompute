#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/passes/convergence_region.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include "helpers.h"

#include <array>
#include <cstdlib>
#include <limits>

namespace luisa::compute::xir {

namespace {

[[nodiscard]] bool restructure_trace_enabled() noexcept {
    static const auto enabled = []() noexcept {
        if (auto value = std::getenv("LUISA_XIR_TRACE_PASSES")) {
            return luisa::string_view{value} == "1";
        }
        return false;
    }();
    return enabled;
}

[[nodiscard]] bool restructure_verify_intermediate_enabled() noexcept {
    if (auto value =
            std::getenv("LUISA_XIR_VERIFY_INTERMEDIATE")) {
        return luisa::string_view{value} == "1";
    }
    return false;
}

struct ScopedTimer {
    Clock clock;
    const char *name;
    ScopedTimer(const char *n) noexcept
        : name(n) {
    }
    ~ScopedTimer() noexcept {
        if (restructure_trace_enabled()) {
            auto ms = clock.toc();
            LUISA_VERBOSE_WITH_LOCATION(
                "[restructure_cfg] {}: {:.3f} ms",
                name, ms);
        }
    }
};

struct CFGTraceStats {
    size_t block_count{0u};
    size_t instruction_count{0u};
    size_t raw_conditional_count{0u};
    size_t raw_indexed_count{0u};
    size_t structured_loop_count{0u};
    size_t structured_selection_count{0u};
};

[[nodiscard]] CFGTraceStats trace_stats(
    FunctionDefinition *def) noexcept {
    CFGTraceStats stats;
    if (def == nullptr) { return stats; }
    for (auto *block : def->basic_blocks()) {
        ++stats.block_count;
        for (auto *instruction : block->instructions()) {
            ++stats.instruction_count;
        }
        if (!block->is_terminated()) { continue; }
        auto *terminator = block->terminator();
        stats.raw_conditional_count +=
            terminator->isa<ConditionalBranchInst>() ? 1u : 0u;
        stats.raw_indexed_count +=
            terminator->isa<IndexedBranchInst>() ? 1u : 0u;
        stats.structured_loop_count +=
            terminator->isa<LoopInst>() ||
                    terminator->isa<SimpleLoopInst>() ?
                1u :
                0u;
        stats.structured_selection_count +=
            terminator->isa<IfInst>() ||
                    terminator->isa<SwitchInst>() ?
                1u :
                0u;
    }
    return stats;
}

void trace_cfg(
    luisa::string_view stage,
    FunctionDefinition *def) noexcept {
    if (!restructure_trace_enabled()) { return; }
    auto stats = trace_stats(def);
    LUISA_VERBOSE_WITH_LOCATION(
        "[restructure_cfg] {}: blocks={}, instructions={}, "
        "raw_conditional={}, raw_indexed={}, structured_loop={}, "
        "structured_selection={}.",
        stage,
        stats.block_count,
        stats.instruction_count,
        stats.raw_conditional_count,
        stats.raw_indexed_count,
        stats.structured_loop_count,
        stats.structured_selection_count);
}

void trace_module_definition(
    luisa::string_view stage,
    size_t index,
    FunctionDefinition *def) noexcept {
    if (!restructure_trace_enabled() || def == nullptr) { return; }
    auto name = def->name();
    LUISA_VERBOSE_WITH_LOCATION(
        "[restructure_cfg] {} definition {}: tag={}, name={}.",
        stage, index, to_string(def->derived_function_tag()),
        name ? *name : luisa::string_view{"<unnamed>"});
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

    // Worklist-driven post-dominator fixed point. The naive full-scan loop is
    // O(N^3) in the block count (N passes, each re-intersecting successor
    // chains for every block); re-processing only the predecessors of blocks
    // whose immediate post-dominator changed converges from the top (nullptr)
    // to the identical maximum fixed point while doing a small, practically
    // linear amount of work.
    luisa::vector<BasicBlock *> worklist;
    for (auto *bb : rpo) {
        if (bb == virt) { continue; }
        if (is_sink(bb)) { worklist.emplace_back(bb); }
    }
    while (!worklist.empty()) {
        auto *bb = worklist.back();
        worklist.pop_back();

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
            bb->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                worklist.emplace_back(pred);
            });
        }
        if (new_ipostdom != nullptr) { set_processed(bb); }
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
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto *sw = static_cast<IndexedBranchTerminatorInstruction *>(term);
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
        default: break;
    }
    return changed;
}

[[nodiscard]] bool terminator_targets(Instruction *term, BasicBlock *target) noexcept {
    if (term == nullptr || target == nullptr) { return false; }
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH:
            return static_cast<BranchInst *>(term)->target_block() == target;
        case DerivedInstructionTag::BREAK:
        case DerivedInstructionTag::CONTINUE:
            return static_cast<BranchTerminatorInstruction *>(term)
                       ->target_block() == target;
        case DerivedInstructionTag::IF:
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto *branch = static_cast<
                ConditionalBranchTerminatorInstruction *>(term);
            return branch->true_block() == target || branch->false_block() == target;
        }
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto *sw = static_cast<IndexedBranchTerminatorInstruction *>(term);
            if (sw->default_block() == target) { return true; }
            for (size_t i = 0u; i < sw->case_count(); i++) {
                if (sw->case_block(i) == target) { return true; }
            }
            return false;
        }
        default: return false;
    }
}

void fix_degenerate_terminator(BasicBlock *bb) noexcept;

struct StructuredLoopExitInfo;

[[nodiscard]] luisa::unordered_set<BasicBlock *>
collect_enclosing_loop_exits(FunctionDefinition *def,
                             BasicBlock *header,
                             const DomTree &dom,
                             const luisa::vector<StructuredLoopExitInfo> *
                                 precomputed_loops = nullptr) noexcept;
[[nodiscard]] BasicBlock *
structured_statement_merge(Instruction *term) noexcept;
[[nodiscard]] BasicBlock *
canonical_exit_target(BasicBlock *target) noexcept;

// Global post-dominance loses a selection's lexical merge when an arm exits an
// enclosing loop or terminates the function. Recover the nearest normal-path
// convergence instead by ignoring enclosing loop boundaries and comparing
// shortest reachability from each distinct arm.
[[nodiscard]] BasicBlock *infer_selection_merge(
    FunctionDefinition *def,
    BasicBlock *header,
    luisa::span<BasicBlock *const> entries,
    const DomTree &dom,
    const luisa::vector<StructuredLoopExitInfo> *precomputed_loops =
        nullptr) noexcept {
    if (def == nullptr || header == nullptr || entries.empty()) {
        return nullptr;
    }
    // The dominator tree is rooted at the executable function entry. An owned
    // but unreachable structural shell deliberately has no node in that tree;
    // its raw selection is rebuilt with a synthetic merge by the caller.
    if (!dom.contains(header)) { return nullptr; }
    auto boundaries =
        collect_enclosing_loop_exits(def, header, dom, precomputed_loops);
    luisa::vector<luisa::unordered_map<BasicBlock *, size_t>>
        distances;
    distances.reserve(entries.size());
    for (auto *entry : entries) {
        luisa::unordered_map<BasicBlock *, size_t> distance;
        if (entry != nullptr && entry != header &&
            !boundaries.contains(entry) && dom.contains(entry) &&
            dom.dominates(header, entry)) {
            luisa::vector<BasicBlock *> queue{entry};
            distance.emplace(entry, 0u);
            for (auto cursor = 0u; cursor < queue.size();
                 cursor++) {
                auto *block = queue[cursor];
                if (!block->is_terminated()) { continue; }
                auto next_distance = distance.at(block) + 1u;
                block->traverse_successors(
                    false, [&](BasicBlock *successor) noexcept {
                        if (successor == nullptr ||
                            successor == header ||
                            boundaries.contains(successor) ||
                            !dom.contains(successor)) {
                            return;
                        }
                        if (auto [iter, inserted] =
                                distance.try_emplace(
                                    successor, next_distance);
                            inserted &&
                            dom.dominates(header, successor)) {
                            queue.emplace_back(successor);
                        } else if (next_distance < iter->second) {
                            iter->second = next_distance;
                        }
                    });
            }
        }
        distances.emplace_back(std::move(distance));
    }

    struct MergeScore {
        BasicBlock *block{nullptr};
        size_t support{0u};
        size_t max_distance{
            std::numeric_limits<size_t>::max()};
        size_t total_distance{
            std::numeric_limits<size_t>::max()};
    };
    MergeScore best;
    MergeScore boundary_proxy_best;
    auto consider =
        [](MergeScore &score, BasicBlock *candidate,
           size_t support, size_t max_distance,
           size_t total_distance) noexcept {
            if (support > score.support ||
                (support == score.support &&
                 max_distance < score.max_distance) ||
                (support == score.support &&
                 max_distance == score.max_distance &&
                 total_distance < score.total_distance)) {
                score = {
                    candidate, support,
                    max_distance, total_distance};
            }
        };
    // Only blocks actually reached by at least one BFS can score. A candidate
    // with support >= 2 must appear in every entry's distance map, so the
    // first map's key set is a superset of all plausible merge blocks; scoring
    // the whole function here would make inference O(blocks x entries) per
    // selection header, which is quadratic in the module size.
    for (auto &&[candidate, _] : distances.front()) {
        if (candidate == nullptr || candidate == header ||
            boundaries.contains(candidate)) {
            continue;
        }
        auto support = size_t{0u};
        auto max_distance = size_t{0u};
        auto total_distance = size_t{0u};
        for (auto &&distance : distances) {
            if (auto iter = distance.find(candidate);
                iter != distance.end()) {
                support++;
                max_distance =
                    std::max(max_distance, iter->second);
                total_distance += iter->second;
            }
        }
        if (support < std::min<size_t>(2u, entries.size())) {
            continue;
        }
        if (boundaries.contains(
                canonical_exit_target(candidate))) {
            // A real convergence block immediately before an enclosing loop
            // boundary is still a valid selection merge. Keep it as a
            // secondary class so an ordinary in-region convergence retains
            // the historical priority. If no ordinary convergence exists,
            // this private proxy must win over the one-normal-arm heuristic:
            // the latter can place the merge in front of only one arm and
            // create a post-merge re-entry into the other.
            consider(
                boundary_proxy_best, candidate, support,
                max_distance, total_distance);
        } else {
            consider(
                best, candidate, support,
                max_distance, total_distance);
        }
    }
    if (best.block != nullptr) { return best.block; }
    if (boundary_proxy_best.block != nullptr) {
        return boundary_proxy_best.block;
    }

    // A selection nested inside an already-recovered selection may have only
    // one normal arm: the other arms can return or leave an enclosing loop.
    // In that case its nearest enclosing selection merge is the lexical
    // continuation even though it is reachable from only one arm.
    for (auto *candidate_header : def->basic_blocks()) {
        if (candidate_header == nullptr ||
            candidate_header == header ||
            !candidate_header->is_terminated() ||
            !dom.contains(candidate_header) ||
            !dom.dominates(candidate_header, header)) {
            continue;
        }
        auto *candidate_term = candidate_header->terminator();
        if (!candidate_term->isa<IfInst>() &&
            !candidate_term->isa<SwitchInst>()) {
            continue;
        }
        auto *candidate = structured_statement_merge(candidate_term);
        if (candidate == nullptr || candidate == header ||
            boundaries.contains(candidate) ||
            boundaries.contains(canonical_exit_target(candidate))) {
            continue;
        }
        auto min_distance = std::numeric_limits<size_t>::max();
        for (auto &&distance : distances) {
            if (auto iter = distance.find(candidate);
                iter != distance.end()) {
                min_distance = std::min(min_distance, iter->second);
            }
        }
        if (min_distance < best.max_distance) {
            best.block = candidate;
            best.max_distance = min_distance;
        }
    }
    if (best.block != nullptr) { return best.block; }

    // If an arm immediately continues with a recovered structured statement,
    // place the current selection's fresh merge in front of that statement.
    // This is the one-normal-arm form of `if (cond) break; continuation;`.
    for (auto *candidate : def->basic_blocks()) {
        if (candidate == nullptr || candidate == header ||
            boundaries.contains(candidate) ||
            boundaries.contains(canonical_exit_target(candidate)) ||
            !candidate->is_terminated() ||
            structured_statement_merge(candidate->terminator()) == nullptr) {
            continue;
        }
        auto min_distance = std::numeric_limits<size_t>::max();
        for (auto &&distance : distances) {
            if (auto iter = distance.find(candidate);
                iter != distance.end()) {
                min_distance = std::min(min_distance, iter->second);
            }
        }
        if (min_distance < best.max_distance) {
            best.block = candidate;
            best.max_distance = min_distance;
        }
    }
    return best.block;
}

[[nodiscard]] SwitchInst *replace_indexed_branch_with_switch(
    IndexedBranchInst *indexed_branch,
    BasicBlock *merge) noexcept {
    if (indexed_branch == nullptr || merge == nullptr) { return nullptr; }
    auto *block = indexed_branch->parent_block();
    auto *value = indexed_branch->value();
    auto *default_block = indexed_branch->default_block();
    luisa::vector<std::pair<
        IndexedBranchTerminatorInstruction::case_value_type,
        BasicBlock *>>
        cases;
    cases.reserve(indexed_branch->case_count());
    for (auto i = 0u; i < indexed_branch->case_count(); i++) {
        cases.emplace_back(
            indexed_branch->case_value(i),
            indexed_branch->case_block(i));
    }
    auto removed = indexed_branch->remove_self();
    XIRBuilder b;
    b.set_insertion_point(block);
    auto *switch_inst = b.switch_(value);
    switch_inst->set_default_block(default_block);
    switch_inst->set_merge_block(merge);
    for (auto [case_value, case_block] : cases) {
        switch_inst->add_case(case_value, case_block);
    }
    for (auto *metadata : removed->metadata_list()) {
        switch_inst->metadata_list().push_front(metadata->clone());
    }
    return switch_inst;
}

// Convert every raw multi-way branch into a structured SwitchInst. A real
// common post-dominator is split through a fresh per-switch merge block so
// nested selections never share merge ownership. If no real post-dominator
// exists (for example all arms return or leave an enclosing loop), the
// structured merge is an unreachable block; later selection-exit
// canonicalization preserves legal break/continue exits and routes other
// multi-exit paths through a dispatch when necessary.
void restructure_indexed_branches(
    FunctionDefinition *def, RestructureCFGInfo &info) noexcept {
    for (;;) {
        auto dom = compute_dom_tree(def, false);
        auto pdom = compute_post_dom(def);
        BasicBlock *header = nullptr;
        IndexedBranchInst *indexed_branch = nullptr;
        size_t best_depth = 0u;
        // Structure is an invariant of every block owned by the definition,
        // including unreachable structural shells retained by DCE. Walking
        // only the executable entry traversal can therefore leave raw
        // IndexedBranchInst nodes behind in such shells.
        for (auto *bb : def->basic_blocks()) {
            if (bb == nullptr) { continue; }
            if (!bb->is_terminated() ||
                !bb->terminator()->isa<IndexedBranchInst>()) {
                continue;
            }
            auto depth = dom_depth(dom, bb);
            if (indexed_branch == nullptr || depth > best_depth) {
                header = bb;
                indexed_branch =
                    static_cast<IndexedBranchInst *>(bb->terminator());
                best_depth = depth;
            }
        }
        if (indexed_branch == nullptr) { break; }

        luisa::vector<BasicBlock *> entries;
        luisa::unordered_set<BasicBlock *> unique_entries;
        auto append_entry = [&](BasicBlock *entry) noexcept {
            if (entry != nullptr && unique_entries.emplace(entry).second) {
                entries.emplace_back(entry);
            }
        };
        append_entry(indexed_branch->default_block());
        for (auto i = 0u; i < indexed_branch->case_count(); i++) {
            append_entry(indexed_branch->case_block(i));
        }
        auto entry_span = luisa::span<BasicBlock *const>{
            entries.data(), entries.size()};
        auto *common_merge =
            infer_selection_merge(def, header, entry_span, dom);
        if (common_merge == nullptr && dom.contains(header)) {
            common_merge = common_postdom(pdom, entry_span);
        }
        auto synthetic_merge =
            common_merge == nullptr ||
            common_merge == pdom.virtual_exit ||
            common_merge == header;

        auto *structural_merge = def->create_basic_block();
        {
            XIRBuilder b;
            b.set_insertion_point(structural_merge);
            if (synthetic_merge) {
                b.unreachable_();
            } else {
                b.br(common_merge);
            }
        }

        if (!synthetic_merge) {
            // Direct header-to-merge cases must enter the fresh structural
            // merge instead of bypassing it.
            (void)retarget_terminator(
                indexed_branch, common_merge, structural_merge);

            luisa::unordered_set<BasicBlock *> visited;
            luisa::vector<BasicBlock *> work;
            for (auto *entry : entries) {
                if (entry != common_merge && entry != structural_merge) {
                    work.emplace_back(entry);
                }
            }
            while (!work.empty()) {
                auto *block = work.back();
                work.pop_back();
                if (block == nullptr || block == header ||
                    block == common_merge || block == structural_merge ||
                    !visited.emplace(block).second) {
                    continue;
                }
                if (dom.contains(header) && dom.contains(block) &&
                    !dom.dominates(header, block)) {
                    continue;
                }
                if (!block->is_terminated()) { continue; }
                auto *term = block->terminator();
                luisa::vector<BasicBlock *> successors;
                block->traverse_successors(
                    false, [&](BasicBlock *successor) noexcept {
                        successors.emplace_back(successor);
                    });
                if (term->isa<BranchInst>() ||
                    term->isa<ConditionalBranchInst>() ||
                    term->isa<IndexedBranchInst>()) {
                    (void)retarget_terminator(
                        term, common_merge, structural_merge);
                    fix_degenerate_terminator(block);
                }
                for (auto *successor : successors) {
                    if (successor != common_merge &&
                        successor != structural_merge) {
                        work.emplace_back(successor);
                    }
                }
            }
        }

        auto *switch_inst = replace_indexed_branch_with_switch(
            indexed_branch, structural_merge);
        LUISA_ASSERT(
            switch_inst != nullptr,
            "Failed to reconstruct SwitchInst from IndexedBranchInst.");
        ++info.restructured_switch_count;
    }
}

[[nodiscard]] bool retarget_loop_exit_to(Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
    if (term == nullptr) { return false; }
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH:
        case DerivedInstructionTag::CONDITIONAL_BRANCH:
        case DerivedInstructionTag::INDEXED_BRANCH:
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

[[nodiscard]] bool retarget_structured_exit_to(
    Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
    if (retarget_loop_exit_to(term, from, to)) { return true; }
    if (term == nullptr ||
        (!term->isa<BreakInst>() && !term->isa<ContinueInst>())) {
        return false;
    }
    auto *branch = static_cast<BranchTerminatorInstruction *>(term);
    if (branch->target_block() != from) { return false; }
    auto *parent = term->parent_block();
    term->remove_self();
    XIRBuilder builder;
    builder.set_insertion_point(parent);
    builder.br(to);
    return true;
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

[[nodiscard]] bool trivial_branch_chain_reaches(
    BasicBlock *from, BasicBlock *target) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    auto *cur = from;
    while (cur != nullptr && visited.emplace(cur).second) {
        if (cur == target) { return true; }
        cur = trivial_branch_target(cur);
    }
    return false;
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
    if (!term->isa<ConditionalBranchInst>() &&
        !term->isa<IndexedBranchInst>() &&
        !term->isa<SwitchInst>()) {
        return false;
    }
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
    if (!term->isa<ConditionalBranchInst>() &&
        !term->isa<IndexedBranchInst>() &&
        !term->isa<SwitchInst>()) {
        return false;
    }
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
    auto dom = compute_dom_tree(def, false);
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
        // Reachability alone does not define lexical loop membership: an
        // enclosing cycle may reach a path which bypasses this loop and joins
        // at its merge. Rewriting such an edge as Break would attach it to a
        // loop that does not dominate the source, and a later single-exit
        // rewrite would leave a stale non-enclosing break target. Every block
        // owned by a structured loop is dominated by its declared entry.
        if (!dom.contains(loop_entry) || !dom.contains(bb) ||
            !dom.dominates(loop_entry, bb)) {
            return;
        }
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
        auto *term = bb->terminator();
        // Loop boundary normalization is local to one construct scope.
        // Descending into a nested Loop or Switch would rewrite an edge in the
        // child as a Break/Continue of the parent, violating the nearest-scope
        // invariant. Treat nested break scopes as atomic and continue from
        // their declared merge; fixup_construct_exits owns any genuine
        // cross-hierarchy exit from the child.
        if (term->isa<LoopInst>() ||
            term->isa<SimpleLoopInst>() ||
            term->isa<SwitchInst>()) {
            if (auto *nested_merge =
                    structured_statement_merge(term);
                nested_merge != nullptr &&
                nested_merge != merge) {
                enqueue(nested_merge);
            }
            continue;
        }
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

[[nodiscard]] bool is_canonical_loop_break_path(
    BasicBlock *target, BasicBlock *merge) noexcept {
    if (target == nullptr || merge == nullptr) { return false; }
    auto *resolved = trivial_branch_chain_target(target);
    if (resolved == merge) { return true; }
    return has_only_terminator(resolved) &&
           resolved->terminator()->isa<BreakInst>() &&
           static_cast<BreakInst *>(resolved->terminator())
                   ->target_block() == merge;
}

[[nodiscard]] bool is_loop_break_target(BasicBlock *target,
                                        BasicBlock *merge) noexcept {
    if (target == nullptr || merge == nullptr) { return false; }
    // A loop merge may be a pure forwarding boundary M ->* T. An edge from
    // inside the loop directly to T has the same executable continuation as
    // Break(M), but it bypasses the declared single-exit boundary and is not
    // legal structured control flow. Treat only side-effect-free forwarding
    // chains as equivalent; canonical_exit_target stops at the first block
    // containing executable payload.
    if (canonical_exit_target(target) ==
        canonical_exit_target(merge)) {
        return true;
    }
    return is_canonical_loop_break_path(target, merge);
}

enum struct LoopBoundaryTargetKind {
    NONE,
    BREAK,
    CONTINUE,
    MIXED,
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
        if (kind != k && kind != LoopBoundaryTargetKind::MIXED) {
            kind = LoopBoundaryTargetKind::MIXED;
        }
        return true;
    };
    luisa::unordered_set<BasicBlock *> visited;
    luisa::vector<BasicBlock *> work{target};
    auto *canonical_merge = canonical_exit_target(merge);
    while (!work.empty()) {
        auto *bb = work.back();
        work.pop_back();
        if (bb == nullptr || !visited.emplace(bb).second) { continue; }
        if (bb == merge ||
            canonical_exit_target(bb) == canonical_merge) {
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
        if (term->isa<LoopInst>() || term->isa<SimpleLoopInst>()) {
            auto *control_flow_merge = term->control_flow_merge();
            if (control_flow_merge == nullptr || control_flow_merge->merge_block() == nullptr) {
                return LoopBoundaryTargetKind::NONE;
            }
            work.emplace_back(control_flow_merge->merge_block());
            continue;
        }
        traverse_structured_successors(bb, [&](BasicBlock *succ) noexcept {
            if (succ != nullptr) { work.emplace_back(succ); }
        });
    }
    return kind;
}

[[nodiscard]] bool normalize_one_loop_boundary_conditional_branch(FunctionDefinition *def,
                                                                  luisa::unordered_set<BasicBlock *> &
                                                                      exit_dispatch_headers,
                                                                  const luisa::unordered_set<BasicBlock *> &
                                                                      generated_exit_dispatch_headers) noexcept {
    auto dom = compute_dom_tree(def, false);
    struct LoopSite {
        BasicBlock *owner{nullptr};
        BasicBlock *entry{nullptr};
        BasicBlock *body{nullptr};
        BasicBlock *continue_target{nullptr};
        BasicBlock *merge{nullptr};
        BasicBlock *selection_merge{nullptr};
        size_t depth{0u};
    };
    luisa::vector<LoopSite> loops;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            loops.emplace_back(
                bb, loop->prepare_block(),
                loop->body_block(),
                loop->update_block(), loop->merge_block(),
                loop->update_block(),
                dom.contains(bb) ? dom_depth(dom, bb) : 0u);
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            loops.emplace_back(
                bb, loop->body_block(),
                loop->body_block(),
                loop->body_block(), loop->merge_block(),
                loop->merge_block(),
                dom.contains(bb) ? dom_depth(dom, bb) : 0u);
        }
    });

    struct Candidate {
        BasicBlock *branch_block{nullptr};
        BasicBlock *true_target{nullptr};
        BasicBlock *false_target{nullptr};
        BasicBlock *loop_entry{nullptr};
        BasicBlock *continue_target{nullptr};
        BasicBlock *merge{nullptr};
        BasicBlock *selection_merge{nullptr};
        Value *condition{nullptr};
        size_t loop_depth{0u};
    };
    luisa::vector<Candidate> candidates;
    auto singular_boundary = [](auto kind) noexcept {
        return kind == LoopBoundaryTargetKind::BREAK ||
               kind == LoopBoundaryTargetKind::CONTINUE;
    };
    // Loop membership is a structural property here, not merely dominance.
    // Walk each loop's executable region from its body, stop at its own
    // prepare/merge boundary, and treat a nested construct as one node by
    // continuing from its merge. Only after every containing loop has been
    // examined may an active dispatch marker be released: the same branch can
    // be ordinary for an inner loop and a boundary guard for an outer loop.
    for (auto site : loops) {
        if (site.owner == nullptr || site.entry == nullptr ||
            site.body == nullptr ||
            site.continue_target == nullptr ||
            site.merge == nullptr ||
            site.selection_merge == nullptr) {
            continue;
        }
        auto append_candidate =
            [&](BasicBlock *branch_block,
                bool allow_one_sided_boundary) noexcept {
                if (branch_block == nullptr ||
                    !branch_block->is_terminated() ||
                    !branch_block->terminator()
                         ->isa<ConditionalBranchInst>()) {
                    return;
                }
                auto *cbr =
                    static_cast<ConditionalBranchInst *>(
                        branch_block->terminator());
                auto *t = cbr->true_block();
                auto *f = cbr->false_block();
                auto true_kind =
                    classify_loop_boundary_path(
                        t, site.continue_target,
                        site.entry, site.merge);
                auto false_kind =
                    classify_loop_boundary_path(
                        f, site.continue_target,
                        site.entry, site.merge);
                auto opposing =
                    (true_kind ==
                         LoopBoundaryTargetKind::BREAK &&
                     false_kind ==
                         LoopBoundaryTargetKind::CONTINUE) ||
                    (true_kind ==
                         LoopBoundaryTargetKind::CONTINUE &&
                     false_kind ==
                         LoopBoundaryTargetKind::BREAK);
                auto one_sided_boundary =
                    singular_boundary(true_kind) !=
                    singular_boundary(false_kind);
                auto generated_boundary_guard =
                    generated_exit_dispatch_headers.contains(
                        branch_block) &&
                    one_sided_boundary;
                if (generated_boundary_guard || opposing ||
                    (allow_one_sided_boundary &&
                     one_sided_boundary)) {
                    candidates.emplace_back(Candidate{
                        branch_block, t, f, site.entry,
                        site.continue_target, site.merge,
                        site.selection_merge,
                        cbr->condition(), site.depth});
                }
            };
        // A canonical conditional prepare is already the native loop guard.
        // A non-canonical one (for example, an exit through a state-writing
        // proxy) is instead a loop-boundary selection. Normalize it before
        // separating prepare from body so natural-loop recovery cannot
        // rediscover the same cycle as a nested loop.
        if (site.entry->is_terminated() &&
            site.entry->terminator()
                ->isa<ConditionalBranchInst>()) {
            auto *prepare_branch =
                static_cast<ConditionalBranchInst *>(
                    site.entry->terminator());
            if (prepare_branch->true_block() != site.body ||
                prepare_branch->false_block() != site.merge) {
                append_candidate(site.entry, true);
            }
        }
        luisa::unordered_set<BasicBlock *> visited;
        luisa::vector<BasicBlock *> work;
        auto enqueue = [&](BasicBlock *block) noexcept {
            if (block == nullptr || block == site.entry ||
                block == site.merge) {
                return;
            }
            if (visited.emplace(block).second) {
                work.emplace_back(block);
            }
        };
        enqueue(site.body);
        while (!work.empty()) {
            auto *branch_block = work.back();
            work.pop_back();
            if (!branch_block->is_terminated()) { continue; }
            auto *term = branch_block->terminator();
            append_candidate(branch_block, false);
            if (term->isa<LoopInst>() ||
                term->isa<SimpleLoopInst>()) {
                if (auto *control_flow_merge =
                        term->control_flow_merge();
                    control_flow_merge != nullptr) {
                    enqueue(control_flow_merge->merge_block());
                }
                continue;
            }
            traverse_structured_successors(
                branch_block,
                [&](BasicBlock *successor) noexcept {
                    enqueue(successor);
                });
        }
    }
    if (candidates.empty()) {
        // A generated dispatch is a loop-boundary guard only if at least one
        // structurally containing loop proves that relation. This point is
        // reached only after the exhaustive region scan above.
        for (auto *header : exit_dispatch_headers) {
            exit_dispatch_headers.erase(header);
            return true;
        }
        return false;
    }

    luisa::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate &lhs,
           const Candidate &rhs) noexcept {
            return lhs.loop_depth > rhs.loop_depth;
        });
    auto cand = candidates.front();
    if (cand.branch_block == nullptr || !cand.branch_block->is_terminated()) { return false; }
    auto *old_term = cand.branch_block->terminator();
    if (!old_term->isa<ConditionalBranchInst>()) { return false; }
    auto true_kind = classify_loop_boundary_path(cand.true_target, cand.continue_target, cand.loop_entry, cand.merge);
    auto false_kind = classify_loop_boundary_path(cand.false_target, cand.continue_target, cand.loop_entry, cand.merge);

    auto opposing =
        (true_kind == LoopBoundaryTargetKind::BREAK &&
         false_kind == LoopBoundaryTargetKind::CONTINUE) ||
        (true_kind == LoopBoundaryTargetKind::CONTINUE &&
         false_kind == LoopBoundaryTargetKind::BREAK);
    auto one_sided_boundary =
        singular_boundary(true_kind) !=
        singular_boundary(false_kind);
    auto generated_boundary_guard =
        generated_exit_dispatch_headers.contains(
            cand.branch_block) &&
        (one_sided_boundary || opposing);
    old_term->remove_self();
    // Keep the exit-dispatch role when a raw dispatch becomes a loop-boundary
    // guard. The IfInst is the structured XIR spelling of a physical branch
    // that does not declare an OpSelectionMerge; treating it as an ordinary
    // selection would route its boundary arm through a fresh dispatch forever.
    if (!generated_boundary_guard) {
        exit_dispatch_headers.erase(
            cand.branch_block);
    }
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
    auto boundary_block = [&](BasicBlock *target, LoopBoundaryTargetKind kind) noexcept {
        if (kind == LoopBoundaryTargetKind::BREAK &&
            is_loop_break_target(target, cand.merge)) {
            return create_boundary_block(true);
        }
        if (kind == LoopBoundaryTargetKind::CONTINUE &&
            is_loop_continue_target(
                target, cand.continue_target,
                cand.loop_entry)) {
            return create_boundary_block(false);
        }
        return target;
    };
    auto *true_block = boundary_block(cand.true_target, true_kind);
    auto *false_block = boundary_block(cand.false_target, false_kind);
    auto *selection_merge = cand.selection_merge;
    if (one_sided_boundary) {
        selection_merge =
            singular_boundary(true_kind) &&
                    !singular_boundary(false_kind) ?
                false_block :
                true_block;
    } else if (opposing) {
        selection_merge = false_block;
    }
    if_inst->set_true_target(true_block);
    if_inst->set_false_target(false_block);
    if_inst->set_merge_block(selection_merge);
    return true;
}

[[nodiscard]] bool normalize_loop_boundary_conditional_branches(FunctionDefinition *def,
                                                                luisa::unordered_set<BasicBlock *> &
                                                                    exit_dispatch_headers,
                                                                const luisa::unordered_set<BasicBlock *> &
                                                                    generated_exit_dispatch_headers) noexcept {
    ScopedTimer _timer_normalize_loop_boundary_conditional_branches(
        "normalize_loop_boundary_conditional_branches");
    auto modified = false;
    // Each successful rewrite replaces one raw conditional branch with an IfInst,
    // so this phase has a finite, monotonic worklist and needs no site-count cap.
    while (normalize_one_loop_boundary_conditional_branch(
        def, exit_dispatch_headers,
        generated_exit_dispatch_headers)) {
        modified = true;
    }
    return modified;
}

void remove_dead_dispatch_expression(
    Value *root,
    luisa::unordered_set<AllocaInst *> &selector_allocas,
    luisa::vector<ManagedPtr<Instruction>> &removed) noexcept {
    luisa::vector<Value *> work{root};
    while (!work.empty()) {
        auto *value = work.back();
        work.pop_back();
        if (value == nullptr || !value->isa<Instruction>()) {
            continue;
        }
        auto *inst = static_cast<Instruction *>(value);
        if (inst->isa<AllocaInst>()) {
            selector_allocas.emplace(
                static_cast<AllocaInst *>(inst));
            continue;
        }
        if (inst->is_terminator() ||
            !inst->use_list().empty() ||
            !get_memory_info(inst)
                 .is_removable_if_unused()) {
            continue;
        }
        if (inst->isa<LoadInst>()) {
            auto *variable =
                static_cast<LoadInst *>(inst)->variable();
            if (variable != nullptr &&
                variable->isa<AllocaInst>()) {
                selector_allocas.emplace(
                    static_cast<AllocaInst *>(
                        variable));
            }
        }
        luisa::vector<Value *> operands;
        for (auto *operand_use : inst->operand_uses()) {
            operands.emplace_back(operand_use->value());
        }
        removed.emplace_back(inst->remove_self());
        for (auto *operand : operands) {
            work.emplace_back(operand);
        }
    }
}

void remove_write_only_dispatch_selectors(
    luisa::unordered_set<AllocaInst *> &selector_allocas,
    luisa::vector<ManagedPtr<Instruction>> &removed) noexcept {
    luisa::vector<AllocaInst *> work;
    work.reserve(selector_allocas.size());
    for (auto *alloca : selector_allocas) {
        work.emplace_back(alloca);
    }
    for (auto cursor = size_t{0u};
         cursor < work.size(); ++cursor) {
        auto *alloca = work[cursor];
        if (alloca == nullptr || !alloca->is_local()) {
            continue;
        }
        luisa::vector<StoreInst *> stores;
        auto write_only = true;
        for (auto &&use : alloca->use_list()) {
            auto *user = use->user();
            if (user == nullptr ||
                !user->isa<StoreInst>()) {
                write_only = false;
                break;
            }
            auto *store = static_cast<StoreInst *>(user);
            if (store->variable() != alloca) {
                write_only = false;
                break;
            }
            stores.emplace_back(store);
        }
        if (!write_only) { continue; }
        luisa::vector<Value *> stored_values;
        stored_values.reserve(stores.size());
        for (auto *store : stores) {
            stored_values.emplace_back(store->value());
            removed.emplace_back(store->remove_self());
        }
        if (alloca->use_list().empty()) {
            removed.emplace_back(alloca->remove_self());
        }
        luisa::unordered_set<AllocaInst *>
            discovered_allocas;
        for (auto *value : stored_values) {
            remove_dead_dispatch_expression(
                value, discovered_allocas, removed);
        }
        for (auto *discovered : discovered_allocas) {
            if (selector_allocas.emplace(
                                    discovered)
                    .second) {
                work.emplace_back(discovered);
            }
        }
    }
}

[[nodiscard]] bool bypass_trivial_loop_prepare_exit_proxies(
    FunctionDefinition *def) noexcept {
    auto modified = false;
    def->traverse_basic_blocks([&](BasicBlock *header) noexcept {
        if (header == nullptr || !header->is_terminated() ||
            !header->terminator()->isa<LoopInst>()) {
            return;
        }
        auto *loop =
            static_cast<LoopInst *>(header->terminator());
        auto *prepare = loop->prepare_block();
        auto *body = loop->body_block();
        auto *merge = loop->merge_block();
        if (prepare == nullptr || body == nullptr ||
            merge == nullptr || !prepare->is_terminated() ||
            !prepare->terminator()
                 ->isa<ConditionalBranchInst>()) {
            return;
        }
        auto *branch = static_cast<ConditionalBranchInst *>(
            prepare->terminator());
        if (branch->true_block() != body ||
            branch->false_block() == merge ||
            !trivial_branch_chain_reaches(
                branch->false_block(), merge)) {
            return;
        }
        branch->set_false_target(merge);
        modified = true;
    });
    return modified;
}

// LLVM SPIRVStructurizer::removeUselessBlocks removes the forwarding chains
// left behind after inner-to-outer exit-state propagation. Do the equivalent
// before mem2reg: if both arms of a generated dispatch consist only of
// unconditional forwarding blocks and end at the same target, the dispatch
// has no control-flow meaning. The selected state already lives in the
// alloca/store protocol, so replacing the conditional with one branch is
// semantics-preserving and prevents a spurious SPIR-V selection.
[[nodiscard]] bool collapse_redundant_exit_dispatches(
    FunctionDefinition *def,
    const luisa::unordered_set<BasicBlock *> &
        generated_exit_dispatch_headers) noexcept {
    ScopedTimer _timer_collapse_redundant_exit_dispatches(
        "collapse_redundant_exit_dispatches");
    auto modified = false;
    luisa::vector<Value *> dead_roots;
    luisa::vector<ManagedPtr<Instruction>> removed;
    luisa::unordered_set<BasicBlock *> live_blocks;
    def->traverse_basic_blocks(
        [&](BasicBlock *block) noexcept {
            live_blocks.emplace(block);
        });
    for (auto *header : generated_exit_dispatch_headers) {
        if (!live_blocks.contains(header) ||
            !header->is_terminated()) {
            continue;
        }
        auto *term = header->terminator();
        if (!term->isa<ConditionalBranchInst>() &&
            !term->isa<IfInst>()) {
            continue;
        }
        auto *branch = static_cast<
            ConditionalBranchTerminatorInstruction *>(term);
        auto *true_target =
            trivial_branch_chain_target(
                branch->true_block());
        auto *false_target =
            trivial_branch_chain_target(
                branch->false_block());
        if (true_target == nullptr ||
            true_target != false_target) {
            continue;
        }
        auto *condition = branch->condition();
        auto old_term = term->remove_self();
        XIRBuilder builder;
        builder.set_insertion_point(header);
        auto *replacement = builder.br(true_target);
        for (auto *metadata : old_term->metadata_list()) {
            replacement->metadata_list().push_front(
                metadata->clone());
        }
        dead_roots.emplace_back(condition);
        modified = true;
    }
    luisa::unordered_set<AllocaInst *>
        selector_allocas;
    for (auto *root : dead_roots) {
        remove_dead_dispatch_expression(
            root, selector_allocas, removed);
    }
    remove_write_only_dispatch_selectors(
        selector_allocas, removed);
    modified |= bypass_trivial_loop_prepare_exit_proxies(
        def);
    return modified;
}

[[nodiscard]] bool normalize_structured_loop_continues(FunctionDefinition *def) noexcept {
    ScopedTimer _timer_normalize_structured_loop_continues(
        "normalize_structured_loop_continues");
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

// A SimpleLoop uses a synthetic SPIR-V continue target because its XIR body
// block is also the logical loop header. A shared forwarding block that ends
// in ContinueInst cannot safely collect edges from sibling selections: any
// Phi recovered in that block would be placed inside one selection and then
// entered from another. Give every incoming edge its own continue block so
// SSA recovery places loop-carried Phis at the loop header instead.
[[nodiscard]] bool split_shared_simple_loop_continues(
    FunctionDefinition *def) noexcept {
    luisa::unordered_set<BasicBlock *> simple_loop_bodies;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb->is_terminated() &&
            bb->terminator()->isa<SimpleLoopInst>()) {
            auto *body = static_cast<SimpleLoopInst *>(
                             bb->terminator())
                             ->body_block();
            if (body != nullptr) {
                simple_loop_bodies.emplace(body);
            }
        }
    });
    if (simple_loop_bodies.empty()) { return false; }

    struct Candidate {
        BasicBlock *block;
        BasicBlock *target;
        luisa::vector<BasicBlock *> predecessors;
    };
    luisa::vector<Candidate> candidates;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!has_only_terminator(bb) ||
            !bb->terminator()->isa<ContinueInst>()) {
            return;
        }
        auto *target =
            static_cast<ContinueInst *>(bb->terminator())
                ->target_block();
        if (!simple_loop_bodies.contains(target)) { return; }
        luisa::vector<BasicBlock *> predecessors;
        bb->traverse_predecessors(
            false, [&](BasicBlock *predecessor) noexcept {
                if (predecessor != nullptr) {
                    predecessors.emplace_back(predecessor);
                }
            });
        if (predecessors.size() > 1u) {
            candidates.emplace_back(Candidate{
                bb, target, std::move(predecessors)});
        }
    });

    auto changed = false;
    XIRBuilder b;
    for (auto &&candidate : candidates) {
        for (auto *predecessor : candidate.predecessors) {
            if (!predecessor->is_terminated() ||
                !terminator_targets(predecessor->terminator(),
                                    candidate.block)) {
                continue;
            }
            auto *proxy = def->create_basic_block();
            b.set_insertion_point(proxy);
            b.continue_(candidate.target);
            if (retarget_terminator(
                    predecessor->terminator(),
                    candidate.block, proxy)) {
                changed = true;
            } else {
                proxy->remove_self();
            }
        }
    }
    return changed;
}

[[nodiscard]] bool is_loop_boundary_if(IfInst *if_inst,
                                       BasicBlock *continue_target,
                                       BasicBlock *loop_entry,
                                       BasicBlock *merge) noexcept {
    if (if_inst == nullptr) { return false; }
    if (continue_target == nullptr || loop_entry == nullptr || merge == nullptr) { return false; }
    auto true_kind = classify_loop_boundary_path(
        if_inst->true_block(), continue_target, loop_entry, merge);
    auto false_kind = classify_loop_boundary_path(
        if_inst->false_block(), continue_target, loop_entry, merge);
    auto singular_boundary = [](auto kind) noexcept {
        return kind == LoopBoundaryTargetKind::BREAK ||
               kind == LoopBoundaryTargetKind::CONTINUE;
    };
    auto *selection_merge = if_inst->merge_block();
    return (true_kind == LoopBoundaryTargetKind::CONTINUE &&
            false_kind == LoopBoundaryTargetKind::BREAK) ||
           (true_kind == LoopBoundaryTargetKind::BREAK &&
            false_kind == LoopBoundaryTargetKind::CONTINUE) ||
           (if_inst->true_block() == selection_merge &&
            singular_boundary(false_kind)) ||
           (if_inst->false_block() == selection_merge &&
            singular_boundary(true_kind));
}

template<typename F>
[[nodiscard]] bool visit_loop_region_blocks(
    FunctionDefinition *def, F &&visitor) noexcept {
    if (def == nullptr) { return false; }
    auto stopped = false;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (stopped || !bb->is_terminated()) { return; }
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
        while (!work.empty() && !stopped) {
            auto *cur = work.back();
            work.pop_back();
            if (visitor(
                    cur, continue_target,
                    loop_entry, merge)) {
                stopped = true;
                return;
            }
            traverse_structured_successors(
                cur, [&](BasicBlock *succ) noexcept {
                    if (succ == loop_entry ||
                        succ == merge) {
                        return;
                    }
                    enqueue(succ);
                });
        }
    });
    return stopped;
}

[[nodiscard]] bool is_loop_boundary_selection_entry(
    BasicBlock *entry,
    FunctionDefinition *def) noexcept {
    if (entry == nullptr || !entry->is_terminated() ||
        !entry->terminator()->isa<IfInst>()) {
        return false;
    }
    auto *if_inst =
        static_cast<IfInst *>(entry->terminator());
    return visit_loop_region_blocks(
        def, [&](BasicBlock *block,
                 BasicBlock *continue_target,
                 BasicBlock *loop_entry,
                 BasicBlock *merge) noexcept {
            return block == entry &&
                   is_loop_boundary_if(
                       if_inst, continue_target,
                       loop_entry, merge);
        });
}

// Invert is_loop_boundary_selection_entry's repeated membership query. For
// one immutable CFG, the predicate is the exact existential relation
//
//   boundary(entry) iff
//       exists loop: entry is structurally reachable inside loop and
//                    entry's IfInst branches only across that loop boundary.
//
// Walking every loop once materializes the same relation for all entries.
[[nodiscard]] luisa::unordered_set<BasicBlock *>
collect_loop_boundary_selection_entries(
    FunctionDefinition *def) noexcept {
    luisa::unordered_set<BasicBlock *> entries;
    static_cast<void>(visit_loop_region_blocks(
        def, [&](BasicBlock *block,
                 BasicBlock *continue_target,
                 BasicBlock *loop_entry,
                 BasicBlock *merge) noexcept {
            if (block->is_terminated() &&
                block->terminator()->isa<IfInst>() &&
                is_loop_boundary_if(
                    static_cast<IfInst *>(
                        block->terminator()),
                    continue_target, loop_entry,
                    merge)) {
                entries.emplace(block);
            }
            return false;
        }));
    return entries;
}

[[nodiscard]] bool canonicalize_loop_boundary_selection_merges(FunctionDefinition *def) noexcept {
    ScopedTimer _timer_canonicalize_loop_boundary_selection_merges(
        "canonicalize_loop_boundary_selection_merges");
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
                auto true_kind = classify_loop_boundary_path(
                    if_inst->true_block(), continue_target,
                    loop_entry, merge);
                auto false_kind = classify_loop_boundary_path(
                    if_inst->false_block(), continue_target,
                    loop_entry, merge);
                auto canonicalize_boundary_arm =
                    [&](BasicBlock *target,
                        LoopBoundaryTargetKind kind) noexcept {
                        auto break_arm =
                            kind == LoopBoundaryTargetKind::BREAK &&
                            is_loop_break_target(target, merge);
                        if (!break_arm) { return target; }
                        // A previously normalized boundary proxy may be
                        // lowered from Break(M) to a pure branch chain ending
                        // at M by an adjacent canonicalization phase. It
                        // already preserves the declared single-exit
                        // boundary. Only a chain that reaches the forwarding
                        // destination *after* M while bypassing M needs a new
                        // proxy.
                        if (is_canonical_loop_break_path(
                                target, merge)) {
                            return target;
                        }
                        auto *proxy = def->create_basic_block();
                        XIRBuilder b;
                        b.set_insertion_point(proxy);
                        b.break_(merge);
                        changed = true;
                        return proxy;
                    };
                auto *old_true = if_inst->true_block();
                auto *old_false = if_inst->false_block();
                auto *new_true = canonicalize_boundary_arm(
                    old_true, true_kind);
                auto *new_false = canonicalize_boundary_arm(
                    old_false, false_kind);
                if (new_true != old_true) {
                    if_inst->set_true_target(new_true);
                }
                if (new_false != old_false) {
                    if_inst->set_false_target(new_false);
                }
                if (is_loop_boundary_if(
                        if_inst, continue_target,
                        loop_entry, merge) &&
                    ((if_inst->merge_block() !=
                          if_inst->true_block() &&
                      if_inst->merge_block() !=
                          if_inst->false_block()) ||
                     if_inst->merge_block() == merge)) {
                    old_false = if_inst->false_block();
                    auto *new_merge = def->create_basic_block();
                    XIRBuilder b;
                    b.set_insertion_point(new_merge);
                    b.br(old_false);
                    if_inst->set_false_target(new_merge);
                    if_inst->set_merge_block(new_merge);
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
    ScopedTimer _timer_canonicalize_loop_update_blocks(
        "canonicalize_loop_update_blocks");
    struct LoopSite {
        LoopInst *loop{nullptr};
        BasicBlock *old_update{nullptr};
        BasicBlock *prepare{nullptr};
        BasicBlock *merge{nullptr};
    };
    luisa::vector<LoopSite> loops;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<LoopInst>()) { return; }
        auto *loop = static_cast<LoopInst *>(term);
        auto *prepare = loop->prepare_block();
        auto *update = loop->update_block();
        auto *merge = loop->merge_block();
        if (prepare == nullptr || update == nullptr ||
            merge == nullptr) {
            return;
        }
        auto canonical = update->is_terminated() && update->terminator()->isa<BranchInst>() &&
                         static_cast<BranchInst *>(update->terminator())->target_block() == prepare;
        if (!canonical) {
            loops.emplace_back(
                LoopSite{loop, update, prepare, merge});
        }
    });
    if (loops.empty()) { return false; }
    for (auto site : loops) {
        // A non-trivial update is an executable region, not merely the
        // structural continue label. Let R be the blocks reachable from the
        // old update U without crossing the next-iteration prepare P or the
        // loop merge M. Continues from outside R enter U and must still execute
        // that region; continues in R complete the update and advance to P.
        //
        // Splitting U into an executable region plus a canonical trampoline U'
        // therefore preserves edges as follows:
        //
        //   Continue(outside R -> U)  => Branch(outside R -> U)
        //   Continue(inside  R -> U)  => Continue(inside R -> U')
        //   U'                        => Branch(U' -> P)
        //
        // Retargeting both classes directly to U' would bypass all state
        // updates and Break paths in R, and can turn a finite loop into an
        // unconditional one.
        luisa::unordered_set<BasicBlock *> update_region;
        luisa::vector<BasicBlock *> work{site.old_update};
        while (!work.empty()) {
            auto *block = work.back();
            work.pop_back();
            if (block == nullptr || block == site.prepare ||
                block == site.merge ||
                !update_region.emplace(block).second ||
                !block->is_terminated()) {
                continue;
            }
            traverse_structured_successors(
                block, [&](BasicBlock *successor) noexcept {
                    if (successor != site.prepare &&
                        successor != site.merge) {
                        work.emplace_back(successor);
                    }
                });
        }

        luisa::vector<ContinueInst *> entering_update;
        luisa::vector<ContinueInst *> completing_update;
        def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
            if (!bb->is_terminated() ||
                !bb->terminator()->isa<ContinueInst>()) {
                return;
            }
            auto *cont = static_cast<ContinueInst *>(
                bb->terminator());
            if (cont->target_block() != site.old_update) {
                return;
            }
            (update_region.contains(bb) ?
                 completing_update :
                 entering_update)
                .emplace_back(cont);
        });

        auto *new_update = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(new_update);
        b.br(site.prepare);
        site.loop->set_update_block(new_update);
        for (auto *cont : completing_update) {
            cont->set_target_block(new_update);
        }
        for (auto *cont : entering_update) {
            auto *parent = cont->parent_block();
            auto old_cont = cont->remove_self();
            b.set_insertion_point(parent);
            auto *branch = b.br(site.old_update);
            for (auto *metadata : old_cont->metadata_list()) {
                branch->metadata_list().push_front(
                    metadata->clone());
            }
        }
    }
    return true;
}

[[nodiscard]] bool has_executable_edge(
    BasicBlock *from, BasicBlock *to) noexcept;
[[nodiscard]] bool retarget_executable_edge(
    Instruction *terminator, BasicBlock *from,
    BasicBlock *to) noexcept;

// Separate the Loop.prepare boundary role from an already-structured body
// header. Native SPIR-V requires prepare to be either Branch(body) or
// ConditionalBranch(condition, body, merge), but generic XIR permits the
// prepare block itself to end in If/Switch/Loop.
//
// For every non-canonical prepare P, insert an empty P' with P' -> P, redirect
// all old executable entries of P through P', and make P the loop body. The
// executable graph is changed only by subdividing incoming edges with P', so
// instruction order, branch conditions, exits, and loop-carried state are
// preserved. The restructuring preflight rejects Phi input, hence moving the
// structural boundary cannot invalidate Phi predecessor labels.
[[nodiscard]] bool canonicalize_loop_prepare_blocks(
    FunctionDefinition *def) noexcept {
    ScopedTimer _timer_canonicalize_loop_prepare_blocks(
        "canonicalize_loop_prepare_blocks");
    luisa::vector<LoopInst *> loops;
    def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        if (block != nullptr && block->is_terminated() &&
            block->terminator()->isa<LoopInst>()) {
            loops.emplace_back(
                static_cast<LoopInst *>(
                    block->terminator()));
        }
    });

    auto changed = false;
    for (auto *loop : loops) {
        auto *owner = loop->parent_block();
        auto *prepare = loop->prepare_block();
        auto *body = loop->body_block();
        auto *merge = loop->merge_block();
        if (owner == nullptr || prepare == nullptr ||
            body == nullptr || merge == nullptr ||
            !prepare->is_terminated()) {
            continue;
        }
        auto *terminator = prepare->terminator();
        auto canonical =
            terminator->isa<BranchInst>() &&
            static_cast<BranchInst *>(terminator)
                    ->target_block() == body;
        if (terminator->isa<ConditionalBranchInst>()) {
            auto *branch =
                static_cast<ConditionalBranchInst *>(
                    terminator);
            canonical =
                branch->condition() != nullptr &&
                branch->condition()->type() ==
                    Type::of<bool>() &&
                branch->true_block() == body &&
                branch->false_block() == merge;
        }
        if (canonical) { continue; }

        luisa::vector<BasicBlock *> old_predecessors;
        def->traverse_basic_blocks(
            [&](BasicBlock *predecessor) noexcept {
                if (predecessor == nullptr ||
                    predecessor == owner ||
                    !predecessor->is_terminated()) {
                    return;
                }
                if (has_executable_edge(
                        predecessor, prepare)) {
                    old_predecessors.emplace_back(
                        predecessor);
                }
            });

        auto *new_prepare = def->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(new_prepare);
        builder.br(prepare);
        for (auto *predecessor : old_predecessors) {
            LUISA_ASSERT(
                predecessor->is_terminated() &&
                    retarget_executable_edge(
                        predecessor->terminator(),
                        prepare, new_prepare),
                "Failed to subdivide an executable "
                "Loop.prepare entry edge.");
            fix_degenerate_terminator(predecessor);
        }
        loop->set_prepare_block(new_prepare);
        loop->set_body_block(prepare);
        changed = true;
    }
    return changed;
}

[[nodiscard]] bool proxy_switch_targets_to_structural_boundaries(FunctionDefinition *def) noexcept {
    ScopedTimer _timer_proxy_switch_targets(
        "proxy_switch_targets_to_structural_boundaries");
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

// Visit executable CFG successors only. Keep this spelling explicit instead
// of deriving edges from arbitrary block-valued fields: structured merge,
// loop-body, update, and continue roles are declarations, not executable
// successors of the construct header.
template<typename Visitor>
void traverse_executable_successors(BasicBlock *block,
                                    Visitor &&visit) noexcept {
    if (block == nullptr || !block->is_terminated()) { return; }
    luisa::vector<BasicBlock *> visited;
    auto visit_once = [&](BasicBlock *successor) noexcept {
        if (successor == nullptr ||
            std::find(visited.begin(), visited.end(), successor) !=
                visited.end()) {
            return;
        }
        visited.emplace_back(successor);
        visit(successor);
    };
    auto *term = block->terminator();
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::IF:
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto *branch = static_cast<
                ConditionalBranchTerminatorInstruction *>(term);
            visit_once(branch->true_block());
            visit_once(branch->false_block());
            break;
        }
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto *branch = static_cast<
                IndexedBranchTerminatorInstruction *>(term);
            visit_once(branch->default_block());
            for (auto i = size_t{0u}; i < branch->case_count(); ++i) {
                visit_once(branch->case_block(i));
            }
            break;
        }
        case DerivedInstructionTag::LOOP:
            visit_once(static_cast<LoopInst *>(term)->prepare_block());
            break;
        case DerivedInstructionTag::SIMPLE_LOOP:
            visit_once(static_cast<SimpleLoopInst *>(term)->body_block());
            break;
        case DerivedInstructionTag::BRANCH:
        case DerivedInstructionTag::BREAK:
        case DerivedInstructionTag::CONTINUE:
        case DerivedInstructionTag::OUTLINE:
            visit_once(static_cast<BranchTerminatorInstruction *>(term)
                           ->target_block());
            break;
        default: {
            auto *declared_merge =
                structured_statement_merge(term);
            for (auto *operand_use : term->operand_uses()) {
                auto *value = operand_use->value();
                if (value == nullptr ||
                    !value->isa<BasicBlock>() ||
                    value == declared_merge) {
                    continue;
                }
                visit_once(static_cast<BasicBlock *>(value));
            }
            break;
        }
    }
}

[[nodiscard]] bool has_executable_edge(
    BasicBlock *from, BasicBlock *to) noexcept {
    auto found = false;
    traverse_executable_successors(
        from, [&](BasicBlock *successor) noexcept {
            found |= successor == to;
        });
    return found;
}

[[nodiscard]] bool retarget_executable_edge(
    Instruction *terminator, BasicBlock *from,
    BasicBlock *to) noexcept {
    if (terminator == nullptr || from == nullptr ||
        to == nullptr) {
        return false;
    }
    auto changed = false;
    switch (terminator->derived_instruction_tag()) {
        case DerivedInstructionTag::IF:
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto *branch = static_cast<
                ConditionalBranchTerminatorInstruction *>(
                terminator);
            if (branch->true_block() == from) {
                branch->set_true_target(to);
                changed = true;
            }
            if (branch->false_block() == from) {
                branch->set_false_target(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto *branch = static_cast<
                IndexedBranchTerminatorInstruction *>(
                terminator);
            if (branch->default_block() == from) {
                branch->set_default_block(to);
                changed = true;
            }
            for (auto i = size_t{0u};
                 i < branch->case_count(); ++i) {
                if (branch->case_block(i) == from) {
                    branch->set_case_block(i, to);
                    changed = true;
                }
            }
            break;
        }
        case DerivedInstructionTag::LOOP: {
            auto *loop =
                static_cast<LoopInst *>(terminator);
            if (loop->prepare_block() == from) {
                loop->set_prepare_block(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::SIMPLE_LOOP: {
            auto *loop =
                static_cast<SimpleLoopInst *>(terminator);
            if (loop->body_block() == from) {
                loop->set_body_block(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::BRANCH:
        case DerivedInstructionTag::BREAK:
        case DerivedInstructionTag::CONTINUE:
        case DerivedInstructionTag::OUTLINE: {
            auto *branch = static_cast<
                BranchTerminatorInstruction *>(
                terminator);
            if (branch->target_block() == from) {
                branch->set_target_block(to);
                changed = true;
            }
            break;
        }
        default: break;
    }
    return changed;
}

struct StructuredLoopExitInfo {
    BasicBlock *header{nullptr};
    luisa::vector<BasicBlock *> exits;
};

[[nodiscard]] luisa::vector<StructuredLoopExitInfo>
collect_structured_loop_exit_info(
    FunctionDefinition *def) noexcept {
    luisa::vector<StructuredLoopExitInfo> loops;
    if (def == nullptr) { return loops; }
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        StructuredLoopExitInfo info{.header = bb};
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            if (auto *prepare = loop->prepare_block();
                prepare != nullptr) {
                info.exits.emplace_back(prepare);
            }
            if (auto *update = loop->update_block();
                update != nullptr) {
                info.exits.emplace_back(update);
            }
            if (auto *merge = loop->merge_block();
                merge != nullptr) {
                info.exits.emplace_back(merge);
            }
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop =
                static_cast<SimpleLoopInst *>(term);
            if (auto *body = loop->body_block();
                body != nullptr) {
                info.exits.emplace_back(body);
            }
            if (auto *merge = loop->merge_block();
                merge != nullptr) {
                info.exits.emplace_back(merge);
            }
        } else {
            return;
        }
        loops.emplace_back(std::move(info));
    });
    return loops;
}

[[nodiscard]] luisa::unordered_set<BasicBlock *>
collect_enclosing_loop_exits(
    FunctionDefinition *def,
    BasicBlock *header,
    const DomTree &dom,
    const luisa::vector<StructuredLoopExitInfo> *precomputed_loops) noexcept {
    luisa::unordered_set<BasicBlock *> exits;
    if (!dom.contains(header)) { return exits; }
    // Re-scanning the whole CFG for structured loops per call makes
    // selection-merge inference quadratic in the block count (every candidate
    // re-walks every loop). Callers that infer merges for many candidates
    // precompute the loop set once and pass it in.
    const auto &loops = precomputed_loops != nullptr
                            ? *precomputed_loops
                            : collect_structured_loop_exit_info(def);
    for (auto &&loop : loops) {
        if (!dom.contains(loop.header) ||
            !dom.dominates(loop.header, header)) {
            continue;
        }
        for (auto *exit : loop.exits) {
            exits.emplace(exit);
        }
    }
    return exits;
}

struct SelectionExitCFGRelations {
    luisa::unordered_set<BasicBlock *>
        loop_boundary_selection_entries;
    luisa::unordered_map<
        BasicBlock *,
        luisa::unordered_set<BasicBlock *>>
        enclosing_loop_exits;
};

// Materialize the exact CFG relations consumed by one selection-exit scan.
// No rewrite occurs while a scan evaluates its sites. A successful rewrite
// returns to the caller, which rebuilds dominance and rematerializes these
// relations before observing the new CFG.
[[nodiscard]] SelectionExitCFGRelations
build_selection_exit_cfg_relations(
    FunctionDefinition *def,
    const DomTree &dom) noexcept {
    SelectionExitCFGRelations relations;
    relations.loop_boundary_selection_entries =
        collect_loop_boundary_selection_entries(def);
    auto loops = collect_structured_loop_exit_info(def);
    def->traverse_basic_blocks(
        [&](BasicBlock *block) noexcept {
            if (!dom.contains(block)) { return; }
            for (auto &&loop : loops) {
                if (!dom.contains(loop.header) ||
                    !dom.dominates(
                        loop.header, block)) {
                    continue;
                }
                auto &exits =
                    relations.enclosing_loop_exits[
                        block];
                for (auto *exit : loop.exits) {
                    exits.emplace(exit);
                }
            }
        });
    return relations;
}

struct SelectionExitEdge {
    BasicBlock *src;
    BasicBlock *dst;
};

enum class SelectionExitRewriteStatus : uint8_t {
    UNCHANGED,
    MODIFIED,
    REPEATED_SITE,
};

struct SelectionExitRewriteResult {
    SelectionExitRewriteStatus status{SelectionExitRewriteStatus::UNCHANGED};
    Instruction *site{nullptr};
};

void append_unique_exit_edge(luisa::vector<SelectionExitEdge> &edges,
                             BasicBlock *src,
                             BasicBlock *dst) noexcept {
    for (auto edge : edges) {
        if (edge.src == src && edge.dst == dst) { return; }
    }
    edges.emplace_back(SelectionExitEdge{src, dst});
}

// A target-state dispatch deliberately represents correlated control flow as a
// conservative CFG: every incoming exit can syntactically reach every target,
// while the stored selector guarantees that only its original target is chosen
// at runtime. This edge expansion can invalidate SSA dominance for a value
// defined on one original exit path and used in its target.
//
// Transport such values through typed local state at the transformation
// boundary. On every dynamically feasible path to a repaired use, the original
// definition executed before its selector store, so the inserted load observes
// exactly the original SSA value. The SPIR-V post-restructure boundary promotes
// these marked slots back to SSA and audits that none remain.
void repair_target_state_dispatch_ssa(
    FunctionDefinition *def) noexcept {
    static_cast<void>(
        reg2mem_pass_repair_cross_block_rvalue_uses_on_function(
            static_cast<Function *>(def)));
}

[[nodiscard]] SelectionExitRewriteResult canonicalize_selection_exits(
    FunctionDefinition *def,
    BasicBlock *header,
    Instruction *term,
    BasicBlock *merge,
    const DomTree &dom,
    const SelectionExitCFGRelations &cfg_relations,
    RestructureCFGInfo &info,
    luisa::unordered_set<Instruction *> &rewritten_sites,
    luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    if (header == nullptr || term == nullptr || merge == nullptr) { return {}; }
    // Exit-state dispatches are the continuation *after* a construct's single
    // exit, matching LLVM SPIRVStructurizer::createSingleExitNode. They must
    // remain transparent while an enclosing construct walks through them, but
    // must not recursively structurize their own routing edges: doing so merely
    // recreates an equivalent dispatch behind a fresh merge forever.
    if (exit_dispatch_headers.contains(header)) {
        return {};
    }
    // A conditional that only chooses between the enclosing loop's break and
    // continue boundaries is not a SPIR-V selection at all. It is one of the
    // explicitly permitted loop-exit branch forms and must not be wrapped in
    // another state dispatch.
    if (cfg_relations.loop_boundary_selection_entries
            .contains(header)) {
        return {};
    }
    auto entries = selection_entries(term);
    if (entries.empty()) { return {}; }
    ++info.selection_exit_enclosing_loop_query_count;
    auto loop_exit_iter =
        cfg_relations.enclosing_loop_exits.find(header);
    auto is_enclosing_loop_exit =
        [&](BasicBlock *block) noexcept {
            return loop_exit_iter !=
                       cfg_relations
                           .enclosing_loop_exits.end() &&
                   loop_exit_iter->second.contains(block);
        };

    luisa::vector<SelectionExitEdge> invalid_exits;
    luisa::vector<SelectionExitEdge> merge_exits;
    luisa::unordered_set<BasicBlock *> region;
    auto entry_is_valid = [&](BasicBlock *entry) noexcept {
        return entry != nullptr && dom.contains(entry) && dom.dominates(header, entry);
    };
    for (auto *entry : entries) {
        if (entry == merge) {
            // An empty arm is a real normal-exit edge. If the selection gets a
            // new structural merge, this header edge must write the same exit
            // state as a non-empty arm; merely retargeting the operand would
            // leave the dispatch selector uninitialized.
            append_unique_exit_edge(merge_exits, header, merge);
            continue;
        }
        if (!entry_is_valid(entry)) {
            // The header itself can directly name an enclosing merge or loop
            // boundary. This is an executable non-local edge just like one
            // found inside an arm, and must be routed through the selection's
            // structural merge.
            if (entry != nullptr) {
                append_unique_exit_edge(
                    invalid_exits, header, entry);
            }
            continue;
        }
        luisa::vector<BasicBlock *> work{entry};
        while (!work.empty()) {
            auto *bb = work.back();
            work.pop_back();
            if (bb == nullptr || bb == merge) { continue; }
            if (!dom.contains(bb) || !dom.dominates(header, bb)) { continue; }
            if (!region.emplace(bb).second) { continue; }
            if (!bb->is_terminated()) { continue; }
            // Sites are processed from inner to outer. Once a nested construct
            // has a single exit, its arm-local state stores belong to that
            // child and must stay inside it; the parent continues from the
            // child's merge. Walking back into the child arms would mistake
            // those stores for independent parent exits and recreate an
            // equivalent state dispatch indefinitely.
            //
            // Loop-boundary guards are the deliberate exception: their IfInst
            // is XIR's structured spelling of a physical conditional without
            // OpSelectionMerge, so an enclosing construct must see both arms.
            auto *nested_merge =
                structured_statement_merge(bb->terminator());
            if (nested_merge != nullptr &&
                !exit_dispatch_headers.contains(bb) &&
                !cfg_relations
                     .loop_boundary_selection_entries
                     .contains(bb)) {
                if (nested_merge == merge) {
                    append_unique_exit_edge(
                        merge_exits, bb, nested_merge);
                } else {
                    work.emplace_back(nested_merge);
                }
                continue;
            }
            traverse_executable_successors(
                bb, [&](BasicBlock *succ) noexcept {
                    if (succ == nullptr) { return; }
                    if (succ == merge) {
                        append_unique_exit_edge(merge_exits, bb, succ);
                        return;
                    }
                    auto *canonical_successor =
                        canonical_exit_target(succ);
                    if (is_enclosing_loop_exit(succ) ||
                        is_enclosing_loop_exit(
                            canonical_successor)) {
                        // A break/continue edge is semantically valid XIR, but it
                        // cannot jump across a surrounding SPIR-V selection. Route
                        // it through this selection's merge; if the selection is
                        // nested, the same state is carried through each enclosing
                        // selection before it is dispatched at loop scope.
                        append_unique_exit_edge(invalid_exits, bb, succ);
                        return;
                    }
                    if (succ == header ||
                        canonical_successor == header) {
                        return;
                    }
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
    if (invalid_exits.empty()) { return {}; }

    // A declared arm may itself be the canonical target of an exit reached
    // from another arm. This is common after duplicate switch labels are
    // split through forwarding proxies: one label still names the shared
    // target directly, while the other labels reach it through their proxies.
    //
    // Once the shared target is moved behind a new single-exit merge, leaving
    // that direct header edge in place would make the target both an arm entry
    // before the merge and a continuation after it. In dominance terms this is
    // exactly a post-merge selection re-entry.
    //
    // The collected boundary edges form a cut between each arm and its exit.
    // Close that cut over canonical-target equivalence classes, but only for an
    // entry whose forwarding path is not already cut. This preserves distinct
    // switch proxies (and therefore distinct case entries) while adding the
    // missing zero-length header-to-exit path of a directly named sink.
    luisa::vector<SelectionExitEdge> reroute_edges;
    reroute_edges.reserve(invalid_exits.size() +
                          merge_exits.size() + entries.size());
    for (auto edge : invalid_exits) {
        reroute_edges.emplace_back(edge);
    }
    for (auto edge : merge_exits) {
        reroute_edges.emplace_back(edge);
    }
    luisa::unordered_set<BasicBlock *> reroute_sources;
    reroute_sources.reserve(reroute_edges.size());
    for (auto edge : reroute_edges) {
        reroute_sources.emplace(edge.src);
    }
    luisa::unordered_set<BasicBlock *> invalid_targets;
    invalid_targets.reserve(invalid_exits.size());
    for (auto edge : invalid_exits) {
        invalid_targets.emplace(
            canonical_exit_target(edge.dst));
    }
    auto forwarding_path_is_cut =
        [&](BasicBlock *entry) noexcept {
            luisa::unordered_set<BasicBlock *> visited;
            auto *block = entry;
            while (block != nullptr &&
                   visited.emplace(block).second) {
                if (reroute_sources.contains(block)) {
                    return true;
                }
                block = trivial_branch_target(block);
            }
            return false;
        };
    for (auto *entry : entries) {
        if (entry == nullptr || entry == merge) { continue; }
        if (invalid_targets.contains(
                canonical_exit_target(entry)) &&
            !forwarding_path_is_cut(entry)) {
            append_unique_exit_edge(
                reroute_edges, header, entry);
        }
    }

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
    // Canonicalize the state-dispatch ladder so loop boundaries are tested
    // before ordinary in-loop continuations. This gives each generated guard
    // exactly one break/continue arm and one fallthrough/merge arm, avoiding a
    // non-progressing "normal vs. (break-or-continue)" MIXED dispatch.
    std::stable_sort(
        targets.begin(), targets.end(),
        [&](BasicBlock *lhs, BasicBlock *rhs) noexcept {
            auto lhs_boundary =
                is_enclosing_loop_exit(lhs);
            auto rhs_boundary =
                is_enclosing_loop_exit(rhs);
            return lhs_boundary && !rhs_boundary;
        });
    target_ids.clear();
    for (auto i = size_t{0u}; i < targets.size(); ++i) {
        target_ids.emplace(
            targets[i],
            static_cast<uint32_t>(i));
    }
    for (auto edge : normalized_edges) {
        if (edge.src == nullptr || !edge.src->is_terminated() ||
            !terminator_targets(edge.src->terminator(), edge.dst)) {
            return {};
        }
    }
    if (!rewritten_sites.emplace(term).second) {
        return {SelectionExitRewriteStatus::REPEATED_SITE, term};
    }
    auto *new_merge = def->create_basic_block();
    XIRBuilder b;
    auto retargeted_any = false;
    if (targets.size() == 1u) {
        for (auto edge : normalized_edges) {
            retargeted_any |= retarget_structured_exit_to(
                edge.src->terminator(), edge.dst, new_merge);
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
            if (!retarget_structured_exit_to(
                    edge.src->terminator(), edge.dst, stub)) {
                stub->remove_self();
                continue;
            }
            retargeted_any = true;
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
        auto *dispatch_header = dispatch;
        for (size_t i = 0u; i + 1u < targets.size(); ++i) {
            exit_dispatch_headers.emplace(dispatch_header);
            auto id = target_ids.at(targets[i]);
            auto *id_const =
                mod->create_constant(Type::of<uint32_t>(), &id);
            auto *condition = b.call(
                Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                {loaded, id_const});
            auto *next = def->create_basic_block();
            b.cond_br(condition, targets[i], next);
            b.set_insertion_point(next);
            dispatch_header = next;
        }
        b.br(targets.back());
    }

    LUISA_ASSERT(retargeted_any,
                 "Selection-exit canonicalization planned a rewrite without a retargetable edge.");

    if (term->isa<IfInst>()) {
        auto *if_inst = static_cast<IfInst *>(term);
        if (if_inst->true_block() == merge) {
            if_inst->set_true_target(new_merge);
        }
        if (if_inst->false_block() == merge) {
            if_inst->set_false_target(new_merge);
        }
        if_inst->set_merge_block(new_merge);
    } else if (term->isa<SwitchInst>()) {
        auto *switch_inst = static_cast<SwitchInst *>(term);
        if (switch_inst->default_block() == merge) {
            switch_inst->set_default_block(new_merge);
        }
        for (auto i = 0u; i < switch_inst->case_count(); i++) {
            if (switch_inst->case_block(i) == merge) {
                switch_inst->set_case_block(i, new_merge);
            }
        }
        switch_inst->set_merge_block(new_merge);
    }
    if (targets.size() > 1u) {
        repair_target_state_dispatch_ssa(def);
    }
    return {SelectionExitRewriteStatus::MODIFIED, term};
}

[[nodiscard]] SelectionExitRewriteResult canonicalize_one_selection_exit(
    FunctionDefinition *def,
    const DomTree &dom,
    RestructureCFGInfo &info,
    luisa::unordered_set<Instruction *> &rewritten_sites,
    luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    auto cfg_relations =
        build_selection_exit_cfg_relations(def, dom);
    ++info.selection_exit_boundary_analysis_count;
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
        ++info.selection_exit_site_query_count;
        auto result = canonicalize_selection_exits(
            def, site.header, site.term, site.merge, dom,
            cfg_relations, info,
            rewritten_sites, exit_dispatch_headers);
        if (result.status != SelectionExitRewriteStatus::UNCHANGED) {
            return result;
        }
    }
    return {};
}

[[nodiscard]] bool drain_selection_exits(FunctionDefinition *def,
                                         DomTree &dom,
                                         PostDomInfo &pdom,
                                         RestructureCFGInfo &info,
                                         luisa::unordered_set<BasicBlock *> &
                                             exit_dispatch_headers) noexcept {
    ScopedTimer _timer_drain_selection_exits(
        "drain_selection_exits");
    luisa::unordered_set<Instruction *> rewritten_sites;
    auto modified = false;
    for (;;) {
        auto result = canonicalize_one_selection_exit(
            def, dom, info, rewritten_sites,
            exit_dispatch_headers);
        if (result.status == SelectionExitRewriteStatus::UNCHANGED) { break; }
        if (result.status == SelectionExitRewriteStatus::REPEATED_SITE) {
            ++info.iteration_limit_count;
            break;
        }
        modified = true;
        dom = compute_dom_tree(def, false);
        pdom = compute_post_dom(def);
    }
    return modified;
}

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
        // A bottom-checked (e.g. rotated) loop carries a conditional branch
        // in its latch and post-dominates through it; only then may the
        // forward collection sweep genuine exit blocks into the body.
        auto any_conditional_latch = false;
        for (auto *latch : valid_latches) {
            if (latch->is_terminated() &&
                latch->terminator()->isa<ConditionalBranchInst>()) {
                any_conditional_latch = true;
                break;
            }
        }
        collect_forward_loop_blocks();
        if (boundary_is_loop_internal && any_conditional_latch) {
            // Prune blocks that cannot reach the header or a latch; they are
            // outside the natural loop.
            luisa::unordered_set<BasicBlock *> reaching;
            luisa::vector<BasicBlock *> reach_work;
            reaching.emplace(header);
            for (auto *latch : valid_latches) {
                if (reaching.emplace(latch).second) {
                    reach_work.emplace_back(latch);
                }
            }
            while (!reach_work.empty()) {
                auto *cur = reach_work.back();
                reach_work.pop_back();
                cur->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                    if (pred == nullptr || !loop_blocks.contains(pred)) { return; }
                    if (reaching.emplace(pred).second) {
                        reach_work.emplace_back(pred);
                    }
                });
            }
            luisa::vector<BasicBlock *> pruned;
            for (auto *lb : loop_blocks) {
                if (!reaching.contains(lb)) { pruned.emplace_back(lb); }
            }
            for (auto *lb : pruned) { loop_blocks.erase(lb); }
        }
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

        // A bottom-checked (rotated) loop carries its only exit condition in
        // the latch. Preserve it as a conditional break/continue through a
        // proxy instead of dropping the condition with the forced back-edge.
        auto latch_keeps_conditional_exit = false;
        if (canonical_latch->is_terminated() &&
            canonical_latch->terminator()->isa<ConditionalBranchInst>()) {
            auto *cb = static_cast<ConditionalBranchInst *>(
                canonical_latch->terminator());
            auto *tb = cb->true_block();
            auto *fb = cb->false_block();
            auto *exit_arm = tb == header && fb == loop_merge ? fb :
                             fb == header && tb == loop_merge ? tb :
                                                                nullptr;
            if (exit_arm != nullptr) {
                auto *proxy = def->create_basic_block();
                {
                    XIRBuilder pb;
                    pb.set_insertion_point(proxy);
                    pb.br(loop_merge);
                }
                if (exit_arm == fb) {
                    cb->set_false_target(proxy);
                } else {
                    cb->set_true_target(proxy);
                }
                loop_blocks.emplace(proxy);
                latch_keeps_conditional_exit = true;
            }
        }
        if (!latch_keeps_conditional_exit) {
            if (canonical_latch->is_terminated()) {
                canonical_latch->terminator()->remove_self();
            }
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
                                            DomTree &dom,
                                            PostDomInfo &pdom,
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
        size_t depth;
    };
    // Merges of constructs that are already structured before this batch runs
    // (loops, previously restructured selections). Their interiors are
    // excluded from candidate scopes; the set is fixed for the whole batch
    // because restructuring retargets terminators but never creates
    // structured constructs. Precompute the interior as a plain set so the
    // per-block scope test stays O(1) during candidate processing.
    luisa::unordered_set<BasicBlock *> structured_merges;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (auto *m = structured_statement_merge(bb->terminator());
            m != nullptr) {
            structured_merges.emplace(m);
        }
    });
    luisa::unordered_set<BasicBlock *> structured_interior;
    for (auto *m : structured_merges) {
        auto *merge_node = dom.node_or_null(m);
        if (merge_node == nullptr) { continue; }
        luisa::vector<const DomTreeNode *> stack{merge_node};
        while (!stack.empty()) {
            auto *node = stack.back();
            stack.pop_back();
            auto *inner = node->block();
            // Merge blocks themselves remain part of the enclosing scope
            // (mirroring the historical worklist that pushed a nested
            // construct's merge without descending into its interior).
            if (inner != m && !structured_merges.contains(inner)) {
                structured_interior.emplace(inner);
            }
            for (auto *child : node->children()) {
                stack.emplace_back(child);
            }
        }
    }

    // Structured loops are immutable during this batch (restructuring only
    // retargets terminators), so collect them once and reuse for every merge
    // inference instead of re-scanning the CFG per candidate.
    auto structured_loops = collect_structured_loop_exit_info(def);

    // Blocks whose terminators may need retargeting (unstructured branches).
    // Retargeting only changes terminator targets and never turns a block into
    // or out of this class, and newly created structural merges are appended
    // below, so this snapshot stays complete for the whole batch. Walking the
    // dominator subtree per candidate instead would be quadratic in the block
    // count on chains of conditional branches.
    luisa::vector<BasicBlock *> unstructured_blocks;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (term->isa<ConditionalBranchInst>() ||
            term->isa<BranchInst>()) {
            unstructured_blocks.emplace_back(bb);
        }
    });

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

        auto entries = std::array{true_bb, false_bb};
        // Fast path: the immediate post-dominator is the merge for ordinary
        // single-exit selections, and is exactly what the historical fallback
        // accepted. Computing it is O(1), while the distance-scoring BFS below
        // re-walks the candidate's whole dominated tail, which is quadratic in
        // the block count for chains of conditional branches. Fall back to the
        // BFS only when no post-dominator exists.
        BasicBlock *merge = nullptr;
        {
            auto ipm_it = pdom.ipostdom.find(bb);
            if (ipm_it != pdom.ipostdom.end() &&
                ipm_it->second != nullptr &&
                ipm_it->second != pdom.virtual_exit &&
                ipm_it->second != bb) {
                merge = ipm_it->second;
            }
        }
        if (merge == nullptr) {
            merge = infer_selection_merge(
                def, bb,
                luisa::span<BasicBlock *const>{
                    entries.data(), entries.size()},
                dom, &structured_loops);
            if (merge == nullptr) {
                auto ipm_it = pdom.ipostdom.find(bb);
                if (ipm_it == pdom.ipostdom.end() ||
                    ipm_it->second == nullptr) {
                    return;
                }
                merge = ipm_it->second;
            }
        }
        if (merge == pdom.virtual_exit) { return; }
        if (merge == bb) { return; }

        if (!dom.strictly_dominates(bb, true_bb)) { return; }
        if (!dom.strictly_dominates(bb, false_bb)) { return; }

        candidates.push_back(
            {bb, cbr, dom_depth(dom, bb)});
    });

    if (candidates.empty()) { return false; }

    // Sort by depth descending (innermost first)
    luisa::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
        return a.depth > b.depth;
    });

    bool any = false;
    auto &created_structural_merges = all_created_structural_merges;
    auto pdom_valid = true;

    // Process all candidates from innermost to outermost.
    // Since we process innermost first, restructuring an inner if does not
    // invalidate the dom/pdom for outer if-candidates. We re-validate each
    // candidate before processing to guard against stale state.
    for (auto &cand : candidates) {
        auto *found_header = cand.header;
        auto *found_cbr = cand.cbr;

        // Re-validate: header may have been restructured by a previous candidate in this batch.
        if (!found_header->is_terminated()) { continue; }
        auto *check_term = found_header->terminator();
        if (!check_term->isa<ConditionalBranchInst>()) { continue; }
        if (static_cast<ConditionalBranchInst *>(check_term) != found_cbr) { continue; }

        auto *true_bb = found_cbr->true_block();
        auto *false_bb = found_cbr->false_block();
        auto *cond = found_cbr->condition();
        if (true_bb == nullptr || false_bb == nullptr ||
            true_bb == false_bb ||
            !dom.contains(found_header) ||
            !dom.strictly_dominates(
                found_header, true_bb) ||
            !dom.strictly_dominates(
                found_header, false_bb)) {
            continue;
        }
        auto entries = std::array{true_bb, false_bb};
        // Same O(1) post-dominator fast path as candidate collection.
        BasicBlock *found_merge = nullptr;
        if (pdom_valid) {
            auto merge_iter = pdom.ipostdom.find(found_header);
            if (merge_iter != pdom.ipostdom.end() &&
                merge_iter->second != nullptr &&
                merge_iter->second != pdom.virtual_exit &&
                merge_iter->second != found_header) {
                found_merge = merge_iter->second;
            }
        }
        if (found_merge == nullptr) {
            found_merge = infer_selection_merge(
                def, found_header,
                luisa::span<BasicBlock *const>{
                    entries.data(), entries.size()},
                dom, &structured_loops);
        }
        if (found_merge == nullptr) {
            // Post-dominance is a fallback for candidates whose merge cannot
            // be inferred from the current dominance tree. Rebuild it lazily
            // after a mutation, immediately before the first query that
            // observes the new CFG. This is equivalent to eager rebuilding
            // after every rewrite while avoiding analyses that no candidate
            // consumes.
            if (!pdom_valid) {
                pdom = compute_post_dom(def);
                ++info.if_batch_post_dom_rebuild_count;
                pdom_valid = true;
            }
            auto merge_iter =
                pdom.ipostdom.find(found_header);
            if (merge_iter == pdom.ipostdom.end() ||
                merge_iter->second == nullptr) {
                continue;
            }
            found_merge = merge_iter->second;
        }
        if (found_merge == pdom.virtual_exit ||
            found_merge == found_header) {
            continue;
        }

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
            unstructured_blocks.emplace_back(structural_merge);
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

        // The scope of (header, merge) is characterized by dominance: blocks
        // dominated by the header, not dominated by the merge, and not inside
        // an already-structured inner construct. After destructuring, such
        // constructs are rare (loops, previously restructured selections), so
        // testing membership against their merge subtrees directly is cheap.
        // Building a reachability set per candidate with a worklist is
        // quadratic in the block count for chains of conditional branches
        // (each candidate re-walks the whole tail of the chain).
        auto in_scope = [&](BasicBlock *bb) noexcept -> bool {
            if (!dom.dominates(found_header, bb)) { return false; }
            if (bb == found_merge || bb == structural_merge) { return false; }
            if (dom.dominates(found_merge, bb)) { return false; }
            if (structured_interior.contains(bb)) { return false; }
            return true;
        };

        // Only retarget unstructured cbr/br blocks that are actually inside
        // the if's scope. Skip IfInst/SwitchInst/LoopInst terminators to avoid
        // corrupting already-structured inner constructs. Equivalent to the
        // historical dominator-subtree walk for every block that can be
        // retargeted (the snapshot above stays complete), but linear in the
        // number of such blocks per candidate instead of the whole subtree.
        {
            for (auto *bb : unstructured_blocks) {
                if (bb != structural_merge && bb != found_header && bb != found_merge &&
                    bb->is_terminated() && in_scope(bb) &&
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

            // Do not eagerly clone successors that are not dominated by this
            // header. Such a reachable set is not necessarily a single-entry
            // region, and node splitting without explicit live-out transport
            // does not preserve SSA. The post-restructure single-exit protocol
            // below handles non-local exits with an explicit target selector
            // and typed value transport.
            info.restructured_if_count++;
            any = true;

            // Keep draining the dominance snapshot. Candidates are ordered
            // innermost first, and a successful rewrite replaces one raw
            // ConditionalBranch with one structured If plus a transparent
            // edge subdivision at its merge. It neither introduces a raw
            // conditional nor changes dominance between pre-existing blocks.
            // Therefore sibling candidates remain independent and outer
            // candidates remain valid after following any structural-merge
            // chain above. The per-candidate terminator identity check is the
            // fail-closed guard for every other stale candidate.
            //
            // This makes the number of remaining raw conditionals a strict
            // descent measure for the batch. Returning after one rewrite
            // would instead rescan the complete candidate set once per
            // conditional, turning a linear dispatch chain into quadratic
            // (or worse, because merge inference walks the CFG) work.
            //
            // Refresh dominance after every mutation so subsequent scope
            // walks include newly inserted structural merges. Mark post-dom
            // stale; the fallback above refreshes it only if a later
            // candidate actually requires an immediate post-dominator.
            dom = compute_dom_tree(def, false);
            pdom_valid = false;
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
    auto *merge = structured_statement_merge(term);
    auto append_entry = [&](BasicBlock *entry) noexcept {
        if (entry == nullptr || entry == merge) { return; }
        for (auto *existing : entries) {
            if (existing == entry) { return; }
        }
        entries.emplace_back(entry);
    };
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::IF: {
            auto *ii = static_cast<IfInst *>(term);
            append_entry(ii->true_block());
            append_entry(ii->false_block());
            break;
        }
        case DerivedInstructionTag::SWITCH: {
            auto *sw = static_cast<SwitchInst *>(term);
            for (size_t i = 0; i < sw->case_count(); i++) {
                append_entry(sw->case_block(i));
            }
            append_entry(sw->default_block());
            break;
        }
        case DerivedInstructionTag::LOOP: {
            auto *lp = static_cast<LoopInst *>(term);
            append_entry(lp->prepare_block());
            // body/update are loop-internal; they may legitimately have multiple preds.
            break;
        }
        case DerivedInstructionTag::SIMPLE_LOOP: {
            auto *sl = static_cast<SimpleLoopInst *>(term);
            append_entry(sl->body_block());
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

[[nodiscard]] bool is_opaque_ray_query_type(
    const Type *type) noexcept {
    return type == Type::custom("LC_RayQueryAll") ||
           type == Type::custom("LC_RayQueryAny");
}

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
                                                 const DomTree &dom,
                                                 bool lower_cloned_structured_branches = false) noexcept {
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

    // Ray-query objects are affine state: one direct query initializer binds
    // one local alloca, and the object is then mutated in place. Node splitting
    // duplicates a mutually exclusive execution path. If that path contains
    // the binding store, sharing the original alloca would create two static
    // initializers for one opaque object; copying the object through ordinary
    // state is likewise undefined. Give the cloned path its own storage and
    // remap every cloned use to it. Ordinary allocas intentionally remain
    // shared so state-dispatch transport retains its value semantics.
    luisa::vector<AllocaInst *> affine_allocas;
    luisa::unordered_set<AllocaInst *> seen_affine_allocas;
    for (auto *old_bb : ordered) {
        for (auto *old_inst : old_bb->instructions()) {
            if (!old_inst->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(old_inst);
            auto *variable = store->variable();
            auto *value = store->value();
            if (variable == nullptr ||
                !variable->isa<AllocaInst>() ||
                value == nullptr ||
                !is_opaque_ray_query_type(variable->type()) ||
                value->type() != variable->type()) {
                continue;
            }
            auto *alloca =
                static_cast<AllocaInst *>(variable);
            if (seen_affine_allocas.emplace(alloca).second) {
                affine_allocas.emplace_back(alloca);
            }
        }
    }
    if (!affine_allocas.empty()) {
        XIRBuilder alloca_builder;
        alloca_builder.set_insertion_point(
            def->body_block()
                ->instructions()
                .head_sentinel());
        for (auto *old_alloca : affine_allocas) {
            auto *new_alloca = static_cast<AllocaInst *>(
                old_alloca->clone_with_metadata(
                    alloca_builder, remap));
            new_alloca->add_comment(
                "opaque state cloned for a split CFG path");
            remap.map[old_alloca] = new_alloca;
        }
    }

    // Clone instructions of each region block into its counterpart.
    XIRBuilder builder;
    for (auto *old_bb : ordered) {
        auto *new_bb = static_cast<BasicBlock *>(remap.map[old_bb]);
        builder.set_insertion_point(new_bb);
        for (auto *old_inst : old_bb->instructions()) {
            Instruction *new_inst = nullptr;
            if (lower_cloned_structured_branches &&
                (old_inst->isa<BreakInst>() ||
                 old_inst->isa<ContinueInst>())) {
                auto *old_branch = static_cast<
                    BranchTerminatorInstruction *>(old_inst);
                auto *new_target = static_cast<BasicBlock *>(
                    remap.resolve(
                        old_branch->target_block()));
                new_inst = builder.br(new_target);
                for (auto *metadata :
                     old_inst->metadata_list()) {
                    new_inst->metadata_list().push_front(
                        metadata->clone());
                }
            } else {
                new_inst = old_inst->clone_with_metadata(
                    builder, remap);
            }
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

struct PostMergeSelectionReentry {
    BasicBlock *header{nullptr};
    BasicBlock *merge{nullptr};
    BasicBlock *reentered_block{nullptr};
    BasicBlock *reentry_predecessor{nullptr};
    luisa::vector<BasicBlock *> entries;
};

// An exit-state dispatch can re-enter an arm of a selection that has already
// merged. In graph terms, the original selection edge and the dispatch edge
// are two entries into the newly formed cycle, so wrapping the dispatch in
// another selection cannot be valid structured control flow. It also cannot
// converge: single-exit canonicalization recreates the same dispatch.
//
// For every edge (P, E) on a side-effect-free forwarding chain starting at a
// dispatch arm, find the deepest selection (H, M) for which H and M dominate
// P, H dominates E, and M does not dominate E. This dominance predicate is
// exactly the definition of crossing from the post-merge region back into the
// selection interior; it does not depend on E being a declared arm entry.
//
// Split the E-owned subgraph with H, M, and sibling entries as its frontier,
// then retarget P to the copy. The copy is dominated by M, while the original
// interior loses this post-merge predecessor. Thus the offending boundary edge
// is removed instead of being hidden behind another selection. Forwarding
// chains and owned subgraphs are finite and cycle-guarded, and selecting the
// deepest owner applies the standard inner-to-outer node-splitting reduction
// for a multi-entry region. The normal loop structurizer handles any resulting
// natural loop on the next fixed-point iteration.
[[nodiscard]] bool split_one_exit_dispatch_selection_reentry(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom,
    RestructureCFGInfo &info,
    const luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    if (exit_dispatch_headers.empty()) { return false; }
    // Loop-boundary membership is a relation of the current immutable CFG.
    // Materialize it once rather than traversing every loop region for every
    // selection considered by every dispatch edge.
    ++info.selection_reentry_boundary_analysis_count;
    const auto loop_boundary_selection_entries =
        collect_loop_boundary_selection_entries(def);
    for (auto *dispatch : exit_dispatch_headers) {
        if (dispatch == nullptr || !dispatch->is_terminated() ||
            !dispatch->terminator()
                 ->isa<ConditionalBranchInst>() ||
            !dom.contains(dispatch)) {
            continue;
        }
        auto *branch = static_cast<ConditionalBranchInst *>(
            dispatch->terminator());
        auto arm_entries =
            std::array{branch->true_block(),
                       branch->false_block()};
        for (auto *arm_entry : arm_entries) {
            if (arm_entry == nullptr) { continue; }
            PostMergeSelectionReentry reentry;
            auto *reentry_predecessor = dispatch;
            auto *reentered_block = arm_entry;
            luisa::unordered_set<BasicBlock *> path;
            while (reentered_block != nullptr &&
                   path.emplace(reentered_block).second) {
                if (dom.contains(reentry_predecessor) &&
                    dom.contains(reentered_block) &&
                    terminator_targets(
                        reentry_predecessor->terminator(),
                        reentered_block)) {
                    ++info.selection_reentry_edge_query_count;
                    // H dominates E iff H is an ancestor of E in the
                    // dominator tree. Walk those ancestors from deepest to
                    // shallowest: the first selection satisfying the other
                    // three dominance predicates is therefore exactly the
                    // deepest owner chosen by the former all-block scan.
                    for (auto *candidate_node =
                             dom.node(reentered_block)->parent();
                         candidate_node != nullptr;
                         candidate_node =
                             candidate_node->parent()) {
                        auto *candidate_header =
                            candidate_node->block();
                        if (candidate_header == nullptr ||
                            !candidate_header->is_terminated() ||
                            exit_dispatch_headers.contains(
                                candidate_header) ||
                            loop_boundary_selection_entries.contains(
                                candidate_header)) {
                            continue;
                        }
                        auto *term =
                            candidate_header->terminator();
                        if (!term->isa<IfInst>() &&
                            !term->isa<SwitchInst>()) {
                            continue;
                        }
                        ++info.selection_reentry_owner_query_count;
                        auto *merge =
                            structured_statement_merge(term);
                        // An edge (P, E) is a post-merge re-entry exactly
                        // when H and M dominate P while H, but not M,
                        // dominates E. H dominates E by construction of this
                        // ancestor walk.
                        if (merge == nullptr ||
                            !dom.contains(merge) ||
                            !dom.dominates(
                                candidate_header,
                                reentry_predecessor) ||
                            !dom.dominates(
                                merge,
                                reentry_predecessor) ||
                            dom.dominates(
                                merge,
                                reentered_block)) {
                            continue;
                        }
                        luisa::vector<BasicBlock *> entries;
                        collect_construct_entries(
                            candidate_header, entries);
                        reentry = {
                            .header = candidate_header,
                            .merge = merge,
                            .reentered_block =
                                reentered_block,
                            .reentry_predecessor =
                                reentry_predecessor,
                            .entries = std::move(entries)};
                        break;
                    }
                }
                auto *next =
                    trivial_branch_target(reentered_block);
                if (next == nullptr) { break; }
                reentry_predecessor = reentered_block;
                reentered_block = next;
            }
            if (reentry.header == nullptr) { continue; }

            // Cloning duplicates definitions along mutually exclusive paths.
            // Transport cross-block values through typed local state first, so
            // either the original arm or its clone writes the same slot before
            // the common continuation reloads it.
            repair_target_state_dispatch_ssa(def);
            dom = compute_dom_tree(def, false);
            pdom = compute_post_dom(def);
            LUISA_ASSERT(
                clone_owned_subgraph_for_edge(
                    def, reentry.header,
                    reentry.reentered_block,
                    reentry.reentry_predecessor,
                    luisa::span<BasicBlock *const>{
                        reentry.entries.data(),
                        reentry.entries.size()},
                    reentry.merge, dom, true),
                "Selection re-entry node splitting made no progress.");
            ++info.canonicalized_cfg_count;
            dom = compute_dom_tree(def, false);
            pdom = compute_post_dom(def);
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool split_exit_dispatch_selection_reentries(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom,
    RestructureCFGInfo &info,
    const luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    ScopedTimer _timer_split_exit_dispatch_selection_reentries(
        "split_exit_dispatch_selection_reentries");
    auto modified = false;
    while (split_one_exit_dispatch_selection_reentry(
        def, dom, pdom, info,
        exit_dispatch_headers)) {
        modified = true;
    }
    return modified;
}

// Per-construct entry-uniqueness fix. Returns true if any edges were rewritten.
[[nodiscard]] bool enforce_construct_entries(FunctionDefinition *def,
                                             BasicBlock *header_bb,
                                             BasicBlock *merge_bb,
                                             RestructureCFGInfo &info,
                                             DomTree &dom,
                                             bool &dom_valid,
                                             luisa::unordered_set<Instruction *> &rewritten_sites) noexcept {
    ScopedTimer _timer_enforce_entries("enforce_construct_entries");
    luisa::vector<BasicBlock *> entries;
    collect_construct_entries(header_bb, entries);
    if (entries.size() <= 1u) { return false; }
    bool changed_any = false;
    bool site_claimed = false;
    auto *site = header_bb->terminator();
    // Iterate entries in their natural order; per Oracle's design, if the sibling-entry
    // graph is acyclic, fixing earlier entries does not create new bad edges into them.
    for (auto *E : entries) {
        luisa::unordered_set<BasicBlock *> rewritten_predecessors;
        for (;;) {
            if (!dom_valid) {
                dom = compute_dom_tree(def, false);
                ++info.construct_entry_dom_tree_count;
                dom_valid = true;
            }
            // Structured-entry legality is defined over the executable CFG.
            // Owned but disconnected blocks are deliberately absent from the
            // dominance tree and cannot introduce another dynamic entry.
            if (!dom.contains(header_bb) || !dom.contains(E)) {
                break;
            }
            luisa::vector<BasicBlock *> offenders;
            E->traverse_predecessors(false, [&](BasicBlock *P) noexcept {
                if (!dom.contains(P)) { return; }
                if (!is_authorized_construct_pred(header_bb->terminator(), E, header_bb, P)) {
                    offenders.emplace_back(P);
                }
            });
            if (offenders.empty()) { break; }
            if (!site_claimed && rewritten_sites.contains(site)) {
                ++info.iteration_limit_count;
                return changed_any;
            }
            for (auto *predecessor : offenders) {
                if (rewritten_predecessors.contains(predecessor)) {
                    ++info.iteration_limit_count;
                    return changed_any;
                }
            }
            bool local_change = false;
            for (auto *P : offenders) {
                if (clone_owned_subgraph_for_edge(def, header_bb, E, P,
                                                  luisa::span<BasicBlock *const>{entries},
                                                  merge_bb, dom)) {
                    local_change = true;
                    rewritten_predecessors.emplace(P);
                }
            }
            if (!local_change) { break; }
            if (!site_claimed) {
                rewritten_sites.emplace(site);
                site_claimed = true;
            }
            changed_any = true;
            // The CFG was modified; the dom tree is now stale.
            dom_valid = false;
        }
    }
    return changed_any;
}

// Visit each structured construct (If/Switch/Loop/SimpleLoop) and enforce the
// invariant. We rescan after each change because the BB list has grown.
void enforce_unique_construct_entries(FunctionDefinition *def,
                                      RestructureCFGInfo &info) noexcept {
    ScopedTimer _timer_enforce_unique("enforce_unique_construct_entries");
    luisa::unordered_set<Instruction *> rewritten_sites;
    // The dominance tree is a function of the executable CFG, not of the
    // construct being inspected. Reuse it across every no-change construct
    // and fixed-point rescan. enforce_construct_entries invalidates it after
    // each mutation batch and rebuilds it before the next dominance query.
    DomTree dom;
    bool dom_valid = false;
    for (;;) {
        auto changed = false;
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
            auto limits_before = info.iteration_limit_count;
            if (enforce_construct_entries(
                    def, hbb, mbb, info, dom, dom_valid,
                    rewritten_sites)) {
                ++info.canonicalized_cfg_count;
                changed = true;
                break;// restart outer loop: BB list and dominance changed
            }
            if (info.iteration_limit_count != limits_before) {
                changed = false;
                return;
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
[[nodiscard]] static bool add_header_to_one_remaining_divergent(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom,
    RestructureCFGInfo &info,
    const luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    ScopedTimer _timer_add_header("add_header_to_remaining_divergent");

    // Recompute structured metadata fresh.
    luisa::unordered_set<BasicBlock *> header_set;
    luisa::unordered_set<BasicBlock *> continue_set;
    luisa::unordered_set<BasicBlock *> loop_prepare_set;
    luisa::unordered_set<BasicBlock *> loop_merge_set;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        auto tag = term->derived_instruction_tag();
        if (tag == DerivedInstructionTag::IF || tag == DerivedInstructionTag::SWITCH ||
            tag == DerivedInstructionTag::LOOP || tag == DerivedInstructionTag::SIMPLE_LOOP) {
            header_set.emplace(bb);
        }
        if (term->isa<LoopInst>()) {
            auto *lp = static_cast<LoopInst *>(term);
            if (lp->merge_block()) { loop_merge_set.emplace(lp->merge_block()); }
            if (lp->update_block()) { continue_set.emplace(lp->update_block()); }
            if (lp->prepare_block()) {
                continue_set.emplace(lp->prepare_block());
                loop_prepare_set.emplace(lp->prepare_block());
            }
        } else if (term->isa<SimpleLoopInst>()) {
            auto *sl = static_cast<SimpleLoopInst *>(term);
            if (sl->merge_block()) { loop_merge_set.emplace(sl->merge_block()); }
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
        if (loop_prepare_set.contains(bb)) { continue; }
        // A single-exit rewrite ends in a raw dispatch *after* the rewritten
        // construct's merge. LLVM's structurizer deliberately leaves this as
        // an ordinary conditional when one arm names an enclosing construct
        // boundary. Turning it into a fresh selection would make that arm
        // leave the new selection without passing through its merge.
        if (exit_dispatch_headers.contains(bb)) { continue; }
        if (!bb->is_terminated()) { continue; }
        auto *term = bb->terminator();
        if (!term->isa<ConditionalBranchInst>()) { continue; }
        auto *cbr = static_cast<ConditionalBranchInst *>(term);

        auto *t = cbr->true_block();
        auto *f = cbr->false_block();
        if (t == nullptr || f == nullptr || t == f) { continue; }
        if (continue_set.contains(t) || continue_set.contains(f) ||
            loop_merge_set.contains(t) ||
            loop_merge_set.contains(f)) {
            continue;
        }

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
                if (header_set.contains(cur)) {
                    if (auto *nested_merge = structured_statement_merge(cur->terminator());
                        nested_merge != nullptr && nested_merge != merge) {
                        work.emplace_back(nested_merge);
                    }
                    continue;
                }
                if (continue_set.contains(cur)) {
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
    dom = compute_dom_tree(def, false);
    pdom = compute_post_dom(def);
    return true;
}

[[nodiscard]] static bool add_headers_to_remaining_divergent(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom,
    RestructureCFGInfo &info,
    const luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    auto modified = false;
    // Like loop-boundary normalization, every successful rewrite consumes one
    // raw ConditionalBranchInst and cannot rediscover the same site.
    while (add_header_to_one_remaining_divergent(
        def, dom, pdom, info, exit_dispatch_headers)) {
        modified = true;
    }
    return modified;
}

// Ensure each structured construct's executable exits respect the SPIR-V
// hierarchy. This follows LLVM SPIRVStructurizer::fixupConstruct:
//
// 1. Rebuild the construct tree from the current merge declarations.
// 2. Visit constructs from inner to outer.
// 3. Compute the construct block set from dominance and ancestor boundaries.
// 4. If an exit targets anything except the construct's own merge/continue,
//    route *all* exits through one new merge, carrying the old target as state.
// 5. Invalidate dominance/post-dominance and rebuild after one rewrite.
//
// Exit-state dispatch headers are intentionally transparent here. LLVM emits
// them as raw branches after the new merge rather than as child constructs;
// XIR may temporarily wrap them in IfInst, so the explicit set preserves the
// same semantics.
[[nodiscard]] static bool fixup_construct_exits(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom,
    RestructureCFGInfo &info,
    luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    ScopedTimer _timer_fixup_exits("fixup_construct_exits");
    static_cast<void>(info);
    auto modified = false;

    // Loop-boundary IfInst nodes are spelled once per immutable CFG; fixup
    // only retargets exits and never changes them, so the membership set is
    // stable for the whole call. The per-header predicate would re-walk every
    // loop region for every construct (quadratic in the block count).
    auto loop_boundary_entries =
        collect_loop_boundary_selection_entries(def);

    for (;;) {
        struct Construct {
            BasicBlock *header{nullptr};
            Instruction *term{nullptr};
            BasicBlock *merge{nullptr};
            BasicBlock *continue_target{nullptr};
            size_t depth{0u};
            Construct *parent{nullptr};
        };
        luisa::vector<Construct> constructs;
        def->traverse_basic_blocks([&](BasicBlock *header) noexcept {
            if (header == nullptr || !header->is_terminated() ||
                !dom.contains(header) ||
                exit_dispatch_headers.contains(header) ||
                loop_boundary_entries.contains(header)) {
                return;
            }
            auto *term = header->terminator();
            auto *merge = structured_statement_merge(term);
            if (merge == nullptr) { return; }
            BasicBlock *continue_target = nullptr;
            if (term->isa<LoopInst>()) {
                auto *loop = static_cast<LoopInst *>(term);
                continue_target = loop->update_block();
                if (continue_target == nullptr) {
                    continue_target = loop->prepare_block();
                }
            } else if (term->isa<SimpleLoopInst>()) {
                continue_target =
                    static_cast<SimpleLoopInst *>(term)
                        ->body_block();
            } else if (!term->isa<IfInst>() &&
                       !term->isa<SwitchInst>()) {
                return;
            }
            constructs.emplace_back(Construct{
                .header = header,
                .term = term,
                .merge = merge,
                .continue_target = continue_target,
                .depth = dom_depth(dom, header)});
        });
        if (constructs.empty()) { break; }

        auto encloses = [&](const Construct &outer,
                            const Construct &inner) noexcept {
            if (&outer == &inner ||
                !dom.contains(outer.header) ||
                !dom.contains(inner.header) ||
                !dom.strictly_dominates(
                    outer.header, inner.header) ||
                inner.header == outer.merge ||
                inner.header == outer.continue_target) {
                return false;
            }
            return !dom.contains(outer.merge) ||
                   !dom.dominates(
                       outer.merge, inner.header);
        };
        for (auto &inner : constructs) {
            for (auto &outer : constructs) {
                if (!encloses(outer, inner)) { continue; }
                if (inner.parent == nullptr ||
                    outer.depth > inner.parent->depth) {
                    inner.parent = &outer;
                }
            }
        }

        luisa::vector<Construct *> construct_order;
        construct_order.reserve(constructs.size());
        for (auto &construct : constructs) {
            construct_order.emplace_back(&construct);
        }
        luisa::sort(
            construct_order.begin(), construct_order.end(),
            [](auto *lhs, auto *rhs) noexcept {
                return lhs->depth > rhs->depth;
            });

        Construct *candidate = nullptr;
        luisa::vector<SelectionExitEdge> candidate_exits;
        for (auto *node_ptr : construct_order) {
            auto &node = *node_ptr;
            if (node.parent == nullptr) { continue; }
            luisa::unordered_set<BasicBlock *> outside_boundaries;
            for (auto *ancestor = node.parent;
                 ancestor != nullptr;
                 ancestor = ancestor->parent) {
                outside_boundaries.emplace(ancestor->merge);
                if (ancestor->continue_target != nullptr) {
                    outside_boundaries.emplace(
                        ancestor->continue_target);
                }
            }

            luisa::unordered_set<BasicBlock *> blocks;
            luisa::vector<BasicBlock *> work{node.header};
            while (!work.empty()) {
                auto *block = work.back();
                work.pop_back();
                if (block == nullptr || block == node.merge ||
                    outside_boundaries.contains(block) ||
                    !dom.contains(block) ||
                    !dom.dominates(node.header, block) ||
                    (dom.contains(node.merge) &&
                     dom.dominates(node.merge, block))) {
                    continue;
                }
                if (!blocks.emplace(block).second) { continue; }
                traverse_executable_successors(
                    block, [&](BasicBlock *successor) noexcept {
                        work.emplace_back(successor);
                    });
            }

            luisa::vector<SelectionExitEdge> exits;
            for (auto *block : def->basic_blocks()) {
                if (!blocks.contains(block)) { continue; }
                traverse_executable_successors(
                    block, [&](BasicBlock *successor) noexcept {
                        if (!blocks.contains(successor)) {
                            append_unique_exit_edge(
                                exits, block, successor);
                        }
                    });
            }
            if (exits.empty()) { continue; }
            auto bad = node.merge == node.parent->merge ||
                       node.merge ==
                           node.parent->continue_target;
            for (auto edge : exits) {
                if (edge.dst != node.merge &&
                    edge.dst != node.continue_target) {
                    bad = true;
                }
            }
            if (bad) {
                candidate = &node;
                // A loop's continue target is internal to that loop, not an
                // exit from the construct. Retargeting it through the fresh
                // merge creates a state dispatch between "continue" and
                // "break"; after an enclosing loop is recovered both arms
                // can become the same outer continue, leaving the loop
                // prepare branch behind a non-canonical proxy. Selection
                // constructs have no continue target, so they still route
                // every normal and non-local exit through their new merge.
                for (auto edge : exits) {
                    if (edge.dst != node.continue_target) {
                        candidate_exits.emplace_back(edge);
                    }
                }
                break;
            }
        }
        if (candidate == nullptr) { break; }

        luisa::unordered_map<BasicBlock *, uint32_t> target_ids;
        luisa::vector<BasicBlock *> targets;
        for (auto edge : candidate_exits) {
            if (!target_ids.contains(edge.dst)) {
                auto id = static_cast<uint32_t>(
                    targets.size());
                target_ids.emplace(edge.dst, id);
                targets.emplace_back(edge.dst);
            }
        }
        LUISA_ASSERT(
            !targets.empty(),
            "Construct fixup selected a construct without exit targets.");

        auto *new_exit = def->create_basic_block();
        auto retargeted_any = false;
        XIRBuilder builder;
        if (targets.size() == 1u) {
            for (auto edge : candidate_exits) {
                retargeted_any |=
                    retarget_structured_exit_to(
                        edge.src->terminator(),
                        edge.dst, new_exit);
                fix_degenerate_terminator(edge.src);
            }
            builder.set_insertion_point(new_exit);
            builder.br(targets.front());
        } else {
            builder.set_insertion_point(
                def->body_block()
                    ->instructions()
                    .head_sentinel());
            auto *selector =
                builder.alloca_local(Type::of<uint32_t>());
            for (auto edge : candidate_exits) {
                auto *stub = def->create_basic_block();
                if (!retarget_structured_exit_to(
                        edge.src->terminator(),
                        edge.dst, stub)) {
                    stub->remove_self();
                    continue;
                }
                retargeted_any = true;
                fix_degenerate_terminator(edge.src);
                auto id = target_ids.at(edge.dst);
                auto *constant =
                    def->parent_module()->create_constant(
                        Type::of<uint32_t>(), &id);
                builder.set_insertion_point(stub);
                builder.store(selector, constant);
                builder.br(new_exit);
            }
            builder.set_insertion_point(new_exit);
            auto *loaded =
                builder.load(Type::of<uint32_t>(), selector);
            auto *dispatch = def->create_basic_block();
            builder.br(dispatch);
            builder.set_insertion_point(dispatch);
            for (auto i = size_t{0u};
                 i + 1u < targets.size(); ++i) {
                exit_dispatch_headers.emplace(dispatch);
                auto id = target_ids.at(targets[i]);
                auto *constant =
                    def->parent_module()->create_constant(
                        Type::of<uint32_t>(), &id);
                auto *condition = builder.call(
                    Type::of<bool>(),
                    ArithmeticOp::BINARY_EQUAL,
                    {loaded, constant});
                auto *next = def->create_basic_block();
                builder.cond_br(
                    condition, targets[i], next);
                builder.set_insertion_point(next);
                dispatch = next;
            }
            builder.br(targets.back());
        }
        LUISA_ASSERT(
            retargeted_any,
            "Construct fixup selected exits that could not be retargeted.");
        auto *control_flow_merge =
            candidate->term->control_flow_merge();
        LUISA_ASSERT(
            control_flow_merge != nullptr &&
                control_flow_merge->merge_block() ==
                    candidate->merge,
            "Construct merge changed during one atomic fixup.");
        control_flow_merge->set_merge_block(new_exit);
        if (targets.size() > 1u) {
            repair_target_state_dispatch_ssa(def);
        }
        modified = true;

        // LLVM Splitter::invalidate(): all containment and exit facts above are
        // stale after one rewrite.
        dom = compute_dom_tree(def, false);
        pdom = compute_post_dom(def);
    }
    return modified;
}

[[nodiscard]] size_t count_unstructured_conditional_branches(
    FunctionDefinition *def) noexcept {
    luisa::unordered_map<BasicBlock *, LoopInst *> loop_prepares;
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr || !block->is_terminated() ||
            !block->terminator()->isa<LoopInst>()) {
            continue;
        }
        auto *loop = static_cast<LoopInst *>(block->terminator());
        if (loop->prepare_block() != nullptr) {
            loop_prepares.emplace(loop->prepare_block(), loop);
        }
    }
    size_t count = 0u;
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr || !block->is_terminated()) {
            continue;
        }
        if (block->terminator()->isa<IndexedBranchInst>()) {
            ++count;
            continue;
        }
        if (!block->terminator()->isa<ConditionalBranchInst>()) { continue; }
        auto *branch = static_cast<ConditionalBranchInst *>(block->terminator());
        auto iter = loop_prepares.find(block);
        auto canonical_loop_prepare = iter != loop_prepares.end() &&
                                      branch->condition() != nullptr &&
                                      branch->condition()->type() == Type::of<bool>() &&
                                      branch->true_block() == iter->second->body_block() &&
                                      branch->false_block() == iter->second->merge_block();
        count += canonical_loop_prepare ? 0u : 1u;
    }
    return count;
}

[[nodiscard]] size_t count_invalid_structured_constructs(
    FunctionDefinition *def) noexcept {
    auto valid_block = [&](BasicBlock *block) noexcept {
        return block != nullptr && block->parent_function() == def;
    };
    size_t count = 0u;
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr || !block->is_terminated()) {
            count += block == nullptr ? 0u : 1u;
            if (block != nullptr) {
                LUISA_VERBOSE_WITH_LOCATION("restructure_cfg: unterminated owned block {}.",
                                            static_cast<void *>(block));
            }
            continue;
        }
        auto *term = block->terminator();
        auto invalid = false;
        switch (term->derived_instruction_tag()) {
            case DerivedInstructionTag::BRANCH:
                invalid = !valid_block(static_cast<BranchInst *>(term)->target_block());
                break;
            case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto *branch = static_cast<ConditionalBranchInst *>(term);
                invalid = branch->condition() == nullptr ||
                          branch->condition()->type() != Type::of<bool>() ||
                          !valid_block(branch->true_block()) ||
                          !valid_block(branch->false_block());
                break;
            }
            case DerivedInstructionTag::IF: {
                auto *if_inst = static_cast<IfInst *>(term);
                invalid = if_inst->condition() == nullptr ||
                          if_inst->condition()->type() != Type::of<bool>() ||
                          !valid_block(if_inst->true_block()) ||
                          !valid_block(if_inst->false_block());
                break;
            }
            case DerivedInstructionTag::SWITCH:
            case DerivedInstructionTag::INDEXED_BRANCH: {
                auto *indexed_branch = static_cast<
                    IndexedBranchTerminatorInstruction *>(term);
                invalid = indexed_branch->value() == nullptr ||
                          !valid_block(indexed_branch->default_block());
                for (size_t i = 0u;
                     i < indexed_branch->case_count() && !invalid; ++i) {
                    invalid = !valid_block(
                        indexed_branch->case_block(i));
                }
                if (!invalid && term->isa<SwitchInst>()) {
                    invalid = !valid_block(
                        static_cast<SwitchInst *>(term)->merge_block());
                }
                break;
            }
            case DerivedInstructionTag::LOOP: {
                auto *loop = static_cast<LoopInst *>(term);
                invalid = !valid_block(loop->prepare_block()) ||
                          !valid_block(loop->body_block()) ||
                          !valid_block(loop->update_block()) ||
                          !valid_block(loop->merge_block());
                break;
            }
            case DerivedInstructionTag::SIMPLE_LOOP: {
                auto *loop = static_cast<SimpleLoopInst *>(term);
                invalid = !valid_block(loop->body_block()) ||
                          !valid_block(loop->merge_block());
                break;
            }
            case DerivedInstructionTag::BREAK:
                invalid = !valid_block(static_cast<BreakInst *>(term)->target_block());
                break;
            case DerivedInstructionTag::CONTINUE:
                invalid = !valid_block(static_cast<ContinueInst *>(term)->target_block());
                break;
            default: break;
        }
        if (!invalid && term->control_flow_merge() != nullptr &&
            term->control_flow_merge()->merge_block() != nullptr) {
            invalid = !valid_block(term->control_flow_merge()->merge_block());
        }
        if (invalid) {
            LUISA_VERBOSE_WITH_LOCATION("restructure_cfg: invalid terminator tag {} in block {}.",
                                        static_cast<int>(term->derived_instruction_tag()),
                                        static_cast<void *>(block));
            ++count;
        }
    }
    return count;
}

[[nodiscard]] size_t count_unauthorized_construct_entries(
    FunctionDefinition *def) noexcept {
    size_t count = 0u;
    auto dom = compute_dom_tree(def, false);
    for (auto *header : def->basic_blocks()) {
        if (header == nullptr || !header->is_terminated() ||
            !dom.contains(header)) {
            continue;
        }
        auto *term = header->terminator();
        switch (term->derived_instruction_tag()) {
            case DerivedInstructionTag::IF:
            case DerivedInstructionTag::SWITCH:
            case DerivedInstructionTag::LOOP:
            case DerivedInstructionTag::SIMPLE_LOOP: break;
            default: continue;
        }
        luisa::vector<BasicBlock *> entries;
        collect_construct_entries(header, entries);
        if (entries.size() <= 1u) { continue; }
        auto invalid = false;
        for (auto *entry : entries) {
            entry->traverse_predecessors(false, [&](BasicBlock *predecessor) noexcept {
                // BasicBlock predecessor traversal follows the complete
                // use-list, including disconnected blocks retained for stable
                // ownership. Construct-entry legality is an executable-CFG
                // property, so only predecessors represented in the same
                // reachable dominance tree participate.
                if (!dom.contains(predecessor)) { return; }
                invalid |= !is_authorized_construct_pred(
                    term, entry, header, predecessor);
            });
        }
        count += invalid ? 1u : 0u;
    }
    return count;
}

[[nodiscard]] size_t count_post_merge_selection_reentries(
    FunctionDefinition *def) noexcept {
    size_t count = 0u;
    auto dom = compute_dom_tree(def, false);
    auto loop_boundary_entries =
        collect_loop_boundary_selection_entries(def);
    for (auto *header : def->basic_blocks()) {
        if (header == nullptr || !header->is_terminated() ||
            !dom.contains(header)) {
            continue;
        }
        auto *term = header->terminator();
        if (!term->isa<IfInst>() &&
            !term->isa<SwitchInst>()) {
            continue;
        }
        // Loop-boundary IfInst nodes are the XIR spelling of physical
        // break/continue guards. The SPIR-V emitter deliberately does not
        // declare them as selection constructs, so selection re-entry rules
        // do not apply to those provenance-checked nodes.
        if (term->isa<IfInst>() &&
            loop_boundary_entries.contains(header)) {
            continue;
        }
        auto *merge =
            term->control_flow_merge()->merge_block();
        if (merge == nullptr || !dom.contains(merge)) {
            continue;
        }

        // For a structured selection (H, M), its executable interior is the
        // H-dominated region before M. An edge from an M-dominated block back
        // into that region is precisely a second entry after the construct
        // has merged. SPIR-V rejects this graph even when the two declared arm
        // entries themselves have unique predecessors.
        auto invalid = false;
        for (auto *block : def->basic_blocks()) {
            if (block == nullptr || block == header ||
                block == merge || !dom.contains(block) ||
                !dom.dominates(header, block) ||
                dom.dominates(merge, block)) {
                continue;
            }
            block->traverse_predecessors(
                false,
                [&](BasicBlock *predecessor) noexcept {
                    invalid |=
                        dom.contains(predecessor) &&
                        dom.dominates(merge, predecessor);
                });
        }
        count += invalid ? 1u : 0u;
    }
    return count;
}

[[nodiscard]] RestructureCFGInfo preflight_restructure_cfg(
    FunctionDefinition *def,
    bool verify_intermediate) noexcept {
    ScopedTimer _timer_preflight("preflight_restructure_cfg");
    RestructureCFGInfo info{};
    {
        ScopedTimer _timer_phi("preflight_count_phi");
        for (auto *block : def->basic_blocks()) {
            if (block == nullptr) { continue; }
            for (auto *inst : block->instructions()) {
                info.invalid_construct_count +=
                    inst->isa<PhiInst>() ? 1u : 0u;
            }
        }
    }
    {
        ScopedTimer _timer_constructs(
            "preflight_count_invalid_structured_constructs");
        info.invalid_construct_count +=
            count_invalid_structured_constructs(def);
    }
    // The bespoke count above records the transform-specific Phi/ownership
    // preconditions. The verifier closes the rest of the input contract:
    // selector types, canonical and unique indexed-branch labels, target
    // ownership, use-def linkage, and SSA dominance must all hold before the
    // first structural merge block is allocated.
    if (info.invalid_construct_count == 0u &&
        verify_intermediate) {
        ScopedTimer _timer_verify("preflight_verify_function");
        ++info.intermediate_verifier_count;
        auto verification = xir_verify_function(
            static_cast<Function *>(def));
        if (!verification.succeeded()) {
            LUISA_WARNING_WITH_LOCATION(
                "restructure_cfg preflight verifier rejected the input: {}",
                verification.errors.front().message);
            ++info.invalid_construct_count;
        }
    }
    if (info.invalid_construct_count != 0u) {
        info.unstructured_branch_count =
            count_unstructured_conditional_branches(def);
        return info;
    }
    {
        ScopedTimer _timer_irreducible(
            "preflight_count_irreducible_regions");
        info.irreducible_region_count =
            count_irreducible_regions(def);
    }
    if (info.irreducible_region_count != 0u) {
        info.unstructured_branch_count =
            count_unstructured_conditional_branches(def);
    }
    return info;
}

class TransactionCloneResolver final
    : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _values;

public:
    void map(const Value *source, Value *clone) noexcept {
        LUISA_ASSERT(source != nullptr && clone != nullptr,
                     "Invalid transaction-clone mapping.");
        auto [iter, inserted] = _values.emplace(source, clone);
        LUISA_ASSERT(inserted || iter->second == clone,
                     "Conflicting transaction-clone mapping.");
    }

    [[nodiscard]] Value *resolve(
        const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        if (value->is_global()) {
            return const_cast<Value *>(value);
        }
        if (auto iter = _values.find(value);
            iter != _values.end()) {
            return iter->second;
        }
        return nullptr;
    }
};

struct ShadowDefinition {
    FunctionDefinition *source{nullptr};
    CallableFunction *shadow{nullptr};
};

[[nodiscard]] bool clone_definition_for_transaction(
    FunctionDefinition *source,
    ShadowDefinition &result) noexcept {
    if (source == nullptr || source->body_block() == nullptr) {
        return false;
    }
    auto *module = source->parent_module();
    auto *shadow = module->create_callable(source->type());
    result = {.source = source, .shadow = shadow};
    TransactionCloneResolver resolver;

    for (auto *argument : source->arguments()) {
        Argument *cloned_argument = nullptr;
        switch (argument->derived_argument_tag()) {
            case DerivedArgumentTag::VALUE:
                cloned_argument =
                    shadow->create_value_argument(argument->type());
                break;
            case DerivedArgumentTag::REFERENCE:
                cloned_argument =
                    shadow->create_reference_argument(argument->type());
                break;
            case DerivedArgumentTag::RESOURCE:
                cloned_argument =
                    shadow->create_resource_argument(argument->type());
                break;
        }
        LUISA_ASSERT(cloned_argument != nullptr,
                     "Failed to clone function argument.");
        resolver.map(argument, cloned_argument);
    }

    struct BlockClone {
        BasicBlock *source;
        BasicBlock *target;
        luisa::vector<Instruction *> instructions;
        size_t next_instruction{0u};
    };
    luisa::vector<BlockClone> blocks;
    blocks.reserve(source->basic_blocks().count_size());
    for (auto *block : source->basic_blocks()) {
        auto *cloned_block = shadow->create_basic_block();
        for (auto *metadata : block->metadata_list()) {
            cloned_block->metadata_list().push_front(
                metadata->clone());
        }
        resolver.map(block, cloned_block);
        blocks.emplace_back(BlockClone{
            .source = block,
            .target = cloned_block});
    }
    shadow->set_body_block(static_cast<BasicBlock *>(
        resolver.resolve(source->body_block())));

    auto remaining_instruction_count = size_t{0u};
    for (auto &block : blocks) {
        for (auto *instruction :
             block.source->instructions()) {
            block.instructions.emplace_back(instruction);
            ++remaining_instruction_count;
        }
    }

    // Clone in a dependency-respecting fixed point while preserving the
    // instruction order within every block. Preflight rejects Phi nodes, so a
    // verifier-valid SSA graph is acyclic and must make progress.
    XIRBuilder builder;
    while (remaining_instruction_count != 0u) {
        auto progressed = false;
        for (auto &block : blocks) {
            if (block.next_instruction ==
                block.instructions.size()) {
                continue;
            }
            auto *instruction =
                block.instructions[block.next_instruction];
            auto operands_ready = true;
            for (auto *use : instruction->operand_uses()) {
                auto *operand = use->value();
                if (operand != nullptr &&
                    resolver.resolve(operand) == nullptr) {
                    operands_ready = false;
                    break;
                }
            }
            if (!operands_ready) { continue; }
            builder.set_insertion_point(block.target);
            auto *cloned_instruction =
                instruction->clone_with_metadata(
                    builder, resolver);
            if (cloned_instruction == nullptr) {
                return false;
            }
            resolver.map(instruction, cloned_instruction);
            ++block.next_instruction;
            --remaining_instruction_count;
            progressed = true;
        }
        if (!progressed) { return false; }
    }
    return true;
}

void discard_shadow_definitions(
    luisa::span<ShadowDefinition> shadows) noexcept {
    for (auto &entry : shadows) {
        if (entry.shadow != nullptr &&
            entry.shadow->is_linked()) {
            auto removed = entry.shadow->remove_self();
            static_cast<void>(removed);
        }
        entry.shadow = nullptr;
    }
}

[[nodiscard]] luisa::unordered_set<Constant *>
snapshot_constants(Module *module) noexcept {
    luisa::unordered_set<Constant *> constants;
    for (auto *constant : module->constant_list()) {
        constants.emplace(constant);
    }
    return constants;
}

void rollback_new_constants(
    Module *module,
    const luisa::unordered_set<Constant *> &snapshot) noexcept {
    luisa::vector<Constant *> created;
    for (auto *constant : module->constant_list()) {
        if (!snapshot.contains(constant)) {
            created.emplace_back(constant);
        }
    }
    for (auto *constant : created) {
        LUISA_ASSERT(
            module->remove_constant_if_unused(constant),
            "Failed to roll back a constant created by "
            "restructure_cfg.");
    }
}

void clear_committed_change_counts(
    RestructureCFGInfo &info) noexcept {
    info.restructured_loop_count = 0u;
    info.restructured_if_count = 0u;
    info.restructured_switch_count = 0u;
    info.canonicalized_cfg_count = 0u;
}

[[nodiscard]] RestructureCFGInfo
restructure_cfg_on_definition_in_place(
    FunctionDefinition *def,
    const RestructureCFGOptions &options,
    bool verify_intermediate) noexcept {
    ScopedTimer _timer_overall("restructure_cfg_on_definition");
    trace_cfg("input", def);
    auto info = preflight_restructure_cfg(
        def, verify_intermediate);
    info.definition_transform_invocation_count = 1u;
    if (info.invalid_construct_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "restructure_cfg rejected {} Phi node(s), malformed construct(s), "
            "or unterminated block(s); run reg2mem for Phi input. The function "
            "was left unchanged.",
            info.invalid_construct_count);
        return info;
    }
    if (info.irreducible_region_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "restructure_cfg rejected {} irreducible multi-entry cyclic region(s); "
            "the function was left unchanged.",
            info.irreducible_region_count);
        return info;
    }
    luisa::unordered_set<BasicBlock *> all_created_structural_merges;
    luisa::unordered_map<BasicBlock *, BasicBlock *> sm_to_header;
    luisa::unordered_set<BasicBlock *> exit_dispatch_headers;
    // This is provenance, not a role marker. A generated raw dispatch may
    // later become an IfInst and leave exit_dispatch_headers, but it is still
    // safe to fold only because both of its arms came from the exit-state
    // protocol. Keeping the sets separate prevents this cleanup from ever
    // rewriting an equivalent-looking user IfInst.
    luisa::unordered_set<BasicBlock *>
        generated_exit_dispatch_headers;
    // Recover native multi-way selection boundaries before generic loop/if
    // structurization. Otherwise those passes can mistake an indexed branch's
    // case subgraph for an ordinary cross-edge region and clone through it.
    restructure_indexed_branches(def, info);
    bool main_last_modified = false;
    for (size_t iteration = 0u;
         iteration < options.main_iteration_limit;
         ++iteration) {
        ScopedTimer _timer_main_iter("main_loop_iteration");
        if (restructure_trace_enabled()) {
            LUISA_VERBOSE_WITH_LOCATION(
                "[restructure_cfg] main iteration {}.",
                iteration);
        }
        auto dom = compute_dom_tree(def, false);
        auto pdom = compute_post_dom(def);
        if (try_restructure_loop(def, dom, pdom, info)) {
            main_last_modified = true;
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
            if (!has_cbr) {
                main_last_modified = false;
                break;
            }
            continue;
        }
        if (try_restructure_if_batch(def, dom, pdom, info, all_created_structural_merges, sm_to_header)) {
            main_last_modified = true;
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
            if (!has_cbr) {
                main_last_modified = false;
                break;
            }
            continue;
        }
        main_last_modified = false;
        break;
    }
    if (main_last_modified) { ++info.iteration_limit_count; }
    enforce_unique_construct_entries(def, info);
    if (split_switch_cases(def)) {
        ++info.canonicalized_cfg_count;
    }

    // Post-restructure fixed-point: each phase drains its independent
    // candidates before returning. This budget therefore guards only cycles
    // caused by interactions between phases, not the number of legal sites.
    bool post_last_modified = false;
    {
        ScopedTimer _timer_post("post_restructure_fixed_point");
        auto dom = compute_dom_tree(def, false);
        auto pdom = compute_post_dom(def);
        for (size_t iteration = 0u;
             iteration < options.post_iteration_limit;
             ++iteration) {
            ScopedTimer _timer_post_iter("post_restructure_iteration");
            auto stats_before = restructure_trace_enabled() ?
                                    trace_stats(def) :
                                    CFGTraceStats{};
            bool local = false;
            auto loop_changed =
                try_restructure_loop(def, dom, pdom, info);
            if (loop_changed) {
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            auto header_changed =
                add_headers_to_remaining_divergent(
                    def, dom, pdom, info,
                    exit_dispatch_headers);
            if (header_changed) {
                local = true;
                // dom/pdom are recomputed after every rewrite in the drained phase.
            }
            auto switch_proxy_changed =
                proxy_switch_targets_to_structural_boundaries(def);
            if (switch_proxy_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            auto limits_before_selection_exits = info.iteration_limit_count;
            auto selection_exit_changed =
                drain_selection_exits(
                    def, dom, pdom, info,
                    exit_dispatch_headers);
            if (selection_exit_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
            }
            auto selection_reentry_changed =
                split_exit_dispatch_selection_reentries(
                    def, dom, pdom, info,
                    exit_dispatch_headers);
            if (selection_reentry_changed) {
                local = true;
            }
            for (auto *header : exit_dispatch_headers) {
                generated_exit_dispatch_headers.emplace(
                    header);
            }
            if (info.iteration_limit_count != limits_before_selection_exits) {
                post_last_modified = false;
                break;
            }
            auto boundary_merge_changed =
                canonicalize_loop_boundary_selection_merges(def);
            if (boundary_merge_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            auto boundary_branch_changed =
                normalize_loop_boundary_conditional_branches(
                    def, exit_dispatch_headers,
                    generated_exit_dispatch_headers);
            if (boundary_branch_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            auto loop_prepare_changed =
                canonicalize_loop_prepare_blocks(def);
            if (loop_prepare_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            auto loop_continue_changed =
                normalize_structured_loop_continues(def);
            if (loop_continue_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            auto loop_update_changed =
                canonicalize_loop_update_blocks(def);
            if (loop_update_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            auto limits_before_fixup = info.iteration_limit_count;
            auto construct_exit_changed =
                fixup_construct_exits(
                    def, dom, pdom, info,
                    exit_dispatch_headers);
            if (construct_exit_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            for (auto *header : exit_dispatch_headers) {
                generated_exit_dispatch_headers.emplace(
                    header);
            }
            auto dispatch_collapse_changed =
                collapse_redundant_exit_dispatches(
                    def,
                    generated_exit_dispatch_headers);
            if (dispatch_collapse_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_dom_tree(def, false);
                pdom = compute_post_dom(def);
            }
            if (restructure_trace_enabled()) {
                auto stats_after = trace_stats(def);
                LUISA_VERBOSE_WITH_LOCATION(
                    "[restructure_cfg] post iteration {}: "
                    "blocks {} -> {}, instructions {} -> {}; "
                    "loop={}, header={}, switch_proxy={}, "
                    "selection_exit={}, selection_reentry={}, "
                    "boundary_merge={}, boundary_branch={}, "
                    "loop_prepare={}, loop_continue={}, loop_update={}, "
                    "construct_exit={}, dispatch_collapse={}.",
                    iteration,
                    stats_before.block_count,
                    stats_after.block_count,
                    stats_before.instruction_count,
                    stats_after.instruction_count,
                    loop_changed,
                    header_changed,
                    switch_proxy_changed,
                    selection_exit_changed,
                    selection_reentry_changed,
                    boundary_merge_changed,
                    boundary_branch_changed,
                    loop_prepare_changed,
                    loop_continue_changed,
                    loop_update_changed,
                    construct_exit_changed,
                    dispatch_collapse_changed);
            }
            if (info.iteration_limit_count != limits_before_fixup) {
                post_last_modified = false;
                break;
            }
            post_last_modified = local;
            if (!local) { break; }
        }
    }
    if (post_last_modified) { ++info.iteration_limit_count; }
    if (split_shared_simple_loop_continues(def)) {
        ++info.canonicalized_cfg_count;
    }
    {
        ScopedTimer _timer_unstructured(
            "post_count_unstructured_branches");
        info.unstructured_branch_count =
            count_unstructured_conditional_branches(def);
    }
    {
        ScopedTimer _timer_constructs(
            "post_count_invalid_structured_constructs");
        info.invalid_construct_count =
            count_invalid_structured_constructs(def);
    }
    auto selection_reentry_count = size_t{0u};
    {
        ScopedTimer _timer_reentries(
            "post_count_selection_reentries");
        selection_reentry_count =
            count_post_merge_selection_reentries(def);
        info.invalid_construct_count += selection_reentry_count;
    }
    if (restructure_trace_enabled() &&
        selection_reentry_count != 0u) {
        auto stats = trace_stats(def);
        LUISA_VERBOSE_WITH_LOCATION(
            "[restructure_cfg] detected {} post-merge selection "
            "re-entry construct(s).",
            selection_reentry_count);
        if (stats.block_count <= 128u &&
            stats.instruction_count <= 4096u) {
            luisa::string dump;
            XIRDebugPrinter printer;
            printer.emit_function(dump, def);
            LUISA_VERBOSE_WITH_LOCATION(
                "[restructure_cfg] failing function dump:\n{}",
                dump);
        }
    }
    if (verify_intermediate &&
        info.unstructured_branch_count == 0u &&
        info.invalid_construct_count == 0u) {
        ScopedTimer _timer_verify("post_verify_function");
        ++info.intermediate_verifier_count;
        auto verification = xir_verify_function(
            static_cast<Function *>(def),
            {.require_no_phi = true,
             .require_unique_merge_blocks = true,
             .require_canonical_break_continue_targets = true});
        if (!verification.succeeded()) {
            LUISA_WARNING_WITH_LOCATION(
                "restructure_cfg output verifier rejected the result: {}",
                verification.errors.front().message);
            ++info.invalid_construct_count;
        }
    }
    if (info.iteration_limit_count != 0u) {
        ScopedTimer _timer_entries(
            "post_count_unauthorized_construct_entries");
        info.invalid_construct_count += count_unauthorized_construct_entries(def);
    }
    if (!info.succeeded()) {
        auto raw_conditional_count = size_t{0u};
        auto raw_indexed_count = size_t{0u};
        for (auto *block : def->basic_blocks()) {
            if (block == nullptr || !block->is_terminated()) { continue; }
            raw_conditional_count +=
                block->terminator()->isa<ConditionalBranchInst>() ? 1u : 0u;
            raw_indexed_count +=
                block->terminator()->isa<IndexedBranchInst>() ? 1u : 0u;
        }
        LUISA_WARNING_WITH_LOCATION(
            "restructure_cfg incomplete: {} unstructured branch(es) "
            "({} conditional, {} indexed), {} invalid construct(s), "
            "{} exhausted iteration budget(s), {} irreducible region(s).",
            info.unstructured_branch_count, raw_conditional_count,
            raw_indexed_count, info.invalid_construct_count,
            info.iteration_limit_count, info.irreducible_region_count);
    }
    return info;
}

}// namespace

RestructureCFGInfo restructure_cfg_pass_run_on_function(
    Function *function,
    const RestructureCFGOptions &options) noexcept {
    if (function == nullptr) { return {}; }
    auto *def = function->definition();
    if (def == nullptr) { return {}; }
    if (def->body_block() == nullptr) {
        // Declaration-like callables have no CFG to restructure. Kernels
        // cannot be declarations, so a bodyless kernel remains malformed.
        RestructureCFGInfo info;
        info.invalid_construct_count =
            function->derived_function_tag() ==
                    DerivedFunctionTag::CALLABLE ?
                0u :
                1u;
        return info;
    }
    const auto verify_intermediate =
        restructure_verify_intermediate_enabled();

    // The public pass contract has one complete input verifier boundary.
    // Structural preconditions below are transform-specific analyses, not
    // replacements for this general XIR validity check.
    auto preflight = RestructureCFGInfo{};
    ++preflight.boundary_verifier_count;
    XIRVerificationResult input_verification;
    {
        ScopedTimer _timer_verify(
            "pass_input_verify_function");
        input_verification =
            xir_verify_function(function);
    }
    if (!input_verification.succeeded()) {
        LUISA_WARNING_WITH_LOCATION(
            "restructure_cfg input verifier rejected the function: {}",
            input_verification.errors.front().message);
        preflight.unstructured_branch_count =
            count_unstructured_conditional_branches(def);
        ++preflight.invalid_construct_count;
        return preflight;
    }
    preflight = preflight_restructure_cfg(
        def, verify_intermediate);
    preflight.boundary_verifier_count = 1u;
    if (!preflight.succeeded()) { return preflight; }

    if (options.mutation_mode ==
        RestructureCFGMutationMode::IN_PLACE_DISCARDABLE) {
        // The caller has declared the input disposable on failure, so run the
        // mutating engine once on the original definition. The same complete
        // boundary verification contract still applies on success.
        auto info = restructure_cfg_on_definition_in_place(
            def, options, verify_intermediate);
        info.boundary_verifier_count = 1u;
        info.intermediate_verifier_count +=
            preflight.intermediate_verifier_count;
        if (info.succeeded()) {
            XIRVerificationResult output_verification;
            {
                ScopedTimer _timer_verify(
                    "pass_output_verify_function");
                output_verification = xir_verify_function(
                    function,
                    {.require_no_phi = true,
                     .require_unique_merge_blocks = true,
                     .require_canonical_break_continue_targets = true});
            }
            ++info.boundary_verifier_count;
            if (!output_verification.succeeded()) {
                LUISA_WARNING_WITH_LOCATION(
                    "restructure_cfg output verifier rejected the function: {}",
                    output_verification.errors.front().message);
                ++info.invalid_construct_count;
            }
        }
        return info;
    }

    auto *module = def->parent_module();
    auto constant_snapshot = snapshot_constants(module);
    ShadowDefinition shadow;
    if (!clone_definition_for_transaction(def, shadow)) {
        luisa::vector shadows{std::move(shadow)};
        discard_shadow_definitions(shadows);
        rollback_new_constants(module, constant_snapshot);
        ++preflight.invalid_construct_count;
        return preflight;
    }

    auto info = restructure_cfg_on_definition_in_place(
        shadow.shadow, options,
        verify_intermediate);
    auto intermediate_verifier_count =
        preflight.intermediate_verifier_count +
        info.intermediate_verifier_count;
    if (info.succeeded()) {
        // Verify the complete candidate output once while it still lives in
        // the shadow definition. A successful result is invariant under the
        // graph-isomorphic replay onto the original definition, so late
        // rejection remains atomic without re-verifying every replay step.
        XIRVerificationResult output_verification;
        {
            ScopedTimer _timer_verify(
                "pass_output_verify_function");
            output_verification = xir_verify_function(
                shadow.shadow,
                {.require_no_phi = true,
                 .require_unique_merge_blocks = true,
                 .require_canonical_break_continue_targets = true});
        }
        info.boundary_verifier_count = 2u;
        if (!output_verification.succeeded()) {
            LUISA_WARNING_WITH_LOCATION(
                "restructure_cfg output verifier rejected the function: {}",
                output_verification.errors.front().message);
            ++info.invalid_construct_count;
        }
    }
    luisa::vector shadows{std::move(shadow)};
    discard_shadow_definitions(shadows);
    rollback_new_constants(module, constant_snapshot);
    if (!info.succeeded()) {
        clear_committed_change_counts(info);
        info.boundary_verifier_count =
            info.boundary_verifier_count == 0u ?
                1u :
                info.boundary_verifier_count;
        info.intermediate_verifier_count =
            intermediate_verifier_count;
        return info;
    }
    // Replay the graph-isomorphic dry run on the original objects so the
    // ordinary pass identity contract is preserved: existing blocks and
    // instructions that are edited in place remain the same objects. The dry
    // run has already proved that every late check succeeds. A replay failure
    // would mean the transform depends on allocation identity rather than CFG
    // structure, which is an internal correctness error, not a recoverable
    // input rejection.
    auto committed = restructure_cfg_on_definition_in_place(
        def, options, verify_intermediate);
    LUISA_ASSERT(
        committed.succeeded(),
        "restructure_cfg deterministic replay diverged from "
        "its successful transactional dry run.");
    committed.boundary_verifier_count = 2u;
    committed.intermediate_verifier_count +=
        intermediate_verifier_count;
    committed.definition_transform_invocation_count +=
        info.definition_transform_invocation_count;
    return committed;
}

RestructureCFGInfo restructure_cfg_pass_run_on_module(
    Module *module, PassReport *report,
    const RestructureCFGOptions &options) noexcept {
    ScopedTimer _timer_module(
        "restructure_cfg_pass_run_on_module");
    RestructureCFGInfo total{};
    auto set_report = [&](const RestructureCFGInfo &info) noexcept {
        if (report == nullptr) { return; }
        report->set("restructured_loop", info.restructured_loop_count);
        report->set("restructured_if", info.restructured_if_count);
        report->set(
            "restructured_switch", info.restructured_switch_count);
        report->set("canonicalized_cfg", info.canonicalized_cfg_count);
        report->set(
            "construct_entry_dom_tree",
            info.construct_entry_dom_tree_count);
        report->set(
            "if_batch_post_dom_rebuild",
            info.if_batch_post_dom_rebuild_count);
        report->set(
            "definition_transform_invocation",
            info.definition_transform_invocation_count);
        report->set(
            "boundary_verifier",
            info.boundary_verifier_count);
        report->set(
            "intermediate_verifier",
            info.intermediate_verifier_count);
        report->set(
            "selection_exit_boundary_analysis",
            info.selection_exit_boundary_analysis_count);
        report->set(
            "selection_exit_site_query",
            info.selection_exit_site_query_count);
        report->set(
            "selection_exit_enclosing_loop_query",
            info.selection_exit_enclosing_loop_query_count);
        report->set(
            "selection_reentry_boundary_analysis",
            info.selection_reentry_boundary_analysis_count);
        report->set(
            "selection_reentry_edge_query",
            info.selection_reentry_edge_query_count);
        report->set(
            "selection_reentry_owner_query",
            info.selection_reentry_owner_query_count);
        report->set(
            "irreducible_region", info.irreducible_region_count);
        report->set(
            "unstructured_branch", info.unstructured_branch_count);
        report->set(
            "invalid_construct", info.invalid_construct_count);
        report->set("iteration_limit", info.iteration_limit_count);
    };
    if (module == nullptr) {
        set_report(total);
        return total;
    }
    const auto verify_intermediate =
        restructure_verify_intermediate_enabled();
    auto accumulate = [](
                          RestructureCFGInfo &dst,
                          const RestructureCFGInfo &src) noexcept {
        dst.restructured_loop_count +=
            src.restructured_loop_count;
        dst.restructured_if_count +=
            src.restructured_if_count;
        dst.restructured_switch_count +=
            src.restructured_switch_count;
        dst.canonicalized_cfg_count +=
            src.canonicalized_cfg_count;
        dst.construct_entry_dom_tree_count +=
            src.construct_entry_dom_tree_count;
        dst.if_batch_post_dom_rebuild_count +=
            src.if_batch_post_dom_rebuild_count;
        dst.definition_transform_invocation_count +=
            src.definition_transform_invocation_count;
        dst.boundary_verifier_count +=
            src.boundary_verifier_count;
        dst.intermediate_verifier_count +=
            src.intermediate_verifier_count;
        dst.selection_exit_boundary_analysis_count +=
            src.selection_exit_boundary_analysis_count;
        dst.selection_exit_site_query_count +=
            src.selection_exit_site_query_count;
        dst.selection_exit_enclosing_loop_query_count +=
            src.selection_exit_enclosing_loop_query_count;
        dst.selection_reentry_boundary_analysis_count +=
            src.selection_reentry_boundary_analysis_count;
        dst.selection_reentry_edge_query_count +=
            src.selection_reentry_edge_query_count;
        dst.selection_reentry_owner_query_count +=
            src.selection_reentry_owner_query_count;
        dst.irreducible_region_count +=
            src.irreducible_region_count;
        dst.unstructured_branch_count +=
            src.unstructured_branch_count;
        dst.invalid_construct_count +=
            src.invalid_construct_count;
        dst.iteration_limit_count +=
            src.iteration_limit_count;
    };

    // The complete input domain consists of every definition with a CFG.
    // Declaration-like callables are outside a CFG transform's domain, while
    // a bodyless kernel remains malformed and must reach the verifier.
    luisa::vector<const Function *> input_functions;
    luisa::vector<FunctionDefinition *> definitions;
    for (auto *function : module->function_list()) {
        auto *def = function->definition();
        if (def == nullptr) { continue; }
        if (def->body_block() == nullptr &&
            function->derived_function_tag() ==
                DerivedFunctionTag::CALLABLE) {
            continue;
        }
        input_functions.emplace_back(function);
        if (def->body_block() != nullptr) {
            definitions.emplace_back(def);
        }
    }

    // Verify that complete transform domain once before any shadow definition
    // or transform-owned constant is created.
    ++total.boundary_verifier_count;
    XIRVerificationResult input_verification;
    {
        ScopedTimer _timer_verify(
            "pass_input_verify_module");
        input_verification =
            xir_verify_functions(input_functions);
    }
    if (!input_verification.succeeded()) {
        LUISA_WARNING_WITH_LOCATION(
            "restructure_cfg input verifier rejected the module: {}",
            input_verification.errors.front().message);
        ++total.invalid_construct_count;
        set_report(total);
        return total;
    }

    {
        ScopedTimer _timer_preflight(
            "module_transaction_preflight");
        for (auto definition_index = size_t{0u};
             definition_index < definitions.size();
             ++definition_index) {
            auto *def = definitions[definition_index];
            trace_module_definition(
                "preflight", definition_index, def);
            trace_cfg("module preflight input", def);
            auto info = preflight_restructure_cfg(
                def, verify_intermediate);
            total.intermediate_verifier_count +=
                info.intermediate_verifier_count;
            total.irreducible_region_count +=
                info.irreducible_region_count;
            total.unstructured_branch_count +=
                info.unstructured_branch_count;
            total.invalid_construct_count +=
                info.invalid_construct_count;
        }
    }
    // A module invocation is a single transaction. A malformed/Phi-bearing
    // function or an irreducible SCC in any function rejects all functions
    // before the first canonicalization split or structured node is created.
    if (!total.succeeded()) {
        set_report(total);
        return total;
    }

    if (options.mutation_mode ==
        RestructureCFGMutationMode::IN_PLACE_DISCARDABLE) {
        // This module is exclusively owned and will be discarded by the
        // caller on failure. Preserve the one-input/one-output verifier
        // boundary contract while avoiding the shadow/replay double transform.
        auto preflight_intermediate_verifier_count =
            total.intermediate_verifier_count;
        total = {};
        total.boundary_verifier_count = 1u;
        total.intermediate_verifier_count =
            preflight_intermediate_verifier_count;
        for (auto definition_index = size_t{0u};
             definition_index < definitions.size();
             ++definition_index) {
            auto *def = definitions[definition_index];
            trace_module_definition(
                "in-place transform", definition_index, def);
            auto info = restructure_cfg_on_definition_in_place(
                def, options, verify_intermediate);
            accumulate(total, info);
            if (!info.succeeded()) { break; }
        }
        if (total.succeeded()) {
            luisa::vector<const Function *> candidate_outputs;
            candidate_outputs.reserve(definitions.size());
            for (auto *def : definitions) {
                candidate_outputs.emplace_back(
                    static_cast<const Function *>(def));
            }
            XIRVerificationResult output_verification;
            {
                ScopedTimer _timer_verify(
                    "pass_output_verify_module");
                output_verification = xir_verify_functions(
                    candidate_outputs,
                    {.require_no_phi = true,
                     .require_unique_merge_blocks = true,
                     .require_canonical_break_continue_targets = true});
            }
            ++total.boundary_verifier_count;
            if (!output_verification.succeeded()) {
                LUISA_WARNING_WITH_LOCATION(
                    "restructure_cfg output verifier rejected the module: {}",
                    output_verification.errors.front().message);
                ++total.invalid_construct_count;
            }
        }
        set_report(total);
        return total;
    }

    auto constant_snapshot = snapshot_constants(module);
    luisa::vector<ShadowDefinition> shadows;
    shadows.reserve(definitions.size());
    for (auto *def : definitions) {
        ScopedTimer _timer_clone(
            "module_transaction_clone_definition");
        ShadowDefinition shadow;
        if (!clone_definition_for_transaction(def, shadow)) {
            shadows.emplace_back(std::move(shadow));
            discard_shadow_definitions(shadows);
            rollback_new_constants(module, constant_snapshot);
            auto boundary_verifier_count =
                total.boundary_verifier_count;
            auto intermediate_verifier_count =
                total.intermediate_verifier_count;
            total = {};
            total.boundary_verifier_count =
                boundary_verifier_count;
            total.intermediate_verifier_count =
                intermediate_verifier_count;
            ++total.invalid_construct_count;
            set_report(total);
            return total;
        }
        shadows.emplace_back(std::move(shadow));
    }

    auto preflight_intermediate_verifier_count =
        total.intermediate_verifier_count;
    total = {};
    total.boundary_verifier_count = 1u;
    total.intermediate_verifier_count =
        preflight_intermediate_verifier_count;
    for (auto definition_index = size_t{0u};
         definition_index < shadows.size();
         ++definition_index) {
        auto &shadow = shadows[definition_index];
        trace_module_definition(
            "transactional dry run", definition_index,
            shadow.shadow);
        auto info = restructure_cfg_on_definition_in_place(
            shadow.shadow, options,
            verify_intermediate);
        accumulate(total, info);
        if (!info.succeeded()) { break; }
    }
    if (!total.succeeded()) {
        discard_shadow_definitions(shadows);
        rollback_new_constants(module, constant_snapshot);
        clear_committed_change_counts(total);
        set_report(total);
        return total;
    }

    // All candidate outputs are checked together by one verifier instance.
    // The committed replay is graph-isomorphic to these shadow definitions;
    // therefore this certificate transfers to the replay while preserving
    // rollback on a late verifier failure.
    luisa::vector<const Function *> candidate_outputs;
    candidate_outputs.reserve(shadows.size());
    for (auto &shadow : shadows) {
        candidate_outputs.emplace_back(shadow.shadow);
    }
    XIRVerificationResult output_verification;
    {
        ScopedTimer _timer_verify(
            "pass_output_verify_module");
        output_verification = xir_verify_functions(
            candidate_outputs,
            {.require_no_phi = true,
             .require_unique_merge_blocks = true,
             .require_canonical_break_continue_targets = true});
    }
    ++total.boundary_verifier_count;
    if (!output_verification.succeeded()) {
        LUISA_WARNING_WITH_LOCATION(
            "restructure_cfg output verifier rejected the module: {}",
            output_verification.errors.front().message);
        ++total.invalid_construct_count;
        discard_shadow_definitions(shadows);
        rollback_new_constants(module, constant_snapshot);
        clear_committed_change_counts(total);
        set_report(total);
        return total;
    }

    auto dry_run_intermediate_verifier_count =
        total.intermediate_verifier_count;
    auto dry_run_transform_invocation_count =
        total.definition_transform_invocation_count;
    discard_shadow_definitions(shadows);
    rollback_new_constants(module, constant_snapshot);

    total = {};
    total.boundary_verifier_count = 2u;
    total.intermediate_verifier_count =
        dry_run_intermediate_verifier_count;
    for (auto definition_index = size_t{0u};
         definition_index < definitions.size();
         ++definition_index) {
        auto *def = definitions[definition_index];
        trace_module_definition(
            "transactional replay", definition_index, def);
        auto info = restructure_cfg_on_definition_in_place(
            def, options, verify_intermediate);
        LUISA_ASSERT(
            info.succeeded(),
            "restructure_cfg module replay diverged from its "
            "successful transactional dry run.");
        accumulate(total, info);
    }
    total.definition_transform_invocation_count +=
        dry_run_transform_invocation_count;
    set_report(total);
    return total;
}

}// namespace luisa::compute::xir
