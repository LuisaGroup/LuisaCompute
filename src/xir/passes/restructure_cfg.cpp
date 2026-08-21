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
#include "irreducible_cfg_analysis.h"
#include "restructure_cfg_loop_boundary.h"
#include "restructure_cfg_post_dom.h"
#include "restructure_cfg_selection_merge.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <set>

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

void trace_preflight_result(
    size_t index,
    const RestructureCFGInfo &info) noexcept {
    if (!restructure_trace_enabled() || info.succeeded()) { return; }
    LUISA_VERBOSE_WITH_LOCATION(
        "[restructure_cfg] preflight definition {} rejected: "
        "irreducible={}, unstructured={}, invalid={}, iteration_limit={}.",
        index, info.irreducible_region_count,
        info.unstructured_branch_count,
        info.invalid_construct_count,
        info.iteration_limit_count);
}

// Return whether recursive SCC decomposition finds a cyclic region with more
// than one entry block. Such a region cannot be represented by XIR's structured
// loop form without state dispatch or node splitting. Detect it before the
// restructuring pipeline mutates anything so failure is atomic and callers can
// choose the dedicated irreducible-CFG lowering.
[[nodiscard]] size_t count_irreducible_regions(FunctionDefinition *def) noexcept {
    auto analysis =
        detail::analyze_cfg_strongly_connected_components(def);
    if (!analysis.irreducible_regions.empty() &&
        restructure_trace_enabled()) {
        for (auto region_index = size_t{0u};
             region_index < analysis.irreducible_regions.size();
             ++region_index) {
            auto &&region =
                analysis.irreducible_regions[region_index];
            luisa::vector<uint8_t> in_region(
                analysis.blocks.size(), 0u);
            for (auto node : region.nodes) {
                in_region[node] = 1u;
            }
            LUISA_VERBOSE_WITH_LOCATION(
                "[restructure_cfg] irreducible region {}: "
                "blocks={}, entries={}.",
                region_index, region.nodes.size(),
                region.entry_nodes.size());
            XIRDebugPrinter printer;
            for (auto node : region.entry_nodes) {
                auto external_predecessor_count = size_t{0u};
                for (auto predecessor : analysis.predecessors[node]) {
                    external_predecessor_count +=
                        in_region[predecessor] == 0u ?
                            1u :
                            0u;
                }
                auto *block = analysis.blocks[node];
                auto terminator_tag = block->is_terminated() ?
                                          to_string(block->terminator()->derived_instruction_tag()) :
                                          luisa::string_view{"unterminated"};
                LUISA_VERBOSE_WITH_LOCATION(
                    "[restructure_cfg] irreducible entry block {}: "
                    "external_predecessors={}, terminator={}",
                    node, external_predecessor_count, terminator_tag);
                luisa::string terminator_dump;
                if (block->is_terminated()) {
                    printer.emit_instruction(
                        terminator_dump, block->terminator());
                    LUISA_VERBOSE_WITH_LOCATION(
                        "[restructure_cfg] irreducible entry block {} "
                        "terminator IR: {}",
                        node, terminator_dump);
                }
                for (auto predecessor : analysis.predecessors[node]) {
                    if (in_region[predecessor] != 0u) {
                        continue;
                    }
                    auto *predecessor_block =
                        analysis.blocks[predecessor];
                    auto predecessor_tag = predecessor_block->is_terminated() ?
                                               to_string(predecessor_block->terminator()->derived_instruction_tag()) :
                                               luisa::string_view{"unterminated"};
                    LUISA_VERBOSE_WITH_LOCATION(
                        "[restructure_cfg] irreducible external edge "
                        "{} -> {} (source terminator={}).",
                        predecessor, node, predecessor_tag);
                    luisa::string predecessor_dump;
                    if (predecessor_block->is_terminated()) {
                        printer.emit_instruction(
                            predecessor_dump,
                            predecessor_block->terminator());
                        LUISA_VERBOSE_WITH_LOCATION(
                            "[restructure_cfg] irreducible external "
                            "source block {} terminator IR: {}",
                            predecessor, predecessor_dump);
                    }
                }
            }
        }
    }
    return analysis.irreducible_region_count();
}

[[nodiscard]] bool is_sink(BasicBlock *block) noexcept {
    return detail::is_restructure_cfg_sink(block);
}

using PostDomInfo = detail::RestructurePostDomInfo;

// Restructuring consumes dominance ancestry on every CFG version, whereas
// only post-merge selection re-entry consumes dominance frontiers. Keep that
// derived relation demand-driven instead of rebuilding it after every edit.
[[nodiscard]] DomTree compute_restructure_dom(
    FunctionDefinition *def) noexcept {
    return compute_dom_tree(
        static_cast<Function *>(def),
        {.compute_dominance_frontiers = false});
}

[[nodiscard]] PostDomInfo compute_post_dom(
    FunctionDefinition *def,
    RestructureCFGInfo &info) noexcept {
    ScopedTimer _timer_pdom("compute_post_dom");
    detail::RestructurePostDomStats stats;
    auto result = detail::compute_restructure_post_dom(
        def, &stats);
    ++info.postdom_analysis_count;
    info.postdom_numbered_block_count +=
        stats.numbered_block_count;
    info.postdom_numbered_edge_count +=
        stats.numbered_edge_count;
    info.postdom_active_block_count +=
        stats.active_block_count;
    info.postdom_fixed_point_iteration_count +=
        stats.fixed_point_iteration_count;
    info.postdom_fixed_point_block_visit_count +=
        stats.fixed_point_block_visit_count;
    info.postdom_fixed_point_edge_visit_count +=
        stats.fixed_point_edge_visit_count;
    info.postdom_intersect_step_count +=
        stats.intersect_step_count;
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

template<typename Dominance>
[[nodiscard]] luisa::unordered_set<BasicBlock *>
collect_enclosing_loop_exits(FunctionDefinition *def,
                             BasicBlock *header,
                             const Dominance &dom) noexcept;
[[nodiscard]] BasicBlock *
structured_statement_merge(Instruction *term) noexcept;
[[nodiscard]] BasicBlock *
canonical_exit_target(BasicBlock *target) noexcept;

// Global post-dominance loses a selection's lexical merge when an arm exits an
// enclosing loop or terminates the function. Recover the nearest normal-path
// convergence instead by ignoring enclosing loop boundaries and comparing
// shortest reachability from each distinct arm.
template<typename Dominance>
[[nodiscard]] BasicBlock *infer_selection_merge(
    FunctionDefinition *def,
    BasicBlock *header,
    luisa::span<BasicBlock *const> entries,
    const Dominance &dom) noexcept {
    if (def == nullptr || header == nullptr || entries.empty()) {
        return nullptr;
    }
    // The dominator tree is rooted at the executable function entry. An owned
    // but unreachable structural shell deliberately has no node in that tree;
    // its raw selection is rebuilt with a synthetic merge by the caller.
    if (!dom.contains(header)) { return nullptr; }
    auto boundaries =
        collect_enclosing_loop_exits(def, header, dom);
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
    for (auto *candidate : def->basic_blocks()) {
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

[[nodiscard]] Constant *create_indexed_branch_case_constant(
    Module *module, const Type *selector_type,
    IndexedBranchTerminatorInstruction::case_value_type bits) noexcept {
    if (module == nullptr || selector_type == nullptr) { return nullptr; }
    bits = IndexedBranchTerminatorInstruction::canonicalize_case_value(
        selector_type, bits);
    switch (selector_type->tag()) {
        case Type::Tag::BOOL: {
            auto value = bits != 0u;
            return module->create_constant(selector_type, &value);
        }
        case Type::Tag::INT8: {
            auto value = luisa::bit_cast<int8_t>(
                static_cast<uint8_t>(bits));
            return module->create_constant(selector_type, &value);
        }
        case Type::Tag::UINT8: {
            auto value = static_cast<uint8_t>(bits);
            return module->create_constant(selector_type, &value);
        }
        case Type::Tag::INT16: {
            auto value = luisa::bit_cast<int16_t>(
                static_cast<uint16_t>(bits));
            return module->create_constant(selector_type, &value);
        }
        case Type::Tag::UINT16: {
            auto value = static_cast<uint16_t>(bits);
            return module->create_constant(selector_type, &value);
        }
        case Type::Tag::INT32: {
            auto value = luisa::bit_cast<int32_t>(
                static_cast<uint32_t>(bits));
            return module->create_constant(selector_type, &value);
        }
        case Type::Tag::UINT32: {
            auto value = static_cast<uint32_t>(bits);
            return module->create_constant(selector_type, &value);
        }
        case Type::Tag::INT64: {
            auto value = luisa::bit_cast<int64_t>(bits);
            return module->create_constant(selector_type, &value);
        }
        case Type::Tag::UINT64:
            return module->create_constant(selector_type, &bits);
        default: return nullptr;
    }
}

[[nodiscard]] bool indexed_branch_has_back_edge(
    IndexedBranchInst *indexed_branch,
    const DomTree &dom) noexcept {
    if (indexed_branch == nullptr) { return false; }
    auto *header = indexed_branch->parent_block();
    if (header == nullptr || !dom.contains(header)) { return false; }
    auto is_back_target = [&](BasicBlock *target) noexcept {
        return target != nullptr && dom.contains(target) &&
               dom.dominates(target, header);
    };
    if (is_back_target(indexed_branch->default_block())) {
        return true;
    }
    for (auto i = size_t{0u};
         i < indexed_branch->case_count(); ++i) {
        if (is_back_target(indexed_branch->case_block(i))) {
            return true;
        }
    }
    return false;
}

// A natural-loop back edge is an edge whose target dominates its source.
// Loop recovery consumes BranchInst and ConditionalBranchInst edges, whereas
// treating a cyclic IndexedBranchInst as a selection first turns that back
// edge into an illegal construct entry and makes node splitting clone the
// whole loop body. Lower only cyclic indexed branches to an ordered equality
// chain before selection recovery. This exposes every back edge to the single
// loop-recovery algorithm while preserving native SwitchInst for acyclic
// multi-way control flow. A zero-case branch is unconditionally canonicalized
// to BranchInst because its selector has no effect.
[[nodiscard]] bool lower_cyclic_indexed_branches(
    FunctionDefinition *def) noexcept {
    auto modified = false;
    for (;;) {
        auto dom = compute_restructure_dom(def);
        IndexedBranchInst *candidate = nullptr;
        for (auto *block : def->basic_blocks()) {
            if (block == nullptr || !block->is_terminated() ||
                !block->terminator()->isa<IndexedBranchInst>()) {
                continue;
            }
            auto *indexed_branch = static_cast<IndexedBranchInst *>(
                block->terminator());
            if (indexed_branch->case_count() == 0u ||
                indexed_branch_has_back_edge(indexed_branch, dom)) {
                candidate = indexed_branch;
                break;
            }
        }
        if (candidate == nullptr) { break; }

        using Case = std::pair<
            IndexedBranchTerminatorInstruction::case_value_type,
            BasicBlock *>;
        auto *header = candidate->parent_block();
        auto *selector = candidate->value();
        auto *default_block = candidate->default_block();
        luisa::vector<Case> cases;
        cases.reserve(candidate->case_count());
        for (auto i = size_t{0u}; i < candidate->case_count(); ++i) {
            // A case targeting the default is semantically redundant: case
            // labels are unique after selector-width canonicalization, so no
            // later case can match the same selector value.
            if (auto *target = candidate->case_block(i);
                target != default_block) {
                cases.emplace_back(candidate->case_value(i), target);
            }
        }

        auto removed = candidate->remove_self();
        Instruction *replacement_terminator = nullptr;
        XIRBuilder builder;
        if (cases.empty()) {
            builder.set_insertion_point(header);
            replacement_terminator = builder.br(default_block);
        } else {
            auto *test_block = header;
            for (auto i = size_t{0u}; i < cases.size(); ++i) {
                auto [case_value, case_block] = cases[i];
                auto *case_constant = create_indexed_branch_case_constant(
                    def->parent_module(), selector->type(), case_value);
                LUISA_ASSERT(
                    case_constant != nullptr,
                    "Verified indexed branch has an unsupported selector type.");
                auto *next_block = i + 1u == cases.size() ?
                                       default_block :
                                       def->create_basic_block();
                builder.set_insertion_point(test_block);
                auto *condition = builder.call(
                    Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                    {selector, case_constant});
                auto *conditional = builder.cond_br(
                    condition, case_block, next_block);
                if (i == 0u) {
                    replacement_terminator = conditional;
                }
                test_block = next_block;
            }
        }
        LUISA_ASSERT(
            replacement_terminator != nullptr,
            "Failed to lower cyclic indexed branch.");
        for (auto *metadata : removed->metadata_list()) {
            replacement_terminator->metadata_list().push_front(
                metadata->clone());
        }
        modified = true;
    }
    return modified;
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
        auto dom = compute_restructure_dom(def);
        auto pdom = compute_post_dom(def, info);
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
    return detail::canonical_trivial_branch_chain_target(bb);
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
    // The loop merge is a structural identity, even when it is itself an
    // empty forwarding block. A fully contracted branch-chain target walks
    // through that identity and would make the already-canonical path
    //
    //     arm ->* loop.merge -> continuation
    //
    // appear to target `continuation`. Test reachability to the declared
    // boundary before taking the ordinary forwarding quotient. This is also
    // the canonical XIR spelling for a loop exit nested in a Switch, where a
    // BreakInst would denote the nearer Switch scope and the edge must remain
    // an ordinary BranchInst to the loop merge.
    if (trivial_branch_chain_reaches(target, merge)) { return true; }
    auto *resolved = trivial_branch_chain_target(target);
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
            if (!add_kind(LoopBoundaryTargetKind::BREAK)) {
                return LoopBoundaryTargetKind::NONE;
            }
            continue;
        }
        if (bb == continue_target || bb == loop_entry) {
            if (!add_kind(LoopBoundaryTargetKind::CONTINUE)) {
                return LoopBoundaryTargetKind::NONE;
            }
            continue;
        }
        if (!bb->is_terminated()) {
            return LoopBoundaryTargetKind::NONE;
        }
        auto *term = bb->terminator();
        if (term->isa<BreakInst>()) {
            auto *br = static_cast<BreakInst *>(term);
            if (br->target_block() != merge) {
                return LoopBoundaryTargetKind::NONE;
            }
            if (!add_kind(LoopBoundaryTargetKind::BREAK)) {
                return LoopBoundaryTargetKind::NONE;
            }
            continue;
        }
        if (term->isa<ContinueInst>()) {
            auto *cont = static_cast<ContinueInst *>(term);
            auto *cont_target = cont->target_block();
            if (cont_target != continue_target &&
                cont_target != loop_entry) {
                return LoopBoundaryTargetKind::NONE;
            }
            if (!add_kind(LoopBoundaryTargetKind::CONTINUE)) {
                return LoopBoundaryTargetKind::NONE;
            }
            continue;
        }
        if (term->isa<ReturnInst>() ||
            term->isa<UnreachableInst>() ||
            term->isa<RasterDiscardInst>()) {
            return LoopBoundaryTargetKind::NONE;
        }
        if (term->isa<LoopInst>() ||
            term->isa<SimpleLoopInst>()) {
            auto *control_flow_merge = term->control_flow_merge();
            if (control_flow_merge == nullptr ||
                control_flow_merge->merge_block() == nullptr) {
                return LoopBoundaryTargetKind::NONE;
            }
            work.emplace_back(control_flow_merge->merge_block());
            continue;
        }
        traverse_structured_successors(
            bb, [&](BasicBlock *succ) noexcept {
                if (succ != nullptr) { work.emplace_back(succ); }
            });
    }
    return kind;
}

// A semantic path may eventually reach a loop boundary after executing
// arbitrary work. That fact alone does not make the first edge a physical
// break/continue edge: SPIR-V may omit a selection merge only when the arm is
// an exclusive, side-effect-free forwarding chain to that boundary.
//
// The proof is local and exact. Every proper forwarding block must contain
// only its terminator and have the immediately preceding chain block as its
// sole predecessor. `selection_merge` is an optional convergence barrier: an
// arm may start at the declared merge, but the sibling arm may not pass
// through it and still claim an independent boundary edge.
[[nodiscard]] LoopBoundaryTargetKind
classify_exclusive_loop_boundary_arm(
    BasicBlock *branch_header,
    BasicBlock *entry,
    BasicBlock *selection_merge,
    BasicBlock *continue_target,
    BasicBlock *loop_entry,
    BasicBlock *merge) noexcept {
    if (branch_header == nullptr || entry == nullptr ||
        continue_target == nullptr || loop_entry == nullptr ||
        merge == nullptr) {
        return LoopBoundaryTargetKind::NONE;
    }
    auto *expected_predecessor = branch_header;
    auto *block = entry;
    luisa::unordered_set<BasicBlock *> visited;
    while (block != nullptr && visited.emplace(block).second) {
        if (block == continue_target || block == loop_entry) {
            return LoopBoundaryTargetKind::CONTINUE;
        }
        if (block == merge) {
            return LoopBoundaryTargetKind::BREAK;
        }
        if ((block == selection_merge && block != entry) ||
            !has_only_terminator(block)) {
            return LoopBoundaryTargetKind::NONE;
        }
        auto predecessor_count = size_t{0u};
        auto has_unexpected_predecessor = false;
        block->traverse_predecessors(
            false, [&](BasicBlock *predecessor) noexcept {
                ++predecessor_count;
                has_unexpected_predecessor |=
                    predecessor != expected_predecessor;
            });
        if (predecessor_count != 1u ||
            has_unexpected_predecessor) {
            return LoopBoundaryTargetKind::NONE;
        }
        auto *terminator = block->terminator();
        if (terminator->isa<BreakInst>()) {
            return static_cast<BreakInst *>(terminator)
                               ->target_block() == merge ?
                       LoopBoundaryTargetKind::BREAK :
                       LoopBoundaryTargetKind::NONE;
        }
        if (terminator->isa<ContinueInst>()) {
            auto *target =
                static_cast<ContinueInst *>(terminator)
                    ->target_block();
            return target == continue_target ||
                           target == loop_entry ?
                       LoopBoundaryTargetKind::CONTINUE :
                       LoopBoundaryTargetKind::NONE;
        }
        if (!terminator->isa<BranchInst>()) {
            return LoopBoundaryTargetKind::NONE;
        }
        expected_predecessor = block;
        block = static_cast<BranchInst *>(terminator)
                    ->target_block();
    }
    return LoopBoundaryTargetKind::NONE;
}

[[nodiscard]] bool normalize_one_loop_boundary_conditional_branch(FunctionDefinition *def,
                                                                  luisa::unordered_set<BasicBlock *> &
                                                                      exit_dispatch_headers,
                                                                  const luisa::unordered_set<BasicBlock *> &
                                                                      generated_exit_dispatch_headers) noexcept {
    auto dom = compute_restructure_dom(def);
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
        LoopBoundaryTargetKind true_kind{
            LoopBoundaryTargetKind::NONE};
        LoopBoundaryTargetKind false_kind{
            LoopBoundaryTargetKind::NONE};
        bool generated_boundary_guard{false};
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
                // Terminal dataflow and physical edge shape are different
                // facts. A generated state-dispatch arm may perform ordinary
                // work before eventually continuing the loop; only the exact
                // forwarding-chain proof permits boundary lowering.
                auto physical_true_kind =
                    classify_exclusive_loop_boundary_arm(
                        branch_block, t, nullptr,
                        site.continue_target, site.entry,
                        site.merge);
                auto physical_false_kind =
                    classify_exclusive_loop_boundary_arm(
                        branch_block, f, nullptr,
                        site.continue_target, site.entry,
                        site.merge);
                auto physical_one_sided_boundary =
                    singular_boundary(physical_true_kind) !=
                    singular_boundary(physical_false_kind);
                auto physical_opposing_boundaries =
                    (physical_true_kind ==
                         LoopBoundaryTargetKind::BREAK &&
                     physical_false_kind ==
                         LoopBoundaryTargetKind::CONTINUE) ||
                    (physical_true_kind ==
                         LoopBoundaryTargetKind::CONTINUE &&
                     physical_false_kind ==
                         LoopBoundaryTargetKind::BREAK);
                auto generated_dispatch =
                    generated_exit_dispatch_headers.contains(
                        branch_block);
                auto generated_boundary_guard =
                    generated_dispatch &&
                    (physical_one_sided_boundary ||
                     physical_opposing_boundaries);
                if (generated_boundary_guard ||
                    (!generated_dispatch &&
                     (opposing ||
                      (allow_one_sided_boundary &&
                       one_sided_boundary)))) {
                    candidates.emplace_back(Candidate{
                        branch_block, t, f, site.entry,
                        site.continue_target, site.merge,
                        site.selection_merge,
                        cbr->condition(),
                        generated_boundary_guard ?
                            physical_true_kind :
                            true_kind,
                        generated_boundary_guard ?
                            physical_false_kind :
                            false_kind,
                        generated_boundary_guard,
                        site.depth});
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
        // The body root belongs to the loop region even when it is also the
        // loop entry, which is exactly the SimpleLoop representation. Only a
        // successor edge back to the entry is a region boundary. Applying the
        // entry exclusion to the root made every SimpleLoop region empty and
        // disagreed with LoopContinueBatchAnalysis about loop membership.
        auto enqueue = [&](BasicBlock *block,
                           bool is_body_root = false) noexcept {
            if (block == nullptr || block == site.merge ||
                (block == site.entry && !is_body_root)) {
                return;
            }
            if (visited.emplace(block).second) {
                work.emplace_back(block);
            }
        };
        enqueue(site.body, true);
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
    auto true_kind = cand.true_kind;
    auto false_kind = cand.false_kind;

    auto opposing =
        (true_kind == LoopBoundaryTargetKind::BREAK &&
         false_kind == LoopBoundaryTargetKind::CONTINUE) ||
        (true_kind == LoopBoundaryTargetKind::CONTINUE &&
         false_kind == LoopBoundaryTargetKind::BREAK);
    auto one_sided_boundary =
        singular_boundary(true_kind) !=
        singular_boundary(false_kind);
    auto generated_boundary_guard =
        cand.generated_boundary_guard;
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
// before mem2reg: quotient each generated dispatch arm by empty unconditional
// forwarding blocks, then compare its terminal effect. Equal Branch(T),
// Break(M), or Continue(U) effects make the selector semantically dead. The
// selected state already lives in the alloca/store protocol, so replacing the
// conditional by that exact effect is semantics-preserving and strictly
// reduces the number of generated conditionals. In particular, it prevents
// later structural phases from repeatedly wrapping two distinct Break(M)
// proxy blocks in equivalent SPIR-V selections.
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
    enum class ExitEffect : uint8_t {
        BLOCK,
        BREAK,
        CONTINUE,
    };
    struct CanonicalExitEffect {
        ExitEffect effect{ExitEffect::BLOCK};
        BasicBlock *target{nullptr};
    };
    // The exit-state selector is semantically dead when both of its arms have
    // the same effect in the quotient CFG formed by removing empty forwarding
    // blocks. Break and Continue are effects rather than ordinary successors:
    // two distinct proxy blocks ending in Break(M), for example, are the same
    // exit even though their block identities differ. Keeping the effect kind
    // in the key prevents the unsound identification of Branch(M), Break(M),
    // and Continue(M), which have different structural roles.
    auto canonical_exit_effect = [](BasicBlock *target) noexcept {
        auto *terminal = trivial_branch_chain_target(target);
        auto result = CanonicalExitEffect{
            .effect = ExitEffect::BLOCK,
            .target = terminal};
        if (!has_only_terminator(terminal)) { return result; }
        auto *terminator = terminal->terminator();
        if (terminator->isa<BreakInst>()) {
            result.effect = ExitEffect::BREAK;
            result.target = static_cast<BreakInst *>(terminator)
                                ->target_block();
        } else if (terminator->isa<ContinueInst>()) {
            result.effect = ExitEffect::CONTINUE;
            result.target =
                static_cast<ContinueInst *>(terminator)
                    ->target_block();
        }
        return result;
    };
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
        auto true_exit =
            canonical_exit_effect(branch->true_block());
        auto false_exit =
            canonical_exit_effect(branch->false_block());
        if (true_exit.target == nullptr ||
            true_exit.effect != false_exit.effect ||
            true_exit.target != false_exit.target) {
            continue;
        }
        auto *condition = branch->condition();
        auto old_term = term->remove_self();
        XIRBuilder builder;
        builder.set_insertion_point(header);
        TerminatorInstruction *replacement = nullptr;
        switch (true_exit.effect) {
            case ExitEffect::BLOCK:
                replacement = builder.br(true_exit.target);
                break;
            case ExitEffect::BREAK:
                replacement = builder.break_(true_exit.target);
                break;
            case ExitEffect::CONTINUE:
                replacement = builder.continue_(true_exit.target);
                break;
        }
        for (auto *metadata : old_term->metadata_list()) {
            replacement->metadata_list().push_front(
                metadata->clone());
        }
        dead_roots.emplace_back(condition);
        if (restructure_trace_enabled()) {
            LUISA_VERBOSE_WITH_LOCATION(
                "[restructure_cfg] dispatch-collapse rewrite: "
                "header={}, effect={}, target={}.",
                static_cast<void *>(header),
                static_cast<uint32_t>(true_exit.effect),
                static_cast<void *>(true_exit.target));
        }
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

[[nodiscard]] bool normalize_structured_loop_continues(
    FunctionDefinition *def,
    DomTree &dom,
    RestructureCFGInfo &info) noexcept {
    ScopedTimer _timer_normalize_structured_loop_continues(
        "normalize_structured_loop_continues");
    struct LoopSite {
        BasicBlock *entry{nullptr};
        BasicBlock *body{nullptr};
        BasicBlock *continue_target{nullptr};
        BasicBlock *merge{nullptr};
    };
    bool changed = false;
    Clock site_collection_clock;
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
    auto site_collection_ms = site_collection_clock.toc();
    Clock ownership_clock;
    LoopContinueBatchAnalysis batch_analysis{def, dom};
    auto ownership_ms = ownership_clock.toc();
    auto dominance_rebuild_ms = 0.0;
    Clock region_discovery_clock;
    ++info.loop_continue_analysis_count;
    for (auto site_index = size_t{0u};
         site_index < loops.size(); ++site_index) {
        auto site = loops[site_index];
        ++info.loop_continue_site_query_count;
        batch_analysis.plan(
            site_index, site.entry, site.body,
            site.continue_target, site.merge);
    }
    auto region_discovery_ms = region_discovery_clock.toc();
    auto &batch_stats = batch_analysis.stats();
    info.loop_continue_region_block_visit_count +=
        batch_stats.region_block_visit_count;
    info.loop_continue_region_edge_visit_count +=
        batch_stats.region_edge_visit_count;
    info.loop_continue_planned_rewrite_count +=
        batch_analysis.rewrite_count();

    // Every query above observes the same immutable CFG and dominance
    // version. Each action carries the original (source, old target)
    // precondition; the mutation helper revalidates it immediately before
    // applying the rewrite. An earlier action can therefore only make a later
    // action fail closed, never make stale analysis authorize a different
    // edge. The outer restructure fixed point analyzes any rejected
    // opportunity on the next exact CFG version.
    Clock rewrite_scan_clock;
    luisa::vector<uint8_t> mutated_sites(loops.size(), 0u);
    for (auto rewrite_index = size_t{0u};
         rewrite_index < batch_analysis.rewrite_count();
         ++rewrite_index) {
        auto &rewrite = batch_analysis.rewrite(rewrite_index);
        auto rewrite_changed = false;
        switch (rewrite.kind) {
            case LoopContinueRewriteKind::BREAK:
                rewrite_changed = retarget_edges_to_break(
                    def, rewrite.block, rewrite.from,
                    rewrite.target);
                break;
            case LoopContinueRewriteKind::CONTINUE:
                rewrite_changed = retarget_edges_to_continue(
                    def, rewrite.block, rewrite.from,
                    rewrite.target);
                break;
        }
        if (!rewrite_changed) { continue; }
        changed = true;
        ++info.loop_continue_applied_rewrite_count;
        LUISA_DEBUG_ASSERT(
            rewrite.site_index < mutated_sites.size(),
            "Loop-continue rewrite site index is out of bounds.");
        if (rewrite.site_index < mutated_sites.size()) {
            mutated_sites[rewrite.site_index] = 1u;
        }
    }
    auto rewrite_scan_ms = rewrite_scan_clock.toc();
    for (auto mutated : mutated_sites) {
        info.loop_continue_invalidation_count +=
            mutated != 0u ? 1u : 0u;
    }
    if (changed) {
        // No analysis query is issued after mutation. Build the one exact
        // ancestry tree retained by the caller. Its dominance-frontier
        // derivative is intentionally left absent until the unique
        // selection-reentry consumer asks for it.
        Clock dominance_rebuild_clock;
        DomTreeBuildStats dominance_stats;
        dom = compute_dom_tree(
            def, {.compute_dominance_frontiers = false},
            &dominance_stats);
        dominance_rebuild_ms += dominance_rebuild_clock.toc();
        ++info.loop_continue_dominance_rebuild_count;
        info.loop_continue_dom_numbered_block_count +=
            dominance_stats.numbered_block_count;
        info.loop_continue_dom_numbered_edge_count +=
            dominance_stats.numbered_edge_count;
        info.loop_continue_dom_fixed_point_iteration_count +=
            dominance_stats.fixed_point_iteration_count;
        info.loop_continue_dom_fixed_point_block_visit_count +=
            dominance_stats.fixed_point_block_visit_count;
        info.loop_continue_dom_fixed_point_edge_visit_count +=
            dominance_stats.fixed_point_edge_visit_count;
        info.loop_continue_dom_intersect_step_count +=
            dominance_stats.intersect_step_count;
    }
    if (restructure_trace_enabled()) {
        LUISA_VERBOSE_WITH_LOCATION(
            "[restructure_cfg] loop_continue_site_collection: {:.3f} ms",
            site_collection_ms);
        LUISA_VERBOSE_WITH_LOCATION(
            "[restructure_cfg] loop_continue_ownership_analysis: {:.3f} ms",
            ownership_ms);
        LUISA_VERBOSE_WITH_LOCATION(
            "[restructure_cfg] loop_continue_region_discovery: {:.3f} ms",
            region_discovery_ms);
        LUISA_VERBOSE_WITH_LOCATION(
            "[restructure_cfg] loop_continue_rewrite_scan: {:.3f} ms",
            rewrite_scan_ms);
        LUISA_VERBOSE_WITH_LOCATION(
            "[restructure_cfg] loop_continue_dominance_rebuild: {:.3f} ms",
            dominance_rebuild_ms);
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

// A semantic path may eventually reach an enclosing loop boundary after
// executing arbitrary work. That does not make its conditional a physical
// loop-boundary guard: SPIR-V may omit OpSelectionMerge only when the boundary
// arm itself is an exclusive, side-effect-free forwarding chain that can be
// collapsed to the loop break/continue edge. The one-sided declared-merge case
// therefore uses the same exclusivity proof as ControlFlowPlan; the two-sided
// case is accepted only for complementary physical break/continue arms.
[[nodiscard]] bool is_physically_lowerable_loop_boundary_if(
    IfInst *if_inst,
    BasicBlock *continue_target,
    BasicBlock *loop_entry,
    BasicBlock *merge) noexcept {
    if (if_inst == nullptr || continue_target == nullptr ||
        loop_entry == nullptr || merge == nullptr) {
        return false;
    }
    auto *selection_merge = if_inst->merge_block();
    auto true_is_merge = if_inst->true_block() == selection_merge;
    auto false_is_merge = if_inst->false_block() == selection_merge;
    auto true_kind =
        classify_exclusive_loop_boundary_arm(
            if_inst->parent_block(), if_inst->true_block(),
            selection_merge, continue_target, loop_entry,
            merge);
    auto false_kind =
        classify_exclusive_loop_boundary_arm(
            if_inst->parent_block(), if_inst->false_block(),
            selection_merge, continue_target, loop_entry,
            merge);
    auto opposing_boundaries =
        (true_kind == LoopBoundaryTargetKind::BREAK &&
         false_kind == LoopBoundaryTargetKind::CONTINUE) ||
        (true_kind == LoopBoundaryTargetKind::CONTINUE &&
         false_kind == LoopBoundaryTargetKind::BREAK);
    auto singular_boundary = [](auto kind) noexcept {
        return kind == LoopBoundaryTargetKind::BREAK ||
               kind == LoopBoundaryTargetKind::CONTINUE;
    };
    return opposing_boundaries ||
           (true_is_merge && singular_boundary(false_kind)) ||
           (false_is_merge && singular_boundary(true_kind));
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
    FunctionDefinition *def,
    RestructureCFGInfo *info = nullptr) noexcept {
    luisa::unordered_set<BasicBlock *> entries;
    if (def == nullptr) { return entries; }
    LoopBoundaryPathDataflow dataflow{def};
    def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        if (block == nullptr || !block->is_terminated()) {
            return;
        }
        auto *term = block->terminator();
        BasicBlock *body = nullptr;
        BasicBlock *continue_target = nullptr;
        BasicBlock *loop_entry = nullptr;
        BasicBlock *merge = nullptr;
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            body = loop->body_block();
            continue_target = loop->update_block();
            if (continue_target == nullptr) {
                continue_target = loop->prepare_block();
            }
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
        if (body == nullptr || continue_target == nullptr ||
            loop_entry == nullptr || merge == nullptr) {
            return;
        }
        dataflow.evaluate(
            body, continue_target, loop_entry, merge);
        if (info != nullptr) {
            ++info->selection_exit_boundary_dataflow_count;
            info->selection_exit_boundary_block_visit_count +=
                dataflow.active_block_count();
            info->selection_exit_boundary_edge_visit_count +=
                dataflow.edge_visit_count();
        }
        for (auto index = size_t{0u};
             index < dataflow.region_size(); ++index) {
            auto *candidate = dataflow.region_block(index);
            if (!candidate->is_terminated() ||
                !candidate->terminator()->isa<IfInst>()) {
                continue;
            }
            auto *if_inst = static_cast<IfInst *>(
                candidate->terminator());
            if (info != nullptr) {
                info->selection_exit_boundary_classification_count +=
                    2u;
            }
            if (is_physically_lowerable_loop_boundary_if(
                    if_inst, continue_target, loop_entry, merge)) {
                entries.emplace(candidate);
            }
        }
    });
    return entries;
}

// Entry uniqueness is a property of physical structured constructs. A
// loop-boundary IfInst is retained in XIR so generic transforms still see a
// structured conditional, but SPIR-V lowers it as the loop's physical
// break/continue branch without OpSelectionMerge. Consequently it has no
// selection interior whose entries need node splitting.
[[nodiscard]] bool requires_unique_construct_entries(
    BasicBlock *header,
    const luisa::unordered_set<BasicBlock *> &
        loop_boundary_selection_entries) noexcept {
    if (header == nullptr || !header->is_terminated()) {
        return false;
    }
    auto *term = header->terminator();
    if (term->isa<IfInst>()) {
        return !loop_boundary_selection_entries.contains(header);
    }
    return term->isa<SwitchInst>() ||
           term->isa<LoopInst>() ||
           term->isa<SimpleLoopInst>();
}

[[nodiscard]] bool canonicalize_loop_boundary_selection_merges(
    FunctionDefinition *def,
    RestructureCFGInfo &info) noexcept {
    ScopedTimer _timer_canonicalize_loop_boundary_selection_merges(
        "canonicalize_loop_boundary_selection_merges");
    struct LoopSite {
        BasicBlock *body = nullptr;
        BasicBlock *continue_target = nullptr;
        BasicBlock *loop_entry = nullptr;
        BasicBlock *merge = nullptr;
    };
    luisa::vector<LoopSite> loops;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        auto site = LoopSite{};
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            site.body = loop->body_block();
            site.continue_target = loop->update_block();
            if (site.continue_target == nullptr) {
                site.continue_target = loop->prepare_block();
            }
            site.loop_entry = loop->prepare_block();
            site.merge = loop->merge_block();
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            site.body = loop->body_block();
            site.continue_target = loop->body_block();
            site.loop_entry = loop->body_block();
            site.merge = loop->merge_block();
        } else {
            return;
        }
        if (site.body != nullptr &&
            site.continue_target != nullptr &&
            site.loop_entry != nullptr &&
            site.merge != nullptr) {
            loops.emplace_back(site);
        }
    });
    if (loops.empty()) { return false; }

    struct Rewrite {
        IfInst *if_inst{nullptr};
        BasicBlock *true_target{nullptr};
        BasicBlock *false_target{nullptr};
        BasicBlock *selection_merge{nullptr};
        BasicBlock *loop_merge{nullptr};
        bool true_break_proxy{false};
        bool false_break_proxy{false};
        bool merge_proxy{false};
        bool merge_proxy_on_true{false};
    };
    auto apply_rewrites = [&](luisa::span<const Rewrite> rewrites) noexcept {
        XIRBuilder builder;
        for (auto &&rewrite : rewrites) {
            auto *new_true = rewrite.true_target;
            auto *new_false = rewrite.false_target;
            if (rewrite.true_break_proxy) {
                new_true = def->create_basic_block();
                builder.set_insertion_point(new_true);
                builder.break_(rewrite.loop_merge);
            }
            if (rewrite.false_break_proxy) {
                new_false = def->create_basic_block();
                builder.set_insertion_point(new_false);
                builder.break_(rewrite.loop_merge);
            }
            if (new_true != rewrite.true_target) {
                rewrite.if_inst->set_true_target(new_true);
            }
            if (new_false != rewrite.false_target) {
                rewrite.if_inst->set_false_target(new_false);
            }
            if (rewrite.merge_proxy) {
                auto *new_merge = def->create_basic_block();
                builder.set_insertion_point(new_merge);
                // Preserve the identity of the arm which represented the
                // old declared merge. A one-sided boundary If may use either
                // arm as its ordinary fallthrough. Always proxying the false
                // arm turns a true-merge spelling into a different CFG role:
                // the boundary arm becomes the declared merge, so construct
                // repair and boundary canonicalization can undo each other
                // forever. Opposing-boundary Ifs have no old merge arm and
                // deliberately choose false as their canonical proxy side.
                auto *old_merge_arm =
                    rewrite.merge_proxy_on_true ? new_true : new_false;
                builder.br(old_merge_arm);
                if (rewrite.merge_proxy_on_true) {
                    rewrite.if_inst->set_true_target(new_merge);
                } else {
                    rewrite.if_inst->set_false_target(new_merge);
                }
                rewrite.if_inst->set_merge_block(new_merge);
            }
        }
    };

    auto changed = false;
    for (;;) {
        // The dataflow and all classifications below describe exactly this
        // immutable CFG version. Applying any loop batch invalidates the
        // block numbering, reverse edges, and terminal facts, so restart from
        // a fresh snapshot before inspecting another loop context.
        LoopBoundaryPathDataflow dataflow{def};
        luisa::unordered_map<BasicBlock *, size_t>
            structured_merge_owner_counts;
        def->traverse_basic_blocks(
            [&](BasicBlock *block) noexcept {
                if (block == nullptr || !block->is_terminated()) {
                    return;
                }
                if (auto *merge = structured_statement_merge(
                        block->terminator());
                    merge != nullptr) {
                    ++structured_merge_owner_counts[merge];
                }
            });
        ++info.boundary_merge_analysis_count;
        auto applied_batch = false;
        for (auto &&loop : loops) {
            dataflow.evaluate(
                loop.body,
                loop.continue_target,
                loop.loop_entry,
                loop.merge);
            ++info.boundary_merge_dataflow_count;
            luisa::vector<Rewrite> rewrites;
            for (auto index = size_t{0u};
                 index < dataflow.region_size(); ++index) {
                auto *cur = dataflow.region_block(index);
                if (!cur->is_terminated() ||
                    !cur->terminator()->isa<IfInst>()) {
                    continue;
                }
                auto *if_inst = static_cast<IfInst *>(
                    cur->terminator());
                auto true_kind = dataflow.classify(
                    if_inst->true_block());
                auto false_kind = dataflow.classify(
                    if_inst->false_block());
                info.boundary_merge_classification_count += 2u;
                auto needs_break_proxy = [merge = loop.merge](
                                             BasicBlock *target,
                                             LoopBoundaryTargetKind kind) noexcept {
                    return kind == LoopBoundaryTargetKind::BREAK &&
                           is_loop_break_target(target, merge) &&
                           !is_canonical_loop_break_path(target, merge);
                };
                auto rewrite = Rewrite{
                    .if_inst = if_inst,
                    .true_target = if_inst->true_block(),
                    .false_target = if_inst->false_block(),
                    .selection_merge = if_inst->merge_block(),
                    .loop_merge = loop.merge,
                    .true_break_proxy = needs_break_proxy(
                        if_inst->true_block(), true_kind),
                    .false_break_proxy = needs_break_proxy(
                        if_inst->false_block(), false_kind)};

                // Break proxies preserve the arm's BREAK fact but replace
                // its target identity. Evaluate the selection predicate
                // on those prospective identities, matching the original
                // arm-first rewrite order without mutating the snapshot.
                auto true_is_selection_merge =
                    !rewrite.true_break_proxy &&
                    rewrite.true_target == rewrite.selection_merge;
                auto false_is_selection_merge =
                    !rewrite.false_break_proxy &&
                    rewrite.false_target == rewrite.selection_merge;
                auto singular_boundary = [](auto kind) noexcept {
                    return kind == LoopBoundaryTargetKind::BREAK ||
                           kind == LoopBoundaryTargetKind::CONTINUE;
                };
                auto boundary_if =
                    (true_kind == LoopBoundaryTargetKind::CONTINUE &&
                     false_kind == LoopBoundaryTargetKind::BREAK) ||
                    (true_kind == LoopBoundaryTargetKind::BREAK &&
                     false_kind == LoopBoundaryTargetKind::CONTINUE) ||
                    (true_is_selection_merge &&
                     singular_boundary(false_kind)) ||
                    (false_is_selection_merge &&
                     singular_boundary(true_kind));
                rewrite.merge_proxy =
                    boundary_if &&
                    ((!true_is_selection_merge &&
                      !false_is_selection_merge) ||
                     rewrite.selection_merge == loop.merge ||
                     structured_merge_owner_counts[
                         rewrite.selection_merge] > 1u);
                rewrite.merge_proxy_on_true =
                    true_is_selection_merge;
                if (rewrite.true_break_proxy ||
                    rewrite.false_break_proxy ||
                    rewrite.merge_proxy) {
                    rewrites.emplace_back(rewrite);
                }
            }
            if (rewrites.empty()) { continue; }

            // Within one loop context, a break proxy substitutes the same
            // BREAK fact. A merge proxy removes only its obsolete declared
            // successor while retaining both executable arms. Facts can
            // therefore only stay equal or lose spurious bits; every action
            // proven on this snapshot remains sound. Newly enabled actions
            // are found after the mandatory version restart below.
            apply_rewrites(rewrites);
            ++info.boundary_merge_rewrite_batch_count;
            changed = true;
            applied_batch = true;
            break;
        }
        if (!applied_batch) { break; }
    }
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
    return detail::canonical_trivial_branch_chain_target(target);
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

template<typename Dominance>
[[nodiscard]] luisa::unordered_set<BasicBlock *>
collect_enclosing_loop_exits(
    FunctionDefinition *def,
    BasicBlock *header,
    const Dominance &dom) noexcept {
    luisa::unordered_set<BasicBlock *> exits;
    if (!dom.contains(header)) { return exits; }
    for (auto &&loop :
         collect_structured_loop_exit_info(def)) {
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
    enum struct ContextKind : uint8_t {
        LOOP,
        SWITCH,
    };
    struct Context {
        size_t parent{SIZE_MAX};
        ContextKind kind{ContextKind::LOOP};
        BasicBlock *break_target{nullptr};
        BasicBlock *continue_target{nullptr};
        size_t dom_depth{0u};
    };
    luisa::unordered_set<BasicBlock *>
        loop_boundary_selection_entries;
    luisa::vector<Context> contexts;
    luisa::unordered_map<BasicBlock *, size_t>
        selection_contexts;
    luisa::unordered_map<BasicBlock *, size_t>
        block_contexts;
    // A physical structured merge has one owner. Reusing another construct's
    // merge as the boundary of a selection would satisfy reachability while
    // violating SPIR-V's role-uniqueness constraint.
    luisa::unordered_set<BasicBlock *> structured_merge_blocks;
    // Empty branch proxies are transparent only until the first structured
    // merge/continue role. Crossing such a role would skip a lexical
    // construct boundary rather than contract a representation-only edge.
    luisa::unordered_set<BasicBlock *> structured_exit_boundaries;
};

// Materialize the exact CFG relations consumed by one selection-exit scan.
// No rewrite occurs while a scan evaluates its sites. A successful rewrite
// returns to the caller, which rebuilds dominance and rematerializes these
// relations before observing the new CFG.
[[nodiscard]] SelectionExitCFGRelations
build_selection_exit_cfg_relations(
    FunctionDefinition *def,
    const DomTree &dom,
    RestructureCFGInfo &info) noexcept {
    ScopedTimer _timer_selection_exit_relations(
        "selection_exit_build_relations");
    SelectionExitCFGRelations relations;
    relations.loop_boundary_selection_entries =
        collect_loop_boundary_selection_entries(def, &info);
    struct ContextSite {
        BasicBlock *header{nullptr};
        BasicBlock *merge{nullptr};
        SelectionExitCFGRelations::ContextKind kind{
            SelectionExitCFGRelations::ContextKind::LOOP};
        BasicBlock *break_target{nullptr};
        BasicBlock *continue_target{nullptr};
        size_t dom_depth{0u};
        bool can_be_active{true};
    };
    luisa::vector<ContextSite> sites;
    luisa::unordered_map<BasicBlock *, size_t>
        site_index_by_header;
    luisa::unordered_map<BasicBlock *, luisa::vector<size_t>>
        merge_events;
    def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        if (block == nullptr || !block->is_terminated()) { return; }
        auto *term = block->terminator();
        if (auto *merge = structured_statement_merge(term);
            merge != nullptr) {
            relations.structured_merge_blocks.emplace(merge);
            relations.structured_exit_boundaries.emplace(merge);
        }
        ContextSite site;
        site.header = block;
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            site.merge = loop->merge_block();
            site.break_target = site.merge;
            site.continue_target = loop->update_block();
            if (site.continue_target == nullptr) {
                site.continue_target = loop->prepare_block();
            }
            relations.structured_exit_boundaries.emplace(
                site.continue_target);
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            site.merge = loop->merge_block();
            site.break_target = site.merge;
            site.continue_target = loop->body_block();
            relations.structured_exit_boundaries.emplace(
                site.continue_target);
        } else if (term->isa<SwitchInst>()) {
            site.kind =
                SelectionExitCFGRelations::ContextKind::SWITCH;
            site.merge = static_cast<SwitchInst *>(term)->merge_block();
            site.break_target = site.merge;
        } else {
            return;
        }
        if (!dom.contains(block)) { return; }
        site.dom_depth = dom_depth(dom, block);
        auto site_index = sites.size();
        site_index_by_header.emplace(block, site_index);
        sites.emplace_back(std::move(site));
    });
    relations.contexts.reserve(sites.size());
    luisa::vector<size_t> context_id_by_site(
        sites.size(), SIZE_MAX);
    for (auto site_index = size_t{0u};
         site_index < sites.size(); ++site_index) {
        auto &site = sites[site_index];
        if (site.merge == nullptr ||
            !dom.contains(site.merge)) {
            continue;
        }
        if (dom.dominates(site.merge, site.header)) {
            // A merge that dominates its header cannot delimit a lexical
            // interior in this dominance version. Cyclic Switches are later
            // loop-wrapped; until then they are not enclosing switch scopes.
            site.can_be_active = false;
        } else if (dom.dominates(site.header, site.merge)) {
            merge_events[site.merge].emplace_back(site_index);
        }
    }

    // Dominance is ancestry in `dom`. Carry a persistent linked construct
    // context down that sparse tree. A loop context contributes its
    // merge/continue boundaries. A Switch contributes its merge because
    // SPIR-V permits a selection to exit to the nearest enclosing switch
    // before the nearest enclosing loop. If/selection contexts need no node:
    // they are not legal non-local exit targets, so retaining only loops and
    // switches is the exact quotient needed by the structured-exit rule.
    struct DomWalkFrame {
        const DomTreeNode *node{nullptr};
        size_t next_child{0u};
        size_t activated_context{SIZE_MAX};
        luisa::vector<size_t> suspended_contexts;
    };
    using ActiveContextKey =
        std::pair<size_t, size_t>;// (dom depth, context id)
    std::set<ActiveContextKey> active_contexts;
    luisa::vector<DomWalkFrame> walk;
    if (dom.root() != nullptr) {
        walk.emplace_back(DomWalkFrame{
            .node = dom.root()});
    }
    while (!walk.empty()) {
        auto &frame = walk.back();
        if (frame.next_child == 0u) {
            auto *block = frame.node->block();
            if (auto event = merge_events.find(block);
                event != merge_events.end()) {
                for (auto site_index : event->second) {
                    auto context_id =
                        context_id_by_site[site_index];
                    if (context_id == SIZE_MAX) { continue; }
                    auto key = ActiveContextKey{
                        relations.contexts[context_id].dom_depth,
                        context_id};
                    if (active_contexts.erase(key) != 0u) {
                        frame.suspended_contexts.emplace_back(
                            context_id);
                    }
                }
            }
            auto current_context = active_contexts.empty() ?
                                       SIZE_MAX :
                                       active_contexts.rbegin()->second;
            relations.block_contexts.emplace(
                block, current_context);
            if (block->is_terminated()) {
                auto *term = block->terminator();
                if (term->isa<IfInst>() || term->isa<SwitchInst>()) {
                    relations.selection_contexts.emplace(
                        block, current_context);
                }
            }
            if (auto site_iter = site_index_by_header.find(block);
                site_iter != site_index_by_header.end() &&
                sites[site_iter->second].can_be_active) {
                auto site_index = site_iter->second;
                auto &site = sites[site_index];
                auto context_id = relations.contexts.size();
                relations.contexts.emplace_back(
                    SelectionExitCFGRelations::Context{
                        .parent = current_context,
                        .kind = site.kind,
                        .break_target = site.break_target,
                        .continue_target = site.continue_target,
                        .dom_depth = site.dom_depth});
                context_id_by_site[site_index] = context_id;
                active_contexts.emplace(
                    site.dom_depth, context_id);
                frame.activated_context = context_id;
                if (site.kind ==
                    SelectionExitCFGRelations::ContextKind::LOOP) {
                    ++info.selection_exit_loop_context_count;
                }
            }
        }
        auto children = frame.node->children();
        if (frame.next_child < children.size()) {
            auto *child = children[frame.next_child++];
            walk.emplace_back(DomWalkFrame{
                .node = child});
            continue;
        }
        if (frame.activated_context != SIZE_MAX) {
            auto context_id = frame.activated_context;
            active_contexts.erase(ActiveContextKey{
                relations.contexts[context_id].dom_depth,
                context_id});
        }
        for (auto context_id : frame.suspended_contexts) {
            active_contexts.emplace(
                relations.contexts[context_id].dom_depth,
                context_id);
        }
        walk.pop_back();
    }
    return relations;
}

struct SelectionExitEdge {
    BasicBlock *src;
    BasicBlock *dst;
};

enum class SelectionExitRewriteStatus : uint8_t {
    UNCHANGED,
    MODIFIED,
    STALLED_SITE,
};

struct SelectionExitRewriteResult {
    SelectionExitRewriteStatus status{SelectionExitRewriteStatus::UNCHANGED};
    Instruction *site{nullptr};
    // A one-target rewrite is only a common funnel in the quotient CFG where
    // side-effect-free forwarding blocks are transparent. It preserves
    // reachability and dominance among canonical pre-existing blocks, so only
    // construct-containment and explicitly bypassed-boundary dependencies
    // need invalidation. Multi-target state dispatches conservatively
    // invalidate every site because their CFG expands correlated reachability.
    bool local_dependency_only{false};
    bool requires_ssa_repair{false};
    // Changing only the declared merge preserves the executable CFG and its
    // dominance relations. Funnel/state-dispatch rewrites set this bit and
    // require the versioned analyses to be rebuilt.
    bool cfg_modified{false};
    luisa::vector<BasicBlock *> mutated_edge_sources;
    luisa::vector<BasicBlock *> bypassed_forwarding_blocks;
};

using SelectionExitProgress =
    luisa::unordered_map<Instruction *, size_t>;

void append_unique_exit_edge(luisa::vector<SelectionExitEdge> &edges,
                             BasicBlock *src,
                             BasicBlock *dst) noexcept {
    for (auto edge : edges) {
        if (edge.src == src && edge.dst == dst) { return; }
    }
    edges.emplace_back(SelectionExitEdge{src, dst});
}

struct SelectionExitAnalysis {
    luisa::vector<BasicBlock *> entries;
    luisa::vector<SelectionExitEdge> invalid_exits;
    luisa::vector<SelectionExitEdge> merge_exits;
};

[[nodiscard]] size_t selection_context(
    const SelectionExitCFGRelations &relations,
    BasicBlock *header) noexcept {
    auto iter = relations.selection_contexts.find(header);
    return iter == relations.selection_contexts.end() ?
               SIZE_MAX :
               iter->second;
}

// SPIR-V structured exits for a selection form a lexical prefix, not the set
// of every dominating boundary. Starting at the selection's current context,
// the first enclosing Switch merge is legal; the first enclosing Loop's
// merge/continue targets are legal and terminate the search. More distant
// loops and switches are not direct exits of the selection construct.
[[nodiscard]] bool is_legal_enclosing_selection_exit(
    const SelectionExitCFGRelations &relations,
    size_t selection_context,
    BasicBlock *block,
    bool allow_enclosing_switch) noexcept {
    auto switch_available = allow_enclosing_switch;
    for (auto context = selection_context;
         context != SIZE_MAX;
         context = relations.contexts[context].parent) {
        auto &&node = relations.contexts[context];
        if (node.kind ==
            SelectionExitCFGRelations::ContextKind::LOOP) {
            return block == node.break_target ||
                   block == node.continue_target;
        }
        if (switch_available &&
            block == node.break_target) {
            return true;
        }
        if (node.kind ==
            SelectionExitCFGRelations::ContextKind::SWITCH) {
            switch_available = false;
        }
    }
    return false;
}

// Empty forwarding chains do not introduce another structured activation;
// they are one edge in the quotient CFG used by the SPIR-V exit rules. Keep
// this normalization in the legality predicate so mutation and final audit
// cannot disagree about a physical loop/switch boundary hidden by proxies.
[[nodiscard]] bool is_legal_enclosing_selection_exit_in_quotient(
    const SelectionExitCFGRelations &relations,
    size_t selection_context,
    BasicBlock *block,
    bool allow_enclosing_switch) noexcept {
    if (is_legal_enclosing_selection_exit(
            relations, selection_context, block,
            allow_enclosing_switch)) {
        return true;
    }
    // Contract only representation-only forwarding blocks. A block carrying
    // any structured merge/continue role is an observable lexical boundary;
    // following through it could incorrectly make a farther loop/switch exit
    // look like the nearest legal one.
    luisa::unordered_set<BasicBlock *> visited;
    auto *current = block;
    while (current != nullptr &&
           visited.emplace(current).second &&
           !relations.structured_exit_boundaries.contains(current)) {
        auto *next = trivial_branch_target(current);
        if (next == nullptr) { break; }
        current = next;
        if (is_legal_enclosing_selection_exit(
                relations, selection_context, current,
                allow_enclosing_switch)) {
            return true;
        }
    }
    return false;
}

// BreakInst denotes the nearest lexical break scope in XIR. Inside a Switch,
// however, SPIR-V also permits an ordinary edge to that Switch construct's
// nearest enclosing loop merge/continue boundary. A destructured or imported
// CFG can spell such an edge as BreakInst(loop_merge), which has the right
// executable target but the wrong lexical operator once the Switch is
// reconstructed.
//
// Normalize only this proven case to BranchInst. The executable graph is
// unchanged. A BreakInst that skips an intervening Loop is not accepted, and
// a canonical BreakInst to the current Switch merge is retained.
[[nodiscard]] bool canonicalize_nonlocal_switch_breaks(
    FunctionDefinition *def,
    const SelectionExitCFGRelations &relations) noexcept {
    auto modified = false;
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr || !block->is_terminated() ||
            !block->terminator()->isa<BreakInst>()) {
            continue;
        }
        auto context_iter =
            relations.block_contexts.find(block);
        if (context_iter == relations.block_contexts.end() ||
            context_iter->second == SIZE_MAX) {
            continue;
        }
        auto context = context_iter->second;
        auto &&nearest = relations.contexts[context];
        if (nearest.kind !=
            SelectionExitCFGRelations::ContextKind::SWITCH) {
            continue;
        }
        auto *break_inst =
            static_cast<BreakInst *>(block->terminator());
        auto *target = break_inst->target_block();
        if (target == nearest.break_target ||
            !is_legal_enclosing_selection_exit(
                relations, context, target, false)) {
            continue;
        }
        auto removed = break_inst->remove_self();
        XIRBuilder builder;
        builder.set_insertion_point(block);
        auto *replacement = builder.br(target);
        for (auto *metadata : removed->metadata_list()) {
            replacement->metadata_list().push_front(
                metadata->clone());
        }
        modified = true;
    }
    return modified;
}

// Compute the exact executable exit cut of a structured selection. The cut is
// defined over each arm's header-dominated region with nested structured
// constructs contracted to their merge. An edge is valid precisely when it
// reaches the selection's declared merge; every other edge is a non-local
// exit that must either become the merge itself or be routed through a
// single-exit protocol. Infinite paths contribute no cut edge, which matches
// SPIR-V's structured-control-flow semantics without requiring classical
// post-dominance through non-terminating loops.
[[nodiscard]] SelectionExitAnalysis analyze_selection_exits(
    BasicBlock *header,
    Instruction *term,
    BasicBlock *merge,
    const DomTree &dom,
    const SelectionExitCFGRelations &cfg_relations,
    RestructureCFGInfo &info,
    const luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    SelectionExitAnalysis analysis;
    analysis.entries = selection_entries(term);
    auto enclosing_context =
        selection_context(cfg_relations, header);
    auto allow_enclosing_switch = !term->isa<SwitchInst>();
    auto is_legal_structured_exit =
        [&](BasicBlock *block) noexcept {
            return is_legal_enclosing_selection_exit_in_quotient(
                cfg_relations, enclosing_context, block,
                allow_enclosing_switch);
        };
    luisa::unordered_set<BasicBlock *> region;
    auto entry_is_valid = [&](BasicBlock *entry) noexcept {
        return entry != nullptr && dom.contains(entry) &&
               dom.dominates(header, entry);
    };
    for (auto *entry : analysis.entries) {
        if (entry == merge) {
            append_unique_exit_edge(
                analysis.merge_exits, header, merge);
            continue;
        }
        if (is_legal_structured_exit(entry)) {
            continue;
        }
        if (!entry_is_valid(entry)) {
            if (entry != nullptr) {
                append_unique_exit_edge(
                    analysis.invalid_exits, header, entry);
            }
            continue;
        }
        luisa::vector<BasicBlock *> work{entry};
        while (!work.empty()) {
            auto *bb = work.back();
            work.pop_back();
            if (bb == nullptr || bb == merge) { continue; }
            if (!dom.contains(bb) ||
                !dom.dominates(header, bb) ||
                !region.emplace(bb).second) {
                continue;
            }
            ++info.selection_exit_region_block_visit_count;
            if (!bb->is_terminated()) { continue; }
            auto *nested_merge =
                structured_statement_merge(bb->terminator());
            if (nested_merge != nullptr &&
                !exit_dispatch_headers.contains(bb) &&
                !cfg_relations.loop_boundary_selection_entries
                     .contains(bb)) {
                if (nested_merge == merge) {
                    append_unique_exit_edge(
                        analysis.merge_exits, bb, nested_merge);
                } else {
                    work.emplace_back(nested_merge);
                }
                continue;
            }
            traverse_executable_successors(
                bb, [&](BasicBlock *successor) noexcept {
                    ++info.selection_exit_region_edge_visit_count;
                    if (successor == nullptr) { return; }
                    if (successor == merge) {
                        append_unique_exit_edge(
                            analysis.merge_exits, bb, successor);
                        return;
                    }
                    auto *canonical_successor =
                        canonical_exit_target(successor);
                    if (is_legal_structured_exit(successor)) {
                        // A selection may leave directly through the nearest
                        // enclosing loop/switch boundary. An otherwise empty
                        // branch chain is the same executable edge in the
                        // quotient CFG; later boundary normalization gives it
                        // the required Break/Continue spelling.
                        return;
                    }
                    if (successor == header ||
                        canonical_successor == header) {
                        return;
                    }
                    if (is_sink(successor) ||
                        !dom.contains(successor) ||
                        !dom.dominates(entry, successor)) {
                        append_unique_exit_edge(
                            analysis.invalid_exits, bb,
                            successor);
                        return;
                    }
                    work.emplace_back(successor);
                });
        }
    }
    return analysis;
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
    SelectionExitProgress &
        rewritten_site_invalid_exit_counts,
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
    if (selection_entries(term).empty()) { return {}; }
    ++info.selection_exit_enclosing_loop_query_count;
    auto enclosing_context =
        selection_context(cfg_relations, header);
    auto allow_enclosing_switch = !term->isa<SwitchInst>();
    auto is_legal_structured_exit =
        [&](BasicBlock *block) noexcept {
            return is_legal_enclosing_selection_exit_in_quotient(
                cfg_relations, enclosing_context, block,
                allow_enclosing_switch);
        };
    auto analysis = analyze_selection_exits(
        header, term, merge, dom, cfg_relations,
        info, exit_dispatch_headers);
    auto &entries = analysis.entries;
    auto &invalid_exits = analysis.invalid_exits;
    auto &merge_exits = analysis.merge_exits;
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

    // If the complete exit cut already has one exact target T, synthesizing
    // N -> T is not progress: a later loop recovery may make N a loop header
    // and expose the same cut again. The canonical SESE boundary is T itself.
    // This is a structural fact, not a peephole: every finite path leaving an
    // arm crosses the cut, every cut edge names T, and H dominates T. Infinite
    // paths need not reach a merge. Reassigning only the declarative merge
    // therefore preserves the executable graph and all SSA facts.
    //
    // A block already owned as another structured merge or serving as an
    // enclosing loop boundary cannot acquire a second physical role. Those
    // cases continue through the explicit single-exit protocol below.
    auto *common_direct_exit =
        reroute_edges.empty() ? nullptr : reroute_edges.front().dst;
    auto has_exact_common_exit = common_direct_exit != nullptr;
    for (auto edge : reroute_edges) {
        has_exact_common_exit &= edge.dst == common_direct_exit;
    }
    if (merge_exits.empty() && has_exact_common_exit &&
        common_direct_exit != merge &&
        dom.contains(common_direct_exit) &&
        dom.dominates(header, common_direct_exit) &&
        !is_legal_structured_exit(common_direct_exit) &&
        !cfg_relations.structured_merge_blocks.contains(
            common_direct_exit)) {
        auto *control_flow_merge = term->control_flow_merge();
        LUISA_ASSERT(
            control_flow_merge != nullptr &&
                control_flow_merge->merge_block() == merge,
            "Selection merge changed during one immutable exit-cut analysis.");
        if (restructure_trace_enabled()) {
            LUISA_VERBOSE_WITH_LOCATION(
                "[restructure_cfg] canonical selection merge: "
                "header={}, old={}, new={}, cut_edges={}.",
                static_cast<void *>(header),
                static_cast<void *>(merge),
                static_cast<void *>(common_direct_exit),
                reroute_edges.size());
        }
        control_flow_merge->set_merge_block(common_direct_exit);
        return SelectionExitRewriteResult{
            .status = SelectionExitRewriteStatus::MODIFIED,
            .site = term,
            .local_dependency_only = true,
            .requires_ssa_repair = false,
            .cfg_modified = false};
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
                is_legal_structured_exit(lhs);
            auto rhs_boundary =
                is_legal_structured_exit(rhs);
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
    auto local_dependency_only = targets.size() == 1u;
    luisa::vector<BasicBlock *> mutated_edge_sources;
    luisa::vector<BasicBlock *> bypassed_forwarding_blocks;
    mutated_edge_sources.reserve(normalized_edges.size());
    for (auto edge : normalized_edges) {
        if (std::find(
                mutated_edge_sources.begin(),
                mutated_edge_sources.end(),
                edge.src) == mutated_edge_sources.end()) {
            mutated_edge_sources.emplace_back(edge.src);
        }
        luisa::vector<BasicBlock *> forwarding_path;
        auto *block = edge.dst;
        while (block != nullptr && block != edge.target &&
               std::find(
                   forwarding_path.begin(),
                   forwarding_path.end(),
                   block) == forwarding_path.end()) {
            forwarding_path.emplace_back(block);
            if (std::find(
                    bypassed_forwarding_blocks.begin(),
                    bypassed_forwarding_blocks.end(),
                    block) == bypassed_forwarding_blocks.end()) {
                bypassed_forwarding_blocks.emplace_back(block);
            }
            block = trivial_branch_target(block);
        }
    }
    auto [rewrite_iter, first_rewrite] =
        rewritten_site_invalid_exit_counts.try_emplace(
            term, invalid_exits.size());
    if (!first_rewrite &&
        invalid_exits.size() >= rewrite_iter->second) {
        return {SelectionExitRewriteStatus::STALLED_SITE, term};
    }
    rewrite_iter->second = invalid_exits.size();
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
    return SelectionExitRewriteResult{
        .status = SelectionExitRewriteStatus::MODIFIED,
        .site = term,
        .local_dependency_only = local_dependency_only,
        .requires_ssa_repair = targets.size() > 1u,
        .cfg_modified = true,
        .mutated_edge_sources =
            std::move(mutated_edge_sources),
        .bypassed_forwarding_blocks =
            std::move(bypassed_forwarding_blocks)};
}

struct SelectionExitDrainResult {
    bool modified{false};
    bool cfg_modified{false};
    bool yielded{false};
};

[[nodiscard]] SelectionExitDrainResult drain_selection_exits(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom,
    RestructureCFGInfo &info,
    luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    ScopedTimer _timer_drain_selection_exits(
        "drain_selection_exits");
    // Within one immutable-phase drain, a site may become eligible again after
    // a nested selection rewrite. It may continue only while its non-local
    // exit count strictly decreases, a well-founded natural-number measure.
    // Do not persist pointer-keyed history across other phases: they may
    // legitimately change the site's structural meaning. Termination of the
    // outer pipeline follows from the phases' shared canonical forms.
    SelectionExitProgress rewritten_site_invalid_exit_counts;
    SelectionExitDrainResult drain;
    auto ssa_repair_requested = false;

    struct Site {
        BasicBlock *header{nullptr};
        Instruction *term{nullptr};
    };
    luisa::vector<Site> sites;
    luisa::unordered_map<BasicBlock *, size_t>
        site_index_by_header;
    {
        ScopedTimer _timer_selection_exit_collect_sites(
            "selection_exit_collect_sites");
        def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            if (!block->is_terminated()) { return; }
            auto *term = block->terminator();
            if (!term->isa<IfInst>() &&
                !term->isa<SwitchInst>()) {
                return;
            }
            auto *merge = term->control_flow_merge();
            if (merge == nullptr ||
                merge->merge_block() == nullptr) {
                return;
            }
            site_index_by_header.emplace(
                block, sites.size());
            sites.emplace_back(Site{block, term});
        });
    }
    luisa::vector<uint8_t> dirty(
        sites.size(), uint8_t{1u});
    auto current_merge = [](Site site) noexcept {
        if (site.header == nullptr || site.term == nullptr ||
            !site.header->is_terminated() ||
            site.header->terminator() != site.term) {
            return static_cast<BasicBlock *>(nullptr);
        }
        auto *merge = site.term->control_flow_merge();
        return merge == nullptr ? nullptr : merge->merge_block();
    };
    auto mark_dirty = [&](size_t index) noexcept {
        if (dirty[index] == uint8_t{0u}) {
            ++info.selection_exit_dependency_requery_count;
            dirty[index] = uint8_t{1u};
        }
    };
    // A structured selection (H, M) can observe a changed block X exactly in
    // subtree(H) - subtree(M). This is the same sparse-dominator construct
    // interior used by construct-exit repair. Checking both the old and new
    // trees covers dominance gained or lost by the rewrite.
    auto mark_enclosing_dependencies =
        [&](const DomTree &snapshot,
            BasicBlock *changed_header) noexcept {
            if (changed_header == nullptr ||
                !snapshot.contains(changed_header)) {
                return;
            }
            for (auto index = size_t{0u};
                 index < sites.size(); ++index) {
                auto site = sites[index];
                if (site.header == changed_header) {
                    mark_dirty(index);
                    continue;
                }
                auto *merge = current_merge(site);
                if (merge == nullptr ||
                    !snapshot.contains(site.header) ||
                    !snapshot.dominates(
                        site.header, changed_header)) {
                    continue;
                }
                auto merge_cuts_interior =
                    snapshot.contains(merge) &&
                    snapshot.dominates(site.header, merge) &&
                    snapshot.dominates(
                        merge, changed_header);
                if (!merge_cuts_interior) {
                    mark_dirty(index);
                }
            }
        };

    for (;;) {
        auto cfg_relations =
            build_selection_exit_cfg_relations(
                def, dom, info);
        ++info.selection_exit_boundary_analysis_count;
        drain.modified |=
            canonicalize_nonlocal_switch_breaks(
                def, cfg_relations);
        struct Query {
            size_t site_index{0u};
            size_t depth{0u};
        };
        luisa::vector<Query> queries;
        queries.reserve(sites.size());
        for (auto index = size_t{0u};
             index < sites.size(); ++index) {
            if (dirty[index] == uint8_t{0u}) { continue; }
            auto site = sites[index];
            if (current_merge(site) == nullptr ||
                !dom.contains(site.header)) {
                dirty[index] = uint8_t{0u};
                continue;
            }
            queries.emplace_back(Query{
                .site_index = index,
                .depth = dom_depth(dom, site.header)});
        }
        {
            ScopedTimer _timer_selection_exit_sort_sites(
                "selection_exit_sort_sites");
            luisa::sort(
                queries.begin(), queries.end(),
                [](auto lhs, auto rhs) noexcept {
                    return lhs.depth > rhs.depth;
                });
        }

        auto version_modified = false;
        {
            ScopedTimer _timer_selection_exit_scan_sites(
                "selection_exit_scan_sites");
            for (auto query_index = size_t{0u};
                 query_index < queries.size();
                 ++query_index) {
                auto query = queries[query_index];
                if (dirty[query.site_index] == uint8_t{0u}) {
                    continue;
                }
                dirty[query.site_index] = uint8_t{0u};
                auto site = sites[query.site_index];
                auto *merge = current_merge(site);
                if (merge == nullptr) { continue; }
                ++info.selection_exit_site_query_count;
                auto result = canonicalize_selection_exits(
                    def, site.header, site.term, merge, dom,
                    cfg_relations, info,
                    rewritten_site_invalid_exit_counts,
                    exit_dispatch_headers);
                if (result.status ==
                    SelectionExitRewriteStatus::UNCHANGED) {
                    continue;
                }
                if (restructure_trace_enabled()) {
                    LUISA_VERBOSE_WITH_LOCATION(
                        "[restructure_cfg] selection-exit worklist: "
                        "{} / {} dirty sites queried before status {} "
                        "at dominance depth {}.",
                        query_index + 1u, queries.size(),
                        static_cast<uint32_t>(result.status),
                        query.depth);
                }
                if (result.status ==
                    SelectionExitRewriteStatus::STALLED_SITE) {
                    ++info.selection_exit_round_yield_count;
                    drain.yielded = true;
                    break;
                }

                drain.modified = true;
                version_modified = true;
                if (!result.cfg_modified) {
                    ++info.selection_exit_merge_canonicalization_count;
                    mark_enclosing_dependencies(
                        dom, site.header);
                    break;
                }
                drain.cfg_modified = true;
                ++info.selection_exit_cfg_invalidation_count;
                if (result.requires_ssa_repair) {
                    ssa_repair_requested = true;
                    ++info.selection_exit_ssa_repair_request_count;
                }
                if (result.local_dependency_only) {
                    ++info.selection_exit_local_invalidation_count;
                    mark_enclosing_dependencies(
                        dom, site.header);
                    for (auto *source :
                         result.mutated_edge_sources) {
                        if (auto iter =
                                site_index_by_header.find(source);
                            iter !=
                            site_index_by_header.end()) {
                            mark_dirty(iter->second);
                        }
                    }
                    for (auto *bypassed :
                         result.bypassed_forwarding_blocks) {
                        if (auto iter =
                                site_index_by_header.find(bypassed);
                            iter !=
                            site_index_by_header.end()) {
                            mark_dirty(iter->second);
                        }
                        for (auto index = size_t{0u};
                             index < sites.size(); ++index) {
                            if (current_merge(sites[index]) ==
                                bypassed) {
                                mark_dirty(index);
                            }
                        }
                    }
                } else {
                    ++info.selection_exit_global_invalidation_count;
                    for (auto index = size_t{0u};
                         index < dirty.size(); ++index) {
                        mark_dirty(index);
                    }
                }
                dom = compute_restructure_dom(def);
                if (result.local_dependency_only) {
                    mark_enclosing_dependencies(
                        dom, site.header);
                }
                break;
            }
        }
        if (drain.yielded) { break; }
        if (!version_modified) { break; }
    }
    // A state-dispatch rewrite changes only control-flow edges. Dominance and
    // subsequent selection-exit queries are independent of instruction
    // operands, so they may observe the temporarily non-SSA CFG. Repair once
    // against the final graph: every dynamically feasible use is still
    // preceded by its original definition, while reg2mem transports values
    // across only the extra syntactic paths introduced by the selectors.
    if (ssa_repair_requested) {
        ScopedTimer _timer_selection_exit_ssa_repair(
            "selection_exit_ssa_repair");
        auto repair =
            reg2mem_pass_repair_cross_block_rvalue_uses_on_function(
                static_cast<Function *>(def));
        ++info.selection_exit_ssa_repair_count;
        info.selection_exit_ssa_repaired_value_count +=
            repair.lowered_cross_block_value_count;
    }
    // Post-dominance has no observer inside the drain: selection-exit
    // classification and rewriting consume only the current dominator tree.
    // Preserve the same observation point for the following phase while
    // coalescing every write in this batch into one version refresh.
    if (drain.cfg_modified) {
        pdom = compute_post_dom(def, info);
        ++info.selection_exit_postdom_refresh_count;
    }
    return drain;
}

// Validate the same exact executable exit cut that drives canonicalization.
// A separate, weaker end condition (for example only checking merge identity
// uniqueness) can accept a selection whose cases first converge at T and then
// reach its declared merge M; SPIR-V rejects that shape because T is outside
// the declared selection construct. Sharing the analysis makes the rewrite
// rule and its proof obligation one definition rather than two approximations.
[[nodiscard]] size_t count_noncanonical_selection_exits(
    FunctionDefinition *def,
    RestructureCFGInfo &info,
    const luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    ScopedTimer _timer_selection_exit_audit(
        "post_count_noncanonical_selection_exits");
    if (def == nullptr) { return 0u; }
    auto dom = compute_restructure_dom(def);
    // Keep proof checking separate from the mutating phase's cost counters.
    // The audit performs no transformation and has its own observable counts.
    RestructureCFGInfo audit_work;
    auto cfg_relations =
        build_selection_exit_cfg_relations(
            def, dom, audit_work);
    auto invalid_count = size_t{0u};
    for (auto *header : def->basic_blocks()) {
        if (header == nullptr || !header->is_terminated() ||
            !dom.contains(header) ||
            exit_dispatch_headers.contains(header) ||
            cfg_relations.loop_boundary_selection_entries
                .contains(header)) {
            continue;
        }
        auto *term = header->terminator();
        if (!term->isa<IfInst>() &&
            !term->isa<SwitchInst>()) {
            continue;
        }
        auto *control_flow_merge =
            term->control_flow_merge();
        auto *merge = control_flow_merge == nullptr ?
                          nullptr :
                          control_flow_merge->merge_block();
        if (merge == nullptr) {
            ++invalid_count;
            continue;
        }
        ++info.selection_exit_audit_selection_count;
        auto analysis = analyze_selection_exits(
            header, term, merge, dom, cfg_relations,
            audit_work, exit_dispatch_headers);
        // analyze_selection_exits uses the same structured-exit predicate as
        // the mutating phase; any surviving edge is therefore illegal by the
        // exact proof obligation, with no weaker audit-side exception.
        invalid_count += analysis.invalid_exits.empty() ? 0u : 1u;
    }
    info.selection_exit_audit_invalid_count +=
        invalid_count;
    return invalid_count;
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

// Dominance overlay for transparent merge blocks inserted by one immutable
// if-restructuring batch. Each overlay block records an old-block anchor whose
// dominators are exactly the old dominators of the overlay. Queries between
// old blocks delegate to the immutable tree; no existing relation changes.
class IfBatchDominanceOverlay {
private:
    const DomTree &_base;
    const luisa::unordered_map<BasicBlock *, BasicBlock *> &_anchors;

public:
    IfBatchDominanceOverlay(
        const DomTree &base,
        const luisa::unordered_map<BasicBlock *, BasicBlock *> &anchors) noexcept
        : _base{base}, _anchors{anchors} {}

    [[nodiscard]] bool contains(BasicBlock *block) const noexcept {
        return _base.contains(block) || _anchors.contains(block);
    }

    [[nodiscard]] bool dominates(
        BasicBlock *source,
        BasicBlock *target) const noexcept {
        if (_base.contains(source)) {
            if (_base.contains(target)) {
                return _base.dominates(source, target);
            }
            if (auto iter = _anchors.find(target);
                iter != _anchors.end()) {
                return _base.dominates(source, iter->second);
            }
        }
        return source == target && contains(source);
    }
};

[[nodiscard]] BasicBlock *nearest_common_dominator(
    const DomTree &dom,
    BasicBlock *lhs,
    BasicBlock *rhs) noexcept {
    if (lhs == nullptr) { return rhs; }
    if (rhs == nullptr) { return lhs; }
    auto *lhs_node = dom.node_or_null(lhs);
    auto *rhs_node = dom.node_or_null(rhs);
    if (lhs_node == nullptr || rhs_node == nullptr) {
        return nullptr;
    }
    auto lhs_depth = dom_depth(dom, lhs);
    auto rhs_depth = dom_depth(dom, rhs);
    while (lhs_depth > rhs_depth) {
        lhs_node = lhs_node->parent();
        --lhs_depth;
    }
    while (rhs_depth > lhs_depth) {
        rhs_node = rhs_node->parent();
        --rhs_depth;
    }
    while (lhs_node != rhs_node) {
        lhs_node = lhs_node->parent();
        rhs_node = rhs_node->parent();
        LUISA_DEBUG_ASSERT(
            lhs_node != nullptr && rhs_node != nullptr,
            "Dominator tree nodes must share the function root.");
    }
    return lhs_node->block();
}

[[nodiscard]] bool try_restructure_if_batch(FunctionDefinition *def,
                                            DomTree &dom,
                                            PostDomInfo &pdom,
                                            RestructureCFGInfo &info,
                                            luisa::unordered_set<BasicBlock *> &all_created_structural_merges,
                                            luisa::unordered_map<BasicBlock *, BasicBlock *> &sm_to_header) noexcept {
    ScopedTimer _timer_try_if("try_restructure_if_batch");
    detail::SelectionMergeBatchAnalysis merge_analysis{def, dom};
    auto accumulate_merge_stats = [&]() noexcept {
        auto &stats = merge_analysis.stats();
        info.if_batch_merge_loop_context_count +=
            stats.loop_context_count;
        info.if_batch_merge_query_count += stats.query_count;
        info.if_batch_merge_block_visit_count +=
            stats.block_visit_count;
        info.if_batch_merge_edge_visit_count +=
            stats.edge_visit_count;
        info.if_batch_merge_aggregate_scan_count +=
            stats.aggregate_scan_count;
        info.if_batch_merge_dominator_ancestor_visit_count +=
            stats.dominator_ancestor_visit_count;
    };
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

        auto entries = std::array{true_bb, false_bb};
        auto *merge = merge_analysis.infer(
            bb,
            luisa::span<BasicBlock *const>{
                entries.data(), entries.size()});
        if (merge == nullptr) {
            auto ipm_it = pdom.ipostdom.find(bb);
            if (ipm_it == pdom.ipostdom.end() ||
                ipm_it->second == nullptr) {
                return;
            }
            merge = ipm_it->second;
        }
        if (merge == pdom.virtual_exit) { return; }
        if (merge == bb) { return; }

        if (!dom.strictly_dominates(bb, true_bb)) { return; }
        if (!dom.strictly_dominates(bb, false_bb)) { return; }

        candidates.push_back(
            {bb, cbr, merge, dom_depth(dom, bb)});
    });

    if (candidates.empty()) {
        accumulate_merge_stats();
        return false;
    }
    ++info.if_batch_analysis_count;
    info.if_batch_candidate_query_count += candidates.size();

    // Sort by depth descending (innermost first)
    luisa::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
        return a.depth > b.depth;
    });

    bool any = false;
    auto &created_structural_merges = all_created_structural_merges;
    luisa::unordered_map<BasicBlock *, BasicBlock *>
        overlay_dominance_anchors;
    IfBatchDominanceOverlay dominance{
        dom, overlay_dominance_anchors};
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
        auto *found_merge = merge_analysis.infer(
            found_header,
            luisa::span<BasicBlock *const>{
                entries.data(), entries.size()},
            &overlay_dominance_anchors);
        // Transparent subdivisions can hide every normal path behind an
        // overlay that is irrelevant to this candidate. In that case the
        // lexical merge proven on the immutable input remains the exact
        // quotient-graph fallback.
        if (found_merge == nullptr) {
            found_merge = cand.merge;
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
            merge_analysis.register_overlay_block(
                structural_merge);
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
                if (auto *nested_merge =
                        structured_statement_merge(bb->terminator());
                    nested_merge != nullptr &&
                    nested_merge != found_merge &&
                    nested_merge != structural_merge) {
                    scope_work.push_back(nested_merge);
                    continue;
                }
                bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                    scope_work.emplace_back(succ);
                });
            }
        }

        // Walk the dominator-tree subtree rooted at found_header.
        // Only retarget unstructured cbr/br blocks that are actually inside
        // the if's scope. Skip IfInst/SwitchInst/LoopInst terminators to avoid
        // corrupting already-structured inner constructs.
        auto retarget_scope_exit = [&](BasicBlock *bb,
                                       bool overlay_block) noexcept {
            if (bb == structural_merge ||
                bb == found_header ||
                bb == found_merge ||
                !bb->is_terminated() ||
                !if_scope_blocks.contains(bb) ||
                allowed_outside_targets.contains(bb)) {
                return;
            }
            auto *term = bb->terminator();
            if (!term->isa<ConditionalBranchInst>() &&
                !term->isa<BranchInst>()) {
                return;
            }
            auto is_loop_update_backedge = false;
            if (auto iter = loop_update_to_prepare.find(bb);
                iter != loop_update_to_prepare.end() &&
                iter->second == found_merge) {
                is_loop_update_backedge = true;
            }
            if (!is_loop_update_backedge) {
                auto retarget = overlay_block;
                if (!retarget && dom.contains(found_merge)) {
                    retarget =
                        !dom.strictly_dominates(found_merge, bb) ||
                        (dom.strictly_dominates(found_merge, bb) &&
                         dom.strictly_dominates(found_header, bb));
                }
                if (retarget) {
                    retarget_terminator(
                        term, found_merge, structural_merge);
                }
            }
            fix_degenerate_terminator(bb);
        };
        auto header_node = dom.node_or_null(found_header);
        if (header_node != nullptr) {
            luisa::vector<const DomTreeNode *> work;
            work.push_back(header_node);
            while (!work.empty()) {
                auto *node = work.back();
                work.pop_back();
                auto *bb = node->block();
                retarget_scope_exit(bb, false);
                for (auto *child : node->children()) {
                    work.push_back(child);
                }
            }
        }
        // The batch inserts only transparent structural merges. Contracting
        // those overlay blocks reproduces the immutable input graph, so
        // dominance between every pre-existing pair of blocks is unchanged.
        // A later outer selection still needs to retarget an overlay merge in
        // its scope; query its exact old-block dominance anchor and process it
        // explicitly instead of rebuilding the whole dominator tree.
        for (auto *bb : if_scope_blocks) {
            if (dom.contains(bb)) { continue; }
            ++info.if_batch_overlay_block_query_count;
            if (!overlay_dominance_anchors.contains(bb) ||
                !dominance.dominates(found_header, bb)) {
                continue;
            }
            retarget_scope_exit(bb, true);
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

            if (!dom.contains(structural_merge) &&
                !overlay_dominance_anchors.contains(
                    structural_merge)) {
                BasicBlock *anchor = nullptr;
                structural_merge->traverse_predecessors(
                    false, [&](BasicBlock *predecessor) noexcept {
                        if (!has_executable_edge(
                                predecessor,
                                structural_merge)) {
                            return;
                        }
                        auto *predecessor_anchor = predecessor;
                        if (!dom.contains(predecessor)) {
                            auto iter =
                                overlay_dominance_anchors.find(
                                    predecessor);
                            if (iter ==
                                overlay_dominance_anchors.end()) {
                                return;
                            }
                            predecessor_anchor = iter->second;
                        }
                        anchor = nearest_common_dominator(
                            dom, anchor, predecessor_anchor);
                    });
                LUISA_DEBUG_ASSERT(
                    anchor != nullptr && dom.contains(anchor),
                    "Transparent selection merge must have a reachable "
                    "dominance anchor.");
                if (anchor == nullptr) {
                    anchor = found_header;
                }
                overlay_dominance_anchors.emplace(
                    structural_merge, anchor);
            }

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
            // The immutable dominance relation remains valid for all old
            // blocks. Newly inserted merges are consumed through the explicit
            // dominance-anchor overlay above, so no per-candidate rebuild is
            // required.
        }
    }

    accumulate_merge_stats();
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
        // Construct-role operands (selection merges and loop body/update/merge
        // declarations) are not dynamic CFG edges. Following them here can
        // make a tiny entry split clone an unrelated, arbitrarily large
        // construct. The owned region is defined over executable reachability.
        traverse_executable_successors(B, [&](BasicBlock *S) noexcept {
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
    // Node splitting applies to a dynamic edge. BasicBlock use-lists also
    // expose declarative construct-role operands, so fail closed before doing
    // any cloning if P does not actually transfer control to E.
    if (!has_executable_edge(P, E)) { return false; }
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
    // Reroute every executable edge in P's terminator that targeted E to the
    // relay. This includes structured arm and loop-entry edges; the generic
    // raw-branch helper deliberately does not cover those instruction kinds.
    LUISA_ASSERT(
        retarget_executable_edge(P->terminator(), E, relay),
        "Failed to retarget executable edge after cloning its owned region.");
    return true;
}

struct PostMergeSelectionReentry {
    BasicBlock *header{nullptr};
    BasicBlock *merge{nullptr};
    BasicBlock *reentered_block{nullptr};
    BasicBlock *reentry_predecessor{nullptr};
    luisa::vector<BasicBlock *> entries;
};

enum struct SelectionReentryAnalysisPurpose : uint8_t {
    TRANSFORM,
    AUDIT,
};

// Return one witness edge for every selection whose current activation can be
// re-entered after its merge. For a selection (H, M), an executable edge
// (P, E) is such a witness exactly when
//
//   H dom E, M !dom E, M dom P, and P -> E,
//
// provided E is not the nearest enclosing loop/switch boundary. The last
// clause quotients the cyclic CFG by lexical construct epochs: completing an
// enclosing loop iteration and entering H again is a new activation, not a
// re-entry into the old one. Since E is in M's dominance frontier whenever
// the first four predicates hold, a sparse frontier walk is complete.
//
// Both the mutating phase and final audit call this analyzer. They can no
// longer disagree by using subtly different structured-exit exceptions.
[[nodiscard]] luisa::vector<PostMergeSelectionReentry>
analyze_post_merge_selection_reentries(
    FunctionDefinition *def,
    DomTree &dom,
    const SelectionExitCFGRelations &cfg_relations,
    const luisa::unordered_set<BasicBlock *> &ignored_headers,
    RestructureCFGInfo &info,
    SelectionReentryAnalysisPurpose purpose,
    bool stop_after_first) noexcept {
    // The frontier is a pure derivative of this immutable dominance snapshot
    // and is observed nowhere else in restructure_cfg. Keeping its
    // materialization at this semantic demand point makes every ancestry-only
    // tree strictly cheaper without weakening the witness theorem below.
    dom.compute_dominance_frontiers();
    ++info.selection_reentry_frontier_materialization_count;
    luisa::vector<PostMergeSelectionReentry> result;
    const auto &loop_boundary_selection_entries =
        cfg_relations.loop_boundary_selection_entries;
    for (auto *header : def->basic_blocks()) {
        if (header == nullptr || !header->is_terminated() ||
            !dom.contains(header) ||
            ignored_headers.contains(header) ||
            loop_boundary_selection_entries.contains(header)) {
            continue;
        }
        auto *term = header->terminator();
        if (!term->isa<IfInst>() &&
            !term->isa<SwitchInst>()) {
            continue;
        }
        auto *merge = structured_statement_merge(term);
        if (merge == nullptr || !dom.contains(merge)) {
            continue;
        }
        if (purpose == SelectionReentryAnalysisPurpose::AUDIT) {
            ++info.selection_reentry_audit_selection_query_count;
        }
        auto enclosing_context =
            selection_context(cfg_relations, header);
        auto allow_enclosing_switch = !term->isa<SwitchInst>();
        auto enclosing_loop_context = enclosing_context;
        while (enclosing_loop_context != SIZE_MAX &&
               cfg_relations.contexts[enclosing_loop_context].kind !=
                   SelectionExitCFGRelations::ContextKind::LOOP) {
            enclosing_loop_context =
                cfg_relations.contexts[enclosing_loop_context].parent;
        }
        // A loop boundary separates dynamic activations of a nested
        // selection. Lazily materialize the portion reachable from M before
        // entering the nearest enclosing loop's merge/continue target. The
        // dominance-frontier query below supplies only sparse candidates, so
        // this bounded reachability is paid only for selections that can
        // actually have a witness.
        auto same_epoch_materialized = false;
        luisa::unordered_set<BasicBlock *> same_epoch_after_merge;
        auto in_same_epoch_after_merge =
            [&](BasicBlock *candidate) noexcept {
                if (enclosing_loop_context == SIZE_MAX) {
                    return true;
                }
                if (!same_epoch_materialized) {
                    same_epoch_materialized = true;
                    auto &&loop =
                        cfg_relations.contexts[enclosing_loop_context];
                    luisa::vector<BasicBlock *> work{merge};
                    while (!work.empty()) {
                        auto *block = work.back();
                        work.pop_back();
                        if (block == nullptr ||
                            (block != merge &&
                             (block == loop.break_target ||
                              block == loop.continue_target)) ||
                            !same_epoch_after_merge.emplace(block).second) {
                            continue;
                        }
                        traverse_executable_successors(
                            block, [&](BasicBlock *successor) noexcept {
                                work.emplace_back(successor);
                            });
                    }
                }
                return same_epoch_after_merge.contains(candidate);
            };
        for (auto *frontier : dom.node(merge)->frontiers()) {
            if (purpose == SelectionReentryAnalysisPurpose::AUDIT) {
                ++info.selection_reentry_audit_frontier_query_count;
            } else {
                ++info.selection_reentry_edge_query_count;
            }
            auto *reentered = frontier->block();
            if (reentered == nullptr || reentered == header ||
                reentered == merge || !dom.contains(reentered) ||
                !dom.dominates(header, reentered) ||
                dom.dominates(merge, reentered) ||
                is_legal_enclosing_selection_exit_in_quotient(
                    cfg_relations, enclosing_context, reentered,
                    allow_enclosing_switch)) {
                continue;
            }
            BasicBlock *offender = nullptr;
            reentered->traverse_predecessors(
                false, [&](BasicBlock *predecessor) noexcept {
                    if (offender != nullptr) { return; }
                    if (purpose ==
                        SelectionReentryAnalysisPurpose::AUDIT) {
                        ++info.selection_reentry_audit_predecessor_query_count;
                    }
                    if (dom.contains(predecessor) &&
                        has_executable_edge(predecessor, reentered) &&
                        dom.dominates(merge, predecessor) &&
                        in_same_epoch_after_merge(predecessor)) {
                        offender = predecessor;
                    }
                });
            if (offender == nullptr) { continue; }
            result.emplace_back(PostMergeSelectionReentry{
                .header = header,
                .merge = merge,
                .reentered_block = reentered,
                .reentry_predecessor = offender});
            if (stop_after_first) { return result; }
            break;
        }
    }
    return result;
}

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
            !dom.contains(dispatch)) {
            continue;
        }
        auto *term = dispatch->terminator();
        if (!term->isa<ConditionalBranchInst>() &&
            !term->isa<IfInst>()) {
            continue;
        }
        auto *branch = static_cast<
            ConditionalBranchTerminatorInstruction *>(term);
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
            dom = compute_restructure_dom(def);
            pdom = compute_post_dom(def, info);
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
            dom = compute_restructure_dom(def);
            pdom = compute_post_dom(def, info);
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool split_one_post_merge_selection_reentry(
    FunctionDefinition *def,
    DomTree &dom,
    PostDomInfo &pdom,
    RestructureCFGInfo &info,
    const luisa::unordered_set<BasicBlock *> &
        exit_dispatch_headers) noexcept {
    ++info.selection_reentry_boundary_analysis_count;
    RestructureCFGInfo relation_work;
    const auto cfg_relations =
        build_selection_exit_cfg_relations(
            def, dom, relation_work);
    auto reentries = analyze_post_merge_selection_reentries(
        def, dom, cfg_relations, exit_dispatch_headers,
        info, SelectionReentryAnalysisPurpose::TRANSFORM,
        true);
    if (reentries.empty()) { return false; }
    auto reentry = std::move(reentries.front());
    collect_construct_entries(
        reentry.header, reentry.entries);
    if (reentry.entries.empty()) { return false; }
    repair_target_state_dispatch_ssa(def);
    dom = compute_restructure_dom(def);
    pdom = compute_post_dom(def, info);
    LUISA_ASSERT(
        clone_owned_subgraph_for_edge(
            def, reentry.header,
            reentry.reentered_block,
            reentry.reentry_predecessor,
            luisa::span<BasicBlock *const>{
                reentry.entries.data(),
                reentry.entries.size()},
            reentry.merge, dom, true),
        "Selection re-entry frontier splitting made no progress.");
    ++info.canonicalized_cfg_count;
    dom = compute_restructure_dom(def);
    pdom = compute_post_dom(def, info);
    return true;
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
    while (split_one_post_merge_selection_reentry(
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
                                             const luisa::unordered_set<BasicBlock *> &loop_boundary_selection_entries,
                                             luisa::unordered_set<Instruction *> &rewritten_sites) noexcept {
    ScopedTimer _timer_enforce_entries("enforce_construct_entries");
    // A loop-boundary IfInst is the structured XIR spelling of a physical
    // break/continue guard. The SPIR-V emitter intentionally emits no
    // OpSelectionMerge for it, so its loop prepare/update/merge targets are
    // allowed to have the loop's ordinary executable predecessors. It is not
    // a selection construct and must not be node-split as one.
    if (!requires_unique_construct_entries(
            header_bb,
            loop_boundary_selection_entries)) {
        return false;
    }
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
                dom = compute_restructure_dom(def);
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
                if (!dom.contains(P) || !has_executable_edge(P, E)) { return; }
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
        // The body of this scan observes one immutable CFG version. This set
        // is the value-numbered result of the boundary predicate for that
        // version; a successful rewrite exits the scan and invalidates it.
        ++info.construct_entry_boundary_analysis_count;
        const auto loop_boundary_selection_entries =
            collect_loop_boundary_selection_entries(def);
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
                    loop_boundary_selection_entries,
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
    dom = compute_restructure_dom(def);
    pdom = compute_post_dom(def, info);
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
// 4. If a construct has an actually illegal exit, or its merge conflicts with
//    its parent's merge/continue role, route its exits through one new merge.
//    Selection exits to the nearest enclosing loop or switch boundary are
//    legal in SPIR-V and do not themselves trigger a rewrite.
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

    for (;;) {
        // No mutation occurs while a candidate is selected. Materialize the
        // boundary relation once for this CFG version; the successful rewrite
        // at the bottom invalidates it together with dominance.
        ++info.construct_exit_boundary_analysis_count;
        RestructureCFGInfo relation_work;
        const auto structured_exit_relations =
            build_selection_exit_cfg_relations(
                def, dom, relation_work);
        const auto &loop_boundary_selection_entries =
            structured_exit_relations
                .loop_boundary_selection_entries;
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
                loop_boundary_selection_entries.contains(header)) {
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

        // Derive the construct hierarchy with one event walk over the sparse
        // dominator tree. For a construct (H, M), the exact set in which it
        // can parent another header X is
        //
        //   subtree(H) - subtree(M) - {continue(H)}.
        //
        // (The M term disappears when M is unreachable or outside H's
        // subtree.) Entering H activates the construct after H's own parent
        // query; entering M suspends it for that complete subtree. The
        // deepest active construct is therefore exactly the former O(C^2)
        // pairwise `encloses` maximum, including the continue-target
        // exception. Events are restored on DFS exit, so sibling subtrees see
        // the same immutable-CFG facts without recomputation.
        luisa::unordered_map<BasicBlock *, size_t>
            construct_index_by_header;
        luisa::unordered_map<BasicBlock *, luisa::vector<size_t>>
            construct_merge_events;
        luisa::vector<uint8_t> construct_can_be_active(
            constructs.size(), uint8_t{1u});
        construct_index_by_header.reserve(constructs.size());
        for (auto i = size_t{0u}; i < constructs.size(); ++i) {
            auto &construct = constructs[i];
            construct_index_by_header.emplace(
                construct.header, i);
            if (!dom.contains(construct.merge)) { continue; }
            if (dom.dominates(
                    construct.merge,
                    construct.header)) {
                // M dominates H: subtree(H) is entirely outside the physical
                // construct interior, so this construct encloses no header.
                construct_can_be_active[i] = uint8_t{0u};
            } else if (dom.dominates(
                           construct.header,
                           construct.merge)) {
                construct_merge_events[construct.merge]
                    .emplace_back(i);
            }
        }

        using ActiveConstructKey =
            std::pair<size_t, size_t>;// (dom depth, construct index)
        std::set<ActiveConstructKey> active_constructs;
        struct DomWalkFrame {
            const DomTreeNode *node{nullptr};
            size_t depth{0u};
            size_t next_child{0u};
            size_t activated_construct{SIZE_MAX};
            luisa::vector<size_t> suspended_constructs;
        };
        luisa::vector<DomWalkFrame> dom_walk;
        if (dom.root() != nullptr) {
            dom_walk.emplace_back(DomWalkFrame{
                .node = dom.root()});
        }
        while (!dom_walk.empty()) {
            auto &frame = dom_walk.back();
            auto *block = frame.node->block();
            if (frame.next_child == 0u) {
                if (auto event_iter =
                        construct_merge_events.find(block);
                    event_iter != construct_merge_events.end()) {
                    for (auto construct_index :
                         event_iter->second) {
                        auto key = ActiveConstructKey{
                            constructs[construct_index].depth,
                            construct_index};
                        if (active_constructs.erase(key) != 0u) {
                            frame.suspended_constructs.emplace_back(
                                construct_index);
                        }
                    }
                }
                if (auto construct_iter =
                        construct_index_by_header.find(block);
                    construct_iter !=
                    construct_index_by_header.end()) {
                    auto construct_index = construct_iter->second;
                    auto &inner = constructs[construct_index];
                    inner.depth = frame.depth;
                    for (auto iter = active_constructs.rbegin();
                         iter != active_constructs.rend(); ++iter) {
                        ++info.construct_exit_parent_query_count;
                        auto &outer = constructs[iter->second];
                        if (outer.continue_target == block) {
                            continue;
                        }
                        inner.parent = &outer;
                        break;
                    }
                    if (construct_can_be_active[construct_index] != 0u) {
                        active_constructs.emplace(
                            inner.depth, construct_index);
                        frame.activated_construct = construct_index;
                    }
                }
            }
            auto children = frame.node->children();
            if (frame.next_child < children.size()) {
                auto *child = children[frame.next_child++];
                dom_walk.emplace_back(DomWalkFrame{
                    .node = child,
                    .depth = frame.depth + 1u});
                continue;
            }
            if (frame.activated_construct != SIZE_MAX) {
                auto construct_index =
                    frame.activated_construct;
                active_constructs.erase(ActiveConstructKey{
                    constructs[construct_index].depth,
                    construct_index});
            }
            for (auto construct_index :
                 frame.suspended_constructs) {
                active_constructs.emplace(
                    constructs[construct_index].depth,
                    construct_index);
            }
            dom_walk.pop_back();
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
            auto selection = node.term->isa<IfInst>() ||
                             node.term->isa<SwitchInst>();
            auto enclosing_context = selection_context(
                structured_exit_relations, node.header);
            auto allow_enclosing_switch =
                !node.term->isa<SwitchInst>();
            auto is_legal_exit =
                [&](BasicBlock *target) noexcept {
                    return target == node.merge ||
                           target == node.continue_target ||
                           (selection &&
                            is_legal_enclosing_selection_exit_in_quotient(
                                structured_exit_relations,
                                enclosing_context, target,
                                allow_enclosing_switch));
                };
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
                // Ordinary dominance alone over-approximates a selection
                // construct when one arm exits an enclosing construct. Once
                // such an arm reaches the nearest legal loop/switch boundary,
                // every block after that boundary belongs to the enclosing
                // construct, even if the selection header happens to dominate
                // it in the executable CFG. This is structural dominance's
                // epoch cut. Do not apply it to the header itself: a selection
                // may begin at a SimpleLoop continue/body target.
                if (selection && block != node.header &&
                    is_legal_enclosing_selection_exit_in_quotient(
                        structured_exit_relations,
                        enclosing_context, block,
                        allow_enclosing_switch)) {
                    continue;
                }
                if (!blocks.emplace(block).second) { continue; }
                traverse_executable_successors(
                    block, [&](BasicBlock *successor) noexcept {
                        work.emplace_back(successor);
                    });
            }

            luisa::vector<SelectionExitEdge> exits;
            for (auto *block : blocks) {
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
                if (!is_legal_exit(edge.dst)) {
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
        dom = compute_restructure_dom(def);
        pdom = compute_post_dom(def, info);
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
    auto dom = compute_restructure_dom(def);
    const auto loop_boundary_selection_entries =
        collect_loop_boundary_selection_entries(def);
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
        if (!requires_unique_construct_entries(
                header,
                loop_boundary_selection_entries)) {
            continue;
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
                if (!dom.contains(predecessor) ||
                    !has_executable_edge(predecessor, entry)) {
                    return;
                }
                invalid |= !is_authorized_construct_pred(
                    term, entry, header, predecessor);
            });
        }
        count += invalid ? 1u : 0u;
    }
    return count;
}

[[nodiscard]] size_t count_post_merge_selection_reentries(
    FunctionDefinition *def,
    RestructureCFGInfo &info) noexcept {
    auto dom = compute_restructure_dom(def);
    RestructureCFGInfo relation_work;
    const auto cfg_relations =
        build_selection_exit_cfg_relations(
            def, dom, relation_work);
    const luisa::unordered_set<BasicBlock *> no_ignored_headers;
    return analyze_post_merge_selection_reentries(
               def, dom, cfg_relations,
               no_ignored_headers, info,
               SelectionReentryAnalysisPurpose::AUDIT,
               false)
        .size();
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
    // Expose cyclic indexed edges to the common loop-recovery algorithm before
    // recovering the remaining native multi-way selection boundaries.
    if (lower_cyclic_indexed_branches(def)) {
        ++info.canonicalized_cfg_count;
    }
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
        auto dom = compute_restructure_dom(def);
        auto pdom = compute_post_dom(def, info);
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
    // Main structurization can recover an opposing break/continue branch as
    // an ordinary IfInst with a third, synthetic merge. Normalize these
    // physical loop guards first: one arm becomes the transparent merge arm,
    // matching their no-OpSelectionMerge lowering. Generic construct-entry
    // enforcement must observe that canonical construct classification and
    // never node-split an enclosing loop prepare/update region for the guard.
    if (canonicalize_loop_boundary_selection_merges(def, info)) {
        ++info.canonicalized_cfg_count;
    }
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
        auto dom = compute_restructure_dom(def);
        auto pdom = compute_post_dom(def, info);
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
                dom = compute_restructure_dom(def);
                pdom = compute_post_dom(def, info);
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
                dom = compute_restructure_dom(def);
                pdom = compute_post_dom(def, info);
            }
            auto selection_exit_drain =
                drain_selection_exits(
                    def, dom, pdom, info,
                    exit_dispatch_headers);
            auto selection_exit_changed =
                selection_exit_drain.modified;
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
            if (selection_exit_drain.yielded &&
                restructure_trace_enabled()) {
                LUISA_VERBOSE_WITH_LOCATION(
                    "[restructure_cfg] selection-exit drain yielded to "
                    "the remaining post canonicalizers after a site revisit.");
            }
            auto boundary_merge_changed =
                canonicalize_loop_boundary_selection_merges(
                    def, info);
            if (boundary_merge_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_restructure_dom(def);
                pdom = compute_post_dom(def, info);
            }
            auto boundary_branch_changed =
                normalize_loop_boundary_conditional_branches(
                    def, exit_dispatch_headers,
                    generated_exit_dispatch_headers);
            if (boundary_branch_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_restructure_dom(def);
                pdom = compute_post_dom(def, info);
            }
            auto loop_prepare_changed =
                canonicalize_loop_prepare_blocks(def);
            if (loop_prepare_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_restructure_dom(def);
                pdom = compute_post_dom(def, info);
            }
            auto loop_continue_changed =
                normalize_structured_loop_continues(
                    def, dom, info);
            if (loop_continue_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                pdom = compute_post_dom(def, info);
            }
            auto loop_update_changed =
                canonicalize_loop_update_blocks(def);
            if (loop_update_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_restructure_dom(def);
                pdom = compute_post_dom(def, info);
            }
            auto limits_before_fixup = info.iteration_limit_count;
            auto construct_exit_changed =
                fixup_construct_exits(
                    def, dom, pdom, info,
                    exit_dispatch_headers);
            if (construct_exit_changed) {
                ++info.canonicalized_cfg_count;
                local = true;
                dom = compute_restructure_dom(def);
                pdom = compute_post_dom(def, info);
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
                dom = compute_restructure_dom(def);
                pdom = compute_post_dom(def, info);
            }
            if (restructure_trace_enabled()) {
                auto stats_after = trace_stats(def);
                LUISA_VERBOSE_WITH_LOCATION(
                    "[restructure_cfg] post iteration {}: "
                    "blocks {} -> {}, instructions {} -> {}; "
                    "raw conditional {} -> {}, structured selections {} -> {}; "
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
                    stats_before.raw_conditional_count,
                    stats_after.raw_conditional_count,
                    stats_before.structured_selection_count,
                    stats_after.structured_selection_count,
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
    info.invalid_construct_count +=
        count_noncanonical_selection_exits(
            def, info, exit_dispatch_headers);
    auto selection_reentry_count = size_t{0u};
    {
        ScopedTimer _timer_reentries(
            "post_count_selection_reentries");
        selection_reentry_count =
            count_post_merge_selection_reentries(def, info);
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
        if (restructure_trace_enabled()) {
            auto stats = trace_stats(def);
            if (stats.block_count <= 128u &&
                stats.instruction_count <= 4096u) {
                luisa::string dump;
                XIRDebugPrinter printer;
                printer.emit_function(dump, def);
                LUISA_VERBOSE_WITH_LOCATION(
                    "[restructure_cfg] incomplete function dump:\n{}",
                    dump);
            }
        }
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
    const auto verify_boundaries =
        xir_pass_has_standalone_verification(
            options.verification_transaction,
            function);
    LUISA_ASSERT(
        verify_boundaries ||
            options.mutation_mode ==
                RestructureCFGMutationMode::IN_PLACE_DISCARDABLE,
        "A restructure_cfg pass inside an enclosing verification transaction "
        "must use disposable in-place mutation; transactional shadow commit "
        "owns and requires its standalone output boundary.");

    // The public pass contract has one complete input verifier boundary.
    // Structural preconditions below are transform-specific analyses, not
    // replacements for this general XIR validity check.
    auto preflight = RestructureCFGInfo{};
    if (verify_boundaries) {
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
    }
    preflight = preflight_restructure_cfg(
        def, verify_intermediate);
    preflight.boundary_verifier_count =
        verify_boundaries ? 1u : 0u;
    if (!preflight.succeeded()) { return preflight; }

    if (options.mutation_mode ==
        RestructureCFGMutationMode::IN_PLACE_DISCARDABLE) {
        // The caller has declared the input disposable on failure, so run the
        // mutating engine once on the original definition. The same complete
        // boundary verification contract still applies on success.
        auto info = restructure_cfg_on_definition_in_place(
            def, options, verify_intermediate);
        info.boundary_verifier_count =
            verify_boundaries ? 1u : 0u;
        info.intermediate_verifier_count +=
            preflight.intermediate_verifier_count;
        if (info.succeeded() && verify_boundaries) {
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
        luisa::vector<ShadowDefinition> shadows{std::move(shadow)};
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
    luisa::vector<ShadowDefinition> shadows{std::move(shadow)};
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
            "construct_entry_boundary_analysis",
            info.construct_entry_boundary_analysis_count);
        report->set(
            "construct_exit_boundary_analysis",
            info.construct_exit_boundary_analysis_count);
        report->set(
            "construct_exit_parent_query",
            info.construct_exit_parent_query_count);
        report->set(
            "if_batch_analysis",
            info.if_batch_analysis_count);
        report->set(
            "if_batch_candidate_query",
            info.if_batch_candidate_query_count);
        report->set(
            "if_batch_overlay_block_query",
            info.if_batch_overlay_block_query_count);
        report->set(
            "if_batch_merge_loop_context",
            info.if_batch_merge_loop_context_count);
        report->set(
            "if_batch_merge_query",
            info.if_batch_merge_query_count);
        report->set(
            "if_batch_merge_block_visit",
            info.if_batch_merge_block_visit_count);
        report->set(
            "if_batch_merge_edge_visit",
            info.if_batch_merge_edge_visit_count);
        report->set(
            "if_batch_merge_aggregate_scan",
            info.if_batch_merge_aggregate_scan_count);
        report->set(
            "if_batch_merge_dominator_ancestor_visit",
            info.if_batch_merge_dominator_ancestor_visit_count);
        report->set(
            "loop_continue_analysis",
            info.loop_continue_analysis_count);
        report->set(
            "loop_continue_site_query",
            info.loop_continue_site_query_count);
        report->set(
            "loop_continue_invalidation",
            info.loop_continue_invalidation_count);
        report->set(
            "loop_continue_dominance_rebuild",
            info.loop_continue_dominance_rebuild_count);
        report->set(
            "loop_continue_region_block_visit",
            info.loop_continue_region_block_visit_count);
        report->set(
            "loop_continue_region_edge_visit",
            info.loop_continue_region_edge_visit_count);
        report->set(
            "loop_continue_planned_rewrite",
            info.loop_continue_planned_rewrite_count);
        report->set(
            "loop_continue_applied_rewrite",
            info.loop_continue_applied_rewrite_count);
        report->set(
            "loop_continue_dom_numbered_block",
            info.loop_continue_dom_numbered_block_count);
        report->set(
            "loop_continue_dom_numbered_edge",
            info.loop_continue_dom_numbered_edge_count);
        report->set(
            "loop_continue_dom_fixed_point_iteration",
            info.loop_continue_dom_fixed_point_iteration_count);
        report->set(
            "loop_continue_dom_fixed_point_block_visit",
            info.loop_continue_dom_fixed_point_block_visit_count);
        report->set(
            "loop_continue_dom_fixed_point_edge_visit",
            info.loop_continue_dom_fixed_point_edge_visit_count);
        report->set(
            "loop_continue_dom_intersect_step",
            info.loop_continue_dom_intersect_step_count);
        report->set(
            "postdom_analysis",
            info.postdom_analysis_count);
        report->set(
            "postdom_numbered_block",
            info.postdom_numbered_block_count);
        report->set(
            "postdom_numbered_edge",
            info.postdom_numbered_edge_count);
        report->set(
            "postdom_active_block",
            info.postdom_active_block_count);
        report->set(
            "postdom_fixed_point_iteration",
            info.postdom_fixed_point_iteration_count);
        report->set(
            "postdom_fixed_point_block_visit",
            info.postdom_fixed_point_block_visit_count);
        report->set(
            "postdom_fixed_point_edge_visit",
            info.postdom_fixed_point_edge_visit_count);
        report->set(
            "postdom_intersect_step",
            info.postdom_intersect_step_count);
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
            "selection_exit_boundary_dataflow",
            info.selection_exit_boundary_dataflow_count);
        report->set(
            "selection_exit_boundary_block_visit",
            info.selection_exit_boundary_block_visit_count);
        report->set(
            "selection_exit_boundary_edge_visit",
            info.selection_exit_boundary_edge_visit_count);
        report->set(
            "selection_exit_boundary_classification",
            info.selection_exit_boundary_classification_count);
        report->set(
            "selection_exit_site_query",
            info.selection_exit_site_query_count);
        report->set(
            "selection_exit_enclosing_loop_query",
            info.selection_exit_enclosing_loop_query_count);
        report->set(
            "selection_exit_region_block_visit",
            info.selection_exit_region_block_visit_count);
        report->set(
            "selection_exit_region_edge_visit",
            info.selection_exit_region_edge_visit_count);
        report->set(
            "selection_exit_loop_context",
            info.selection_exit_loop_context_count);
        report->set(
            "selection_exit_cfg_invalidation",
            info.selection_exit_cfg_invalidation_count);
        report->set(
            "selection_exit_local_invalidation",
            info.selection_exit_local_invalidation_count);
        report->set(
            "selection_exit_global_invalidation",
            info.selection_exit_global_invalidation_count);
        report->set(
            "selection_exit_ssa_repair_request",
            info.selection_exit_ssa_repair_request_count);
        report->set(
            "selection_exit_ssa_repair",
            info.selection_exit_ssa_repair_count);
        report->set(
            "selection_exit_ssa_repaired_value",
            info.selection_exit_ssa_repaired_value_count);
        report->set(
            "selection_exit_dependency_requery",
            info.selection_exit_dependency_requery_count);
        report->set(
            "selection_exit_postdom_refresh",
            info.selection_exit_postdom_refresh_count);
        report->set(
            "selection_exit_round_yield",
            info.selection_exit_round_yield_count);
        report->set(
            "boundary_merge_analysis",
            info.boundary_merge_analysis_count);
        report->set(
            "boundary_merge_dataflow",
            info.boundary_merge_dataflow_count);
        report->set(
            "boundary_merge_classification",
            info.boundary_merge_classification_count);
        report->set(
            "boundary_merge_rewrite_batch",
            info.boundary_merge_rewrite_batch_count);
        report->set(
            "selection_reentry_boundary_analysis",
            info.selection_reentry_boundary_analysis_count);
        report->set(
            "selection_reentry_frontier_materialization",
            info.selection_reentry_frontier_materialization_count);
        report->set(
            "selection_reentry_edge_query",
            info.selection_reentry_edge_query_count);
        report->set(
            "selection_reentry_owner_query",
            info.selection_reentry_owner_query_count);
        report->set(
            "selection_reentry_audit_selection_query",
            info.selection_reentry_audit_selection_query_count);
        report->set(
            "selection_reentry_audit_frontier_query",
            info.selection_reentry_audit_frontier_query_count);
        report->set(
            "selection_reentry_audit_predecessor_query",
            info.selection_reentry_audit_predecessor_query_count);
        report->set(
            "irreducible_region", info.irreducible_region_count);
        report->set(
            "unstructured_branch", info.unstructured_branch_count);
        report->set(
            "invalid_construct", info.invalid_construct_count);
        report->set("iteration_limit", info.iteration_limit_count);
    };
    if (module == nullptr) {
        LUISA_ASSERT(
            options.verification_transaction == nullptr,
            "A null module cannot belong to an enclosing XIR pass "
            "verification transaction.");
        set_report(total);
        return total;
    }
    const auto verify_boundaries =
        xir_pass_has_standalone_verification(
            options.verification_transaction,
            module);
    LUISA_ASSERT(
        verify_boundaries ||
            options.mutation_mode ==
                RestructureCFGMutationMode::IN_PLACE_DISCARDABLE,
        "A module restructure_cfg pass inside an enclosing verification "
        "transaction must use disposable in-place mutation; transactional "
        "shadow commit owns and requires its standalone output boundary.");
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
        dst.construct_entry_boundary_analysis_count +=
            src.construct_entry_boundary_analysis_count;
        dst.construct_exit_boundary_analysis_count +=
            src.construct_exit_boundary_analysis_count;
        dst.construct_exit_parent_query_count +=
            src.construct_exit_parent_query_count;
        dst.if_batch_analysis_count +=
            src.if_batch_analysis_count;
        dst.if_batch_candidate_query_count +=
            src.if_batch_candidate_query_count;
        dst.if_batch_overlay_block_query_count +=
            src.if_batch_overlay_block_query_count;
        dst.if_batch_merge_loop_context_count +=
            src.if_batch_merge_loop_context_count;
        dst.if_batch_merge_query_count +=
            src.if_batch_merge_query_count;
        dst.if_batch_merge_block_visit_count +=
            src.if_batch_merge_block_visit_count;
        dst.if_batch_merge_edge_visit_count +=
            src.if_batch_merge_edge_visit_count;
        dst.if_batch_merge_aggregate_scan_count +=
            src.if_batch_merge_aggregate_scan_count;
        dst.if_batch_merge_dominator_ancestor_visit_count +=
            src.if_batch_merge_dominator_ancestor_visit_count;
        dst.loop_continue_analysis_count +=
            src.loop_continue_analysis_count;
        dst.loop_continue_site_query_count +=
            src.loop_continue_site_query_count;
        dst.loop_continue_invalidation_count +=
            src.loop_continue_invalidation_count;
        dst.loop_continue_dominance_rebuild_count +=
            src.loop_continue_dominance_rebuild_count;
        dst.loop_continue_region_block_visit_count +=
            src.loop_continue_region_block_visit_count;
        dst.loop_continue_region_edge_visit_count +=
            src.loop_continue_region_edge_visit_count;
        dst.loop_continue_planned_rewrite_count +=
            src.loop_continue_planned_rewrite_count;
        dst.loop_continue_applied_rewrite_count +=
            src.loop_continue_applied_rewrite_count;
        dst.loop_continue_dom_numbered_block_count +=
            src.loop_continue_dom_numbered_block_count;
        dst.loop_continue_dom_numbered_edge_count +=
            src.loop_continue_dom_numbered_edge_count;
        dst.loop_continue_dom_fixed_point_iteration_count +=
            src.loop_continue_dom_fixed_point_iteration_count;
        dst.loop_continue_dom_fixed_point_block_visit_count +=
            src.loop_continue_dom_fixed_point_block_visit_count;
        dst.loop_continue_dom_fixed_point_edge_visit_count +=
            src.loop_continue_dom_fixed_point_edge_visit_count;
        dst.loop_continue_dom_intersect_step_count +=
            src.loop_continue_dom_intersect_step_count;
        dst.postdom_analysis_count +=
            src.postdom_analysis_count;
        dst.postdom_numbered_block_count +=
            src.postdom_numbered_block_count;
        dst.postdom_numbered_edge_count +=
            src.postdom_numbered_edge_count;
        dst.postdom_active_block_count +=
            src.postdom_active_block_count;
        dst.postdom_fixed_point_iteration_count +=
            src.postdom_fixed_point_iteration_count;
        dst.postdom_fixed_point_block_visit_count +=
            src.postdom_fixed_point_block_visit_count;
        dst.postdom_fixed_point_edge_visit_count +=
            src.postdom_fixed_point_edge_visit_count;
        dst.postdom_intersect_step_count +=
            src.postdom_intersect_step_count;
        dst.definition_transform_invocation_count +=
            src.definition_transform_invocation_count;
        dst.boundary_verifier_count +=
            src.boundary_verifier_count;
        dst.intermediate_verifier_count +=
            src.intermediate_verifier_count;
        dst.selection_exit_boundary_analysis_count +=
            src.selection_exit_boundary_analysis_count;
        dst.selection_exit_boundary_dataflow_count +=
            src.selection_exit_boundary_dataflow_count;
        dst.selection_exit_boundary_block_visit_count +=
            src.selection_exit_boundary_block_visit_count;
        dst.selection_exit_boundary_edge_visit_count +=
            src.selection_exit_boundary_edge_visit_count;
        dst.selection_exit_boundary_classification_count +=
            src.selection_exit_boundary_classification_count;
        dst.selection_exit_site_query_count +=
            src.selection_exit_site_query_count;
        dst.selection_exit_enclosing_loop_query_count +=
            src.selection_exit_enclosing_loop_query_count;
        dst.selection_exit_region_block_visit_count +=
            src.selection_exit_region_block_visit_count;
        dst.selection_exit_region_edge_visit_count +=
            src.selection_exit_region_edge_visit_count;
        dst.selection_exit_loop_context_count +=
            src.selection_exit_loop_context_count;
        dst.selection_exit_cfg_invalidation_count +=
            src.selection_exit_cfg_invalidation_count;
        dst.selection_exit_local_invalidation_count +=
            src.selection_exit_local_invalidation_count;
        dst.selection_exit_global_invalidation_count +=
            src.selection_exit_global_invalidation_count;
        dst.selection_exit_ssa_repair_request_count +=
            src.selection_exit_ssa_repair_request_count;
        dst.selection_exit_ssa_repair_count +=
            src.selection_exit_ssa_repair_count;
        dst.selection_exit_ssa_repaired_value_count +=
            src.selection_exit_ssa_repaired_value_count;
        dst.selection_exit_dependency_requery_count +=
            src.selection_exit_dependency_requery_count;
        dst.selection_exit_postdom_refresh_count +=
            src.selection_exit_postdom_refresh_count;
        dst.selection_exit_round_yield_count +=
            src.selection_exit_round_yield_count;
        dst.boundary_merge_analysis_count +=
            src.boundary_merge_analysis_count;
        dst.boundary_merge_dataflow_count +=
            src.boundary_merge_dataflow_count;
        dst.boundary_merge_classification_count +=
            src.boundary_merge_classification_count;
        dst.boundary_merge_rewrite_batch_count +=
            src.boundary_merge_rewrite_batch_count;
        dst.selection_reentry_boundary_analysis_count +=
            src.selection_reentry_boundary_analysis_count;
        dst.selection_reentry_frontier_materialization_count +=
            src.selection_reentry_frontier_materialization_count;
        dst.selection_reentry_edge_query_count +=
            src.selection_reentry_edge_query_count;
        dst.selection_reentry_owner_query_count +=
            src.selection_reentry_owner_query_count;
        dst.selection_reentry_audit_selection_query_count +=
            src.selection_reentry_audit_selection_query_count;
        dst.selection_reentry_audit_frontier_query_count +=
            src.selection_reentry_audit_frontier_query_count;
        dst.selection_reentry_audit_predecessor_query_count +=
            src.selection_reentry_audit_predecessor_query_count;
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
    if (verify_boundaries) {
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
            trace_preflight_result(definition_index, info);
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
        total.boundary_verifier_count =
            verify_boundaries ? 1u : 0u;
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
        if (total.succeeded() && verify_boundaries) {
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
