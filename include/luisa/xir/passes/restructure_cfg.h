#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;

struct RestructureCFGInfo {
    size_t restructured_loop_count{0u};
    size_t restructured_if_count{0u};
    size_t restructured_switch_count{0u};
    // Number of successful auxiliary CFG-canonicalization phases. These
    // rewrites do not create a new structured construct, but they still mutate
    // the IR (for example by splitting duplicate Switch case targets).
    size_t canonicalized_cfg_count{0u};
    // Diagnostic operation count for dominance trees built while enforcing
    // unique structured-construct entries. This is not a change count.
    size_t construct_entry_dom_tree_count{0u};
    // Loop-boundary selection membership is a value of one immutable CFG
    // version. Entry enforcement materializes the complete relation once per
    // version instead of rediscovering it separately for every construct.
    size_t construct_entry_boundary_analysis_count{0u};
    // Construct-exit repair follows the same versioned-analysis contract. A
    // successful repair invalidates the relation and the next scan rebuilds
    // it exactly once.
    size_t construct_exit_boundary_analysis_count{0u};
    // Candidate inspections performed while deriving the physical construct
    // hierarchy from the sparse dominator tree. This grows with actual active
    // nesting, not with the square of the number of constructs.
    size_t construct_exit_parent_query_count{0u};
    // If restructuring analyzes one immutable dominance snapshot, stores the
    // lexical merge of every raw conditional, then inserts only transparent
    // edge subdivisions. Candidate queries therefore scale with raw If sites;
    // new overlay blocks are handled through explicit owner provenance rather
    // than rebuilding dominance after each rewrite.
    size_t if_batch_analysis_count{0u};
    size_t if_batch_candidate_query_count{0u};
    size_t if_batch_overlay_block_query_count{0u};
    // Selection merges share one dense value-numbering workspace and one
    // persistent loop-context tree per immutable if batch. Epoch arrays make
    // query cost proportional to actually visited blocks and edges without
    // rebuilding pointer hash maps for each arm.
    size_t if_batch_merge_loop_context_count{0u};
    size_t if_batch_merge_query_count{0u};
    size_t if_batch_merge_block_visit_count{0u};
    size_t if_batch_merge_edge_visit_count{0u};
    // Loop-continue normalization shares ownership and dominance across all
    // site queries in one immutable CFG version. Any successful site rewrite
    // invalidates both analyses before the next site is inspected.
    size_t loop_continue_analysis_count{0u};
    size_t loop_continue_site_query_count{0u};
    size_t loop_continue_invalidation_count{0u};
    // Physical invocations of the per-definition mutating transform. A
    // successful transactional pass invokes it twice per definition (shadow
    // validation plus identity-preserving replay); an in-place pass invokes
    // it once. This is a diagnostic operation count, not a change count.
    size_t definition_transform_invocation_count{0u};
    // Full XIR verifier invocations at the public pass boundaries.
    // A successful public function/module pass has exactly two: one for the
    // complete input and one for the complete candidate output.
    size_t boundary_verifier_count{0u};
    // Additional per-definition verifier invocations are diagnostic only and
    // disabled by default. Set LUISA_XIR_VERIFY_INTERMEDIATE=1 to enable them.
    size_t intermediate_verifier_count{0u};
    // Selection-exit scans classify loop-boundary selections once per
    // observed CFG version, then reuse that exact relation for every site.
    // These are diagnostic operation counts, not change counts.
    size_t selection_exit_boundary_analysis_count{0u};
    // A boundary analysis numbers its blocks once, solves one sparse
    // monotone dataflow per reachable loop, then performs O(1) lookups for
    // each IfInst arm. No arm launches an independent CFG search.
    size_t selection_exit_boundary_dataflow_count{0u};
    // Each dataflow solves only the structurally reachable loop region plus
    // its explicit boundary nodes. These counts expose the sparse induced
    // subgraphs actually visited across all loop contexts.
    size_t selection_exit_boundary_block_visit_count{0u};
    size_t selection_exit_boundary_edge_visit_count{0u};
    size_t selection_exit_boundary_classification_count{0u};
    size_t selection_exit_site_query_count{0u};
    size_t selection_exit_enclosing_loop_query_count{0u};
    // Actual sparse selection-region work across all site queries.
    size_t selection_exit_region_block_visit_count{0u};
    size_t selection_exit_region_edge_visit_count{0u};
    // Persistent enclosing-loop context nodes materialized while scanning
    // selection exits. There is exactly one node per reachable structured
    // loop per observed CFG version, never one loop-exit set per block.
    size_t selection_exit_loop_context_count{0u};
    // Selection-exit rewrites invalidate dominance immediately because the
    // next site query consumes it. Post-dominance is not observed inside the
    // drain, so one refresh is deferred until the complete mutation batch has
    // finished. These counters make that analysis-lifetime contract visible.
    size_t selection_exit_cfg_invalidation_count{0u};
    size_t selection_exit_local_invalidation_count{0u};
    size_t selection_exit_global_invalidation_count{0u};
    // Clean site results invalidated by the precise construct-dependency
    // worklist after a localized rewrite.
    size_t selection_exit_dependency_requery_count{0u};
    size_t selection_exit_postdom_refresh_count{0u};
    // Rewriting a nested selection may make a site already handled in the
    // current drain round eligible again. The selection phase yields so later
    // canonicalizers can collapse the generated protocol before the next
    // bounded outer fixed-point round.
    size_t selection_exit_round_yield_count{0u};
    // Loop-boundary merge canonicalization is analyzed against an immutable
    // CFG snapshot. Each snapshot numbers blocks once, solves one sparse
    // reverse-CFG dataflow per visited loop, and classifies each IfInst arm
    // by an O(1) lookup. A rewrite batch invalidates the complete snapshot.
    size_t boundary_merge_analysis_count{0u};
    size_t boundary_merge_dataflow_count{0u};
    size_t boundary_merge_classification_count{0u};
    size_t boundary_merge_rewrite_batch_count{0u};
    // Post-merge selection re-entry scans materialize loop-boundary
    // membership once per immutable CFG version, then inspect only dominator
    // ancestors of each forwarding edge destination. These operation counts
    // make that complexity contract observable to scale regressions.
    size_t selection_reentry_boundary_analysis_count{0u};
    size_t selection_reentry_edge_query_count{0u};
    size_t selection_reentry_owner_query_count{0u};
    // The final selection-reentry audit is expressed as sparse dominance-
    // frontier queries. It never scans every block for every selection.
    size_t selection_reentry_audit_selection_query_count{0u};
    size_t selection_reentry_audit_frontier_query_count{0u};
    size_t selection_reentry_audit_predecessor_query_count{0u};
    size_t irreducible_region_count{0u};
    size_t unstructured_branch_count{0u};
    size_t invalid_construct_count{0u};
    size_t iteration_limit_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return restructured_loop_count != 0u ||
               restructured_if_count != 0u ||
               restructured_switch_count != 0u ||
               canonicalized_cfg_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return irreducible_region_count == 0u &&
               unstructured_branch_count == 0u &&
               invalid_construct_count == 0u &&
               iteration_limit_count == 0u;
    }
};

enum struct RestructureCFGMutationMode {
    // Preserve the input on every failure. The pass transforms and validates
    // shadow definitions, then deterministically replays the successful
    // transformation onto the original definitions to preserve object
    // identity.
    TRANSACTIONAL,
    // Transform the original definitions exactly once. Boundary verification
    // is unchanged, but a transform or output-verifier failure may leave a
    // partially rewritten input. This mode is only valid when the caller has
    // exclusive ownership and will discard the whole module on failure.
    IN_PLACE_DISCARDABLE,
};

struct RestructureCFGOptions {
    // These are safety bounds, not semantic tuning knobs. Exhaustion rejects
    // the pass; whether the input is preserved is selected by mutation_mode.
    size_t main_iteration_limit{10000u};
    size_t post_iteration_limit{64u};
    RestructureCFGMutationMode mutation_mode{
        RestructureCFGMutationMode::TRANSACTIONAL};
};

// Converts reducible plain CFG regions into structured control flow. A function
// containing an irreducible (multi-entry) cyclic SCC is rejected before any IR
// mutation and reported through irreducible_region_count. The default mutation
// mode is transactional: any late failure, including an exhausted safety bound
// or output-verifier rejection, discards the shadow CFG. The explicitly
// selected in-place mode may mutate on failure and therefore requires a
// disposable, exclusively owned input. Declaration-like callables with no body
// own no CFG and are successful no-ops.
[[nodiscard]] LUISA_XIR_API RestructureCFGInfo restructure_cfg_pass_run_on_function(
    Function *function,
    const RestructureCFGOptions &options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API RestructureCFGInfo restructure_cfg_pass_run_on_module(
    Module *module, PassReport *report = nullptr,
    const RestructureCFGOptions &options = {}) noexcept;

}// namespace luisa::compute::xir
