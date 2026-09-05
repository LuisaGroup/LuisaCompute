"""Keep published URLs working without keeping duplicate source documents."""

from sphinx.errors import ExtensionError


_ROOT = "source/"
_TILE = _ROOT + "tile/"
_COMPILER = _ROOT + "internals/tile/"
_PERF = _ROOT + "performance/tile/"

REDIRECTS = {
    _ROOT + "tile_programming_design": _TILE + "design",
    _ROOT + "tile_programming_poc_kernels": _TILE + "kernels",
    _ROOT + "tile_execution_planner": _COMPILER + "planner",
    _ROOT + "tile_native_runtime": _COMPILER + "runtime",
    _ROOT + "tile_xir_design": _COMPILER + "xir",
    _ROOT + "tile_status_report": _PERF + "index",
    _ROOT + "tile_tirx_reduction_report": _COMPILER + "reductions",
}

# These are the section ids from the published Sphinx HTML, not GitHub's
# numeric-heading slugs. Split pages retain their original section ids.
FRAGMENTS = {
    _ROOT + "tile_status_report": {
        "technical-summary": _PERF + "checkpoints",
        "reading-map-and-scope": _PERF + "checkpoints",
        "architecture-decision-ledger": _COMPILER + "decisions",
        "what-is-implemented-and-what-remains-design": _PERF + "implementation",
        "next-work-and-acceptance-criteria": _PERF + "implementation",
        **dict.fromkeys([
            "correctness-common-llm-operators-now-use-both-bridges",
            "five-failure-investigations-changed-the-implementation",
            "llvm-coexistence-is-a-build-runtime-constraint",
            "phi-transfers-must-be-simultaneous",
            "bounds-proofs-belong-before-schedule-expansion",
            "distributed-initialization-does-not-create-shared-private-storage",
            "shared-ssa-must-survive-until-target-resource-planning",
        ], _PERF + "validation"),
        **dict.fromkeys([
            "performance-preserve-the-measurement-basis",
            "metal-subgroup-reductions-close-the-measured-normalization-defect",
            "new-xir-simd-planner-pilot",
            "balanced-metal-evidence-mpp-cost-v2-closes-this-gemm-cohort",
            "cpu-tirx-reference-gaps-and-proved-provider-realizations",
            "whole-gemm-cblas-realization",
            "shared-ssa-and-reduction-realization",
            "target-specific-residual-layernorm-materialization",
        ], _PERF + "results"),
    },
    _ROOT + "tile_tirx_reduction_report": dict.fromkeys([
        "outcome-at-a-glance",
        "performance-evidence",
        "base-reductions-versus-eager-pytorch",
        "layernorm-and-cross-entropy-versus-eager-pytorch",
        "rmsnorm-causal-a-b-against-the-old-lowering",
        "layernorm-and-cross-entropy-causal-a-b",
        "fused-residual-layernorm-and-materialization-choice",
        "target-aware-widths-gpu-and-dispatch-acceptance",
        "budgeted-immutable-input-reuse",
        "joint-resource-and-execution-mapping",
        "whole-launch-policy-shape-held-out-gains-and-small-case-failures",
        "tail-packs-a-structural-repair-after-width-reuse-ablation",
        "what-this-closes-and-what-remains",
    ], _PERF + "reductions"),
}

RENAMED_FRAGMENTS = {
    _ROOT + "tile_programming_design": {
        "luisa-tile-dsl-a-from-scratch-design": "tile-language-and-layout",
    },
    _ROOT + "tile_programming_poc_kernels": {
        "luisa-tile-dsl-executable-kernel-gallery": "tile-kernel-examples",
    },
    _ROOT + "tile_status_report": {
        "technical-summary": "recorded-checkpoints",
        "tile-programming-implementation-and-evidence-report": "tile-status-and-performance",
    },
    _ROOT + "tile_tirx_reduction_report": {
        "tileir-to-tirx-metal-reductions-design-and-evidence": "tirx-metal-reduction-lowering",
    },
}


# Topic splits keep the overview URLs, while old section links go directly
# to the single page that now owns their content. Keep original Docutils ids.
SPLIT_FRAGMENTS = {
    "source/tile/design": {
        **dict.fromkeys([
            "execution-structure-first",
            "the-semantic-skeleton",
            "the-precise-relationship-to-halide",
            "dimensions-and-spaces",
            "a-small-complete-structured-region-calculus",
            "execution-is-a-nest-not-yet-a-layout",
            "logical-anchor-and-execution-frontier",
            "execution-transform-calculus",
        ], "source/tile/execution"),
        **dict.fromkeys([
            "the-canonical-layout-algebra",
            "decision",
            "typed-map",
            "algebraic-operators",
            "replication-and-non-injectivity",
            "bounds-and-data-dependent-indexing",
            "what-complete-means",
            "compatibility-embeddings",
            "proof-discipline-and-algebra-laws",
            "value-distribution-is-a-layout-not-another-algebra",
        ], "source/tile/layouts"),
        **dict.fromkeys([
            "elementwise-operators-lift-directly-to-tiles",
            "mma-is-a-portable-value-primitive",
            "reduction-is-a-structured-algebraic-region",
            "ordering-and-selection-stay-logical-tile-operations",
        ], "source/tile/values"),
        **dict.fromkeys([
            "views-values-and-addressable-memory",
            "three-surface-objects",
            "memory-ownership",
            "implemented-explicit-memory-path",
            "the-execution-to-memory-equation",
        ], "source/tile/memory"),
        **dict.fromkeys([
            "pipeline-is-a-temporal-producer-consumer-nest",
            "stage-boundaries-are-lexical",
            "scheduling-and-versioning",
        ], "source/tile/pipeline"),
        **dict.fromkeys([
            "direct-assignment-and-hidden-ssa-plumbing",
            "surface-rule",
            "addressable-storage-uses-explicit-effects",
            "c-staging-and-jit",
            "one-scoped-builder-no-builder-prefixes",
            "ordinary-configuration-creates-variants",
            "target-schedules",
        ], "source/tile/staging"),
        **dict.fromkeys([
            "tileir-as-a-thin-but-transformable-ir",
            "in-memory-structure",
            "ownership-and-mutation-contract",
            "minimal-operations",
            "forms-and-invariants",
            "essential-analyses",
            "capture-algorithm",
            "shared-ssa-preserves-a-resource-planning-choice",
            "required-verifier-invariants",
        ], "source/internals/tile/ir"),
        **dict.fromkeys([
            "compiler-bridges-and-native-backends",
            "boundary",
            "layout-bridge",
            "execution-bridge",
            "guarded-native-metal-matrix-realization",
            "implemented-native-software-prefetch-path",
            "pipeline-and-memory-bridge",
            "bootstrap-lowering-path",
            "target-catalog",
        ], "source/internals/tile/lowering"),
        **dict.fromkeys([
            "minimal-implementation-plan",
            "phase-a-algebra-and-ir-skeleton",
            "phase-b-elegant-c-capture",
            "phase-c-scheduling-core",
            "phase-d-tvm-bootstrap",
            "phase-e-native-optimization",
            "final-decisions",
        ], "source/internals/tile/decisions"),
    },
    "source/internals/tile/planner": {
        **dict.fromkeys([
            "native-mpp-experiment-operation-scope-is-not-launch-size",
            "implemented-matrix-mapping-family",
            "realization-changes",
            "direct-accumulator-output",
            "implemented-relative-work-models",
            "simd-group-reference-basis",
            "metal-mpp-memory-v2-basis",
            "cooperative-copy-batching",
            "dependence-aware-group-synchronization",
            "independent-subgroup-programs-legality-is-not-profitability",
            "implemented-solver-enumeration-plus-pareto-dynamic-programming",
        ], "source/internals/tile/matrix"),
        **dict.fromkeys([
            "llvm-compiler-temporary-storage-realization",
            "cartesian-cpu-register-packs",
            "cpu-immutable-input-expressions",
            "full-vector-guard-specialization",
            "cpu-root-launch-cost-guard",
            "shared-tile-ssa-target-materialization-and-cpu-provider-atoms",
        ], "source/internals/tile/cpu"),
        **dict.fromkeys([
            "backend-owned-execution-cost-policy",
            "whole-launch-service-policy-and-shape-held-out-check",
        ], "source/internals/tile/cost-policy"),
    },
}

# The earlier flat documentation URLs skip the intermediate landing page.
FRAGMENTS[_ROOT + "tile_programming_design"] = SPLIT_FRAGMENTS[_TILE + "design"]
FRAGMENTS[_ROOT + "tile_execution_planner"] = SPLIT_FRAGMENTS[_COMPILER + "planner"]


def page_context(app, pagename, templatename, context, doctree):
    fragments = SPLIT_FRAGMENTS.get(pagename, {})
    context["moved_fragment_urls"] = {
        anchor: context["pathto"](target)
        for anchor, target in fragments.items()
    }


def check_structure(app, env):
    parents = {page: [] for page in env.found_docs}
    for parent, children in env.toctree_includes.items():
        for child in children:
            if child in parents:
                parents[child].append(parent)
    errors = []
    for page, owners in sorted(parents.items()):
        expected = 0 if page == app.config.root_doc else 1
        if len(owners) != expected:
            errors.append(f"{page}: expected {expected} owning toctree(s), got {sorted(owners)}")
    reachable = set()
    pending = [app.config.root_doc]
    while pending:
        page = pending.pop()
        if page not in reachable:
            reachable.add(page)
            pending.extend(env.toctree_includes.get(page, []))
    for page in sorted(env.found_docs - reachable):
        errors.append(f"{page}: unreachable from the root toctree")
    for source, fragments in SPLIT_FRAGMENTS.items():
        for target in {source, *fragments.values()}:
            if target not in env.found_docs:
                errors.append(f"Missing split-page target: {source} -> {target}")
    if errors:
        raise ExtensionError("Documentation ownership errors:\n" + "\n".join(errors))


def collect_pages(app):
    for old, destination in REDIRECTS.items():
        fragments = FRAGMENTS.get(old, {})
        for target in {destination, *fragments.values()}:
            if target not in app.env.found_docs:
                raise ExtensionError(f"Missing redirect target: {old} -> {target}")
        yield old, {
            "destination": destination,
            "fragments": fragments,
            "renamed_fragments": RENAMED_FRAGMENTS.get(old, {}),
            "destination_titles": {
                page: app.env.titles[page].astext()
                for page in {destination, *fragments.values()}
            },
        }, "redirect.html"


def setup(app):
    app.connect("env-check-consistency", check_structure)
    app.connect("html-page-context", page_context)
    app.connect("html-collect-pages", collect_pages)
    return {"version": "2", "parallel_read_safe": True, "parallel_write_safe": True}
