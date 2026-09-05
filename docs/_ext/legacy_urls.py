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
    app.connect("html-collect-pages", collect_pages)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
