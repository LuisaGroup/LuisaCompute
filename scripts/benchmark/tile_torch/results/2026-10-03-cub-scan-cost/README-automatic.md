# Actual automatic CUB scan cost measurements — 2026-10-03

This is a separate dataset from `observations.json` and its explicit-recipe experiments. It measures the implemented optional factory search and final guarded dispatch. It does not relabel any earlier measurement as automatic.

Each case retains seven graph samples for fresh default, automatic, default recheck, and a new default-stage Torch run. Each round uses its own Torch denominator. All cases, original fallbacks, parity mismatches and regressions remain included.

| Cohort | Rows × width | Type | Installed T | Guard selected | Default µs | Auto µs | Recheck µs | Fresh Torch µs | Auto/Torch | Search ms |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| heldout | 3 × 2048 | fp16 | 256 | True | 1.2734 | 0.9849 | 1.2586 | 0.9699 | 1.0155 | 4177.057 |
| heldout | 3 × 2048 | bf16 | 256 | True | 1.2598 | 0.9901 | 1.2639 | 1.4081 | 0.7031 | 4891.988 |
| heldout | 65 × 4096 | fp16 | 512 | True | 3.7070 | 1.9004 | 3.7009 | 4.1829 | 0.4543 | 6739.154 |
| heldout | 65 × 4096 | bf16 | 512 | True | 3.7057 | 1.9287 | 3.7122 | 1.8605 | 1.0367 | 7273.120 |
| heldout | 129 × 8192 | fp16 | 256 | True | 12.3249 | 4.1116 | 12.3570 | 6.1743 | 0.6659 | 14326.421 |
| heldout | 129 × 8192 | bf16 | 256 | True | 12.3343 | 4.1980 | 12.3702 | 6.2082 | 0.6762 | 12085.562 |
| heldout | 256 × 16384 | fp16 | 128 | True | 66.0045 | 12.5298 | 66.2285 | 34.9901 | 0.3581 | 14869.894 |
| heldout | 256 × 16384 | bf16 | 128 | True | 65.9949 | 33.5357 | 66.2195 | 35.3834 | 0.9478 | 9782.018 |

heldout: geometry-equal automatic/original ratios **0.453097 / 0.452872** against initial/recheck; automatic/fresh-Torch **0.690023**. Engineering status: `passed_measured_engineering_gates`.

**Cases slower than their same-round Torch denominator:** heldout [3, 2048] fp16: 1.015475; heldout [65, 4096] bf16: 1.036662

## Preserved diagnostics

- The formal BF16 256 x 16384 automatic observation is about 33.54 us, versus about 13 us in prior explicit rounds. It is retained unchanged. A separate closed fixed/automatic repetition moved the slow state to FP16 despite identical source, key, resources and choice; see README-outlier.md and automatic-outlier.json. The cause remains unresolved and stable automatic/fixed runtime parity is not established.
- FP16 3 x 2048 and BF16 65 x 4096 remain slower than their fresh Torch denominator. Passing measured engineering gates against original Tile does not mean beating Torch on every case or resolving the separate runtime instability.

## Scope and reproduction

Run `python reproduce_automatic.py` in this directory. It checks `manifest-automatic.json`, all retained medians and event-span denominators, frozen feature/score/choice calculations, candidate/winner source and key bindings, fresh denominators, both-control ratios, every regression and engineering gates. It uses only the Python standard library, never fits or invokes a compiler/GPU.

The package retains 24 native and 8 Torch observations (224 raw graph samples), plus 32 per-recipe attempt records. Cohorts remain separate; eligibility has no speedup requirement.

Upstream validators rechecked the complete unchanged FP64/per-element oracle, all saved outputs, guards/read-only, generated sources, candidate SHA256 values, compiler identities, and full-build/final cleanup gates. This compact package preserves those receipts and does not contain tensor payloads or rerun the GPU oracle. Hashes of omitted artifacts are provenance, not reconstructable data.

The installed winner and actual command BufferView guard prediction are separate. Receipts are host-selector evidence, not a raw Driver trace. Resources describe queried ordinary-CUDA functions, not physical Tile workers. Historical candidate-source/resource parity is checked without pooling historical timings.

Search time and per-candidate compiler-method times remain separate from dispatch. Compiler calls can hit the PTX LRU: counts are not NVRTC cache misses. Default, automatic and recheck total compile time also includes their respective original compilation/setup. These cold observations are not part of the graph dispatch ratio.

The frozen seven coefficients, strict 0.95 score rule and profile identity remain unchanged. Engineering limits are geometry-equal ratio ≤0.95 versus both controls, no selected regression >3%, and control drift ≤3%; sample-extrema threshold overlap is descriptive and inconclusive, not a confidence interval. A single queue does not establish cross-device or future-run guarantees.

Original profile ID: `67127e819f80a395aeecae55cd999c8e3f8b1bcf894fdf0672c58611cca87f66`. Original profile bytes SHA: `d38f317cb43b1e646fcdb959e30ea89675b79722508d973ee037609db24d9186`. `original_summary_bytes` identifies the unprojected local validator output; compact redacted bytes have distinct hashes in `manifest-automatic.json`. The complete implementation-map digest likewise identifies the original map, not its compact subset. Source64/compile64 keys remain distinct from SHA256.

All original explicit-recipe files are untouched. The original orchestration preflight failure, if present, is recorded separately. Any later diagnostic must use a new record and cannot replace this queue’s samples.
