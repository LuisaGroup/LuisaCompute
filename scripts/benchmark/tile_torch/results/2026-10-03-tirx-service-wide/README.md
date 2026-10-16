# CUDA TIRx Service profile: twelve-case holdout

This experimental profile improved the geometric mean over fixed T128/P2 TIRx
by **18.69%** on its eleven supported cases. Against the current native Tile
recheck, the twelve-case geometric-mean gain was **2.61%**, with **four losses
larger than 5%**. This is not a default-route recommendation. All positive,
negative and unsupported results are retained.

The private selector uses the existing `ServiceExecutionCostPolicy` on actual
compiler `ReductionCandidate` facts. It adds no DSL or execution-nest primitive.
It tries six ordered existing geometries, T/P = **128/2, 64/1, 128/1, 128/4,
256/4, 256/1**, with lane elements 1, unroll 64, input cache disabled and the
existing private scalar budget 64. Exact ties keep the earlier candidate. A
legal 128/2 incumbent changes only for a score **strictly below 95%** of its
score. An illegal incumbent permits the best legal candidate. Only proved exact
mapping rejections may be skipped; unknown compiler failures are not candidates.

The model was frozen before this holdout. It was fitted to 29 unique schedules
from six earlier norm fixtures with equal fixture weighting and train-only
capacity selection. Their cost facts came from source-identical posthoc CPU
replay, not timing-time callback collection. This holdout did not refit it. The
new 256/1 schedule was absent from that training set; it was selected five times
here, once as the only legal candidate. The fitted capacity of 24 is a model
parameter, not an occupancy query. The private profile is gated to fast math,
SM89, 24 SMs, warp32, and CUDA toolkit/Driver API 13040.

## Complete results

Graph-event medians are microseconds per complete operation. `Native` gives
initial/recheck controls. `Fixed` means explicit T128/P2/L1/U64 TIRx, not the
production default. Negative deltas are faster. Torch is freshly measured in
this same cohort, using the initial native fixture; no historical denominator
is used.

| Case | Native initial/recheck | Fixed TIRx | Selected TIRx | Fresh Torch | T/P | Selected/native recheck |
|---|---:|---:|---:|---:|---:|---:|
| SUM 256×128 BF16 | 1.2491 / 1.2426 | 1.0781 | 0.9196 | 0.9866 | 128/4 | -26.00% |
| SUM 129×2048 FP16 | 1.3603 / 1.3604 | 1.4308 | 1.4385 | 1.3177 | 128/2 | +5.74% |
| SUM 3×8191 FP32 | 1.3601 / 1.3638 | 2.1504 | 1.3633 | 1.3028 | 256/1 | -0.03% |
| MAX 256×128 BF16 | 1.2068 / 1.2011 | 1.0805 | 0.9181 | 1.1426 | 128/4 | -23.56% |
| MAX 129×2048 FP16 | 1.3616 / 1.3560 | 1.4386 | 1.4461 | 1.7160 | 128/2 | +6.64% |
| MAX 3×8191 FP32 | 1.2982 / 1.2946 | 1.9780 | 1.2838 | 1.3079 | 256/1 | -0.84% |
| LayerNorm 128×1024 BF16 | 1.9086 / 1.9145 | 1.8463 | 1.8460 | 1.8183 | 128/2 | -3.58% |
| LayerNorm 17×1536 FP16 | 1.2896 / 1.2927 | 2.2753 | 1.4471 | 1.5550 | 128/1 | +11.94% |
| LayerNorm 5×1025 FP32 | 1.2132 / 1.2123 | 1.8411 | 1.3223 | 2.2361 | 256/1 | +9.08% |
| RMSNorm 32×4096 FP16 | 1.6437 / 1.6521 | 2.1435 | 1.6005 | 1.4936 | 256/1 | -3.13% |
| Softmax 129×1024 BF16 | 1.5837 / 1.5911 | 1.6382 | 1.6368 | 1.6566 | 128/2 | +2.87% |
| Softmax 3×8192 FP32 | 2.1113 / 2.1156 | unsupported | 2.0756 | 2.9309 | 256/1 | -1.89% |

The unsupported fixed softmax case has no timing, source or correctness result
and is not a zero-time observation. All twelve profile choices passed. Counts
are **47 native runs + 12 Torch runs, 413 primary samples**, and **72 actual CPU
candidate attempts: 64 legal, eight exact mapping rejections**. The geometric
mean selected/Torch ratio is 0.90365; selected/native recheck is 0.97389, and
selected/fixed TIRx is 0.81309 on eleven supported pairs. Native control drift
was -0.515% to +0.511%. Small differences are not evidence of robust gains.

Seven cases tie exactly between the same-two-warps-per-row schedules 128/2,
64/1 and 256/4. Four have identical values for all six Service features; no
coefficient refit of this formula can separate them. The other three ties rely
on the fitted zero program-byte coefficient and saturated concurrency feature.
The model ranks TIRx schedules; it does not price the native Tile route. Thus a
large improvement over fixed TIRx can still lose to native, as both small
LayerNorm cases show. No operation-name exception or post-hoc threshold change
was introduced.

## Validation and measurement scope

The serial stages were native+fresh Torch, fixed TIRx, selected TIRx, and native
recheck. Each admitted route retained seven samples, a 100 ms target, at least
500 ms warmup and graphs of 100 operations, with four host threads and affinity
mask 0x15400. Adaptive replay event spans and denominators are bundled. These
are graph stream spans per complete operation, not isolated instruction timing.
The 80% target floor applies to the final calibration span; later sample spans
may fluctuate below that floor and are retained without filtering.

Original inputs, complete FP64 references and per-element error bounds are
identical across routes, apart from the declared lowering label. The closed
summary re-read every saved logical output against that original oracle.
Runtime reports checked all readonly inputs, guards and finite outputs; the
CPU summary did not reread guard allocations. Fresh Torch passed before/after
compiled checks and saved-output validation. No tolerance was widened.

Both TIRx routes checked the actual CUfunction, entry, full grid/block and
buffer-binding permutation/final pointers for **2,300 graph nodes** in total.
Native controls have validated graph100 timing and numerical checks, but no
private per-node Driver observation. The selected Runtime CUDA source and cost
callback matched its CPU candidate, and candidate 0 matched the separately
measured fixed source when legal. The public packet retains those observations
and original hashes; it does not bundle the graph CSVs or CUDA source files.

Queried registers/static-shared/local/max-threads are saved Driver attributes,
not occupancy or spill-traffic measurements. All admitted TIRx local-byte
reports were zero. The selected RMS case changed 80 to 39 registers relative to
fixed; the selected LayerNorm 17 case improved strongly while retaining 40
registers and 80 shared bytes. These observations do not establish causality.

Fast math is explicitly enabled throughout, but route implementations need not
use identical arithmetic. TIRx reduction merges preserve subnormals through
explicit PTX helpers; its elementwise code uses the requested CUDA fast policy.
The original oracle is the correctness contract, not bitwise equality between
reduction trees or equivalence to a Torch algorithm such as Welford.

## Search and startup costs

All twelve `frontend_search_ms` values are retained: **45.75–310.39 ms**, median
**162.03 ms**. They time the bounded six-candidate CPU frontend search/selection,
not six NVRTC compilations. Only the chosen option proceeds through Runtime
shader compilation; its original `compile_ms` is saved separately. Runtime may
use its existing compiler cache, so these fields do not prove actual NVRTC
invocation counts. The old `cold_ms` is the first dispatch, not total startup.
Search time is excluded from both old fields and all reported GPU sample ratios.
CPU selection and extra source-receipt I/O therefore have costs beyond the GPU
figures; this packet makes no end-to-end startup improvement claim.

## Reproduce the compact projection

```sh
python reproduce.py
```

The standard-library script checks bundled file digests, all 413 samples and
their event-span normalization, medians/ratios/control drift, all 72 candidate
records, frozen Service score arithmetic and stable selection rules. It checks
recorded fixture hashes, reported correctness counts, source/ABI identity and
launch/resource descriptors. It **does not run GPU code, compile, refit, read
the omitted tensors/source/graph CSVs, or independently reproduce numerical
correctness**. These are original validated evidence receipts, not a new run.

`evidence.json` is a portable, compact projection; its digest in `index.json`
is distinct from the original closed summary SHA256
`93c72c8919d9b4694c7be5b667f493d314bb78250c1ff94f3563ce398b3a0945`.
The frozen training analysis SHA256 is
`fe7e04b6e5f5c9337b656615fc497d93854d340e6d800316f6cffcee101a6b01`.
Repository-relative references locate unbundled evidence; their hashes are
provenance, not assertions that those files are included. No large compiler
cache, tensor, binary or unrelated historical timing is copied into this packet.
