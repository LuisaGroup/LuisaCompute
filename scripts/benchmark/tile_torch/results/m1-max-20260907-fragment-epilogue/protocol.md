# Closed matrix epilogue protocol

Question: can a same-owner scalar DAG execute in MPP destination fragments
without changing its mathematical result, and where is this candidate faster?
The primary artifact is the existing Sphinx internals/performance Tile section.
No separate dashboard or replacement documentation hierarchy is created.

The two initial `baseline/` and `fused/` directories are exploratory auto-worker
screens, not paired acceptance evidence. They retain all 18 attempted cases
per compiler, including the three ragged rejections. Compilers and benchmark
features were still being developed during this screen. Never infer a causal
speedup from these sequential sessions or treat discarded storage as timing.

The replay freezes a common schedule for every graph and shape: 64×64 output
blocks, nominal K=4096, one pipeline stage, 256 workers, default cost coefficients,
and MPP read-only views. The 256-worker constraint is a diagnostic common legal
binding after the auto-1024 ragged rejection, not a promoted universal rule.
Use the same final binary/compiler with only `fuse_matrix_epilogues` changed.
Four rounds cover both variant and native/Torch order factors. Each run has
seven samples, 20 ms host sample windows and 100 ms warmup, plus separately
instrumented compute-pass and no-counter command-buffer timing phases.
Fresh native processes retain JIT/setup separately from warm timing.

The graphs are GEMM, clamp(0.125 GEMM + 0.25, min=0), and tanh-GELU of the same
affine transform. Inputs are matched FP32; every complete output is compared
to FP64 matmul plus the specified transform, at atol=rtol=1e-4. Torch uses
preallocated output and intermediate buffers: `mm.out`, in-place scale/shift,
then activation.out. It is eager Torch, not compiled fused Torch or MPSGraph.
Plain MPS multiplication is not a matched fused-graph baseline and is not
substituted for one. Nondyadic, nonzero-initializer, tail, transposed-destination,
observation and manual-memory cases are covered by native correctness tests.

Primary timing: no-counter command-buffer GPU batch interval, including GPU
work and intra-buffer gaps. It is not isolated kernel time. Counter-based
compute intervals are diagnostics because instrumentation can perturb both
paths. Preserve batched E2E, single-dispatch E2E and all raw samples separately.
Summaries use within-round medians and medians/ranges of paired ratios, not
confidence intervals or ratios of pooled means. No failed/slow row is dropped.
Shape/operator lookup tables are descriptive, never inputs to production
matching, cost coefficients or solver decisions.

Controls compare the frozen pre-extension stack with the final default-off
path, including matrix epilogues and nonmatrix Metal/CPU programs. Metal source
must match exactly. CPU comparison may normalize only bijective TVM TBAA
allocation-address labels, retaining the alias graph and width/offset suffixes.
Check phases retain complete logs and nonzero assertion counts. The existing
user edit `metal::mem_flags(3)` → `metal::mem_flags(2)` is present in both tested
Luisa stacks but excluded from this change; two existing source-string suites
are known to reject it. Do not describe those suites as green.

The MPP element interface is optional and the new planner candidate defaults
off until general profitability evidence or measured JIT selection supports
it. CPU/SIMD and independent native-MPP gain no new emission from this change.
Bias/residual memory operands and free variables are outside the first closed
DAG contract. Generalization means proved expression/access rules, not universal
hardware or shape coverage.

Report plan: answer/limits in current status; exact per-shape timings and nearby
interpretation in results; proof/lowering diagram in internals/matrix; raw
samples, sources, hashes, tests and failed probes here. Exact tables support
shape lookup and mixed timing units; no aggregate leaderboard or decorative
chart is planned. QA must check source equivalence, arithmetic, retained errors,
local links and rendered wide/narrow Sphinx pages before handoff.
