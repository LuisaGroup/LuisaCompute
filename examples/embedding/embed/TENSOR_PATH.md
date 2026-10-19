# bge-m3 embedding engine — tensor-path optimization

Engineering doc for the tensor / cooperative-vector compute paths of the
bge-m3 inference engine in `examples/embedding/embed/`. Sources of truth:
the plan (`tensor_optimize.md` at the repo root), the selection
infrastructure (`tensor_path.{h,cpp}`), the kernel contracts
(`kernels/kernels_coop.h`, `kernels/kernels_norm_vec.h`,
`kernels/kernels_embed_vec.h`, `kernels/kernels_heads_vec.h`), the
integration (`executor.cpp`, `session.cpp`), and the per-phase implementation
reports under `.kimix_cache/embed_reports/done_*.md`. This document describes
what is actually shipped and measured; anything not verified is labeled.

---

## 1. Feature summary

The engine has four compute paths for the projection / elementwise hot spots
(plan §2.1, `tensor_path.h`):

| Path | Name | What it is |
|---|---|---|
| `Legacy` | `legacy` | Verbatim current kernel selection. **The kill switch (plan §9): byte-identical dispatch sequence, compiles none of the new shaders, always available on every backend.** |
| `Vector` | `vector` | Plain float4 / DOT / FMA vector ops: the `*_vec4` bandwidth/compute twins plus the `rbv_*` vector register-blocked GEMM family. Eligible on DX / VK / CUDA. |
| `Coop` | `coop` | Cooperative-vector matvec projections (`kernels_coop.h`). **DX and VK ONLY.** An explicit request the device cannot satisfy is a hard startup error — never a silent fallback (plan §2.1). |
| `Auto` | `auto` | `coop` when the full capability query passes, `vector` otherwise, with the fallback reason logged. |

**Default = `legacy`.** `LUISA_EMBED_TENSOR_PATH` unset resolves to
`Legacy` — the new paths are strictly opt-in (plan §0.2), so default startup
cost and bit-level behavior are unchanged. Setting the env to a non-legacy
value makes the Executor compile the extra shaders once per Executor and swap
the per-site dispatch behind shape guards (§4/§5 below).

Kill-switch semantics: an unknown `LUISA_EMBED_TENSOR_PATH` value logs
`LUISA_WARNING` and falls back to `legacy` — the engine must never fail to
start over a config typo. The same warn-once-and-default idiom applies to
`LUISA_EMBED_COOP_CONFIG` and (pre-existing) `LUISA_EMBED_GEMM_CONFIG`.

Path selection is resolved from the process environment exactly once per
process (thread-safe magic-static, the same idiom as `embed_p1_mode()`), and
the resolved choice is always logged (§4).

---

## 2. Environment variables

| Variable | Values | Default | Semantics |
|---|---|---|---|
| `LUISA_EMBED_TENSOR_PATH` | `legacy` \| `vector` \| `coop` \| `auto` | unset → `legacy` | Case-insensitive; surrounding whitespace (incl. trailing CR from `setx`) trimmed; exact lowercase preferred. Unknown value → `LUISA_WARNING` + `legacy` (never a startup failure). |
| `LUISA_EMBED_COOP_CONFIG` | exact coop variant name (§6 table) | `coop_k64_n16_t1` | Selects the coop matvec variant by exact name, mirroring the whole-variant arm of `LUISA_EMBED_GEMM_CONFIG`. Unknown name → `LUISA_WARNING` + shipped default `coop_k64_n16_t1`. |

Pre-existing knobs that compose with the tensor path (each keeps its own
semantics; only interactions are listed here):

| Variable | Owner | Interaction with the tensor path |
|---|---|---|
| `LUISA_EMBED_GEMM_CONFIG` (+ `LUISA_EMBED_GEMM_TILE/BLOCK/KSLAB/REGTILE`) | `kernels_gemm.cpp` | Chooses the register-tile GEMM variant for the `legacy`/`vector` paths (the `rbv_*` vec4 family enters through the same table). On the `coop` path the projection sites are replaced, so the resolved GEMM config no longer drives them; the executor log line reports the coop variant name instead. |
| `LUISA_EMBED_FLASH_CONFIG` (+ `LUISA_EMBED_FLASH_BLOCK/_TILE/_REG`) | `kernels_attention.cpp` | Orthogonal: the attention path is untouched by the tensor-path integration (Phase 3, §8), so the flash knobs behave identically under every tensor path. |
| `LUISA_EMBED_WEIGHTS` | `executor.cpp` WeightStore | `fp32|bf16|fp16`, **fail-closed** on unknown values (deliberate contrast with the tensor-path parse). Composition rule: the coop matvec reads plain fp32 weights only. When the WeightStore holds **packed half/bf16** GEMM weights, a `coop`-resolved path is **demoted to `vector` before any coop shader is compiled**, with a loud `LUISA_WARNING` and the demotion recorded in the selection log's `reason` field. Set `LUISA_EMBED_WEIGHTS=fp32` (A/B convert-on-upload mode) to keep coop on a packed package. |

---

## 3. Capability matrix (per backend)

| Backend | vector path | coop path | Requirements / observed state |
|---|---|---|---|
| DX | yes | gated | Coop kernels compile to SM 6.9, which requires `DirectXDeviceConfigExt::UseExperimental()` set at **device creation** (before any device is created — the gate is process-global) **and** runtime acceptance (`ExperimentalFeaturesEnabled()` reading back true). On this dev machine `D3D12EnableExperimentalFeatures` fails with **`0x80004002`** ("Failed to enable D3D12 experimental feature 'ExperimentalShaderModels'… Continuing without experimental features"), so coop is unavailable from the DX side here. |
| VK | yes | gated | Requires `VK_NV_cooperative_vector` + `VK_EXT_shader_replicated_composites` + the Vulkan memory model **and** an all-FP32 component config. Observed on RTX 5070 Ti Laptop: the extension is present but the driver reports **"FP32 float config NOT supported (quantized configs only)"**. Additionally a VK codegen bug was hit: the coop `mul_add` lowering emits `OpCooperativeVectorMatrixMulAddNV` whose bias operand SPIR-V validation rejects ("must be an array type"), and the JIT compile surface **aborts** (`LUISA_ERROR` is `[[noreturn]]`). All coop compile sites therefore gate first through the non-aborting AOT `Device::compile_to()`; on this machine that returns false → recorded no-go. |
| CUDA | yes | **never** | Plan §0.3/§2: CUDA never selects, simulates, or silently degrades the coop path. Enforced three ways: the backend allowlist in `query_cooperative_support()` refuses anything but `"dx"`/`"vk"` without touching the compiler; the AST/generated-source negative checks (tests assert `use_cooperative_operations() == false` on every tensor-path kernel); and an explicit `coop` request on CUDA is a startup error. |
| fallback | vector only | never | CPU numeric oracle; coop is refused by the same allowlist. |

The shipped probe surface: there is **no public `Device::query()` capability
key** for cooperative vectors on either backend, so the decisive check is the
try-compile-and-run pattern (`query_cooperative_support()` in
`tensor_path.cpp`, §4). Caveat documented there: Luisa fatal paths
(`LUISA_ERROR`) abort rather than throw, so a VK-native-route coop rejection
mid-compile cannot be contained from host code — the caller decides whether to
reach the probe at all for a given device configuration.

---

## 4. Path resolution semantics and logs

`resolve_tensor_path(device, requested)`:

- `legacy` → `legacy` (no capability query at all).
- `vector` → `vector` (no coop query — CUDA-safe by construction).
- `coop` → `coop` **only if** the support query passed; otherwise
  `LUISA_ERROR_WITH_LOCATION("LUISA_EMBED_TENSOR_PATH=coop requested but
  {}; falling back is NOT allowed for explicit coop", reason)` — a clear
  startup abort, never a silent wrong-result path.
- `auto` → `coop` when the query passed, else `vector` with the reason logged
  at INFO. Never returns `coop` when support is absent.

The cooperative query chains the plan §2.2 gates: (1) backend allowlist
(dx/vk), (2) feature-enabled state — folded into (3) because no public query
key exists, (3) a **real compile+run probe**: `make_coop_probe_kernel()`
(K=16, N=8, fp32 matvec+bias, one thread, `dispatch(32)`) executed with the
plan §5.1 C0 **non-uniform, layout-sensitive matrix**
`W[k,n] = 0.01k − 0.03n + ((k+3n)%7)`, mixed-sign/scale `x`, bias spanning
−1.0…−0.3, compared per element against an fp64 CPU reference under
`abs(diff) <= 1e-3 * max(1, |ref|)` — and (4) layout validation, implied by
the probe aliasing a canonical row-major `[K,N]` fp32 buffer (an identity
matrix would let a wrong layout pass; hence the C0 recipe). The result,
including its failure reason, is cached per-process; the probe self-contains
its stream (create → commit → synchronize → RAII destroy) and pays one small
AOT compile + one dispatch on first use.

Log lines (all `LUISA_INFO` unless noted; resolved once per process / per
Executor):

```text
embedding tensor path: LUISA_EMBED_TENSOR_PATH unset -> 'legacy' (source=default; new paths are opt-in).
embedding tensor path: requested 'vector' (source=env).
embedding tensor path: backend='dx', requested='auto', resolved='vector', reason='vector'
embedding tensor path: auto -> 'vector' (cooperative unavailable: <reason>).
embedding coop probe: passed on backend 'vk' (K=16, N=8, fp32 matvec+bias vs fp64 reference, gate 1e-3*max(1,|ref|), worst abs err <w>).
embedding tensor path: resolved='coop', gemm_variant='coop_k64_n16_t1', weight_format='fp32'.
```

Fields of the selection line (plan §2.1 logging requirement): `backend` =
`device.backend_name()`; `requested` = the env-parsed path; `resolved` = the
actual path after capability gating (and after the packed-weights demotion in
§2, whose `reason` then reads `coop demoted to vector: packed '<fmt>' weights
cannot enter the coop path`); `reason` = the resolved path name in the normal
case. The summary line's `gemm_variant` field carries the coop variant name
on the coop path and the resolved register-GEMM config + source string
otherwise; `weight_format` is the WeightStore's device mode. The Session
ready line additionally echoes the requested path name.

---

## 5. Numerical contracts (vec4 / vector path)

- **Summation order changes are intra-thread only.** Widening loads to float4
  changes each thread's partial set (thread `tx` now owns vec4 lanes
  `{tx, tx+bs4, …}`), and every vec4→scalar horizontal fold uses the
  **pinned order `(a.x+a.y)+(a.z+a.w)` — never `dot()`** (this keeps the
  shipped variants the `vec4_io` flavor; a DOT/FMA-reassociation split is not
  implemented, §8). The block reduction (shared-memory halving tree, barrier
  discipline, flattened-block-id geometry) is mirrored bit-faithfully from the
  legacy kernels. `embed_lookup_vec4` even mirrors the legacy pass-3
  expression order `((x-mean)*inv)*gamma+beta`.
- **Legacy-vs-vector comparisons use the existing tolerance gates**, never
  bit-identity: LN / embed / heads at the 1e-4 class (layernorm 1e-4,
  sparse head `1e-5 + 1e-5·|ref|`, colbert `<=1e-3` rel — each file's
  pre-existing gate), GEMM at the hybrid `abs(diff) <= 1e-3 * max(1,|ref|)`,
  and **`residual_add_vec4` is bit-exact** (pure elementwise fp32 add; the
  test gate is a differing-bit-pattern count of 0).
- **Same-path determinism is preserved**: the lane order and tree order are
  fixed at emission, so repeated execution of the same path is
  deterministic (plan §8.2 category 2). The legacy kernels remain the
  bit-level regression oracle (category 1).
- **Vec4 host contract** (`kernels_norm_vec.h` / `kernels_embed_vec.h` /
  `kernels_heads_vec.h`): the caller MUST guarantee (1) 16-byte alignment of
  every bound `Buffer<float4>` element-0 address — the executor binds
  zero-copy `BufferView<float>::as<float4>()` views only at offset 0 or at
  element offsets that are multiples of 4, which are multiples of 16 bytes
  whenever the size guard holds; and (2) `H % 4 == 0` (row length) — plus
  `N % 4 == 0` for `colbert_proj_vec4` and `head_dim % 4 == 0` for the
  attention vec4 twins. **The kernels do not self-guard.** Every executor
  record site is wrapped in a per-site `H % 4 == 0`-style predicate: a ragged
  shape stays on the legacy scalar kernel (no tail fast/slow split is
  implemented — the engine shapes H=1024, I=4096, head_dim=64 are all exact
  multiples of 4). The residual+LN pair switches as one coupled unit so no
  half-vec route can read an unwritten buffer slot.

---

## 6. Coop kernel contract (`kernels_coop.h` / `.cpp`)

One thread per token group computes `y[N] = W[K,N] @ x[K] + bias[N]` with the
device cooperative-vector matvec ops, replacing the tiled GEMM at the four
projection sites (qkv, attn.out, ffn.up → bias+GELU-erf, ffn.down).

- **fp32 weights only.** Packed half/bf16 must not enter the coop path
  (plan §1.3/§5.5); the factory asserts `GemmWeightFormat::Fp32` at the
  visible boundary and the executor demotes coop→vector earlier (§2).
- **`K % kslab == 0` and `N % ntile == 0` are guaranteed by the caller**; the
  kernel carries `device_assert` backstops. Ragged K/N fall back to the
  regular GEMM on the host — no tail predicates are emitted for coop. Ragged
  M is handled by a per-token guard.
- **Canonical row-major `[K,N]` weight layout is validated, not assumed.**
  DX lowering uses `MUL_OPTIMAL` and VK uses `InferencingOptimalNV`; neither
  is proof that a canonical `[K,N]` row-major alias works. The C0 probe
  (§4) aliases exactly that layout through the `CoopMatrixRef` byte offsets
  with a non-uniform matrix, and the executor gates on a coop variant only
  when that probe passed. If the layout is wrong, the probe fails numerically.
- **Variant table** (emission-time constants; `LUISA_EMBED_COOP_CONFIG`
  selects by exact name):

  | name | kslab | ntile | tokens_per_block (TPB) |
  |---|---|---|---|
  | `coop_k64_n16_t1` (shipped default) | 64 | 16 | 1 |
  | `coop_k64_n32_t1` | 64 | 32 | 1 |
  | `coop_k128_n32_t1` | 128 | 32 | 1 |
  | `coop_k64_n32_t2` | 64 | 32 | 2 |

- **Fixed-order slab accumulation.** Per N-tile: zero a register
  `CoopVector<float>` accumulator, loop over `K/kslab` slabs, load the x slab
  with `cooperative_vector_load`, run `cooperative_mat_mul` per slab — and
  `cooperative_mat_mul_add` (fused bias from the bias buffer at the column
  base) **on the last slab only** — then accumulate element-wise
  `acc[i] += partial[i]` in a C++-unrolled fixed order. **No
  `cooperative_vector_accumulate`** (global RMW ordering is forbidden, plan
  §1.3/§5.3) and no future-marked coop element-wise CallOps. Epilogues
  emitted: `Bias` and `BiasGeluErf` (GELU applied after the fused bias).
- Launch: block fixed at 256 (`kCoopGemmBlock`); `dispatch` counts
  **total threads** = `ceil(M / (256·TPB)) · 256` via
  `coop_gemm_dispatch_size_config`. Buffers (x `[M*K]`, weight `[K*N]`, bias
  `[N]`, y `[M*N]`, all fp32) are bound as `ByteBufferView` zero-copy aliases
  of the typed buffers; byte offsets are computed in 64-bit DSL arithmetic and
  truncated to the uint32 the ref API takes (exact within the bge-m3 envelope,
  largest offset < 64 MiB).
- Shader names bake variant + epilogue for profiling, e.g.
  `embed_coop_gemm_bias_k64_n16_t1` / `embed_coop_gemm_bias_gelu_erf_k64_n16_t1`.

---

## 7. Measured results

> **Label: DEBUG build, RTX 5070 Ti Laptop, single-machine runs — not
> release-conclusive** (plan §7.2 requires release-mode, multi-process,
> median-based gates before any promotion decision; that matrix has not been
> run, §8).

GEMM micro sweep (`benchmark_embed_gemm` on DX, shapes = the four real engine
projection geometries, correctness gates before timing):

- `rbv_64x64_b128_r84v4` (the vec4 register-blocked GEMM) **passes every
  correctness gate**: the ragged proxy shape 37×69×53 (bias + gelu, worst
  error `2.384e-06`) and the 16-point per-shape sample gate on every timed
  shape (`sample_bad=0`).
- It is **perf-parity with its scalar twin `rb_64x64_b128_r84`**, e.g.:
  - qkv, rows=4096: 11613.8 vs 11611.2 GFLOP/s (~11.6 TFLOP/s both);
  - ffn_down, rows=4096: 11077.4 vs 11065.9 GFLOP/s.
- `rbv_128x64_b256_r84v4` is **slower** (≈10 % behind the scalar twin, e.g.
  ffn_down rows=4096: 10003.1 vs 11065.9 GFLOP/s).
- Therefore the vec4 GEMM stays opt-in: no net speed gain at kernel level on
  this machine, and plan §7.3 forbids promotion without an end-to-end win.
  Note also that the executor's vector/coop paths do not select GEMM variants
  by themselves — the register-GEMM variant is always the one resolved from
  the LUISA_EMBED_GEMM_* knobs — so the vec4 GEMM is used (engine-wide) only
  when chosen explicitly, e.g. LUISA_EMBED_GEMM_CONFIG=rbv_64x64_b128_r84v4.

Kernel correctness tests (`test_embed_kernel_{layernorm,embed,heads,gemm}`
including the vec4 checks) **pass on DX and VK** (exit 0). Worst observed
numeric diffs are far inside the gates: the norm-suite aggregate gate reports
`worst max-abs-diff 5.007e-06` (at `layernorm_vec4` H=4096), and
`residual_add_vec4` is bit-exact (0 differing bit patterns across all swept
counts). The tensor microtest suite (`test_embed_tensor_micro`) passes on any
backend; the recorded DX run shows FMA and REDUCE_SUM with worst_abs 0.0 and
length_squared worst abs 3.815e-06 (each vs the test's fp64 CPU reference at
its stated gate).

Coop: **zero device numbers exist** — no coop kernel has ever executed a
timed or correctness case on this machine (§8); the probe records documented
no-go evidence on both backends (§3).

---

## 8. Known limitations / not done (honest list)

- **Attention vec4 kernels are NOT wired.** The naive QK/PV vec4 twin
  factories exist (`make_attention_scores_vec4_kernel`,
  `make_attention_pv_vec4_kernel` + their dispatch helpers,
  `kernels_attention.{h,cpp}`) but `record_encode` still issues the legacy
  `_attn_scores` / `_attn_softmax` / `_attn_pv` / flash sequence on every
  tensor path. Phase 3 isolated benchmarks are pending; the attention kernels
  have no vec4 test coverage yet, and no isolated vec4-attention benchmark
  exists yet. (Note: a vec4 flash variant was planned but is not present in
  the shipped flash table — only the scalar `legacy/shared/reg_*` families
  are emitted.)
- **PAIRS sparse-compact mode is not vectorized**: `sparse_head_compact`
  keeps the legacy scalar kernel in v1 (no vec4 compact variant exists);
  WINDOW/LEGACY modes use `_sparse_head_vec4` under the usual guard.
- **Coop has never produced device numbers.** On the dev machine it is a
  validated no-go with evidence: DX — `D3D12EnableExperimentalFeatures` fails
  with `0x80004002`, so SM 6.9 is unavailable; VK — the extension is present
  but the all-FP32 component config is not ("quantized configs only"), and
  the coop `mul_add` lowering additionally hits a VK codegen bug
  (`OpCooperativeVectorMatrixMulAddNV` bias operand rejected by SPIR-V
  validation), surfaced as an AOT `compile_to` false rather than an abort.
  The paths and gating are implemented and tested (`test_embed_coop_probe`
  exits 0 on vk/dx/cuda with the recorded reasons), but the plan-§5.1 "Go"
  condition has not been met on this hardware.
- **The vec4_io vs vec4_fma kernel-level split is not implemented**: each vec4
  kernel ships a single variant with the pinned horizontal order; the
  separate DOT/FMA-reassociation benchmark arm from plan §4.1 is absent.
- **The release-mode benchmark matrix of plan §7 has not been run** (release
  builds, ≥3 independent processes, Session e2e, cold-start/memory
  accounting). All numbers in §7 are debug-build micro/kernel results.
- **E2E golden tests need generated data**: the dense/ColBERT/sparse golden
  files are produced by `scripts/embedding/gen_goldens.py` and must exist
  before `test_embed_e2e`-family goldens validate a non-legacy path
  end-to-end.
- `slice_cls` (non-fused dense pool route) has no vec4 twin and stays scalar;
  the coop epilogue set is `Bias`/`BiasGeluErf` only (no coop ColBERT or
  sparse-head reuse yet).

---

## 9. How to run

Backend argument `<be>` is the installed plugin name: `dx`, `vk`, `cuda`,
`fallback`. Tests are device tests registered in both build systems.

```text
# Phase-0 tensor microtest (float4 alias/alignment/DOT/FMA/REDUCE/length²;
# runs on any backend):
xmake run test_embed_tensor_micro <be>

# Cooperative capability probe (C0 layout/numerics + K×N battery + AST
# positive/negative checks; no-go recorded as PASS on unsupported machines):
xmake run test_embed_coop_probe <be>

# Kernel correctness incl. the vec4 twins (norm/embed/heads/gemm table checks):
xmake run test_embed_kernel_layernorm <be>
xmake run test_embed_kernel_embed <be>
xmake run test_embed_kernel_heads <be>
xmake run test_embed_kernel_gemm <be>

# GEMM micro benchmark sweep (correctness-gated, then timed; --variants
# matches BOTH tiled names incl. rbv_* AND coop names):
xmake run benchmark_embed_gemm <be> --rows 512,4096 \
    --variants rb_64x64_b128_r84,rbv_64x64_b128_r84v4,rbv_128x64_b256_r84v4
xmake run benchmark_embed_gemm <be> --variants coop_k64_n16_t1   # dx/vk only
```

What the output proves:

- **Path actually taken** — the per-Executor selection line (§4):
  `embedding tensor path: backend='<be>', requested='<req>',
  resolved='<res>', reason='<reason>'`. Trust `resolved=`, never `requested=`:
  `auto` that loses the probe prints `resolved='vector'` plus the INFO line
  `auto -> 'vector' (cooperative unavailable: <reason>)`, and a packed-weight
  coop request prints `resolved='vector'` with reason
  `coop demoted to vector: packed '…' weights…`. The following summary line
  `embedding executor tensor path: resolved='…', gemm_variant='…',
  weight_format='…'` pins the GEMM variant source and weight format.
- **Which coop variant is live** — `embed coop: variant '<name>' selected by
  LUISA_EMBED_COOP_CONFIG (kslab …, ntile …, tokens_per_block …)` (or the
  default/warning line), and shader names `embed_coop_gemm_*_<tile-tag>`.
- **Probe outcome** — success: `embedding coop probe: passed on backend …`
  (with worst abs err); failure: the returned reason string (e.g. backend not
  allowed, `… failed to compile (SM 6.9 / NV cooperative-vector codegen
  rejected …)`, or the numeric-failure message). In `test_embed_coop_probe`
  the DX early no-go prints `coop probe: DX experimental features NOT enabled
  … coop disabled -> PASS`, and VK prints the AOT no-go line.
- **Vector kernels in use** — on DX/VK/CUDA with `vector` resolved, dispatch
  sites switch to the `*_vec4` shaders only where the per-site %4 guards
  hold; kernel-level verification is the `_vec4` checks inside the kernel
  tests (all `exit=0` = pass, `residual_add_vec4` reporting `bit_diffs=0`).
