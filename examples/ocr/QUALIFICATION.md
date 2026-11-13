# OCR qualification matrix (Phase 12)

Status: **no REAL model is validated**. All 15 requirement models (17 selectable
targets counting the three PP-OCRv6 sizes) remain `planned`. `Session::create`
and `Session::recognize` validate their arguments and then fail closed with
`unvalidated` for every real artifact. The machine-readable form of this matrix
is `example_ocr --list-models` (offline; no backend/package/input required;
exit 0), emitted by `ocr/cli.cpp::model_support_report_json()` from the same
default planned table Session will consult once real loading is enabled
(`real_model_validated=false`, `session_status="unvalidated"`, per-target
`support_state="planned"`, `validated=false`).

## What "qualified" means here

- **Synthetic parity** (what Phases 0–11 actually have): small, declared
  synthetic architectures execute on the host and on device and match
  independent double-accumulating CPU oracles at layered golden points. This
  exercises the ENGINE (operators, executors, caches, schedulers, output
  parsers), **not** any official model. Synthetic tests are NOT real model
  support and never upgrade a recipe's support state.
- **Real-model qualification** (NOT met for any target): requires pinned
  official checkpoint revision + config fingerprint + package hash, official
  processor/tokenizer parity, official layered goldens, deterministic
  end-to-end decode on real documents, accuracy gates (CER/mAP/table/JSON as
  applicable per MODELS.md), and measured performance/memory on the declared
  target hardware. None of these artifacts exists in this repository.

## Measurement environment

- The device evidence below was measured on an NVIDIA GeForce RTX 4080
  SUPER, cuda backend. The final Phase 12 tree passed the same 27 host suites
  (2863 assertions / 253 tests) and 36 device suites (6719 assertions / 268
  tests) under BOTH xmake and a fresh CMake Debug build (Clang 22.1, CUDA
  13.0). These are correctness counts, not performance measurements.

- Historical `vk` passes are recorded ONLY where the phase sections in
  README.md / MODELS.md already document them (Phases 2–6, pre-merge). Since
  the upstream merge, `vk` device creation is blocked by an upstream
  regression (`VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT` rejected
  by the installed validation layer); no backend edits are made here. Those
  historical notes are not current-environment measurements.
- **No RTX 4060 measurements exist.** RTX 4060 8 GB is a requirement target,
  not a measured configuration. No latency/throughput/residency number in
  this repository is a benchmark result; correctness suites are not
  benchmarks, and no benchmark-based performance claim is made.
- No weights are bundled or redistributed and no runtime downloads exist.

## 17-target matrix

Legend for the last five columns: every cell is **NOT QUALIFIED** — no pinned
official checkpoint revision, no official package, no validated precision
matrix, no qualified output mode, and no measured performance exists for any
target. The synthetic suite column lists the engine-parity evidence only.

| Target | Family | Synthetic suite / evidence (NOT real-model support) | Backend actually measured | Checkpoint revision | Package | Precision | Mode | Performance |
|---|---|---|---|---|---|---|---|---|
| pp_ocrv6:tiny | cnn | `test_ocr_model_pp_ocrv6` (per-size detector→DB→crop→CTC vs unfolded oracle) | cuda (RTX 4080 SUPER); vk historical (Phase 5, pre-merge) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| pp_ocrv6:small | cnn | `test_ocr_model_pp_ocrv6` | cuda (RTX 4080 SUPER); vk historical (Phase 5, pre-merge) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| pp_ocrv6:medium | cnn | `test_ocr_model_pp_ocrv6` | cuda (RTX 4080 SUPER); vk historical (Phase 5, pre-merge) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| paddle_ocr_vl_1_6 | layout_vlm | `test_ocr_model_paddle_ocr_vl`, `test_ocr_region_scheduler`, `test_ocr_layout` (NaViT+connector+ERNIE parity; Phase 11 graph counters) | cuda (RTX 4080 SUPER); vk historical (Phase 6, pre-merge) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| ovis_ocr2 | hybrid_gdn | `test_ocr_model_ovis_ocr2`, `test_ocr_gdn_*` (GDN conv/recurrent/chunked/cache parity) | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| tele_ocr | dense_vlm | `test_ocr_model_tele_ocr` (window/global vision + Qwen3 decoder parity) | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| glm_ocr | layout_vlm | `test_ocr_model_glm_ocr`, `test_ocr_region_scheduler` | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| mineru_2_5_pro_2605_1_2b | layout_vlm | `test_ocr_model_mineru`, `test_ocr_region_scheduler` | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| hunyuan_ocr_1_5 | custom_vlm | `test_ocr_model_hunyuan` (XD-RoPE/custom conventions parity) | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| surya_2_5 | hybrid_gdn | `test_ocr_model_surya`, `test_ocr_gdn_*` (detector stand-in + GDN decoder) | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| open_doc_0_1b | layout_vlm | `test_ocr_model_open_doc`, `test_ocr_layout` (PP-DocLayoutV2 + UniRec CNN+CTC two-stage) | cuda (RTX 4080 SUPER); vk historical (Phase 6, pre-merge) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| fire_red_ocr | dense_vlm | `test_ocr_model_fire_red` (shared Qwen3 graph via its own tensor map) | cuda (RTX 4080 SUPER); vk historical (Phase 4, pre-merge) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| qwen3_vl_2b_instruct | dense_vlm | `test_ocr_model_qwen3_vl` (M-RoPE + QK-Norm + DeepStack layered parity) | cuda (RTX 4080 SUPER); vk historical (Phase 4, pre-merge) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| deepseek_ocr_2 | moe_vlm | `test_ocr_model_deepseek_ocr2`, `test_ocr_moe_*` (router/top-k/permutation/experts + exact custom mask) | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| dots_mocr | dense_vlm | `test_ocr_model_dots_mocr` (packed-INT4 W4A32 decoder parity) | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32/W4A32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| got_ocr2_0 | dense_vlm | `test_ocr_model_got_ocr2` | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |
| olm_ocr_2 | dense_vlm | `test_ocr_model_olm_ocr2` (relative-position vision attention parity; W4A32 decode A/B vs dequantized oracle) | cuda (RTX 4080 SUPER) | NOT QUALIFIED (missing) | NOT QUALIFIED (missing) | NOT QUALIFIED (synthetic FP32/W4A32 only) | NOT QUALIFIED | NOT QUALIFIED (no measurements) |

Shared engine evidence (all synthetic, cuda on RTX 4080 SUPER; vk historical
for Phases 2–3 pre-merge): `test_ocr_ops_*` (image/gemm/norm/rope/attention/
conv/vision), `test_ocr_runtime_device`, `test_ocr_decode_prefill`,
`test_ocr_decode_cache`, `test_ocr_decode_sampling`, `test_ocr_decode_stop`,
and the 27 host suites under `src/tests/unit/ocr/` (CTest label `unit;unit_ocr`,
no GPU/model needed).

## Current execution boundaries (Phases 11–12)

- Phase 11 added: exact-capacity P7 ragged multi-crop vision graph chunking
  (Paddle/OpenDoc recognizers remain per-crop); optional
  `DenseDecodeOptions.enable_cuda_graph` single-sequence CUDA graph
  capture/update with explicit fallback counters; `SchedulerLimits.
  enable_cuda_graph` forwarding through the layout orchestrator; P7/Paddle
  `cuda_graph_steps/updates/fallbacks` counters; optional
  `DenseExecutorConfig.enable_fused_residual_rms_norm` (default off, reference
  command sequence preserved). All are opt-in engine features exercised by
  synthetic parity tests — none is a real-model performance claim.
- Text decode is still independent batch-1 per region;
  `plan_decode_batches` / `max_decode_batch` remain planning-only (no batched
  decode executor exists).
- Still unsupported unless code proves otherwise: native shader fast paths,
  GLM MTP, Hunyuan DFlash, paged KV (fails closed), batched decode, grouped
  persistent MoE GEMM, and CUDA graphs for the GDN/MoE/Surya executors (only
  the dense/P7/Paddle paths implement them).
- `Session::create` / `Session::recognize` remain **unvalidated** for every
  real artifact; the unsafe experimental registry override never enables
  inference.

## Reproducing the synthetic qualification (no real artifacts needed)

Host suites (no GPU/model; from the repository root):

```bash
xmake build test_ocr_cli
xmake run test_ocr_cli
# all OCR host suites:
ctest --test-dir build -L unit_ocr        # CMake build tree
```

Offline support report (no backend/package/input; exit 0):

```bash
xmake build example_ocr
xmake run example_ocr --list-models
```

Device suites (synthetic parity; explicit backend, cuda measured on the
RTX 4080 SUPER in the current environment):

```bash
xmake build test_ocr_ops_gemm test_ocr_decode_prefill test_ocr_decode_stop \
  test_ocr_model_pp_ocrv6 test_ocr_model_qwen3_vl test_ocr_region_scheduler \
  test_ocr_model_deepseek_ocr2
xmake run test_ocr_ops_gemm cuda
xmake run test_ocr_decode_prefill cuda
xmake run test_ocr_decode_stop cuda          # Phase 11 graph capture/update A/B
xmake run test_ocr_model_pp_ocrv6 cuda
xmake run test_ocr_model_qwen3_vl cuda
xmake run test_ocr_region_scheduler cuda     # Phase 11 chunking + graph counters
xmake run test_ocr_model_paddle_ocr_vl cuda  # Phase 11 Paddle graph counters
xmake run test_ocr_model_deepseek_ocr2 cuda
```

The full per-phase suite lists and tolerances are in README.md and MODELS.md.
A green run proves synthetic engine parity ONLY.

## Populating the real rows (external artifacts; never automatic)

Real qualification requires artifacts the user supplies under their own
license. Nothing here downloads anything, and **tests never modify or
regenerate goldens**: the frozen synthetic fixtures in `src/tests/data/ocr/`
are regenerated only by `tools/generate_fixtures.py`, never by tests; official
goldens are exported explicitly by `tools/export_golden.py` into a separate
output directory and compared read-only.

1. Obtain the official checkpoint/config/processor/tokenizer locally at an
   exact revision (outside this repository; no runtime download exists).
2. Pack and validate the artifact offline:

   ```bash
   python examples/ocr/tools/pack_model.py --checkpoint-dir <official-dir> \
     --config <official-dir>/config.json --revision <exact-revision> \
     --license <SPDX> --model <target> --recipe-version <v> \
     --preprocessing-version <v> --tokenizer-version <v> \
     --output-parser-version <v> --output model.lcocr
   python examples/ocr/tools/validate_package.py model.lcocr --json
   ```

3. Export official processor features and layered goldens (explicit local
   artifacts; recorded revisions):

   ```bash
   python examples/ocr/tools/preprocess.py --processor-dir <official-dir> \
     --processor-revision <rev> --image page.png --output page.ocrfeat
   python examples/ocr/tools/export_golden.py --model-dir <official-dir> \
     --revision <exact-revision> --input page.png --output-dir <golden-dir>
   ```

4. Pin the recipe identity (revision + config fingerprint + package format
   version + exact tensor signature) in the registry, run the layered golden
   and deterministic end-to-end comparisons against the exported references,
   and measure accuracy/performance on the declared hardware. Only after ALL
   gates in MODELS.md pass may a recipe move to `validated` and a row above
   replace its NOT QUALIFIED cells with the measured facts (revision, package
   hash, precision matrix, qualified modes, measured performance with the
   exact hardware/driver/backend recorded). Until then every row stays NOT
   QUALIFIED and Session keeps failing closed.
