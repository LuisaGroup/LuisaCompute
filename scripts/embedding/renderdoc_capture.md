# RenderDoc capture workflow for the bge-m3 engine (opt plan O0.3)

Purpose: per-dispatch GPU timings for ONE encode — the real hot list that host
timers cannot see inside a committed CommandList (plan §7.5: spot-check after
each wave; §3 ordering: "optimize the measured top-3 kernels, not the guessed
ones").

## Procedure

1. Build release: `xmake f -p windows -a x64 -m release --lc_bin_dir=build/opt_rel/bin`
   then `xmake build benchmark_embed`. RenderDoc injection works on both dx and
   vk; dx captures are the most complete on Windows.
2. Launch with injection (renderdoccli must match the installed RenderDoc):
   ```
   renderdoccli.exe --capture-file benchmark_results/captures/embed_b1s512.rdc \
       --working-dir D:\compute --inject \
       --exe D:\compute\build\opt_rel\bin\release\benchmark_embed.exe \
       -- dx --seq 512 --batch 1 --iters 3 --warmup 1
   ```
   (`--inject` instead of launching the exe directly keeps argv passing clean;
   alternatively launch `benchmark_embed.exe` from the RenderDoc UI with
   working dir D:\compute and args `dx --seq 512 --batch 1 --iters 3`.)
3. Trigger: with `--iters 3 --warmup 1`, capture the WHOLE run (4 encodes) —
   F12 (default) or `renderdoccli --capture-*` trigger keys. One capture is
   enough; each encode is identical work.
4. In the RenderDoc Event Browser: the captured frame contains 4 back-to-back
   CommandList executions of ~150 dispatches each (upload → embed → 24×
   {qkv GEMM, attn scores, softmax chunk(s), attn pv, out GEMM, residual,
   LN, ffn.up GELU GEMM, ffn.down GEMM, residual, LN} → slice/cls + l2norm →
   readback).
5. Per-dispatch GPU time: select a dispatch → Pipeline State / Timer column;
   or use the python shell (RenderDoc > Tools > Python Console):
   ```python
   # actions = controller.GetRootActions(); walk children, action.GetDuration()
   # group by shader name (the kernels carry opt.name set at compile time:
   # gemm_bias, gemm_bias_gelu_erf, attn_scores, softmax_row, attn_pv,
   # layernorm, residual_add, emb_lookup, slice_cls, l2_normalize, ...)
   ```
6. Aggregate: sum per kernel name across one encode; rank. Expectation to
   confirm/refute (plan §1 hypothesis): gemm family dominates; attn trio
   significant at S=512; softmax chunked rows visible as repeated softmax_row
   dispatches at S>4092/head-rows.
7. Archive: save .rdc under benchmark_results/captures/<date>_<backend>_b<b>s<s>.rdc
   and append the ranked table to benchmark_results/embed_opt_log.md.

## Notes / traps

- vk captures on NVIDIA only expose queue timings if GPU counters are enabled
  (RenderDoc prompts; accept "enable unlimited" for profiling use).
- The dx DEVICE_REMOVED defect (opt log B-1) can poison a capture session —
  capture B=1 shapes first; batch captures only after B-1 is fixed.
- Do NOT capture with LUISA_ENABLE_VALIDATION=1 (validation layers add
  per-command overhead and skew the hot list).
- Keep the capture workload deterministic: same text (bench synthesizes it),
  warmup excluded from the timed iterations.
