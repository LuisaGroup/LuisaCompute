# Collective cost and streaming validation — 2026-10-03

This validates the frozen experimental profile described in [the earlier checkpoint](../2026-10-03-collective-cost/README.md). That checkpoint correctly records held-out cases as unmeasured at its freeze. The model was not changed after these twelve cases were measured.

The twelve strict held-out cases completed four sequential phases: default with PyTorch, cost, default recheck, and cost repeat. All 48 native results and twelve PyTorch results passed their unchanged full-output FP64 oracle. Four cases selected compiler hint 8 and eight retained the default; retaining zero is not an optimization.

The geometric mean across the twelve case ratios is 0.97928 for cost/default and 0.98141 for cost-repeat/default-recheck. Ratios below one are lower measured medians. These are descriptive means, not confidence estimates; the four selected cases and eight default cases are kept separate in cost-statistics.json.

Default recheck drift ranged from -0.620% to +0.718%. RMSNorm 7x2051 BF16 improved about 9% and softmax 9x769 FP16 about 11% in both cost phases. The softmax result still trails PyTorch. LayerNorm's roughly 3% change and MAX's near-zero change are not called robust gains. This does not establish that all reductions or scans match PyTorch.

| Case | default | cost | recheck | cost-repeat | Initial Torch |
|---|---:|---:|---:|---:|---:|
| heldout-reduce_sum-37x257-fp32-br4 | 0.9965 us | 0.9946 us [default, hint 0, predicted-default] (-0.2%) | 0.9947 us (-0.2%) | 1.0021 us [default, hint 0, predicted-default] (+0.6%) | 0.9761 us |
| heldout-reduce_sum-19x2051-fp16 | 1.0313 us | 1.0253 us [default, hint 0, predicted-default] (-0.6%) | 1.0250 us (-0.6%) | 1.0326 us [default, hint 0, predicted-default] (+0.1%) | 1.3520 us |
| heldout-reduce_max-37x257-bf16-br4 | 0.9897 us | 0.9851 us [default, hint 0, predicted-default] (-0.5%) | 0.9902 us (+0.0%) | 0.9909 us [default, hint 0, predicted-default] (+0.1%) | 0.9195 us |
| heldout-reduce_max-3x4097-fp32 | 1.1081 us | 1.1052 us [selected, hint 8, predicted-saving] (-0.3%) | 1.1066 us (-0.1%) | 1.1109 us [selected, hint 8, predicted-saving] (+0.3%) | 1.2225 us |
| heldout-scan-23x513-fp32 | 1.0003 us | 1.0046 us [default, hint 0, prefix] (+0.4%) | 0.9958 us (-0.5%) | 0.9972 us [default, hint 0, prefix] (-0.3%) | 1.0224 us |
| heldout-scan-11x4097-fp16 | 2.1903 us | 2.1898 us [default, hint 0, prefix] (-0.0%) | 2.1894 us (-0.0%) | 2.1902 us [default, hint 0, prefix] (-0.0%) | 4.2278 us |
| heldout-rmsnorm-7x2051-bf16 | 1.7391 us | 1.5838 us [selected, hint 8, predicted-saving] (-8.9%) | 1.7380 us (-0.1%) | 1.5904 us [selected, hint 8, predicted-saving] (-8.5%) | 2.0951 us |
| heldout-rmsnorm-256x1536-fp16 | 3.5558 us | 3.5519 us [default, hint 0, predicted-default] (-0.1%) | 3.5813 us (+0.7%) | 3.5203 us [default, hint 0, predicted-default] (-1.0%) | 2.5848 us |
| heldout-layernorm-5x1025-fp32 | 1.3921 us | 1.3506 us [selected, hint 8, predicted-saving] (-3.0%) | 1.3845 us (-0.5%) | 1.3492 us [selected, hint 8, predicted-saving] (-3.1%) | 1.4932 us |
| heldout-layernorm-64x3073-bf16 | 3.0389 us | 3.0434 us [default, hint 0, predicted-default] (+0.1%) | 3.0361 us (-0.1%) | 3.0382 us [default, hint 0, predicted-default] (-0.0%) | 6.2544 us |
| heldout-softmax-9x769-fp16 | 1.4820 us | 1.3159 us [selected, hint 8, predicted-saving] (-11.2%) | 1.4831 us (+0.1%) | 1.3216 us [selected, hint 8, predicted-saving] (-10.8%) | 1.1412 us |
| heldout-masked_softmax-33x1537-fp32 | 2.9403 us | 2.9463 us [default, hint 0, predicted-default] (+0.2%) | 2.9397 us (-0.0%) | 2.9396 us [default, hint 0, predicted-default] (-0.0%) | 2.7779 us |

Percent changes use the fresh initial native default. Recheck drift and cost repeat versus recheck are retained separately in JSON; raw samples are never pooled or filtered. Default-retained stages are not optimizations. Historical source preservation contains no historical timing comparison.


| Case | default | stream1024 | stream2048 | recheck | Initial Torch |
|---|---:|---:|---:|---:|---:|
| aligned16-scan-128x8192-fp16 | 12.3091 us | 14.8439 us [host expects stream entry] (+20.6%) | 15.1229 us [host expects stream entry] (+22.9%) | 12.3102 us (+0.0%) | 5.4863 us |
| aligned16-scan-128x8192-bf16 | 12.3400 us | 14.8048 us [host expects stream entry] (+20.0%) | 15.1120 us [host expects stream entry] (+22.5%) | 12.3266 us (-0.1%) | 5.4737 us |
| aligned16-control-scan-128x8192-fp32 | 13.8589 us | 14.9520 us [host expects stream entry] (+7.9%) | 15.4328 us [host expects stream entry] (+11.4%) | 13.8507 us (-0.1%) | 19.8081 us |
| stream-tail-37x2051-fp32-br4 | 5.3508 us | 9.3554 us [host expects stream entry] (+74.8%) | 9.5157 us [host expects stream entry] (+77.8%) | 5.3529 us (+0.0%) | 4.1296 us |
| stream-tail-37x2051-fp16-br4 | 7.1914 us | 10.4616 us [host expects stream entry] (+45.5%) | 13.1002 us [host expects stream entry] (+82.2%) | 7.1922 us (+0.0%) | 3.7662 us |
| stream-tail-37x2051-bf16-br4 | 5.1209 us | 9.4143 us [host expects stream entry] (+83.8%) | 9.4742 us [host expects stream entry] (+85.0%) | 5.1202 us (-0.0%) | 3.7693 us |
| stream-tail-5x8191-fp16-br1 | 3.1067 us | 6.7939 us [host expects stream entry] (+118.7%) | 6.1782 us [host expects stream entry] (+98.9%) | 3.1069 us (+0.0%) | 4.7764 us |
| stream-128x4096-bf16-br1 | 6.4713 us | 7.7686 us [host expects stream entry] (+20.0%) | 7.9745 us [host expects stream entry] (+23.2%) | 6.4897 us (+0.3%) | 2.8630 us |

Percent changes use the fresh initial native default. Recheck drift and cost repeat versus recheck are retained separately in JSON; raw samples are never pooled or filtered. Default-retained stages are not optimizations. Historical source preservation contains no historical timing comparison.


Both streaming chunk candidates passed correctness but regressed on all eight cases and are **rejected for automatic selection**. Long FP16/BF16 scans regressed about 20% with chunk 1024 and 22–23% with chunk 2048; some tail cases regressed much more. The existing cost policy continues to retain the default for prefix scans. No physical register/state reduction is claimed from these source-level changes.

The fresh PyTorch FP32 128x8192 scan measured 19.808 us, versus 10.480 us in the earlier checkpoint. The retained max-autotune caches selected different configurations for the identical Triton template: fresh XBLOCK8/R0_BLOCK512/warps4 versus old XBLOCK1/R0_BLOCK4096/warps16. The fresh seven samples cluster between 19.674 and 19.962 us. This baseline variant must not be presented as a Luisa improvement or stable advantage over PyTorch: Luisa's default source is byte-identical to the old source. Only the fresh timing is used for the table; scan-torch-diagnostic.json retains both raw records/configurations and process-window telemetry, with the causal limits stated.

Each phase retains seven graph samples and calibration. Each sample repeats a graph of 100 operations to target approximately 100 ms; replay caps and actual spans remain in JSON. Warmup is 500 ms, four CPU threads use affinity 0x15400, and all phases share the same LLVM22 MSVC build. The initial PyTorch baseline is measured once per case, not in the three later native-only phases. Raw samples are never pooled or removed.

Every saved output was checked against the original bounds. Cost source bytes differ only by the documented entry worker hint. Worker counts are compiler hints, not an observed physical warp count. Streaming selection, when included, is the existing shared host selector receipt based on actual views/static ranges, not a device trace. Old/new default source comparisons do not reuse historical timings from different binaries.

Validation also includes the recorded LLVM22 full build, LLVM23 related-target build, and the two streaming GPU correctness runs: each chunk size passed 436190 assertions in four groups and recorded graph updates accepted 4/rejected 0. Host tests on both LLVM trees passed 522 assertions/four groups and 2119/six groups; Python passed 110 tests and the no-throw check covered 2767 files. The latter host/Python counts were observed in session output without retained local logs, as explicitly recorded in verification-receipts.json. No graph-update rejection was observed or claimed tested at runtime in these runs.

Public JSON files retain all raw seven-sample arrays, phase decisions, cold costs, fixture/source hashes, source proofs, and receipt summaries. Paths use placeholders and GPU UUIDs are removed. Original local artifact hashes identify original bytes; receipts.json hashes the actual public bytes. All text uses UTF-8 and LF.

Run `python verify_public.py` from this directory to verify all public file hashes and recompute the medians, ratios, subgroups, geometric means and stored source-proof relationships. This verifies published evidence consistency; it does not rerun GPU execution or recreate missing raw binaries.

Reproduce measurement from the repository root after a successful full CMake build and the documented CUDA/PyTorch environment setup:

```powershell
$python = '.deps/torch-cuda-venv/Scripts/python.exe'
$caseFile = 'scripts/benchmark/tile_torch/results/2026-10-03-collective-validation/cost-cases.json'
$common = @('--cases', $caseFile, '--routes', 'native', '--build-dir', 'build-msvc-llvm', '--build-marker', 'build-msvc-llvm/logs/full-build-success.json', '--torch-python', $python, '--affinity-mask', '0x15400', '--threads', '4', '--samples', '7', '--sample-ms', '100', '--warmup-ms', '500', '--graph-batch', '100', '--native-timeout', '180', '--torch-timeout', '180', '--ranking-contract', 'standard')
& $python scripts/benchmark/tile_torch/cuda_matrix.py @common --output .deps/new-cost-default
& $python scripts/benchmark/tile_torch/cuda_matrix.py @common --native-only --native-collective-cost --output .deps/new-cost-candidate
& $python scripts/benchmark/tile_torch/cuda_matrix.py @common --native-only --output .deps/new-cost-recheck
& $python scripts/benchmark/tile_torch/cuda_matrix.py @common --native-only --native-collective-cost --output .deps/new-cost-repeat
```

For the independent streaming queue use streaming-cases.json, then fresh default, `--native-only --native-streaming-scan 1024`, `--native-only --native-streaming-scan 2048`, and a fresh native-only default recheck. Keep source, executable and helper bytes unchanged across all phases and stop if any command fails. The repository runner performs the full packet and final artifact gates.
