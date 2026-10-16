# CUB scan recipe cost evidence — 2026-10-03

The frozen model chose an explicit CUB recipe for all eight independent cases. Against the original Tile path, the initial sweep geometry-equal time ratio was **0.4024 / 0.4016** (initial / recheck control), and the fresh chosen-recipe repeat was **0.4013 / 0.4020**. Both rounds passed the predeclared engineering gates; these are finite-sample observations, not confidence bounds.

**These performance packets measured explicit recipes selected by the frozen CPU model. They do not measure automatic factory search or automatic-policy dispatch performance.** Separately, the implemented automatic cost mode has passed its actual CUDA direct/graph correctness tests (47,200,048 assertions in five tests), and LLVM22/23 host parity tests passed (144 assertions each). See implementation-validation.json for exact log receipts. Automatic-policy timing remains unmeasured in this package. The policy remains opt-in; default Tile source, kernel fallback, DSL and execution primitives are unchanged.

Same-round policy/Torch geometric time ratios are **0.6514** for the initial heldout sweep and **0.6067** for the fresh repeat. Each uses that round's separately executed Torch result. These denominators are not pooled, and neither is a historical Torch number.

## Complete selected-case table

Times are seven-sample medians in microseconds. Ratios below one mean less time. All per-recipe observations, including unavailable/fallback requests and regressions, remain in observations.json.

| Cohort | Rows × width | Type | T | Original | Recheck | Selected | Same-round Torch | Selected/Torch |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| calibration | 1 × 2048 | fp16 | 256 | 1.2738 | 1.2641 | 0.9837 | 0.9822 | 1.0015 |
| calibration | 1 × 2048 | bf16 | 256 | 1.2650 | 1.2645 | 0.9932 | 1.3713 | 0.7243 |
| calibration | 17 × 4096 | fp16 | 512 | 1.9461 | 1.9452 | 1.2360 | 1.8181 | 0.6799 |
| calibration | 17 × 4096 | bf16 | 512 | 1.9569 | 1.9586 | 1.2432 | 1.2061 | 1.0307 |
| calibration | 128 × 2048 | fp16 | 256 | 3.5796 | 3.5826 | 1.7068 | 1.7979 | 0.9494 |
| calibration | 128 × 2048 | bf16 | 256 | 3.5740 | 3.5792 | 1.7211 | 2.5419 | 0.6771 |
| calibration | 4 × 8192 | fp16 | 1024 | 2.9537 | 2.9655 | 1.7429 | 2.9503 | 0.5908 |
| calibration | 4 × 8192 | bf16 | 1024 | 2.9619 | 2.9710 | 1.7521 | 1.8799 | 0.9320 |
| calibration | 128 × 8192 | fp16 | 256 | 12.3482 | 12.3474 | 4.1335 | 5.2631 | 0.7854 |
| calibration | 128 × 8192 | bf16 | 256 | 12.3362 | 12.3550 | 4.2117 | 5.4830 | 0.7681 |
| calibration | 128 × 16384 | fp16 | 256 | 36.3944 | 36.4080 | 7.4204 | 10.0985 | 0.7348 |
| calibration | 128 × 16384 | bf16 | 256 | 36.3922 | 36.4054 | 7.5605 | 14.0054 | 0.5398 |
| heldout | 3 × 2048 | fp16 | 256 | 1.2564 | 1.2609 | 0.9864 | 1.1301 | 0.8729 |
| heldout | 3 × 2048 | bf16 | 256 | 1.2578 | 1.2632 | 0.9906 | 1.1421 | 0.8674 |
| heldout | 65 × 4096 | fp16 | 512 | 3.6999 | 3.7142 | 1.8978 | 2.3754 | 0.7990 |
| heldout | 65 × 4096 | bf16 | 512 | 3.6994 | 3.7123 | 1.9298 | 2.3896 | 0.8076 |
| heldout | 129 × 8192 | fp16 | 256 | 12.3404 | 12.3524 | 4.1201 | 5.4767 | 0.7523 |
| heldout | 129 × 8192 | bf16 | 256 | 12.3636 | 12.3632 | 4.2009 | 6.2331 | 0.6740 |
| heldout | 256 × 16384 | fp16 | 128 | 66.2554 | 66.2675 | 12.3598 | 34.9971 | 0.3532 |
| heldout | 256 × 16384 | bf16 | 128 | 66.2758 | 66.2603 | 13.0095 | 35.1102 | 0.3705 |
| fresh-repeat | 3 × 2048 | fp16 | 256 | 1.2738 | 1.2579 | 0.9855 | 1.4062 | 0.7008 |
| fresh-repeat | 3 × 2048 | bf16 | 256 | 1.2640 | 1.2584 | 0.9930 | 1.4102 | 0.7041 |
| fresh-repeat | 65 × 4096 | fp16 | 512 | 3.7155 | 3.7145 | 1.8932 | 2.3756 | 0.7970 |
| fresh-repeat | 65 × 4096 | bf16 | 512 | 3.7121 | 3.7194 | 1.9321 | 2.7266 | 0.7086 |
| fresh-repeat | 129 × 8192 | fp16 | 256 | 12.3510 | 12.3549 | 4.1223 | 5.4865 | 0.7514 |
| fresh-repeat | 129 × 8192 | bf16 | 256 | 12.3702 | 12.3664 | 4.1984 | 6.2351 | 0.6733 |
| fresh-repeat | 256 × 16384 | fp16 | 128 | 66.2336 | 66.2406 | 12.5402 | 35.0249 | 0.3580 |
| fresh-repeat | 256 × 16384 | bf16 | 128 | 66.1587 | 66.2432 | 12.8760 | 35.4088 | 0.3636 |

Calibration **17×4096 BF16 remains slower than Torch**: selected 1.243187 µs / Torch 1.206138 µs = 1.030717 (3.07% more time). The broad objective of beating Torch for every operator/geometry is not achieved. Other slower recipes and original fallbacks are preserved, not filtered out.

## Frozen model and scope

Device: RTX 4060 Laptop, SM89, 24 SMs, warp32, CUDA13.4/Driver API13040. Physical resources and resident CTA capacity are Driver queries of the installed ordinary-CUDA entry, not inferred from Tile block1. The first realizer admits closed unordered FP32 +0 inclusive sums with F16/BF16 storage, one full row per program, width divisible by 8T, and actual final-pointer 16-byte alignment plus complete input/output disjoint intervals. Unsupported recipes retain the original.

`I=8`, `C=8T`, `J=N/C`, `G=ceil(T/warp)`, `A=queried resident CTA capacity`, `H=ceil(R/(SM*A))`, `B=R*N*(input_bytes+output_bytes)`. CUB features are `[1, B/(SM*2^20), H*J*I, H*J*G]`. Tile features are `[1, B/(SM*2^20), ceil(R/SM)*W]`; no physical worker count is assigned to Tile.

CUB coefficients: `[0.6027021529393383, 12.227593513162688, 0.027854484240835357, 0.022923736019806327]`. Tile coefficients: `[0.7284119403590213, 0.0, 0.00026641302734655293]`. Score is a learned positive time proxy. Choose the lowest score only if candidate/original `< 0.95`; ties favor original, then ascending T. Unknown/failed resources, nonzero local bytes, incompatible launch capacity, and unavailable candidates are excluded rather than replaced with zero.

The calibration has 12 typed cases in six geometries, with all dtype/recipe variants held out together in leave-one-geometry-out diagnostics. The original fit used equal geometry weights and training-only scaling. The public reproducer uses the already frozen full-fit and fold coefficients; it never refits on calibration or heldout outcomes. Detailed residuals, rank/conditioning and all fold decisions are retained in model-diagnostics.json.

## Validation and reproduction

Run `python reproduce.py` in this directory (Python standard library only). It verifies public file hashes, every retained seven-sample median, frozen feature formulas/resource exclusions, all full-fit and heldout decisions, fresh-repeat choice consistency, both-control ratios, same-round Torch denominators, LOGO frozen-coefficient residuals, aggregate metrics and engineering gates. The complete replay passed: **144 native observations +28 Torch observations, 1,204 raw samples, and 100 full-fit/LOGO residuals**. Sixteen unavailable recipe requests remain explicit original fallbacks, alongside 56 original-control observations. No compiler, GPU, network, tensor file or fitting operation is used.

The upstream closed validators rechecked complete exported FP64 references/per-element bounds, all output elements, input/read-only buffers, guards, native/CUB source bytes, selector receipts, original controls and Torch compiler artifacts. Their output receipts and correctness records are retained. This compact package does not contain tensor payloads, cubins, or a raw-graph Driver trace and does not rerun those oracle checks. Hashes identify retained external artifacts; omitted raw files cannot be reconstructed from hashes.

Separate implementation checks also passed fixed T256/T1024 regressions (3,008,097 /9,741,921 assertions), the CUDA Python suite (136 tests), and the no-project-exceptions check. **Broad Python discovery did not pass**: 391 tests ran with 28 errors in unchanged non-CUDA/platform fixtures (Windows path/symlink privilege, POSIX timeout and macOS dynamic-library assumptions). Its failed log is preserved separately. Default native regression also passed: 1,442,097 assertions in 25 tests, with its separate closed log receipt.

The engineering limits are geometric time ratio ≤0.95 against each control, no selected case >1.03 against either control, control drift ≤3%, no unresolved sample-extrema threshold overlap, and consistent fresh choices. Sample extrema are descriptive, not confidence intervals. Seven samples and one repeat do not prove cross-device, cross-Driver or future-run reliability.

Original profile ID: `67127e819f80a395aeecae55cd999c8e3f8b1bcf894fdf0672c58611cca87f66`. Original profile byte SHA: `d38f317cb43b1e646fcdb959e30ea89675b79722508d973ee037609db24d9186`. Compact projected bytes have their own manifest hashes; redacting paths and omitting the full3159-file snapshot does not preserve the original profile content hash. Relevant implementation/tool receipts and complete-map digest are in provenance.json. Source64-bit keys, compile64-bit keys and SHA256 receipts remain distinct.

Cold compile observations are separate from dispatch samples. Explicit-recipe compile time does not measure four-candidate automatic search overhead. No speedup claim here treats setup/compilation as dispatch work.

## Fresh automatic-policy measurement

The following PowerShell commands reconstruct the eight independent cases from the bundled case definitions and run a new original+Torch control, automatic-policy native measurement, and original native recheck, serially. Run from the repository root on the supported Windows/CUDA device after a successful full MSVC build. `full-build-success.json` must be the real completed-build marker for those binaries; do not fabricate it. Set `$torch` to a CUDA-enabled PyTorch environment with `torch.compile` dependencies; the recorded machine used the path below. Required CUDA DLL directories must be on `PATH`. Use a new output directory and keep other GPU work idle.

```powershell
$packet = 'scripts/benchmark/tile_torch/results/2026-10-03-cub-scan-cost'
$torch = '.deps/torch-cuda-venv/Scripts/python.exe'
$runner = 'scripts/benchmark/tile_torch/cuda_matrix.py'
$output = '.deps/cub-scan-cost-fresh-v1'
if (Test-Path -LiteralPath $output) { throw 'Choose a fresh output directory' }
New-Item -ItemType Directory -Path $output | Out-Null
& $torch -c "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); rows=[r['case'] for r in json.loads((p/'observations.json').read_text())['cases'] if r['cohort']=='heldout']; assert len(rows)==8 and len({r['id'] for r in rows})==8; pathlib.Path(sys.argv[2]).write_text(json.dumps({'schema':1,'cases':rows},indent=2)+'\n')" $packet "$output/cases.json"
if ($LASTEXITCODE) { throw 'Inventory reconstruction failed' }
$common = @('--cases', "$output/cases.json", '--build-dir', 'build-msvc-llvm',
    '--build-marker', 'build-msvc-llvm/logs/full-build-success.json',
    '--routes', 'native', '--torch-python', $torch, '--torch-mode', 'max-autotune',
    '--affinity-mask', '0x15400', '--threads', '4', '--samples', '7',
    '--sample-ms', '100', '--warmup-ms', '500', '--graph-batch', '100')
& $torch $runner @common --output "$output/default"
if ($LASTEXITCODE) { throw 'Original/Torch control failed; retain its outputs' }
& $torch $runner @common --native-only --native-cub-scan-cost --output "$output/automatic"
if ($LASTEXITCODE) { throw 'Automatic measurement failed; retain its outputs' }
& $torch $runner @common --native-only --output "$output/recheck"
if ($LASTEXITCODE) { throw 'Original recheck failed; retain its outputs' }
```

The fixed affinity mask identifies this machine's documented CPU set; a different CPU topology needs a valid mask with at least four distinct physical cores. The runner clears other native scheduling experiments per route and validates complete outputs, guards, source receipts and selector metadata. Accept a cohort only after its final aggregate is passed and its cleanup succeeds. Compare automatic graph-event medians independently with both new native controls and with the new default cohort's Torch median; preserve every case, fallback, failure and raw sample. This recipe does not rerun the historical measurements or promise identical future timings. Automatic search/setup wall time remains separate from dispatch time, and the compact `reproduce.py` continues to replay only the frozen explicit-recipe evidence above.
