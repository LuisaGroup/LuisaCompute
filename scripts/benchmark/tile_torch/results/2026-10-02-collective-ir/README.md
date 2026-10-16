# CUDA Tile collective IR: worker and structural experiments

MSVC/LLVM 22, system CUDA 13.4, RTX 4060 Laptop GPU. All strict saved-output/oracle, input/guard, generated-source and cohort-completion gates passed. The measured implementation is identified by the source/binary hashes in [receipt-summary.json](receipt-summary.json); measurements preceded publication of the implementation commit. No candidate becomes a production default from this experiment.

The worker experiment contains **16 configurations, 10 operation/shape pairs and 12 operation/shape/dtype pairs**, including repeated schedules. Each has default0, requested worker4, requested worker8 and final default0 visits; only the first default0 visit runs Torch. The structural experiment repeats seven of those configurations: seven controls, three scan-chunk1024, three scan-chunk2048 and four independent-axis1 visits. These are not additional independent workload shapes.

[timings.csv](timings.csv) retains all seven samples for every one of the 97 route visits (64 worker-native, 16 Torch, 17 structural-native). The [worker checkpoint](worker/checkpoint.json) and [structural checkpoint](structural/checkpoint.json) additionally retain complete calibration attempts, actual graph replay counts/spans, warmup, cold phases, full validation receipts and source proof. No raw sample is discarded. CUDA graph timing is per complete logical operation: seven samples target 100 ms each using calibrated repeated replays of a 100-operation graph, with 500 ms warmup. Native preallocated outputs and Torch functional graph-pool allocations differ. Samples are correlated; percentage changes are observed medians, not statistically established wins.

## Observed worker behavior

The final default0 recheck differs from the first by **-0.44% to +0.98%**. Explicit worker4 differs by -0.19% to +1.48%, with no material improvement. Worker8 helps some low-program-count reductions and normalization but substantially regresses other geometries. These are compiler hints, not measured hardware thread counts.

| Configuration | Default0 us | Worker8 us | Baseline Torch us | Worker8 change |
|---|---:|---:|---:|---:|
| Scan 128x8192 FP16 | 12.344 | 16.206 | 5.465 | +31.3% |
| Scan 128x8192 BF16 | 12.357 | 16.137 | 5.480 | +30.6% |
| Scan 128x8192 FP32 | 13.844 | 18.408 | 10.480 | +33.0% |
| Sum 17x1024 FP16, BR4 | 2.199 | 1.477 | 0.908 | -32.8% |
| Max 17x1024 FP16, BR4 | 2.165 | 1.559 | 0.914 | -28.0% |
| RMSNorm 1x8192 FP16 | 2.890 | 2.272 | 1.414 | -21.4% |
| LayerNorm 1x8192 BF16 | 3.274 | 2.618 | 1.916 | -20.0% |
| LayerNorm 128x8192 BF16 | 11.216 | 9.882 | 8.091 | -11.9% |
| Softmax 1024x512 FP16 | 5.721 | 8.949 | 3.282 | +56.4% |

The five improved configurations remain about 1.22-1.71 times their baseline Torch time. Native default narrow wide scans remain about 2.26 times Torch. BR1 is already faster than either BR4 variant for the tested 17x1024 reduction configurations, so a worker-hint improvement does not establish the globally best schedule. Torch references above are the same-fixture baseline measurement, not contemporaneous Torch repetitions for each hint. See the [complete 16-row table](worker/README.md), including near ties and regressions. There is no pooled win rate.

## Pure structural experiments

The source transformations preserve the original global-memory materialization and work grid while partitioning pure Tile values. Both scan chunk sizes were **4.4-6.3% slower** on the three 128x8192 scans. Independent-axis1 lowered BR4 sum17x1024 FP16 from 2.208 to 1.692 us (-23.4%), but regressed sum128x65 BR4 by39.4%, scan17x1024 BR4 by10.5%, and max17x1024 BR4 by20.3%. These native-only structural cohorts have no Torch comparison. The [complete table](structural/README.md) preserves all cases.

Worker sources were proven byte-identical after removing only the exact worker hint and its extern-C wrapper. Structural sources intentionally differ: their complete diffs/hashes and rewrite counters are preserved. Entry ABI, memory address/load/store and bid-coordinate statements are identical after line whitespace stripping; reported block/program counts and inferred fixture grids agree. This is lexical artifact evidence plus numerical validation, not a general semantic proof or a Driver launch trace. [source-hashes.csv](source-hashes.csv) contains all 81 native-source receipts and worker-normalized hashes.

## Reproduction and limitations

Inputs are [worker16](cases-worker16.json), [structural7](cases-structural7.json), [scan3](cases-scan3.json) and [independent4](cases-independent4.json). All use true FP32/FP16/BF16 storage, FP32 computation, unchanged strict FP64 per-element bounds, seed20261001 and recorded patterns. Scan retains unordered-tree semantics. The separate [heldout12 inventory](cases-heldout12-unmeasured.json) was frozen but **has not been measured or used to select a policy**.

After a full CMake build, use the existing `cuda_matrix.py` with `--routes native --samples 7 --sample-ms 100 --warmup-ms 500 --graph-batch 100 --threads 4 --ranking-contract standard`, an explicit four-physical-core affinity mask, the completed-build marker and a fresh output directory. The measured machine used `0x15400`; choose an appropriate mask on another machine. Use `--torch-mode max-autotune` and the CUDA Torch Python environment for the single baseline. Torch is fullgraph and its actual compiler evidence is retained. Native and Torch have separate cold-compile scopes.

The measured worker order was requested4 native-only, requested8 native-only, default0 plus Torch, and default0 native-only recheck. Set `--native-worker-warps` explicitly for each cohort. Structural cohorts use `--native-only --native-worker-warps 0`, with either `--native-scan-chunk 1024`, `--native-scan-chunk 2048`, or `--native-independent-axis 1`; controls leave both structural values zero. Alignment remains off throughout. Run cohorts serially from identical frozen sources/binaries; do not alter the oracle, timing or policy based on heldout outcomes.

Historical v1 records are excluded, not repaired: the worker queue was interrupted after undocumented worker1/2 attempts, and the structural chunk2048 aggregate failed on a Windows checkpoint replacement permission error even though its GPU child passed. The v2 runner adds a bounded atomic-replacement retry and reruns complete cohorts. Those old failures remain preserved in local raw logs. Raw binary/output files are intentionally not copied here; compact hashes, validation summaries, original artifact references and all timing samples are retained. JSON/CSV/Markdown are LF with local Git attributes so published hashes remain stable.
