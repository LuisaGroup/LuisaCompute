# CUDA Tile aligned structured loads on SM89

The opt-in aligned entry improves the four 256×128 SUM/MAX cases from 1.279–1.337 µs to 0.874–0.882 µs (31.0–34.6% lower time), and BF16 GEMM512³ from 28.074 to 23.150 µs (17.5% lower). It regresses FP16 scan128×8192 from 12.313 to 16.905 µs (**37.3% higher**). This representation is therefore still an explicit experiment, not a universally profitable default. All 48 native runs and 16 same-cohort Torch runs passed; three ineligible cases used the unchanged original entry.

The existing `VIEW_LOAD` is emitted as `tensor_span` / `partition_view` only where static contiguous bounds and chunk origins are proved, inside the existing aligned16 alternate entry. Runtime selection checks the actual final BufferView pointers. The original entry, arithmetic, store order, grid and Tile block ABI remain unchanged. No new DSL execution primitive, operator-name specialization or precision relaxation is involved.

Measured on an RTX 4060 Laptop GPU (SM89, 24 SMs), Windows/MSVC, system CUDA 13.4; Torch 2.14.1+cu130 / Triton Windows 3.8.0.post29. Every case is strict FP32 computation with the stated FP16/BF16 storage and unchanged full FP64 per-element bounds. Sequence: default native+Torch, aligned16 native, default native recheck. Each route retains seven graph-v2 samples, 100 ms target, 500 ms warmup, batch100, four CPU cores (`0x15400`). Times are complete-operation graph throughput, not isolated kernel latency. Native uses preallocated output and a hazard DAG; Torch functional output allocation is a different contract.

A/D, A/R and A/T are aligned-request time divided by default, recheck and same-cohort Torch time; lower is better. `mask/loads` gives the guarded root mask and rewritten load count. “fallback” means no alternate was eligible. Selection in this report is a host prediction from recorded final-pointer residues, not a Driver trace.

| Operation / dimensions | Storage | Tile | Default µs | Aligned µs | Recheck µs | Torch µs | A/D | A/R | A/T | mask/loads |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| reduce_sum 256×128 | fp16 | 8×128×1 | 1.307 | 0.874 | 1.303 | 1.016 | 0.669 | 0.671 | 0.861 | 0x1/1 |
| reduce_max 256×128 | fp16 | 8×128×1 | 1.331 | 0.874 | 1.341 | 1.021 | 0.657 | 0.652 | 0.856 | 0x1/1 |
| reduce_sum 256×128 | bf16 | 8×128×1 | 1.279 | 0.882 | 1.285 | 0.926 | 0.690 | 0.686 | 0.953 | 0x1/1 |
| reduce_max 256×128 | bf16 | 8×128×1 | 1.337 | 0.875 | 1.340 | 1.139 | 0.654 | 0.653 | 0.769 | 0x1/1 |
| reduce_sum 64×256 | fp16 | 4×256×1 | 1.010 | 0.840 | 1.011 | 0.905 | 0.832 | 0.831 | 0.928 | 0x1/1 |
| reduce_max 128×512 | bf16 | 8×512×1 | 1.253 | 1.146 | 1.253 | 1.227 | 0.915 | 0.915 | 0.935 | 0x1/1 |
| reduce_sum 128×1024 | bf16 | 4×1024×1 | 1.304 | 1.301 | 1.306 | 1.213 | 0.997 | 0.996 | 1.073 | 0x1/1 |
| reduce_max 128×8192 | fp16 | 1×8192×1 | 2.533 | 2.533 | 2.546 | 2.408 | 1.000 | 0.995 | 1.052 | 0x1/1 |
| reduce_sum 257×128 | bf16 | 8×128×1 | 1.131 | 1.130 | 1.140 | 1.143 | 0.999 | 0.991 | 0.989 | fallback |
| reduce_max 31×513 | fp16 | 4×1024×1 | 1.371 | 1.368 | 1.367 | 0.996 | 0.998 | 1.001 | 1.373 | fallback |
| rmsnorm 128×1024 | bf16 | 1×1024×1 | 1.679 | 1.629 | 1.685 | 1.350 | 0.970 | 0.967 | 1.207 | 0xb/2 |
| softmax 32×512 | fp16 | 1×512×1 | 1.200 | 1.193 | 1.203 | 1.084 | 0.994 | 0.991 | 1.100 | 0x9/1 |
| scan 128×8192 | fp16 | 1×8192×1 | 12.313 | 16.905 | 12.342 | 8.127 | 1.373 | 1.370 | 2.080 | 0x9/1 |
| scan 129×8191 | bf16 | 1×8192×1 | 13.054 | 13.050 | 13.053 | 26.592 | 1.000 | 1.000 | 0.491 | fallback |
| gemm 512×512×512 | bf16 | 64×64×32 | 28.074 | 23.150 | 28.253 | 16.097 | 0.825 | 0.819 | 1.438 | 0xb/2 |
| bmm 2×64×64×64 | fp16 | 16×32×32 | 1.200 | 1.216 | 1.212 | 1.496 | 1.013 | 1.004 | 0.813 | 0xb/2 |

All rechecks are within 5% of their original controls; that is a drift check, not replicated randomized evidence. The scan slowdown is present against both controls. Near-equal rows should not be read as robust improvements. The three fallback rows have byte-identical source and expose ordinary timing variation. No rejected rows or samples were removed.

[The companion JSON](aligned-view-sm89.json) retains all **448 raw event-time samples**, event/host spans, calibration and cold costs, every fixture/oracle/source/output receipt, native readonly/guard checks, Torch full-output checks, and the validated queue/build/checkpoint identities. The checkpoint SHA256 is `b0188ea894f30edae3d431b60ce0f73b45264e7cdb18c26ee3f15a17d681bcb9`. Exact measured files are bound by the build snapshot; captured HEAD alone does not describe its uncommitted changes. No standalone Driver-probe result is pooled into this table.

Reproduce with the common case inventory and `scripts/benchmark/tile_torch/cuda_matrix.py`: baseline `--routes native`, alternate `--routes native --native-only --native-aligned16`, then baseline `--routes native --native-only`; keep seven samples, 100 ms/sample, 500 ms warmup and graph batch100. Direct API use requires `LUISA_CUDA_TILE_IR=1` and `LUISA_CUDA_TILE_IR_ALIGNED16=1`; cuda_matrix sets these for the corresponding flags. All worker, partition, cost and streaming experiments were disabled here. The measured inventory is retained in `aligned-view-sm89-cases.json`. Run `python reproduce.py` beside the JSON to check all 448 samples and reproduce every median/ratio without GPU access. Checkpoint/source hashes identify original evidence, not this projected JSON; reproduce.py reports the public JSON SHA256 separately.
