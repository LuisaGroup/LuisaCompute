# Independent Torch static appendix

This appendix describes retained artifacts from the separate, closed V4 fresh-Torch run. It contributes no timing denominator to V5. All graph/stream timings and native/Torch ratios are intentionally omitted from `torch-static.json`; the independent original review receipt is retained.

Torch 2.14.1+cu130 (git 5c4886908584029761b579af026dcfb627c84070), triton-windows 3.8.0.post29/Triton 3.8.0. A saved best_config is matched uniquely to source filename, X/R extent, warps and stages. This is retained configuration evidence, not an instrumented launch trace. Actual cache cubin resources were inspected with cuobjdump without recompilation or GPU execution.

R128×16384 uses one streaming FP32 contribution-tile reduction per storage type, then one post-loop reduction. XBLOCK=2/RBLOCK=2048; FP32/BF16 have 16 warps, FP16 32. PTX requires 512/1024/512 threads. TTGIR distributes FP32's two rows through each thread's register tile ([1,4] elements/thread, [1,16] warps/CTA), whereas BF16 distributes rows across warp groups ([1,8], [2,8]); FP16 is [1,4], [2,16]. SASS loads are 128/64/128-bit respectively. Retained REG values are 40/30/39; dynamic shared metadata is 128/128/64 B, static shared and local/stack allocations are zero. No LDL/STL appears.

All three use `ld.global.L1::evict_first.L2::cache_hint` with `createpolicy.fractional.L2::evict_first.b64 ..., 1.0`; no cg/ca/nc/evict_last distinction separates them. Contiguous adjacent-lane addresses and vector widths establish coalescible patterns, not measured transaction count or cache residency. Thread-limit-only capacity suggests 1 CTA/SM for 1024 threads versus 3 for 512 on this 1536-resident-thread/SM device; this is an upper bound, not actual occupancy or a timing explanation.

At R3×16384, a real two-kernel split forms six 8192-element partial sums, then reduces two partials per output. R128×16384 has enough independent outputs for the installed heuristic to avoid splitting. The local source receipts document threshold/configuration and coordinate-descent rules. No coefficient, policy, baseline, oracle or candidate was altered from this inspection.

The historical review's unusual process timings have no established causal explanation. Configuration, allocation and cache history could interact; this static appendix proves neither a cache bottleneck nor stable native superiority. V5 resources and ratios are reported separately.
