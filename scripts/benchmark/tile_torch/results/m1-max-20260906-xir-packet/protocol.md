# SIMD packet-index proof: frozen compiler comparison

Predeclared after the 128-cubed smoke test, before the balanced replay.
This is a generic XIR-to-Schedule proof change, not a GEMM-name fast path or
a cost coefficient fit. The planner and Tile program are identical in both
compiler arms. LLVM math remains strict; no FMA/reassociation change.

Baseline: rebuild commit `47beacf2f` with the pre-existing unrelated dirty
worktree, then copy the complete build `bin/` to
`/tmp/luisa-xir-packet-baseline.FFXCie/bin` before changing production code.
Candidate: the same LLVM 21 build with only the new Schedule proof. The
pre-existing TIRx `cooperative.cpp` change is not part of this optimization;
neither CPU arm executes TIRx. Hash both executable/library sets before and
after timing. Keep the exact source diff alongside the report.

GEMM M,N,K: `32,32,32`; `128,128,128`; `512,512,512`; `1024,1024,1024`;
`128,2048,512`; `127,193,61`. Fix Tile `1,1,8`, planned root order/block
packing, W8 and eight CPU workers. The ragged case is a negative performance
control: its quotient/remainder must not receive aligned-packet annotations.
Use six balanced permutations of baseline/candidate/eager Torch, five
samples, 20 ms target and 100 ms warmup. Rotate shape order each round.

All times are synchronized **host-wall dispatch** throughput and latency,
excluding JIT/allocation/transfers. They are not CPU kernel-only timings.
Do not run builds, tests, GPU work or profilers during timing. Compare paired
rounds, not the earlier smoke-test measurements. Preserve every failure and
regression. Full-output FP64 validation uses the existing dyadic inputs and
unchanged `atol=rtol=1e-4`; also run the non-dyadic guarded Runtime tests.

Inspect generated LLVM IR for the annotations' effect. IR instruction counts
are not claims about final CPU ISA. External Torch ratios remain distinct
from the before/after compiler speedup. Report compile time and both dispatch
metrics; do not call a reduction in dispatch overhead a kernel-only gain.

MPS capture/profiling is a separate diagnostic workload, never mixed into
these acceptance samples or presented as an uninstrumented GPU time.

## Post-replay conservative refinement

The initial `replay/` completed all 108 checks and its `audit.json` receipt
verified 38 unchanged artifacts. A subsequent source review found that
Schedule IR accepts non-power-of-two widths too. The new uint32 packet
alignment proof is now explicitly limited to power-of-two widths, with a W3
negative test. This leaves the W8 realization unchanged but changes the
compiler artifact. Preserve the first report as historical evidence; rebuild
and repeat the identical predeclared protocol in `final-replay/`. Do not
substitute its old timings for the final binary or overwrite its old receipt.
