# Realization-derived matrix work

The baseline is commit `71bd1d6a5` plus the unchanged user-owned shared-only
barrier edit. Full TVM/Luisa builds completed before copying the benchmark and
its Tile/bridge/TVM libraries into `/tmp/luisa-mpp-realized-baseline.mbTNjX`.
Do not attribute that barrier edit to this change or commit it with this work.

The defect is a mismatch between modeled and emitted work: bounded MPP input
views can issue a smaller physical K than the captured tile, and resident /
direct accumulators remove scalar domains that the score still charges.
Derive those features from the existing matrix and recurrence proofs, preserve
the nominal domain for legality, and apply candidate-dependent work before
Pareto pruning. Do not fit coefficients, change numerical permissions, bypass
resource limits, or add operator-name / shape dispatch rules.

Verification: analytical versus independently enumerated work, one/multiple K
iterations, transposes, partial M/N/K, nonzero initial state, staged/global
inputs, retained scalar sinks and observation boundaries. Keep full CPU/Metal
matrix/execution/operator regression tests. Check emitted source against the
new diagnostics; a smaller modeled number alone is not a faster kernel.

Performance: freeze the old and new model-selected schedules, then compare
them without retuning. Test aligned and ragged square/rectangular shapes and
physical K below/equal/above the captured tile. Keep GPU command-buffer batch
and single-call intervals distinct from synchronized host E2E. Torch and direct
MPS use preallocated FP32 outputs and the same complete-output FP64 oracle.
Counterbalance provider/variant order and retain every reversal. Concurrent
desktop activity makes timing diagnostic unless repeatability controls support
acceptance. No isolated-kernel or universal parity claim follows from a finite
cohort. Results belong in the existing Sphinx architecture/performance sections.

Registered before any selection: eight shapes, in order, 512³, 4096³,
1025×1025×1024, 4096×4096×11008, then held-out 257×769×113,
2049×4097×1025, 4097×4097×4096 and 8192³. Same fifteen candidates for both
stacks: output blocks 32×64, 64×64, 128×32, 128×64, 64×128 crossed with
BK=128/512/4096, automatic thread solver, copy batch/pipeline window one.
Select on the model only; single-sample timings during selection are not
acceptance. Freeze those choices for six fresh-JIT rounds, nine samples,
30 ms windows and 100 ms warmup. Model coefficients are never fitted to this
cohort, including the held-out half.
