# CPU input-view option: other-operator smoke coverage

Add, row sum, and softmax each ran at four sizes with fresh native/Torch full
output validation: **24/24 outputs valid**, 3,164,132 checked elements. Maximum
absolute error is 3.537064198824713e-9 (softmax); all 12 generated LLVM hashes
were independently verified.

This is a single-pass correctness/exploratory timing run, not a counterbalanced
causal performance comparison. Preserve all timings, including slower Tile
softmax/reduction cases; do not treat individual wins as established speedups.
The input-view option does not promise that every eligible-looking operation
actually forwards; legality may preserve the original snapshot.

Configuration: CPU worker binding, automatic vectorization, 8192-byte stack
budget, 64 logical pack lanes, input views enabled, requested eight threads,
100 ms warmup, five samples × 20 ms. Full commands, raw timings, numerical
errors, source hashes, and software identities are in [results.json](results.json).
The [four-round GEMM A/B](../m1-max-20260904-cpu-views-replay/notes.md) and
[six-order library comparison](../m1-max-20260904-cpu-views-system/notes.md)
provide the repeated performance evidence separately.
