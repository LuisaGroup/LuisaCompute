# Other CPU operators after full-vector guard specialization

All **24 complete outputs** pass: Tile and Torch for four shapes each of
add, row sum and softmax. The run checks 3,164,132 elements; maximum absolute
error is 3.5370642e-9 (softmax). All twelve archived LLVM hashes were verified.

This is a **single-pass correctness/performance smoke run**, not a controlled
before/after comparison. Five samples target 20 ms after 100 ms warmup, with
eight requested CPU threads, worker binding, automatic vectorization,
8192-byte stack budget, 64 logical pack lanes and input views. No concurrent
build, test or profiler. Full shapes, timings and errors are in
[results.md](results.md) and [results.json](results.json).

The slow cases remain visible: add 17×257, sum 17×257 and 64×4096, and softmax
128×1024/64×4096 are behind Torch in this run. This does not establish a
regression caused by the new pass, nor operator-wide parity. Guard versioning
is defined on independent element domains and does not create a special GEMM
primitive. The [GEMM A/B](../m1-max-20260905-cpu-guards-replay/notes.md) supplies
the controlled performance evidence for the current change.
