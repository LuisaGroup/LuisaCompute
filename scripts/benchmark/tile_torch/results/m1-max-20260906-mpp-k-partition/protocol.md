# MPP K-partition diagnostic

Freeze this diagnostic before collecting timings. Source checkpoint is
`99b3e0579`, with the existing user-owned barrier flag 3→2 change in both
controls and candidates. No production changes or coefficient fitting in
this experiment. Complete the selected CMake tree before executing binaries.

Keep M/N block = 128×32, group = 128 threads, pipeline window = 1, copy batch
= 1, conservative subgroup fences, immutable MPP input views. Change only
the captured K block across 128, 512, 1024, 4096. Test M×N×K =
1024×1024×1537, 4096³, 4096×4096×11008. Repeat with both candidate order and
shape order reversed. Preserve actual subgroup plans and generated sources;
a change in the selected local rectangle is a confound to report, not ignore.

Use the existing run.py staged/JIT experiment, with direct MPS and eager
Torch controls, 5 samples, 20 ms target windows, 100 ms warmup, 8 CPU threads,
300-second process timeout, and full FP64 output validation. Compilation,
allocation, upload/download and oracle work are outside timing. The optional
timing helper records instrumented diagnostics and separate no-counter GPU
command-buffer intervals. Only the no-counter intervals are the GPU search
metric; E2E batch and synchronized single-call latency remain separate.

This is an exploratory two-order diagnostic, not balanced acceptance. The
framework's automatically selected minimum and fresh replay are retained but
are not promoted to production defaults. Do not claim a new implementation
speedup, MPS parity, or a cache/occupancy diagnosis from timing alone. Look for
consistent K sensitivity in both orders before choosing the next structural
change. Keep every failure and regression. Do not run builds, tests or GPU
profilers concurrently with the experiment. Fingerprint the five linked TVM
libraries before and after, in addition to the harness's Luisa artifacts.
