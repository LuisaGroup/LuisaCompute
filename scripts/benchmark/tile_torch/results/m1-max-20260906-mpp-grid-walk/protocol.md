# MPP grid-walk diagnostic

Before timings, freeze the hand-MPP change that adds an optional physical
program permutation. The default remains the old 2D launch. Candidate 1 uses
a linear launch with the same row-major logical coordinates; candidates 4
and 8 traverse column-first within consecutive stripes of 4 or 8 program
rows. A partial last stripe uses its actual height. Group count, output
ownership, per-group work and all memory layouts are unchanged. No inter-group
execution-order guarantee is assumed.

Use fixed native MPP geometry 32,32,1,1,0,1,4,4: independent 32×32
SIMD-group outputs, effective group output 128×32, whole K, FP32 input/output,
no fast math or relaxed precision, alpha=1, beta=0, no packing. Compare all
four walks and direct MPS on shapes M×N×K = 1024×1024×1537, 4096³,
8192³, 4096×4096×11008, and 2049×4097×1025. The last shape covers a partial
program stripe and both output tails. Precede timing with small correctness
cases, including partial stripes and collective-scope MPP configurations.

Run two orders, rotating path order by shape; reverse each shape's exact path
order in the second round and reverse shape order. Five samples, 20 ms target
windows, 100 ms warmup, 300-second process timeout. Retain every failure and
all full FP64 output validation receipts, generated hand-MPP sources and
unchanged executable hashes. Input generation/allocation/transfers/JIT/oracle
are untimed. No other tests, builds or GPU profilers during timing.

Report raw no-counter GPU command-buffer batch and synchronized single-call
intervals separately from E2E batch and single-call time. These are not
isolated kernel timestamps. This is a two-order exploratory screen, not
balanced acceptance or evidence of cache hit rates. Do not install the search
minimum, mutate the production planner, or claim MPS/Torch parity from it.

Motivation: Apple, [MPP Programming Guide, §§2.3.3–2.3.4](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf),
discusses locality-preserving program walks and K synchronization. Its M5
heuristics are not assumed to apply unchanged to this M1 Max. The experiment
tests a simple bounded stripe permutation, not Morton order or a measured
cache model.

## Predeclared follow-up after the stripe screen

The completed `results.json` rejects both one-column stripe candidates: each
is slower than linear in both orders on every measured shape. Do not overwrite
that result or relabel it as a successful locality optimization.

Before further timing, add an orthogonal column span to the same bounded
permutation. Traverse row-major within 2×8, 4×16 and 8×32 **program** rectangles;
with the fixed 128×32 group-output geometry these describe square 256²,
512² and 1024² output regions. Partial row stripes and column rectangles use
actual dimensions, so there are no padded programs. Keep the linear and MPS
controls, the same five shapes, and the exact two-order protocol. Save this
exploratory follow-up separately as `results-rectangles.json`; do not combine
its timings with the earlier cohort. Run another 48 small complete-output
checks on the final binary before the new timings. This still is not a
production planner choice or a measured cache-hit model.
