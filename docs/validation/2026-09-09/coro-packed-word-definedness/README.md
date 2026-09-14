# Packed coroutine words require a defined seed

## Proof obligation before the correction

For suspension edge `e` and packed word `w`, let `D(e,w)` be the mask of
Boolean lanes whose logical values the continuation stores, and `L(e,w)`
the mask live on the outgoing edge, both from the certified distilled CFG.
Only `P(e,w) = L(e,w) & ~D(e,w)` must pass through unchanged. The physical
output is therefore

```
new_word = (P != 0 ? old_word & P : 0) | encode(stored_values)
```

For every live lane in `D`, the output equals its new Boolean value; for
every live lane outside `D`, it equals the incoming value. Other bits are
dead. If `P` is empty, the expression has no incoming-word dependency.
If `P` is nonempty, materialization already includes this physical word in
the continuation input set. This is a whole-word transfer requirement, not
an additional logical frame value or a frame-pool initialization policy.

The previous split lowering unconditionally loaded the old physical word,
then applied an AND/OR sequence for each stored bit. A new local CoroFrame
is instantiated with UNDEFINED; no payload is required at coroutine entry.
Metal4 represents XIR UNDEFINED as LLVM poison. Bitwise operations on poison
do not make the overwritten bits defined, so the generated initial packed
word store can disappear. Changing backend undefined semantics, clearing the
pool, or disabling sorting/tail would conceal this violated precondition.

## Original-module reduction (before correction)

Psycles renderer implementation `9e3ba165`, Luisa `03a0f5158`, Apple M1 Max,
Metal4 graph, sorting and automatic tail enabled. A host-only diagnostic
driver repeats the same legal sample range `[0,64)` twice at 64x64 while
keeping the sampler's total sample count at 256. Production renderer
algorithms are unchanged. This is a replay-invariance check, not a renderer
reference or a performance benchmark.

- Keeping only physical field 90 stale (all other frame fields cleared)
  reproduces DiffCol relative RMSE 0.14322 and mean ratio 0.85914.
- This uint field is payload slot 83, packing five logical Boolean values
  into two interfering bit lanes. Its diagnostic name `_reg_101849` is not
  a unique logical variable identity and is not the sorting key.
- A readback immediately after the first entry launch in each replay gives
  low-two-bit histograms `[131072,0,0,0]` and
  `[29290,12377,89405,0]`. The second fresh entry retains previous payload.
- Tail-pool readback found no global pool modifications by the tail itself.
  Clearing the pool makes replay pass; sorting-off and tail-off controls
  also pass, but none is an acceptable compiler fix.

The permanent minimal XIR regression extends the existing six-interfering-
Boolean split test: its entry defines all live bits and must contain no
incoming-word load. Before correction: 1 failure among 38 tests / 386
assertions. After correction: all 38 tests / 386 assertions pass.

## Correction and focused validation

`coro_packed_word.h` computes one shared `D/P` projection for split lowering
and materialization's physical input metadata. Split starts from zero when
`P` is empty, otherwise masks the incoming word to `P`, then ORs encoded new
bits. Suspend and ordinary branch transitions use their exact outgoing
edge's liveness. No extra slots, pool clears, backend conditionals, changes
to undefined semantics, or renderer changes are introduced.

Host suites pass: split 38 tests / 386 assertions; materialize 15 / 113;
distill 56 / 493; dataflow 2 / 17. The existing all-schedulers suite
passes 20 / 100 on both Metal and Metal4. The new device regression passes
1 / 14 on both Metal and Metal4, checking
40 dispatches of 4096 instances: state-machine, wavefront, and graph with
AoS/SoA, sorting off/on, and tail off/on. Graph uses a 128-frame pool for
4096 instances and changes every Boolean truth-table case between replays;
all tail-enabled cases assert that the tail actually executes.

With all temporary SDK diagnostics removed, the original-module 64x64
same-range replay now gives:

| Pass | Relative RMSE | Second/first mean |
| --- | ---: | ---: |
| Normal | 5.7967241e-7 | 1.0000000661 |
| DiffCol | 3.4739715e-7 | 1.0000000059 |
| Combined | 4.3721874e-7 | 1.0000000033 |

All inspected values are finite. Sorting and automatic tail remain enabled.
Logs are retained under Psycles' ignored
`build-macos/benchmarks/2026-09-09/lone-monk-metal-schedulers` directory.

## Full original-module Cycles gate

The unchanged upstream film implementation `9e3ba165` is rendered again at
1920x1080 / 256 spp, four 64-sample batches, using the corrected Luisa build
with all temporary diagnostics removed. Both compact/populate switches are
1, fast math remains enabled, the main shader cache is disabled, and graph
sorting/automatic tail remain enabled. No builds, profiling or other device
work overlap this gate. A fresh version-verified Blender Cycles 5.2.1 Metal
reference uses the same source blend, frame, seed, integrator and 15 passes.

- Both actual EXRs have exactly 46 finite channels. Every pixel's integer
  sample count is checked against 256 by the original output path.
- Combined relative RMSE: **0.007764527**, mean luminance / Cycles:
  **0.999891523**, versus 0.230399849 / 0.775731912 in the original failure.
- DiffCol relative RMSE: **0.000850013**, luminance ratio **1.000004020**.
- The full-resolution Combined comparison removes the coherent dark regions;
  per-pass residuals, including maximum errors, remain in the report and are
  not waived as universal Cycles compatibility.
- Render-only wall: 328.193 s; session initialization: 106.560 s; fresh Cycles
  main-loop wall: 50.778 s. This is one correctness-gate observation, not a
  repeated performance conclusion or an improvement over invalid renders.
- The frame remains 7 stages / 91 fields / 456 B, capacity 131072.

Evidence: `packed-word-original-gate/graph/run-1/benchmark.json`, the
15-pass comparison `report.json`, `packed-word-original-gate-audit.json`,
binary/source hashes, and original logs in the directory above. This gate's
manifest records the pre-commit parent and dirty fix plus exact implementation
hashes. New-upstream performance measurements use a separate published cohort.
