# Target-complete reduction widths: implementation checkpoint

September 5, 2026; Apple M1 Max; FP32; TileIR to TIRx Metal.

## Structural findings

The test/benchmark Runtime previously selected Metal with only its 32-lane
subgroup attribute. TVM's installed target definition supplied the remaining
default `max_num_threads=256`. Independently, the bridge restricted its
reduction family to eight subgroups and automatic powers of two. Thus legal
96-thread candidates were omitted automatically, and 512/1024-thread
candidates were rejected before device compilation.

The Runtime now queries `DeviceAPI::kMaxThreadsPerBlock` and forwards the
result as both Metal target thread-capacity attributes. The native benchmark
records it as `metal_max_threads`. The installed Metal runtime implements
that query with the device's `maxThreadsPerThreadgroup`; a compiled pipeline
may still impose a tighter resource-dependent constraint.

The bridge enumerates every whole-subgroup cooperating width through
`min(32, target_max_threads/32)`. The 32-subgroup cap is algorithmic: the
second collective assigns one lane to each partial. Packed independent rows
retain the existing separate 1..8-program family, so there are at most 39
automatic candidates. Search-budget exhaustion fails explicitly; exact
widths do not need the full automatic budget. The reduction-tree numerical
permission, ownership proofs, resource bounds and no-fallback exact requests
are unchanged.

## Cost-policy features and remaining work

Backend policies now receive physical threadgroup count, useful scalar
element count and useful lane-work fraction, in addition to worker width,
packing, consecutive elements, scalar rounds and private/shared storage.
Plans/JSON expose corresponding facts. They are not measured occupancy.
The default scalar-round coefficients are unchanged: this checkpoint repairs
the admissible family and target information, not its active-group/issue cost
model. Widths alone do not prove a performance win.

## Verification

Full selected-tree builds succeeded. CPU/Metal execution tests pass,
including 14 new V=1/4 softmax cases covering automatic mapping and exact
96/160/224/288/512/1024-thread layouts with ragged tails. Independent ownership
counts check private storage and useful-lane fractions; a custom backend
policy proves every legal width is presented, including non-powers of two.
Budget exhaustion and over-limit exact requests fail closed.

The old auto-layout assertions are replaced with independent exhaustive
evaluation of the documented objective. A separate matrix reference fixture
now explicitly requests its original 256 threads so it continues exercising
the atom-wave loop on devices supporting 1024 threads; its source/numerical
assertions are retained. Benchmark metadata tests pass **83/83**. The final
full Tile rerun passes **31/33** in 101.98 seconds; the only failures are the
two pre-existing cooperative/memory source-assertion conflicts with the user's
unowned `mem_flags(2)` edit. Their numerical checks pass. See [full log](tests.log).
The local edit and those assertions were not changed or submitted. Project
clangd checks pass for all four changed translation units (both shared headers
are checked through their consumers).

Finite search and independent GPU/E2E replay will be recorded separately.
No timing result or universal Torch/MPS performance claim is inferred from
these correctness checks.
