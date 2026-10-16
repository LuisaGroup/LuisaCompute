# Same-domain shared Tile SSA: balanced GELU-add A/B

September 5, 2026; Apple M1 Max; macOS 26.6.2; PyTorch 2.14.0.
Assessment: **share with the timing and eager-graph caveats below**.

The fused mapper now scalarizes a proved, compiler-owned same-domain producer
chain before constructing the program × element grid. Each shared value has
one scalar definition per worker. No kernel-name matching, manually exposed
scratch, or expression recomputation is involved.

## Question and result

Does shared SSA still force elementwise execution back to one thread per
logical program? Previously yes; this extension removes that boundary for
pointwise producers. In four counterbalanced rounds, the only A/B option
changed is `--element-grid reference` versus `auto`. Both use immutable input
views, the same full Tile-SSA preservation policy, the same binary/compiler,
FP32 tanh-approximate GELU(A+B), and preallocated inputs/output.

| Shape | Reference µs | Fused µs | Paired reference/fused median [min, max] | Paired fused/Torch |
|---|---:|---:|---:|---:|
| 1×127 | 84.540 | 2.585 | 32.701× [31.008, 33.326] | 0.280× |
| 17×257 | 141.631 | 2.637 | 52.636× [50.877, 57.412] | 0.270× |
| 128×1024 | 52.764 | 5.293 | 9.908× [9.644, 11.792] | 0.325× |
| 4096×256 | 75.641 | 19.929 | 3.834× [3.767, 3.886] | 0.326× |

Times are medians of per-round batched p50 **synchronized host-wall** times.
Ratios are medians of paired per-round ratios, not ratios of the displayed
medians. Ranges are observed round extrema, **not confidence intervals**.
All 32 native and 32 Torch outputs pass the complete FP64 oracle. No candidate
selection or best-of-N timing enters this replay.

Torch uses eager `add.out` followed by `gelu.out(approximate=tanh)`, with both
intermediate and final buffers preallocated. Thus the Torch comparison is one
fused native graph versus two eager operations; it is not a Torch-compile or
universal activation-performance claim. These older measurements contain no
GPU counters and must never be relabeled as pure kernel time.

## Proof and regression scope

Admitted producers have versioned compiler provenance, matching static
domains, unique dominating definitions, exact pointwise local read/write
coordinates, no escaping storage, no effects, and no manual-memory marker.
Nonzero loop minima are normalized. Negative input origins and ragged tails
preserve guarded zero-fill. Neighbor/transpose reads, conditional/repeated
writes, unmarked/manual allocations, changed inputs, explicit execution
bindings and mismatched domains decline fusion. Exact row-reduction options
cannot silently fall through to the element-grid family.

The emitted GELU shader has one `tile_storage_6_element` definition for A+B
and no full shared-producer Tile array. Tests separately use a shared `exp`
producer and check that source contains one `exp` call, with two scalar
temporaries and a full numerical oracle. Tanh's TVMx expansion contains
several source-level exponentials; we do not infer hardware instruction counts
from those strings.

## Reproduction and fingerprints

Frozen schedules: [reference](../m1-max-20260905-shared-element-reference/results.json)
and [candidate](../m1-max-20260905-shared-element-fused/results.json).
The complete replay, every raw sample and source hash are in
[results.json](results.json); [results.md](results.md) is the generated table.

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260905-shared-element-reference/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-shared-element-fused/results.json \
  --output /tmp/shared-element-replay \
  --operations gelu_add --rounds 4 --samples 7 --sample-ms 40 --warmup-ms 100 \
  --capture-sources \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_compiler.dylib
```

The archived binary SHA256 is
`7921a45639a218bc83e60981aadfd58dd9b55dfc106c023747da859bf0a37a9c`;
TIRx bridge `f9323401b9b6e476ca0561e2512ec5c010354e79e742b37aaca081beeba896c3`;
TVMx compiler `44a277c13f8400925b6eb7170148b0c0e03ca727a70d29f33713d0bc0c8d5c89`.
The replay's full artifact map is unchanged before/after. Subsequent exact-
constraint guards and timing-harness changes produce different binary hashes;
this report records the actual experiment, not a fictitious rebuilt binary.

No build, syntax check or other benchmark ran concurrently with this replay.
The later GPU/dispatch smoke is a separate measurement and is not pooled here.
