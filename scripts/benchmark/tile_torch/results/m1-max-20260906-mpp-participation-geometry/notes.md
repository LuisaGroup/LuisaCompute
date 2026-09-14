# Matched MPP participation: no universal replacement

## Technical summary

The existing handwritten MPP benchmark executes either independent 32×32
subgroup operations or one whole-group operation at identical output geometry
and thread count. All **70 complete outputs** pass. Collective participation
helps some configurations, but it is not a consistently faster replacement:
the 4096³ and ragged cases regress in every matched geometry. This screen
does not change production TIRx/native lowering, default plans or coefficients.

The canonical narrative remains in the existing Sphinx performance section;
this file is its reproducibility record, not a second report application.
The technical report roles are summary; evidence below; scope/methods in the
protocol; robustness and next questions below. The exact geometry-by-shape
lookup belongs in a table, with both orders retained rather than a chart of
selected minima. No performance acceptance or causal cache model is inferred.

## Scope and evidence

The [predeclared protocol](protocol.md) fixes five FP32 shapes, three matched
geometry pairs and direct MPS. Two rounds reverse arm and shape precedence;
there are five samples per GPU/E2E batch/single phase, 20 ms requested windows
and 100 ms warmup. Every MPP invocation processes whole physical K with
dynamic inline tensors, cooperative output, fast math and relaxed precision
disabled. MPS output is preallocated. There is no Torch or TIRx arm here.

[Raw results](screen/results.json) retain all cases, sample arrays, commands
and full FP64-oracle receipts at atol=rtol=1e-4. They cover **1,335,054,350
checked output elements**, with zero maximum error for the deterministic dyadic
inputs. Sixty generated MSL files are fingerprinted. The standalone programs
record no-counter command-buffer GPU intervals, not isolated kernel times;
host E2E is an independent timing phase. Unlike the TIRx timing helper, these
receipts retain already-normalized GPU sample arrays, not raw nanosecond
intervals for an independent denominator reconstruction.

The [independent audit](screen/audit.json) validates the exact arm geometry,
participation, precision flags, source stability, sample medians, visit order,
and recorded complete-output counts. Five deliberately malformed reports are
rejected. It preserves all four timing views and both rounds. Run it with:

```sh
uv run --offline --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260906-program-order/audit.py participation
```

## Interpretation and next questions

The 8192³ 128×32 collective/independent GPU ratios are **0.827 and 0.854**,
but the independent 128×64 arm is faster than that collective arm in both
rounds, and both remain behind MPS. On 256×11008×4096, collective 64×64 helps
its matched independent arm, while the other two pairs do not consistently
improve. A match-local win therefore does not establish a better global plan.

Substantial session variation remains: the 512³ MPS control changes from
49.232 to 61.531 µs between orders. No application was closed or clock/GPU
idle state established. Two rounds are exploratory, not confidence intervals,
held-out calibration or accepted library parity. A future whole-group atom
needs numerical/participation contracts and a separate frozen replay before
promotion. The next generic experiment instead exposes bounded program-grid
traversal jointly with physical K, without changing per-group participation.

The before/after executable and driver hashes are equal **during this run**.
The report predates subsequent `run.py` traversal options; its historical
hash is intentionally preserved and is not claimed to equal today's script.
