# Frozen reference plans

These twelve valid plan-collection runs use the unchanged analytic reduction
policy, automatic whole-subgroup width, input reload and V=4/U=1/P=1 on the
four predeclared held-out shapes. All 24 executed native/Torch outputs pass.
The full metadata, raw GPU/E2E samples and exact native commands are in
[results.json](results.json); generated Metal sources are hash-addressed in
`sources/`.

These initial measurements are not the paired acceptance result. The
[complete service-policy report](../m1-max-20260905-service-policy-validation/notes.md)
includes the independently replayed gains and regressions, fitting boundary,
timing definitions, audit and frozen compiler artifacts.
