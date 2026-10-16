# Worker model v2 specification, frozen before new calibration results

This revision follows the completed original worker16 training diagnostics.
It does not read the new independent-axis16 outcomes while being prepared and
does not read the external heldout12 inventory or outcomes. The old v1 script
and its recorded all-default result remain unchanged.

Candidate set is fixed to0/8. Any IR prefix contribution keeps0; all prefix
measurements remain in reports as semantic exclusions, including regressions.
Non-prefix measurements, including every tie and regression, train a numeric
axis-aligned regression tree with maximum depth2. Each leaf must contain at
least two distinct logical geometry groups. No operation/benchmark string,
explicit tensor shape lookup, precision label or hand-selected physical cutoff
is a predictor. Thresholds are midpoints of the training numeric feature values.

The four features are log2(1+programs/SMs), log2(1+largest explicit Tile elements),
log2(1+peak explicit Tile bytes/(warp size*4)), and
log2(1+elementwise element-work/total collective input elements). They describe
logical demand, state and arithmetic mix, not actual registers or occupancy.
The target is ln(median worker8 / median default0). Each geometry group has equal
total fitting weight, so repeated controls and storage/arithmetic variants do
not get extra voting weight merely through row count. Splits minimize weighted
squared log-score error; ties use feature order then lower threshold.

Geometry grouping remains (programs, sorted unique (width, independent) pairs).
Outer leave-one-group-out diagnostics refit the entire tree, support bounds,
calibration residuals and decision policy without the held group's outcomes.
An inner group holdout records every residual and whether the held point was
inside its learned leaf's coordinate-wise support box. The report preserves
the global maximum absolute residual, including unsupported predictions.

Each full-training leaf gets a separately labeled empirical envelope from inner
predictions that were themselves inside their training leaf's support, mapped
to this leaf by their features. At least two distinct calibrated groups are
required. Its margin is the maximum absolute supported residual plus twice the
largest timing log deviation among all training rows in the leaf, plus the
largest default0 recheck log drift there. Any leaf row lacking recheck, drift
above5%, insufficient calibration coverage or support-box violation keeps0.
A leaf's upper log score must beat ln(.95) before diagnostic8 is emitted.

This is an explicitly conditional support/envelope policy, not a smaller global
error quantile or a statistical confidence interval. Unsupported/bad residuals
are never erased; their exclusion from a leaf's eligible prediction population
is recorded. Final outer diagnostics and subsequent replicated heldout
validation must establish that this conditioning does not hide regressions.
Deployment remains disabled until that validation; this can be completed in the
current task. There is no hyperparameter search, outcome-based sample pruning,
or automatic relaxation if the policy still abstains.

Inputs may contain the original0/4/8/0 summary and additional0/8/0 native-only
summaries. Torch is neither required nor synthesized for new calibration.
Each summary must be completed and fully validated; identities, paired source,
fixtures and default recheck must remain intact. Implementation receipt identity
and native measurement configuration must agree before merging. Same logical
case repeated across independent runs is preserved and grouped together.
The original summary writer omitted zero structural settings; only absent
`native_scan_chunk`/`native_independent_axis` are normalized to explicit0.
Present nonzero or malformed settings reject. Exact original input JSON remains
in the output receipts; no other configuration field is silently normalized.
