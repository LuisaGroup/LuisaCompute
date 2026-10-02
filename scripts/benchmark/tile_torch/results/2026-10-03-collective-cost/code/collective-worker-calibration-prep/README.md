# Offline worker-hint calibration diagnostic

This ignored tool fits **relative log scores**, not latency, hardware thread
counts, register pressure or occupancy. It does not run a process, compile code,
access a GPU, change a runtime policy or select a production hint. Every report
and every prediction explicitly keeps `production_hint=0`.

```powershell
.deps/torch-cuda-venv/Scripts/python.exe .deps/collective-worker-calibration-prep/test_calibrate.py
.deps/torch-cuda-venv/Scripts/python.exe .deps/collective-worker-calibration-prep/calibrate.py --summary .deps/COMPLETED_SUMMARY/report.json --device-facts .deps/oct02-collective-device-facts.json --output .deps/NEW_CALIBRATION_DIRECTORY
```

The summary must be the `completed_validated` result of the separately reviewed
worker sweep summarizer: default0, one explicit4 and one explicit8 cohort,
optional final default0 recheck, all final cohort and case gates passed. Missing
facts, a failed cohort, changed source/fixture, malformed timing or any rejected
row rejects the entire fit. Slower samples and near ties are all retained. The
tool trusts that summary's raw-file validation; it does not repeat expensive
oracle/file checks while GPU measurements are running. Exact summary and device
JSON bytes are copied into a new output directory, including on rejection;
input/script hashes, NumPy version, source/fixture identities, raw seven-sample
vectors, medians, all ratios and all logical facts remain in `report.json`.

Device JSON schema1 uses positive integers `compute_capability`, `sm_count`,
`warp_size`, `max_resident_warps_per_sm`, a nonempty `identity` object and an
`evidence` list of declared path/SHA256 receipts. The root's
`oct02-collective-device-facts.json` matches this schema. The diagnostic accepts
only the current emitter's admitted SM89 hint experiment. Maximum resident
warps are only a reference normalization budget. Declared evidence is preserved;
the tool does not assert that it independently observed the timed device.

Eight input features follow the earlier model design. They use only numeric
`collective-work-v1` and `collective=kindN:widthN:independentN` records from the
real native realization plus those device facts. The IR kind enum determines
the prefix fraction. Benchmark names, operation strings, external tensor
shapes, precision names, identifiers and source text are not scoring features.
Case records and source hashes remain solely for identity/provenance.

For worker4 and worker8 separately, the target is `ln(median_worker/median_0)`.
Ridge regularization is predeclared at alpha1 with an unpenalized intercept;
features are standardized using training rows only. There is no parameter
search. Reports expose standardized and original-feature coefficients,
normalization and coordinate-wise support minima/maxima. A support box is a
limited extrapolation guard, not proof of joint feature coverage.

Leave-one-group-out diagnostics group rows by the IR geometry signature:
program count and the sorted unique set of `(collective width, independent
elements)` pairs. Storage and arithmetic variants with the same geometry stay
together. For each outer held group, fitting, standardization, support bounds
and uncertainty calibration see **only the remaining training groups**. Inner
group holdouts determine the largest training prediction error. The empirical
margin also includes twice the largest training timing log deviation and the
largest training default0 recheck drift. This conservative envelope is not a
confidence interval: correlated timing samples and one run per cohort do not
establish calibrated statistical coverage. Outer held measurements are used
only after the decision to report error and observed regression.

At least four training geometry groups and complete inner residuals are needed
for a nonzero diagnostic candidate. Missing recheck, training drift above5%,
out-of-support features, overlapping candidate envelopes, or lack of a5%
improvement in the conservative upper envelope all retain diagnostic0. A
nonzero diagnostic candidate still never enables production selection. The
report preserves worst observed regressions, all fold coefficients and each
fold's exact training case identities. Full-data fitted coefficients are labeled
training diagnostics; they are not validated deployment parameters.

The external heldout12 inventory and outcomes are never loaded by this tool.
Freeze the complete tool/policy/profile identity before any later heldout
evaluation; do not use heldout measurements to refit, choose alpha, change the
margin or select a different worker per case. Deployment requires subsequent
replicated heldout validation, which may be completed as part of the current task.

`test_calibrate.py` uses plainly synthetic data to test parser/receipt admission,
feature equations, geometry grouping, preservation of regressions, nested
holdout isolation, support/default behavior, fixed-ridge scoring and rejection.
Its numbers and fitted coefficients are not performance evidence. No real fit
is included while the completed production worker sweep is unavailable.
