# Model-selected resource and execution plans

The profile and validation shapes were frozen in `47314e616` before these
measurements. For each of twelve cases, two separately captured reload/cache
configurations use the service-policy C++ width solver. The staged/JIT driver
selects by summed whole-kernel **model score**, never by the recorded GPU or
host timings, then freshly captures and runs the winner. All 72 executed
native/Torch outputs pass; the independent audit reconstructs all 32 width
scores for each resource choice and verifies the selection.

[results.json](results.json) preserves every trial, selected/fresh plan,
source hash, raw sample and exact profile argument. This is not compile-only
search and does not measure regret against an exhaustive timed optimum.
See the [complete report](../m1-max-20260905-service-policy-validation/notes.md)
for the later four-round acceptance replay, including small-case regressions.
