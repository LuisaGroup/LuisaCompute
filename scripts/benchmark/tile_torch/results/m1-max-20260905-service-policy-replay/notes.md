# Shape-held-out service-policy replay

Four counterbalanced rounds compare the frozen old automatic planner/reload
with the frozen model-only width/reuse choice on twelve cases. All 192
executed native/Torch outputs pass. Complete plans and generated source hashes
match the plan-collection records; 21 compiler/runtime/calibration artifacts
are unchanged. See [raw results](results.json), [tables](results.md), and
the [independent audit](../m1-max-20260905-service-policy-validation/audit.txt).

Nine cases improve in every no-counter GPU-throughput pair. At 768×6144,
softmax/RMSNorm/LayerNorm gain 1.360×/1.287×/1.231× GPU throughput. However,
37×1537 softmax and LayerNorm regress in every GPU/E2E-throughput pair and
small RMSNorm GPU is mixed. The profile remains opt-in. These are shape-held-
out observations, not operator/device generalization or isolated-kernel
timestamps. Full methods, separate E2E/single-call measurements and limitations
are in the [primary technical report](../m1-max-20260905-service-policy-validation/notes.md).
