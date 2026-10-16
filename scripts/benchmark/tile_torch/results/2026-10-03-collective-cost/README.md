# Collective cost profile checkpoint, 2026-10-03

This checkpoint freezes a small, optional worker-hint model before independent heldout testing. **The twelve heldout configurations have not been measured.** It records completed native measurements and offline model diagnostics; it does not claim that deploying the profile has already passed GPU correctness or improved independent workloads.

The v3 leave-one-geometry-group-out (LOGO) policy ratio is **0.97396 with equal weight per geometry group** (about 2.60% lower than always using the default). The secondary row-weighted ratio is 0.94198; repeated shapes, dtypes and algebra variants make that a different weighting, not stronger evidence. There are 32 training rows from two 16-configuration inventories, grouped into 18 logical IR geometries. Each complete geometry group is removed before fitting its fold.

| Offline diagnostic | Training rows / LOGO groups | Nondefault LOGO choices | Observed policy result |
|---|---:|---:|---|
| v1 ridge model with empirical margin | 16 / 7 | 0 / 16 | All default; ratio 1.0 |
| v2 tree with leaf-support and empirical-margin gates | 32 / 18 | 0 / 32 | All default; ratio 1.0 |
| frozen v3 tree, uncalibrated point score | 32 / 18 | 10 / 32 | Geometry-weighted 0.97396; row-weighted 0.94198 |

V3 retains three selected regressions; the largest is 1.01971 (about 1.97%). None of the selected LOGO regressions exceeds 5%, but this is not a risk bound or a claim about unseen inputs. The largest absolute prediction residual is 0.43449 in log time. Always requesting worker8 regresses on 22/32 rows, with a worst ratio of 1.56439. Every positive and negative observation, decision and residual remains in `models/v3.json`; v1/v2 abstentions and predictions remain in their corresponding files.

The profile uses nine generic IR features: programs per SM, largest logical Tile, explicit live Tile bytes, element work per collective input, contribution width, independent extent, SUM/MAX input fractions, and nominal read-plus-write bytes. These are logical predictors, not measured physical registers, occupancy or memory traffic. A depth-two tree gives each training geometry equal total weight. It selects worker8 only when the leaf point score is strictly below `log(0.95)`; prefix scans and unmeasured MINIMUM algebra default to zero. There is no confidence guarantee or support-box gate. The runtime profile is opt-in, strict-math, and restricted to the declared SM89/24-SM/CUDA13.4 environment. A worker hint is a compiler request, not an observed thread count.

The four motivating SUM/MAX contrasts mix algebra with masked versus fully proven in-bounds accesses, FP16/BF16/FP32 storage and reduction-axis aspect ratio. These measurements do not isolate their physical causes. V3 was frozen before examining independent heldout outcomes; changes based on future failures require a new training/validation split.

## Evidence and timing scope

`extra48/checkpoint.json` preserves all 48 validated extra visits, including correctness/output rechecks, input/oracle/source hashes, seven graph samples, graph calibration/replay counts, cold timings, default recheck drift and final cohort gates. It is a sanitized projection of the completed earlier validation, not a new check of today's source or binaries. The default recheck range is approximately -0.413% to +0.074% for these extra cases. The original16 checkpoint is published separately in the preceding `2026-10-02-collective-ir` results directory; its calibration-relevant fields and all native samples are also included here.

`native-samples.csv` contains 112 native visits (64 original plus 48 extra), each with all seven samples and source hashes. This count is not 112 independent workloads. Both training datasets use the frozen adaptive graph replay protocol, 100 operations per graph, seven samples targeting 100 ms, and a 500 ms warmup. Replay caps and actual spans are retained in the extra checkpoint. Cold compile/capture measurements remain separate. The extra runs are native-only and introduce no new Torch comparison. Interrupted v1 worker attempts and the failed WinError5 structural cohort remain excluded; successful child output never rescues a failed final aggregate.

## Reproduce the offline result

Use Python 3.11 or newer and the NumPy version pinned in `requirements.txt`. No Torch, CUDA initialization, compiler or GPU is required. From this directory:

```powershell
python -m pip install -r requirements.txt
python -B reproduce.py
```

The verifier checks every published file's hash, reruns v1/v2/v3 on the public inputs and requires exact equality of the numerical projections: training features/samples, fitted models and tree nodes, full-training choices, and every LOGO fold's predictions and decisions. The original scripts are byte-identical copies under three sibling directories in `code/`, so their original dependency-hash checks still work. No import-path patch or model refit is hidden in publication.

For a standalone v3 report in a new local directory:

```powershell
python -B code/collective-worker-calibration-v3-prep/calibrate_v3.py --summary inputs/summary-0.json --summary inputs/summary-1.json --device-facts inputs/device.json --output local-reproduction-v3
```

That newly generated report includes the new public input hashes and local output provenance, so its opaque profile ID differs from the original. Its numerical tree/scores/decisions reproduce the frozen model. `models/v3-cpp-parity.json` retains all 32 original C++ parity vectors and decisions; `cases-heldout12-unmeasured.json` contains only the frozen independent input inventory.

## Original versus public hashes

The original v3 profile ID is `357c15e0c745797a75e8d11c9de9fafc0ab204f71d1619fedd9251323e7abb35`. It is provenance for the original local report, not the SHA256 of any redacted public document. `receipts.json` separates original source artifact receipts from `public_files`, whose hashes describe the actual published bytes. Device UUIDs and absolute user paths were removed and inputs were compacted; the public JSON files therefore have new hashes. Original script bytes are preserved, and all public text uses LF. Model confidence remains uncalibrated and independent heldout validation is still pending.
