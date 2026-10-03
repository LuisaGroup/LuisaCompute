# Eligibility queue stopped in default Tile compilation

This is a preserved failed cohort, separate from the successful automatic heldout measurements and the fixed/automatic instability diagnostic.

The six-case default stage completed with five native cases passed and one native compiler failure: BF16 scan `[3,8192]`, Tile `[4,8192,1]` (`scan-eligibility-3x8192-bf16-br4`). The retained result reports `tileiras` exit `0x00000005`, “failed to compile Tile IR program”. All six Torch processes passed. The queue stopped after this default stage: **automatic and recheck were never started**.

No speed ratios or cohort metrics are produced. The default failure is not attributed to the cost path, and the log does not establish resource exhaustion. The exact failed native result, all six case statuses, original failed journal and source receipts remain in `eligibility-failure.json`. Successful rows have not been promoted into a successful cohort or pooled into the formal result.

Run `python reproduce_failure.py` to check the compact public bytes and failure classification. It verifies provenance relationships/statuses, not the omitted tensor payloads. Existing `automatic.json`, `automatic-outlier.json` and their public manifests remain byte-identical to the prior package.

Package entry points: `README-automatic.md` for actual automatic performance; `README-outlier.md` for separate runtime instability; this file for incomplete eligibility evidence. Original local receipt hashes and the compact public hashes identify different byte streams.
