#!/usr/bin/env python3
"""Audit raw Metal controls and alpha-renamed CPU TBAA object labels."""
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_tbaa(text):
    # Only native TVM's object-address label in a complete metadata definition
    # may vary. Keep metadata IDs, all references, and the label equivalence
    # classes. Do not strip instructions, arbitrary hex constants or metadata.
    labels = {}
    pattern = re.compile(r'(^!\d+ = !\{!")(0x[0-9a-f]+)(", !\d+, i64 0\}$)', re.MULTILINE)
    def replace(match):
        label = labels.setdefault(match[2], len(labels))
        return match[1] + f"tbaa_object_{label}" + match[3]
    normalized = pattern.sub(replace, text)
    return normalized, len(labels)


def main():
    report = json.loads((ROOT / "controls.json").read_text())
    assert report["artifacts_unchanged"] is True and len(report["runs"]) == 7
    assert all(row["passed"] for row in report["runs"])
    assert sum(row["expected_cases"] for row in report["runs"]) == 32
    rows = []
    for row in report["source_controls"]:
        cohort, name = row["cohort"], row["name"]
        suffix = ".ll" if cohort == "cpu" else ".metal"
        old_path = ROOT / f"control-{cohort}-old/sources" / (row["old"] + suffix)
        new_path = ROOT / ("new-reference" if cohort == "large-old" else f"control-{cohort}-new") / "sources" / (row["new"] + suffix)
        old, new = old_path.read_bytes(), new_path.read_bytes()
        assert digest(old) == row["old"] and digest(new) == row["new"]
        result = dict(cohort=cohort, name=name, raw_identical=old == new)
        if cohort == "cpu":
            a, count_a = canonical_tbaa(old.decode())
            b, count_b = canonical_tbaa(new.decode())
            assert count_a == count_b > 0 and a == b, name
            result.update(tbaa_alpha_equivalent=True, unique_object_labels=count_a,
                          canonical_sha256=digest(a.encode()))
            # A changed instruction or changed alias equivalence relation
            # cannot pass merely because address labels are normalized.
            assert canonical_tbaa(old.decode() + "\nret i32 123\n")[0] != b
        else:
            assert old == new, name
        rows.append(result)
    assert len(rows) == 18
    # Labels must remain distinct and repeated labels must remain equal.
    pattern = '!1 = !{!"0xaaa", !2, i64 0}\n!3 = !{!"0xbbb", !2, i64 0}'
    changed = pattern.replace("0xbbb", "0xaaa")
    assert canonical_tbaa(pattern)[0] != canonical_tbaa(changed)[0]
    result = dict(passed=True, control_graph_runs=32, native_and_torch_outputs=64,
                  artifacts_unchanged=True, artifact_count=len(report["artifacts_before"]), controls=rows,
                  interpretation="14 byte-identical Metal controls; 4 CPU controls equal after bijective TBAA object-label renaming only; no timing claims",
                  original_receipt="controls.json retains its raw-hash failure, caused by nondeterministic CPU metadata labels")
    (ROOT / "controls-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
