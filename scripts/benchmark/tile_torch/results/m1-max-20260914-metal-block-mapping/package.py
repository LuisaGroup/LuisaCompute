"""Package frozen Metal block evidence after all native writers have stopped.

No compiler, GPU work or numerical oracle is executed. The original raw tree is
read-only. Only receipt-listed tensor roots and plan/test-bound sources are
included; native binaries remain fingerprints, not a dynamic-loader closure.
"""

import argparse
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

sys.dont_write_bytecode = True
from audit import CASES, HELPERS, TEST_HELPERS, ORDER, fingerprint, helper_sources, read, require, sha, validate_plan

OUT = Path(__file__).resolve().parent
EXCLUDED = {"venv", ".venv", "__pycache__"}
BINARY_SUFFIXES = {".so", ".dylib", ".dll", ".exe", ".o", ".a"}


def collect_raw(root):
    files, excluded = {}, []
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        kept = []
        for name in sorted(dirs):
            path = parent / name
            if name in EXCLUDED:
                excluded.append(path.relative_to(root).as_posix())
            else:
                require(not path.is_symlink(), "refusing raw directory symlink: " + str(path))
                kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            path = parent / name
            require(path.suffix not in BINARY_SUFFIXES, "unexpected binary in raw evidence; do not silently archive it: " + str(path))
            files[path.relative_to(root).as_posix()] = fingerprint(path)
    return files, sorted(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--helpers-dir", type=Path, required=True)
    args = parser.parse_args()
    raw, helpers = args.raw.resolve(strict=True), args.helpers_dir.resolve(strict=True)
    require(raw != OUT and not OUT.is_relative_to(raw) and not raw.is_relative_to(OUT), "raw/package must be disjoint")
    for name in ("evidence.tar.xz", "package-inventory.json", "SHA256SUMS"):
        require(not (OUT / name).exists(), "preserve prior packaging attempt; output exists: " + name)
    plan, result, _ = validate_plan(raw)
    tested = helper_sources(raw, plan, helpers)
    before, excluded = collect_raw(raw)
    members = {"raw/" + name: (raw / name, info) for name, info in before.items()}

    # Exactly one explicitly declared tensor root per visit, with all four
    # exported arrays. No scanning /tmp or sibling experiments.
    tensor_roots, tensors = {}, {}
    for case_index, (case, op, dims, block, dispatch, auto_block) in enumerate(CASES):
        for position, requested in enumerate(ORDER):
            label = f"{case}-{position}-block{requested}"
            row = result["rows"][case_index * 8 + position]
            relative = label + "/cohort/results.json"
            require(row["result"] == relative, "unexpected cohort path")
            cohort = read(raw / relative)
            require(cohort["cohort_valid"] is cohort["artifacts_unchanged"] is True and len(cohort["results"]) == 1,
                    "unfinished/invalid cohort")
            capture = cohort["results"][0]
            require(capture["status"] == "OK" and capture["valid"] is True, "failed capture")
            root_declared = Path(cohort["temporary_tensor_root"])
            root_resolved = root_declared.resolve(strict=True)
            tensor_roots[label] = str(root_declared)
            output = Path(capture["command"][-1])
            expected = [output, *(Path(str(output) + f".input{i}.f32") for i in range(3))]
            require(set(capture["tensor_receipts"]) == set(map(str, expected)), "unexpected tensor receipt set")
            for original in expected:
                require(original.is_relative_to(root_declared) and original.resolve(strict=True).is_relative_to(root_resolved),
                        "external tensor escapes declared root")
                info = fingerprint(original)
                require(info == capture["tensor_receipts"][str(original)], "tensor identity changed")
                name = "tensors/" + label + "/" + original.relative_to(root_declared).as_posix()
                require(name not in members, "duplicate tensor member")
                tensors[name] = info
                members[name] = (original, info)
    require(len(tensor_roots) == 48 and len(tensors) == 192, "incomplete tensor evidence")

    selected = Path(plan["selected_source"])
    selected_resolved = selected.resolve(strict=True)
    selected_members = {}
    for original, expected in plan["selected_source_sha256"].items():
        path = Path(original)
        require(path.is_relative_to(selected) and path.resolve(strict=True).is_relative_to(selected_resolved),
                "selected source outside declared root")
        info = fingerprint(path)
        require(info["sha256"] == expected, "selected source changed: " + original)
        name = "sources/selected/" + path.relative_to(selected).as_posix()
        selected_members[name] = info
        members[name] = (path, info)
    require(bool(selected_members), "no plan-bound selected source snapshots")
    helper_members = {}
    for name in HELPERS + TEST_HELPERS:
        path = helpers / name
        info = fingerprint(path)
        archive_name = "sources/helpers/" + name
        helper_members[archive_name] = info
        members[archive_name] = (path, info)

    # One-thread low-preset compression with a large dictionary keeps repeated
    # stdout, receipts and tensors compact without changing archive contents.
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 0, "dict_size": 128 * 1024 * 1024}]
    with (OUT / "evidence.tar.xz").open("xb") as destination:
        with lzma.LZMAFile(destination, "w", filters=filters) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, (path, info) in sorted(members.items()):
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mode = info["bytes"], 0o644
                    with path.open("rb") as stream:
                        archive.addfile(entry, stream)
    require(collect_raw(raw) == (before, excluded), "raw changed during packaging; preserve failed output")
    for path, info in members.values():
        require(fingerprint(path) == info, "member changed during packaging: " + str(path))
    expected_members = {name: info for name, (_, info) in members.items()}
    actual_members = {}
    import hashlib
    with tarfile.open(OUT / "evidence.tar.xz", "r:xz") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            require(entry.isfile() and not name.is_absolute() and ".." not in name.parts and entry.name not in actual_members,
                    "unsafe or duplicate archive member")
            actual_members[entry.name] = dict(bytes=entry.size,
                sha256=hashlib.file_digest(archive.extractfile(entry), "sha256").hexdigest())
    require(actual_members == expected_members, "complete archive readback differs")
    manifest = dict(format="metal-block-mapping-evidence-v1", original_raw=str(raw), source_head=plan["source_head"],
                    evidence_sha256=sha(OUT / "evidence.tar.xz"), evidence_bytes=(OUT / "evidence.tar.xz").stat().st_size,
                    member_count=len(members), member_uncompressed_bytes=sum(v["bytes"] for v in expected_members.values()),
                    members=expected_members, excluded_raw_directories=excluded, raw_unchanged=True,
                    compression=dict(codec="lzma2", preset=0, dictionary_bytes=128 * 1024 * 1024, threads=1),
                    selected_source_count=len(selected_members), external_tensor_roots=tensor_roots,
                    external_tensor_members=tensors, helper_members=helper_members, tested_sources=tested,
                    artifact_fingerprints=plan["frozen_sha256"],
                    boundary="Raw logs include the original failed build and the successful compiler invocation with retained wrapper false-positive diagnostics, plus its successful confirmation. Only plan-bound selected sources and six tested Python sources are included. Native binaries are fingerprints only. Not a complete recursive-source rebuild, dependency or dynamic-loader closure; no native execution during packaging.")
    with (OUT / "package-inventory.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    with (OUT / "SHA256SUMS").open("x") as stream:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(sha(path) + "  " + path.name + "\n")
    print(json.dumps(dict(status="packaged", evidence_bytes=manifest["evidence_bytes"],
                          evidence_sha256=manifest["evidence_sha256"], members=len(members), tensors=len(tensors),
                          selected_sources=len(selected_members), raw_unchanged=True)))


if __name__ == "__main__":
    main()
