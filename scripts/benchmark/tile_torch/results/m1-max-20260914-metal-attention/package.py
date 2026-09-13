"""Package reviewed Metal evidence only after every native writer has stopped.

Never executes a compiler, GPU workload, or numerical oracle. Excludes Python
environments/caches; copies only receipt-listed external tensors and selected
source files whose frozen hashes match. Binary fingerprints are not a loader
closure, and this is not a self-contained rebuild package.
"""

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import tarfile

OUT = Path(__file__).resolve().parent
EXCLUDED = {"venv", ".venv", "__pycache__"}
HELPERS = ("metal4_timing.py", "compare_llm.py", "run.py", "repeat.py")


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected a regular, nonsymlink file: " + str(path))
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def collect_raw(root):
    files, excluded = {}, []
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        kept = []
        for name in sorted(dirs):
            path = parent / name
            if name in EXCLUDED:
                excluded.append(path.relative_to(root).as_posix())
            elif path.is_symlink():
                raise ValueError("refusing a raw symlink directory: " + str(path))
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            path = parent / name
            files[path.relative_to(root).as_posix()] = fingerprint(path)
    return files, sorted(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--helpers-dir", type=Path, required=True)
    parser.add_argument("--test-helper-sha256", required=True,
                        help="reviewed final test_metal4_timing.py hash; not a production capture dependency")
    args = parser.parse_args()
    raw, helpers = args.raw.resolve(strict=True), args.helpers_dir.resolve(strict=True)
    if raw == OUT or OUT.is_relative_to(raw) or raw.is_relative_to(OUT):
        raise ValueError("raw and package must be disjoint")
    for name in ("evidence.tar.xz", "package-inventory.json", "SHA256SUMS"):
        if (OUT / name).exists():
            raise ValueError("preserve previous package attempt; output exists: " + name)
    before, excluded = collect_raw(raw)
    members = {"raw/" + name: (raw / name, info) for name, info in before.items()}
    cohort = read_json(raw / "metal4-fixed/cohort/results.json")
    query = read_json(raw / "query-tiles-results.json")
    if query.get("status") != "complete" or query.get("artifacts_unchanged") is not True or len(query["rows"]) != 16:
        raise ValueError("query-tile experiment is incomplete")
    cohort_paths = [("metal4-fixed", raw / "metal4-fixed/cohort/results.json")]
    for row in query["rows"]:
        label = f"query-{row['case']}-{row['position']}-bq{row['query_tile']}"
        if row["result"] != label + "/cohort/results.json" or Path(label).name != label:
            raise ValueError("out-of-scope query cohort path")
        cohort_paths.append((label, raw / row["result"]))
    if len({label for label, _ in cohort_paths}) != 17:
        raise ValueError("duplicate cohort identity")

    # Exactly the fixed cohort and sixteen declared query cohorts. Never
    # scan sibling temporary directories or include unreferenced tensors.
    tensor_receipts, tensor_roots = {}, {}
    for label, report_path in cohort_paths:
        report = read_json(report_path)
        if not report.get("finished_utc") or not report.get("artifacts_unchanged"):
            raise ValueError("cohort is unfinished or artifacts changed: " + label)
        tensor_root = Path(report["temporary_tensor_root"]).resolve(strict=True)
        tensor_roots[label] = str(tensor_root)
        for row in report["results"]:
            for original, receipt in row["tensor_receipts"].items():
                if Path(original).is_symlink():
                    raise ValueError("refusing an external tensor symlink")
                path = Path(original).resolve(strict=True)
                if not path.is_relative_to(tensor_root) or receipt is None:
                    raise ValueError("missing or out-of-scope external tensor")
                info = fingerprint(path)
                if info != receipt:
                    raise ValueError("external tensor receipt mismatch: " + original)
                name = "tensors/" + label + "/" + path.relative_to(tensor_root).as_posix()
                previous = tensor_receipts.setdefault(name, info)
                if previous != info:
                    raise ValueError("ambiguous external tensor identity")
                members[name] = (path, info)

    provenance = read_json(raw / "native-provenance.json")
    declared_selected = Path(provenance["selected_source"])
    selected = declared_selected.resolve(strict=True)
    selected_sources = {}
    for original, expected in provenance["files_sha256"].items():
        path = Path(original)
        # Exactly the source prefix declared before native execution. Bin
        # paths remain fingerprints only; no dylib or virtualenv is copied.
        if not path.is_relative_to(declared_selected):
            continue
        if not path.resolve(strict=True).is_relative_to(selected):
            raise ValueError("source path resolves outside the declared scope")
        info = fingerprint(path)
        if info["sha256"] != expected:
            raise ValueError("frozen selected source changed: " + original)
        name = "sources/selected/" + path.relative_to(declared_selected).as_posix()
        selected_sources[name] = info
        members[name] = (path, info)
    if not selected_sources:
        raise ValueError("no frozen selected sources")

    helper_sources = {}
    for name in (*HELPERS, "test_metal4_timing.py"):
        path = helpers / name
        info = fingerprint(path)
        if name == "test_metal4_timing.py":
            expected = args.test_helper_sha256
        else:
            matches = [v for p, v in cohort["artifacts_before"].items() if Path(p).resolve() == path]
            if len(matches) != 1:
                raise ValueError("helper absent/ambiguous in capture receipts: " + name)
            expected = matches[0]["sha256"]
        if info["sha256"] != expected:
            raise ValueError("reviewed Python helper changed: " + name)
        archive_name = "sources/helpers/" + name
        helper_sources[archive_name] = info
        members[archive_name] = (path, info)

    filters = [{"id": lzma.FILTER_LZMA2, "preset": 0, "dict_size": 128 * 1024 * 1024}]
    with (OUT / "evidence.tar.xz").open("xb") as destination:
        with lzma.LZMAFile(destination, "w", filters=filters) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, (path, info) in sorted(members.items()):
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mode = info["bytes"], 0o644
                    with path.open("rb") as stream:
                        archive.addfile(entry, stream)

    if collect_raw(raw) != (before, excluded):
        raise ValueError("raw changed during packaging; preserve failed output")
    expected_members = {name: info for name, (_, info) in members.items()}
    for path, expected in members.values():
        if fingerprint(path) != expected:
            raise ValueError("source/tensor/raw changed while packaging: " + str(path))
    actual_members = {}
    with tarfile.open(OUT / "evidence.tar.xz", "r:xz") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            if not entry.isfile() or entry.name in actual_members or name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe or duplicate archive member")
            actual_members[entry.name] = dict(bytes=entry.size, sha256=hashlib.file_digest(archive.extractfile(entry), "sha256").hexdigest())
    if actual_members != expected_members:
        raise ValueError("complete archive readback mismatch")
    manifest = dict(
        format="metal-attention-recovery-evidence-v1", original_raw=str(raw),
        evidence_sha256=sha(OUT / "evidence.tar.xz"), evidence_bytes=(OUT / "evidence.tar.xz").stat().st_size,
        member_count=len(members), member_uncompressed_bytes=sum(v["bytes"] for v in expected_members.values()),
        members=expected_members, excluded_raw_directories=excluded, raw_unchanged=True,
        compression=dict(codec="lzma2", preset=0, dictionary_bytes=128 * 1024 * 1024, threads=1),
        frozen_selected_source_count=len(selected_sources), native_source_head=provenance["head"],
        external_tensor_roots=tensor_roots, external_tensor_members=tensor_receipts,
        helper_members=helper_sources,
        test_snapshot_scope="Final reviewed test source; not a production dependency attested before capture.",
        boundary="Regular raw evidence plus receipt-listed external tensors and selected source snapshots. All hashes/readback checked; no live native execution. Not a whole-HEAD build, self-contained rebuild, or dynamic-loader closure.")
    with (OUT / "package-inventory.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    with (OUT / "SHA256SUMS").open("x") as stream:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(sha(path) + "  " + path.name + "\n")
    print(json.dumps(dict(status="packaged", bytes=manifest["evidence_bytes"], sha256=manifest["evidence_sha256"],
                          members=len(members), selected_sources=len(selected_sources), tensors=len(tensor_receipts), raw_unchanged=True)))


if __name__ == "__main__":
    main()
