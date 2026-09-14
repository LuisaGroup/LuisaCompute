#!/usr/bin/env python3
"""Create or independently verify the complete map-fusion evidence archive.

Standard library only. This does not run kernels, builds, JIT or archived code.
All archive members are regular files, and verification streams every byte.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile


SCHEMA = "luisa-map-fusion-evidence-v1"
CHUNK_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 50_000_000
RAW_BUNDLE_BYTES = 45_000_000
RUN_DIRECTORIES = {
    "captures", "prepared", "replay", "v3", "v3-preflight-error", "v4", "evidence-qa",
}
RUN_EXCLUDED_DIRECTORIES = {
    "docs-html": "Rebuildable whole-site Sphinx output, excluded deliberately; build logs and targeted rendered screenshots/receipts are retained under run/evidence-qa/docs.",
}
RUN_SUFFIXES = {".py", ".log", ".json", ".xml", ".txt"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(value):
    require(isinstance(value, str) and value != "", "empty/non-string member name")
    require("\\" not in value and ":" not in value and "\x00" not in value,
            f"unsafe member name: {value!r}")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value,
            f"non-canonical member name: {value!r}")
    require(all(part not in {"", ".", ".."} for part in value.split("/")),
            f"traversal member name: {value!r}")
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=unique_object)


def hash_stream(source):
    digest = hashlib.sha256()
    size = 0
    while data := source.read(CHUNK_BYTES):
        digest.update(data)
        size += len(data)
    return size, digest.hexdigest()


def hash_file(path):
    require(stat.S_ISREG(path.lstat().st_mode), f"not a regular file: {path}")
    with path.open("rb") as source:
        return hash_stream(source)


def root_directory(value):
    path = Path(value)
    require(not path.is_symlink(), f"symlink source root: {path}")
    path = path.resolve(strict=True)
    require(path.is_dir() and path.parent != path, f"unsafe/non-directory root: {path}")
    return path


def inventory_tree(root, prefix):
    """Never follow symlinks; reject special files rather than silently omit them."""
    result = []
    for current, directories, files in os.walk(root, followlinks=False):
        current = Path(current)
        for name in directories:
            child = current / name
            require(stat.S_ISDIR(child.lstat().st_mode), f"non-directory/symlink: {child}")
        for name in sorted(files):
            child = current / name
            require(stat.S_ISREG(child.lstat().st_mode), f"non-regular source: {child}")
            member = (PurePosixPath(prefix) / child.relative_to(root).as_posix()).as_posix()
            safe_name(member)
            result.append((member, child))
    return sorted(result)


def inventory_run(root):
    result = []
    for child in sorted(root.iterdir()):
        mode = child.lstat().st_mode
        if stat.S_ISDIR(mode):
            if child.name in RUN_EXCLUDED_DIRECTORIES:
                continue
            require(child.name in RUN_DIRECTORIES, f"unreviewed run directory: {child.name}")
            result.extend(inventory_tree(child, f"run/{child.name}"))
        else:
            require(stat.S_ISREG(mode), f"non-regular run entry: {child.name}")
            require(not child.name.startswith(".") and child.suffix in RUN_SUFFIXES,
                    f"unreviewed root file: {child.name}")
            result.append((f"run/{child.name}", child))
    return sorted(result)


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.size = 0
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.source.read(size)
        self.size += len(data)
        self.digest.update(data)
        return data


def create_bundle(output, number, files):
    name = f"evidence-{number:02d}.tar.gz"
    destination = output / name
    members = []
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for member_name, source_path in files:
                    before = source_path.stat()
                    require(stat.S_ISREG(source_path.lstat().st_mode), f"source changed type: {source_path}")
                    info = tarfile.TarInfo(member_name)
                    info.size = before.st_size
                    info.mode = stat.S_IMODE(before.st_mode)
                    info.mtime = 0
                    with source_path.open("rb") as source:
                        reader = HashingReader(source)
                        archive.addfile(info, reader)
                        require(not source.read(1), f"source grew: {source_path}")
                    after = source_path.stat()
                    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                            f"source changed while archiving: {source_path}")
                    require(reader.size == before.st_size, f"truncated source: {source_path}")
                    members.append({"path": member_name, "size": reader.size,
                                    "sha256": reader.digest.hexdigest(), "mode": info.mode,
                                    "original_path": str(source_path)})
    size, digest = hash_file(destination)
    require(size < MAX_BUNDLE_BYTES, f"bundle exceeds 50 MB: {destination}")
    print(f"created {name}: {len(members)} files, {size} compressed bytes", flush=True)
    return {"path": name, "size": size, "sha256": digest, "members": members}


def create(args):
    require(args.run and args.final_source and args.base_commit,
            "--create needs --run, --final-source and --base-commit")
    require(re.fullmatch(r"[0-9a-f]{40}", args.base_commit), "base commit must be a full Git SHA-1")
    run = root_directory(args.run)
    final_source = root_directory(args.final_source)
    require(not (final_source / ".git").exists(), "--final-source must be a narrow frozen overlay, not a checkout")
    output = Path(args.output).absolute()
    require(not output.is_symlink(), "symlink output directory")
    output.mkdir(parents=False, exist_ok=True)
    output = output.resolve(strict=True)
    require(not output.is_relative_to(run) and not output.is_relative_to(final_source),
            "output must be outside both input trees")
    require(all(p.name in {"archive.py", "README.md", "qa-docs.cjs"} and p.is_file() and not p.is_symlink()
                for p in output.iterdir()), "output already contains evidence or unrelated files")
    final_files = inventory_tree(final_source, "final-source")
    require(0 < len(final_files) <= 128 and sum(p.stat().st_size for _, p in final_files) < 16_000_000,
            "final source must be a nonempty narrow overlay (<=128 files / <16 MB)")
    files = inventory_run(run) + final_files
    require(files, "empty evidence")
    names = [name for name, _ in files]
    require(len(set(names)) == len(names), "duplicate logical paths")
    bundles = []
    current = []
    raw_size = 10240
    for name, path in files:
        # Include ample PAX/header overhead; this corpus has no >32 MiB members.
        contribution = ((path.stat().st_size + 511) // 512) * 512 + 4096
        require(contribution + 10240 <= RAW_BUNDLE_BYTES,
                f"single member too large for safe bundle budget: {path}")
        if current and raw_size + contribution > RAW_BUNDLE_BYTES:
            bundles.append(create_bundle(output, len(bundles), current))
            current, raw_size = [], 10240
        current.append((name, path))
        raw_size += contribution
    if current:
        bundles.append(create_bundle(output, len(bundles), current))
    # Re-enumerate and rehash: no files added/removed or modified during packing.
    require(files == inventory_run(run) + inventory_tree(final_source, "final-source"),
            "source inventory changed during packing")
    for bundle in bundles:
        for member in bundle["members"]:
            require(hash_file(Path(member["original_path"])) == (member["size"], member["sha256"]),
                    f"source changed after packing: {member['original_path']}")
    manifest = {
        "schema": SCHEMA,
        "base_commit": args.base_commit,
        "roles": {
            "run/captures": "v2 captures; diagnostic Runtime timings, not the pure-entry timing result",
            "run/prepared": "frozen v2 ORC objects, native dylibs, helpers, inputs and FP64 oracles",
            "run/replay": "v2 measured native-entry ABBA visits, raw samples and complete outputs",
            "run/v3": "v3 source and recapture with bytewise v2 artifact comparisons; not a timing replay",
            "final-source": "later v4 budget-fix source overlay; validation is not a new timed cohort",
            "run/v4": "v4 source and validation recapture: 16 LLVM / 16 ORC / 50 data bytewise comparisons with v2; not a native-entry timing replay",
        },
        "limitations": [
            "Integrity verification is not numerical, semantic or performance validation.",
            "The v2 pre/post binary inventory omitted .so plugins; later hashes cannot establish their historical state.",
            "The v3 inventory includes 24 binary paths (executable, libraries and plugins) but does not repair the missing v2 hashes.",
            "The full original v2 C++ overlay was hashed but not snapshotted; its generated LLVM/ORC objects are retained.",
            "v3 artifact equivalence is limited to the checked case/artifact corpus; it is not general compiler equivalence.",
            "v4 bytewise artifact equivalence also applies only to the checked 16 variants and artifact kinds.",
            "v4 final validation and v2 measured timings are distinct source/time snapshots.",
            "Single-thread native-entry timings are not Runtime E2E throughput, legacy Tile, Torch, MPS or BLAS timings.",
        ],
        "logical_file_count": len(files),
        "logical_bytes": sum(m["size"] for b in bundles for m in b["members"]),
        "explicitly_excluded_directories": [
            {"path": f"run/{name}", "reason": reason}
            for name, reason in RUN_EXCLUDED_DIRECTORIES.items() if (run / name).is_dir()
        ],
        "bundles": bundles,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2)
        destination.write("\n")
    verify(output)


def verify(output):
    output = root_directory(output)
    manifest = read_json(output / "manifest.json")
    require(manifest.get("schema") == SCHEMA, "unexpected manifest schema")
    require(isinstance(manifest.get("bundles"), list) and manifest["bundles"], "missing bundles")
    bundle_names = set()
    member_names = set()
    logical_bytes = 0
    for bundle in manifest["bundles"]:
        name = bundle["path"]
        require(len(safe_name(name).parts) == 1 and name.endswith(".tar.gz"), "unsafe bundle name")
        require(name not in bundle_names, f"duplicate bundle: {name}")
        bundle_names.add(name)
        require(type(bundle["size"]) is int and 0 < bundle["size"] < MAX_BUNDLE_BYTES, "invalid bundle size")
        require(SHA256_PATTERN.fullmatch(bundle["sha256"]), "invalid bundle SHA-256")
        require(hash_file(output / name) == (bundle["size"], bundle["sha256"]), f"corrupt bundle: {name}")
        expected = {}
        for member in bundle["members"]:
            path = member["path"]
            safe_name(path)
            require(path not in member_names, f"duplicate member across archive: {path}")
            member_names.add(path)
            require(type(member["size"]) is int and member["size"] >= 0, f"invalid size: {path}")
            require(SHA256_PATTERN.fullmatch(member["sha256"]), f"invalid SHA-256: {path}")
            require(type(member["mode"]) is int and 0 <= member["mode"] <= 0o7777, f"invalid mode: {path}")
            expected[path] = member
        require(expected, f"empty bundle: {name}")
        seen = set()
        with tarfile.open(output / name, mode="r|gz") as archive:
            for info in archive:
                safe_name(info.name)
                require(info.isfile() and not info.issparse(), f"non-regular tar member: {info.name}")
                require(info.name not in seen and info.name in expected, f"unexpected/duplicate tar member: {info.name}")
                seen.add(info.name)
                member = expected[info.name]
                require(info.size == member["size"] and info.mode == member["mode"], f"metadata differs: {info.name}")
                source = archive.extractfile(info)
                require(source is not None, f"missing data: {info.name}")
                with source:
                    require(hash_stream(source) == (member["size"], member["sha256"]), f"corrupt member: {info.name}")
                logical_bytes += info.size
        require(seen == set(expected), f"missing tar members: {name}")
        print(f"verified {name}: {len(seen)} members", flush=True)
    require(manifest["logical_file_count"] == len(member_names), "file count differs")
    require(manifest["logical_bytes"] == logical_bytes, "logical byte count differs")
    print(json.dumps({"status": "verified", "bundles": len(bundle_names),
                      "files": len(member_names), "logical_bytes": logical_bytes}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--create", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--run", help="explicit experiment directory, never a whole checkout")
    parser.add_argument("--final-source", help="explicit frozen v4 task-only source overlay")
    parser.add_argument("--base-commit", help="full baseline Git commit for the overlay")
    args = parser.parse_args()
    try:
        if args.create:
            create(args)
        else:
            require(not (args.run or args.final_source or args.base_commit), "verification never consults external sources")
            verify(Path(args.output))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
