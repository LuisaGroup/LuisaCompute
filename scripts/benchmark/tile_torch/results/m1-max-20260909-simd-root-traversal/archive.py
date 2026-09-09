#!/usr/bin/env python3
"""Create or independently verify the complete SIMD GEMM diagnostic archive.

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


SCHEMA = "luisa-simd-root-traversal-evidence-v1"
CHUNK_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 50_000_000
CASE_PREFIXES = (
    "baseline", "baseline-full", "m8n1", "m8n1-full", "m8n1-w4-full",
    "root32-rect", "root32-4096",
)
RUN_DIRECTORIES = {"root-v1", "final-source", "evidence-qa", "docs-qa"}
RUN_SUFFIXES = {".py", ".log", ".json", ".xml", ".txt", ".md", ".f32", ".f64", ".cjs"}
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
    require(args.run and args.base_commit, "--create needs --run and --base-commit")
    require(re.fullmatch(r"[0-9a-f]{40}", args.base_commit), "base commit must be a full Git SHA-1")
    run = root_directory(args.run)
    require(not (run / ".git").exists(), "--run must be the narrow experiment, not a checkout")
    output = Path(args.output).absolute()
    require(not output.is_symlink(), "symlink output directory")
    output.mkdir(parents=False, exist_ok=True)
    output = output.resolve(strict=True)
    require(not output.is_relative_to(run), "output must be outside the input tree")
    require(all(p.name in {"archive.py", "README.md"} and p.is_file() and not p.is_symlink()
                for p in output.iterdir()), "output already contains evidence or unrelated files")
    files = inventory_run(run)
    require(files, "empty evidence")
    require(len({name for name, _ in files}) == len(files), "duplicate logical paths")
    groups = {prefix: [] for prefix in CASE_PREFIXES}
    groups["source-and-receipts"] = []
    for name, path in files:
        group = next((prefix for prefix in CASE_PREFIXES
                      if path.parent == run and path.name.startswith(prefix + ".")), "source-and-receipts")
        groups[group].append((name, path))
    for prefix in CASE_PREFIXES:
        found = {path.name for _, path in groups[prefix]}
        for suffix in ("input0.f32", "input1.f32", "expected.f64", "output.f32", "source.txt", "log"):
            require(prefix + "." + suffix in found, f"missing complete case member: {prefix}.{suffix}")
    require((run / "final-source").is_dir(), "missing final-source snapshot")
    bundles = []
    for group, members in groups.items():
        if members:
            # Keep each case intact even for its 128 MiB FP64 oracle.
            # No member truncation/sampling; reject if the compressed result reaches 50 MB.
            bundle = create_bundle(output, len(bundles), members)
            bundle["group"] = group
            bundles.append(bundle)
    require(files == inventory_run(run), "source inventory changed during packing")
    for bundle in bundles:
        for member in bundle["members"]:
            require(hash_file(Path(member["original_path"])) == (member["size"], member["sha256"]),
                    f"source changed after packing: {member['original_path']}")
    manifest = {
        "schema": SCHEMA,
        "base_commit": args.base_commit,
        "roles": {
            "baseline": "4096 cubed FP32 GEMM, original pipeline source; completed whole-process diagnostic",
            "baseline-full": "4096 cubed FP32 GEMM, full-packet specialization enabled (not full-K)",
            "m8n1": "4096 cubed FP32 GEMM, changed source tile shape",
            "m8n1-full": "4096 cubed FP32 GEMM, changed source tile shape and full-packet specialization",
            "m8n1-w4-full": "4096 cubed FP32 GEMM, W4 and full-packet specialization",
            "root32-rect": "rectangular FP32 GEMM, opt-in root traversal mapping",
            "root32-4096": "4096 cubed FP32 GEMM, opt-in root traversal mapping",
            "run/root-v1/source": "measured root traversal implementation snapshot",
            "run/final-source": "later implementation/test source snapshot; not a rerun of all seven diagnostics",
        },
        "limitations": [
            "All timings are single-sample synchronized Runtime E2E diagnostics, not native-entry or pure-kernel measurements.",
            "The seven cases are not randomized ABBA pairs or a statistically qualified benchmark matrix.",
            "All seven sources use k_tile_pipeline; some cases change tile shape, full-packet specialization or SIMD width. Do not attribute every difference to root mapping.",
            "Root temporal cache cost is unmodeled; the mapping option is not an automatically calibrated solver choice.",
            "The original archived 4096-cubed SIMD matrix Error remains unchanged; these are separate completed diagnostic attempts.",
            "No new legacy Tile, Torch, MPS or BLAS timing comparison is established.",
            "Archive-time hashes are integrity receipts, not retroactive pre-run binary freeze evidence.",
            "Passing archive verification is not a numerical or semantic validation by itself.",
        ],
        "logical_file_count": len(files),
        "logical_bytes": sum(m["size"] for b in bundles for m in b["members"]),
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
    parser.add_argument("--base-commit", help="full baseline Git commit for the overlay")
    args = parser.parse_args()
    try:
        if args.create:
            create(args)
        else:
            require(not (args.run or args.base_commit), "verification never consults external sources")
            verify(Path(args.output))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
