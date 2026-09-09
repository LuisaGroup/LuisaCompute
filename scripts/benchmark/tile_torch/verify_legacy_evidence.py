#!/usr/bin/env python3
"""Verify legacy/current Tile evidence, optionally restoring to a new directory.

Only Python's standard library is needed. No benchmark is run and no archive
path is extracted directly: regular files are streamed to exclusive new files.
"""

import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def relative_path(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"unsafe relative path: {name!r}")
    return Path(*path.parts)


def checked_digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"invalid SHA-256: {value!r}")
    return value


def checked_size(value):
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid byte size: {value!r}")
    return value


def stream_digest(source, expected, destination=None):
    size = checked_size(expected["size"])
    digest = hashlib.sha256()
    count = 0
    while data := source.read(min(1024 * 1024, size - count + 1)):
        count += len(data)
        if count > size:
            raise ValueError("content exceeds its declared byte size")
        digest.update(data)
        if destination is not None:
            destination.write(data)
    if count != size or digest.hexdigest() != checked_digest(expected["sha256"]):
        raise ValueError("content size or SHA-256 mismatch")


def verify_file(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    with path.open("rb") as source:
        stream_digest(source, expected)


def output_file(root, name):
    if root is None:
        return nullcontext(None)
    path = root / relative_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("xb")


def verify_bundles(root, manifest, restore):
    count = 0
    for name, record in manifest.items():
        relative_path(name)
        if not name.endswith(".tar.gz") or "/" in name:
            raise ValueError(f"invalid bundle name: {name}")
        verify_file(root / name, record)
        expected = {}
        for member in record["members"]:
            relative_path(member["path"])
            if member["path"] in expected:
                raise ValueError(f"duplicate tar manifest member: {member['path']}")
            expected[member["path"]] = member
        seen = set()
        bundle_root = restore / name.removesuffix(".tar.gz") if restore else None
        with tarfile.open(root / name, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or member.name in seen or member.name not in expected:
                    raise ValueError(f"unexpected/non-regular tar member: {name}:{member.name}")
                record = expected[member.name]
                if member.size != checked_size(record["size"]):
                    raise ValueError(f"tar header size mismatch: {name}:{member.name}")
                with archive.extractfile(member) as source, output_file(bundle_root, member.name) as output:
                    try:
                        stream_digest(source, record, output)
                    except ValueError as error:
                        raise ValueError(f"{name}:{member.name}: {error}") from error
                seen.add(member.name)
        if seen != expected.keys():
            raise ValueError(f"missing tar members: {name}")
        count += len(seen)
        print(f"Verified {name}: {len(seen)} members", flush=True)
    return count


def verify_matrix(root, manifest, restore):
    if manifest["format"] != "sha256-chunks-zip-v1":
        raise ValueError("unsupported matrix evidence format")
    chunk_size = checked_size(manifest["chunk_size"])
    if not 0 < chunk_size <= 64 * 1024 * 1024:
        raise ValueError("unsupported chunk size")
    expected_by_shard = {name: set() for name in manifest["shards"]}
    for key, record in manifest["chunks"].items():
        checked_digest(key)
        if not 0 < checked_size(record["size"]) <= chunk_size:
            raise ValueError(f"invalid chunk size: {key}")
        expected_by_shard[record["shard"]].add(key)
    with ExitStack() as stack:
        shards = {}
        for name, record in manifest["shards"].items():
            relative_path(name)
            if "/" in name:
                raise ValueError(f"invalid shard name: {name}")
            verify_file(root / name, record)
            archive = stack.enter_context(zipfile.ZipFile(root / name))
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != expected_by_shard[name]:
                raise ValueError(f"duplicate/missing/unexpected ZIP entries: {name}")
            shards[name] = archive
        for key, record in manifest["chunks"].items():
            archive = shards[record["shard"]]
            info = archive.getinfo(key)
            if info.file_size != record["size"] or info.is_dir():
                raise ValueError(f"invalid ZIP entry: {key}")
            with archive.open(key) as source:
                stream_digest(source, {"size": record["size"], "sha256": key})
        used_chunks = set()
        matrix_root = restore / "matrix" if restore else None
        for name, record in manifest["files"].items():
            relative_path(name)
            expected_size = checked_size(record["size"])
            expected_hash = checked_digest(record["sha256"])
            digest = hashlib.sha256()
            count = 0
            with output_file(matrix_root, name) as output:
                for index, key in enumerate(record["chunks"]):
                    chunk = manifest["chunks"][key]
                    used_chunks.add(key)
                    if index + 1 < len(record["chunks"]) and chunk["size"] != chunk_size:
                        raise ValueError(f"short non-final chunk: {name}")
                    with shards[chunk["shard"]].open(key) as source:
                        while data := source.read(1024 * 1024):
                            count += len(data)
                            if count > expected_size:
                                raise ValueError(f"reconstructed size exceeds manifest: {name}")
                            digest.update(data)
                            if output is not None:
                                output.write(data)
            if count != expected_size or digest.hexdigest() != expected_hash:
                raise ValueError(f"reconstructed size/SHA-256 mismatch: {name}")
        if used_chunks != manifest["chunks"].keys():
            raise ValueError("unreferenced matrix chunks")
    # The convenient unpacked summary must match the copy inside the archive.
    if (root / "matrix-results.json").exists():
        verify_file(root / "matrix-results.json", manifest["files"]["results.json"])
    print(f"Verified matrix: {len(manifest['files'])} files, "
          f"{len(manifest['chunks'])} chunks, {len(shards)} shards", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="directory containing both manifests")
    parser.add_argument("--restore", type=Path, help="new directory (must not already exist)")
    args = parser.parse_args()
    root = args.evidence.resolve(strict=True)
    bundles = read_json(root / "manifest.json")
    matrix = read_json(root / "matrix-manifest.json")
    restore = None
    if args.restore:
        # Resolve only the existing parent; never follow a final symlink or reuse
        # a directory. A private root and exclusive files prevent overwrites.
        parent = args.restore.parent.resolve(strict=True)
        restore = parent / args.restore.name
        if args.restore.name in ("", ".", ".."):
            raise ValueError("restore must name a new child directory")
        restore.mkdir(mode=0o700, exist_ok=False)
    count = verify_bundles(root, bundles, restore)
    verify_matrix(root, matrix, restore)
    print(f"PASS: {len(bundles)} bundles / {count} tar members and all matrix evidence")
    if restore:
        print(f"Restored to {restore}")


if __name__ == "__main__":
    main()
