#!/usr/bin/env python3
"""preprocess.py -- official-AutoProcessor feature bundle exporter (offline).

Produces a .ocrfeat v1 feature bundle for differential testing against the C++
production preprocessing. Requires an EXPLICIT local official processor
(checkpoint directory with the official processing code/config); nothing is
downloaded. This is a DEBUG/interchange path, not ordinary input handling.

The bundle records the processor identity/revision, the source image hash and
the deterministic decode settings, plus the preprocessed tensors with their
shape/layout/dtype. Standard-library-only metadata; the official processor
itself (transformers/PIL/numpy) is imported lazily and only when present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcocr_format import (  # noqa: E402
    OCRF_ALIGNMENT,
    OCRF_HEADER_SIZE,
    OCRF_MAGIC,
    FEATURE_DTYPE_F32,
    FEATURE_DTYPE_I64,
    FormatError,
    align_up,
    checked_add,
    checked_mul,
    element_count,
    pack_u32,
    pack_u64,
)


def _die(msg: str):
    print(f"preprocess: error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FeatureWriter:
    """Streaming .ocrfeat v1 writer (matches feature_bundle.cpp)."""

    def __init__(self, path):
        self._path = path
        self._file = open(path, "wb")
        self._offset = OCRF_HEADER_SIZE
        self._payload_base = 0
        self._tensors = []
        self._file.write(b"\x00" * OCRF_HEADER_SIZE)

    def _write_all(self, data):
        if data:
            self._file.write(data)
            self._offset = checked_add(self._offset, len(data))

    def _pad(self):
        aligned = align_up(self._offset, OCRF_ALIGNMENT)
        if aligned > self._offset:
            self._write_all(b"\x00" * (aligned - self._offset))

    def add_tensor(self, name, dtype, shape, payload):
        count = element_count(shape)
        dtype_size = {FEATURE_DTYPE_F32: 4, FEATURE_DTYPE_I64: 8}.get(dtype)
        if dtype_size is None:
            raise FormatError(f"unsupported feature dtype {dtype}")
        if len(payload) != checked_mul(count, dtype_size):
            raise FormatError(f"payload size mismatch for {name}")
        self._pad()
        if self._payload_base == 0:
            self._payload_base = self._offset
        self._tensors.append({
            "name": name, "dtype": dtype, "shape": list(shape),
            "element_count": count, "payload_offset": self._offset,
            "payload_size": len(payload),
        })
        self._write_all(payload)

    def finish(self, meta):
        meta = dict(meta)
        meta.setdefault("format_version", 1)
        meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        meta_offset = self._offset
        self._write_all(meta_json)
        table_offset = self._offset
        table = bytearray(pack_u64(len(self._tensors)))
        for t in self._tensors:
            name_bytes = t["name"].encode("utf-8")
            table += pack_u32(len(name_bytes))
            table += pack_u32(t["dtype"])
            table += pack_u32(len(t["shape"]))
            table += pack_u32(0)
            table += pack_u64(t["element_count"])
            for d in t["shape"]:
                table += pack_u64(d)
            table += pack_u64(t["payload_offset"])
            table += pack_u64(t["payload_size"])
            table += name_bytes
        self._write_all(bytes(table))
        file_size = self._offset
        header = bytearray()
        header += OCRF_MAGIC
        header += pack_u32(1)
        header += pack_u32(OCRF_HEADER_SIZE)
        header += pack_u32(0)
        header += pack_u64(meta_offset)
        header += pack_u64(len(meta_json))
        header += pack_u64(table_offset)
        header += pack_u64(len(table))
        header += pack_u64(self._payload_base if self._payload_base else file_size)
        header += pack_u64(file_size)
        assert len(header) == OCRF_HEADER_SIZE
        self._file.seek(0)
        self._file.write(bytes(header))
        self._file.flush()
        self._file.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess.py",
        description="Export an official-AutoProcessor .ocrfeat feature bundle (offline, explicit local artifacts).",
    )
    p.add_argument("--processor-dir", required=True,
                   help="Local official processor directory (config + processing code; no download).")
    p.add_argument("--processor-revision", required=True,
                   help="Exact official processor revision (recorded).")
    p.add_argument("--image", required=True, help="Local source image path.")
    p.add_argument("--output", required=True, help="Destination .ocrfeat path.")
    p.add_argument("--decode-settings", default='{"deterministic":true}',
                   help="JSON text of deterministic decode settings (recorded).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.processor_dir):
        _die(f"processor dir not found: {args.processor_dir} (offline; no implicit download)")
    if not os.path.isfile(args.image):
        _die(f"image not found: {args.image}")
    try:
        json.loads(args.decode_settings)
    except ValueError:
        _die("--decode-settings must be valid JSON")
    try:
        from transformers import AutoProcessor  # type: ignore
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        _die("transformers/PIL/numpy are required for the official processor path; "
             "install them locally (this tool never downloads them)")
    processor = AutoProcessor.from_pretrained(args.processor_dir, local_files_only=True)
    image = Image.open(args.image).convert("RGB")
    features = processor(images=image, return_tensors="np")
    writer = FeatureWriter(args.output)
    for name, value in features.items():
        arr = np.asarray(value)
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            writer.add_tensor(name, FEATURE_DTYPE_F32, list(arr.shape),
                              arr.astype("<f4").tobytes())
        elif arr.dtype in (np.int64, np.int32):
            writer.add_tensor(name, FEATURE_DTYPE_I64, list(arr.shape),
                              arr.astype("<i8").tobytes())
        else:
            _die(f"unsupported feature dtype for {name}: {arr.dtype}")
    meta = {
        "producer": "preprocess.py",
        "processor": getattr(processor, "name_or_path", args.processor_dir),
        "processor_revision": args.processor_revision,
        "source_image_hash": _sha256_file(args.image),
        "decode_settings": args.decode_settings,
    }
    writer.finish(meta)
    print(f"preprocess: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())