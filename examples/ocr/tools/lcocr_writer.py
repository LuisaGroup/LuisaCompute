"""Streaming .lcocr v1 writer (standard-library only).

Writes the SAME byte layout as examples/ocr/ocr/package.cpp (PackageWriter).
Tensors are streamed to disk one at a time; no whole-model host dictionary is
retained. Used by pack_model.py and by the frozen-fixture generator.
"""
from __future__ import annotations

import json
import os
import struct

from lcocr_format import (
    LCOCR_HEADER_SIZE,
    LCOCR_MAGIC,
    LCOCR_MAX_NAME_BYTES,
    LCOCR_MAX_RANK,
    LCOCR_PAYLOAD_ALIGNMENT,
    FormatError,
    align_up,
    checked_add,
    checked_mul,
    dense_element_size,
    element_count,
    group_count,
    int4_packed_size,
    is_quantized,
    pack_u32,
    pack_u64,
    row_major_strides,
    QUANT_INT4_ASYMMETRIC,
    QUANT_INT4_SYMMETRIC,
    QUANT_INT8_SYMMETRIC,
    QUANT_NONE,
    ENCODING_INT4_PACKED,
    ENCODING_INT8,
)


class PackageWriter:
    def __init__(self, path):
        self._path = os.fspath(path)
        self._file = open(self._path, "wb")
        self._offset = LCOCR_HEADER_SIZE
        self._payload_base = 0
        self._tensors = []
        self._finished = False
        self._file.write(b"\x00" * LCOCR_HEADER_SIZE)
        # payload_base is set lazily on the first payload (after alignment),
        # matching the C++ PackageWriter. An EMPTY package writes no padding:
        # its payload region is empty and payload_base == manifest_offset.

    def _write_all(self, data: bytes):
        if not data:
            return
        self._file.write(data)
        self._offset = checked_add(self._offset, len(data))

    def _pad(self):
        aligned = align_up(self._offset, LCOCR_PAYLOAD_ALIGNMENT)
        padding = aligned - self._offset
        if padding:
            self._write_all(b"\x00" * padding)

    def add_tensor(self, name, encoding, layout, shape, payload,
                   quant_kind=QUANT_NONE, group_size=0, scales=None, zero_points=None):
        if self._finished:
            raise FormatError("writer already finished")
        if not name or len(name.encode("utf-8")) > LCOCR_MAX_NAME_BYTES:
            raise FormatError(f"invalid tensor name: {name!r}")
        if "\x00" in name:
            raise FormatError("tensor name contains NUL")
        if any(t["name"] == name for t in self._tensors):
            raise FormatError(f"duplicate tensor name: {name}")
        if len(shape) > LCOCR_MAX_RANK:
            raise FormatError(f"rank exceeds maximum: {len(shape)}")
        count = element_count(shape)
        scales = list(scales) if scales is not None else []
        zero_points = list(zero_points) if zero_points is not None else []

        if is_quantized(encoding):
            if encoding == ENCODING_INT4_PACKED:
                if quant_kind not in (QUANT_INT4_SYMMETRIC, QUANT_INT4_ASYMMETRIC):
                    raise FormatError("INT4 encoding requires an INT4 quant kind")
                expected = int4_packed_size(count)
            else:
                if quant_kind != QUANT_INT8_SYMMETRIC:
                    raise FormatError("INT8 encoding requires int8_symmetric")
                expected = count
            if len(payload) != expected:
                raise FormatError(f"payload size mismatch for {name}: {len(payload)} != {expected}")
            groups = group_count(count, group_size)
            if len(scales) != groups:
                raise FormatError(f"scale count mismatch for {name}")
            for s in scales:
                if not (s == s) or s in (float("inf"), float("-inf")) or s < 0:
                    raise FormatError(f"non-finite/negative scale for {name}")
            if quant_kind == QUANT_INT4_ASYMMETRIC:
                if len(zero_points) != groups:
                    raise FormatError(f"zero-point count mismatch for {name}")
                for zp in zero_points:
                    if not 0 <= int(zp) <= 15:
                        raise FormatError(f"zero point out of range for {name}")
            elif zero_points:
                raise FormatError("symmetric quantization must not store zero points")
        else:
            if quant_kind != QUANT_NONE or group_size != 0 or scales or zero_points:
                raise FormatError("dense encoding must not carry quantization metadata")
            expected = checked_mul(count, dense_element_size(encoding))
            if len(payload) != expected:
                raise FormatError(f"payload size mismatch for {name}: {len(payload)} != {expected}")

        self._pad()
        if self._payload_base == 0:
            self._payload_base = self._offset
        record = {
            "name": name,
            "encoding": encoding,
            "layout": layout,
            "quant_kind": quant_kind,
            "group_size": group_size,
            "shape": list(shape),
            "strides": row_major_strides(shape),
            "element_count": count,
            "payload_offset": self._offset,
            "payload_size": len(payload),
            "scales_offset": 0,
            "scales_size": 0,
            "zero_points_offset": 0,
            "zero_points_size": 0,
        }
        self._write_all(payload)
        if scales:
            self._pad()
            record["scales_offset"] = self._offset
            record["scales_size"] = len(scales) * 4
            self._write_all(struct.pack(f"<{len(scales)}f", *scales))
        if zero_points:
            self._pad()
            record["zero_points_offset"] = self._offset
            record["zero_points_size"] = len(zero_points)
            self._write_all(bytes(zero_points))
        self._tensors.append(record)

    def finish(self, manifest: dict):
        if self._finished:
            raise FormatError("writer already finished")
        manifest = dict(manifest)
        manifest.setdefault("format_version", 1)
        manifest_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        # Canonical empty layout: with no payloads, pad to the 256-byte payload
        # alignment so payload_base == manifest_offset is aligned (matches C++).
        if not self._tensors:
            self._pad()
        manifest_offset = self._offset
        self._write_all(manifest_json)
        table_offset = self._offset
        table = bytearray()
        table += pack_u64(len(self._tensors))
        for t in self._tensors:
            name_bytes = t["name"].encode("utf-8")
            table += pack_u32(len(name_bytes))
            table += pack_u32(len(t["shape"]))
            table += pack_u32(t["encoding"])
            table += pack_u32(t["layout"])
            table += pack_u32(t["quant_kind"])
            table += pack_u32(t["group_size"])
            table += pack_u32(0)
            table += pack_u64(t["element_count"])
            for d in t["shape"]:
                table += pack_u64(d)
            for s in t["strides"]:
                table += pack_u64(s)
            table += pack_u64(t["payload_offset"])
            table += pack_u64(t["payload_size"])
            table += pack_u64(t["scales_offset"])
            table += pack_u64(t["scales_size"])
            table += pack_u64(t["zero_points_offset"])
            table += pack_u64(t["zero_points_size"])
            table += name_bytes
        self._write_all(bytes(table))
        file_size = self._offset
        header = bytearray()
        header += LCOCR_MAGIC
        header += pack_u32(1)
        header += pack_u32(LCOCR_HEADER_SIZE)
        header += pack_u64(manifest_offset)
        header += pack_u64(len(manifest_json))
        header += pack_u64(table_offset)
        header += pack_u64(len(table))
        header += pack_u64(self._payload_base if self._payload_base else manifest_offset)
        header += pack_u64(file_size)
        assert len(header) == LCOCR_HEADER_SIZE
        self._file.seek(0)
        self._file.write(bytes(header))
        self._file.flush()
        self._file.close()
        self._finished = True