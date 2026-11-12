#!/usr/bin/env python3
"""validate_package.py -- reopen a .lcocr artifact and check every range.

Standard-library only. Reopens the ACTUAL artifact (not an in-memory model),
parses the header/manifest/table, validates every offset/size/name/shape/
stride/scale/zero-point, checks alignment/overlap/padding/trailing data, and
decodes sample tensors (dense FP32/FP16/BF16 and grouped INT4/INT8) to prove
the quantization metadata is sufficient to decode without reinterpretation.

Exit code 0 = valid; 2 = malformed/unsupported. Ordinary malformed input is a
reported error, never a crash.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcocr_format import (  # noqa: E402
    LCOCR_HEADER_SIZE,
    LCOCR_MAGIC,
    LCOCR_MAX_RANK,
    LCOCR_PAYLOAD_ALIGNMENT,
    FormatError,
    checked_add,
    checked_mul,
    dense_element_size,
    element_count,
    group_count,
    int4_packed_size,
    is_quantized,
    unpack_int4_codes,
    QUANT_INT4_ASYMMETRIC,
    QUANT_INT4_SYMMETRIC,
    QUANT_INT8_SYMMETRIC,
    QUANT_NONE,
    ENCODING_INT4_PACKED,
    ENCODING_INT8,
    ENCODING_FP16,
    ENCODING_BF16,
)


class Validator:
    def __init__(self, path):
        self._path = path
        self._file = open(path, "rb")
        self._file.seek(0, os.SEEK_END)
        self._file_size = self._file.tell()
        self._file.seek(0)
        self.tensors = []
        self.manifest = {}

    def close(self):
        self._file.close()

    def _read(self, offset, size):
        end = checked_add(offset, size)
        if end > self._file_size:
            raise FormatError(f"read range [{offset},{end}) exceeds file size {self._file_size}")
        self._file.seek(offset)
        data = self._file.read(size)
        if len(data) != size:
            raise FormatError("short read")
        return data

    def validate(self, decode_samples=True):
        if self._file_size < LCOCR_HEADER_SIZE:
            raise FormatError("file smaller than header")
        header = self._read(0, LCOCR_HEADER_SIZE)
        magic = header[0:8]
        if magic != LCOCR_MAGIC:
            raise FormatError(f"bad magic: {magic!r}")
        version, header_size = struct.unpack_from("<II", header, 8)
        if version != 1:
            raise FormatError(f"unsupported version {version}")
        if header_size != LCOCR_HEADER_SIZE:
            raise FormatError(f"unexpected header size {header_size}")
        (manifest_offset, manifest_size, table_offset, table_size,
         payload_base, file_size) = struct.unpack_from("<QQQQQQ", header, 16)
        if file_size != self._file_size:
            raise FormatError(f"header file_size {file_size} != actual {self._file_size}")
        if manifest_size == 0:
            raise FormatError("empty manifest")
        manifest_end = checked_add(manifest_offset, manifest_size)
        if manifest_offset < LCOCR_HEADER_SIZE or manifest_end > file_size:
            raise FormatError("manifest range out of bounds")
        table_end = checked_add(table_offset, table_size)
        if table_size < 8 or table_offset < manifest_end or table_end > file_size:
            raise FormatError("table range out of bounds/overlaps manifest")
        if table_end != file_size:
            raise FormatError("trailing data after the tensor table")
        if payload_base < LCOCR_HEADER_SIZE or payload_base > manifest_offset or payload_base % LCOCR_PAYLOAD_ALIGNMENT != 0:
            raise FormatError("payload base misaligned/out of bounds")

        manifest_bytes = self._read(manifest_offset, manifest_size)
        self.manifest = json.loads(manifest_bytes.decode("utf-8"))
        for key in ("model", "recipe_version", "checkpoint_revision", "config_fingerprint",
                    "producer", "producer_version", "preprocessing_version",
                    "tokenizer_version", "output_parser_version"):
            if not self.manifest.get(key):
                raise FormatError(f"manifest missing required field {key}")

        table = self._read(table_offset, table_size)
        pos = 0

        def u32():
            nonlocal pos
            v = struct.unpack_from("<I", table, pos)[0]
            pos += 4
            return v

        def u64():
            nonlocal pos
            v = struct.unpack_from("<Q", table, pos)[0]
            pos += 8
            return v

        count = u64()
        ranges = []
        names = set()
        for _ in range(count):
            name_size = u32()
            rank = u32()
            encoding = u32()
            layout = u32()
            quant_kind = u32()
            group_size = u32()
            reserved32 = u32()
            elem_count = u64()
            if reserved32 != 0:
                raise FormatError("reserved tensor field nonzero")
            if name_size == 0 or name_size > 4096:
                raise FormatError("invalid name size")
            if rank > LCOCR_MAX_RANK:
                raise FormatError(f"rank {rank} exceeds max")
            if encoding > ENCODING_INT8:
                raise FormatError(f"unsupported encoding {encoding}")
            shape = [u64() for _ in range(rank)]
            strides = [u64() for _ in range(rank)]
            payload_offset = u64()
            payload_size = u64()
            scales_offset = u64()
            scales_size = u64()
            zp_offset = u64()
            zp_size = u64()
            name = table[pos:pos + name_size].decode("utf-8")
            pos += name_size
            if name in names:
                raise FormatError(f"duplicate tensor name {name}")
            names.add(name)
            if element_count(shape) != elem_count:
                raise FormatError(f"{name}: shape product != element_count")
            # row-major strides
            acc = 1
            for i in range(rank - 1, -1, -1):
                if strides[i] != acc:
                    raise FormatError(f"{name}: strides not row-major")
                acc = checked_mul(acc, shape[i])
            if is_quantized(encoding):
                if encoding == ENCODING_INT4_PACKED:
                    if quant_kind not in (QUANT_INT4_SYMMETRIC, QUANT_INT4_ASYMMETRIC):
                        raise FormatError(f"{name}: INT4 needs INT4 quant kind")
                    expected = int4_packed_size(elem_count)
                else:
                    if quant_kind != QUANT_INT8_SYMMETRIC:
                        raise FormatError(f"{name}: INT8 needs int8_symmetric")
                    expected = elem_count
                groups = group_count(elem_count, group_size)
                if scales_size != groups * 4:
                    raise FormatError(f"{name}: scales size mismatch")
                expected_zp = groups if quant_kind == QUANT_INT4_ASYMMETRIC else 0
                if zp_size != expected_zp:
                    raise FormatError(f"{name}: zero-point size mismatch")
            else:
                if quant_kind != QUANT_NONE or group_size != 0 or scales_size != 0 or zp_size != 0:
                    raise FormatError(f"{name}: dense carries quant metadata")
                expected = checked_mul(elem_count, dense_element_size(encoding))
            if payload_size != expected:
                raise FormatError(f"{name}: payload size mismatch")
            for off, size, what in ((payload_offset, payload_size, "payload"),
                                    (scales_offset, scales_size, "scales"),
                                    (zp_offset, zp_size, "zero_points")):
                if size == 0:
                    # Canonical zero-size rule (matches C++): offset 0 OR any
                    # 256-byte aligned in-bounds position in the payload region.
                    if off != 0 and (off < payload_base or off > manifest_offset
                                     or off % LCOCR_PAYLOAD_ALIGNMENT != 0):
                        raise FormatError(f"{name}: empty {what} offset is not 0 or an aligned in-bounds position")
                    continue
                end = checked_add(off, size)
                if off < payload_base or end > file_size or off % LCOCR_PAYLOAD_ALIGNMENT != 0:
                    raise FormatError(f"{name}: {what} range out of bounds/misaligned")
                ranges.append((off, size, name))
            self.tensors.append({
                "name": name, "encoding": encoding, "quant_kind": quant_kind,
                "group_size": group_size, "shape": shape, "element_count": elem_count,
                "payload_offset": payload_offset, "payload_size": payload_size,
                "scales_offset": scales_offset, "scales_size": scales_size,
                "zero_points_offset": zp_offset, "zero_points_size": zp_size,
            })
        if pos != len(table):
            raise FormatError("trailing bytes after table")
        # overlap / padding / coverage. An EMPTY package has no payload ranges:
        # its payload region is empty and payload_base == manifest_offset.
        if not ranges and payload_base != manifest_offset:
            raise FormatError("empty package must have payload_base == manifest_offset")
        ranges.sort()
        cursor = LCOCR_HEADER_SIZE
        for off, size, name in ranges:
            if off < cursor:
                raise FormatError(f"{name}: payload overlaps another blob")
            if off > cursor:
                gap = self._read(cursor, off - cursor)
                if any(b != 0 for b in gap):
                    raise FormatError("nonzero padding bytes")
            cursor = off + size
        # Remaining [cursor, manifest_offset) is trailing zero padding (the whole
        # payload region for an EMPTY package). It must be all zero.
        if cursor > manifest_offset:
            raise FormatError("payload region overruns the manifest")
        if cursor < manifest_offset:
            gap = self._read(cursor, manifest_offset - cursor)
            if any(b != 0 for b in gap):
                raise FormatError("nonzero trailing padding bytes")
        # scale / zero-point values + sample decode
        for t in self.tensors:
            if not is_quantized(t["encoding"]):
                continue
            scales = struct.unpack(f"<{t['scales_size'] // 4}f",
                                   self._read(t["scales_offset"], t["scales_size"]))
            for s in scales:
                if not math.isfinite(s) or s < 0:
                    raise FormatError(f"{t['name']}: non-finite/negative scale")
            zp = self._read(t["zero_points_offset"], t["zero_points_size"]) if t["zero_points_size"] else b""
            if t["quant_kind"] == QUANT_INT4_ASYMMETRIC:
                for z in zp:
                    if z > 15:
                        raise FormatError(f"{t['name']}: zero point out of range")
            if decode_samples:
                self._decode_sample(t, scales, zp)
        return True

    def _decode_sample(self, t, scales, zp):
        payload = self._read(t["payload_offset"], t["payload_size"])
        n = t["element_count"]
        gs = t["group_size"]
        if t["encoding"] == ENCODING_INT4_PACKED:
            codes = unpack_int4_codes(payload, n)
            symmetric = t["quant_kind"] == QUANT_INT4_SYMMETRIC
            for i in range(min(n, 256)):
                z = 8 if symmetric else zp[i // gs]
                _ = (codes[i] - z) * scales[i // gs]
        elif t["encoding"] == ENCODING_INT8:
            codes = struct.unpack(f"<{n}b", payload)
            for i in range(min(n, 256)):
                if codes[i] == -128:
                    raise FormatError(f"{t['name']}: reserved INT8 code -128")
                _ = codes[i] * scales[i // gs]

    def decode_dense_sample(self, name, limit=256):
        t = next((x for x in self.tensors if x["name"] == name), None)
        if t is None:
            raise FormatError(f"no such tensor {name}")
        if is_quantized(t["encoding"]):
            raise FormatError("dense decode on a quantized tensor")
        payload = self._read(t["payload_offset"], t["payload_size"])
        n = min(t["element_count"], limit)
        if t["encoding"] == ENCODING_FP16:
            return list(struct.unpack(f"<{n}e", payload[:n * 2]))
        if t["encoding"] == ENCODING_BF16:
            out = []
            for i in range(n):
                bits = struct.unpack_from("<H", payload, i * 2)[0] << 16
                out.append(struct.unpack("<f", struct.pack("<I", bits))[0])
            return out
        return list(struct.unpack(f"<{n}f", payload[:n * 4]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate_package.py",
        description="Reopen a .lcocr artifact and validate every range/name/shape/scale; decode samples.",
    )
    p.add_argument("package", help="Path to the .lcocr artifact to validate.")
    p.add_argument("--no-decode", action="store_true", help="Skip sample decoding.")
    p.add_argument("--json", action="store_true", help="Emit a JSON summary on success.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.package):
        print(f"validate_package: error: not found: {args.package}", file=sys.stderr)
        return 2
    validator = Validator(args.package)
    try:
        validator.validate(decode_samples=not args.no_decode)
    except (FormatError, ValueError, struct.error, UnicodeDecodeError) as e:
        print(f"validate_package: INVALID: {e}", file=sys.stderr)
        return 2
    finally:
        validator.close()
    if args.json:
        print(json.dumps({
            "ok": True,
            "tensors": len(validator.tensors),
            "model": validator.manifest.get("model"),
            "file_size": os.path.getsize(args.package),
        }, indent=2))
    else:
        print(f"validate_package: OK ({len(validator.tensors)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())