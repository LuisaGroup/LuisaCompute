"""Shared .lcocr v1 / .ocrfeat v1 format constants and checked helpers.

This module is the single source of truth for the on-disk layout shared by
pack_model.py, validate_package.py, preprocess.py and the C++ reader/writer
(examples/ocr/ocr/package.cpp, feature_bundle.cpp). It is standard-library
only so fixture generation and validation never need third-party packages.

The layout is documented in examples/ocr/FORMAT.md. Every offset/size is
64-bit little-endian; payloads are 256-byte aligned with zero padding.
"""
from __future__ import annotations

import struct

# --- .lcocr v1 ---
LCOCR_MAGIC = b"LCOCR1\x00\x00"  # 8 bytes
LCOCR_FORMAT_VERSION = 1
LCOCR_HEADER_SIZE = 64
LCOCR_PAYLOAD_ALIGNMENT = 256
LCOCR_MAX_RANK = 8
LCOCR_MAX_NAME_BYTES = 4096
LCOCR_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
LCOCR_MAX_TABLE_BYTES = 256 * 1024 * 1024
LCOCR_MAX_TENSORS = 1 << 20

# TensorEncoding values (must match package.h).
ENCODING_FP32 = 0
ENCODING_FP16 = 1
ENCODING_BF16 = 2
ENCODING_INT4_PACKED = 3
ENCODING_INT8 = 4

# TensorLayout values.
LAYOUT_ROW_MAJOR = 0
LAYOUT_KN = 1
LAYOUT_NK = 2
LAYOUT_CONV_OIHW = 3
LAYOUT_CONV_OHWI = 4

# QuantKind values (must match quantization.h).
QUANT_NONE = 0
QUANT_INT4_SYMMETRIC = 1
QUANT_INT4_ASYMMETRIC = 2
QUANT_INT8_SYMMETRIC = 3

SUPPORTED_GROUP_SIZES = (64, 128)

# --- .ocrfeat v1 ---
OCRF_MAGIC = b"OCRF"  # u32 little-endian
OCRF_FORMAT_VERSION = 1
OCRF_HEADER_SIZE = 64
OCRF_ALIGNMENT = 256
OCRF_MAX_RANK = 8

FEATURE_DTYPE_F32 = 0
FEATURE_DTYPE_I32 = 1
FEATURE_DTYPE_I64 = 2
FEATURE_DTYPE_U8 = 3


class FormatError(ValueError):
    """Raised on a malformed/overflowing artifact (ordinary input error)."""


def checked_add(a: int, b: int) -> int:
    r = a + b
    if r < 0 or r > 0xFFFFFFFFFFFFFFFF:
        raise FormatError(f"checked add overflow: {a} + {b}")
    return r


def checked_mul(a: int, b: int) -> int:
    r = a * b
    if r < 0 or r > 0xFFFFFFFFFFFFFFFF:
        raise FormatError(f"checked mul overflow: {a} * {b}")
    return r


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise FormatError(f"alignment must be a power of two: {alignment}")
    return (value + alignment - 1) & ~(alignment - 1)


def element_count(shape) -> int:
    count = 1
    for d in shape:
        count = checked_mul(count, int(d))
    return count


def group_count(count: int, group_size: int) -> int:
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise FormatError(f"group size must be 64 or 128: {group_size}")
    return (count + group_size - 1) // group_size


def int4_packed_size(count: int) -> int:
    return (count + 1) // 2


def dense_element_size(encoding: int) -> int:
    return {ENCODING_FP32: 4, ENCODING_FP16: 2, ENCODING_BF16: 2}.get(encoding, 0)


def is_quantized(encoding: int) -> bool:
    return encoding in (ENCODING_INT4_PACKED, ENCODING_INT8)


def row_major_strides(shape):
    strides = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc = checked_mul(acc, int(shape[i]))
    return strides


def pack_u32(v: int) -> bytes:
    return struct.pack("<I", v)


def pack_u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def pack_int4_codes(codes) -> bytes:
    """Pack unsigned nibble codes (0..15), low nibble first, zero pad."""
    for c in codes:
        if not 0 <= int(c) <= 15:
            raise FormatError(f"INT4 code out of range: {c}")
    out = bytearray(int4_packed_size(len(codes)))
    for i, c in enumerate(codes):
        out[i // 2] |= int(c) << ((i % 2) * 4)
    return bytes(out)


def unpack_int4_codes(packed: bytes, count: int):
    if len(packed) != int4_packed_size(count):
        raise FormatError("packed size does not match element count")
    if count % 2 == 1 and packed and (packed[-1] & 0xF0) != 0:
        raise FormatError("nonzero padding nibble in odd-length INT4 payload")
    codes = []
    for i in range(count):
        codes.append((packed[i // 2] >> ((i % 2) * 4)) & 0x0F)
    return codes