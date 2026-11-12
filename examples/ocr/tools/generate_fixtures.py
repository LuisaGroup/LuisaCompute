#!/usr/bin/env python3
"""generate_fixtures.py -- regenerate the FROZEN synthetic OCR test fixtures.

Standard-library only (no numpy/PIL/safetensors). This is the ONLY writer of
src/tests/data/ocr; tests NEVER create or overwrite these files. Run it
explicitly to regenerate after an intentional format change:

    python examples/ocr/tools/generate_fixtures.py

Everything here is SYNTHETIC (tiny hand-built tensors/images/tokenizers), not
official model output. golden_manifest.json labels each artifact accordingly.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcocr_format import (  # noqa: E402
    ENCODING_BF16,
    ENCODING_FP16,
    ENCODING_FP32,
    ENCODING_INT4_PACKED,
    ENCODING_INT8,
    LAYOUT_KN,
    LAYOUT_ROW_MAJOR,
    QUANT_INT4_ASYMMETRIC,
    QUANT_INT4_SYMMETRIC,
    QUANT_INT8_SYMMETRIC,
    QUANT_NONE,
    pack_int4_codes,
)
from lcocr_writer import PackageWriter  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "src", "tests", "data", "ocr")
DATA_DIR = os.path.normpath(DATA_DIR)


def _fp32_bytes(values):
    return struct.pack(f"<{len(values)}f", *values)


def _fp16_bytes(values):
    # struct 'e' is IEEE half (RNE).
    return struct.pack(f"<{len(values)}e", *values)


def _bf16_bytes(values):
    out = bytearray()
    for v in values:
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        lsb = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + lsb) & 0xFFFFFFFF
        out += struct.pack("<H", rounded >> 16)
    return bytes(out)


def _manifest(model="glm_ocr", component=""):
    return {
        "format_version": 1,
        "model": model,
        "recipe_version": "1",
        "checkpoint_revision": "synthetic-fixture-rev-0",
        "config_fingerprint": "synthetic-fingerprint-0",
        "license": "synthetic-test-only",
        "producer": "generate_fixtures.py",
        "producer_version": "1.0.0",
        "preprocessing_version": "1",
        "tokenizer_version": "1",
        "output_parser_version": "1",
        "component": component,
        "dependencies": [],
        "source_hashes": [
            {"path": "synthetic", "algorithm": "sha256", "digest": "0" * 64},
        ],
    }


def gen_package_basic(path):
    w = PackageWriter(path)
    # FP32 [2,3]
    w.add_tensor("vision.embed", ENCODING_FP32, LAYOUT_ROW_MAJOR, [2, 3],
                 _fp32_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    # FP16 [4]
    w.add_tensor("vision.norm", ENCODING_FP16, LAYOUT_ROW_MAJOR, [4],
                 _fp16_bytes([0.5, -0.5, 1.5, -1.5]))
    # BF16 [4]
    w.add_tensor("decoder.norm", ENCODING_BF16, LAYOUT_ROW_MAJOR, [4],
                 _bf16_bytes([1.0, 2.0, -1.0, 0.25]))
    # INT4 symmetric [8] group 64 (one partial group), codes 0..15 subset
    codes = [0, 8, 15, 7, 1, 9, 14, 3]
    w.add_tensor("decoder.q_proj", ENCODING_INT4_PACKED, LAYOUT_KN, [8],
                 pack_int4_codes(codes), quant_kind=QUANT_INT4_SYMMETRIC,
                 group_size=64, scales=[0.125])
    # INT4 asymmetric [5] odd count, group 64
    codes5 = [1, 2, 3, 4, 5]
    w.add_tensor("decoder.k_proj", ENCODING_INT4_PACKED, LAYOUT_KN, [5],
                 pack_int4_codes(codes5), quant_kind=QUANT_INT4_ASYMMETRIC,
                 group_size=64, scales=[0.5], zero_points=[8])
    # INT8 [7] odd count, group 64
    w.add_tensor("decoder.v_proj", ENCODING_INT8, LAYOUT_ROW_MAJOR, [7],
                 struct.pack("<7b", -127, -1, 0, 1, 127, -64, 64),
                 quant_kind=QUANT_INT8_SYMMETRIC, group_size=64, scales=[0.01])
    # INT4 group 128, 130 elements -> 2 groups (second partial)
    codes130 = [(i * 7) % 16 for i in range(130)]
    w.add_tensor("decoder.big", ENCODING_INT4_PACKED, LAYOUT_KN, [130],
                 pack_int4_codes(codes130), quant_kind=QUANT_INT4_SYMMETRIC,
                 group_size=128, scales=[0.25, 0.5])
    w.finish(_manifest())


def gen_package_components(path):
    # A layout component package referenced by a parent manifest.
    w = PackageWriter(path)
    w.add_tensor("layout.backbone", ENCODING_FP32, LAYOUT_ROW_MAJOR, [2, 2],
                 _fp32_bytes([0.1, 0.2, 0.3, 0.4]))
    w.finish(_manifest(component="layout"))


def gen_package_multirank(path):
    w = PackageWriter(path)
    # rank 0 scalar, rank 1, rank 4 conv, rank 8 (max)
    w.add_tensor("scalar", ENCODING_FP32, LAYOUT_ROW_MAJOR, [], _fp32_bytes([3.5]))
    w.add_tensor("vector", ENCODING_FP32, LAYOUT_ROW_MAJOR, [3], _fp32_bytes([1, 2, 3]))
    w.add_tensor("conv", ENCODING_FP32, LAYOUT_ROW_MAJOR, [2, 3, 2, 2],
                 _fp32_bytes([float(i) for i in range(24)]))
    w.add_tensor("rank8", ENCODING_FP32, LAYOUT_ROW_MAJOR, [1, 1, 1, 1, 1, 1, 1, 2],
                 _fp32_bytes([7.0, 8.0]))
    w.finish(_manifest())


def gen_safetensors(path):
    # Minimal synthetic safetensors: header JSON + data.
    tensors = {
        "weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        "half": {"dtype": "F16", "shape": [4], "data_offsets": [24, 32]},
        "bf": {"dtype": "BF16", "shape": [2], "data_offsets": [32, 36]},
        "byte": {"dtype": "U8", "shape": [3], "data_offsets": [36, 39]},
    }
    header = {"__metadata__": {"format": "pt", "producer": "generate_fixtures.py"}}
    header.update(tensors)
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    data = _fp32_bytes([1, 2, 3, 4, 5, 6]) + _fp16_bytes([0.5, -0.5, 1.5, -1.5]) \
        + _bf16_bytes([1.0, -2.0]) + bytes([10, 20, 30])
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(data)


def gen_tokenizer_bpe(path):
    # GPT-2 byte encoder/decoder.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = {str(b): chr(c) for b, c in zip(bs, cs)}
    byte_decoder = {chr(c): b for b, c in zip(bs, cs)}
    # Build a tiny vocab: all byte tokens + a few merges.
    vocab = {}
    idx = 0
    for b in range(256):
        vocab[byte_encoder[str(b)]] = idx
        idx += 1
    # Add merged pieces for "hello" and Chinese chars via merges.
    merges = []
    # "Ġh" + "e" -> "Ġhe" etc. We add a handful of useful merges.
    def add_merge(a, b):
        nonlocal idx
        merged = a + b
        if merged not in vocab:
            vocab[merged] = idx
            idx += 1
        merges.append(f"{a} {b}")
    # Ġ is the GPT-2 encoding of space (byte 32 -> 256+? ). Compute it.
    space_piece = byte_encoder["32"]
    add_merge(space_piece, "h")          # " h"
    add_merge(space_piece + "h", "e")    # " he"
    add_merge(space_piece + "he", "l")   # " hel"
    add_merge(space_piece + "hel", "l")  # " hell"
    add_merge(space_piece + "hell", "o") # " hello"
    added_tokens = [
        {"id": idx, "content": "<|image|>", "special": True, "lstrip": False,
         "rstrip": False, "single_word": False},
    ]
    image_id = idx
    vocab["<|image|>"] = idx
    idx += 1
    added_tokens.append({"id": idx, "content": "<s>", "special": True, "lstrip": False,
                         "rstrip": False, "single_word": False})
    bos_id = idx
    vocab["<s>"] = idx
    idx += 1
    added_tokens.append({"id": idx, "content": "</s>", "special": True, "lstrip": False,
                         "rstrip": False, "single_word": False})
    eos_id = idx
    vocab["</s>"] = idx
    idx += 1
    tokenizer = {
        "version": "1.0",
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split",
                 "pattern": {"Regex": "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                 "use_regex": True},
            ],
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [bos_id], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [eos_id], "tokens": ["</s>"]},
            },
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
            "byte_encoder": byte_encoder,
            "byte_decoder": byte_decoder,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False)
    return {"image_id": image_id, "bos_id": bos_id, "eos_id": eos_id,
            "vocab_size": idx}


def gen_tokenizer_unigram(path):
    # SentencePiece-style unigram with byte fallback + metaspace.
    pieces = []
    def add(piece, score):
        pieces.append([piece, score])
    add("<unk>", 0.0)
    add("<s>", 0.0)
    add("</s>", 0.0)
    # metaspace pieces
    add("\u2581", -1.0)
    add("\u2581hello", -2.0)
    add("\u2581world", -2.5)
    add("h", -3.0)
    add("e", -3.0)
    add("l", -3.0)
    add("o", -3.0)
    add("\u2581\u4f60\u597d", -2.0)  # 你好
    add("\u4f60", -3.0)
    add("\u597d", -3.0)
    # byte fallback tokens
    for b in range(256):
        add(f"<0x{b:02X}>", -10.0)
    tokenizer = {
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": True, "lstrip": False, "rstrip": False, "single_word": False},
            {"id": 1, "content": "<s>", "special": True, "lstrip": False, "rstrip": False, "single_word": False},
            {"id": 2, "content": "</s>", "special": True, "lstrip": False, "rstrip": False, "single_word": False},
        ],
        "normalizer": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": True},
        "pre_tokenizer": None,
        "post_processor": None,
        "model": {
            "type": "Unigram",
            "unk_id": 0,
            "byte_fallback": True,
            "vocab": pieces,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False)


def _png_chunk(tag, data):
    chunk = tag + data
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)


def gen_png(path, width, height, rgb_fn):
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            r, g, b = rgb_fn(x, y)
            raw += bytes([r, g, b])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += _png_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def _jpeg_exif_orientation(orientation):
    # Build a minimal JPEG with an EXIF APP1 segment carrying the orientation.
    # TIFF header (little-endian) + IFD0 with one tag (0x0112).
    tiff = bytearray()
    tiff += b"II"
    tiff += struct.pack("<H", 42)
    tiff += struct.pack("<I", 8)  # IFD offset
    tiff += struct.pack("<H", 1)  # one entry
    tiff += struct.pack("<H", 0x0112)  # orientation tag
    tiff += struct.pack("<H", 3)  # SHORT
    tiff += struct.pack("<I", 1)  # count
    tiff += struct.pack("<H", orientation)
    tiff += b"\x00\x00"  # value padding
    tiff += struct.pack("<I", 0)  # next IFD
    exif_payload = b"Exif\x00\x00" + bytes(tiff)
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif_payload) + 2) + exif_payload
    return app1


def gen_jpeg_with_exif(path, orientation):
    # A tiny baseline JPEG (8x8 gray) produced by a minimal encoder is complex;
    # instead we embed EXIF into a valid 1x1 JPEG. We construct a minimal
    # valid JPEG byte stream (SOI + APP1 + DQT + SOF0 + DHT + SOS + EOI).
    # For fixture purposes we use a known-good 1x1 white JPEG base and insert
    # the EXIF APP1 right after SOI.
    base = bytes([
        0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0x00,
    ]) + bytes([8] * 64) + bytes([
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
        0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03,
        0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04,
        0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71,
        0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1,
        0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A,
        0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
        0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
        0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74,
        0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4,
        0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8,
        0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3,
        0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6,
        0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9,
        0xFA,
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x7B, 0x40,
        0xFF, 0xD9,
    ])
    # Insert EXIF APP1 after SOI (first 2 bytes).
    out = base[:2] + _jpeg_exif_orientation(orientation) + base[2:]
    with open(path, "wb") as f:
        f.write(out)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    gen_package_basic(os.path.join(DATA_DIR, "synthetic_basic.lcocr"))
    gen_package_components(os.path.join(DATA_DIR, "synthetic_layout.lcocr"))
    gen_package_multirank(os.path.join(DATA_DIR, "synthetic_multirank.lcocr"))
    gen_safetensors(os.path.join(DATA_DIR, "synthetic.safetensors"))
    bpe_meta = gen_tokenizer_bpe(os.path.join(DATA_DIR, "tokenizer_bpe.json"))
    gen_tokenizer_unigram(os.path.join(DATA_DIR, "tokenizer_unigram.json"))
    # Tiny images: blank, wide, mixed-color.
    gen_png(os.path.join(DATA_DIR, "blank_4x4.png"), 4, 4, lambda x, y: (255, 255, 255))
    gen_png(os.path.join(DATA_DIR, "wide_16x2.png"), 16, 2, lambda x, y: (x * 16 % 256, 128, 255 - x * 16 % 256))
    gen_png(os.path.join(DATA_DIR, "mixed_8x8.png"), 8, 8,
            lambda x, y: ((x * 32) % 256, (y * 32) % 256, ((x + y) * 16) % 256))
    gen_jpeg_with_exif(os.path.join(DATA_DIR, "exif_rotate90.jpg"), 6)
    gen_jpeg_with_exif(os.path.join(DATA_DIR, "exif_normal.jpg"), 1)
    # Document manifest referencing the tiny images.
    doc_manifest = {
        "format_version": 1,
        "producer": "generate_fixtures.py",
        "source": "synthetic_document",
        "rasterizer": "synthetic",
        "dpi": 72,
        "pages": [
            {"source_page_index": 0, "image": "blank_4x4.png", "width": 4, "height": 4},
            {"source_page_index": 1, "image": "mixed_8x8.png", "width": 8, "height": 8},
        ],
    }
    with open(os.path.join(DATA_DIR, "document_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(doc_manifest, f, ensure_ascii=False, indent=2)
    # Golden manifest: labels synthetic vs official.
    golden_manifest = {
        "format_version": 1,
        "note": "All artifacts below are SYNTHETIC test fixtures, NOT official model outputs. "
                "Official goldens are produced separately by tools/export_golden.py and are "
                "never regenerated by tests.",
        "artifacts": [
            {"file": "synthetic_basic.lcocr", "kind": "synthetic", "type": "package",
             "description": "FP32/FP16/BF16 + INT4 sym/asym + INT8, group 64/128, odd counts"},
            {"file": "synthetic_layout.lcocr", "kind": "synthetic", "type": "package",
             "description": "layout component package"},
            {"file": "synthetic_multirank.lcocr", "kind": "synthetic", "type": "package",
             "description": "rank 0/1/4/8 tensors"},
            {"file": "synthetic.safetensors", "kind": "synthetic", "type": "safetensors",
             "description": "F32/F16/BF16/U8 tensors"},
            {"file": "tokenizer_bpe.json", "kind": "synthetic", "type": "tokenizer",
             "description": "byte-level BPE, GPT-2 byte mapping, added/special tokens",
             "meta": bpe_meta},
            {"file": "tokenizer_unigram.json", "kind": "synthetic", "type": "tokenizer",
             "description": "Unigram with byte fallback + metaspace"},
            {"file": "blank_4x4.png", "kind": "synthetic", "type": "image", "description": "blank white 4x4"},
            {"file": "wide_16x2.png", "kind": "synthetic", "type": "image", "description": "very wide 16x2"},
            {"file": "mixed_8x8.png", "kind": "synthetic", "type": "image", "description": "mixed-color 8x8"},
            {"file": "exif_rotate90.jpg", "kind": "synthetic", "type": "image", "description": "JPEG EXIF orientation 6"},
            {"file": "exif_normal.jpg", "kind": "synthetic", "type": "image", "description": "JPEG EXIF orientation 1"},
            {"file": "document_manifest.json", "kind": "synthetic", "type": "document",
             "description": "2-page synthetic document manifest"},
        ],
        "official": [],
    }
    with open(os.path.join(DATA_DIR, "golden_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(golden_manifest, f, ensure_ascii=False, indent=2)
    print(f"generate_fixtures: wrote fixtures to {DATA_DIR}")


if __name__ == "__main__":
    main()