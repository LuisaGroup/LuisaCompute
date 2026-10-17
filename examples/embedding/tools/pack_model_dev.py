#!/usr/bin/env python3
"""pack_model_dev.py -- developer-side helpers for the bge-m3 weight packer
(embedding_model_plan.md section 3.3 / milestone M2; examples/embedding/tools/
pack_model.cpp is the real, dependency-free implementation).

Dev-only convenience script. It is NOT part of the build or the runtime, and it
does not need onnx/torch (stdlib only; numpy is used when available for the
optional cross-check against the ONNX external data, but is not required).

Usage
  python pack_model_dev.py inspect <model.bgm> [--entries N] [--names]
      Parse and verify the BGM3 header + tensor table: magic/version/precision/
      tensor_count, 256-byte blob alignment, size_bytes == prod(dims)*4, rank vs
      dims, ascending non-overlapping offsets, blobs inside the file. Resolves
      every name_hash back to the canonical name (FNV-1a-64 over the list below,
      which mirrors the spec in pack_model.cpp) and prints the first N entries.

  python pack_model_dev.py dump <model.bgm> <canonical name> <out.bin>
      Extract one tensor blob (raw float32 little-endian) from a packed model.

  python pack_model_dev.py dump-pt <file.pt> [--outdir DIR]
      List the tensors inside a torch .pt checkpoint (zip + pickle, stdlib only),
      convert float16 storages to float32, and write each tensor to
      DIR/<name>.bin (raw float32 LE). Handy for eyeballing the retrieval heads.

  python pack_model_dev.py cross <model_dir> <model.bgm> <name> [--layer I]
      Cross-check a packed tensor against the source assets: re-reads the ONNX
      external data by offset/length (using the minimal protobuf walk below is
      out of scope here -- it uses the onnx package if installed) and compares.
      Requires the `onnx` + `numpy` dev packages; purely optional.
"""

import argparse
import json
import io
import os
import pickle
import struct
import sys
import zipfile

LAYERS, HIDDEN, INTERMEDIATE = 24, 1024, 4096


def canonical_names():
    """The emission-order list documented at the top of pack_model.cpp."""
    names = [(f"emb.word", (250002, HIDDEN)), (f"emb.pos", (8194, HIDDEN)),
             (f"emb.ln.gamma", (HIDDEN,)), (f"emb.ln.beta", (HIDDEN,))]
    for i in range(LAYERS):
        p = f"layer{i}."
        names += [(p + "qkv", (HIDDEN, 3 * HIDDEN)), (p + "qkv.bias", (3 * HIDDEN,)),
                  (p + "attn.out", (HIDDEN, HIDDEN)), (p + "attn.out.bias", (HIDDEN,)),
                  (p + "ln1.gamma", (HIDDEN,)), (p + "ln1.beta", (HIDDEN,)),
                  (p + "ffn.up", (HIDDEN, INTERMEDIATE)), (p + "ffn.up.bias", (INTERMEDIATE,)),
                  (p + "ffn.down", (INTERMEDIATE, HIDDEN)), (p + "ffn.down.bias", (HIDDEN,)),
                  (p + "ln2.gamma", (HIDDEN,)), (p + "ln2.beta", (HIDDEN,))]
    names += [("head.sparse.w", (HIDDEN, 1)), ("head.sparse.b", (1,)),
              ("head.colbert.w", (HIDDEN, HIDDEN)), ("head.colbert.b", (HIDDEN,))]
    return names


def fnv1a_64(s: str) -> int:
    h = 0xcbf29ce484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def align_up(v, a):
    return (v + a - 1) // a * a


def read_table(path):
    with open(path, "rb") as f:
        size = os.path.getsize(path)
        head = f.read(16)
        if len(head) != 16:
            raise SystemExit("file too small for a BGM3 header")
        magic = head[:4]
        version, precision, count = struct.unpack("<III", head[4:16])
        if magic != b"BGM3":
            raise SystemExit(f"bad magic {magic!r}, expected b'BGM3'")
        table = f.read(48 * count)
        entries = []
        for i in range(count):
            e = table[48 * i: 48 * (i + 1)]
            name_hash, offset, size_bytes = struct.unpack("<QQQ", e[:24])
            rank = struct.unpack("<I", e[24:28])[0]
            dims = struct.unpack("<IIII", e[28:44])
            entries.append(dict(hash=name_hash, offset=offset, size=size_bytes, rank=rank, dims=dims))
    return size, version, precision, count, entries


def cmd_inspect(args):
    size, version, precision, count, entries = read_table(args.bgm)
    known = {fnv1a_64(n): n for n, _ in canonical_names()}
    shapes = {fnv1a_64(n): d for n, d in canonical_names()}
    print(f"{args.bgm}: {size} bytes")
    print(f"magic BGM3, version {version}, precision {'fp32' if precision == 0 else f'#{precision}'}, "
          f"tensor_count {count}")
    print(f"table = 16 + 48*{count} = {16 + 48 * count} bytes; first blob expected at "
          f"{align_up(16 + 48 * count, 256)}")
    problems = []
    expect_off = align_up(16 + 48 * count, 256)
    payload = 0
    for i, e in enumerate(entries):
        name = known.get(e["hash"], f"<unknown 0x{e['hash']:016x}>")
        elems = 1
        for d in e["dims"][: e["rank"]]:
            elems *= d
        if e["rank"] < 1 or e["rank"] > 4:
            problems.append(f"[{i}] rank {e['rank']} out of range")
        if any(d == 0 for d in e["dims"][: e["rank"]]):
            problems.append(f"[{i}] {name}: zero dim")
        if any(d != 0 for d in e["dims"][e["rank"]:]):
            problems.append(f"[{i}] {name}: dims beyond rank are not zero")
        if e["size"] != elems * 4:
            problems.append(f"[{i}] {name}: size {e['size']} != {elems}*4")
        if e["offset"] % 256:
            problems.append(f"[{i}] {name}: offset {e['offset']} is not 256-byte aligned")
        if e["offset"] != expect_off:
            problems.append(f"[{i}] {name}: offset {e['offset']} != expected {expect_off}")
        if e["offset"] + e["size"] > size:
            problems.append(f"[{i}] {name}: blob runs past the file")
        if shapes.get(e["hash"]) and tuple(e["dims"][: e["rank"]]) != shapes[e["hash"]]:
            problems.append(f"[{i}] {name}: dims {e['dims'][:e['rank']]} != spec {shapes[e['hash']]}")
        expect_off = align_up(e["offset"] + e["size"], 256)
        payload += e["size"]
    if expect_off != size:
        problems.append(f"file size {size} != computed {expect_off}")
    print(f"  payload {payload} bytes ({payload / 2**30:.3f} GiB), "
          f"alignment padding {size - 16 - 48 * count - payload} bytes")
    print(f"  first {args.entries} table entries:")
    for i, e in enumerate(entries[: args.entries]):
        name = known.get(e["hash"], f"<unknown 0x{e['hash']:016x}>")
        print(f"   [{i:3d}] {name:20s} hash 0x{e['hash']:016x} offset 0x{e['offset']:012x} "
              f"size {e['size']:10d} rank {e['rank']} dims {e['dims'][:e['rank']]}")
    if args.names:
        print("  all canonical names (in table order):")
        for i, e in enumerate(entries):
            print(f"   [{i:3d}] {known.get(e['hash'], '<unknown>')}")
    unknown = [e for e in entries if e["hash"] not in known]
    if len(entries) != len(canonical_names()):
        problems.append(f"tensor_count {count} != spec {len(canonical_names())}")
    if unknown:
        problems.append(f"{len(unknown)} entries have hashes outside the canonical name set")
    if problems:
        print("PROBLEMS:")
        for p in problems[:40]:
            print("  -", p)
        return 1
    print("package structure verified: alignment, sizes, dims, ordering and the "
          "canonical name set all check out")
    return 0


def cmd_dump(args):
    _, _, _, _, entries = read_table(args.bgm)
    want = fnv1a_64(args.name)
    hit = [e for e in entries if e["hash"] == want]
    if not hit:
        raise SystemExit(f"'{args.name}' is not in the package")
    e = hit[0]
    with open(args.bgm, "rb") as f:
        f.seek(e["offset"])
        blob = f.read(e["size"])
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"wrote {args.out}: {e['size']} bytes, dims {e['dims'][:e['rank']]}")
    return 0


class _StorageClass:
    """Stand-in for torch.<X>Storage globals referenced by data.pkl."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):  # never actually constructed by these checkpoints
        return None

    def __repr__(self):
        return f"<{self.name}>"


class _Storage:
    def __init__(self, cls, key, device, size):
        self.cls, self.key, self.device, self.size = cls, key, device, size


class _Tensor:
    def __init__(self, storage, offset, size):
        self.storage, self.offset, self.size = storage, offset, size


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, hooks):
    return _Tensor(storage, storage_offset, tuple(size))


def _noop(*a, **k):
    return None


class _StateDictUnpickler(pickle.Unpickler):
    """Reads a torch state_dict pickle without importing torch: the storage
    classes and _rebuild_tensor_v2 are replaced by the metadata holders above."""

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if name.endswith("Storage"):
            return _StorageClass(name)
        if module == "collections" and name == "OrderedDict":
            return dict
        return _noop

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            _, cls, key, device, *rest = pid
            return _Storage(cls, key, device, rest[-1] if rest else None)
        raise SystemExit(f"unsupported persistent id: {pid!r}")


def cmd_dump_pt(args):
    zf = zipfile.ZipFile(args.pt)
    names = zf.namelist()
    pkl_name = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]
    byteorder = zf.read(prefix + "byteorder").decode()
    print(f"{args.pt}: members={names} byteorder={byteorder}")
    if byteorder != "little":
        raise SystemExit("only little-endian checkpoints are supported")
    state = _StateDictUnpickler(io.BytesIO(zf.read(pkl_name))).load()
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pt))[0]
    for key, t in state.items():
        if not isinstance(t, _Tensor):
            print(f"  {key}: <not a tensor: {type(t).__name__}>")
            continue
        cls = getattr(t.storage.cls, "name", str(t.storage.cls))
        esz = {"HalfStorage": 2, "FloatStorage": 4, "DoubleStorage": 8,
               "IntStorage": 4, "LongStorage": 8}.get(cls, 0)
        if esz == 0:
            raise SystemExit(f"{key}: unsupported storage type '{cls}'")
        raw = zf.read(prefix + "data/" + t.storage.key)
        n = 1
        for d in t.size:
            n *= d
        vals = struct.unpack("<" + {2: "e", 4: "f", 8: "d"}[esz] * n, raw[t.offset * esz: t.offset * esz + n * esz])
        out = os.path.join(args.outdir, f"{stem}.{key}.bin")
        with open(out, "wb") as f:
            f.write(struct.pack("<" + "f" * n, *(float(v) for v in vals)))
        print(f"  {key}: sizes={tuple(t.size)} storage={cls}/{t.storage.key} -> {out} ({n * 4} bytes fp32)")
    return 0


def cmd_cross(args):
    try:
        import numpy as np
        import onnx
    except ImportError:
        raise SystemExit("the 'cross' subcommand needs the dev packages: pip install onnx numpy")
    m = onnx.load(os.path.join(args.model_dir, "onnx", "model.onnx"), load_external_data=False)
    inits = {i.name: i for i in m.graph.initializer}
    data = os.path.join(args.model_dir, "onnx", "model.onnx_data")
    _, _, _, _, entries = read_table(args.bgm)
    known = {fnv1a_64(n): n for n, _ in canonical_names()}
    name = known.get(fnv1a_64(args.name)) or args.name
    e = next(x for x in entries if x["hash"] == fnv1a_64(name))
    if args.layer is not None and "layer" not in name:
        raise SystemExit("--layer only makes sense for layer tensors")
    src = args.source or name
    if src not in inits:
        raise System(f"source initializer '{src}' not in the ONNX graph")
    i = inits[src]
    ext = {x.key: x.value for x in i.external_data}
    with open(data, "rb") as f:
        f.seek(int(ext["offset"]))
        blob = f.read(int(ext["length"]))
    a = np.frombuffer(blob, dtype="<f4").reshape(list(i.dims))
    with open(args.bgm, "rb") as f:
        f.seek(e["offset"])
        b = np.frombuffer(f.read(e["size"]), dtype="<f4").reshape(list(e["dims"][:e["rank"]]))
    if a.shape != b.shape:
        a = a.T
    same = a.shape == b.shape and np.array_equal(a, b)
    print(f"{name} vs {src}: shapes {a.shape}/{b.shape} identical={same} "
          f"maxdiff={np.abs(a - b).max() if a.shape == b.shape else 'n/a'}")
    return 0 if same else 1


def main():
    ap = argparse.ArgumentParser(description="bge-m3 packer dev helpers")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("inspect")
    p.add_argument("bgm")
    p.add_argument("--entries", type=int, default=8)
    p.add_argument("--names", action="store_true")
    p.set_defaults(fn=cmd_inspect)
    p = sp.add_parser("dump")
    p.add_argument("bgm")
    p.add_argument("name")
    p.add_argument("out")
    p.set_defaults(fn=cmd_dump)
    p = sp.add_parser("dump-pt")
    p.add_argument("pt")
    p.add_argument("--outdir", default=".")
    p.set_defaults(fn=cmd_dump_pt)
    p = sp.add_parser("cross")
    p.add_argument("model_dir")
    p.add_argument("bgm")
    p.add_argument("name")
    p.add_argument("--source")
    p.add_argument("--layer", type=int)
    p.set_defaults(fn=cmd_cross)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
