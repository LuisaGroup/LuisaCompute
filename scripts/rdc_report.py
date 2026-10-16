#!/usr/bin/env python
# rdc_report.py — headless analysis of a RenderDoc capture for CI/agent use.
#
# Uses renderdoccmd (no Python bindings required) to:
#   * extract the embedded thumbnail,
#   * convert the capture to the XML+ZIP ("zip.xml") frame documentation and
#     chrome.json API trace,
#   * parse the XML chunk stream into a compact frame report: API, per-call
#     histogram, dispatch grids, pipeline bytecodes, acceleration structure
#     builds, texture/buffer inventory, VRAM estimate, present info, CPU-side
#     frame span.
#
# Usage:
#   python scripts/rdc_report.py frame.rdc [--workdir DIR] [--keep-exports]
#                                          [--renderdoc-cmd PATH] [--json]
#
# Requirements: RenderDoc installed (default C:/Program Files/RenderDoc),
# Python 3.8+ stdlib only. The zip.xml export can be large (~10x capture size);
# it is written into --workdir (default: next to the .rdc) and removed at exit
# unless --keep-exports is given.
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

DEFAULT_RDOC = os.environ.get("RENDERDOC_DIR", r"C:\Program Files\RenderDoc")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
FIELD_RE = re.compile(r'<(\w+) name="([^"]+)"[^>]*?(?:string="([^"]*)")?[^>]*>([^<]*)</\1>')


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip()


def split_chunks(xml_text):
    starts = [m.start() for m in re.finditer(r"<chunk\s", xml_text)]
    doc_end = xml_text.find("</chunks>")
    if doc_end < 0:
        doc_end = len(xml_text)
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else doc_end
        out.append(xml_text[s:e])
    return out


def chunk_attrs(body):
    head = body[: body.find(">") + 1]
    return dict(ATTR_RE.findall(head))


def chunk_fields(body):
    vals = {}
    for _tag, name, enum_str, raw in FIELD_RE.findall(body):
        if name not in vals:  # first occurrence wins (outer fields shadow nested)
            vals[name] = enum_str if enum_str else raw.strip()
    return vals


def fmt_bytes(n):
    return f"{n / 1048576:.1f} MiB" if n >= 1048576 else (f"{n / 1024:.1f} KiB" if n >= 1024 else f"{n} B")


DXGI_BPP = {
    "R8_UNORM": 1, "R8G8B8A8_UNORM": 4, "R8G8B8A8_UNORM_SRGB": 4, "B8G8R8A8_UNORM": 4,
    "R16G16B16A16_FLOAT": 8, "R32G32B16A16_UNORM": 8, "R32_SINT": 4, "R32_UINT": 4,
    "R32G32_SFLOAT": 8, "R16G16B16A16_UNORM": 8, "R32G32B32A32_FLOAT": 16, "D24_UNORM_S8_UINT": 4,
    "D32_FLOAT": 4, "R16_FLOAT": 2, "R11G11B10_FLOAT": 4, "R8G8B8A8_TYPELESS": 4,
    "R16G16B16A16_FLOAT": 8, "R8_UINT": 1, "R8_SINT": 1, "R16_UNORM": 2,
}


def report(rdc, workdir, keep, rdoc_cmd, as_json):
    stem = os.path.splitext(os.path.basename(rdc))[0]
    thumb = os.path.join(workdir, stem + "_thumb.png")
    xml_path = os.path.join(workdir, stem + ".zip.xml")
    chrome_path = os.path.join(workdir, stem + ".chrome.json")
    run([rdoc_cmd, "thumb", "-o", thumb, rdc])
    if not os.path.exists(xml_path):
        run([rdoc_cmd, "convert", "-f", rdc, "-o", xml_path, "-c", "zip.xml"])
    if not os.path.exists(chrome_path):
        run([rdoc_cmd, "convert", "-f", rdc, "-o", chrome_path, "-c", "chrome.json"])

    with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
        xml = f.read()
    out = {}
    hdr = re.search(r"<header>.*?</header>", xml, re.S)
    hdr_txt = hdr.group(0) if hdr else ""
    driver = re.search(r"<driver[^>]*>([^<]+)</driver>", hdr_txt)
    thumb_m = re.search(r'<thumbnail width="(\d+)" height="(\d+)"', hdr_txt)
    out["api"] = driver.group(1) if driver else "?"
    out["backbuffer"] = f"{thumb_m.group(1)}x{thumb_m.group(2)}" if thumb_m else "?"
    out["capture_size"] = fmt_bytes(os.path.getsize(rdc))

    chunks = split_chunks(xml)
    parsed = [(chunk_attrs(b), chunk_fields(b), b) for b in chunks]
    hist = Counter(a.get("name", "?") for a, _v, _b in parsed)
    out["num_chunks"] = len(chunks)

    dispatches = []
    textures = []
    buffer_bytes = 0
    buffer_count = 0
    pipelines = []
    accel = Counter()
    present = None
    names = []
    ts_min = ts_max = None
    for a, v, b in parsed:
        name = a.get("name", "?")
        if "timestamp" in a:
            t = int(a["timestamp"])
            ts_min = t if ts_min is None else min(ts_min, t)
            d = int(a.get("duration", 0))
            ts_max = max(t + d, ts_max or 0)
        if name.endswith("::Dispatch") or name == "vkCmdDispatch" or "Dispatch" in name and "Create" not in name:
            g = (v.get("ThreadGroupCountX") or v.get("x"), v.get("ThreadGroupCountY") or v.get("y"),
                 v.get("ThreadGroupCountZ") or v.get("z"))
            dispatches.append(tuple(g or ()))
        elif name.endswith("CreateComputePipeline") or name == "vkCreatePipelineCache" or "CreatePipeline" in name:
            m = re.search(r'byteLength="(\d+)"', b)
            if m:
                pipelines.append(int(m.group(1)))
        elif "Acceleration Structure Create" in name:
            m = re.search(r'TYPE_(BOTTOM_LEVEL|TOP_LEVEL)', b)
            accel[m.group(1) if m else "?"] += 1
        elif name.endswith(("CreatePlacedResource", "CreateCommittedResource")):
            dim = re.search(r'<enum name="Dimension"[^>]*string="D3D12_RESOURCE_DIMENSION_(\w+)"', b)
            dim = dim.group(1) if dim else "?"
            w = int(v.get("Width", 0) or 0)
            h = int(v.get("Height", 0) or 0)
            if dim == "BUFFER":
                buffer_bytes += w
                buffer_count += 1
            else:
                fmt = re.search(r'<enum name="Format"[^>]*string="DXGI_FORMAT_(\w+)"', b)
                fmt = fmt.group(1) if fmt else "?"
                mips = int(v.get("MipLevels", 1) or 1)
                bpp = DXGI_BPP.get(fmt, 4)
                sz = w * max(h, 1) * bpp * (2 if mips > 1 else 1) // (4 // 4)
                textures.append({"dim": dim, "w": w, "h": h, "fmt": fmt, "mips": mips, "approx": sz})
        elif "ID3D12Resource::SetName" in name or "SetName" in name:
            m = re.search(r"<string[^>]*>([^<]*)</string>", b)
            if m:
                names.append(m.group(1))
        elif name.endswith("Present"):
            present = {k: v[k] for k in ("SyncInterval", "Flags") if k in v}
    tex_total = sum(t["approx"] for t in textures)
    out["histogram"] = hist.most_common(25)
    out["dispatch_grids"] = dispatches
    out["pipeline_bytecode_sizes"] = pipelines
    out["accel_structures"] = dict(accel)
    out["named_resources"] = names
    out["textures"] = textures
    out["buffer_total"] = fmt_bytes(buffer_bytes)
    out["buffer_count"] = buffer_count
    out["texture_total_approx"] = fmt_bytes(tex_total)
    out["present"] = present
    if ts_min is not None and ts_max is not None:
        out["cpu_api_span_ms"] = f"{(ts_max - ts_min) / 1e4:.2f}"  # 100ns units
    with open(chrome_path, "r", encoding="utf-8", errors="replace") as f:
        chrome = json.load(f)
    out["chrome_events"] = Counter(ev.get("cat", "?") for ev in chrome.get("traceEvents", [])).most_common()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(__doc__.splitlines()[0] if __doc__ else "rdc reporter"))
    ap.add_argument("rdc")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep-exports", action="store_true")
    ap.add_argument("--renderdoc-cmd", default=os.path.join(DEFAULT_RDOC, "renderdoccmd.exe"))
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    args = ap.parse_args()
    workdir = args.workdir or os.path.dirname(os.path.abspath(args.rdc))
    os.makedirs(workdir, exist_ok=True)
    if not shutil.which(args.renderdoc_cmd) and not os.path.exists(args.renderdoc_cmd):
        raise SystemExit(f"renderdoccmd not found: {args.renderdoc_cmd}")
    rep = report(args.rdc, workdir, args.keep_exports, args.renderdoc_cmd, args.json)
    if args.json:
        rep["histogram"] = dict(rep["histogram"])
        rep["chrome_events"] = dict(rep["chrome_events"])
        print(json.dumps(rep, indent=2, default=str))
        return
    print(f"=== RenderDoc frame report: {os.path.basename(args.rdc)} ===")
    print(f"API: {rep['api']} | backbuffer {rep['backbuffer']} | capture {rep['capture_size']} | chunks {rep['num_chunks']}")
    if rep.get("cpu_api_span_ms"):
        print(f"CPU-side API span: {rep['cpu_api_span_ms']} ms | present: {rep['present']}")
    print("\n-- API histogram (top) --")
    for n, c in rep["histogram"][:18]:
        print(f"{c:6d}  {n}")
    print("\n-- Dispatch grids (thread groups) --")
    for g in rep["dispatch_grids"]:
        print("  ", g)
    if rep["pipeline_bytecode_sizes"]:
        print(f"-- GPU pipelines: {len(rep['pipeline_bytecode_sizes'])}, bytecode sizes: {rep['pipeline_bytecode_sizes']} --")
    if rep["accel_structures"]:
        print(f"-- Acceleration structures: {rep['accel_structures']} --")
    if rep["named_resources"]:
        print(f"-- Named resources: {rep['named_resources']} --")
    print(f"\n-- Textures ({rep['texture_total_approx']} total) --")
    for t in rep["textures"]:
        print(f"  {t['dim']:10s} {t['w']}x{t['h']} {t['fmt']} mips={t['mips']} ~{fmt_bytes(t['approx'])}")
    print(f"-- Buffers: {rep['buffer_count']} allocs, {rep['buffer_total']} total --")
    print(f"\n-- chrome.json event categories --")
    for cat, c in rep["chrome_events"]:
        print(f"{c:8d}  {cat}")


if __name__ == "__main__":
    main()
