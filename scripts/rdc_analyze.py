#!/usr/bin/env python
# analyze.py — batch RenderDoc (.rdc) capture deserializer & search tool.
#
# Two modes:
#
#   DESERIALIZE (default)
#       python analyze.py 1.rdc 2.rdc 3.rdc [-o outdir]
#     Each capture is fully deserialized into outdir/<stem>/ containing:
#       thumbnail.png        embedded frame thumbnail
#       capture.zip.xml      complete XML chunk stream (the "all information" dump)
#       capture.chrome.json  chrome://tracing JSON API trace
#       chunks.jsonl         every chunk: name, timestamp, duration, parsed fields
#       report.json          structured summary (header, histogram, dispatches,
#                            pipelines, accel structures, textures, buffers,
#                            named resources, present, CPU API span)
#       report.txt           human-readable rendering of report.json
#
#   SEARCH
#       python analyze.py 1.rdc --search "vkCmdDispatch"
#       python analyze.py --search "R11G11B10" 1.rdc 2.rdc 3.rdc
#     Case-insensitive substring search across chunk names, resource names and
#     all parsed field values. Prints a compact hit list plus context fields.
#
# Requirements: RenderDoc installed (renderdoccmd.exe), Python 3.8+ stdlib only.
# The zip.xml export can be ~10x the capture size; it is kept in the output
# directory (that is the point of deserialization), so mind disk space.
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

DXGI_BPP = {
    "R8_UNORM": 1, "R8G8B8A8_UNORM": 4, "R8G8B8A8_UNORM_SRGB": 4, "B8G8R8A8_UNORM": 4,
    "B8G8R8A8_UNORM_SRGB": 4, "R16G16B16A16_FLOAT": 8, "R16G16B16A16_UNORM": 8,
    "R32_SINT": 4, "R32_UINT": 4, "R32G32_SFLOAT": 8, "R32G32_FLOAT": 8,
    "R32G32B32A32_FLOAT": 16, "D24_UNORM_S8_UINT": 4, "D32_FLOAT": 4, "D32_FLOAT_S8X24_UINT": 8,
    "R16_FLOAT": 2, "R11G11B10_FLOAT": 4, "R8G8B8A8_TYPELESS": 4, "R8_UINT": 1, "R8_SINT": 1,
    "R16_UNORM": 2, "R16_UINT": 2, "R16_SINT": 2, "R32G32B32_FLOAT": 12, "R32G32B32_UINT": 12,
    "R32G32B32_SINT": 12, "R10G10B10A2_UNORM": 4, "BC1_UNORM": 0.5, "BC3_UNORM": 1,
}


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip()


def fmt_bytes(n):
    n = int(n)
    return f"{n / 1048576:.1f} MiB" if n >= 1048576 else (f"{n / 1024:.1f} KiB" if n >= 1024 else f"{n} B")


# ---------------------------------------------------------------- XML parsing
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


def chunk_string(body):
    m = re.search(r"<string[^>]*>([^<]*)</string>", body)
    return m.group(1) if m else None


# ------------------------------------------------------------- deserialize
def export_capture(rdc, outdir, rdoc_cmd, force=False):
    """Export thumbnail / zip.xml / chrome.json into outdir (skip if present)."""
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(rdc))[0]
    paths = {
        "thumb": os.path.join(outdir, "thumbnail.png"),
        "xml": os.path.join(outdir, "capture.zip.xml"),
        "chrome": os.path.join(outdir, "capture.chrome.json"),
    }
    if force or not os.path.exists(paths["thumb"]):
        run([rdoc_cmd, "thumb", "-o", paths["thumb"], rdc])
    if force or not os.path.exists(paths["xml"]):
        run([rdoc_cmd, "convert", "-f", rdc, "-o", paths["xml"], "-c", "zip.xml"])
    if force or not os.path.exists(paths["chrome"]):
        run([rdoc_cmd, "convert", "-f", rdc, "-o", paths["chrome"], "-c", "chrome.json"])
    return paths


def build_report(rdc, xml_text, chrome):
    """Parse the zip.xml chunk stream into a structured, detailed report."""
    hdr = re.search(r"<header>.*?</header>", xml_text, re.S)
    hdr_txt = hdr.group(0) if hdr else ""
    driver = re.search(r"<driver[^>]*>([^<]+)</driver>", hdr_txt)
    thumb_m = re.search(r'<thumbnail width="(\d+)" height="(\d+)"', hdr_txt)

    chunks = split_chunks(xml_text)
    parsed = []
    for b in chunks:
        a = chunk_attrs(b)
        parsed.append({
            "name": a.get("name", "?"),
            "id": int(a["id"]) if a.get("id", "").isdigit() else None,
            "timestamp": int(a["timestamp"]) if a.get("timestamp", "").isdigit() else None,
            "duration": int(a["duration"]) if a.get("duration", "").isdigit() else None,
            "fields": chunk_fields(b),
            "string": chunk_string(b),
        })

    out = {
        "capture": os.path.basename(rdc),
        "api": driver.group(1) if driver else "?",
        "backbuffer": f"{thumb_m.group(1)}x{thumb_m.group(2)}" if thumb_m else "?",
        "capture_size": fmt_bytes(os.path.getsize(rdc)),
        "num_chunks": len(chunks),
    }

    dispatches, textures, names = [], [], []
    hist = Counter()
    buffer_bytes = buffer_count = 0
    present = None
    ts_min = ts_max = None

    for p in parsed:
        name, v, b = p["name"], p["fields"], None
        hist[name] += 1
        if p["timestamp"] is not None:
            ts_min = p["timestamp"] if ts_min is None else min(ts_min, p["timestamp"])
            ts_max = max(p["timestamp"] + (p["duration"] or 0), ts_max or 0)

        if name.endswith("::Dispatch") or name == "vkCmdDispatch" or ("Dispatch" in name and "Create" not in name):
            g = (v.get("ThreadGroupCountX") or v.get("x"), v.get("ThreadGroupCountY") or v.get("y"),
                 v.get("ThreadGroupCountZ") or v.get("z"))
            dispatches.append({"event": name, "grid": g,
                               "timestamp": p["timestamp"], "duration_us": (p["duration"] or 0) / 10.0})
        elif name.endswith(("CreatePlacedResource", "CreateCommittedResource")):
            dim_m = re.search(r'D3D12_RESOURCE_DIMENSION_(\w+)', json.dumps(v))
            dim = dim_m.group(1) if dim_m else "?"
            w = int(v.get("Width", 0) or 0)
            h = int(v.get("Height", 0) or 0)
            if dim == "BUFFER":
                buffer_bytes += w
                buffer_count += 1
            else:
                fmt_m = re.search(r'DXGI_FORMAT_(\w+)', json.dumps(v))
                fmt = fmt_m.group(1) if fmt_m else "?"
                mips = int(v.get("MipLevels", 1) or 1)
                bpp = DXGI_BPP.get(fmt, 4)
                approx = int(w * max(h, 1) * bpp * (1.33 if mips > 1 else 1.0))
                textures.append({"dim": dim, "w": w, "h": h, "fmt": fmt, "mips": mips,
                                 "approx_bytes": approx, "name": p["string"]})
        elif "SetName" in name:
            if p["string"]:
                names.append(p["string"])
        elif name.endswith("Present"):
            present = {k: v[k] for k in ("SyncInterval", "Flags") if k in v}

    # second pass over raw chunk bodies for things best found with regex
    raw_pipelines, raw_accel = [], Counter()
    for b in split_chunks(xml_text):
        a = chunk_attrs(b)
        name = a.get("name", "?")
        if "CreatePipeline" in name or name == "vkCreatePipelineCache":
            m = re.search(r'byteLength="(\d+)"', b)
            if m:
                raw_pipelines.append({"event": name, "bytecode_bytes": int(m.group(1))})
        elif "Acceleration Structure Create" in name:
            m = re.search(r'TYPE_(BOTTOM_LEVEL|TOP_LEVEL)', b)
            raw_accel[m.group(1) if m else "?"] += 1

    tex_total = sum(t["approx_bytes"] for t in textures)
    out.update({
        "histogram": hist.most_common(),
        "dispatches": dispatches,
        "pipelines": raw_pipelines,
        "accel_structures": dict(raw_accel),
        "named_resources": names,
        "textures": textures,
        "texture_total_approx": fmt_bytes(tex_total),
        "buffer_count": buffer_count,
        "buffer_total": fmt_bytes(buffer_bytes),
        "present": present,
    })
    if ts_min is not None and ts_max is not None:
        out["cpu_api_span_ms"] = round((ts_max - ts_min) / 1e4, 3)  # 100ns units
    if chrome:
        out["chrome_event_categories"] = Counter(
            ev.get("cat", "?") for ev in chrome.get("traceEvents", [])).most_common()
    return out, parsed


def write_outputs(outdir, report, parsed):
    with open(os.path.join(outdir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for p in parsed:
            f.write(json.dumps(p, default=str) + "\n")
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(render_text(report))


def render_text(rep):
    L = [f"=== RenderDoc frame report: {rep['capture']} ===",
         f"API: {rep['api']} | backbuffer {rep['backbuffer']} | capture {rep['capture_size']} | chunks {rep['num_chunks']}"]
    if rep.get("cpu_api_span_ms"):
        L.append(f"CPU-side API span: {rep['cpu_api_span_ms']} ms | present: {rep['present']}")
    L.append("\n-- API histogram --")
    for n, c in rep["histogram"][:40]:
        L.append(f"{c:6d}  {n}")
    L.append("\n-- Dispatch grids (thread groups) --")
    for d in rep["dispatches"]:
        L.append(f"  {d['event']}: {d['grid']}  ts={d['timestamp']} dur={d['duration_us']:.1f}us")
    if rep["pipelines"]:
        L.append(f"\n-- GPU pipelines ({len(rep['pipelines'])}) --")
        for p in rep["pipelines"]:
            L.append(f"  {p['event']}: {fmt_bytes(p['bytecode_bytes'])}")
    if rep["accel_structures"]:
        L.append(f"\n-- Acceleration structures: {rep['accel_structures']} --")
    if rep["named_resources"]:
        L.append(f"\n-- Named resources ({len(rep['named_resources'])}) --")
        for n in rep["named_resources"]:
            L.append(f"  {n}")
    L.append(f"\n-- Textures ({rep['texture_total_approx']} approx total, {len(rep['textures'])}) --")
    for t in rep["textures"]:
        L.append(f"  {t['dim']:10s} {t['w']}x{t['h']} {t['fmt']} mips={t['mips']} ~{fmt_bytes(t['approx_bytes'])}"
                 + (f"  [{t['name']}]" if t["name"] else ""))
    L.append(f"\n-- Buffers: {rep['buffer_count']} allocs, {rep['buffer_total']} total --")
    if rep.get("chrome_event_categories"):
        L.append("\n-- chrome.json event categories --")
        for cat, c in rep["chrome_event_categories"]:
            L.append(f"{c:8d}  {cat}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- search
def search_captures(rdc, pattern, rdoc_cmd, workdir=None):
    """Case-insensitive substring search across all parsed chunk data."""
    # reuse a prior full deserialize if present, else export to a temp dir
    stem = os.path.splitext(os.path.basename(rdc))[0]
    cached = os.path.join(os.path.dirname(os.path.abspath(rdc)), "analyzed", stem, "capture.zip.xml")
    if os.path.exists(cached):
        paths = {"xml": cached}
    else:
        workdir = workdir or os.path.join(os.path.dirname(os.path.abspath(rdc)), ".analyze_tmp")
        os.makedirs(workdir, exist_ok=True)
        paths = export_capture(rdc, workdir, rdoc_cmd)
    with open(paths["xml"], "r", encoding="utf-8", errors="replace") as f:
        xml = f.read()
    chunks = split_chunks(xml)
    pat = pattern.lower()
    hits = []
    for b in chunks:
        if pat in b.lower():
            a = chunk_attrs(b)
            v = chunk_fields(b)
            hits.append({"name": a.get("name", "?"),
                         "id": a.get("id"), "timestamp": a.get("timestamp"),
                         "duration_us": (int(a["duration"]) / 10.0) if a.get("duration", "").isdigit() else None,
                         "fields": {k: val for k, val in v.items() if pat in str(val).lower()
                                    or pat in k.lower()} or dict(list(v.items())[:6]),
                         "string": chunk_string(b)})
    print(f"== {os.path.basename(rdc)}: {len(hits)}/{len(chunks)} chunks match '{pattern}' ==")
    for h in hits[:int(os.environ.get('SEARCH_MAX', '200'))]:
        ts = f" ts={h['timestamp']}" if h["timestamp"] else ""
        dur = f" dur={h['duration_us']:.1f}us" if h["duration_us"] is not None else ""
        print(f"  [{h['id']}] {h['name']}{ts}{dur}")
        shown = 0
        for k, val in h["fields"].items():
            if shown >= 8:
                print("      ...")
                break
            print(f"      {k} = {val}")
            shown += 1
        if h["string"] and pat in h["string"].lower():
            print(f"      string = {h['string']}")
    return len(hits)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "rdc analyzer")
    ap.add_argument("rdc", nargs="+", help="capture file(s), e.g. 1.rdc 2.rdc 3.rdc")
    ap.add_argument("-o", "--outdir", default="analyzed", help="output root dir (default: ./analyzed)")
    ap.add_argument("--search", metavar="PATTERN", help="search captures for PATTERN instead of full deserialize")
    ap.add_argument("--force", action="store_true", help="re-export even if outputs exist")
    ap.add_argument("--renderdoc-cmd", default=os.path.join(DEFAULT_RDOC, "renderdoccmd.exe"))
    args = ap.parse_args()

    if not shutil.which(args.renderdoc_cmd) and not os.path.exists(args.renderdoc_cmd):
        raise SystemExit(f"renderdoccmd not found: {args.renderdoc_cmd}")

    if args.search:
        for rdc in args.rdc:
            search_captures(rdc, args.search, args.renderdoc_cmd)
        return

    for rdc in args.rdc:
        stem = os.path.splitext(os.path.basename(rdc))[0]
        outdir = os.path.join(args.outdir, stem)
        print(f"[{stem}] exporting ...")
        paths = export_capture(rdc, outdir, args.renderdoc_cmd, force=args.force)
        print(f"[{stem}] parsing ...")
        with open(paths["xml"], "r", encoding="utf-8", errors="replace") as f:
            xml = f.read()
        chrome = None
        if os.path.exists(paths["chrome"]):
            with open(paths["chrome"], "r", encoding="utf-8", errors="replace") as f:
                try:
                    chrome = json.load(f)
                except json.JSONDecodeError:
                    pass
        report, parsed = build_report(rdc, xml, chrome)
        write_outputs(outdir, report, parsed)
        print(f"[{stem}] -> {outdir}")
        print(render_text(report).splitlines()[0])
        print(f"      {report['api']} | {report['backbuffer']} | {report['num_chunks']} chunks | "
              f"{report.get('cpu_api_span_ms', '?')} ms CPU span | report.json + chunks.jsonl written")


if __name__ == "__main__":
    main()
