---
name: renderdoc
description: RenderDoc GPU frame-capture analysis — headless .rdc capture, batch deserialization, frame reports and full-text search scripts, plus the manual Event Browser / Pipeline State / shader-debug workflow. Use when an image is wrong (missing, black, NaN, wrong color, flickering), when pipeline state or resource contents need ground truth, or when a frame looks slow.
---

# RenderDoc Frame Analysis

RenderDoc = single-frame **API capture + GPU replay + inspection** (Vulkan 1.4, D3D11/D3D12, OpenGL 3.2+, GLES 2.0–3.2). It answers "which event produced this pixel/vertex, what did the shader actually read, is this state correct, what does this buffer contain, why is this pass expensive".

It is **not** an API validator (run validation layers at runtime for ground truth on invalid use) and replay is not a perfect reproduction: timing-sensitive bugs can vanish, extra memory is allocated, and Vulkan/D3D12 captures often do not replay across different vendors (D3D11/GL captures are portable).

## Repo toolchain

| File | Role |
|---|---|
| `scripts/rdc_analyze.md` | Manual (UI) methodology: capture hygiene, Event Browser bisection, Texture Viewer/overlays/pixel history, Pipeline State checklist, Mesh Viewer, shader debugging, defect patterns, performance signatures, root-cause table. **Read sections on demand — do not load all 746 lines.** |
| `scripts/renderdoc_capture.sh` | Headless capture of a windowed target: `renderdoccmd capture -w` + window wait + synthetic F12 + poll for the `.rdc`. |
| `scripts/rdc_report.py` | One capture → compact frame report (text or `--json`). Fast triage in CI/agent runs. |
| `scripts/rdc_analyze.py` | Many captures → full deserialization into greppable artifacts, plus substring `--search`. |

All three scripts need RenderDoc installed (`renderdoccmd.exe`), Python 3.8+ stdlib only, and honor `RENDERDOC_DIR` / `--renderdoc-cmd`.

## 1. Capture a frame

```bash
bash scripts/renderdoc_capture.sh --exe build/bin/example_path_tracing.exe --args dx \
     --out build/renderdoc_captures            # last stdout line = the .rdc path
```

- The target **must present a frame** (GUI mode). `--offline`, `--compare/-c` and `--out_ref` all force headless mode → no window → no capture (`examples/common/reference_compare.h`). Compute-only binaries likewise produce nothing; for those, embed the RenderDoc app API (`StartFrameCapture`/`EndFrameCapture`/`TriggerCapture` from `renderdoc_app.h`).
- `--args "dx --spp 16"` passes program args; `--timeout`, `--workdir`, `--keep-running` tune the run; logs go to stderr, `capture.log` lands in `--out`.
- Prefer the **`dx` backend for captures**: it forwards `Resource::set_name(...)` (`include/luisa/runtime/rhi/resource.h`) to `ID3D12Resource::SetName`, so named buffers/textures/PSOs appear in the UI and in `named_resources`. The Vulkan backend's `Device::set_name` is currently a no-op (`src/backends/vk/device.cpp`), so those captures show unnamed `ResourceId`s.
- Name your own resources before capturing (`resource.set_name("SceneColor")`); kernel names come free (`kernel_16hex`), and `LUISA_CORO_SHADER_MAP=1` names coro scheduler shaders. `LUISA_DUMP_SOURCE=1` dumps generated source so disassembly can be mapped back to it.
- Enable API validation in capture options when you suspect invalid use (D3D12 SDK layers ship in `build/bin`), then read RenderDoc's Debug Messages.

## 2. Headless triage with the scripts

```bash
# single capture, human summary (or --json for machine-readable)
python scripts/rdc_report.py build/renderdoc_captures/pt_frame1071.rdc [--json] [--workdir DIR]

# batch: full deserialize into ./analyzed/<stem>/
python scripts/rdc_analyze.py 1.rdc 2.rdc 3.rdc -o analyzed
# re-export even if artifacts exist:
python scripts/rdc_analyze.py 1.rdc -o analyzed --force

# substring search over every chunk name / resource name / field value
python scripts/rdc_analyze.py captures/frame.rdc --search "R32G32B10A2"      # formats
python scripts/rdc_analyze.py captures/frame.rdc --search "SceneColor"       # resource names
python scripts/rdc_analyze.py captures/*.rdc --search "vkCmdDispatch"        # across captures
```

Use `rdc_report.py` for "what is in this frame / is it even the frame I think" and `rdc_analyze.py` for anything you must grep (formats, resource ids, descriptors, dispatch grids, shader names). Both parse the `zip.xml` chunk stream produced by `renderdoccmd convert`.

Report contents: API/driver, backbuffer size, capture size, chunk count, CPU-side API span, present flags, per-API-call histogram, compute dispatch grids (+duration), pipeline bytecode sizes, BLAS/TLAS build counts, named resources, texture inventory with approximate VRAM, buffer alloc count/total, chrome.json event categories.

`analyzed/<stem>/` artifacts:

| File | Contents |
|---|---|
| `thumbnail.png` | embedded frame thumbnail (look at it first) |
| `capture.zip.xml` | complete XML chunk stream (~10× capture size: 41 MiB `.rdc` → 305 MiB) |
| `capture.chrome.json` | chrome://tracing API trace — use for timeline/GPU-gap reading |
| `chunks.jsonl` | one JSON per chunk: `name`, `id`, `timestamp`, `duration`, `fields`, `string` |
| `report.json` / `report.txt` | structured summary and its text rendering |

Query `chunks.jsonl` instead of re-parsing the 300 MB XML:

```python
import json
rows = [json.loads(l) for l in open("analyzed/pt_frame1071/chunks.jsonl")]
for r in rows:
    if "Dispatch" in r["name"]:
        f = r["fields"]
        print(r["id"], r["name"], (f.get("ThreadGroupCountX") or f.get("x"),
              f.get("ThreadGroupCountY") or f.get("y"), f.get("ThreadGroupCountZ") or f.get("z")),
              f"{(r['duration'] or 0)/10:.1f}us")
```

Search hits print `[chunk id] name ts=… dur=…us` plus up to 8 matching/leading fields; cap the printed hits with `SEARCH_MAX` (default 200). Chunk `id`/`timestamp` are capture-stream ordering keys — map them to UI EIDs by name/position, they are not the same numbers.

## 3. Manual investigation workflow (`scripts/rdc_analyze.md`)

Section map: §1 capabilities/limits · §2 capture setup (validation, debuggable shaders, naming, minimal repro) · §3 the windows as one system · §4 Event Browser + bisection · §5 Texture Viewer, HDR range, overlays, pixel history · §6 Pipeline State checklist · §7 Mesh Viewer · §8 pixel/vertex/compute debugging · §9 constant-buffer/texture/buffer/usage inspection · §10 defect patterns · §11 performance · §12 custom visualization + hot shader replacement · §13 Python API automation · §14 end-to-end sequence · §15 root-cause table · §16 tips · §17 stuck checklist.

Order that works (never start with shader debugging):

1. **Health check** — API, GPU, driver, does the bug reproduce in the capture? Debug Messages first.
2. **Frame structure** — group events into passes (shadow → gbuffer → lighting → transparent → post → UI); count draws; hunt unexpected repeats/fullscreen passes.
3. **Bisect to the first bad event** — open the final output, jump to mid-frame, walk earlier/later until the introducing EID is found; bookmark first-bad and last-good.
4. **Follow the resource chain** — is the bad value already in an *input* of that event? Trace the producer with Resource Inspector / usage (ascending EID order) until the first writer of bad data.
5. **Validate the pipeline contract** — targets & formats & mip/slice, vertex input (index buffer, strides, layout), shader + entry point + permutation, constants (transpose, stale frame CB, wrong index), textures/samplers (sRGB vs linear, wrong mip), rasterizer (cull/winding/scissor/viewport), depth-stencil, blend/write mask, synchronization. Note Pipeline State shows the state **after** the selected action — compare adjacent EIDs for before/after.
6. **Route by data type** — geometry → Mesh Viewer (VS Input right/wrong splits the bug before/after the vertex shader) · pixel/color → Texture Viewer + Pixel History (green=passed, red=failed test, gray=unknown write) · state/resource → Pipeline State + Resource Inspector · validity → Debug Messages · compute output → UAV/buffer raw inspection (custom HLSL/GLSL-lite formats) · performance → per-action timings + Performance Counter Viewer.
7. **Hypothesis → test** — compare against a good sibling draw, toggle postprocess, hot-replace a shader with a constant color, or use NaN/negative/INF + depth/stencil pass-fail + wireframe/quad-overdraw/triangle-size overlays.

Overlays worth turning on immediately for a LuisaCompute renderer: **NaN/negative/INF** (bloom/volumetrics/light-accum poisoning → black screen), **depth pass/fail** and **stencil pass/fail** (missing object), **wireframe** (did the triangles exist at all), **quad overdraw / triangle size** (performance).

## 4. Symptom → first place to look

| Symptom | Look at | Typical root cause |
|---|---|---|
| Object missing | Event Browser, Mesh Viewer, Pixel History | culled / zero count / bad index offset / wrong RT slice / scissor zero / depth always fails / winding-cull mismatch |
| Wrong position or scale | VS constants | matrix transpose or row/column-major, wrong per-object CB slot, stale frame constants, instance stride |
| Exploded / distorted mesh | Mesh Viewer VS input↔output | stride-format mismatch, bad skinning weights/bones, NaNs, meshlet payload packing |
| Wrong color/lighting | G-buffer + PS constants + Texture Viewer | wrong texture/mip, sRGB-linear mismatch, tangent basis, light index, GGX denominator |
| Black after postprocess | NaN/INF overlay on intermediates | divide by zero, unnormalized zero vector, log/pow of negative, unclamped HDR, temporal history |
| Flicker / intermittent | Resource usage + validation messages | missing barrier, uninitialized data, stale descriptor/dynamic index, previous-frame hazard |
| Indirect draw wrong | argument buffer inspection | count not reset, alignment/stride, ExecuteIndirect layout, state between compute and draw |
| Low FPS / expensive pass | per-action timings + counters | thousands of tiny draws, overdraw, tiny triangles, bandwidth, barrier storms, dispatch far larger than data |

Counter units differ (seconds, %, ratio, bytes, cycles, …) — never compare raw values across units; treat counters as hypotheses ranked by per-action GPU duration.

## 5. Script gotchas (verified)

- Exports dominate disk: `zip.xml` ≈ 10× the `.rdc`; delete `analyzed/` and `*.zip.xml` after the session. Export+parse of a 41 MiB, 165-chunk D3D12 capture took ~7 s.
- `rdc_report.py --keep-exports` is accepted but ignored — the exports always stay in `--workdir` (default: next to the `.rdc`).
- `rdc_report.py` texture/buffer inventory is **D3D12-specific** (`CreatePlacedResource`/`CreateCommittedResource` + `DXGI_FORMAT_*`); Vulkan/GL captures still report histogram, dispatches, pipelines, present, but an empty texture/buffer table.
- The reported `cpu_api_span_ms` spans the whole capture including initialization (a path-tracing capture reported 35 s) — it is not frame time. Read `capture.chrome.json` or per-chunk durations instead.
- `rdc_analyze.py -o` is relative to the **current** directory, but `--search` only reuses a cache at `<dir-of-.rdc>/analyzed/<stem>/capture.zip.xml`; otherwise it re-exports into `<dir-of-.rdc>/.analyze_tmp`. Deserialize with `-o <rdc dir>/analyzed` to make searches instant.
- Field extraction flattens nested structs and **first occurrence wins**, so a value printed for e.g. `Format` may come from an outer field, not the one you expected — confirm against `capture.zip.xml` or the UI before drawing a conclusion.
- Both scripts raise `SystemExit` on the first failing `renderdoccmd` call; if `renderdoccmd` is missing set `RENDERDOC_DIR` (e.g. `D:/RenderDoc`) instead of editing paths.

## 6. When stuck, answer in order

exact symptom → first RT showing it → first EID introducing it → is bad data already in an input → right shader bound → constants correct at that EID → expected resources bound → formats/strides/offsets/mips/slices/sRGB → viewport/scissor/cull/depth/stencil/blend/RT → pixel history pass-fail → VS input vs output → validation messages → does it vanish with postprocess disabled → draw/resource/state/synchronization/performance defect?
