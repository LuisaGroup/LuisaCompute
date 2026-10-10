#!/usr/bin/env python3
"""Build and pack example_native_shader into a standalone, xmake-free directory.

Steps:
  1. `xmake config -m release` + `xmake build example_native_shader`
  2. Resolve the exe's real DLL dependency graph (PE import tables + modules
     loaded dynamically at runtime, e.g. luisa-backend-*, dxcompiler,
     dstorage) and copy only the reachable DLLs from bin/release into
     build/native_shader/
  3. Copy examples/compute/native_shader_examples/README.md, rewriting the
     xmake-specific parts so the binary can be run directly
  4. Copy two self-contained samples (dispatch JSON + shader sources)

Usage:
  python scripts/pack_native_shader.py [--skip-build] [--no-verify]
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "compute" / "native_shader_examples"
BIN_DIR = PROJECT_ROOT / "bin" / "release"
OUT_DIR = PROJECT_ROOT / "build" / "native_shader"
TARGET_NAME = "example_native_shader"

IS_WINDOWS = sys.platform == "win32"
EXE_NAME = f"{TARGET_NAME}.exe" if IS_WINDOWS else TARGET_NAME


# --- DLL dependency resolution ------------------------------------------------
#
# Two mechanisms bring in DLLs:
#   * link-time imports — walked from the PE import table;
#   * runtime loads (LoadLibrary/dlopen) — the backend plugins are resolved as
#     "luisa-backend-" + the backend name from the command line (see
#     src/runtime/context.cpp), the DXC shader compiler loads "dxcompiler" /
#     "dxil" (src/backends/common/hlsl/shader_compiler.cpp) and the dx backend
#     loads "dstorage" / "dstoragecore" (src/backends/dx/DXApi/ext.cpp).
# The runtime names are stored as bare stems in the binaries ("dxcompiler",
# with ".dll" appended by DynamicModule::load), so we scan module bytes for the
# stem with identifier boundaries instead of the full file name.


def pe_imported_dlls(data: bytes) -> list[str]:
    """Return the DLL names of a PE file's import table (lowercased)."""
    peoff = struct.unpack_from("<I", data, 0x3C)[0]
    if data[peoff:peoff + 4] != b"PE\0\0":
        raise ValueError("not a PE file")
    nsections = struct.unpack_from("<H", data, peoff + 6)[0]
    optsize = struct.unpack_from("<H", data, peoff + 20)[0]
    opt = peoff + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd_base = opt + (112 if magic == 0x20B else 96)  # PE32+ vs PE32
    import_rva = struct.unpack_from("<I", data, dd_base + 8)[0]
    if not import_rva:
        return []
    sections = []
    sec_base = opt + optsize
    for i in range(nsections):
        b = sec_base + 40 * i
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, b + 8)
        sections.append((vaddr, max(vsize, rawsize), rawptr))

    def rva_to_offset(rva: int):
        for vaddr, size, rawptr in sections:
            if vaddr <= rva < vaddr + size:
                return rawptr + (rva - vaddr)
        return None

    dlls = []
    off = rva_to_offset(import_rva)
    while off is not None:
        oft, _ts, _fc, name_rva, _ft = struct.unpack_from("<IIIII", data, off)
        if oft == 0 and name_rva == 0:
            break
        noff = rva_to_offset(name_rva)
        if noff is None:
            break
        dlls.append(data[noff:data.index(b"\0", noff)].decode(errors="replace").lower())
        off += 20
    return dlls


def _is_ident_byte(c: bytes) -> bool:
    return c.isalnum() or c in b"_."


def stem_referenced(data: bytes, stem: str) -> bool:
    """True if the ASCII stem appears with non-identifier boundaries."""
    s = stem.encode()
    start = 0
    while True:
        i = data.find(s, start)
        if i < 0:
            return False
        before = data[i - 1:i]
        after = data[i + len(s):i + len(s) + 1]
        if (not before or not _is_ident_byte(before)) and \
           (not after or not _is_ident_byte(after)):
            return True
        start = i + 1


def resolve_dependencies(exe: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Walk the dependency graph of `exe` over the DLLs in BIN_DIR.

    Returns (link_deps, runtime_deps): module name -> the module that pulled
    it in, for link-time imports and runtime-loaded modules respectively.
    """
    candidates = {p.name.lower(): p for p in BIN_DIR.glob("*.dll")}
    stems = {name: name[:-4] for name in candidates}  # strip ".dll"
    cache: dict[str, bytes] = {}
    link_deps: dict[str, str] = {}
    runtime_deps: dict[str, str] = {}

    def read(name: str) -> bytes:
        if name not in cache:
            cache[name] = (BIN_DIR / name).read_bytes() \
                if name == exe.name.lower() else candidates[name].read_bytes()
        return cache[name]

    def add(name: str, reason: str, table: dict[str, str]) -> None:
        if name in link_deps or name in runtime_deps or name == exe.name.lower():
            return
        table[name] = reason

    # The exe itself plus every built backend plugin: the backend is chosen
    # at runtime from the command line, so all of them must ship.
    pending = [exe.name.lower()]
    for backend in sorted(BIN_DIR.glob("luisa-backend-*.dll")):
        add(backend.name.lower(), "backend plugin (CLI-selected)", runtime_deps)
    pending.extend(runtime_deps)

    while pending:
        mod = pending.pop()
        data = read(mod)
        for dep in pe_imported_dlls(data):
            if dep in candidates and dep not in link_deps and dep not in runtime_deps:
                add(dep, mod, link_deps)
                pending.append(dep)
        for cand, stem in stems.items():
            if cand not in link_deps and cand not in runtime_deps and \
                    stem_referenced(data, stem):
                add(cand, mod, runtime_deps)
                pending.append(cand)
    return link_deps, runtime_deps


def pack_binary() -> Path:
    exe_src = BIN_DIR / EXE_NAME
    if not exe_src.is_file():
        raise FileNotFoundError(
            f"{exe_src} not found — did the release build succeed?")
    out_exe = OUT_DIR / EXE_NAME
    shutil.copy2(exe_src, out_exe)

    link_deps, runtime_deps = resolve_dependencies(exe_src)
    reasons = {**{k: f"imported by {v}" for k, v in link_deps.items()},
               **{k: f"loaded at runtime by {v}" for k, v in runtime_deps.items()}}
    for name in sorted(reasons):
        log(f"  {name} ({reasons[name]})")
        shutil.copy2(BIN_DIR / name, OUT_DIR / name)
    log(f"packed {out_exe.name} + {len(reasons)} DLL(s) into {OUT_DIR}")
    return out_exe


# --- README rewrite: exact replacements of the xmake-specific paragraphs -----

README_REPLACEMENTS = [
    # 1. The "how xmake run works" note after the CLI synopsis.
    (
        "`xmake run <target> ...` runs the binary with the *target's* output "
        "directory as\nthe working directory, so a relative document path must "
        "be given relative to\n`bin/<mode>/` (or use an absolute path). "
        "`--self-test` locates its corpus by\nsearching upwards from the "
        "working directory, so it works from anywhere.",
        "Run `example_native_shader.exe` (Windows) or `./example_native_shader` "
        "(Linux)\ndirectly from this directory — no build system or xmake "
        "environment is needed.\nA relative document path resolves against the "
        "current working directory (or use\nan absolute path). `--self-test` "
        "locates its corpus by searching upwards from\nthe working directory, "
        "so it works from anywhere.",
    ),
    # 2. The output-dir paragraph tied to bin/<mode>/.
    (
        "Because `xmake run` sets the working directory to `bin/<mode>/`, the "
        "default\n`output_dir` of `\"native_shader_output\"` produces artifacts "
        "under\n`bin/<mode>/native_shader_output/`. A checkout that ran the "
        "example therefore\nlooks like:",
        "Relative output paths resolve against the current working directory, "
        "so the\ndefault `output_dir` of `\"native_shader_output\"` produces "
        "artifacts under\n`./native_shader_output/`. A directory that ran the "
        "example therefore looks\nlike:",
    ),
    # 3. The "Run it" instruction for simple_add.json.
    (
        "`dispatch` is in threads, exactly one per element, so no bounds check "
        "is\nneeded. Run it (from the repository root, or pass an absolute "
        "document\npath — `xmake run` makes `bin/<mode>/` the working "
        "directory, see the note\nabove):\n\n"
        "```\n"
        "bin/release/example_native_shader.exe cuda examples/compute/"
        "native_shader_examples/simple_add.json\n"
        "```",
        "`dispatch` is in threads, exactly one per element, so no bounds check "
        "is\nneeded. Run it from this directory (the self-contained sample "
        "ships in\n`examples/simple_add/`):\n\n"
        "```\n"
        "example_native_shader.exe cuda examples/simple_add/simple_add.json\n"
        "```",
    ),
]

# --- Samples packed into <out>/examples --------------------------------------

# (source relative to EXAMPLE_DIR, destination relative to OUT_DIR)
SAMPLE_FILES = [
    # Self-contained document: the shader code is inline in the JSON.
    ("simple_add.json", "examples/simple_add/simple_add.json"),
    # Document that loads shader sources from files (keeps the
    # document-relative shaders/ layout its include_dirs entry expects).
    ("scale_offline.json", "examples/scale_offline/scale_offline.json"),
    ("shaders/scale.hlsl", "examples/scale_offline/shaders/scale.hlsl"),
    ("shaders/scale.glsl", "examples/scale_offline/shaders/scale.glsl"),
    ("shaders/scale.cuda", "examples/scale_offline/shaders/scale.cuda"),
    ("shaders/native_shader_math.h",
     "examples/scale_offline/shaders/native_shader_math.h"),
]


def log(msg: str) -> None:
    print(f"[pack_native_shader] {msg}")


def run(cmd: list[str], cwd: Path) -> None:
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build() -> None:
    run(["xmake", "config", "-m", "release", "-y"], cwd=PROJECT_ROOT)
    run(["xmake", "build", TARGET_NAME], cwd=PROJECT_ROOT)


def rewrite_readme() -> None:
    src = EXAMPLE_DIR / "README.md"
    text = src.read_text(encoding="utf-8")
    for old, new in README_REPLACEMENTS:
        if old not in text:
            log(f"warning: README paragraph not found, left unchanged:\n  {old[:80]}...")
            continue
        text = text.replace(old, new, 1)
    dst = OUT_DIR / "README.md"
    dst.write_text(text, encoding="utf-8")
    log(f"wrote {dst}")


def pack_samples() -> None:
    for rel_src, rel_dst in SAMPLE_FILES:
        src = EXAMPLE_DIR / rel_src
        if not src.is_file():
            log(f"warning: sample file missing, skipped: {src}")
            continue
        dst = OUT_DIR / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    log(f"packed {len(SAMPLE_FILES)} sample file(s) into {OUT_DIR / 'examples'}")


def verify(out_exe: Path) -> None:
    # --help exercises exe + runtime DLL loading without needing a GPU.
    log("verifying packed binary: " + out_exe.name + " --help")
    result = subprocess.run([str(out_exe), "--help"], cwd=OUT_DIR,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"packed binary failed --help (exit {result.returncode})")
    log("packed binary runs without the xmake environment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse the existing bin/release binaries")
    parser.add_argument("--no-verify", action="store_true",
                        help="do not run the packed binary's --help smoke test")
    args = parser.parse_args()

    if not IS_WINDOWS:
        raise SystemExit("dependency analysis currently supports PE (Windows) "
                         "binaries only")

    if not args.skip_build:
        build()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    out_exe = pack_binary()
    rewrite_readme()
    pack_samples()
    if not args.no_verify:
        verify(out_exe)

    log("done. Try:")
    print(f"  cd {OUT_DIR}")
    print(f"  .\\{EXE_NAME} --help")
    print(f"  .\\{EXE_NAME} dx examples\\simple_add\\simple_add.json")


if __name__ == "__main__":
    main()
