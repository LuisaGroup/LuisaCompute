#!/usr/bin/env python3
"""Check the exact isolated source overlay and archive its local identity."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ISOLATED = Path("/tmp/luisa-reduction-checkpoint.gQUERp/source")
FILES = [
    "include/luisa/tile/bridge/xir/lower.h", "include/luisa/tile/bridge/xir/planner.h",
    "src/tile/bridge/xir/lower.cpp", "src/tile/bridge/xir/planner.cpp", "src/tile/bridge/xir/representation.h",
    "src/backends/simd/runtime/simd_tile.cpp", "src/backends/simd/runtime/simd_shader.cpp",
    "src/backends/simd/simd_compiler.cpp", "src/backends/simd/simd_compiler.h",
    "src/backends/simd/llvm/llvm_schedule_codegen.h", "src/backends/simd/llvm/llvm_schedule_codegen.cpp",
    "src/backends/simd/llvm/llvm_schedule_emitter.h", "src/backends/simd/llvm/llvm_schedule_emitter.cpp",
    "src/backends/simd/llvm/llvm_schedule_emitter_control.cpp",
    "src/tests/unit/tile/bridge/test_xir.cpp", "src/tests/unit/tile/bridge/test_xir_runtime.cpp",
    "src/tests/unit/tile/bridge/test_xir_llm.cpp", "src/tests/unit/simd/test_llvm_schedule_codegen.cpp",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    overlay = {}
    for name in FILES:
        current, tested = digest(ROOT / name), digest(ISOLATED / name)
        if current != tested:
            raise ValueError("untested current source: " + name)
        overlay[name] = current
    helpers = {name: digest(ISOLATED / "src/tests/common" / name) for name in
               ("tile_llm_benchmark.h", "tile_llm_test_utils.h", "tile_xir_test_utils.h", "tile_reduction_policy_test_utils.h")}
    diff = subprocess.check_output(["git", "diff", "3b96c263d", "--", *FILES], cwd=ROOT)
    (HERE / "source-overlay.patch.gz").write_bytes(gzip.compress(diff, mtime=0))
    report = dict(scope="selected source overlay and isolated helper identity; not a full external-dependency lock",
                  base_source="f5daf25e6", helper_checkpoint="db52dbc59", baseline_xir_checkpoint="3b96c263d",
                  source_root=str(ISOLATED), current_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  overlay_sha256=overlay, isolated_helpers_sha256=helpers,
                  build_cache_sha256=digest(ISOLATED.parent / "build/CMakeCache.txt"),
                  overlay_matches_isolated=True)
    (HERE / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: all", len(overlay), "source overlays match the tested build")


if __name__ == "__main__":
    main()
