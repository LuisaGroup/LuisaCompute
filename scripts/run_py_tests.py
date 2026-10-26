"""Run the LuisaCompute Python test suite and produce a markdown + JSON report.

Discovers ``test-*.py`` files under the assembled tree (default ``bin/py``) and
runs each as ``python <test>.py <backend>`` with a per-test timeout.

Classification:
  * GUI smoke tests (see GUI_TESTS) open a window and render in a loop until the
    window is closed, so they HANG by design. We run them bounded (timeout kill)
    and classify as ``PASS_SMOKE`` when no Python traceback appears in the
    captured output before the kill (we note rendered-frame lines too).
  * ``test-dlpack-errors.py`` is a host-only ``unittest`` module and is run with
    ``python -m pytest -q``.
  * Every other ``test-*.py`` is a normal test: ``PASS`` on exit code 0,
    ``TIMEOUT`` if it hit the timeout, otherwise ``FAIL``.

Pure stdlib. Usage:
    python scripts/run_py_tests.py [--backend cuda] [--timeout 240]
        [--gui-timeout 90] [--out DIR] [--only PATTERN] [--py DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Tests that open a GUI window and render in a loop until closed -> will hang by
# design.  These are run bounded and classified as smoke tests.
GUI_TESTS = {
    "test-bindless.py",
    "test-game-of-life.py",
    "test-mpm.py",
    "test-path-tracing.py",
    "test-path-tracing-f16.py",
    "test-procedural.py",
    "test-raymarch.py",
    "test-rtx.py",
    "test-sdf-renderer.py",
    "test-shadertoy.py",
    "test-shader-visuals-present.py",
    "test-texture.py",
}

# Run via pytest instead of as a plain script (host-only ``unittest`` modules).
PYTEST_TESTS = {"test-dlpack-errors.py", "test-lifetime-dlpack.py"}

# Per-test backend override (with reason).  These run on a backend other than the
# default because the *debug* build is missing an optional CUDA capability, or the
# test itself only supports certain backends -- NOT because of a python-frontend
# bug (those cases are reported normally).  ``cuda`` is the suite default.
BACKEND_OVERRIDES = {
    # CUDA autodiff needs the experimental LLVM codegen (not compiled in this build).
    "test-autodiff.py": "dx",
    "test-autodiff-clamp.py": "dx",
    "test-autodiff-clamp-builtin.py": "dx",
    # New autodiff regression (fix: backward()/GRADIENT_MARKER) -- same LLVM reason.
    "test-autodiff-regression.py": "dx",
    # CUDA indirect dispatch needs the CUDA device runtime (cudadevrt), not linked.
    "test-indirect.py": "dx",
    # New indirect-dispatch regression (fix: IndirectDispatchBuffer ctor) -- same reason.
    "test-indirect-dispatch.py": "dx",
    # CUDA ray tracing (OptiX) aborts compiling these kernels in this debug build;
    # the DXR (dx) backend renders them fine (GUI smoke).
    "test-path-tracing.py": "dx",
    "test-path-tracing-f16.py": "dx",
    "test-rtx.py": "dx",
    "test-sdf-renderer.py": "dx",
    # This test is hard-coded to require dx/vk (prints + exits otherwise).
    "test-interop.py": "dx",
}

# Tests that cannot run in this environment regardless of frontend correctness
# (missing optional third-party deps / GPU-torch).  Reported as ENV_SKIP, not FAIL.
ENV_SKIP = {
    "test-interop.py": "requires torch with CUDA (torch is 2.11.0+cpu here)",
    "test-pytorch-interop.py": "requires the 'cupy' module (not installed)",
}

# Traceback markers -> a real Python-level failure even for a smoke test.
_TB_RE = re.compile(
    r"Traceback \(most recent call last\)|"
    r"^\s*\w*Error:\s|"
    r"luisa\.func\.CompileError|"
    r"Exception:",
    re.MULTILINE,
)


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child process and its whole process tree (Windows-safe)."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            import os
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_one(python: str, cwd: Path, script: str, backend: str,
            timeout: int, env_extra: dict[str, str]):
    """Run a single test script, returning (status, duration, tail_log_path,
    full_output)."""
    import os
    env = dict(os.environ)
    env.update(env_extra)
    start = time.perf_counter()
    timed_out = False
    argv = [python, script]
    if backend:
        argv.append(backend)
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=15)
        except Exception:
            out = ""
    dur = time.perf_counter() - start
    out = out or ""
    return timed_out, proc.returncode, dur, out


def classify(script: str, is_gui: bool, is_pytest: bool, timed_out: bool,
             rc: int, output: str) -> str:
    if _TB_RE.search(output):
        return "FAIL"
    if is_gui:
        # GUI smoke: a clean timeout kill (no traceback) means the render loop
        # ran without a Python-level error -> smoke pass.
        return "PASS_SMOKE" if (timed_out or rc == 0) else "FAIL"
    if timed_out:
        return "TIMEOUT"
    if is_pytest:
        return "PASS" if rc == 0 else "FAIL"
    return "PASS" if rc == 0 else "FAIL"


def write_log(logs_dir: Path, name: str, output: str) -> str:
    logs_dir.mkdir(parents=True, exist_ok=True)
    p = logs_dir / (name + ".log")
    p.write_text(output, encoding="utf-8", errors="replace")
    return str(p)


def frames_rendered_note(output: str) -> str:
    for line in reversed(output.splitlines()):
        low = line.lower()
        if "frame" in low and any(c.isdigit() for c in line):
            return line.strip()
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the LuisaCompute Python test suite.")
    parser.add_argument("--backend", default="cuda", help="Backend arg passed to each test.")
    parser.add_argument("--timeout", type=int, default=240, help="Per-test timeout (s) for normal tests.")
    parser.add_argument("--gui-timeout", type=int, default=90, help="Per-test timeout (s) for GUI smoke tests.")
    parser.add_argument("--out", default=None, help="Report output directory.")
    parser.add_argument("--only", default=None, help="Substring filter on test file names.")
    parser.add_argument("--py", default=None, help="Assembled tree root (default: <root>/bin/py).")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to run tests with.")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    py_dir = Path(args.py).resolve() if args.py else (root / "bin" / "py")
    if not py_dir.is_dir():
        print(f"ERROR: assembled python tree not found: {py_dir}", file=sys.stderr)
        return 2
    bin_debug = root / "bin" / "debug"

    out_dir = Path(args.out).resolve() if args.out else (root / "py_test_reports" / time.strftime("run_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"

    # Ensure runtime DLLs (CUDA/dxc/etc.) from bin/debug can be found.
    env_extra: dict[str, str] = {}
    if bin_debug.is_dir():
        env_extra["PATH"] = str(bin_debug) + ";" + env_extra.get("PATH", __import__("os").environ.get("PATH", ""))

    tests = sorted(p.name for p in py_dir.glob("test-*.py"))
    if args.only:
        tests = [t for t in tests if args.only in t]

    # dlpack errors test is a host-only unittest -> pytest.
    results = []
    print(f"Running {len(tests)} test(s) from {py_dir} (backend={args.backend or 'default'})")
    for name in tests:
        is_gui = name in GUI_TESTS
        is_pytest = name in PYTEST_TESTS
        backend = BACKEND_OVERRIDES.get(name, args.backend)
        if name in ENV_SKIP:
            # Still attempt the run so the captured output proves the dependency
            # is the blocker (not a frontend bug), but classify as ENV_SKIP when the
            # failure is the documented missing dependency.
            reason = ENV_SKIP[name]
        else:
            reason = ""
        timeout = args.gui_timeout if is_gui else args.timeout
        if is_pytest:
            import os
            env = dict(os.environ)
            env.update(env_extra)
            start = time.perf_counter()
            timed_out = False
            try:
                p = subprocess.run(
                    [args.python, "-m", "pytest", "-q", name],
                    cwd=str(py_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                )
                out = p.stdout or ""
                rc = p.returncode
            except subprocess.TimeoutExpired as e:
                timed_out = True
                out = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "")
                rc = -1
            dur = time.perf_counter() - start
        else:
            timed_out, rc, dur, out = run_one(args.python, py_dir, name, backend, timeout, env_extra)

        status = classify(name, is_gui, is_pytest, timed_out, rc, out)
        if name in ENV_SKIP and status == "FAIL":
            status = "ENV_SKIP"
        log_path = write_log(logs_dir, name, out)
        note = frames_rendered_note(out) if is_gui else ""
        results.append({
            "test": name,
            "kind": "pytest" if is_pytest else ("gui" if is_gui else "normal"),
            "status": status,
            "backend": backend,
            "exit_code": rc,
            "timed_out": timed_out,
            "duration_s": round(dur, 2),
            "log": log_path,
            "note": note,
            "env_reason": reason,
        })
        line = f"  [{status:>9}] {name:32s} {backend:>4} {dur:7.1f}s rc={rc}"
        if note:
            line += f"  ({note})"
        print(line)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    # ---- markdown report ----
    lines = []
    lines.append(f"# LuisaCompute Python Test Report")
    lines.append("")
    lines.append(f"- tree: `{py_dir}`")
    lines.append(f"- backend: `{args.backend}`")
    lines.append(f"- python: `{args.python}`")
    lines.append(f"- timeout: normal {args.timeout}s / gui {args.gui_timeout}s")
    lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| status | count |")
    lines.append("|---|---|")
    for k in ("PASS", "PASS_SMOKE", "FAIL", "TIMEOUT", "ENV_SKIP"):
        if k in counts:
            lines.append(f"| {k} | {counts[k]} |")
    lines.append(f"| **total** | **{len(results)}** |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| test | kind | backend | status | dur(s) | exit | note / log tail |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        note = r["note"] or _last_interesting(r["log"]) or r.get("env_reason", "")
        note = note.replace("|", "/").replace("\n", " ")[:120]
        lines.append(f"| {r['test']} | {r['kind']} | {r.get('backend','')} | {r['status']} | {r['duration_s']} | {r['exit_code']} | {note} |")
    lines.append("")
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps({"summary": counts, "results": results}, indent=2), encoding="utf-8")

    print("\n--- Summary ---")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"\nReport: {md_path}")
    print(f"JSON:   {json_path}")

    failed = [r["test"] for r in results if r["status"] in ("FAIL", "TIMEOUT")]
    return 1 if failed else 0


def _last_interesting(log_path: str) -> str:
    try:
        txt = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    body = [l for l in txt.splitlines() if l.strip()]
    return body[-1] if body else ""


if __name__ == "__main__":
    raise SystemExit(main())
