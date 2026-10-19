#!/usr/bin/env python3
"""Run GitHub Actions workflows locally.

Reads ``.github/workflows/build-cmake.yml`` and
``.github/workflows/build-xmake.yml``, expands their job matrices, and
executes the ``run:`` steps on the local machine.

* Linux jobs are executed through WSL when the host is Windows.
* ``actions/checkout`` is emulated with ``git submodule update``.
* Other ``uses:`` actions (SDK/toolchain installers, artifact upload) are
  skipped with a notice, assuming the corresponding tools are already
  installed locally.
* After each matrix combination finishes, the temporary data and cache it
  generated (build trees, xmake config cache, downloaded keyring/scripts,
  step scripts) are removed -- unless ``--keep`` is given.

Examples:
    python scripts/run_ci.py --list
    python scripts/run_ci.py --workflow cmake --job build-windows --config Release
    python scripts/run_ci.py --workflow xmake --job build-linux --compiler clang
"""

import argparse
import glob
import itertools
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = {
    "cmake": ROOT / ".github" / "workflows" / "build-cmake.yml",
    "xmake": ROOT / ".github" / "workflows" / "build-xmake.yml",
}
CI_TMP = ROOT / ".ci_tmp"

# Actions that cannot (and need not) be reproduced locally. Mapped to a
# human-readable reason that is printed when the step is skipped.
SKIP_ACTIONS = {
    "actions/upload-artifact@v4": "artifact upload is CI-only",
    "johnnynunez/setup-vulkan-sdk@v1": "Vulkan SDK is assumed to be installed locally",
    "Jimver/cuda-toolkit@v0.2.22": "CUDA toolkit is assumed to be installed locally",
    "ilammy/msvc-dev-cmd@v1": "MSVC environment is set up via local vcvars64.bat",
    "ashutoshvarma/setup-ninja@master": "Ninja is assumed to be installed locally",
    "xmake-io/github-action-setup-xmake@v1": "xmake is assumed to be installed locally",
    "ZhongRuoyu/setup-llvm@v0": "LLVM is assumed to be installed locally",
}

# Files/dirs produced by the workflow runs and removed during cleanup.
GENERATED_PATHS = (
    "build",
    "build-eastl-fallback",
    ".xmake",
    "llvm.sh",
    "cuda-keyring_1.1-1_all.deb",
)

EXPR_RE = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def log(msg=""):
    print(msg, flush=True)


def header(msg):
    bar = "=" * 72
    log(f"\n{bar}\n{msg}\n{bar}")


class MatrixView:
    """Read-only attribute view over a matrix dict for expression eval."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        if isinstance(value, dict):
            return MatrixView(value)
        return value

    def __repr__(self):
        return repr(self._data)


def eval_expr(expr, matrix):
    """Evaluate a GitHub Actions expression (matrix context only)."""
    expr = expr.strip()
    # GitHub expressions use && / || / !, translate to Python.
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!([^=])", r"not \1", expr)
    return bool(eval(  # noqa: S307 - restricted namespace
        expr, {"__builtins__": {}}, {"matrix": MatrixView(matrix)}
    ))


def substitute(text, matrix):
    """Replace ${{ matrix.* }} placeholders inside a run block."""

    def repl(match):
        expr = match.group(1).strip()
        m = re.fullmatch(r"matrix\.([A-Za-z_][A-Za-z0-9_.]*)", expr)
        if not m:
            raise ValueError(f"unsupported expression: ${{{{ {expr} }}}}")
        value = matrix
        for part in m.group(1).split("."):
            value = value[part]
        return str(value)

    return EXPR_RE.sub(repl, text)


def host_os():
    s = platform.system()
    return {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(s, s)


# ---------------------------------------------------------------------------
# workflow parsing
# ---------------------------------------------------------------------------

def load_workflow(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_matrix(job):
    """Expand a job's strategy.matrix into a list of matrix dicts."""
    strategy = job.get("strategy") or {}
    matrix = strategy.get("matrix")
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    values = [v if isinstance(v, list) else [v] for v in matrix.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def step_enabled(step, matrix):
    cond = step.get("if")
    if cond is None:
        return True
    return eval_expr(str(cond), matrix)


def job_os(job_name, job, matrix):
    runs_on = substitute(str(job.get("runs-on", "")), matrix)
    for tag, env in (("ubuntu", "linux"), ("windows", "windows"), ("macos", "macos")):
        if tag in runs_on:
            return env
    raise ValueError(f"cannot determine OS from runs-on: {runs_on!r} (job {job_name})")


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def find_vcvars():
    patterns = [
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for pattern in patterns:
        matches = sorted(m for m in glob.glob(pattern) if os.path.isfile(m))
        if matches:
            return matches[0]
    return None


class WslRunner:
    """Execute bash steps inside WSL from a Windows host."""

    def __init__(self, distro=None):
        self.distro = distro or self._default_distro()
        self.posix_root = self._to_posix(ROOT)

    @staticmethod
    def _default_distro():
        out = subprocess.run(
            ["wsl", "-l", "-q"], capture_output=True, text=True, encoding="utf-16-le"
        ).stdout
        names = [line.strip().strip("\x00") for line in out.splitlines() if line.strip()]
        if not names:
            raise RuntimeError("no WSL distribution found; install one or use --wsl-distro")
        return names[0]

    @staticmethod
    def _to_posix(path):
        # wsl.exe mangles backslashes in arguments; use forward slashes.
        win_path = str(path).replace("\\", "/")
        out = subprocess.run(
            ["wsl", "wslpath", "-a", win_path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out

    def bash(self, script_path):
        rel = script_path.relative_to(ROOT).as_posix()
        cmd = ["wsl", "-d", self.distro, "--",
               "bash", "-c",
               f"cd {self.posix_root} && bash --noprofile --norc -eo pipefail {rel}"]
        return subprocess.run(cmd).returncode


def run_step(step, matrix, env, tmp_dir, wsl):
    """Run a single step; returns the exit code (0 for skipped actions)."""
    name = step.get("name") or "Run"
    label = f"----- step: {name}"
    if "uses" in step:
        action = step["uses"]
        if action.startswith("actions/checkout"):
            log(f"{label} (checkout -> git submodule update)")
            return subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=ROOT,
            ).returncode
        reason = SKIP_ACTIONS.get(action, "assumed to be available locally")
        log(f"{label} (skip: {action}: {reason})")
        return 0

    script = substitute(step["run"], matrix)
    shell = step.get("shell")

    if env == "linux":
        script_path = tmp_dir / "step.sh"
        script_path.write_text(script, encoding="utf-8")
        if host_os() == "windows":
            log(f"{label} (bash via WSL: {wsl.distro})")
            return wsl.bash(script_path)
        log(f"{label} (bash)")
        return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail",
                               str(script_path)], cwd=ROOT).returncode

    if env == "windows":
        if shell == "cmd" or (shell is None and os.name == "nt"):
            bat = tmp_dir / "step.bat"
            vcvars = find_vcvars()
            prefix = f'call "{vcvars}" >nul && ' if vcvars else ""
            bat.write_text(script, encoding="utf-8")
            log(f"{label} (cmd)")
            return subprocess.run(f'{prefix}"{bat}"', shell=True, cwd=ROOT).returncode
        # pwsh (or default on non-Windows host pretending to be Windows)
        ps1 = tmp_dir / "step.ps1"
        ps1.write_text(script, encoding="utf-8")
        exe = shutil.which("pwsh") or shutil.which("powershell")
        if exe:
            log(f"{label} (pwsh)")
            return subprocess.run([exe, "-NoProfile", "-File", str(ps1)],
                                  cwd=ROOT).returncode
        log(f"{label} (skip: no PowerShell found for step: {name})")
        return 0

    raise RuntimeError(f"unsupported env: {env}")


def cleanup(run_tmp, preexisting):
    """Remove temp data and cache generated by a finished workflow run.

    Only paths that did NOT exist before the run are removed, so pre-existing
    user data (e.g. a local ``build/`` tree) is never touched.
    """
    removed = []
    for rel in list(GENERATED_PATHS) + [run_tmp]:
        path = ROOT / rel if isinstance(rel, str) else rel
        key = str(path)
        if key in preexisting or not path.exists():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass
        removed.append(rel if isinstance(rel, str) else rel.name)
    # Drop the .ci_tmp root once it is empty.
    try:
        if CI_TMP.is_dir() and not any(CI_TMP.iterdir()):
            CI_TMP.rmdir()
    except OSError:
        pass
    return removed


def run_matrix_entry(workflow_name, job_name, job, matrix, keep, wsl_distro):
    env = job_os(job_name, job, matrix)
    host = host_os()
    parts = [f"{k}={v}" for k, v in matrix.items()]
    title = f"{workflow_name} / {job_name}" + (" / " + ", ".join(parts) if parts else "")
    header(title)

    if env == "macos" and host != "macos":
        log(f"SKIP: macOS job cannot run on a {host} host.")
        return None
    if env == "windows" and host != "windows":
        log(f"SKIP: Windows job cannot run on a {host} host.")
        return None

    wsl = None
    if env == "linux" and host == "windows":
        wsl = WslRunner(wsl_distro)

    # Snapshot which generated paths already exist so cleanup is non-destructive.
    generated = [ROOT / p for p in GENERATED_PATHS]
    preexisting = {str(p) for p in generated if p.exists()}

    slug = "-".join(re.sub(r"[^A-Za-z0-9_.-]", "_", f"{k}_{v}")
                    for k, v in matrix.items())
    tmp_dir = CI_TMP / f"{workflow_name}-{job_name}-{slug}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    try:
        for step in job.get("steps", []):
            if not step_enabled(step, matrix):
                log(f"----- step: {step.get('name') or 'Run'} (skipped by if-condition)")
                continue
            code = run_step(step, matrix, env, tmp_dir, wsl)
            if code != 0:
                # Any failing step fails the job (fail-fast: false in the
                # workflow only governs other matrix entries, handled by
                # --keep-going).
                log(f"step failed with exit code {code}")
                ok = False
                break
    finally:
        if keep:
            log(f"cleanup skipped (--keep); temp dir kept at {tmp_dir}")
        else:
            removed = cleanup(tmp_dir, preexisting)
            if removed:
                log("cleaned: " + ", ".join(removed))

    log(f"RESULT: {title} -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run GitHub Actions workflows locally (see file docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workflow", choices=["cmake", "xmake", "all"], default="all",
                   help="which workflow file(s) to run (default: all)")
    p.add_argument("--job", action="append",
                   help="job id to run, e.g. build-linux (repeatable); default: all jobs "
                        "matching the host, or host-compatible jobs")
    p.add_argument("--config", action="append",
                   help="matrix config filter, e.g. Release / Debug / release / debug")
    p.add_argument("--compiler", "--toolchain", dest="compiler", action="append",
                   help="matrix compiler/toolchain filter, e.g. gcc / clang / cl / msvc")
    p.add_argument("--wsl-distro", default=None,
                   help="WSL distribution for Linux jobs on Windows (default: first listed)")
    p.add_argument("--keep", action="store_true",
                   help="do not clean temp data/cache after each run")
    p.add_argument("--keep-going", action="store_true",
                   help="continue running remaining matrix entries after a failure "
                        "(workflows set fail-fast: false)")
    p.add_argument("--list", action="store_true",
                   help="list the expanded matrix entries and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print the steps that would run without executing them")
    return p.parse_args(argv)


def matrix_matches(matrix, configs, compilers):
    if configs:
        values = {str(v).lower() for v in matrix.values()}
        if not any(c.lower() in values for c in configs):
            return False
    if compilers:
        wanted = {c.lower() for c in compilers}
        values = {str(v).lower() for v in matrix.values()}
        if not (wanted & values):
            return False
    return True


def main(argv=None):
    args = parse_args(argv)
    names = ["cmake", "xmake"] if args.workflow == "all" else [args.workflow]

    # Collect (workflow, job, matrix) triples.
    entries = []
    for name in names:
        wf = load_workflow(WORKFLOWS[name])
        for job_name, job in wf["jobs"].items():
            if args.job and job_name not in args.job:
                continue
            for matrix in expand_matrix(job):
                if not matrix_matches(matrix, args.config, args.compiler):
                    continue
                entries.append((name, job_name, job, matrix))

    if not entries:
        log("no matrix entries match the given filters.")
        return 0

    if args.list or args.dry_run:
        for name, job_name, job, matrix in entries:
            env = job_os(job_name, job, matrix)
            parts = ", ".join(f"{k}={v}" for k, v in matrix.items())
            log(f"[{name}] {job_name} ({env}): {parts or '(no matrix)'}")
            if args.dry_run:
                for step in job.get("steps", []):
                    tag = step.get("name") or ("uses " + step["uses"] if "uses" in step else "Run")
                    status = ""
                    if "run" in step and not step_enabled(step, matrix):
                        status = "  [skipped by if]"
                    elif "run" in step:
                        status = f"  [{step.get('shell') or 'bash' if env != 'windows' else step.get('shell') or 'pwsh/cmd'}]"
                    log(f"    - {tag}{status}")
        return 0

    results = []
    for name, job_name, job, matrix in entries:
        ok = run_matrix_entry(name, job_name, job, matrix, args.keep, args.wsl_distro)
        results.append((f"{name}/{job_name}/{matrix}", ok))
        if ok is False and not args.keep_going:
            log("\naborting remaining entries (use --keep-going to emulate fail-fast: false)")
            break

    header("summary")
    failed = 0
    skipped = 0
    for label, ok in results:
        state = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        if ok is False:
            failed += 1
        elif ok is None:
            skipped += 1
        log(f"{state:4}  {label}")
    if failed:
        log(f"\n{failed} job(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
