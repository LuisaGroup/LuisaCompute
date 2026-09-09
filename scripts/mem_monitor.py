"""Launch a process and monitor its memory usage (Windows / Linux).

Windows: samples Private Bytes / Working Set via GetProcessMemoryInfo.
Linux:   samples private memory (Private_Clean + Private_Dirty from
         /proc/<pid>/smaps_rollup, fallback RssAnon + VmSwap from
         /proc/<pid>/status) and Working Set (VmRSS).
Kills the process tree if private memory exceeds --kill-gb
(Windows: taskkill /F /T; Linux: SIGKILL to the child's process group).

Usage:
    python scripts/mem_monitor.py --kill-gb 20 --interval 0.25 -- cmd args...
"""
import argparse
import subprocess
import sys
import time

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    def get_memory(pid):
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb):
                return None
            return {
                "private": counters.PrivateUsage,
                "working_set": counters.WorkingSetSize,
                "pagefile": counters.PagefileUsage,
            }
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    def popen_kwargs():
        return {}

    def kill_tree(proc):
        # /T kills the process tree (DXC children etc.)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)

else:
    import os
    import signal

    def _read_status_kb(pid):
        fields = {}
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                for key in ("VmRSS:", "VmSwap:", "RssAnon:"):
                    if line.startswith(key):
                        fields[key] = int(line.split()[1]) * 1024
        return fields

    def _read_smaps_rollup_private(pid):
        private = 0
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith(("Private_Clean:", "Private_Dirty:")):
                    private += int(line.split()[1]) * 1024
        return private

    def get_memory(pid):
        try:
            fields = _read_status_kb(pid)
        except OSError:
            return None  # process gone
        rss = fields.get("VmRSS:", 0)
        swap = fields.get("VmSwap:", 0)
        try:
            private = _read_smaps_rollup_private(pid)
        except OSError:
            # smaps_rollup unavailable (old kernel or restricted): fall back
            # to anonymous RSS + swap as a private-memory approximation.
            private = fields.get("RssAnon:", rss) + swap
        return {"private": private, "working_set": rss, "pagefile": swap}

    def popen_kwargs():
        # own session => own process group, so killpg can take the whole tree
        return {"start_new_session": True}

    def kill_tree(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill-gb", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--log", default="mem_monitor.log")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2

    proc = subprocess.Popen(cmd, **popen_kwargs())
    kill_bytes = int(args.kill_gb * (1 << 30))
    peak = 0
    t0 = time.time()
    killed = False
    samples = []
    with open(args.log, "w") as f:
        f.write("# time_s private_mb working_set_mb\n")
        while True:
            mem = get_memory(proc.pid)
            t = time.time() - t0
            if mem is not None:
                peak = max(peak, mem["private"])
                samples.append((t, mem["private"], mem["working_set"]))
                f.write(f"{t:9.3f} {mem['private'] / 2**20:12.1f} "
                        f"{mem['working_set'] / 2**20:12.1f}\n")
                f.flush()
                if mem["private"] > kill_bytes and not killed:
                    print(f"[monitor] KILLING pid={proc.pid}: private bytes "
                          f"{mem['private'] / 2**30:.2f} GiB > {args.kill_gb} GiB",
                          flush=True)
                    kill_tree(proc)
                    killed = True
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(args.interval)
        if killed:
            # wait for termination
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    # print compact summary: first/last few samples + peak
    print(f"[monitor] exit_code={proc.returncode} killed={killed}")
    print(f"[monitor] peak private bytes: {peak / 2**30:.3f} GiB "
          f"({peak / 2**20:.0f} MiB)")
    if samples:
        print("[monitor] first samples:")
        for t, priv, ws in samples[:5]:
            print(f"  t={t:8.2f}s private={priv / 2**20:10.1f} MiB ws={ws / 2**20:10.1f} MiB")
        print("[monitor] last samples:")
        for t, priv, ws in samples[-5:]:
            print(f"  t={t:8.2f}s private={priv / 2**20:10.1f} MiB ws={ws / 2**20:10.1f} MiB")
    return proc.returncode if proc.returncode is not None else 3


if __name__ == "__main__":
    sys.exit(main())
