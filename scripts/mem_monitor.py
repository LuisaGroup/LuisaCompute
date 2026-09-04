"""Launch a process and monitor its memory usage (Windows).

Samples Private Bytes and Working Set via GetProcessMemoryInfo.
Kills the process if private bytes exceed --kill-gb.

Usage:
    python scripts/mem_monitor.py --kill-gb 20 --interval 0.25 -- out.exe args...
"""
import argparse
import ctypes
import subprocess
import sys
import time
from ctypes import wintypes

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_TERMINATE = 0x0001


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

    proc = subprocess.Popen(cmd)
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
                    # /T kills the process tree (DXC children etc.)
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
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
