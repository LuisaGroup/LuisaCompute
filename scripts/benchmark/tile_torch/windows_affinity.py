"""Read Windows CPU Set topology; affinity writes require an explicit owned handle."""
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime
import json
from pathlib import Path
import struct
import sys

k = c.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = w.HANDLE
k.GetSystemCpuSetInformation.argtypes = [c.c_void_p, w.ULONG, c.POINTER(w.ULONG), w.HANDLE, w.ULONG]
k.GetSystemCpuSetInformation.restype = w.BOOL
k.GetProcessAffinityMask.argtypes = [w.HANDLE, c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)]
k.GetProcessAffinityMask.restype = w.BOOL
k.SetProcessAffinityMask.argtypes = [w.HANDLE, c.c_size_t]
k.SetProcessAffinityMask.restype = w.BOOL
CREATE_SUSPENDED = 0x4


class ThreadEntry(c.Structure):
    _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('thread_id', w.DWORD),
                ('owner_pid', w.DWORD), ('base_priority', w.LONG),
                ('delta_priority', w.LONG), ('flags', w.DWORD)]


k.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
k.CreateToolhelp32Snapshot.restype = w.HANDLE
k.Thread32First.argtypes = [w.HANDLE, c.POINTER(ThreadEntry)]
k.Thread32First.restype = w.BOOL
k.Thread32Next.argtypes = [w.HANDLE, c.POINTER(ThreadEntry)]
k.Thread32Next.restype = w.BOOL
k.OpenThread.argtypes = [w.DWORD, w.BOOL, w.DWORD]
k.OpenThread.restype = w.HANDLE
k.ResumeThread.argtypes = [w.HANDLE]
k.ResumeThread.restype = w.DWORD
k.CloseHandle.argtypes = [w.HANDLE]
k.CloseHandle.restype = w.BOOL


def resume_owned_primary_thread(process):
    # Popen closes CreateProcess's primary-thread handle. A process created
    # suspended has one thread; reopen only that owned PID's thread to resume it.
    snapshot = k.CreateToolhelp32Snapshot(0x4, 0)
    if snapshot == c.c_void_p(-1).value:
        raise c.WinError(c.get_last_error())
    threads = []
    try:
        entry = ThreadEntry(size=c.sizeof(ThreadEntry))
        more = k.Thread32First(snapshot, c.byref(entry))
        while more:
            if entry.owner_pid == process.pid:
                threads.append(entry.thread_id)
            entry.size = c.sizeof(ThreadEntry)
            more = k.Thread32Next(snapshot, c.byref(entry))
    finally:
        k.CloseHandle(snapshot)
    if len(threads) != 1:
        raise RuntimeError(f'Expected one suspended thread in owned PID {process.pid}, found {threads}')
    thread = k.OpenThread(0x2, False, threads[0])
    if not thread:
        raise c.WinError(c.get_last_error())
    try:
        previous = k.ResumeThread(thread)
        if previous != 1:
            raise RuntimeError(f'Unexpected owned primary-thread suspend count: {previous}')
    finally:
        k.CloseHandle(thread)
    return threads[0]


def affinity(handle):
    process, system = c.c_size_t(), c.c_size_t()
    if not k.GetProcessAffinityMask(handle, c.byref(process), c.byref(system)):
        raise c.WinError(c.get_last_error())
    return {'process_mask': hex(process.value), 'system_mask': hex(system.value)}


def set_owned_affinity(handle, mask):
    if not k.SetProcessAffinityMask(handle, mask):
        raise c.WinError(c.get_last_error())
    actual = affinity(handle)
    if int(actual['process_mask'], 16) != mask:
        raise RuntimeError(f'Affinity readback differs from request: {actual}')
    return actual


def topology():
    length = w.ULONG()
    handle = k.GetCurrentProcess()
    success = k.GetSystemCpuSetInformation(None, 0, c.byref(length), handle, 0)
    if not success and c.get_last_error() != 122:
        raise c.WinError(c.get_last_error())
    buffer = c.create_string_buffer(length.value)
    if not k.GetSystemCpuSetInformation(buffer, len(buffer), c.byref(length), handle, 0):
        raise c.WinError(c.get_last_error())
    raw, offset, records = buffer.raw[:length.value], 0, []
    while offset < len(raw):
        size, kind = struct.unpack_from('<II', raw, offset)
        if size < 8 or offset + size > len(raw):
            raise RuntimeError('Malformed variable-size CPU Set record')
        if kind == 0:
            if size < 32:
                raise RuntimeError('CPU Set record smaller than documented structure')
            cpu_id, group, logical, core, cache, numa, efficiency, flags = struct.unpack_from('<IHBBBBBB', raw, offset + 8)
            records.append({'cpu_set_id': cpu_id, 'group': group, 'logical_processor': logical,
                            'core_index': core, 'last_level_cache_index': cache,
                            'numa_node_index': numa, 'efficiency_class': efficiency,
                            'scheduling_class': raw[offset + 20],
                            'parked': bool(flags & 1), 'allocated': bool(flags & 2),
                            'allocated_to_target': bool(flags & 4)})
        offset += size
    return {'queried_at': datetime.now().astimezone().isoformat(),
            'api': 'GetSystemCpuSetInformation', 'records': records,
            'caller_affinity': affinity(handle),
            'references': ['https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information',
                           'https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setprocessaffinitymask']}


class OwnedJob:
    """Kill-on-close job owning only children attached before their startup."""

    def __init__(self):
        class BasicLimits(c.Structure):
            _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64),
                        ('flags', w.DWORD), ('min_working_set', c.c_size_t),
                        ('max_working_set', c.c_size_t), ('active_process_limit', w.DWORD),
                        ('affinity', c.c_size_t), ('priority_class', w.DWORD),
                        ('scheduling_class', w.DWORD)]

        class IoCounters(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in
                        ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

        class ExtendedLimits(c.Structure):
            _fields_ = [('basic', BasicLimits), ('io', IoCounters),
                        ('process_memory_limit', c.c_size_t), ('job_memory_limit', c.c_size_t),
                        ('peak_process_memory', c.c_size_t), ('peak_job_memory', c.c_size_t)]

        k.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        k.CreateJobObjectW.restype = w.HANDLE
        k.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        k.SetInformationJobObject.restype = w.BOOL
        k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        k.AssignProcessToJobObject.restype = w.BOOL
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.get_last_error()
            self.close()
            raise c.WinError(error)

    def attach(self, process):
        if not k.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise c.WinError(c.get_last_error())

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            if not k.CloseHandle(handle):
                raise c.WinError(c.get_last_error())


if __name__ == '__main__':
    result = topology()
    text = json.dumps(result, indent=2) + '\n'
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(text, encoding='utf-8')
    print(text)
