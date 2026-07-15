#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debugger.py

A lightweight Windows debugger implemented in Python using `ctypes` to call
the Windows Debug API and DbgHelp.dll.  It launches a native x64 executable
under the debugger, waits for a crash (second-chance exception) and prints a
symbolic stack trace by reading the matching PDB file(s).

Usage:
    python debugger.py <path_to_exe> [pdb_search_path] [-- <args>...]

Arguments after `--` are passed to the target executable.

Requirements:
    - Windows x64
    - Python 3.x (64-bit is recommended so the pointer widths match the target)
    - The target EXE must have a matching PDB (either next to the EXE or in the
      supplied search path).

Note:
    This is intentionally minimal.  For production use consider generating a
    minidump and analyzing it with cdb/windbg or Breakpad/Crashpad.
"""

import ctypes
import ctypes.wintypes as wintypes
import os
import sys

# =============================================================================
# 1. Windows constants
# =============================================================================

DEBUG_PROCESS = 0x00000001
DEBUG_ONLY_THIS_PROCESS = 0x00000002

EXCEPTION_DEBUG_EVENT = 1
CREATE_THREAD_DEBUG_EVENT = 2
CREATE_PROCESS_DEBUG_EVENT = 3
EXIT_THREAD_DEBUG_EVENT = 4
EXIT_PROCESS_DEBUG_EVENT = 5
LOAD_DLL_DEBUG_EVENT = 6
UNLOAD_DLL_DEBUG_EVENT = 7
OUTPUT_DEBUG_STRING_EVENT = 8
RIP_EVENT = 9

DBG_EXCEPTION_HANDLED = 0x00010001
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001

THREAD_ALL_ACCESS = 0x001FFFFF

# x64 CONTEXT flags
CONTEXT_AMD64 = 0x00100000
CONTEXT_CONTROL = CONTEXT_AMD64 | 0x00000001
CONTEXT_INTEGER = CONTEXT_AMD64 | 0x00000002
CONTEXT_SEGMENTS = CONTEXT_AMD64 | 0x00000004
CONTEXT_FLOATING_POINT = CONTEXT_AMD64 | 0x00000008
CONTEXT_DEBUG_REGISTERS = CONTEXT_AMD64 | 0x00000010
CONTEXT_FULL = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS  # 0x0010000B
CONTEXT_ALL = CONTEXT_FULL | CONTEXT_FLOATING_POINT | CONTEXT_DEBUG_REGISTERS

IMAGE_FILE_MACHINE_AMD64 = 0x8664
ADDRESS_MODE_FLAT = 3  # AddrModeFlat from dbghelp.h, NOT 0

# Common exception codes
EXCEPTION_ACCESS_VIOLATION = 0xC0000005
EXCEPTION_BREAKPOINT = 0x80000003
EXCEPTION_SINGLE_STEP = 0x80000004

# DbgHelp symbol options
SYMOPT_LOAD_LINES = 0x00000010
SYMOPT_UNDNAME = 0x00000100
SYMOPT_DEFERRED_LOADS = 0x00000004
SYMOPT_CASE_INSENSITIVE = 0x00000001

# =============================================================================
# 2. Windows structures
# =============================================================================


class STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", wintypes.LPBYTE),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("ExceptionCode", wintypes.DWORD),
        ("ExceptionFlags", wintypes.DWORD),
        ("ExceptionRecord", wintypes.LPVOID),
        ("ExceptionAddress", wintypes.LPVOID),
        ("NumberParameters", wintypes.DWORD),
        ("ExceptionInformation", ctypes.c_ulonglong * 15),
        ("dwFirstChance", wintypes.DWORD),
    ]


class DEBUG_EVENT(ctypes.Structure):
    # The union is 160 bytes on x64; the whole structure is 176 bytes.
    # Windows inserts 4 bytes of padding after dwThreadId to 8-byte-align
    # the union, so we expose it explicitly.
    _fields_ = [
        ("dwDebugEventCode", wintypes.DWORD),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
        ("_padding", wintypes.DWORD),
        ("u", ctypes.c_ubyte * 160),
    ]


class CONTEXT(ctypes.Structure):
    """x64 CONTEXT structure.

    Only the registers we need are exposed by name; the remaining bytes are
    stored in a filler array so the overall size matches Windows' 1232 bytes.
    """
    _fields_ = [
        ("P1Home", ctypes.c_ulonglong),
        ("P2Home", ctypes.c_ulonglong),
        ("P3Home", ctypes.c_ulonglong),
        ("P4Home", ctypes.c_ulonglong),
        ("P5Home", ctypes.c_ulonglong),
        ("P6Home", ctypes.c_ulonglong),
        ("ContextFlags", wintypes.DWORD),
        ("MxCsr", wintypes.DWORD),
        ("SegCs", wintypes.WORD),
        ("SegDs", wintypes.WORD),
        ("SegEs", wintypes.WORD),
        ("SegFs", wintypes.WORD),
        ("SegGs", wintypes.WORD),
        ("SegSs", wintypes.WORD),
        ("EFlags", wintypes.DWORD),
        ("Dr0", ctypes.c_ulonglong),
        ("Dr1", ctypes.c_ulonglong),
        ("Dr2", ctypes.c_ulonglong),
        ("Dr3", ctypes.c_ulonglong),
        ("Dr6", ctypes.c_ulonglong),
        ("Dr7", ctypes.c_ulonglong),
        ("Rax", ctypes.c_ulonglong),
        ("Rcx", ctypes.c_ulonglong),
        ("Rdx", ctypes.c_ulonglong),
        ("Rbx", ctypes.c_ulonglong),
        ("Rsp", ctypes.c_ulonglong),
        ("Rbp", ctypes.c_ulonglong),
        ("Rsi", ctypes.c_ulonglong),
        ("Rdi", ctypes.c_ulonglong),
        ("R8", ctypes.c_ulonglong),
        ("R9", ctypes.c_ulonglong),
        ("R10", ctypes.c_ulonglong),
        ("R11", ctypes.c_ulonglong),
        ("R12", ctypes.c_ulonglong),
        ("R13", ctypes.c_ulonglong),
        ("R14", ctypes.c_ulonglong),
        ("R15", ctypes.c_ulonglong),
        ("Rip", ctypes.c_ulonglong),
        # Padding to reach sizeof(CONTEXT) == 1232 on x64.
        # Rip ends at offset 256; the remaining 976 bytes are the FPU/vector
        # register state and debugging metadata that StackWalk64 may need.
        ("_Reserved", ctypes.c_ubyte * (1232 - 256)),
    ]


class ADDRESS64(ctypes.Structure):
    _fields_ = [
        ("Offset", ctypes.c_ulonglong),
        ("Segment", wintypes.DWORD),
        ("Mode", wintypes.DWORD),
    ]


class KDHELP64(ctypes.Structure):
    _fields_ = [
        ("Thread", ctypes.c_ulonglong),
        ("ThCallbackStack", wintypes.DWORD),
        ("ThCallbackBStore", wintypes.DWORD),
        ("NextCallback", wintypes.DWORD),
        ("FramePointer", wintypes.DWORD),
        ("KiCallUserMode", ctypes.c_ulonglong),
        ("KeUserCallbackDispatcher", ctypes.c_ulonglong),
        ("SystemRangeStart", ctypes.c_ulonglong),
        ("KiUserExceptionDispatcher", ctypes.c_ulonglong),
        ("StackBase", ctypes.c_ulonglong),
        ("StackLimit", ctypes.c_ulonglong),
        ("BuildVersion", wintypes.DWORD),
        ("RetpolineStubFunctionTableSize", wintypes.DWORD),
        ("RetpolineStubFunctionTable", ctypes.c_ulonglong),
        ("RetpolineStubOffset", wintypes.DWORD),
        ("RetpolineStubSize", wintypes.DWORD),
        ("Reserved0", ctypes.c_ulonglong * 2),
    ]


class STACKFRAME64(ctypes.Structure):
    _fields_ = [
        ("AddrPC", ADDRESS64),
        ("AddrReturn", ADDRESS64),
        ("AddrFrame", ADDRESS64),
        ("AddrStack", ADDRESS64),
        ("AddrBStore", ADDRESS64),
        ("FuncTableEntry", wintypes.LPVOID),
        ("Params", ctypes.c_ulonglong * 4),
        ("Far", wintypes.BOOL),
        ("Virtual", wintypes.BOOL),
        ("Reserved", ctypes.c_ulonglong * 3),
        ("KdHelp", KDHELP64),
    ]


class SYMBOL_INFO(ctypes.Structure):
    # Layout matches DbgHelp.h SYMBOL_INFO (Name[1] is expanded to 2000 chars).
    _fields_ = [
        ("SizeOfStruct", wintypes.DWORD),
        ("TypeIndex", wintypes.DWORD),
        ("Reserved", ctypes.c_ulonglong * 2),
        ("Index", wintypes.DWORD),
        ("Size", wintypes.DWORD),
        ("ModBase", ctypes.c_ulonglong),
        ("Flags", wintypes.DWORD),
        ("Value", ctypes.c_ulonglong),
        ("Address", ctypes.c_ulonglong),
        ("Register", wintypes.DWORD),
        ("Scope", wintypes.DWORD),
        ("Tag", wintypes.DWORD),
        ("NameLen", wintypes.DWORD),
        ("MaxNameLen", wintypes.DWORD),
        ("Name", wintypes.CHAR * 2000),
    ]


class IMAGEHLP_LINE64(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", wintypes.DWORD),
        ("Key", wintypes.LPVOID),
        ("LineNumber", wintypes.DWORD),
        ("FileName", wintypes.PCHAR),
        ("Address", ctypes.c_ulonglong),
    ]


class CREATE_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hFile", wintypes.HANDLE),
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("lpBaseOfImage", wintypes.LPVOID),
        ("dwDebugInfoFileOffset", wintypes.DWORD),
        ("nDebugInfoSize", wintypes.DWORD),
        ("lpThreadLocalBase", wintypes.LPVOID),
        ("lpStartAddress", wintypes.LPVOID),
        ("lpImageName", wintypes.LPVOID),
        ("fUnicode", wintypes.WORD),
        ("_padding", wintypes.WORD * 3),
    ]


class LOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hFile", wintypes.HANDLE),
        ("lpBaseOfDll", wintypes.LPVOID),
        ("dwDebugInfoFileOffset", wintypes.DWORD),
        ("nDebugInfoSize", wintypes.DWORD),
        ("lpImageName", wintypes.LPVOID),
        ("fUnicode", wintypes.WORD),
        ("_padding", wintypes.WORD * 3),
    ]


# =============================================================================
# 3. Load DLLs and set function prototypes
# =============================================================================

kernel32 = ctypes.windll.kernel32
dbghelp = ctypes.windll.dbghelp

# Kernel32
CreateProcess = kernel32.CreateProcessW
CreateProcess.argtypes = [
    wintypes.LPCWSTR, wintypes.LPWSTR,
    wintypes.LPVOID, wintypes.LPVOID,
    wintypes.BOOL, wintypes.DWORD, wintypes.LPVOID, wintypes.LPCWSTR,
    ctypes.POINTER(STARTUPINFO), ctypes.POINTER(PROCESS_INFORMATION)
]
CreateProcess.restype = wintypes.BOOL

WaitForDebugEvent = kernel32.WaitForDebugEvent
WaitForDebugEvent.argtypes = [ctypes.POINTER(DEBUG_EVENT), wintypes.DWORD]
WaitForDebugEvent.restype = wintypes.BOOL

ContinueDebugEvent = kernel32.ContinueDebugEvent
ContinueDebugEvent.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
ContinueDebugEvent.restype = wintypes.BOOL

OpenThread = kernel32.OpenThread
OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
OpenThread.restype = wintypes.HANDLE

GetThreadContext = kernel32.GetThreadContext
GetThreadContext.argtypes = [wintypes.HANDLE, ctypes.POINTER(CONTEXT)]
GetThreadContext.restype = wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

ReadProcessMemory = kernel32.ReadProcessMemory
ReadProcessMemory.argtypes = [
    wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]
ReadProcessMemory.restype = wintypes.BOOL

# DbgHelp
SymSetOptions = dbghelp.SymSetOptions
SymSetOptions.argtypes = [wintypes.DWORD]
SymSetOptions.restype = wintypes.DWORD

SymInitialize = dbghelp.SymInitialize
SymInitialize.argtypes = [wintypes.HANDLE, wintypes.LPCSTR, wintypes.BOOL]
SymInitialize.restype = wintypes.BOOL

SymCleanup = dbghelp.SymCleanup
SymCleanup.argtypes = [wintypes.HANDLE]
SymCleanup.restype = wintypes.BOOL

SymRefreshModuleList = dbghelp.SymRefreshModuleList
SymRefreshModuleList.argtypes = [wintypes.HANDLE]
SymRefreshModuleList.restype = wintypes.BOOL

SymLoadModule64 = dbghelp.SymLoadModule64
SymLoadModule64.argtypes = [
    wintypes.HANDLE, wintypes.HANDLE, wintypes.LPCSTR, wintypes.LPCSTR,
    ctypes.c_ulonglong, wintypes.DWORD
]
SymLoadModule64.restype = ctypes.c_ulonglong

SymFromAddr = dbghelp.SymFromAddr
SymFromAddr.argtypes = [
    wintypes.HANDLE, ctypes.c_ulonglong,
    ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(SYMBOL_INFO)
]
SymFromAddr.restype = wintypes.BOOL

SymGetLineFromAddr64 = dbghelp.SymGetLineFromAddr64
SymGetLineFromAddr64.argtypes = [
    wintypes.HANDLE, ctypes.c_ulonglong,
    ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(IMAGEHLP_LINE64)
]
SymGetLineFromAddr64.restype = wintypes.BOOL

SymFunctionTableAccess64 = dbghelp.SymFunctionTableAccess64
SymFunctionTableAccess64.argtypes = [wintypes.HANDLE, ctypes.c_ulonglong]
SymFunctionTableAccess64.restype = wintypes.LPVOID

SymGetModuleBase64 = dbghelp.SymGetModuleBase64
SymGetModuleBase64.argtypes = [wintypes.HANDLE, ctypes.c_ulonglong]
SymGetModuleBase64.restype = ctypes.c_ulonglong

StackWalk64 = dbghelp.StackWalk64
StackWalk64.argtypes = [
    wintypes.DWORD, wintypes.HANDLE, wintypes.HANDLE,
    ctypes.POINTER(STACKFRAME64), ctypes.POINTER(CONTEXT),
    wintypes.LPVOID, wintypes.LPVOID, wintypes.LPVOID, wintypes.LPVOID
]
StackWalk64.restype = wintypes.BOOL


# =============================================================================
# 4. Helpers
# =============================================================================


def decode_symbol_name(buf):
    """Decode a null-terminated byte buffer/pointer to a Python string."""
    if not buf:
        return ""
    # ctypes.c_char_p is a pointer; read the C string it points to.
    if isinstance(buf, ctypes.POINTER(ctypes.c_char)) or isinstance(buf, ctypes.c_char_p):
        if not buf:
            return ""
        return ctypes.string_at(buf).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    if isinstance(buf, bytes):
        return buf.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    # For fixed-size arrays find the null terminator.
    end = buf.find(b"\x00")
    if end == -1:
        end = len(buf)
    return buf[:end].decode("utf-8", errors="ignore")


def read_remote_string(h_process, ptr, is_unicode):
    """Read a remote process image-name string described by a debug event."""
    if not ptr:
        return None
    # lpImageName points to another pointer in the target address space.
    string_ptr = ctypes.c_void_p()
    read = ctypes.c_size_t()
    if not ReadProcessMemory(
        h_process, ptr, ctypes.byref(string_ptr),
        ctypes.sizeof(ctypes.c_void_p), ctypes.byref(read)
    ):
        return None
    if not string_ptr.value:
        return None
    buf = ctypes.create_string_buffer(512)
    if not ReadProcessMemory(
        h_process, string_ptr.value, buf, 512, ctypes.byref(read)
    ):
        return None
    data = buf.raw[:read.value]
    if is_unicode:
        try:
            return data.decode("utf-16-le").split("\x00", 1)[0]
        except UnicodeDecodeError:
            return None
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


def get_image_size(h_process, base_address):
    """Read SizeOfImage from the PE headers of a loaded module."""
    try:
        buf = ctypes.create_string_buffer(0x1000)
        read = ctypes.c_size_t()
        if ReadProcessMemory(h_process, base_address, buf, 0x1000, ctypes.byref(read)):
            data = buf.raw
            pe_offset = int.from_bytes(data[0x3C:0x40], "little")
            if 0 < pe_offset < 0x1000 and data[pe_offset:pe_offset + 2] == b"PE":
                size_offset = pe_offset + 0x50
                if size_offset + 4 <= len(data):
                    return int.from_bytes(data[size_offset:size_offset + 4], "little")
    except Exception:
        pass
    return 0


def format_exception_code(code):
    """Return a human-readable name for common exception codes."""
    names = {
        EXCEPTION_ACCESS_VIOLATION: "EXCEPTION_ACCESS_VIOLATION",
        EXCEPTION_BREAKPOINT: "EXCEPTION_BREAKPOINT",
        EXCEPTION_SINGLE_STEP: "EXCEPTION_SINGLE_STEP",
        0xC000008C: "EXCEPTION_ARRAY_BOUNDS_EXCEEDED",
        0xC0000094: "EXCEPTION_INT_DIVIDE_BY_ZERO",
        0xC0000095: "EXCEPTION_INT_OVERFLOW",
        0xC00000FD: "EXCEPTION_STACK_OVERFLOW",
        0xE06D7363: "C++ EH / MSVC exception",
    }
    return names.get(code, "UNKNOWN")


def print_stack_trace(h_process, h_thread, ctx, max_frames=64):
    """Walk the stack and print function names / file:line from PDB symbols."""
    frame = STACKFRAME64()
    frame.AddrPC.Offset = ctx.Rip
    frame.AddrPC.Mode = ADDRESS_MODE_FLAT
    # For x64 StackWalk64 uses RSP-based unwinding; RBP may not be a frame chain.
    frame.AddrFrame.Offset = ctx.Rsp
    frame.AddrFrame.Mode = ADDRESS_MODE_FLAT
    frame.AddrStack.Offset = ctx.Rsp
    frame.AddrStack.Mode = ADDRESS_MODE_FLAT
    frame.AddrReturn.Mode = ADDRESS_MODE_FLAT
    frame.AddrBStore.Mode = ADDRESS_MODE_FLAT

    print("\n=== C++ stack trace (with PDB symbols) ===")
    frame_count = 0

    while frame_count < max_frames:
        ok = StackWalk64(
            IMAGE_FILE_MACHINE_AMD64,
            h_process,
            h_thread,
            ctypes.byref(frame),
            ctypes.byref(ctx),
            None,
            SymFunctionTableAccess64,
            SymGetModuleBase64,
            None
        )
        if not ok or frame.AddrPC.Offset == 0:
            break

        addr = frame.AddrPC.Offset
        func_name = ""
        file_name = ""
        line_num = 0

        # Resolve function name.
        symbol = SYMBOL_INFO()
        symbol.SizeOfStruct = 88          # sizeof(SYMBOL_INFO) in C
        symbol.MaxNameLen = 2000
        displacement = ctypes.c_ulonglong(0)
        if SymFromAddr(h_process, addr, ctypes.byref(displacement), ctypes.byref(symbol)):
            func_name = decode_symbol_name(symbol.Name)
        else:
            func_name = "<unknown>"

        # Resolve source file and line number (requires PDB with line info).
        line = IMAGEHLP_LINE64()
        line.SizeOfStruct = 40            # sizeof(IMAGEHLP_LINE64) in C
        line_displacement = wintypes.DWORD(0)
        if SymGetLineFromAddr64(h_process, addr, ctypes.byref(line_displacement), ctypes.byref(line)):
            if line.FileName:
                file_name = decode_symbol_name(line.FileName)
            line_num = line.LineNumber

        location = f"{file_name}:{line_num}" if file_name and line_num else ""
        print(f"  #{frame_count:2d} 0x{addr:016X} {func_name}")
        if location:
            print(f"       {location}")

        frame_count += 1

    print("==========================================\n")


# =============================================================================
# 5. Debugger main loop
# =============================================================================


def start_debugger(exe_path, pdb_search_path=None, exe_args=None, wait_timeout_ms=5000):
    """Launch `exe_path` under the debugger and print a symbolic stack on crash."""
    if exe_args is None:
        exe_args = []
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    exe_dir = os.path.dirname(os.path.abspath(exe_path))
    if pdb_search_path is None:
        pdb_search_path = exe_dir

    # Symbol path: use a local cache for the Microsoft symbol server plus the
    # user-supplied directory.  Use semicolons to separate paths.
    sym_path = f"SRV*C:\\Symbols*https://msdl.microsoft.com/download/symbols;{pdb_search_path}"

    si = STARTUPINFO()
    si.cb = ctypes.sizeof(STARTUPINFO)
    pi = PROCESS_INFORMATION()

    # Build command line: module name followed by any extra arguments.
    command_line = exe_path
    if exe_args:
        command_line += " " + " ".join(exe_args)

    ok = CreateProcess(
        exe_path, command_line, None, None, False,
        DEBUG_PROCESS | DEBUG_ONLY_THIS_PROCESS,
        None, None, ctypes.byref(si), ctypes.byref(pi)
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())

    print(f"[+] Started process PID={pi.dwProcessId}, TID={pi.dwThreadId}")

    symbols_initialized = False
    debug_event = DEBUG_EVENT()

    try:
        # Configure DbgHelp options before initializing symbols.
        SymSetOptions(
            SYMOPT_LOAD_LINES
            | SYMOPT_UNDNAME
            | SYMOPT_DEFERRED_LOADS
            | SYMOPT_CASE_INSENSITIVE
        )

        # fInvadeProcess must be False while the child is still at its initial
        # breakpoint.  We load the module list explicitly below.
        if not SymInitialize(pi.hProcess, sym_path.encode("utf-8"), False):
            err = ctypes.get_last_error()
            print(f"[-] SymInitialize failed: {err}")
        else:
            symbols_initialized = True
            print(f"[+] Symbol engine initialized (path: {sym_path})")

        while True:
            if not WaitForDebugEvent(ctypes.byref(debug_event), wait_timeout_ms):
                # Timeout - just loop.  The target may still be running.
                continue

            pid = debug_event.dwProcessId
            tid = debug_event.dwThreadId
            code = debug_event.dwDebugEventCode

            if code == EXCEPTION_DEBUG_EVENT:
                exc_info = ctypes.cast(
                    ctypes.addressof(debug_event.u),
                    ctypes.POINTER(EXCEPTION_DEBUG_INFO)
                ).contents

                exc_name = format_exception_code(exc_info.ExceptionCode)
                print(f"\n[!] Caught exception: PID={pid}, TID={tid}")
                print(f"    Code:        0x{exc_info.ExceptionCode:08X} ({exc_name})")
                print(f"    Address:     0x{exc_info.ExceptionAddress:016X}")
                print(f"    FirstChance: {exc_info.dwFirstChance}")

                # Open the faulting thread and retrieve its context.
                h_thread = OpenThread(THREAD_ALL_ACCESS, False, tid)
                if h_thread:
                    ctx = CONTEXT()
                    ctx.ContextFlags = CONTEXT_ALL
                    if GetThreadContext(h_thread, ctypes.byref(ctx)):
                        print(f"[+] Got thread context, RIP=0x{ctx.Rip:016X}")
                        print_stack_trace(pi.hProcess, h_thread, ctx)
                    else:
                        print(f"[-] GetThreadContext failed: {ctypes.get_last_error()}")
                    CloseHandle(h_thread)
                else:
                    print(f"[-] OpenThread failed: {ctypes.get_last_error()}")

                # Second-chance exceptions are unhandled crashes.
                if exc_info.dwFirstChance == 0:
                    print("[-] Second-chance exception; the process is terminating.")
                    break

                # Continue appropriately.  For breakpoints / single-step use
                # DBG_CONTINUE so the debuggee does not need to handle them.
                if exc_info.ExceptionCode in (EXCEPTION_BREAKPOINT, EXCEPTION_SINGLE_STEP):
                    cont_status = DBG_CONTINUE
                else:
                    cont_status = DBG_EXCEPTION_NOT_HANDLED
                if not ContinueDebugEvent(pid, tid, cont_status):
                    print(f"[-] ContinueDebugEvent failed: {ctypes.get_last_error()}")
                    break
                continue

            elif code == EXIT_PROCESS_DEBUG_EVENT:
                print("[+] Process exited normally.")
                break

            elif code == CREATE_THREAD_DEBUG_EVENT:
                print(f"[+] Thread created: TID={tid}")

            elif code == CREATE_PROCESS_DEBUG_EVENT:
                cpdi = ctypes.cast(
                    ctypes.addressof(debug_event.u),
                    ctypes.POINTER(CREATE_PROCESS_DEBUG_INFO)
                ).contents
                if symbols_initialized:
                    size = get_image_size(pi.hProcess, cpdi.lpBaseOfImage)
                    base = SymLoadModule64(
                        pi.hProcess, cpdi.hFile, exe_path.encode("utf-8"),
                        None, ctypes.c_ulonglong(cpdi.lpBaseOfImage), size
                    )
                    if base:
                        print(f"[+] Loaded module {exe_path} at 0x{base:016X}")
                    else:
                        print(f"[-] SymLoadModule64 failed: {ctypes.get_last_error()}")

            elif code == LOAD_DLL_DEBUG_EVENT:
                lddi = ctypes.cast(
                    ctypes.addressof(debug_event.u),
                    ctypes.POINTER(LOAD_DLL_DEBUG_INFO)
                ).contents
                if symbols_initialized:
                    size = get_image_size(pi.hProcess, lddi.lpBaseOfDll)
                    base = SymLoadModule64(
                        pi.hProcess, lddi.hFile, None, None,
                        ctypes.c_ulonglong(lddi.lpBaseOfDll), size
                    )
                    if base:
                        dll_name = read_remote_string(pi.hProcess, lddi.lpImageName, lddi.fUnicode)
                        if dll_name:
                            print(f"[+] Loaded DLL {dll_name}")
                    # Failures for system DLLs are usually harmless; DbgHelp can
                    # still resolve many addresses via the public symbol server.

            # For all other events, let the process continue.
            ContinueDebugEvent(pid, tid, DBG_CONTINUE)

    finally:
        if symbols_initialized:
            SymCleanup(pi.hProcess)
        CloseHandle(pi.hProcess)
        CloseHandle(pi.hThread)
        print("[+] Debugger exiting.")


# =============================================================================
# 6. Entry point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debugger.py <path_to_exe> [pdb_search_path] [-- <args>...]")
        sys.exit(1)

    exe = sys.argv[1]
    # Split arguments: everything before `--` (if present) belongs to the
    # debugger; everything after `--` is forwarded to the target executable.
    if "--" in sys.argv:
        split_idx = sys.argv.index("--")
        debugger_args = sys.argv[1:split_idx]
        exe_args = sys.argv[split_idx + 1:]
    else:
        debugger_args = sys.argv[1:]
        exe_args = []

    exe = debugger_args[0]
    pdb_path = debugger_args[1] if len(debugger_args) > 1 else None
    start_debugger(exe, pdb_path, exe_args)
