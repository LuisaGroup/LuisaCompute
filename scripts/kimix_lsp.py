#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kimix_lsp — Kimix session-server LSP bridge for the LuisaCompute repo.

A self-contained, **stdlib-only** JSON-RPC 2.0 session server implementing the
Kimix session-server wire protocol (docs/session_server_rpc.md §1–§7) plus a
clangd-backed C++ syntax bridge for the ``connect`` tool hook (``before_tool``).

Behaviour summary:

* ``before_tool`` with action ``read``  → compile-database info (compiler,
  standard, defines, include dirs) for files inside ``compile_commands.json``.
* ``before_tool`` with action ``write``/``edit`` → clangd syntax diagnostics of
  the *pre-edit* on-disk content for files inside the compile database.

Both payloads are shaped for brevity: root-relative paths, short capped lists,
one clipped line per diagnostic, and Info/Hint diagnostics dropped.
* ``get_compile_info`` / ``check_syntax`` / ``reload`` / ``ping`` RPC methods.
* Everything degrades gracefully: an unavailable clangd, a missing compile
  database, timeouts — the bridge caller only ever sees "no extra info",
  never an exception.

Usage::

    python scripts/kimix_lsp.py --network tcp --host <address> --port <port> \
        [--root DIR] [--stop-file PATH] [--compile-commands PATH] \
        [--clangd PATH] [--no-clangd] [--timeout SECONDS] [--verbose]

Transports: TCP with newline-delimited JSON (fully), and HTTP ``POST /rpc``
mirroring the reference dummy server. Responses are compact UTF-8 JSON.
Startup logs (``kimix_lsp listening on tcp://host:port``, clangd and compile-db
status) go to **stderr** — stdout is never written to.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── JSON-RPC 2.0 constants (session_server_rpc.md §3) ─────────────────────────
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Same framing limit as the reference server: 16 MiB per TCP line.
TCP_STREAM_LIMIT = 16 * 1024 * 1024

# Message shaping caps (spec §6/§7). Everything the bridge injects into a tool
# result — and everything an RPC result carries — is deliberately terse: the
# agent pays for these tokens, so lists are short, paths are root-relative,
# and a diagnostic is one clipped line.
MAX_INLINE_ITEMS = 6         # defines / include dirs per inline list
MAX_SYNTAX_LINES = 6         # diagnostics shown in a before_tool message
MAX_FORMATTED = 10           # diagnostics kept in check_syntax results
MAX_PATHS_HANDLED = 4        # before_tool multi-file cap
RESULT_LIST_CAP = 48         # defines / include dirs per RPC result
MAX_DIAG_CHARS = 180         # hard clip for one formatted diagnostic
MAX_INLINE_PATH_CHARS = 60   # clip for one inline include dir
MAX_INLINE_DEFINE_CHARS = 80 # clip for one inline define

DB_FILENAME = "compile_commands.json"
SOURCE_EXTS = frozenset({".cpp", ".cc", ".cxx", ".c", ".cu", ".cp"})
HEADER_EXTS = frozenset({".h", ".hpp", ".hh", ".hxx", ".inl", ".h++", ".tcc"})
_MSVC_STYLE_COMPILERS = frozenset({
    "cl", "cl.exe", "clang-cl", "clang-cl.exe", "icl", "icl.exe",
    "icx-cl", "icx-cl.exe", "hostcl", "hostx64\\cl.exe",
})

# UTF-8 stdio for logging without breaking the protocol stream. (The protocol
# itself only ever writes to sockets, never to stdout.)
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_VERBOSE = False


class RpcError(Exception):
    """Handler-level error carrying a JSON-RPC error code."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _vlog(line: str) -> None:
    if _VERBOSE:
        _log(f"[verbose] {line}")


def _norm_key(path_like: Any) -> str:
    """Index key: normcase makes this case-insensitive on Windows and
    identity-with-slash-normalisation on POSIX — the only platform branch."""
    return os.path.normcase(str(path_like))


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _split_compile_command(command: str) -> list[str]:
    """Tokenize one ``command`` string into arguments.

    On POSIX this is ``shlex.split(..., posix=True)`` (same result as
    check_cpp_syntax's ``posix=os.name != "nt"`` formula there). On Windows
    ``shlex`` in non-POSIX mode still breaks a quoted inline value like
    ``/I"<include dir with spaces>"`` at the inner space, so tokenize
    manually: double quotes toggle grouping, backslashes stay literal, and surrounding quotes are stripped from
    whole-token (``"cl.exe"``) and flag-inline values (``/I"..."``, ``/D"..."``,
    ``-I"..."``, ``-D"..."``) alike — mirroring how cl.exe itself reads its
    command line.
    """
    if os.name != "nt":
        return shlex.split(command, posix=True)
    arguments: list[str] = []
    current: list[str] = []
    in_quotes = False
    for char in command:
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
        elif char in " \t" and not in_quotes:
            if current:
                arguments.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        arguments.append("".join(current))
    stripped = []
    for argument in arguments:
        if len(argument) >= 2 and argument[0] == argument[-1] == '"':
            argument = argument[1:-1]
        else:
            match = re.match(r'^(/[^"\s]+|-I|-D)"(.*)"$', argument)
            if match:
                argument = match.group(1) + match.group(2)
        stripped.append(argument)
    return stripped


def _syntax_only_arguments(arguments: list[str]) -> list[str]:
    """Demote warning-as-error flags without hiding compiler warnings."""
    result = []
    for argument in arguments:
        if argument == "-Werror":
            result.append("-Wno-error")
        elif argument.startswith("-Werror="):
            result.append("-Wno-error=" + argument[len("-Werror="):])
        elif argument == "-pedantic-errors":
            result.append("-pedantic")
        elif argument.lower() == "/wx":
            result.append("/WX-")
        else:
            result.append(argument)
    return result


def _syntax_only_entry(entry: dict) -> dict:
    sanitized = dict(entry)
    arguments = entry.get("arguments")
    if isinstance(arguments, list) and all(
        isinstance(argument, str) for argument in arguments
    ):
        parsed = list(arguments)
    else:
        command = entry.get("command")
        if not isinstance(command, str):
            return sanitized
        try:
            parsed = _split_compile_command(command)
        except ValueError:
            return sanitized
    sanitized["arguments"] = _syntax_only_arguments(parsed)
    sanitized.pop("command", None)
    return sanitized


# ── flag extraction ───────────────────────────────────────────────────────────

_MACRO_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MSVC_INLINE_DEFINE_RE = re.compile(r"[A-Z_][A-Z0-9_]*(=.*?)?")
_DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")


def _looks_like_define_name(token: str) -> bool:
    """A bare ``-D <token>`` value: NAME or NAME=VALUE, never a path/flag."""
    if not token or token.startswith("-"):
        return False
    if "/" in token or "\\" in token:
        return False
    name = token.split("=", 1)[0]
    return bool(_MACRO_NAME_RE.fullmatch(name))


def _looks_like_std_value(token: str) -> bool:
    return bool(token) and "/" not in token and "\\" not in token and " " not in token


def _msvc_style(compiler_basename: str | None) -> bool:
    if not compiler_basename:
        return False
    return compiler_basename.lower() in _MSVC_STYLE_COMPILERS


def _extract_flags(arguments: list[str]) -> tuple[str | None, str | None, list[str], list[str]]:
    """Return (compiler, standard, defines, include_dirs) for one TU.

    Understands the clang/GCC spellings (``-D X``, ``-DX``, ``-D X=Y``,
    ``-I dir``, ``-Idir``, ``-isystem dir``, ``-std=...``) and the MSVC
    spellings (``/D``, ``/DX``, ``/I``, ``/Idir``, ``/std:c++20``,
    ``/external:I dir``, case-insensitive).  Slash-prefixed inline forms are
    only trusted when the entry's compiler is MSVC-flavoured (``cl.exe`` /
    ``clang-cl``) so POSIX include paths such as ``/include/x.h`` are not
    misread as ``/I`` flags; the unambiguous separate-token forms
    (``/D`` + name, ``/I`` + dir) are honoured for every compiler.  Order is
    preserved and duplicates are dropped (defines by user-visible spelling,
    include dirs by their resolved normalized path).
    """
    compiler = Path(arguments[0]).name if arguments else None
    msvc = _msvc_style(compiler)
    defines: list[str] = []
    includes: list[str] = []
    standard: str | None = None

    def add_define(token: str | None) -> None:
        if token and token not in defines:
            defines.append(token)

    def add_include(token: str | None) -> None:
        if token and token not in includes:
            includes.append(token)

    i, n = 0, len(arguments)
    while i < n:
        a = arguments[i]
        nxt = arguments[i + 1] if i + 1 < n else None
        consumed = 1
        low = a.lower()
        # ── clang/GCC dash spellings (always) ──
        if a == "-D" and nxt is not None and _looks_like_define_name(nxt):
            add_define(nxt)
            consumed = 2
        elif a.startswith("-D") and len(a) > 2:
            add_define(a[2:])
        elif a == "-I" and nxt is not None and not nxt.startswith("-"):
            add_include(nxt)
            consumed = 2
        elif a.startswith("-I") and len(a) > 2:
            add_include(a[2:])
        elif a == "-isystem" and nxt is not None and not nxt.startswith("-"):
            add_include(nxt)
            consumed = 2
        elif a.startswith("-std=") or a.startswith("--std="):
            value = a[a.index("std=") + 4:]
            if value and not standard:
                standard = value.lower()
        # ── MSVC spellings ──
        elif low.startswith("/external:i"):
            inline = a[len("/external:i"):]
            if a == "/external:I" or low == "/external:i":
                if nxt is not None:
                    add_include(nxt)
                    consumed = 2
            elif inline:
                add_include(inline)
        elif a == "/D" and nxt is not None and _looks_like_define_name(nxt):
            add_define(nxt)
            consumed = 2
        elif msvc and len(a) > 2 and a[0] == "/" and low[1] == "d" \
                and _looks_like_define_name(a[2:]):
            add_define(a[2:])
        elif not msvc and len(a) > 2 and a[0] == "/" and low[1] == "d":
            # "/DNAME=V" with an unambiguous all-caps macro spelling.
            inline = a[2:]
            if _MSVC_INLINE_DEFINE_RE.fullmatch(inline):
                add_define(inline)
        elif a == "/I" and nxt is not None and not nxt.startswith("-"):
            add_include(nxt)
            consumed = 2
        elif len(a) > 2 and a[0] == "/" and low[1] == "i" and not a[2:].startswith("/"):
            inline = a[2:]
            if msvc or _DRIVE_RE.match(inline):
                add_include(inline)
        elif low.startswith("/std:") and _looks_like_std_value(a[len("/std:"):]):
            value = a[len("/std:") :]
            if value and not standard:
                standard = value.lower()
        i += consumed
    return compiler, standard, defines, includes


# ── compile database ──────────────────────────────────────────────────────────

@dataclass
class CompileInfo:
    """Per-source compile flags extracted from one database entry."""

    path: str
    in_compile_db: bool
    compiler: str | None = None
    standard: str | None = None
    defines: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    note: str | None = None  # e.g. "inferred from sibling TU"
    arguments: list[str] = field(default_factory=list)
    sanitized: dict = field(default_factory=dict)

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        """Compact RPC payload: capped lists, root-relative include dirs."""
        defines = self.defines[:RESULT_LIST_CAP]
        include_dirs = [
            _short_dir(inc, root) for inc in self.include_dirs[:RESULT_LIST_CAP]
        ]
        return {
            "found": True,
            "path": _short_file(self.path, root),
            "in_compile_db": self.in_compile_db,
            "compiler": self.compiler,
            "standard": self.standard,
            "defines": defines,
            "include_dirs": include_dirs,
            "truncated": (
                len(self.defines) > RESULT_LIST_CAP
                or len(self.include_dirs) > RESULT_LIST_CAP
            ),
        }


@dataclass
class _Record:
    key: str
    info: CompileInfo


def find_compile_db(root: Path, explicit: str | os.PathLike | None = None) -> Path | None:
    """Locate compile_commands.json: explicit (file or dir) → .vscode →
    build → root. Returns None when nothing usable exists."""
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = root / candidate
        db = candidate if candidate.name == DB_FILENAME else candidate / DB_FILENAME
        return db if db.is_file() else None
    for base in (root / ".vscode", root / "build", root):
        db = base / DB_FILENAME
        if db.is_file():
            return db
    return None


class CompileDatabase:
    """Index of compile_commands.json keyed by resolved source path.

    Public surface used by the RPC handlers and the test suite:
    ``.load(path_or_dir)``, ``.entries``, ``.lookup(path, cwd=None)``, ``.path``.
    A missing / unparseable / non-list database yields an empty index — the
    server keeps running (never raises).
    """

    def __init__(self, root: Path | str = ".", explicit: str | os.PathLike | None = None) -> None:
        self.root = Path(root)
        self.explicit = explicit
        self.path: str | None = None
        self._by_source: dict[str, _Record] = {}
        self._by_dir: dict[str, list[_Record]] = {}

    def load(self, path_or_dir: str | os.PathLike | None = None) -> "CompileDatabase":
        self._by_source.clear()
        self._by_dir.clear()
        self.path = None
        db = Path(path_or_dir) if path_or_dir is not None else find_compile_db(self.root, self.explicit)
        if db is not None:
            db = db if db.name == DB_FILENAME else db / DB_FILENAME
            if not db.is_absolute():
                db = self.root / db
            if not db.is_file():
                db = None
        if db is None:
            _log("kimix_lsp: compile db not found")
            return self
        try:
            raw_entries = json.loads(db.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            _log(f"kimix_lsp: compile db unparseable ({exc}); continuing without it")
            return self
        if not isinstance(raw_entries, list):
            _log("kimix_lsp: compile db is not a JSON list; continuing without it")
            return self
        self.path = str(db)
        db_dir = db.parent
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            record = self._build_record(entry, db_dir)
            if record is None:
                continue
            self._by_source[record.key] = record
            dir_key = _norm_key(Path(record.info.path).parent)
            self._by_dir.setdefault(dir_key, []).append(record)
        return self

    @property
    def entries(self) -> int:
        return len(self._by_source)

    def _build_record(self, entry: dict, db_dir: Path) -> _Record | None:
        file_value = entry.get("file")
        if not isinstance(file_value, str) or not file_value:
            return None
        directory_value = entry.get("directory")
        directory = (
            Path(directory_value)
            if isinstance(directory_value, str) and directory_value
            else db_dir
        )
        if not directory.is_absolute():
            directory = db_dir / directory
        source = Path(file_value)
        if not source.is_absolute():
            source = directory / source
        try:
            resolved = source.resolve()
        except OSError:
            return None

        arguments = entry.get("arguments")
        if not (isinstance(arguments, list) and all(isinstance(a, str) for a in arguments)):
            command = entry.get("command")
            arguments = None
            if isinstance(command, str):
                with contextlib.suppress(ValueError):
                    arguments = _split_compile_command(command)
        if arguments is None:
            arguments = []

        compiler, standard, defines, includes = _extract_flags(list(arguments))
        resolved_includes = []
        seen_inc: set[str] = set()
        for inc in includes:
            p = Path(inc)
            if not p.is_absolute():
                p = directory / p
            cleaned = os.path.normpath(str(p))
            key = _norm_key(cleaned)
            if key not in seen_inc:
                seen_inc.add(key)
                resolved_includes.append(cleaned)

        info = CompileInfo(
            path=str(resolved),
            in_compile_db=True,
            compiler=compiler,
            standard=standard,
            defines=defines,
            include_dirs=resolved_includes,
            arguments=list(arguments),
            sanitized=_syntax_only_entry({**entry, "directory": str(directory)}),
        )
        return _Record(key=_norm_key(resolved), info=info)

    def lookup(self, path: Any, cwd: str | os.PathLike | None = None, root: Path | None = None) -> CompileInfo | None:
        """Resolve ``path`` (relative → cwd or root) and find its flags.

        Sources are matched exactly. Headers (.h/.hpp/…) not in the database
        fall back to the flags of a sibling TU in the same directory, or of a
        TU whose include_dirs contain the header's directory; such results are
        marked ``in_compile_db=False`` with ``note="inferred from sibling TU"``.
        """
        if not isinstance(path, (str, os.PathLike)):
            return None
        p = Path(path)
        if not p.is_absolute():
            p = (Path(cwd) if cwd else (root or self.root)) / p
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        record = self._by_source.get(_norm_key(resolved))
        if record is not None:
            return record.info
        if resolved.suffix.lower() in HEADER_EXTS:
            borrowed = self._borrow_for_header(resolved)
            if borrowed is not None:
                return CompileInfo(
                    path=str(resolved),
                    in_compile_db=False,
                    compiler=borrowed.compiler,
                    standard=borrowed.standard,
                    defines=list(borrowed.defines),
                    include_dirs=list(borrowed.include_dirs),
                    note="inferred from sibling TU",
                )
        return None

    def _borrow_for_header(self, header: Path) -> CompileInfo | None:
        parent_key = _norm_key(header.parent)
        candidates = sorted(
            (r.info for r in self._by_dir.get(parent_key, ())),
            key=lambda info: _norm_key(info.path),
        )
        if candidates:
            return candidates[0]
        for record in sorted(self._by_source.values(), key=lambda r: r.key):
            if any(_norm_key(inc) == parent_key for inc in record.info.include_dirs):
                return record.info
        return None

    def sanitized_entries(self) -> list[dict]:
        """The whole index with warning-as-error flags demoted — written once
        into the temp compile-commands dir handed to clangd."""
        return [record.info.sanitized for record in sorted(self._by_source.values(), key=lambda r: r.key)]


# ── clangd bridge (silently optional) ─────────────────────────────────────────

def _severity(diag: dict) -> int:
    value = diag.get("severity", 1)
    return value if isinstance(value, int) and 1 <= value <= 4 else 1


_SEVERITY_NAMES = ("", "error", "warning", "info", "hint")
# Info/Hint diagnostics are noise for a syntax verdict: never formatted.
MAX_FORMATTED_SEVERITY = 2


def _clip(text: str, limit: int) -> str:
    """One line, whitespace-collapsed, hard-clipped with an ellipsis."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: max(limit - 1, 1)].rstrip() + "…"


def _short_path(path_like: Any, root: Path | None, limit: int | None = None) -> str:
    """Root-relative POSIX spelling when possible, else the given path.

    A leading ``…`` marks a tail-clipped absolute path (kept when the file is
    outside the project root, where the suffix is the identifying part).
    """
    text = str(path_like)
    if root is not None:
        try:
            text = Path(text).relative_to(root).as_posix()
        except (ValueError, OSError):
            pass
    if limit is not None and len(text) > limit:
        return "…" + text[-(limit - 1):]
    return text


def _short_file(path_like: Any, root: Path | None) -> str:
    return _short_path(path_like, root)


def _short_dir(path_like: Any, root: Path | None) -> str:
    return _short_path(path_like, root, MAX_INLINE_PATH_CHARS)


def format_diagnostic(diag: dict) -> str:
    """One terse line: ``L<line>:<col> <severity>: <message> [<code>]``.

    Positions are 1-based; the message is whitespace-collapsed and clipped to
    :data:`MAX_DIAG_CHARS` so a single verbose clangd note cannot flood a tool
    result.
    """
    start = ((diag.get("range") or {}).get("start") or {})
    line = start.get("line", 0)
    col = start.get("character", 0)
    if not isinstance(line, int):
        line = 0
    if not isinstance(col, int):
        col = 0
    severity_str = _SEVERITY_NAMES[_severity(diag)]
    message = _clip(diag.get("message", ""), MAX_DIAG_CHARS)
    code = diag.get("code", "")
    text = f"L{line + 1}:{col + 1} {severity_str}: {message}"
    if code not in (None, ""):
        text += f" [{code}]"
    return _clip(text, MAX_DIAG_CHARS + 32)


def _format_all(diagnostics: list[dict]) -> list[str]:
    """Errors and warnings only, in source order, capped at MAX_FORMATTED."""
    kept = [
        d for d in diagnostics
        if isinstance(d, dict) and _severity(d) <= MAX_FORMATTED_SEVERITY
    ]
    formatted = [format_diagnostic(d) for d in kept]
    if len(formatted) > MAX_FORMATTED:
        extra = len(formatted) - MAX_FORMATTED
        formatted = formatted[:MAX_FORMATTED] + [f"… (+{extra} more)"]
    return formatted


def _read_lsp_message(pipe: Any) -> dict | None:
    """Read one Content-Length framed LSP message from a binary pipe."""
    headers: list[bytes] = []
    while True:
        byte = pipe.read(1)
        if not byte:
            return None
        if byte == b"\n":
            if headers and headers[-1].strip() == b"":
                break
            continue
        line = byte
        while not line.endswith(b"\n"):
            more = pipe.read(1)
            if not more:
                return None
            line += more
        headers.append(line.strip())
        if line.strip() == b"":
            break
    length = 0
    for header in headers:
        low = header.lower()
        if low.startswith(b"content-length:"):
            with contextlib.suppress(ValueError):
                length = int(header.split(b":", 1)[1].strip())
    if length <= 0:
        return None
    body = pipe.read(length)
    if len(body) < length:
        return None
    try:
        parsed = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


class ClangdChecker:
    """Persistent lazily-started clangd child over stdio (Content-Length framing).

    The whole check is synchronous and runs inside ``asyncio.to_thread`` from
    the RPC handlers; a single ``threading.Lock`` serialises checks. Any
    failure (spawn, EOF, broken pipe, timeout) degrades to
    ``checked=False`` + reason, killing the child so the next call restarts
    it. Never raises to the caller.
    """

    def __init__(self, clangd_path: str | None, db: CompileDatabase, timeout: float = 4.0, verbose: bool = False) -> None:
        self.clangd_path = clangd_path
        self.db = db
        self.timeout = float(timeout)
        self.verbose = verbose
        self.enabled = clangd_path is not None
        self.unavailable_reason = "clangd not available"
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._messages: queue.Queue | None = None
        self._reader: threading.Thread | None = None
        self._req_id = 0
        self._db_dir: str | None = None

    # ── process plumbing ──
    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        """Launch the clangd child; tests monkeypatch this hook to run a stub
        interpreter through the same LSP plumbing."""
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _write_temp_db(self) -> str:
        if self._db_dir is None or not Path(self._db_dir).is_dir():
            self._db_dir = tempfile.mkdtemp(prefix="kimix-lsp-cc-")
        (Path(self._db_dir) / DB_FILENAME).write_bytes(_json_bytes(self.db.sanitized_entries()))
        return self._db_dir

    def _start_reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        messages: queue.Queue = queue.Queue()

        def _loop() -> None:
            try:
                while True:
                    msg = _read_lsp_message(self._proc.stdout)
                    if msg is None:
                        break
                    messages.put(msg)
            except Exception as exc:  # reader crash → surface to waiters
                messages.put(exc)
            finally:
                messages.put(None)

        self._messages = messages
        self._reader = threading.Thread(target=_loop, name="kimix-lsp-clangd-reader", daemon=True)
        self._reader.start()

    def invalidate(self) -> None:
        """Kill the current child WITHOUT taking the lock (an outer asyncio
        timeout uses this to unblock a worker thread stuck in ``check``)."""
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            if self._proc is not None and self._proc.poll() is None:
                self._send({"jsonrpc": "2.0", "id": -1, "method": "shutdown", "params": None})
                self._send({"jsonrpc": "2.0", "method": "exit", "params": None})
                self._proc.wait(timeout=1)
        self.invalidate()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        self._messages = None
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2)
        if self._reader is not None:
            with contextlib.suppress(Exception):
                self._reader.join(timeout=1)
            self._reader = None

    def _send(self, message: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        body = _json_bytes(message)
        self._proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict) -> int:
        self._req_id += 1
        message = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params}
        if self.verbose:
            _vlog(f"clangd --> {method}")
        self._send(message)
        return self._req_id

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _next_message(self, timeout: float) -> dict:
        assert self._messages is not None
        try:
            message = self._messages.get(timeout=max(timeout, 0.001))
        except queue.Empty:
            raise TimeoutError("timed out waiting for clangd response") from None
        if message is None:
            code = self._proc.poll() if self._proc is not None else None
            raise RuntimeError(f"clangd exited unexpectedly (exit code {code})")
        if isinstance(message, Exception):
            raise RuntimeError(f"failed to read clangd response: {message}") from message
        return message

    def _ensure_started(self, deadline: float) -> None:
        if self._proc is not None and self._proc.poll() is None and self._messages is not None:
            return
        self._kill()
        db_dir = self._write_temp_db()
        cmd = [
            self.clangd_path,
            f"--compile-commands-dir={db_dir}",
            "--log=error",
            "--background-index=false",
            "--pch-storage=memory",
            "--limit-results=0",
            "--clang-tidy=false",
        ]
        if self.verbose:
            _vlog(f"starting clangd: {cmd}")
        try:
            self._proc = self._spawn(cmd)
        except OSError as exc:
            self.enabled = False
            self.unavailable_reason = f"clangd failed to start: {exc}"
            raise RuntimeError(self.unavailable_reason) from exc
        self._start_reader()
        root_uri = Path(db_dir).resolve().as_uri()
        req_id = self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "capabilities": {
                    "textDocument": {
                        # declare pull support or clangd answers
                        # textDocument/diagnostic with "method not found"
                        "diagnostic": {"dynamicRegistration": False,
                                        "versionSupport": False},
                        "publishDiagnostics": {"relatedInformation": False},
                    }
                },
                "workspaceFolders": [{"uri": root_uri, "name": Path(db_dir).name}],
            },
        )
        while True:
            msg = self._next_message(deadline - time.monotonic())
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"clangd initialization failed: {msg['error']}")
                break
        self._notify("initialized", {})

    # ── public check ──
    def check(self, resolved_path: Path, content: str | None, timeout: float) -> dict[str, Any]:
        """Synchronous clangd round trip. Returns the check_syntax core dict."""
        result: dict[str, Any] = {"checked": False, "errors": 0, "warnings": 0, "formatted": [], "reason": None}
        if not self.enabled or not self.clangd_path:
            result["reason"] = self.unavailable_reason or "clangd not available"
            return result
        with self._lock:
            deadline = time.monotonic() + max(timeout, 0.05)
            try:
                if content is None:
                    content = Path(resolved_path).read_text(encoding="utf-8", errors="replace")
                self._ensure_started(deadline)
                # drop stale frames from previous checks: clangd answers
                # didClose with an empty publishDiagnostics for the same URI,
                # which would otherwise masquerade as this check's result
                while True:
                    try:
                        item = self._messages.get_nowait()
                    except queue.Empty:
                        break
                    if item is None or isinstance(item, Exception):
                        self._messages.put(item)   # keep the EOF/error signal
                        break
                uri = Path(resolved_path).as_uri()
                self._notify("textDocument/didOpen", {"textDocument": {
                    "uri": uri, "languageId": "cpp", "version": 1, "text": content,
                }})
                req_id = self._request("textDocument/diagnostic", {
                    "textDocument": {"uri": uri}, "identifier": "kimix-lsp",
                })
                diagnostics: list | None = None
                pull_supported = True
                while diagnostics is None:
                    msg = self._next_message(deadline - time.monotonic())
                    if msg.get("id") == req_id and pull_supported:
                        if "error" in msg:
                            # pull API unavailable (older clangd / missing
                            # capability): fall back to pushed diagnostics
                            pull_supported = False
                        else:
                            got = msg.get("result")
                            diagnostics = got.get("items", []) if isinstance(got, dict) else []
                    elif msg.get("method") == "textDocument/publishDiagnostics":
                        params = msg.get("params") or {}
                        if params.get("uri") == uri:
                            diagnostics = params.get("diagnostics") or []
                with contextlib.suppress(Exception):
                    self._notify("textDocument/didClose", {"textDocument": {"uri": uri}})
                diagnostics = [d for d in diagnostics if isinstance(d, dict)]
                result["checked"] = True
                result["errors"] = sum(1 for d in diagnostics if _severity(d) <= 1)
                result["warnings"] = sum(1 for d in diagnostics if _severity(d) == 2)
                result["formatted"] = _format_all(diagnostics)
            except (TimeoutError, RuntimeError, OSError, ValueError) as exc:
                self._kill()
                result["reason"] = str(exc) or repr(exc)
            except Exception as exc:  # never leak a traceback to the caller
                self._kill()
                result["reason"] = repr(exc)
        return result


# ── application state ─────────────────────────────────────────────────────────

@dataclass
class AppState:
    root: Path
    db: CompileDatabase
    checker: ClangdChecker
    compile_commands: str | None = None

    def reload_db(self) -> None:
        db = CompileDatabase(root=self.root, explicit=self.compile_commands)
        db.load()
        self.db = db
        self.checker.db = db

    def resolve(self, raw: str, cwd: str | None = None) -> Path:
        p = Path(raw)
        if not p.is_absolute():
            p = (Path(cwd) if cwd else self.root) / p
        try:
            return p.resolve()
        except OSError:
            return p

    def rel(self, resolved: Path, raw: str) -> str:
        try:
            return Path(resolved).relative_to(self.root).as_posix()
        except ValueError:
            return raw


_STATE: AppState | None = None


def _state() -> AppState:
    global _STATE
    if _STATE is None:
        root = Path.cwd()
        db = CompileDatabase(root=root)
        _STATE = AppState(root=root, db=db, checker=ClangdChecker(None, db))
    return _STATE


def _is_glob_or_url(raw: str) -> bool:
    if "://" in raw or raw.startswith("data:"):
        return True
    return "*" in raw or "?" in raw


# ── JSON-RPC message handling (§2) ────────────────────────────────────────────

def _sanitize_id(value: Any) -> int | str | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    return value


def _error_body(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": error}


def _require(params: Any, key: str, typ: Any) -> Any:
    if not isinstance(params, dict) or key not in params:
        raise RpcError(INVALID_PARAMS, f"missing required param {key!r}")
    value = params[key]
    if not isinstance(value, typ) or isinstance(value, bool):
        raise RpcError(
            INVALID_PARAMS,
            f"param {key!r} must be {typ.__name__ if isinstance(typ, type) else typ}, got {value!r}",
            repr(value),
        )
    return value


def _optional(params: Any, key: str, typ: Any, default: Any) -> Any:
    if not isinstance(params, dict) or key not in params or params[key] is None:
        return default
    value = params[key]
    if not isinstance(value, typ) or isinstance(value, bool):
        raise RpcError(
            INVALID_PARAMS,
            f"param {key!r} must be {typ.__name__ if isinstance(typ, type) else typ}, got {value!r}",
            repr(value),
        )
    return value


def _as_object(params: Any, method: str) -> dict:
    """Modeled methods accept dict or None (→ {}); arrays / scalars → -32602."""
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise RpcError(INVALID_PARAMS, f"params for {method} must be an object, got {type(params).__name__}",
                       repr(params))
    return params


# ── method implementations (§3 table) ─────────────────────────────────────────

async def rpc_ping(params: Any) -> str:
    """Health check: params ignored entirely, bare-string result."""
    return "pong"


async def rpc_get_compile_info(params: Any) -> dict[str, Any]:
    st = _state()
    p = _as_object(params, "get_compile_info")
    raw = _require(p, "path", str)
    cwd = _optional(p, "cwd", str, None)
    resolved = st.resolve(raw, cwd)
    info = st.db.lookup(resolved, cwd, root=st.root)
    if info is None:
        return {
            "found": False, "path": _short_file(resolved, st.root), "in_compile_db": False,
            "compiler": None, "standard": None, "defines": [], "include_dirs": [],
            "truncated": False,
        }
    return info.to_dict(st.root)


def _numeric(params: Any, key: str, default: float) -> float:
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RpcError(INVALID_PARAMS, f"param {key!r} must be a number, got {value!r}", repr(value))
    if value <= 0:
        return default
    return float(value)


async def rpc_check_syntax(params: Any) -> dict[str, Any]:
    st = _state()
    p = _as_object(params, "check_syntax")
    raw = _require(p, "path", str)
    content = _optional(p, "content", str, None)
    timeout = st.checker.timeout
    if p.get("timeout") is not None:
        timeout = _numeric(p, "timeout", st.checker.timeout)
    resolved = st.resolve(raw)
    info = st.db.lookup(resolved, root=st.root)
    in_db = info is not None and info.in_compile_db
    base = {
        "enabled": st.checker.enabled, "path": _short_file(resolved, st.root),
        "in_compile_db": in_db, "checked": False,
        "errors": 0, "warnings": 0, "formatted": [],
    }
    if not st.checker.enabled:
        return {**base, "reason": st.checker.unavailable_reason or "clangd not available"}
    if info is None:
        # Not a source in the db and no header fallback matched either.
        return {**base, "reason": "not in compile database"}
    # ``not in_db`` with info != None means an inferred header; clangd checks
    # it with the borrowed sibling-TU flags via its nearest-source heuristic.
    return await _run_clangd(st, resolved, base, timeout, content)


async def _run_clangd(st: AppState, resolved: Path, base: dict, timeout: float, content: str | None = None) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        None, functools.partial(st.checker.check, resolved, content, timeout),
    )
    try:
        core = await asyncio.wait_for(future, timeout + 1.0)
    except asyncio.TimeoutError:
        st.checker.invalidate()  # kill child so the worker thread unblocks
        core = {"checked": False, "errors": 0, "warnings": 0, "formatted": [],
                "reason": f"clangd check timed out after {timeout + 1.0:.1f}s"}
    except Exception as exc:  # belt-and-braces; checker.check itself never raises
        core = {"checked": False, "errors": 0, "warnings": 0, "formatted": [],
               "reason": repr(exc)}
    result = {**base, **core}
    result["enabled"] = st.checker.enabled
    return result


def _cap_inline(items: list[str], limit: int) -> str:
    """Comma-joined clip of ``items``: at most ``limit`` entries, each clipped."""
    shown = [_clip(item, limit) for item in items[:MAX_INLINE_ITEMS]]
    text = ", ".join(shown)
    extra = len(items) - MAX_INLINE_ITEMS
    if extra > 0:
        text += f", … (+{extra} more)"
    return text


def _compile_info_message(st: AppState, rel: str, info: CompileInfo) -> str:
    """File meta info in at most three short lines.

    Include dirs are root-relative (absolute paths clipped from the left) and
    both lists are capped at :data:`MAX_INLINE_ITEMS` entries with an overflow
    marker — the counts in the header keep the totals honest.
    """
    inner = ", ".join(part for part in (info.compiler, info.standard, info.note) if part)
    head = f"[lsp] compile info for {rel}"
    if inner:
        head += f" ({inner})"
    head += f": {len(info.defines)} defines, {len(info.include_dirs)} include dirs"
    lines = [head]
    if info.defines:
        lines.append("[lsp] defines: "
                     + _cap_inline(info.defines, MAX_INLINE_DEFINE_CHARS))
    if info.include_dirs:
        dirs = [_short_dir(inc, st.root) for inc in info.include_dirs]
        lines.append("[lsp] include_dirs: " + _cap_inline(dirs, MAX_INLINE_PATH_CHARS))
    return "\n".join(lines)


async def rpc_before_tool(params: Any) -> dict[str, Any]:
    """Connect-bridge hook: never raises; internal failures degrade to "".
    Missing/non-string ``tool`` is the only error (-32602)."""
    st = _state()
    tool = ""
    action = "pre"
    message = ""
    try:
        p = _as_object(params, "before_tool")
        if not isinstance(p.get("tool"), str) or not p.get("tool"):
            raise RpcError(INVALID_PARAMS, "missing required param 'tool'")
        tool = p["tool"]
        if isinstance(p.get("action"), str) and p["action"]:
            action = p["action"]
        cwd = p.get("cwd") if isinstance(p.get("cwd"), str) else None
        raw_paths: list[str] = []
        if isinstance(p.get("path"), str) and p["path"]:
            raw_paths.append(p["path"])
        paths = p.get("paths")
        if isinstance(paths, list):
            for extra in paths:
                if isinstance(extra, str) and extra and extra not in raw_paths:
                    raw_paths.append(extra)
        skipped = max(0, len(raw_paths) - MAX_PATHS_HANDLED)
        chunks: list[str] = []
        for raw in raw_paths[:MAX_PATHS_HANDLED]:
            chunk = await _before_tool_for_path(st, tool, action, raw, cwd)
            if chunk:
                chunks.append(chunk)
        message = "\n".join(chunks)
        if skipped:
            suffix = f"(+{skipped} more files skipped)"
            message = f"{message}\n{suffix}" if message else suffix
    except RpcError:
        raise
    except Exception as exc:  # spec §1.3: internal failures → "no extra info"
        _vlog(f"before_tool degraded: {exc!r}")
        message = ""
    return {"message": message, "tool": tool, "action": action, "handled": True}


async def _before_tool_for_path(st: AppState, tool: str, action: str, raw: str, cwd: str | None) -> str:
    if _is_glob_or_url(raw):
        return ""
    resolved = st.resolve(raw, cwd)
    rel = st.rel(resolved, raw)
    info = st.db.lookup(resolved, cwd, root=st.root)
    is_read = action == "read" or (action == "pre" and tool in ("read", "read_image"))
    if is_read:
        if info is None:
            return ""
        return _compile_info_message(st, rel, info)
    if action in ("write", "edit"):
        if info is None or not info.in_compile_db or not st.checker.enabled:
            return ""  # silent: only in-db files with clangd available
        base = {"enabled": True, "path": _short_file(resolved, st.root), "in_compile_db": True,
                "checked": False, "errors": 0, "warnings": 0, "formatted": []}
        result = await _run_clangd(st, resolved, base, st.checker.timeout)
        if not result.get("checked"):
            return ""
        if result["errors"] == 0 and result["warnings"] == 0:
            return f"[lsp] clangd syntax OK (pre-edit {rel})"
        lines = [f"[lsp] clangd (pre-edit {rel}): "
                 f"{result['errors']} errors, {result['warnings']} warnings"]
        shown = result["formatted"][:MAX_SYNTAX_LINES]
        lines.extend(shown)
        hidden = len(result["formatted"]) - len(shown)
        if hidden > 0:
            lines.append(f"… (+{hidden} more)")
        return "\n".join(lines)
    return ""


async def rpc_reload(params: Any) -> dict[str, Any]:
    _as_object(params, "reload")
    st = _state()
    st.reload_db()
    return {
        "entries": st.db.entries,
        "compile_db": st.db.path,
        "clangd": st.checker.enabled,
    }


METHODS: dict[str, Any] = {
    "ping": rpc_ping,
    "before_tool": rpc_before_tool,
    "get_compile_info": rpc_get_compile_info,
    "check_syntax": rpc_check_syntax,
    "reload": rpc_reload,
}


async def _dispatch_one(msg: Any) -> dict[str, Any] | None:
    """Handle one request/notification; None means "write nothing back"."""
    if not isinstance(msg, dict):
        return _error_body(None, INVALID_REQUEST, "Invalid Request")
    if msg.get("jsonrpc") != "2.0":
        return _error_body(_sanitize_id(msg.get("id")), INVALID_REQUEST, "Invalid Request")
    req_id = _sanitize_id(msg.get("id"))
    is_notification = "id" not in msg
    method = msg.get("method")
    if not isinstance(method, str):
        if is_notification:
            return None
        return _error_body(req_id, INVALID_REQUEST, "Invalid Request")
    params = msg.get("params")
    if params is not None and not isinstance(params, (dict, list)):
        # request-model check (§2.1): params must be object / array / null
        if is_notification:
            return None
        return _error_body(req_id, INVALID_REQUEST, "Invalid Request",
                           "params must be an object, array or null")
    handler = METHODS.get(method)
    if handler is None:
        if is_notification:
            return None
        return _error_body(req_id, METHOD_NOT_FOUND, f"Method not found: {method}")
    try:
        result = await handler(params)
    except RpcError as exc:
        if is_notification:
            return None
        return _error_body(req_id, exc.code, exc.message, exc.data)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # handler crash → -32603 with repr(exc)
        if is_notification:
            return None
        return _error_body(req_id, INTERNAL_ERROR, "Internal error", repr(exc))
    if is_notification:
        return None
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


async def handle_message(raw: bytes | str | dict | list) -> Any:
    """Parse + dispatch one line/body. Returns the response payload
    (dict/list), or None when nothing should be written back.

    A pre-parsed ``dict`` / ``list`` (in-process use, e.g. tests) is dispatched
    directly; ``bytes``/``str`` go through ``json.loads`` (→ -32700 on failure).
    """
    if isinstance(raw, (dict, list)):
        msg = raw
    else:
        try:
            msg = json.loads(raw)
        except Exception as exc:
            return _error_body(None, PARSE_ERROR, "Parse error", str(exc))
    if isinstance(msg, list):
        if not msg:
            return _error_body(None, INVALID_REQUEST, "Invalid Request")
        responses = await asyncio.gather(
            *(_dispatch_one(item) for item in msg), return_exceptions=True
        )
        out: list[dict] = []
        for response in responses:
            if isinstance(response, asyncio.CancelledError):
                raise response
            if isinstance(response, BaseException):
                out.append(_error_body(None, INTERNAL_ERROR, "Internal error", repr(response)))
            elif response is not None:
                out.append(response)
        return out or None
    return await _dispatch_one(msg)


# ── transports (§1) ─────────────────────────────────────────────────────────────

# Live client connections. Shutdown must close these itself: since Python
# 3.12 asyncio's Server.wait_closed() also waits for ACTIVE connections, so a
# lingering bridge connection would otherwise block --stop-file exit forever.
_LIVE_WRITERS: set = set()
_LIVE_TRANSPORTS: set = set()


async def _tcp_client_connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    _LIVE_WRITERS.add(writer)
    try:
        while True:
            try:
                line = await reader.readuntil(b"\n")
            except asyncio.LimitOverrunError:
                # Over-long frame: close without responding (spec §1.1).
                writer.close()
                return
            except (asyncio.IncompleteReadError, ConnectionResetError):
                return
            if not line.strip():
                continue  # blank lines are skipped silently
            try:
                response = await handle_message(line.rstrip(b"\n"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = _error_body(None, INTERNAL_ERROR, "Internal error", repr(exc))
            if response is not None:
                writer.write(_json_bytes(response) + b"\n")
                await writer.drain()
    except (ConnectionResetError, BrokenPipeError, RuntimeError):
        pass
    except Exception:
        pass
    finally:
        _LIVE_WRITERS.discard(writer)
        with contextlib.suppress(Exception):
            writer.close()


async def _process_http(request: bytes, transport: Any) -> None:
    """POST /rpc: same dispatch as TCP, HTTP envelope from §1.2."""
    try:
        head, sep, body = request.partition(b"\r\n\r\n")
        if not sep:
            body = b""
        lines = head.split(b"\r\n")
        request_line = lines[0].decode("latin-1").split() if lines else []
        method = request_line[0] if request_line else ""
        target = request_line[1] if len(request_line) > 1 else ""
        if target != "/rpc":
            status, payload = 404, b'{"error":"not found"}'
        elif method != "POST":
            status, payload = 405, b'{"error":"method not allowed"}'
        elif not body.strip():
            status, payload = 200, _json_bytes(_error_body(None, INVALID_REQUEST, "Invalid Request"))
        else:
            response = await handle_message(body)
            if response is None:
                status, payload = 204, b""
            else:
                status, payload = 200, _json_bytes(response)
        reason = {200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found",
                  405: "Method Not Allowed"}.get(status, "OK")
        envelope = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("latin-1")
        transport.write(envelope + payload)
    except Exception:
        with contextlib.suppress(Exception):
            transport.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
    finally:
        with contextlib.suppress(Exception):
            transport.close()


class _HttpProtocol(asyncio.Protocol):
    """Feed whole request bytes into the coroutine-style handler above."""

    def __init__(self) -> None:
        self._buffer = b""
        self._transport: asyncio.Transport | None = None
        self._expected: int | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]
        _LIVE_TRANSPORTS.add(transport)

    def connection_lost(self, exc: BaseException | None) -> None:
        if self._transport is not None:
            _LIVE_TRANSPORTS.discard(self._transport)
            self._transport = None

    def data_received(self, data: bytes) -> None:
        self._buffer += data
        if self._expected is None:
            marker = self._buffer.find(b"\r\n\r\n")
            if marker < 0:
                return
            header = self._buffer[:marker].decode("latin-1", "replace")
            length = 0
            for line in header.split("\r\n")[1:]:
                key, _, value = line.partition(":")
                if key.strip().lower() == "content-length":
                    with contextlib.suppress(ValueError):
                        length = int(value.strip())
            self._expected = marker + 4 + length
        if len(self._buffer) < self._expected:
            return
        request, self._buffer = self._buffer[: self._expected], self._buffer[self._expected:]
        self._expected = None
        if self._transport is not None:
            asyncio.ensure_future(_process_http(request, self._transport))


# ── lifecycle ─────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    global _STATE
    root = Path(args.root).resolve()
    db = CompileDatabase(root=root, explicit=args.compile_commands)
    db.load()  # logs failures to stderr; never raises
    if args.no_clangd:
        clangd_path = None
        clangd_note = "clangd disabled (--no-clangd)"
    else:
        clangd_path = args.clangd or (shutil.which("clangd") or None)
        clangd_note = (
            f"clangd enabled ({clangd_path})" if clangd_path
            else "clangd disabled (clangd not found on PATH; use --clangd PATH)"
        )
    checker = ClangdChecker(clangd_path, db, timeout=args.timeout, verbose=args.verbose)
    _STATE = AppState(root=root, db=db, checker=checker, compile_commands=args.compile_commands)

    host, port = args.host, args.port
    if args.network == "tcp":
        server = await asyncio.start_server(_tcp_client_connected, host, port, limit=TCP_STREAM_LIMIT)
    else:
        loop = asyncio.get_running_loop()
        server = await loop.create_server(_HttpProtocol, host, port)
    sock = server.sockets[0] if server.sockets else None
    bound_host, bound_port = (sock.getsockname()[:2] if sock else (host, port))

    _log(f"kimix_lsp listening on {args.network}://{bound_host}:{bound_port}")
    _log(f"kimix_lsp: {clangd_note}")
    if db.path:
        _log(f"kimix_lsp: compile db {db.path} ({db.entries} entries)")
    else:
        _log("kimix_lsp: compile db not found")

    stopped = asyncio.Event()

    async def _watch() -> None:
        if args.stop_file:
            stop_file = Path(args.stop_file)
            while not stopped.is_set():
                if stop_file.exists():
                    break
                await asyncio.sleep(0.2)
        else:
            await stopped.wait()
        stopped.set()

    def _close_live_connections() -> None:
        for writer in list(_LIVE_WRITERS):
            with contextlib.suppress(Exception):
                writer.close()
        for transport in list(_LIVE_TRANSPORTS):
            with contextlib.suppress(Exception):
                transport.close()

    watcher = asyncio.create_task(_watch())
    try:
        await stopped.wait()
    finally:
        server.close()
        # close lingering clients FIRST so wait_closed() cannot be held open
        # by an idle bridge connection (BUG 2), then bound the wait anyway
        _close_live_connections()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.wait_closed(), 2.0)
        # last resort: abort whatever remains so handler coroutines unwind
        for writer in list(_LIVE_WRITERS):
            with contextlib.suppress(Exception):
                writer.transport.abort()
        for transport in list(_LIVE_TRANSPORTS):
            with contextlib.suppress(Exception):
                transport.abort()
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        with contextlib.suppress(Exception):
            await asyncio.sleep(0.05)   # let aborted handlers run their finally
        with contextlib.suppress(Exception):
            await asyncio.to_thread(checker.shutdown)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kimix_lsp",
        description="Kimix session-server LSP bridge (JSON-RPC 2.0, stdlib only).",
    )
    parser.add_argument("--network", choices=["tcp", "http"], default="tcp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="Bind port; the client allocates it via <port>.")
    parser.add_argument("--root", default=".", metavar="DIR",
                        help="Project root: base for the compile db lookup and raw bridge paths.")
    parser.add_argument("--stop-file", default=None, metavar="PATH", help="Exit once this file appears.")
    parser.add_argument("--compile-commands", default=None, metavar="PATH",
                        help="Explicit compile_commands.json file or directory containing one.")
    parser.add_argument("--clangd", default=None, metavar="PATH", help="Explicit clangd executable (skips PATH lookup).")
    parser.add_argument("--no-clangd", action="store_true", help="Force the syntax module off.")
    parser.add_argument("--timeout", type=float, default=4.0, help="Per-file clangd syntax budget in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Log extra details to stderr.")
    args = parser.parse_args(argv)
    global _VERBOSE
    _VERBOSE = bool(args.verbose)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
