#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++ Syntax Checker using clangd LSP

This script uses clangd Language Server Protocol to check syntax of C++ files.
It locates a compilation database containing the requested source file so
clangd receives the matching compilation flags.
"""

import argparse
from contextlib import contextmanager
import orjson
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def resolve_executable(command: str) -> str | None:
    """Resolve an executable path using the host platform's PATH rules."""
    resolved = shutil.which(os.path.expanduser(command))
    if resolved is None:
        return None
    return str(Path(resolved).resolve())


class ClangdLSPClient:
    """Minimal LSP client for clangd to get diagnostics."""

    def __init__(
        self,
        clangd_path: str,
        compile_commands_dir: str,
        clang_tidy: bool = False,
        verbose: bool = False,
    ):
        self.clangd_path = clangd_path
        self.compile_commands_dir = compile_commands_dir
        self.clang_tidy = clang_tidy
        self.process = None
        self.request_id = 0
        self.verbose = verbose
        self._messages = queue.Queue()
        self._reader_thread = None

    def start(self):
        """Start clangd process."""
        cmd = [
            self.clangd_path,
            "--compile-commands-dir=" + self.compile_commands_dir,
            "--log=error",
            "--clang-tidy=" + ("true" if self.clang_tidy else "false"),
            "--completion-style=bundled",
            "--pch-storage=memory",
        ]

        if self.verbose:
            print(f"[verbose] Starting clangd: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if self.verbose else subprocess.DEVNULL,
        )
        self._reader_thread = threading.Thread(
            target=self._read_messages,
            name="clangd-lsp-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def stop(self):
        """Stop clangd process."""
        if self.process:
            try:
                if self.process.poll() is None:
                    self._send_request("shutdown", {})
                    self._send_notification("exit", {})
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=2)
            finally:
                if self._reader_thread is not None:
                    self._reader_thread.join(timeout=1)
                    self._reader_thread = None
                self.process = None

    def _send_message(self, message: bytes):
        """Send a message to clangd."""
        header = f"Content-Length: {len(message)}\r\n\r\n".encode()
        if self.verbose:
            print(f"[verbose] LSP --> {message.decode()}")
        self.process.stdin.write(header + message)
        self.process.stdin.flush()

    def _send_request(self, method: str, params: dict) -> int:
        """Send a request to clangd."""
        self.request_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params,
        }
        self._send_message(orjson.dumps(message))
        return self.request_id

    def _send_notification(self, method: str, params: dict):
        """Send a notification to clangd."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._send_message(orjson.dumps(message))

    def _read_message_from_pipe(self) -> dict | None:
        """Read one message from clangd's stdout pipe."""
        # Read header
        header = b""
        while True:
            byte = self.process.stdout.read(1)
            if not byte:
                return None
            header += byte
            if header.endswith(b"\r\n\r\n"):
                break

        # Parse Content-Length
        content_length = 0
        for line in header.decode().split("\r\n"):
            if line.startswith("Content-Length:"):
                content_length = int(line.split(":")[1].strip())
                break

        if content_length == 0:
            return None

        # Read body
        body = self.process.stdout.read(content_length)
        msg = orjson.loads(body)
        if self.verbose:
            print(f"[verbose] LSP <-- {orjson.dumps(msg).decode()}")
        return msg

    def _read_messages(self):
        """Forward blocking pipe reads to a queue so callers can time out."""
        try:
            while True:
                message = self._read_message_from_pipe()
                if message is None:
                    break
                self._messages.put(message)
        except Exception as error:
            self._messages.put(error)
        finally:
            self._messages.put(None)

    def _next_message(self, timeout: float) -> dict:
        """Read the next queued message, failing closed on timeout or EOF."""
        try:
            message = self._messages.get(timeout=max(timeout, 0.0))
        except queue.Empty as error:
            raise TimeoutError(
                "timed out waiting for clangd diagnostics"
            ) from error
        if isinstance(message, Exception):
            raise RuntimeError(f"failed to read clangd response: {message}") \
                from message
        if message is None:
            return_code = (
                None if self.process is None else self.process.poll()
            )
            raise RuntimeError(
                f"clangd exited before producing diagnostics"
                f" (exit code {return_code})"
            )
        return message

    def initialize(self, timeout: float = 10.0):
        """Initialize the LSP connection."""
        root_uri = Path(self.compile_commands_dir).resolve().as_uri()
        if self.verbose:
            print(f"[verbose] Initializing LSP with rootUri: {root_uri}")
        request_id = self._send_request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "capabilities": {},
                "workspaceFolders": [
                    {"uri": root_uri, "name": Path(self.compile_commands_dir).name}
                ],
            },
        )

        deadline = time.monotonic() + timeout
        while True:
            msg = self._next_message(deadline - time.monotonic())
            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                raise RuntimeError(
                    f"clangd initialization failed: {msg['error']}"
                )
            if "result" in msg:
                break

        self._send_notification("initialized", {})

    def open_document(self, file_path: str, content: str):
        """Open a document in clangd."""
        uri = Path(file_path).resolve().as_uri()
        if self.verbose:
            print(f"[verbose] Opening document: {uri}")
        self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "cpp",
                    "version": 1,
                    "text": content,
                }
            },
        )

    def get_diagnostics(self, file_path: str, timeout: float = 10.0) -> list:
        """Wait for clangd's push diagnostics for the requested document."""
        uri = Path(file_path).resolve().as_uri()
        deadline = time.monotonic() + timeout
        if self.verbose:
            print(f"[verbose] Waiting for diagnostics (timeout={timeout}s)...")
        while True:
            msg = self._next_message(deadline - time.monotonic())
            if msg.get("method") == "textDocument/publishDiagnostics":
                params = msg.get("params", {})
                if params.get("uri") == uri:
                    return params.get("diagnostics", [])


def _compile_command_source(database: Path, entry: dict) -> Path | None:
    """Resolve one compilation-database entry's source path."""
    if not isinstance(entry, dict):
        return None
    source_value = entry.get("file")
    if not isinstance(source_value, str) or not source_value:
        return None
    source = Path(source_value)
    if not source.is_absolute():
        directory_value = entry.get("directory")
        directory = (
            Path(directory_value)
            if isinstance(directory_value, str) and directory_value
            else database.parent
        )
        if not directory.is_absolute():
            directory = database.parent / directory
        source = directory / source
    try:
        return source.resolve()
    except OSError:
        return None


def _load_compile_command_entries(database: Path) -> list[dict]:
    try:
        entries = orjson.loads(database.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return []
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _compile_commands_contains(database: Path, file_path: Path) -> bool:
    """Return whether a compilation database contains the requested file."""
    requested = file_path.resolve()
    for entry in _load_compile_command_entries(database):
        if _compile_command_source(database, entry) == requested:
            return True
    return False


def _split_compile_command(command: str) -> list[str]:
    arguments = shlex.split(command, posix=os.name != "nt")
    if os.name == "nt":
        arguments = [
            argument[1:-1]
            if len(argument) >= 2 and argument[0] == argument[-1] == '"'
            else argument
            for argument in arguments
        ]
    return arguments


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


@contextmanager
def syntax_compile_commands(
    compile_commands_dir: str | Path,
    file_path: str | Path,
):
    """Yield a per-TU database with warning-as-error flags demoted."""
    directory = Path(compile_commands_dir).resolve()
    database = directory / "compile_commands.json"
    if not database.is_file():
        yield str(directory)
        return
    requested = Path(file_path).resolve()
    entries = [
        _syntax_only_entry(entry)
        for entry in _load_compile_command_entries(database)
        if _compile_command_source(database, entry) == requested
    ]
    if not entries:
        yield str(directory)
        return
    with tempfile.TemporaryDirectory(
        prefix="luisa-clangd-syntax-"
    ) as temporary_directory:
        temporary_database = (
            Path(temporary_directory) / "compile_commands.json"
        )
        temporary_database.write_bytes(orjson.dumps(entries))
        yield temporary_directory


def load_compile_commands(
    project_root: str | Path = ".",
    file_path: str | Path | None = None,
    explicit_path: str | Path | None = None,
    verbose: bool = False,
) -> str:
    """Find a compilation database directory for the requested source file."""
    root = Path(project_root).resolve()
    if explicit_path is not None:
        candidate = Path(explicit_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        database = (
            candidate
            if candidate.name == "compile_commands.json"
            else candidate / "compile_commands.json"
        )
        if not database.is_file():
            raise FileNotFoundError(
                f"Could not find compile_commands.json at {database}"
            )
        if file_path is not None and not _compile_commands_contains(
            database, Path(file_path)
        ):
            raise FileNotFoundError(
                f"Compilation database does not contain {Path(file_path).resolve()}: {database}"
            )
        if verbose:
            print(f"[verbose] Using compile_commands.json in: {database.parent}")
        return str(database.parent)

    candidate_directories = [root / ".vscode", root / "build", root]
    candidate_directories.extend(
        path for path in sorted(root.glob("build*")) if path.is_dir()
    )
    seen = set()
    for directory in candidate_directories:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        database = directory / "compile_commands.json"
        if not database.is_file():
            continue
        if file_path is not None and not _compile_commands_contains(
            database, Path(file_path)
        ):
            continue
        if verbose:
            print(f"[verbose] Found compile_commands.json in: {directory}")
        return str(directory)

    requested = (
        ""
        if file_path is None
        else f" containing {Path(file_path).resolve()}"
    )
    raise FileNotFoundError(
        f"Could not find compile_commands.json{requested} under {root}"
    )


def format_diagnostic(diag: dict) -> str:
    """Format a diagnostic message."""
    range_info = diag.get("range", {})
    start = range_info.get("start", {})
    line = start.get("line", 0) + 1  # LSP uses 0-based indexing
    character = start.get("character", 0) + 1

    severity = diag.get("severity", 1)
    severity_str = ["Error", "Error", "Warning", "Info", "Hint"][
        min(severity, 4)
    ]

    message = diag.get("message", "")
    code = diag.get("code", "")
    source = diag.get("source", "clangd")

    result = f"{severity_str}: {message}"
    if code:
        result += f" [{code}]"
    result += f" at line {line}, col {character}"

    return result


def check_syntax(
    file_path: str,
    project_root: str | Path = ".",
    clangd_path: str = "clangd",
    compile_commands_path: str | Path | None = None,
    diagnostic_timeout: float = 30.0,
    clang_tidy: bool = False,
    verbose: bool = False,
) -> int:
    """Check syntax of a C++ file using clangd.

    Returns:
        0 if no errors, 1 if errors found, 2 on other failures
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 2

    if verbose:
        print(f"[verbose] Checking file: {file_path}")

    # Find compile_commands.json
    try:
        compile_commands_dir = load_compile_commands(
            project_root,
            file_path=file_path,
            explicit_path=compile_commands_path,
            verbose=verbose,
        )
    except FileNotFoundError as e:
        if compile_commands_path is not None:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        if verbose:
            print(f"[verbose] {e}")
        compile_commands_dir = str(Path(project_root).resolve())

    # Read file content
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 2

    # Create and use clangd client
    try:
        with syntax_compile_commands(
            compile_commands_dir, file_path
        ) as diagnostic_database_dir:
            client = ClangdLSPClient(
                clangd_path,
                diagnostic_database_dir,
                clang_tidy=clang_tidy,
                verbose=verbose,
            )
            try:
                client.start()
                client.initialize(timeout=diagnostic_timeout)
                client.open_document(str(file_path), content)
                diagnostics = client.get_diagnostics(
                    str(file_path), timeout=diagnostic_timeout
                )
            finally:
                client.stop()
        
        if not diagnostics:
            print("[OK] No issues found!")
            return 0

        errors = 0
        warnings = 0

        for diag in diagnostics:
            print(format_diagnostic(diag))
            severity = diag.get("severity", 1)
            if severity <= 1:
                errors += 1
            elif severity == 2:
                warnings += 1

        print("-" * 60)
        print(f"Total: {errors} error(s), {warnings} warning(s)")

        return 1 if errors > 0 else 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def main():
    parser = argparse.ArgumentParser(
        description="Check C++ file syntax using clangd LSP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s myfile.cpp
  %(prog)s --project-root .. src/main.cpp
  %(prog)s --clangd /usr/bin/clangd file.cpp
        """,
    )

    parser.add_argument("file", help="C++ file to check")
    parser.add_argument(
        "--project-root",
        default=Path('.'),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--clangd",
        default="clangd",
        help="Path to clangd executable (default: clangd)",
    )
    parser.add_argument(
        "--compile-commands-dir",
        default=None,
        help="Directory or file for compile_commands.json (overrides auto-detection)",
    )
    parser.add_argument(
        "--diagnostic-timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for initialization and diagnostics (default: 30)",
    )
    parser.add_argument(
        "--clang-tidy",
        action="store_true",
        help="Include clang-tidy diagnostics (disabled for syntax-only checks)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output (show LSP messages and debug info)",
    )

    args = parser.parse_args()

    # Check clangd is available
    clangd_path = args.clangd
    
    # Read clangd.path from .vscode/settings.json if not explicitly provided
    if clangd_path == "clangd":
        settings_path = Path(args.project_root) / ".vscode" / "settings.json"
        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = orjson.loads(f.read())
                config_clangd_path = settings.get("clangd.path")
                if config_clangd_path:
                    # Resolve relative path from project root
                    resolved_path = Path(args.project_root) / config_clangd_path
                    if resolved_path.exists():
                        clangd_path = str(resolved_path.resolve())
                    else:
                        # Try as absolute path
                        config_path = Path(config_clangd_path)
                        if config_path.exists():
                            clangd_path = str(config_path.resolve())
            except (orjson.JSONDecodeError, IOError):
                pass  # Fall back to default behavior
    resolved_clangd_path = resolve_executable(clangd_path)
    if resolved_clangd_path is None:
        print(
            f"Error: clangd not found: {clangd_path}",
            file=sys.stderr,
        )
        print(
            "Please install clangd or provide correct path with --clangd",
            file=sys.stderr,
        )
        sys.exit(2)
    clangd_path = resolved_clangd_path
    exit_code = check_syntax(
        args.file,
        project_root=args.project_root,
        clangd_path=clangd_path,
        compile_commands_path=args.compile_commands_dir,
        diagnostic_timeout=args.diagnostic_timeout,
        clang_tidy=args.clang_tidy,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
