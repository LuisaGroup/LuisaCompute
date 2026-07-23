#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++ Syntax Checker using clangd LSP

This script uses clangd Language Server Protocol to check syntax of C++ files.
It reads compile_commands.json from .vscode directory for proper compilation flags.
"""

import argparse
import orjson
import os
import subprocess
import sys
import time
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class ClangdLSPClient:
    """Minimal LSP client for clangd to get diagnostics."""

    def __init__(self, clangd_path: str, compile_commands_dir: str, verbose: bool = False):
        self.clangd_path = clangd_path
        self.compile_commands_dir = compile_commands_dir
        self.process = None
        self.request_id = 0
        self.diagnostics = []
        self.verbose = verbose

    def start(self):
        """Start clangd process."""
        cmd = [
            self.clangd_path,
            "--compile-commands-dir=" + self.compile_commands_dir,
            "--log=error",
            "--clang-tidy=true",
            "--completion-style=bundled",
            "--pch-storage=memory",
            "--cross-file-rename=false",
        ]

        if self.verbose:
            print(f"[verbose] Starting clangd: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def stop(self):
        """Stop clangd process."""
        if self.process:
            try:
                self._send_request("shutdown", {})
                self._send_notification("exit", {})
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
            finally:
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

    def _read_message(self) -> dict:
        """Read a message from clangd."""
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

    def initialize(self):
        """Initialize the LSP connection."""
        root_uri = Path(self.compile_commands_dir).resolve().as_uri()
        if self.verbose:
            print(f"[verbose] Initializing LSP with rootUri: {root_uri}")
        self._send_request(
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

        # Wait for initialize response
        while True:
            msg = self._read_message()
            if msg and "id" in msg and msg.get("result"):
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
        """Get diagnostics for a file using textDocument/diagnostic."""
        uri = Path(file_path).resolve().as_uri()
        req_id = self._send_request(
            "textDocument/diagnostic",
            {
                "textDocument": {"uri": uri},
                "identifier": "syntax-check",
            },
        )

        start_time = time.time()
        if self.verbose:
            print(f"[verbose] Waiting for diagnostics (timeout={timeout}s)...")
        while time.time() - start_time < timeout:
            msg = self._read_message()
            if msg is None:
                break

            # Check for diagnostic response
            if msg.get("id") == req_id and "result" in msg:
                result = msg["result"]
                if isinstance(result, dict) and "items" in result:
                    return result["items"]
                return []

            # Check for publishDiagnostics notification
            if msg.get("method") == "textDocument/publishDiagnostics":
                params = msg.get("params", {})
                if params.get("uri") == uri:
                    return params.get("diagnostics", [])

        return []


def load_compile_commands(project_root: str = ".", verbose: bool = False) -> str:
    """Find and validate compile_commands.json location."""
    vscode_dir = Path(project_root) / ".vscode"
    compile_commands = vscode_dir / "compile_commands.json"

    if compile_commands.exists():
        if verbose:
            print(f"[verbose] Found compile_commands.json in: {vscode_dir}")
        return str(vscode_dir)

    # Try build directory
    build_dir = Path(project_root) / "build"
    compile_commands = build_dir / "compile_commands.json"
    if compile_commands.exists():
        if verbose:
            print(f"[verbose] Found compile_commands.json in: {build_dir}")
        return str(build_dir)

    raise FileNotFoundError(
        "Could not find compile_commands.json in .vscode or build directory"
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


def check_syntax(file_path: str, project_root: str = ".", clangd_path: str = "clangd", verbose: bool = False) -> int:
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
        compile_commands_dir = load_compile_commands(project_root, verbose=verbose)
    except FileNotFoundError as e:
        if verbose:
            print(f"[verbose] {e}")
        compile_commands_dir = project_root

    # Read file content
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 2

    # Create and use clangd client
    client = ClangdLSPClient(clangd_path, compile_commands_dir, verbose=verbose)

    try:
        client.start()

        client.initialize()

        client.open_document(str(file_path), content)

        # Wait for clangd to process
        time.sleep(0.5)

        diagnostics = client.get_diagnostics(str(file_path))
        
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
    finally:
        client.stop()


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
        default=".vscode/compile_commands.json",
        help="Directory containing compile_commands.json (overrides auto-detection)",
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
    if not Path(clangd_path).exists():
        # Try to find in PATH
        try:
            result = subprocess.run(
                ["where", clangd_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(
                    f"Error: clangd not found: {clangd_path}",
                    file=sys.stderr,
                )
                print(
                    "Please install clangd or provide correct path with --clangd",
                    file=sys.stderr,
                )
                sys.exit(2)
            clangd_path = result.stdout.strip().split("\n")[0].strip()
        except Exception as e:
            print(f"Error finding clangd: {e}", file=sys.stderr)
            sys.exit(2)
    exit_code = check_syntax(
        args.file,
        project_root=args.project_root,
        clangd_path=clangd_path,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
