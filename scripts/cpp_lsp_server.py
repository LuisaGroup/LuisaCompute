#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++ LSP Server using clangd over HTTP.

Loads compile_commands.json, starts clangd, and exposes HTTP endpoints
for syntax checking and symbol queries.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Fix Windows encoding issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Helpers (mirrored from check_cpp_syntax.py)
# ---------------------------------------------------------------------------

def load_compile_commands(project_root: str = ".", verbose: bool = False) -> str:
    """Find and validate compile_commands.json location."""
    root = Path(project_root).resolve()
    candidates = [root / ".vscode", root / "build", root]
    for d in candidates:
        cc = d / "compile_commands.json"
        if cc.exists():
            if verbose:
                print(f"[verbose] Found compile_commands.json in: {d}")
            return str(d)
    raise FileNotFoundError(
        "Could not find compile_commands.json in .vscode, build, or project root"
    )


def format_diagnostic(diag: dict) -> str:
    """Format a diagnostic message for human reading."""
    range_info = diag.get("range", {})
    start = range_info.get("start", {})
    line = start.get("line", 0) + 1  # LSP uses 0-based indexing
    character = start.get("character", 0) + 1
    severity = diag.get("severity", 1)
    severity_str = ["Error", "Error", "Warning", "Info", "Hint"][min(severity, 4)]
    message = diag.get("message", "")
    code = diag.get("code", "")
    result = f"{severity_str}: {message}"
    if code:
        result += f" [{code}]"
    result += f" at line {line}, col {character}"
    return result


# ---------------------------------------------------------------------------
# Clangd LSP wrapper
# ---------------------------------------------------------------------------

class ClangdProcess:
    """Low-level clangd wrapper with synchronous JSON-RPC over stdio."""

    def __init__(self, clangd_path: str, compile_commands_dir: str, verbose: bool = False):
        self.clangd_path = clangd_path
        self.compile_commands_dir = compile_commands_dir
        self.process: subprocess.Popen | None = None
        self.lock = asyncio.Lock()
        self.request_id = 0
        self.verbose = verbose

    def start(self):
        cmd = [
            self.clangd_path,
            f"--compile-commands-dir={self.compile_commands_dir}",
            "--log=error",
            "--clang-tidy=true",
            "--completion-style=bundled",
            "--pch-storage=memory",
        ]
        if self.verbose:
            print(f"[verbose] Starting clangd: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr=subprocess.DEVNULL,
        )

    def stop(self):
        if self.process:
            try:
                self._send_request_raw("shutdown", {})
                self._send_notification_raw("exit", {})
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
            finally:
                self.process = None

    def _send_message(self, message: bytes):
        header = f"Content-Length: {len(message)}\r\n\r\n".encode()
        if self.verbose:
            print(f"[verbose] LSP --> {message.decode()}")
        self.process.stdin.write(header + message)
        self.process.stdin.flush()

    def _send_request_raw(self, method: str, params: dict) -> int:
        self.request_id += 1
        req_id = self.request_id
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self._send_message(json.dumps(msg).encode())
        return req_id

    def _send_notification_raw(self, method: str, params: dict):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._send_message(json.dumps(msg).encode())

    def _read_message(self) -> dict | None:
        header = b""
        while True:
            byte = self.process.stdout.read(1)
            if not byte:
                return None
            header += byte
            if header.endswith(b"\r\n\r\n"):
                break
        content_length = 0
        for line in header.decode().split("\r\n"):
            if line.startswith("Content-Length:"):
                content_length = int(line.split(":", 1)[1].strip())
                break
        if content_length == 0:
            return None
        body = self.process.stdout.read(content_length)
        msg = json.loads(body)
        if self.verbose:
            print(f"[verbose] LSP <-- {json.dumps(msg)}")
        return msg

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def initialize(self, args):
        root_uri = Path(self.compile_commands_dir).resolve().as_uri()
        if self.verbose:
            print(f"[verbose] Initializing LSP with rootUri: {root_uri}")
        req_id = self._send_request_raw(
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
        while True:
            msg = self._read_message()
            if msg is None:
                if self.process.poll() is not None:
                    raise RuntimeError("clangd process exited unexpectedly during initialization")
                continue
            if msg.get("id") == req_id and "result" in msg:
                break
        self._send_notification_raw("initialized", {})

    # -----------------------------------------------------------------------
    # Document helpers
    # -----------------------------------------------------------------------

    def open_document(self, file_uri: str, content: str):
        self._send_notification_raw(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": file_uri,
                    "languageId": "cpp",
                    "version": 1,
                    "text": content,
                }
            },
        )

    def close_document(self, file_uri: str):
        self._send_notification_raw(
            "textDocument/didClose",
            {"textDocument": {"uri": file_uri}},
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_diagnostics(self, file_uri: str, timeout: float = 10.0) -> list:
        req_id = self._send_request_raw(
            "textDocument/diagnostic",
            {"textDocument": {"uri": file_uri}, "identifier": "syntax-check"},
        )
        start = time.time()
        while time.time() - start < timeout:
            msg = self._read_message()
            if msg is None:
                break
            if msg.get("id") == req_id and "result" in msg:
                result = msg["result"]
                if isinstance(result, dict) and "items" in result:
                    return result["items"]
                return []
            if msg.get("method") == "textDocument/publishDiagnostics":
                params = msg.get("params", {})
                if params.get("uri") == file_uri:
                    return params.get("diagnostics", [])
        return []

    # -----------------------------------------------------------------------
    # Generic symbol request
    # -----------------------------------------------------------------------

    def request_symbol(self, method: str, params: dict, timeout: float = 10.0):
        req_id = self._send_request_raw(method, params)
        start = time.time()
        while time.time() - start < timeout:
            msg = self._read_message()
            if msg is None:
                break
            if msg.get("id") == req_id and "result" in msg:
                return msg["result"]
        return None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global clangd_process
    if clangd_process:
        await asyncio.to_thread(clangd_process.stop)


app = FastAPI(title="C++ LSP Server", lifespan=lifespan)


class CheckSyntaxRequest(BaseModel):
    file_path: str
    content: str | None = None
    verbose: bool = False
    timeout: float = 10.0


class SymbolRequest(BaseModel):
    file_path: str
    line: int = Field(..., description="0-based line number")
    character: int = Field(..., description="0-based character number")
    content: str | None = None
    action: str = Field(
        "definition",
        description=(
            "One of: definition, declaration, typeDefinition, implementation, "
            "references, hover, documentSymbol"
        ),
    )
    timeout: float = 10.0


# Global clangd handle – populated in main() before uvicorn starts.
clangd_process: ClangdProcess | None = None


@app.get("/health")
async def health():
    running = (
        clangd_process.process.poll() is None
        if (clangd_process and clangd_process.process)
        else False
    )
    return {"status": "ok", "clangd_running": running}


@app.post("/check_syntax")
async def check_syntax(req: CheckSyntaxRequest):
    if clangd_process is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    file_path = Path(req.file_path).resolve()
    if not file_path.exists() and req.content is None:
        raise HTTPException(status_code=404, detail="File not found and no content provided")

    content = req.content if req.content is not None else file_path.read_text(encoding="utf-8")
    file_uri = file_path.as_uri()

    async with clangd_process.lock:
        await asyncio.to_thread(clangd_process.open_document, file_uri, content)

    await asyncio.sleep(0.5)

    async with clangd_process.lock:
        diagnostics = await asyncio.to_thread(
            clangd_process.get_diagnostics, file_uri, req.timeout
        )
        await asyncio.to_thread(clangd_process.close_document, file_uri)

    errors = sum(1 for d in diagnostics if d.get("severity", 1) <= 1)
    warnings = sum(1 for d in diagnostics if d.get("severity", 1) == 2)
    formatted = [format_diagnostic(d) for d in diagnostics]

    return {
        "success": True,
        "diagnostics": diagnostics,
        "errors": errors,
        "warnings": warnings,
        "formatted": formatted if (req.verbose or errors or warnings) else None,
    }


@app.post("/symbol")
async def symbol(req: SymbolRequest):
    if clangd_process is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    file_path = Path(req.file_path).resolve()
    if not file_path.exists() and req.content is None:
        raise HTTPException(status_code=404, detail="File not found and no content provided")

    content = req.content if req.content is not None else file_path.read_text(encoding="utf-8")
    file_uri = file_path.as_uri()

    async with clangd_process.lock:
        await asyncio.to_thread(clangd_process.open_document, file_uri, content)

    await asyncio.sleep(0.3)

    result = None
    try:
        if req.action == "documentSymbol":
            async with clangd_process.lock:
                result = await asyncio.to_thread(
                    clangd_process.request_symbol,
                    "textDocument/documentSymbol",
                    {"textDocument": {"uri": file_uri}},
                    req.timeout,
                )
        elif req.action in (
            "definition",
            "declaration",
            "typeDefinition",
            "implementation",
            "references",
            "hover",
        ):
            params = {
                "textDocument": {"uri": file_uri},
                "position": {"line": req.line, "character": req.character},
            }
            if req.action == "references":
                params["context"] = {"includeDeclaration": True}
            async with clangd_process.lock:
                result = await asyncio.to_thread(
                    clangd_process.request_symbol,
                    f"textDocument/{req.action}",
                    params,
                    req.timeout,
                )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {req.action}")
    finally:
        async with clangd_process.lock:
            await asyncio.to_thread(clangd_process.close_document, file_uri)

    return {"success": result is not None, "result": result}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C++ LSP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--clangd", default="clangd", help="Path to clangd executable")
    parser.add_argument(
        "--compile-commands-dir",
        default=None,
        help="Override compile_commands.json directory",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose LSP traffic")
    args = parser.parse_args()

    print("[1/5] Parsing arguments... done")

    compile_commands_dir = args.compile_commands_dir
    if compile_commands_dir is None:
        print("[2/5] Locating compile_commands.json... ", end="", flush=True)
        try:
            compile_commands_dir = load_compile_commands(args.project_root, verbose=args.verbose)
            print(f"found at {compile_commands_dir}")
        except FileNotFoundError:
            compile_commands_dir = args.project_root
            print("not found, falling back to project root")
    else:
        print(f"[2/5] Using provided compile_commands directory: {compile_commands_dir}")

    print("[3/5] Spawning clangd process... ", end="", flush=True)
    global clangd_process
    clangd_process = ClangdProcess(args.clangd, compile_commands_dir, verbose=args.verbose)
    clangd_process.start()
    print("ok")

    print("[4/5] Initializing LSP session... ", end="", flush=True)
    
    clangd_process.initialize(args)
    print("ok")

    print(f"[5/5] Starting HTTP server on http://{args.host}:{args.port}", flush=True)
    print("        Press Ctrl+C to stop\n", flush=True)

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info" if args.verbose else "warning",
        )
    except KeyboardInterrupt:
        print("\n[shutdown] Received interrupt, exiting cleanly.")
        clangd_process.stop()
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[shutdown] Received interrupt, exiting cleanly.")
        sys.exit(0)
