---
name: lsp
description: clangd LSP over HTTP for C++ syntax checks and symbol navigation in LuisaCompute sources.
---

# C++ LSP

Provides clangd-powered syntax checks and symbol queries over an HTTP bridge.

## Architecture

- **Server** (`scripts/cpp_lsp_server.py`): FastAPI + uvicorn wrapper around one `clangd` subprocess driven over raw JSON-RPC/stdio (no `pygls`). Requires `compile_commands.json`.
  - `GET /health` → `{"status": "ok", "clangd_running": bool}`
  - `POST /check_syntax` → `{file_path, content?, verbose, timeout}`
  - `POST /symbol` → `{file_path, line, character, content?, action, timeout}`
- **Client** (`scripts/cpp_lsp_client.py`): httpx CLI with `check` / `symbol` subcommands that talks to the server.

## Workflow

1. **Start the server in the background** before any C++ editing session (the command blocks until stopped — run it as a background job):
```bash
python scripts/cpp_lsp_server.py --project-root . --port 8000
```
The server auto-discovers `compile_commands.json` in `.vscode/`, `build/`, or project root (first hit wins — `.vscode/` is checked before `build/`), and unlike `scripts/check_cpp_syntax.py` it does **not** scan other `build*` directories. If none is found it falls back to the project root, i.e. clangd runs without per-TU flags.

2. **Use the client** to query information. Common commands:

### Check Syntax
```bash
python scripts/cpp_lsp_client.py check src/foo.cpp -v
# Override content ad-hoc:
python scripts/cpp_lsp_client.py check src/foo.cpp --content "int main() { return 0; }"
```

### Symbol Navigation
```bash
# Go to definition at line 10, character 5 (0-based)
python scripts/cpp_lsp_client.py symbol src/foo.cpp 10 5 --action definition

# Hover info
python scripts/cpp_lsp_client.py symbol src/foo.cpp 10 5 --action hover

# Document-level symbols
python scripts/cpp_lsp_client.py symbol src/foo.cpp --action documentSymbol
```

Available actions: `definition`, `declaration`, `typeDefinition`, `implementation`, `references`, `hover`, `documentSymbol`.

`--content` exists only on `check`; the `symbol` subcommand has no `--content` flag (the `/symbol` endpoint accepts an optional `content` field, reachable only via a direct HTTP POST). `symbol`'s `line`/`character` positionals are optional (default `0`) and ignored by `documentSymbol`.

### Server Health
```bash
curl http://127.0.0.1:8000/health
```

## Parameters

| Client flag | Scope | Default | Description |
|-------------|-------|---------|-------------|
| `--server` | global | `http://127.0.0.1:8000` | Server base URL |
| `--timeout` | global | `3.0` | HTTP request timeout (s) |
| `--lsp-timeout` | subcommand | `10.0` | clangd wait time (s), sent as the body `timeout` |
| `-v`, `--verbose` | `check` | off | Also print Info/Hint diagnostics |
| `--content` | `check` | unset | Check this text instead of the file on disk |
| `line` `character` | `symbol` | `0` `0` | 0-based position (optional; omit for `documentSymbol`) |
| `--action` | `symbol` | `definition` | One of the 7 actions listed above |

`--server`/`--timeout` belong to the top-level parser and must come **before** the subcommand; `--lsp-timeout`, `-v`, `--content`, `--action` come after it.

| Server flag | Default | Description |
|-------------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Bind port |
| `--project-root` | `.` | Where to search for `compile_commands.json` |
| `--clangd` | `clangd` | Path to clangd executable (no pre-check, no `settings.json` lookup) |
| `--compile-commands-dir` | auto | Directory containing `compile_commands.json` — a directory, not a file path |
| `--verbose` | off | Print LSP traffic and raise uvicorn log level to `info` |

## Exit Codes

- `check` subcommand: `0` (no errors — warnings alone still exit `0`), `1` (at least one error), `2` (connection/HTTP/server error: unreachable server, missing file without `--content` → HTTP 404).
- `symbol` subcommand: `0` (result returned), `1` (no result: clangd returned `null` or the LSP call timed out), `2` (connection/server error). An action outside the 7 above is rejected by the client's argparse; a direct HTTP call gets 400.
- Same 0/1/2 semantics as `scripts/check_cpp_syntax.py` (0 clean, 1 errors, 2 tool failure).

## Requirements

- `clangd` resolvable through PATH. The server passes `--clangd` verbatim to `subprocess.Popen`; it does **not** read `clangd.path` from `.vscode/settings.json` the way `scripts/check_cpp_syntax.py` does, and does not verify the binary first — give `--clangd <path>` explicitly when clangd is not on PATH.
- `compile_commands.json` generated, e.g.:
  - XMake: `xmake project -k compile_commands --lsp=clangd .vscode` (`.vscode` is the server's first search dir)
  - CMake: already forced by the root `CMakeLists.txt` (`set(CMAKE_EXPORT_COMPILE_COMMANDS ON)`), so a normal CMake configure writes `build/compile_commands.json`
- Python packages: `fastapi`, `uvicorn`, `httpx`, `pydantic` (the scripts' only third-party imports; not declared in `pyproject.toml`, install them separately)

## Notes

- The server holds a single clangd process. Restart it if `compile_commands.json` changes — there is no idle timeout and no hot-reload; stop it with Ctrl+C (the lifespan hook sends LSP `shutdown`/`exit` and kills clangd after a 2 s grace).
- Every clangd call is serialized behind one `asyncio.Lock`, and each endpoint sleeps first (`0.5 s` before diagnostics, `0.3 s` before a symbol query), so concurrent client calls queue instead of running in parallel.
- The server always starts clangd with `--clang-tidy=true` (plus `--log=error --completion-style=bundled --pch-storage=memory`), unlike `check_cpp_syntax.py`, where clang-tidy is off unless `--clang-tidy`.
- `line` and `character` in `symbol` requests are 0-based, matching the LSP protocol; the diagnostic positions `check` prints are converted to 1-based `line`/`col`.
