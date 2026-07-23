---
name: lsp
description: clangd LSP over HTTP for C++ syntax checks and symbol navigation in LuisaCompute sources.
---

# C++ LSP

Provides clangd-powered syntax checks and symbol queries over an HTTP bridge.

## Architecture

- **Server** (`scripts/cpp_lsp_server.py`): FastAPI wrapper around `clangd`. Requires `compile_commands.json`.
- **Client** (`scripts/cpp_lsp_client.py`): httpx CLI that talks to the server.

## Workflow

1. **Start the server in the background** before any C++ editing session:
   ```bash
   python scripts/cpp_lsp_server.py --project-root . --port 8000
   ```
   The server auto-discovers `compile_commands.json` in `.vscode/`, `build/`, or project root. If none is found, it falls back to the project root.

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

You can also override file content for symbol queries:
```bash
python scripts/cpp_lsp_client.py symbol src/foo.cpp 10 5 --action hover --content "int x = 0;"
```

### Server Health
```bash
curl http://127.0.0.1:8000/health
```

## Parameters

| Client flag | Default | Description |
|-------------|---------|-------------|
| `--server` | `http://127.0.0.1:8000` | Server base URL |
| `--timeout` | `3.0` | HTTP request timeout (s) |
| `--lsp-timeout` | `10.0` | Internal clangd timeout (s) passed to the server |

| Server flag | Default | Description |
|-------------|---------|-------------|
| `--host` / `--port` | `127.0.0.1:8000` | Bind address |
| `--project-root` | `.` | Where to search for `compile_commands.json` |
| `--clangd` | `clangd` | Path to clangd executable |
| `--compile-commands-dir` | auto | Override `compile_commands.json` location |
| `--verbose` | off | Print LSP traffic |

## Exit Codes

- `check` subcommand: `0` (no issues), `1` (errors found), `2` (connection/server error).
- `symbol` subcommand: `0` (result returned), `1` (no result), `2` (connection/server error).

## Requirements

- `clangd` in PATH
- `compile_commands.json` generated, e.g.:
  - XMake: `xmake project -k compile_commands --lsp=clangd .vscode`
  - CMake: `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
- Python packages: `fastapi`, `uvicorn`, `httpx`, `pydantic`

## Notes

- The server holds a single clangd process. Restart it if `compile_commands.json` changes.
- `line` and `character` are 0-based, matching the LSP protocol and clangd's expectations.
