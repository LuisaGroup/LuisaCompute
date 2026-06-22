---
name: git
description: Show uncommitted changes and commit history via git. Use when: (1) inspecting what changed in a file before committing, (2) checking diff of tracked/new/deleted files, (3) reviewing a specific commit's changes, (4) listing files changed in a commit.
---

# Git

Two scripts for inspecting changes at file and commit level.

## git_diff — Per-file diff

Show diff information for given files via `scripts/git_diff.py`.

```bash
python scripts/git_diff.py <file> [file ...]
```

### Behavior

| File Status | Output |
|-------------|--------|
| **Tracked, uncommitted** | Staged diff first, then working-tree diff; last-commit diff between `HEAD~` and `HEAD` |
| **New (untracked)** | Prints full file content |
| **Deleted (not on disk)** | Shows diff from `HEAD` if previously tracked; otherwise notes it was never tracked |

### Examples

```bash
# Check a modified source file
python scripts/git_diff.py src/xir/passes/restructure_cfg.cpp

# Check multiple files
python scripts/git_diff.py src/foo.cpp include/bar.h scripts/baz.py

# Check a new untracked file
python scripts/git_diff.py src/new_file.cpp
```

### Notes

- Operates from the project root (parent of `scripts/`).
- Binary files are flagged with their size instead of printing content.
- Empty diff sections are omitted.

## git_commit — Per-commit diff

Show changed files and their diffs for a specific commit via `scripts/git_commit.py`.

```bash
python scripts/git_commit.py <commit-hash>
```

### Behavior

| Category | Output |
|----------|--------|
| **Created files** | List of `+ <path>` entries |
| **Deleted files** | List of `- <path>` entries |
| **Updated files** | Full diff per file from `commit^..commit` (or `--root` for initial commit) |

### Examples

```bash
# Inspect the last commit
python scripts/git_commit.py HEAD

# Inspect a specific commit
python scripts/git_commit.py 97dfeeaab

# Inspect a short hash
python scripts/git_commit.py cba8a5d
```

### Notes

- Exits with code 1 and prints to stderr on invalid commit hash.
- Handles root (initial) commits by using `git show` instead of `git diff`.
- Renamed files are listed under "updated" and show the new path.

## update_sha256 — DX SDK zip update

Update DX SDK zip name and SHA256 across all files that reference it, via `scripts/update_sha256.py`.

```bash
python scripts/update_sha256.py [zip_name]
```

### Behavior

| Arg | Effect |
|-----|--------|
| **zip_name given** | Uses `SDKs/<zip_name>` as the local zip; must exist. |
| **zip_name omitted** | Parses zip name from `LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL` in `src/backends/dx/CMakeLists.txt`. |

Computes SHA256 of the local zip, then updates:

| File | What changes |
|------|-------------|
| `src/backends/dx/CMakeLists.txt` | `LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL` + `LUISA_COMPUTE_DX_SDK_SHA256` |
| `scripts/download_sdks.cmake` | URL + SHA256 inside `if (sdk STREQUAL "dx")` block |
| `scripts/find_sdk.lua` | `name` field under `dx_sdk` entry |

### Examples

```bash
# Provide zip name explicitly
python scripts/update_sha256.py dx_sdk_20260622.zip

# Auto-detect from CMakeLists.txt URL
python scripts/update_sha256.py
```

### Notes

- Zip must live under `SDKs/` relative to repo root.
- Skips files not found with a warning (non-fatal).
- No-op if all values are already up-to-date.
