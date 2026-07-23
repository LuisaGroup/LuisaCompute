---
name: git
description: Show uncommitted changes and commit history via git. Use when: (1) inspecting what changed in a file before committing, (2) checking diff of tracked/new/deleted files, (3) reviewing a specific commit's changes, (4) listing files changed in a commit.
---

# Git

Use native, non-interactive git commands to inspect changes. Run them from the
repository root and keep path lists explicit so unrelated user work stays out
of a commit.

## Inspect working-tree changes

```bash
git status --short
git diff --check
git diff --stat
git diff -- <tracked-file> [tracked-file ...]
git diff --no-index -- /dev/null <new-untracked-file>
```

Before committing, inspect every intended tracked file and every intended new
file. Do not stage unrelated untracked files merely because they appear in
`git status`.

## Inspect commits

```bash
git show --stat --oneline <commit>
git show --format=fuller --find-renames <commit> -- <path> [path ...]
git diff <commit>^ <commit> -- <path> [path ...]
```

For a root commit, use `git show <commit>` because `<commit>^` does not exist.
Use `git diff --name-status` when only the changed-file inventory is needed.

## update_sha256 — DX SDK zip update

Update DX SDK zip name and SHA256 across all files that reference it, via `scripts/update_sha256.py`.

```bash
python scripts/update_sha256.py [zip_name]
```

### Behavior

| Arg | Effect |
|-----|--------|
| **zip_name given** | Uses `SDKs/<zip_name>` as the local zip; must exist. |
| **zip_name omitted** | Parses zip name from `LUISA_COMPUTE_DX_SDK` in `scripts/sdks.cmake`. |

Computes SHA256 of the local zip, then updates:

| File | What changes |
|------|-------------|
| `scripts/sdks.cmake` | `LUISA_COMPUTE_DX_SDK` (URL + SHA256 as one CMake list) |
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
