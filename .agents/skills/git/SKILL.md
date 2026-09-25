---
name: git
description: Show uncommitted changes and commit history via git. Use when: (1) inspecting what changed in a file before committing, (2) checking diff of tracked/new/deleted files, (3) reviewing a specific commit's changes, (4) listing files changed in a commit.
---

# Git

Use native, non-interactive git commands to inspect changes. Run them from the
repository root and keep path lists explicit so unrelated user work stays out
of a commit. In an interactive terminal prefix `--no-pager`
(`git --no-pager show <commit>`); piped output is never paged.

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

Non-zero exit codes are data, not failures: `git diff --check` exits 2 when it
finds whitespace errors, and `git diff --no-index` exits 1 when the two sides
differ (0 when identical).

17 vendored libraries under `src/ext/` are git submodules, so a touched one
shows up as ` M src/ext/<name>` (the remaining `src/ext/` entries are ordinary
tracked files/dirs). Read a pointer change with
`git diff --ignore-submodules=none -- src/ext/<name>`, hide these entries with
`git diff --ignore-submodules=all`, list them with `git submodule status`, and
never stage a `src/ext/*` gitlink unless updating it is the task.

## Inspect commits

```bash
git show --stat --oneline <commit>
git show --format=fuller --find-renames <commit> -- <path> [path ...]
git diff <commit>^ <commit> -- <path> [path ...]
```

For a root commit, use `git show <commit>` because `<commit>^` does not exist.
Use `git diff --name-status` when only the changed-file inventory is needed.

For a merge commit, plain `git show` prints only the combined diff (files that
differ from *every* parent), which usually looks nearly empty. Count parents with
`git rev-list --parents -n1 <commit>`, then read per-parent changes with
`git show -m --stat --oneline <commit>` or first-parent changes with
`git diff --stat <commit>^1 <commit> -- <path>`.

## update_sha256 — DX SDK zip update

Update the DX SDK zip name and its SHA256 pin in the two files that record them,
via `scripts/update_sha256.py` (only `scripts/sdks.cmake` carries the hash).

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
python scripts/update_sha256.py dx_sdk_20260815.zip

# Auto-detect the zip name from scripts/sdks.cmake
python scripts/update_sha256.py
```

### Notes

- Zip must live under `SDKs/` relative to repo root (the parent of `scripts/`).
- Skips files not found with a warning (non-fatal) — see the exit-1 cases below.
- No-op if all values are already up-to-date.
- **Write recipe, not inspection**: it rewrites both files in place, and the URL
  written into `sdks.cmake` is always
  `https://github.com/LuisaGroup/SDKs/releases/download/sdk/<zip_name>`.
- DX SDK only: the `vk_sdk` entry in `scripts/find_sdk.lua` and
  `LUISA_COMPUTE_VK_SDK_DOWNLOAD_URL` / `LUISA_COMPUTE_VK_SDK_SHA256` in
  `src/backends/vk/CMakeLists.txt` are left untouched and must be edited by hand.
- There is no `--dry-run`; only the first argument is read, so trailing arguments
  are silently ignored (`--dry-run` alone fails as an invalid zip name).
- A missing `SDKs/<zip>`, a name without the `.zip` suffix, or an unparsable
  `LUISA_COMPUTE_DX_SDK` prints `[ERROR] ...` and exits 1. The non-fatal `[WARN]`
  skip only applies to a missing `scripts/find_sdk.lua` (or a missing
  `scripts/sdks.cmake` when a zip name is given); with no argument a missing
  `scripts/sdks.cmake` is fatal.
