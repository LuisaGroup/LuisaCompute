#!/usr/bin/env python3
"""Integration tests in a disposable repository; no network or submodule checkout."""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOOD = "4244e2a1ec56dd9c714755453ba840a48a0b5ee5"
BAD = "9a9505f992c02b5e8dea642bceef71a3da46fe30"


class PinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = pathlib.Path(self.tmp.name)
        for folder in ("scripts", ".githooks"):
            (self.repo / folder).mkdir()
        for name in ("check_llvm_downgrade_pin.py", "install_git_hooks.py"):
            shutil.copy(ROOT / "scripts" / name, self.repo / "scripts" / name)
        for name in ("pre-commit", "pre-merge-commit", "pre-push", "run-pin-check"):
            shutil.copy(ROOT / ".githooks" / name, self.repo / ".githooks" / name)
        self.run_cmd("git", "init", "-q")
        self.run_cmd("git", "config", "user.name", "Test")
        self.run_cmd("git", "config", "user.email", "test@example.invalid")
        self.cmake = self.repo / "src/ext/CMakeLists.txt"
        self.cmake.parent.mkdir(parents=True)
        self.cmake.write_text(f'set(LUISA_LLVM_DOWNGRADE_EXPECTED_COMMIT\n "{GOOD}")\n')
        self.run_cmd("git", "add", "src/ext/CMakeLists.txt")
        self.pin(GOOD)
        self.run_cmd("git", "commit", "-qm", "fixture")

    def run_cmd(self, *args, ok=True, data=None):
        result = subprocess.run(args, cwd=self.repo, input=data,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result.stdout.strip()

    def pin(self, sha):
        self.run_cmd("git", "update-index", "--add", "--cacheinfo",
                     f"160000,{sha},src/ext/llvm-downgrade")

    def guard(self, *args, ok=True):
        return self.run_cmd(sys.executable, "scripts/check_llvm_downgrade_pin.py",
                            *args, ok=ok)

    def test_index_and_commit_without_submodule_checkout(self):
        self.guard()
        self.guard("--ref", "HEAD")
        self.pin(BAD)
        self.guard(ok=False)
        self.guard("--ref", "HEAD")

    def test_unstaged_pin_cannot_mask_staged_mismatch(self):
        self.pin(BAD)
        self.cmake.write_text(f'set(LUISA_LLVM_DOWNGRADE_EXPECTED_COMMIT "{BAD}")')
        self.guard(ok=False)
        self.run_cmd("git", "add", "src/ext/CMakeLists.txt")
        self.guard()

    def test_missing_or_non_gitlink_is_rejected(self):
        self.run_cmd("git", "update-index", "--force-remove", "src/ext/llvm-downgrade")
        self.guard(ok=False)
        blob = self.run_cmd("git", "hash-object", "-w", "--stdin", data="not a submodule")
        self.run_cmd("git", "update-index", "--add", "--cacheinfo",
                     f"100644,{blob},src/ext/llvm-downgrade")
        self.guard(ok=False)

    def test_missing_pin_fails_closed(self):
        self.cmake.write_text("# no pin\n")
        self.run_cmd("git", "add", "src/ext/CMakeLists.txt")
        self.guard(ok=False)

    def test_hooks_and_push_tip_not_index(self):
        self.run_cmd(sys.executable, "scripts/install_git_hooks.py")
        self.run_cmd(sys.executable, "scripts/install_git_hooks.py")
        self.pin(BAD)
        for name in ("pre-commit", "pre-merge-commit"):
            self.run_cmd("sh", ".githooks/" + name, ok=False)
        good = self.run_cmd("git", "rev-parse", "HEAD")
        self.run_cmd("git", "-c", "core.hooksPath=/dev/null",
                     "commit", "-qm", "bad fixture")
        bad = self.run_cmd("git", "rev-parse", "HEAD")
        self.pin(GOOD)
        self.run_cmd("sh", ".githooks/pre-push",
                     data=f"refs/heads/a {bad} refs/heads/a {good}\n", ok=False)
        self.run_cmd("sh", ".githooks/pre-push",
                     data=f"refs/heads/a {good} refs/heads/a {bad}\n")
        self.run_cmd("sh", ".githooks/pre-push",
                     data=f"(delete) {'0' * 40} refs/heads/a {bad}\n")

    def test_installer_preserves_existing_setup(self):
        self.run_cmd("git", "config", "--local", "core.hooksPath", "custom-hooks")
        self.run_cmd(sys.executable, "scripts/install_git_hooks.py", ok=False)
        self.assertEqual(self.run_cmd("git", "config", "--get", "core.hooksPath"), "custom-hooks")
        self.run_cmd("git", "config", "--unset", "core.hooksPath")
        (self.repo / ".git/hooks/pre-commit").write_text("# custom")
        self.run_cmd(sys.executable, "scripts/install_git_hooks.py", ok=False)


if __name__ == "__main__":
    unittest.main()
