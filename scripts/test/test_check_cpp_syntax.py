#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_all_cpp_syntax import check_file
from check_cpp_syntax import (
    check_syntax,
    load_compile_commands,
    resolve_executable,
)


class ResolveExecutableTest(unittest.TestCase):

    def test_resolves_explicit_executable(self):
        self.assertEqual(
            resolve_executable(sys.executable),
            str(Path(sys.executable).resolve()),
        )

    def test_resolves_executable_from_path(self):
        with tempfile.TemporaryDirectory() as directory:
            name = "clangd-test.exe" if os.name == "nt" else "clangd-test"
            executable = Path(directory) / name
            executable.write_text("", encoding="utf-8")
            executable.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": directory}):
                self.assertEqual(
                    resolve_executable(name),
                    str(executable.resolve()),
                )

    def test_missing_executable_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"PATH": directory}):
                self.assertIsNone(
                    resolve_executable("clangd-that-does-not-exist"))


class LoadCompileCommandsTest(unittest.TestCase):

    @staticmethod
    def _write_database(directory: Path, source: Path):
        directory.mkdir(parents=True)
        (directory / "compile_commands.json").write_text(
            '[{"directory": "' + str(directory) +
            '", "file": "' + str(source) +
            '", "command": "c++ -c source.cpp"}]',
            encoding="utf-8",
        )

    def test_discovers_matching_build_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            source.write_text("", encoding="utf-8")
            self._write_database(root / "build-z", source)
            self.assertEqual(
                load_compile_commands(root, file_path=source),
                str((root / "build-z").resolve()),
            )

    def test_explicit_database_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_compile_commands(
                    root, explicit_path=root / "missing-build"
                )

    def test_explicit_database_failure_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            source.write_text("", encoding="utf-8")
            error = StringIO()
            with redirect_stderr(error):
                result = check_syntax(
                    str(source),
                    project_root=root,
                    clangd_path=sys.executable,
                    compile_commands_path=root / "missing-build",
                )
            self.assertEqual(result, 2)
            self.assertIn("Could not find compile_commands.json", error.getvalue())


class CheckAllSyntaxTest(unittest.TestCase):

    @mock.patch("check_all_cpp_syntax.subprocess.run")
    def test_forwards_explicit_database(self, run):
        run.return_value = mock.Mock(
            returncode=0,
            stdout="",
            stderr="",
        )
        check_file(
            "source.cpp",
            "check_cpp_syntax.py",
            "/project",
            "/usr/bin/clangd",
            "/project/build/compile_commands.json",
        )
        command = run.call_args.args[0]
        self.assertIn("--compile-commands-dir", command)
        self.assertIn("/project/build/compile_commands.json", command)


if __name__ == "__main__":
    unittest.main()
