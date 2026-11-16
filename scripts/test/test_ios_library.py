#!/usr/bin/env python3
"""Host-only regression tests for iOS CI orchestration, not iOS compilation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class IOSLibraryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.env = dict(os.environ, PATH=f"{self.path}:{os.environ['PATH']}")
        self.log = self.path / "cmake.log"
        self.env["MOCK_LOG"] = str(self.log)
        self.tool("cmake", 'printf "%s\\n" "$@" >> "$MOCK_LOG"')
        self.tool("xcodebuild", ":")
        self.tool("lipo", "echo arm64")
        self.tool("xcrun", "echo '    platform IOS'")
        self.tool("otool", "printf 'binary:\\n /usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\\n'")
        self.llvm = self.path / "llvm"
        self.llvm.mkdir()
        (self.llvm / "LLVMConfig.cmake").touch()

    def tool(self, name, body):
        path = self.path / name
        path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def run_script(self, script, *args):
        return subprocess.run(["bash", str(ROOT / "scripts" / script), *map(str, args)],
                              env=self.env, text=True, capture_output=True)

    def test_backend_configuration_and_link_target(self):
        for backend in ("metal", "metal4"):
            with self.subTest(backend=backend):
                build = self.path / backend
                executable = build / "bin/Release" / f"test_ios_{backend}_link.app" / f"test_ios_{backend}_link"
                executable.parent.mkdir(parents=True)
                executable.touch()
                self.log.write_text("")
                result = self.run_script("build_ios_library.sh", "--backend", backend,
                                         "--build-dir", build, "--llvm-dir", self.llvm)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = self.log.read_text().splitlines()
                self.assertIn("-DLUISA_COMPUTE_BUILD_IOS_LINK_TESTS=ON", args)
                self.assertIn("-DLUISA_COMPUTE_BUILD_TESTS=OFF", args)
                self.assertIn("-DCMAKE_SYSTEM_NAME=iOS", args)
                self.assertIn("luisa-ios-link-tests", args)
                self.assertIn("CODE_SIGNING_ALLOWED=NO", args)
                self.assertIn(f"-DLUISA_COMPUTE_ENABLE_METAL={'ON' if backend == 'metal' else 'OFF'}", args)
                self.assertIn(f"-DLUISA_COMPUTE_ENABLE_METAL4={'ON' if backend == 'metal4' else 'OFF'}", args)
                self.assertFalse(any("IOS_EXAMPLES" in arg or "IOS_BENCHMARKS" in arg for arg in args))

    def test_invalid_backend_and_missing_llvm(self):
        self.assertNotEqual(self.run_script("build_ios_library.sh", "--backend", "invalid").returncode, 0)
        self.assertNotEqual(self.run_script("build_ios_library.sh", "--backend", "metal4",
                                           "--llvm-dir", self.path / "absent").returncode, 0)

    def test_audit_fails_closed(self):
        executable = self.path / "test_ios_link"
        self.assertNotEqual(self.run_script("audit_ios_link_test.sh", executable).returncode, 0)
        executable.touch()
        self.assertEqual(self.run_script("audit_ios_link_test.sh", executable).returncode, 0)
        for tool, body in (("lipo", "echo x86_64"),
                           ("xcrun", "echo 'platform MACOS'"),
                           ("otool", "printf 'binary:\\n @rpath/libLLVM.dylib\\n'"),
                           ("otool", "exit 1")):
            with self.subTest(tool=tool, body=body):
                path = self.path / tool
                original = path.read_text()
                self.tool(tool, body)
                self.assertNotEqual(self.run_script("audit_ios_link_test.sh", executable).returncode, 0)
                path.write_text(original)


if __name__ == "__main__":
    unittest.main()
