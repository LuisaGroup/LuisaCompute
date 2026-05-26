import shutil
import sys
from common import run_cmd, get_targets, PROJECT_ROOT

RUNTIME_TARGETS = [
    "test_atomic",
    "test_atomic_queue",
    "test_byte_buffer",
    "test_context",
    "test_copy",
    "test_decoupled_look_back",
    "test_matrix_multiply",
    "test_mipmap",
    "test_pinned_mem",
    "test_printer",
    "test_printer_custom_callback",
    "test_sampler",
    "test_shared_memory",
    "test_softmax",
    "test_texture_compress",
    "test_texture_io",
    "test_warp",
    "test_warp_prefix_scan",
    "test_buffer_io",
    "test_buffer",
    "test_buffer_view",
    "test_device_test",
    "test_external_buffer",
    "test_gemm",
    "test_shared_mem",
    "test_fp4",
    "test_fp4_quantization",
    "test_fp8",
    "test_fp8_quantization",
]

BACKENDS = ["dx", "vk", "cuda", "metal"]


def _copy_logo():
    """Copy logo.png to bin/debug and bin/release before running tests."""
    src = PROJECT_ROOT / "src" / "tests" / "logo.png"
    if not src.exists():
        print(f"WARNING: {src} not found, skipping logo copy")
        return

    for subdir in ("debug", "release"):
        dst_dir = PROJECT_ROOT / "bin" / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "logo.png"
        print(f"Copying {src} -> {dst}")
        shutil.copy2(src, dst)


def main():
    args = sys.argv[1:]
    mode = None
    backend = None

    if args and args[0] in ("debug", "release"):
        mode = args.pop(0)
    if args:
        backend = args.pop(0)
    if args:
        print(f"Usage: python {sys.argv[0]} [mode] [backend]")
        sys.exit(1)

    backends = [backend] if backend else BACKENDS

    ret = run_cmd(["xmake", "f", "-m", mode if mode else "release", "-c"])
    if ret != 0:
        print("ERROR: xmake config failed")
        sys.exit(ret)

    _copy_logo()

    available = set(get_targets())
    targets = [t for t in RUNTIME_TARGETS if t in available]

    if not targets:
        print("No runtime targets found.")
        sys.exit(1)

    failures = []
    for target in targets:
        print(f"\n=== Building {target} ===")
        if run_cmd(["xmake", "build", target]) != 0:
            failures.append(f"build:{target}")
            continue
        for backend in backends:
            print(f"--- Running {target} with backend {backend} ---")
            if run_cmd(["xmake", "run", target, backend]) != 0:
                failures.append(f"run:{target}:{backend}")

    if failures:
        print("\n!!! FAILURES !!!")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\nAll runtime tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
