import shutil
import sys
from common import run_cmd, get_targets, PROJECT_ROOT, get_backends

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
    _copy_logo()
    backends = get_backends(["dx", "vk", "cuda", "metal"])

    available = set(get_targets())
    targets = [t for t in RUNTIME_TARGETS if t in available]

    if not targets:
        print("No runtime targets found.")
        sys.exit(1)

    failures = []
    for target in targets:
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
