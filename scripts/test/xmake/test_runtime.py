import shutil
import sys
from common import run_cmd, get_targets, PROJECT_ROOT, get_backends, format_backends

RUNTIME_TARGETS = [
    "test_accel_build_modes",
    "test_accel_visibility",
    "test_atomic",
    "test_atomic_queue",
    "test_bindless_mip",
    "test_byte_buffer",
    "test_context",
    "test_copy",
    "test_decoupled_look_back",
    "test_direct_texture_sampling",
    "test_matrix_multiply",
    "test_mipmap",
    "test_mha_warp_reduction",
    "test_pinned_mem",
    "test_printer",
    "test_printer_custom_callback",
    "test_sampler",
    "test_shared_memory",
    "test_softmax",
    "test_texture_compress",
    "test_texture_io",
    "test_timeline_event",
    "test_warp",
    "test_warp_prefix_scan",
    "test_warp_sparse_collectives",
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

# These executables validate HIP-specific contracts and intentionally skip on
# other devices. Run them only with HIP so a skip can never be reported as
# coverage for another requested backend.
HIP_RUNTIME_TARGETS = [
    "test_hip_codegen_arithmetic",
    "test_hip_curve_ray_query",
    "test_hip_motion_instance_device_ops",
    "test_hip_motion_instance_matrix",
    "test_hip_motion_instance_srt",
    "test_hip_motion_ray_query",
    "test_hip_motion_workgroup_rows",
    "test_hip_ray_query_pipeline",
    "test_hip_signed_texture_io",
    "test_hip_wave_size",
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
    backends = get_backends(["dx", "vk", "cuda", "hip", "metal"])

    available = set(get_targets())
    targets = [t for t in RUNTIME_TARGETS if t in available]
    hip_targets = [t for t in HIP_RUNTIME_TARGETS if t in available] if "hip" in backends else []
    missing_hip_targets = [t for t in HIP_RUNTIME_TARGETS if t not in available] if "hip" in backends else []

    if not targets and not hip_targets:
        print("No runtime targets found.")
        sys.exit(1)

    failures = [f"missing-target:{target}:hip" for target in missing_hip_targets]
    for target in missing_hip_targets:
        print(f"ERROR: required HIP regression target is unavailable: {target}")
    for target in targets:
        for backend in backends:
            print(f"--- Running {target} with backend {backend} ---")
            if run_cmd(["xmake", "run", target, backend]) != 0:
                failures.append(f"run:{target}:{backend}")
    for target in hip_targets:
        print(f"--- Running {target} with backend hip ---")
        if run_cmd(["xmake", "run", target, "hip"]) != 0:
            failures.append(f"run:{target}:hip")

    if failures:
        print("\n!!! FAILURES !!!")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nAll runtime tests passed for backends: {format_backends(backends)}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
