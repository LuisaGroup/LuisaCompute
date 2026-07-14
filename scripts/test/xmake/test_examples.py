import sys
from common import run_cmd, get_targets, PROJECT_ROOT, get_backends
ALLOWED_EXAMPLES = {
    "example_path_tracing",
    "example_path_tracing_camera",
    "example_path_tracing_cutout",
    "example_path_tracing_hdr",
    "example_path_tracing_nested_callable",
    "example_path_tracing_ray_masks",
    "example_path_tracing_spectrum",
    "example_photon_mapping",
    "example_sdf_renderer",
    "example_blackhole",
    "example_voxel_raytracer",
    "example_procedural",
    "example_shader_toy",
    "example_shader_toy_spacex",
    "example_shader_visuals_present",
    "example_path_tracing_xir2ast",
    "example_sdf_renderer_xir2ast",
    "example_fire_simulation",
    "example_game_of_life",
    "example_mpm3d",
    "example_mpm88",
    "example_nbody_simulation",
    "example_wave_equation",
}


def main():
    backends = get_backends(["dx", "vk", "cuda", "metal"])
    available = set(get_targets())
    targets = sorted([t for t in available if t in ALLOWED_EXAMPLES])

    if not targets:
        print("No example targets found.")
        sys.exit(1)

    failures = []
    for target in targets:
        for backend in backends:
            print(f"--- Running {target} with backend {backend} ---")
            if run_cmd(["xmake", "run", target, backend, "--offline"]) != 0:
                failures.append(f"run:{target}:{backend}")

    if failures:
        print("\n!!! FAILURES !!!")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\nAll example tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
