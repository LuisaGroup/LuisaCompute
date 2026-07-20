import sys
from common import run_cmd, get_targets, PROJECT_ROOT, get_backends, format_backends

ALLOWED_EXAMPLES = {
    "example_path_tracing",
    "example_path_tracing_camera",
    "example_path_tracing_cutout",
    "example_path_tracing_hdr",
    "example_path_tracing_ir",
    "example_path_tracing_nested_callable",
    "example_path_tracing_ray_masks",
    "example_path_tracing_spectrum",
    "example_photon_mapping",
    "example_sdf_renderer",
    "example_sdf_renderer_ir",
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
    "example_image_processing",
}

# Each entry is backed by an existing docs/gallery image and by comparison
# handling in the corresponding offline example. The IR/XIR-to-AST variants
# intentionally render the same scene and write the same output as their base
# examples, so they share the same references.
EXAMPLE_REFERENCES = {
    "example_path_tracing": "test_path_tracing.png",
    "example_path_tracing_camera": "test_path_tracing_camera.png",
    "example_path_tracing_cutout": "test_path_tracing_cutout.png",
    "example_path_tracing_hdr": "test_path_tracing_hdr.png",
    "example_path_tracing_ir": "test_path_tracing.png",
    "example_path_tracing_nested_callable": "test_path_tracing_nested_callable.png",
    "example_path_tracing_ray_masks": "test_path_tracing_ray_masks.png",
    "example_path_tracing_spectrum": "test_path_tracing_spectrum.png",
    "example_path_tracing_xir2ast": "test_path_tracing.png",
    "example_photon_mapping": "test_photon_mapping.png",
    "example_sdf_renderer": "sdf_renderer.png",
    "example_sdf_renderer_ir": "sdf_renderer.png",
    "example_sdf_renderer_xir2ast": "sdf_renderer.png",
    "example_blackhole": "test_blackhole.png",
    "example_voxel_raytracer": "test_voxel_raytracer.png",
    "example_procedural": "test_procedural.png",
    "example_shader_toy": "test_shader_toy.png",
    "example_shader_toy_spacex": "test_shader_toy_spacex.png",
    "example_fire_simulation": "test_fire_simulation.png",
    "example_game_of_life": "test_game_of_life.png",
    "example_mpm3d": "test_mpm3d.png",
    "example_mpm88": "test_mpm88.png",
    "example_nbody_simulation": "test_nbody_simulation.png",
    "example_wave_equation": "test_wave_equation.png",
    "example_image_processing": "test_image_processing.png",
}

PATH_TRACING_EXAMPLES = {
    target for target in EXAMPLE_REFERENCES if target.startswith("example_path_tracing")
}


def main():
    backends = get_backends(["dx", "vk", "cuda", "hip", "metal"])
    available = set(get_targets())
    targets = sorted([t for t in available if t in ALLOWED_EXAMPLES])

    if not targets:
        print("No example targets found.")
        sys.exit(1)

    failures = []
    render_only_targets = []
    for target in targets:
        for backend in backends:
            print(f"--- Running {target} with backend {backend} ---")
            cmd = ["xmake", "run", target, backend, "--offline"]
            reference_name = EXAMPLE_REFERENCES.get(target)
            if reference_name is not None:
                reference = PROJECT_ROOT / "docs" / "gallery" / reference_name
                if not reference.is_file():
                    print(f"ERROR: gallery reference not found: {reference}")
                    failures.append(f"reference:{target}:{backend}")
                    continue
                if target in PATH_TRACING_EXAMPLES:
                    cmd.extend(["--spp", "1024"])
                cmd.extend(["--compare", str(reference)])
            else:
                print(f"NOTE: {target} has no mapped gallery reference; this is a render-only run.")
                render_only_targets.append(target)
            if run_cmd(cmd) != 0:
                failures.append(f"run:{target}:{backend}")

    if failures:
        print("\n!!! FAILURES !!!")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nAll offline example runs passed for backends: {format_backends(backends)}.")
        if render_only_targets:
            names = ", ".join(sorted(set(render_only_targets)))
            print(f"Render-only (not reference-validated): {names}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
