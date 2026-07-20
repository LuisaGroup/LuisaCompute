# Gallery

This gallery showcases output images from LuisaCompute examples and integration tests.
Checked-in reference images are stored in `docs/gallery/`. Offline rendering does
not create or select a reference automatically; pass the reference explicitly.

To render offline and compare against a checked-in reference:

```bash
<build-dir>/bin/<example_or_test> <backend> --offline --compare docs/gallery/<reference>.png
```

---

## Rendering

### Path Tracing

Cornell Box rendered with Monte Carlo path tracing, demonstrating global illumination,
soft shadows, and color bleeding.

```{image} ../gallery/test_path_tracing.png
:alt: Path Tracing (Cornell Box)
:width: 512px
```

**Source:** `examples/rendering/path_tracing.cpp`

### Path Tracing (HDR)

High dynamic range path tracing with tone mapping.

```{image} ../gallery/test_path_tracing_hdr.png
:alt: Path Tracing HDR
:width: 512px
```

**Source:** `examples/rendering/path_tracing_hdr.cpp`

### Path Tracing (Camera)

Path tracing with depth of field camera model.

```{image} ../gallery/test_path_tracing_camera.png
:alt: Path Tracing with Camera
:width: 512px
```

**Source:** `examples/rendering/path_tracing_camera.cpp`

### Path Tracing (Cutout)

Path tracing with alpha-tested cutout geometry.

```{image} ../gallery/test_path_tracing_cutout.png
:alt: Path Tracing Cutout
:width: 512px
```

**Source:** `examples/rendering/path_tracing_cutout.cpp`

### Path Tracing (Nested Callable)

Path tracing demonstrating nested callable composition.

```{image} ../gallery/test_path_tracing_nested_callable.png
:alt: Path Tracing Nested Callable
:width: 512px
```

**Source:** `examples/rendering/path_tracing_nested_callable.cpp`

### Path Tracing (Ray Masks)

Path tracing with per-instance ray visibility masks.

```{image} ../gallery/test_path_tracing_ray_masks.png
:alt: Path Tracing Ray Masks
:width: 512px
```

**Source:** `examples/rendering/path_tracing_ray_masks.cpp`

### Path Tracing (Spectral)

Spectral path tracing with wavelength-dependent rendering.

```{image} ../gallery/test_path_tracing_spectrum.png
:alt: Path Tracing Spectral
:width: 512px
```

**Source:** `examples/rendering/path_tracing_spectrum.cpp`

### Path Tracing (IR Backend)

Path tracing compiled through the IR pipeline.

```{image} ../gallery/test_path_tracing.png
:alt: Path Tracing IR
:width: 512px
```

**Source:** `examples/rendering/path_tracing_ir.cpp`

### Photon Mapping

Global illumination via photon mapping with caustics.

```{image} ../gallery/test_photon_mapping.png
:alt: Photon Mapping
:width: 512px
```

**Source:** `examples/rendering/photon_mapping.cpp`

### SDF Renderer

Signed distance field renderer with ray marching.

```{image} ../gallery/sdf_renderer.png
:alt: SDF Renderer
:width: 512px
```

**Source:** `examples/rendering/sdf_renderer.cpp`

### SDF Renderer (IR Backend)

SDF renderer compiled through the IR pipeline.

```{image} ../gallery/sdf_renderer.png
:alt: SDF Renderer IR
:width: 512px
```

**Source:** `examples/rendering/sdf_renderer_ir.cpp`

### Voxel Ray Tracer

Real-time voxel rendering with ray tracing.

```{image} ../gallery/test_voxel_raytracer.png
:alt: Voxel Ray Tracer
:width: 512px
```

**Source:** `examples/rendering/voxel_raytracer.cpp`

### Black Hole

Interstellar-style black hole with gravitational lensing and accretion disk.

```{image} ../gallery/test_blackhole.png
:alt: Black Hole
:width: 512px
```

**Source:** `examples/rendering/blackhole.cpp`

### ShaderToy

Classic ShaderToy-style procedural rendering.

```{image} ../gallery/test_shader_toy.png
:alt: ShaderToy
:width: 512px
```

**Source:** `examples/rendering/shader_toy.cpp`

### ShaderToy (SpaceX)

SpaceX-inspired ShaderToy effect.

```{image} ../gallery/test_shader_toy_spacex.png
:alt: ShaderToy SpaceX
:width: 512px
```

**Source:** `examples/rendering/shader_toy_spacex.cpp`

### Procedural Geometry

Procedural geometry rendering with ray tracing.

```{image} ../gallery/test_procedural.png
:alt: Procedural
:width: 512px
```

**Source:** `examples/rendering/procedural.cpp`

---

## Simulation

### MPM88 Fluid

Material Point Method fluid simulation.

```{image} ../gallery/test_mpm88.png
:alt: MPM88 Fluid Simulation
:width: 512px
```

**Source:** `examples/simulation/mpm88.cpp`

### Game of Life

Conway's Game of Life cellular automaton.

```{image} ../gallery/test_game_of_life.png
:alt: Game of Life
:width: 512px
```

**Source:** `examples/simulation/game_of_life.cpp`

### Wave Equation

Interactive wave equation PDE solver.

```{image} ../gallery/test_wave_equation.png
:alt: Wave Equation
:width: 512px
```

**Source:** `examples/simulation/wave_equation.cpp`

### Fire Simulation

Physics-based fire particle system.

```{image} ../gallery/test_fire_simulation.png
:alt: Fire Simulation
:width: 512px
```

**Source:** `examples/simulation/fire_simulation.cpp`

### N-Body Simulation

Gravitational N-body particle simulation.

```{image} ../gallery/test_nbody_simulation.png
:alt: N-Body Simulation
:width: 512px
```

**Source:** `examples/simulation/nbody_simulation.cpp`

---

## Compute

### Image Processing

Multi-pass image processing pipeline with filters and effects.

```{image} ../gallery/test_image_processing.png
:alt: Image Processing
:width: 512px
```

**Source:** `examples/compute/image_processing.cpp`

---

## Integration Tests

These images are produced by integration tests and verify correctness of specific
framework features.

### RTX Ray Tracing

Basic RTX intersection test with triangle meshes.

```{image} ../gallery/test_rtx.png
:alt: RTX Test
:width: 512px
```

**Source:** `src/tests/integration/runtime/test_rtx.cpp`

### Indirect RTX

Indirect dispatch ray tracing test. No checked-in reference image is currently
available for this test.

**Source:** `src/tests/integration/runtime/test_indirect_rtx.cpp`

### Procedural Callable

Procedural callable rendering integration test.

```{image} ../gallery/test_procedural_callable.png
:alt: Procedural Callable Test
:width: 512px
```

**Source:** `src/tests/integration/runtime/test_procedural_callable.cpp`

### Native Code Include

External native shader code inclusion test.

```{image} ../gallery/test_native_code.png
:alt: Native Code Test
:width: 512px
```

**Source:** `src/tests/integration/runtime/test_native_include.cpp`

### Motion Blur

Motion blur rendering with mesh and curve keyframes. No checked-in reference
image is currently available for this test.

**Source:** `src/tests/integration/runtime/test_motion_blur.cpp`

### 3D Texture Volume Rendering

Perlin noise volume rendering with ray marching.

```{image} ../gallery/test_texture3d.png
:alt: 3D Texture Test
:width: 512px
```

**Source:** `src/tests/integration/runtime/test_texture3d.cpp`
