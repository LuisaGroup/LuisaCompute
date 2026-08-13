# Tutorials

This section provides step-by-step tutorials for building practical applications with LuisaCompute.

## Table of Contents

1. [ShaderToy-Style Mandelbrot](#shadertoy-style-mandelbrot-set) - A gentle introduction
2. [Path Tracing Renderer](#path-tracing-renderer) - Global illumination with ray tracing
3. [MPM Fluid Simulation](#mpm-fluid-simulation) - Material Point Method for fluids
4. [Conway's Game of Life](#conways-game-of-life) - Cellular automata with ping-pong buffering
5. [Wave Equation Simulation](#wave-equation-simulation) - Interactive PDE solver
6. [Image Processing Pipeline](#image-processing-pipeline) - Multi-pass filters and effects
7. [N-Body Simulation](#n-body-gravitational-simulation) - Gravitational particle physics
8. [Fire Particle System](#fire-particle-system) - Physics-based fire simulation
9. [Reaction-Diffusion](#reaction-diffusion-simulation) - Gray-Scott pattern formation
10. [Voxel Ray Tracer](#voxel-ray-tracer) - Real-time voxel rendering
11. [Black Hole Renderer](#black-hole-renderer) - Interstellar-style gravitational lensing

---

## ShaderToy-Style Mandelbrot Set

Let's start with a simple yet visually appealing example: rendering the Mandelbrot set. This mimics the style of [ShaderToy](https://www.shadertoy.com/), a popular platform for fragment shaders.

### Final Result

A colorful visualization of the Mandelbrot set with smooth coloring.

### Step-by-Step Implementation

#### Step 1: Project Setup

Create a new project with the following structure:

```
mandelbrot/
├── CMakeLists.txt
└── main.cpp
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.23)
project(mandelbrot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Add LuisaCompute
add_subdirectory(LuisaCompute)

add_executable(mandelbrot main.cpp)
target_link_libraries(mandelbrot PRIVATE luisa::compute)
```

#### Step 2: Basic Structure

**main.cpp - Initial Setup:**

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    // Step 1: Create context and device
    Context context{argv[0]};
    Device device = context.create_device("cuda");  // or "metal", "dx", "vk", "hip", "fallback"
    Stream stream = device.create_stream();
    
    // Step 2: Create an image to render to
    static constexpr uint2 resolution = make_uint2(1024u, 1024u);
    Image<float> image = device.create_image<float>(
        PixelStorage::FLOAT4, resolution.x, resolution.y);
    
    // We'll add the kernel here...
    
    return 0;
}
```

#### Step 3: Understanding the Mandelbrot Set

The Mandelbrot set is defined by the iteration:
```
z_{n+1} = z_n² + c
```

Where `c` is a complex number corresponding to pixel coordinates. If the sequence remains bounded (|z| ≤ 2), the point is in the set.

#### Step 4: Implementing the Kernel

```cpp
// Inside main(), after creating the image:

Kernel2D mandelbrot_kernel = [&](ImageFloat image) noexcept {
    // Get pixel coordinates
    UInt2 coord = dispatch_id().xy();
    Float2 uv = make_float2(coord) / make_float2(resolution);
    
    // Map to complex plane: centered at (-0.5, 0), zoomed in
    Float2 c = (uv - 0.5f) * 3.0f - make_float2(0.5f, 0.0f);
    
    // Mandelbrot iteration
    Float2 z = make_float2(0.0f);
    Float n = 0.0f;
    static constexpr int M = 100;  // Maximum iterations
    
    $for (i, M) {
        // z = z² + c
        Float2 z_new = make_float2(
            z.x * z.x - z.y * z.y + c.x,
            2.0f * z.x * z.y + c.y
        );
        z = z_new;
        
        // Check divergence
        $if (dot(z, z) > 4.0f) {
            $break;
        };
        n += 1.0f;
    };
    
    // Color using cosine palette (Inigo Quilez's technique)
    Float t = n / static_cast<float>(M);
    Float3 color = 0.5f + 0.5f * cos(6.28318f * (t + make_float3(0.0f, 0.33f, 0.67f)));
    
    // Write to image
    image.write(coord, make_float4(color, 1.0f));
};
```

#### Step 5: Compile and Execute

```cpp
// Compile the kernel
auto shader = device.compile(mandelbrot_kernel);

// Execute
stream << shader(image).dispatch(resolution)
       << synchronize();

// Save result
luisa::vector<uint8_t> pixels(resolution.x * resolution.y * 4);
stream << image.copy_to(pixels.data())
       << synchronize();

stbi_write_png("mandelbrot.png", resolution.x, resolution.y, 4, pixels.data(), 0);
LUISA_INFO("Saved to mandelbrot.png");
```

#### Complete Code

<details>
<summary>Click to expand complete code</summary>

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    
    static constexpr uint2 resolution = make_uint2(1024u, 1024u);
    Image<float> image = device.create_image<float>(
        PixelStorage::FLOAT4, resolution.x, resolution.y);
    
    Kernel2D mandelbrot_kernel = [&](ImageFloat image) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float2 uv = make_float2(coord) / make_float2(resolution);
        
        Float2 c = (uv - 0.5f) * 3.0f - make_float2(0.5f, 0.0f);
        Float2 z = make_float2(0.0f);
        Float n = 0.0f;
        static constexpr int M = 100;
        
        $for (i, M) {
            Float2 z_new = make_float2(
                z.x * z.x - z.y * z.y + c.x,
                2.0f * z.x * z.y + c.y
            );
            z = z_new;
            $if (dot(z, z) > 4.0f) { $break; };
            n += 1.0f;
        };
        
        Float t = n / static_cast<float>(M);
        Float3 color = 0.5f + 0.5f * cos(6.28318f * (t + make_float3(0.0f, 0.33f, 0.67f)));
        image.write(coord, make_float4(color, 1.0f));
    };
    
    auto shader = device.compile(mandelbrot_kernel);
    stream << shader(image).dispatch(resolution) << synchronize();
    
    luisa::vector<uint8_t> pixels(resolution.x * resolution.y * 4);
    stream << image.copy_to(pixels.data()) << synchronize();
    stbi_write_png("mandelbrot.png", resolution.x, resolution.y, 4, pixels.data(), 0);
    
    return 0;
}
```
</details>

#### Exercises

1. **Zoom Animation**: Modify `c` to animate over time
2. **Julia Set**: Keep `c` constant and vary initial `z`
3. **Smooth Coloring**: Use `n - log2(log2(dot(z,z)))` for smoother gradients

---

## Path Tracing Renderer

Now let's build a more sophisticated renderer using ray tracing. We'll implement a basic path tracer for the Cornell Box scene.

### Final Result

A physically-based rendered image with global illumination.

### Step-by-Step Implementation

#### Step 1: Scene Setup

We need geometry (triangles), materials, and acceleration structures:

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

// Cornell Box geometry data
static constexpr float3 vertices[] = {
    // Floor
    {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f},
    // Ceiling
    {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f},
    // ... more vertices
};

static constexpr uint3 triangles[] = {
    {0, 1, 2}, {0, 2, 3},  // Floor
    // ... more triangles
};
```

#### Step 2: Create Acceleration Structure

```cpp
// Create buffers
Buffer<float3> vertex_buffer = device.create_buffer<float3>(num_vertices);
Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(num_triangles);

// Upload data
stream << vertex_buffer.copy_from(vertices)
       << triangle_buffer.copy_from(triangles)
       << synchronize();

// Create mesh and acceleration structure
Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
Accel accel = device.create_accel();
accel.emplace_back(mesh, make_float4x4(1.0f));  // Identity transform

stream << mesh.build() << accel.build() << synchronize();
```

#### Step 3: Helper Callables

```cpp
// Linear congruential generator for random numbers
Callable lcg = [](UInt &state) noexcept {
    constexpr uint a = 1664525u;
    constexpr uint c = 1013904223u;
    state = a * state + c;
    return cast<float>(state) / cast<float>(std::numeric_limits<uint>::max());
};

// Cosine-weighted hemisphere sampling
Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
    Float r = sqrt(u.x);
    Float theta = 2.0f * pi * u.y;
    return make_float3(r * cos(theta), r * sin(theta), sqrt(1.0f - u.x));
};

// Build orthonormal basis from normal
Callable make_onb = [](Float3 normal) noexcept {
    Float3 b1 = ite(abs(normal.y) < 0.999f, 
                     normalize(cross(normal, make_float3(0.0f, 1.0f, 0.0f))),
                     make_float3(1.0f, 0.0f, 0.0f));
    Float3 b2 = cross(normal, b1);
    return make_float3x3(b1, b2, normal);
};
```

#### Step 4: The Path Tracing Kernel

```cpp
Kernel2D path_tracer = [&](ImageFloat image, ImageUInt seed_image, 
                           AccelVar accel, UInt frame_index) noexcept {
    set_block_size(16u, 16u, 1u);
    
    UInt2 coord = dispatch_id().xy();
    Float2 resolution = make_float2(dispatch_size().xy());
    
    // Initialize random seed
    $if (frame_index == 0u) {
        seed_image.write(coord, make_uint4(coord.x * 1234567u + coord.y));
    };
    
    UInt seed = seed_image.read(coord).x;
    
    // Generate camera ray
    Float2 uv = (make_float2(coord) + make_float2(lcg(seed), lcg(seed))) / resolution;
    Float2 ndc = uv * 2.0f - 1.0f;
    
    Float3 origin = make_float3(0.0f, 0.0f, 3.0f);
    Float3 direction = normalize(make_float3(ndc.x, ndc.y, -1.0f));
    
    // Path tracing loop
    Float3 radiance = def(make_float3(0.0f));
    Float3 throughput = def(make_float3(1.0f));
    
    $for (bounce, 5u) {  // Maximum 5 bounces
        // Trace ray
        Var<Ray> ray = make_ray(origin, direction);
        Var<TriangleHit> hit = accel.intersect(ray, {});
        
        $if (hit->miss()) { $break; };
        
        // Get hit information
        Float3 p0 = vertex_buffer->read(hit.prim * 3u + 0u);
        Float3 p1 = vertex_buffer->read(hit.prim * 3u + 1u);
        Float3 p2 = vertex_buffer->read(hit.prim * 3u + 2u);
        Float3 p = hit->interpolate(p0, p1, p2);
        Float3 n = normalize(cross(p1 - p0, p2 - p0));
        
        // Simple diffuse material (white)
        Float3 albedo = make_float3(0.8f);
        
        // Add emission if hit light
        $if (hit.prim >= light_triangle_start) {
            radiance += throughput * make_float3(10.0f);  // Emissive
            $break;
        };
        
        // Sample next direction (cosine-weighted)
        Float2 u = make_float2(lcg(seed), lcg(seed));
        Float3 local_dir = cosine_sample_hemisphere(u);
        Var<Float3x3> onb = make_onb(n);
        Float3 new_dir = onb * local_dir;
        
        // Update throughput
        Float cos_theta = dot(new_dir, n);
        throughput *= albedo * cos_theta * (1.0f / pi);
        
        // Russian roulette
        Float p_survive = max(throughput.x, max(throughput.y, throughput.z));
        $if (lcg(seed) > p_survive) { $break; };
        throughput *= 1.0f / p_survive;
        
        // Setup next ray
        origin = p + n * 1e-4f;
        direction = new_dir;
    };
    
    // Accumulate over frames
    Float4 prev = image.read(coord);
    Float count = prev.w + 1.0f;
    Float3 avg = lerp(prev.xyz(), radiance, 1.0f / count);
    image.write(coord, make_float4(avg, count));
    
    seed_image.write(coord, make_uint4(seed));
};
```

#### Step 5: Rendering Loop

```cpp
// Create images
Image<float> framebuffer = device.create_image<float>(
    PixelStorage::FLOAT4, resolution.x, resolution.y);
Image<uint> seed_image = device.create_image<uint>(
    PixelStorage::INT1, resolution.x, resolution.y);

auto renderer = device.compile(path_tracer);

// Render multiple frames for accumulation
for (uint frame = 0; frame < 100; frame++) {
    stream << renderer(framebuffer, seed_image, accel, frame).dispatch(resolution)
           << synchronize();
}

// Tonemap and save
Kernel2D tonemap = [&](ImageFloat hdr, ImageFloat ldr) noexcept {
    UInt2 coord = dispatch_id().xy();
    Float3 color = hdr.read(coord).xyz();
    // Simple Reinhard tonemapping
    color = color / (color + 1.0f);
    // Gamma correction
    color = pow(color, 1.0f / 2.2f);
    ldr.write(coord, make_float4(color, 1.0f));
};

Image<float> ldr = device.create_image<float>(
    PixelStorage::BYTE4, resolution.x, resolution.y);
stream << device.compile(tonemap)(framebuffer, ldr).dispatch(resolution)
       << synchronize();

// Save...
```

#### Key Concepts Explained

1. **Path Tracing**: Simulates light transport by tracing random paths from camera
2. **Russian Roulette**: Probabilistically terminates paths to avoid infinite recursion
3. **Cosine Sampling**: Importance sampling for diffuse surfaces
4. **Accumulation**: Multiple samples averaged for noise reduction

---

## MPM Fluid Simulation

Material Point Method (MPM) is a hybrid Lagrangian-Eulerian method for simulating continua. We'll implement a basic 3D fluid simulation.

### Final Result

An interactive 3D fluid simulation with particles.

### Step-by-Step Implementation

#### Step 1: Simulation Parameters

```cpp
static constexpr int n_grid = 64;           // Grid resolution
static constexpr uint n_particles = 50000;  // Number of particles
static constexpr float dx = 1.0f / n_grid;  // Grid spacing
static constexpr float dt = 8e-5f;          // Time step
static constexpr float p_vol = (dx * 0.5f) * (dx * 0.5f) * (dx * 0.5f);
static constexpr float p_mass = 1.0f * p_vol;
static constexpr float E = 400.0f;          // Young's modulus
static constexpr float gravity = 9.8f;
```

#### Step 2: Data Structures

```cpp
// Particle data (Lagrangian)
Buffer<float3> x = device.create_buffer<float3>(n_particles);  // Positions
Buffer<float3> v = device.create_buffer<float3>(n_particles);  // Velocities
Buffer<float3x3> C = device.create_buffer<float3x3>(n_particles);  // Affine momentum
Buffer<float> J = device.create_buffer<float>(n_particles);    // Volume ratio

// Grid data (Eulerian)
Buffer<float4> grid = device.create_buffer<float4>(n_grid * n_grid * n_grid);
// xyz = momentum, w = mass
```

#### Step 3: Grid Clearing Kernel

```cpp
auto clear_grid = device.compile<3>([&] {
    set_block_size(8, 8, 1);
    UInt idx = dispatch_id().x + dispatch_id().y * n_grid 
               + dispatch_id().z * n_grid * n_grid;
    grid->write(idx, make_float4(0.0f));
});
```

#### Step 4: Particle-to-Grid Transfer

This is the heart of MPM - transferring particle data to the grid:

```cpp
auto particle_to_grid = device.compile<1>([&] {
    set_block_size(64, 1, 1);
    UInt p = dispatch_id().x;
    
    // Particle position in grid coordinates
    Float3 Xp = x->read(p) / dx;
    Int3 base = make_int3(Xp - 0.5f);
    Float3 fx = Xp - make_float3(base);
    
    // Quadratic B-spline weights
    auto w = [&](Float3 fx) {
        return make_float3(
            0.5f * (1.5f - fx) * (1.5f - fx),
            0.75f - (fx - 1.0f) * (fx - 1.0f),
            0.5f * (fx - 0.5f) * (fx - 0.5f)
        );
    };
    Float3 wx = w(fx.x), wy = w(fx.y), wz = w(fx.z);
    
    // Stress-based force (simplified for fluids)
    Float stress = -4.0f * dt * E * p_vol * (J->read(p) - 1.0f) / (dx * dx);
    Float3x3 affine = make_float3x3(stress, 0.0f, 0.0f,
                                     0.0f, stress, 0.0f,
                                     0.0f, 0.0f, stress);
    Float3 vp = v->read(p);
    
    // Scatter to 3x3x3 grid nodes
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                Float3 offset = make_float3(i, j, k);
                Float3 dpos = (offset - fx) * dx;
                Float weight = wx[i] * wy[j] * wz[k];
                
                Float3 v_add = weight * (p_mass * vp + affine * dpos);
                UInt idx = (base.x + i) + (base.y + j) * n_grid 
                          + (base.z + k) * n_grid * n_grid;
                
                // Atomic add to grid
                grid->atomic(idx).x.fetch_add(v_add.x);
                grid->atomic(idx).y.fetch_add(v_add.y);
                grid->atomic(idx).z.fetch_add(v_add.z);
                grid->atomic(idx).w.fetch_add(weight * p_mass);
            }
        }
    }
});
```

#### Step 5: Grid Velocity Update

```cpp
auto grid_update = device.compile<3>([&] {
    set_block_size(8, 8, 1);
    Int3 coord = make_int3(dispatch_id().xyz());
    
    UInt idx = coord.x + coord.y * n_grid + coord.z * n_grid * n_grid;
    Float4 v_and_m = grid->read(idx);
    
    Float3 v = v_and_m.xyz();
    Float m = v_and_m.w;
    
    // Normalize by mass
    v = ite(m > 0.0f, v / m, v);
    
    // Gravity
    v.y -= dt * gravity;
    
    // Boundary conditions
    static constexpr int bound = 3;
    v = ite((coord < bound && v < 0.0f) || 
            (coord > n_grid - bound && v > 0.0f), 
            0.0f, v);
    
    grid->write(idx, make_float4(v, m));
});
```

#### Step 6: Grid-to-Particle Transfer

```cpp
auto grid_to_particle = device.compile<1>([&] {
    set_block_size(64, 1, 1);
    UInt p = dispatch_id().x;
    
    Float3 Xp = x->read(p) / dx;
    Int3 base = make_int3(Xp - 0.5f);
    Float3 fx = Xp - make_float3(base);
    
    // Same weights as P2G
    // ... weight computation ...
    
    Float3 new_v = def(make_float3(0.0f));
    Float3x3 new_C = def(make_float3x3(0.0f));
    
    // Gather from grid
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                Float3 offset = make_float3(i, j, k);
                Float3 dpos = (offset - fx) * dx;
                Float weight = wx[i] * wy[j] * wz[k];
                
                UInt idx = (base.x + i) + (base.y + j) * n_grid 
                          + (base.z + k) * n_grid * n_grid;
                Float3 g_v = grid->read(idx).xyz();
                
                new_v += weight * g_v;
                new_C += 4.0f * weight * outer_product(g_v, dpos) / (dx * dx);
            }
        }
    }
    
    // Update particle state
    v->write(p, new_v);
    x->write(p, x->read(p) + new_v * dt);
    J->write(p, J->read(p) * (1.0f + dt * trace(new_C)));
    C->write(p, new_C);
});
```

#### Step 7: Simulation Loop

```cpp
// Initialize particles
luisa::vector<float3> x_init(n_particles);
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.2f, 0.8f);

for (uint i = 0; i < n_particles; i++) {
    x_init[i] = make_float3(dis(gen), dis(gen), dis(gen));
}

stream << x.copy_from(x_init.data())
       << v.copy_from(luisa::vector<float3>(n_particles, make_float3(0.0f)).data())
       << J.copy_from(luisa::vector<float>(n_particles, 1.0f).data())
       << synchronize();

// Simulation loop
Window window{"MPM Fluid", 1024, 1024};
Swapchain swapchain = device.create_swapchain(stream, {
    .display = window.native_display(),
    .window = window.native_handle(),
    .size = make_uint2(1024u, 1024u),
});
Image<float> display = device.create_image<float>(
    swapchain.backend_storage(), make_uint2(1024u, 1024u));

while (!window.should_close()) {
    CommandList cmd;
    
    // MPM sub-steps
    for (int i = 0; i < 25; i++) {
        cmd << clear_grid().dispatch(n_grid, n_grid, n_grid)
            << particle_to_grid().dispatch(n_particles)
            << grid_update().dispatch(n_grid, n_grid, n_grid)
            << grid_to_particle().dispatch(n_particles);
    }
    
    // Render particles to display
    cmd << clear_display().dispatch(1024, 1024)
        << draw_particles().dispatch(n_particles)
        << swapchain.present(display);
    
    stream << cmd.commit();
    window.poll_events();
}
```

#### Key Concepts Explained

1. **Lagrangian vs Eulerian**: Particles track material; grid handles physics
2. **APIC (Affine Particle-in-Cell)**: Uses affine momentum matrix for better accuracy
3. **Transfer**: P2G (particle to grid) and G2P (grid to particle) are the core operations
4. **MLS-MPM**: Moving Least Squares MPM for stability

---

## Conway's Game of Life

Cellular automata are fascinating discrete systems. Let's implement Conway's Game of Life, a classic example that demonstrates **ping-pong buffering** - a technique where we alternate between two buffers to avoid read-write conflicts.

### Final Result

An interactive simulation where cells evolve based on simple rules, creating complex emergent patterns.

### Step-by-Step Implementation

#### Step 1: Understanding the Rules

Conway's Game of Life follows three simple rules:
1. **Survival**: Live cells with 2-3 neighbors survive
2. **Birth**: Dead cells with exactly 3 neighbors become alive
3. **Death**: All other cells die or stay dead

#### Step 2: The ImagePair Structure

We'll use two images that swap each frame:

```cpp
struct ImagePair {
    Image<uint> prev;
    Image<uint> curr;
    ImagePair(Device &device, PixelStorage storage, uint width, uint height) noexcept
        : prev{device.create_image<uint>(storage, width, height)},
          curr{device.create_image<uint>(storage, width, height)} {}
    void swap() noexcept { std::swap(prev, curr); }
};
```

#### Step 3: The Update Kernel

```cpp
// Helper to read cell state
Callable read_state = [](ImageUInt prev, UInt2 uv) noexcept {
    return prev.read(uv).x == 255u;
};

Kernel2D kernel = [&](ImageUInt prev, ImageUInt curr) noexcept {
    set_block_size(16, 16, 1);
    UInt count = def(0u);
    UInt2 uv = dispatch_id().xy();
    UInt2 size = dispatch_size().xy();
    Bool state = read_state(prev, uv);
    Int2 p = make_int2(uv);
    
    // Check all 8 neighbors with toroidal wrapping
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                Int2 q = p + make_int2(dx, dy) + make_int2(size);
                Bool neighbor = read_state(prev, make_uint2(q) % size);
                count += ite(neighbor, 1, 0);
            }
        }
    }
    
    // Apply Conway's rules
    Bool c0 = count == 2u;
    Bool c1 = count == 3u;
    curr.write(uv, make_uint4(make_uint3(ite((state & c0) | c1, 255u, 0u)), 255u));
};
```

#### Step 4: Display Kernel

We upscale the low-res simulation for better visibility:

```cpp
Kernel2D display_kernel = [&](ImageUInt in_tex, ImageFloat out_tex) noexcept {
    set_block_size(16, 16, 1);
    UInt2 uv = dispatch_id().xy();
    UInt2 coord = uv / 4u;  // 4x upscaling
    UInt4 value = in_tex.read(coord);
    out_tex.write(uv, make_float4(value) / 255.0f);
};
```

#### Step 5: Initialization and Main Loop

```cpp
// Initialize with random state (25% alive)
luisa::vector<uint> host_image(width * height);
for (auto &v : host_image) {
    auto x = (rng() % 4u == 0u) * 255u;
    v = x * 0x00010101u | 0xff000000u;  // White or black
}
stream << image_pair.prev.copy_from(host_image.data()) << synchronize();

// Main loop
while (!window.should_close()) {
    stream << shader(image_pair.prev, image_pair.curr).dispatch(width, height)
           << display_shader(image_pair.curr, display).dispatch(width * 4u, height * 4u)
           << swap_chain.present(display);
    image_pair.swap();  // Ping-pong: swap buffers
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    window.poll_events();
}
```

#### Key Concepts

1. **Ping-Pong Buffering**: We read from `prev` and write to `curr`, then swap
2. **Toroidal Boundary**: `% size` makes the grid wrap around (edges connect)
3. **Neighborhood**: The 8 surrounding cells determine each cell's fate

#### Exercises

1. **Different Rules**: Try HighLife (born with 3 or 6 neighbors)
2. **Pattern Library**: Implement a pattern loader for gliders, blinkers, etc.
3. **Multi-State**: Add fading effects for recently-dead cells

---

## Wave Equation Simulation

Simulate realistic water ripples using the **finite difference method** to solve the 2D wave equation. This demonstrates PDE solving and interactive GPU computing with beautiful visual results.

### Final Result

An interactive water simulation where you can click and drag to create waves that propagate realistically, interfere with each other, and gradually dampen. The rendering includes caustics effects and foam at wave crests.

### The Physics

The wave equation: `∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)`

Discretized form: `u_next = 2*u_curr - u_prev + c²*dt²*laplacian(u)`

For stability, the CFL condition requires: `c² * dt² < 0.5`

### Step-by-Step Implementation

#### Step 1: Setup Three Buffers

We need three height buffers for time integration (past, present, future):

```cpp
Image<float> height_prev = device.create_image<float>(PixelStorage::FLOAT1, width, height);
Image<float> height_curr = device.create_image<float>(PixelStorage::FLOAT1, width, height);
Image<float> height_next = device.create_image<float>(PixelStorage::FLOAT1, width, height);
```

#### Step 2: The Wave Solver Kernel

```cpp
Kernel2D wave_step = [&](ImageFloat prev, ImageFloat curr, ImageFloat next) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    Var u_curr = curr.read(uv).x;

    // Sample neighbors with clamping (not wrapping)
    auto sample = [&](Int2 offset) noexcept {
        Int2 p = make_int2(uv) + offset;
        p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
        p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
        return curr.read(make_uint2(p)).x;
    };

    Var u_left = sample(make_int2(-1, 0));
    Var u_right = sample(make_int2(1, 0));
    Var u_up = sample(make_int2(0, -1));
    Var u_down = sample(make_int2(0, 1));

    // Laplacian: sum of neighbors - 4*center
    Var laplacian = u_left + u_right + u_up + u_down - 4.0f * u_curr;

    // Read previous height
    Var u_prev = prev.read(uv).x;

    // Wave equation integration
    Var u_next = 2.0f * u_curr - u_prev + (c * c * dt * dt) * laplacian;
    
    // Apply damping for energy loss
    u_next *= damping;
    u_next = clamp(u_next, -2.0f, 2.0f);

    next.write(uv, make_float4(u_next, 0.0f, 0.0f, 1.0f));
};
```

#### Step 3: Interactive Droplet Drop

```cpp
// Setup mouse callbacks
window.set_mouse_callback([&mouse_down, &mouse_pos](MouseButton, Action a, float2 p) noexcept {
    if (a == Action::ACTION_PRESSED) {
        mouse_down = true;
    } else if (a == Action::ACTION_RELEASED) {
        mouse_down = false;
    }
    mouse_pos = p;
});

// Droplet kernel - creates Gaussian ripple
Kernel2D drop_droplet = [](ImageFloat height, UInt2 center, Float strength) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();

    Var dx = cast<float>(cast<int>(uv.x) - cast<int>(center.x));
    Var dy = cast<float>(cast<int>(uv.y) - cast<int>(center.y));
    Var dist_sq = dx * dx + dy * dy;
    
    Var radius = 15.0f;
    Var gaussian = exp(-dist_sq / (radius * radius * 0.5f));

    Var current = height.read(uv).x;
    height.write(uv, make_float4(current + strength * gaussian, 0.0f, 0.0f, 1.0f));
};
```

#### Step 4: Rendering with Caustics Effect

```cpp
Kernel2D render_kernel = [](ImageFloat height, ImageFloat output, Float time) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    Var h = height.read(uv).x;

    // Sample neighbors for slope (caustics effect)
    Var h_left = sample(make_int2(-2, 0));
    Var h_right = sample(make_int2(2, 0));
    Var h_up = sample(make_int2(0, -2));
    Var h_down = sample(make_int2(0, 2));

    // Slope magnitude creates caustic highlights
    Var slope_x = abs(h_right - h_left);
    Var slope_y = abs(h_down - h_up);
    Var slope = sqrt(slope_x * slope_x + slope_y * slope_y);

    // Water color palette
    Var deep_color = make_float3(0.0f, 0.1f, 0.3f);
    Var shallow_color = make_float3(0.0f, 0.4f, 0.6f);
    Var foam_color = make_float3(0.9f, 0.95f, 1.0f);
    Var highlight_color = make_float3(0.6f, 0.8f, 1.0f);

    // Base color based on wave height
    Var t = clamp(h * 0.5f + 0.5f, 0.0f, 1.0f);
    Var base_color = lerp(deep_color, shallow_color, t);

    // Add caustics based on slope
    Var caustics = smoothstep(0.1f, 0.5f, slope);
    base_color = lerp(base_color, highlight_color, caustics * 0.5f);

    // Foam at wave peaks
    Var foam = smoothstep(0.5f, 1.0f, h);
    base_color = lerp(base_color, foam_color, foam * 0.7f);

    output.write(uv, make_float4(base_color, 1.0f));
};
```

#### Step 5: Main Simulation Loop

```cpp
while (!window.should_close()) {
    // Handle mouse interaction
    if (mouse_down) {
        uint sim_x = clamp(cast<uint>(mouse_pos.x / scale), 0u, width - 1);
        uint sim_y = clamp(cast<uint>(mouse_pos.y / scale), 0u, height - 1);
        
        // Interpolate for smooth trails when dragging
        stream << drop_shader(height_curr, make_uint2(sim_x, sim_y), -0.5f)
                      .dispatch(width, height);
    }

    // Update wave simulation (multiple steps per frame)
    for (int step = 0; step < 4; step++) {
        stream << wave_shader(height_prev, height_curr, height_next).dispatch(width, height);
        std::swap(height_prev, height_curr);
        std::swap(height_curr, height_next);
    }

    // Render
    stream << render(height_curr, display, time).dispatch(width, height)
           << swap_chain.present(display);
}
```

#### Key Concepts

1. **Finite Differences**: Approximate spatial derivatives using neighbor samples
2. **Time Integration**: Three-buffer scheme for second-order time integration (Verlet-style)
3. **Stability**: CFL condition requires careful parameter tuning
4. **Caustics**: Surface slope creates realistic water shimmer

#### Tips for Better Results

1. **Parameters**: Try `c = 0.2f`, `dt = 1.0f`, `damping = 0.995f` for nice propagation
2. **Multiple Steps**: Run 4 simulation steps per frame for faster wave propagation
3. **Smooth Trails**: Interpolate droplet positions when dragging mouse
4. **Clamped Boundaries**: Edges act as walls (wave reflection) rather than wrapping

#### Exercises

1. **Variable Depth**: Make wave speed vary by location for refraction effects
2. **Obstacles**: Add static obstacles that block/reflect waves
3. **Rain**: Add periodic random droplets for ambient activity
4. **Underwater**: Add objects beneath the surface with distortion

---

## Image Processing Pipeline

Build a multi-stage image processing pipeline demonstrating **separable filters**, **convolution**, and **multi-pass rendering**.

### Final Result

A real-time image processing demo showing Gaussian blur, Sobel edge detection, and creative compositing.

### Step-by-Step Implementation

#### Step 1: Generate Test Pattern

```cpp
Kernel2D generate_pattern = [](ImageFloat image) noexcept {
    set_block_size(16, 16, 1);
    Var uv = make_float2(dispatch_id().xy()) / make_float2(dispatch_size().xy());

    // Checkerboard
    Var checker = sin(uv.x * 20.0f) * sin(uv.y * 20.0f);
    
    // Concentric circles
    Var center = uv - make_float2(0.5f);
    Var radius = length(center);
    Var circles = sin(radius * 50.0f);
    
    // Combine patterns
    Var pattern = checker * 0.3f + circles * 0.4f;
    
    // Add "face-like" features
    Var eye1 = exp(-length(uv - make_float2(0.35f, 0.4f)) * 50.0f);
    Var eye2 = exp(-length(uv - make_float2(0.65f, 0.4f)) * 50.0f);
    Var mouth = exp(-pow(uv.y - 0.6f - 0.1f * pow(uv.x - 0.5f, 2.0f), 2.0f) * 200.0f);

    Var final_val = pattern * 0.5f + 0.5f + eye1 * 0.3f + eye2 * 0.3f + mouth * 0.2f;
    image.write(dispatch_id().xy(), make_float4(make_float3(final_val), 1.0f));
};
```

#### Step 2: Separable Gaussian Blur

Instead of a 2D convolution (81 samples), we do two 1D passes (18 samples total):

```cpp
// Horizontal pass
Kernel2D gaussian_blur_h = [](ImageFloat input, ImageFloat output) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    // Gaussian weights (sigma=2.0)
    Var weights = make_float9(0.006f, 0.028f, 0.084f, 0.168f, 0.224f,
                              0.168f, 0.084f, 0.028f, 0.006f);
    Var offsets = make_int9(-4, -3, -2, -1, 0, 1, 2, 3, 4);

    Var sum = make_float3(0.0f);
    for (uint i = 0u; i < 9u; i++) {
        Var sample_uv = make_uint2(
            clamp(cast<int>(uv.x) + offsets[i], 0, cast<int>(size.x) - 1),
            uv.y);
        sum += input.read(sample_uv).xyz() * weights[i];
    }
    output.write(uv, make_float4(sum, 1.0f));
};

// Vertical pass (similar structure)
Kernel2D gaussian_blur_v = [](ImageFloat input, ImageFloat output) noexcept { ... };
```

#### Step 3: Sobel Edge Detection

```cpp
Kernel2D sobel_edge = [](ImageFloat input, ImageFloat output) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    auto sample = [&](Int2 offset) noexcept {
        Var sample_uv = make_uint2(
            clamp(cast<int>(uv.x) + offset.x, 0, cast<int>(size.x) - 1),
            clamp(cast<int>(uv.y) + offset.y, 0, cast<int>(size.y) - 1));
        return input.read(sample_uv).x;  // Luminance
    };

    // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
    Var gx = sample(make_int2(-1, -1)) * -1.0f +
             sample(make_int2(1, -1)) * 1.0f +
             sample(make_int2(-1, 0)) * -2.0f +
             sample(make_int2(1, 0)) * 2.0f +
             sample(make_int2(-1, 1)) * -1.0f +
             sample(make_int2(1, 1)) * 1.0f;

    // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
    Var gy = sample(make_int2(-1, -1)) * -1.0f +
             sample(make_int2(0, -1)) * -2.0f +
             sample(make_int2(1, -1)) * -1.0f +
             sample(make_int2(-1, 1)) * 1.0f +
             sample(make_int2(0, 1)) * 2.0f +
             sample(make_int2(1, 1)) * 1.0f;

    // Gradient magnitude and direction
    Var magnitude = sqrt(gx * gx + gy * gy);
    Var angle = atan2(gy, gx) / 3.14159f * 0.5f + 0.5f;
    
    output.write(uv, make_float4(magnitude, angle, 0.0f, 1.0f));
};
```

#### Step 4: Creative Compositing

```cpp
Kernel2D composite = [](ImageFloat original, ImageFloat blurred,
                       ImageFloat edges, ImageFloat output, Float edge_intensity) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();

    Var orig = original.read(uv);
    Var blur = blurred.read(uv);
    Var edge = edges.read(uv);

    // Unsharp mask sharpening
    Var sharpen_amount = 1.5f;
    Var sharpened = orig.xyz() + (orig.xyz() - blur.xyz()) * sharpen_amount;
    sharpened = clamp(sharpened, 0.0f, 1.0f);

    // Edge overlay with cyan color
    Var edge_color = make_float3(0.0f, 0.8f, 1.0f) * edge.x * edge_intensity;
    Var final_color = lerp(sharpened, edge_color, edge.x * 0.5f);

    // Vignette effect
    Var coord = make_float2(uv) / make_float2(dispatch_size().xy());
    Var center_dist = length(coord - make_float2(0.5f));
    Var vignette = 1.0f - center_dist * 0.5f;

    output.write(uv, make_float4(final_color * vignette, 1.0f));
};
```

#### Step 5: Pipeline Execution

```cpp
// Create intermediate images
Image<float> source_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
Image<float> temp_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
Image<float> blurred_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
Image<float> edge_image = device.create_image<float>(PixelStorage::FLOAT4, width, height);

// Generate source
stream << pattern_shader(source_image).dispatch(width, height);

// Run pipeline once
stream << blur_h_shader(source_image, temp_image).dispatch(width, height)
       << blur_v_shader(temp_image, blurred_image).dispatch(width, height)
       << sobel_shader(source_image, edge_image).dispatch(width, height);

// Animate in main loop
while (!window.should_close()) {
    float edge_intensity = (sinf(time) + 1.0f) * 0.5f;  // Pulsing effect
    stream << composite_shader(source_image, blurred_image, edge_image,
                               display, edge_intensity).dispatch(width, height)
           << swap_chain.present(display);
}
```

#### Key Concepts

1. **Separable Filters**: Decompose 2D convolution into two 1D passes for efficiency
2. **Multi-Pass Rendering**: Chain multiple kernels for complex effects
3. **Unsharp Mask**: Sharpening using `original + (original - blurred) * amount`

#### Exercises

1. **Bloom Effect**: Add a bright-pass filter before blur
2. **Custom Kernels**: Implement emboss or motion blur filters
3. **Live Input**: Use camera input instead of procedural pattern

---

## N-Body Gravitational Simulation

Simulate thousands of stars interacting through gravity. This demonstrates **particle systems**, **tile-based optimization**, and **3D rendering** on the GPU.

### Final Result

A rotating galaxy simulation with thousands of gravitationally interacting particles.

### Step-by-Step Implementation

#### Step 1: Particle Structure

```cpp
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
    float pad[3];  // Alignment padding
};

LUISA_STRUCT(Particle, position, velocity, mass, pad) {};
```

#### Step 2: Galaxy Initialization

```cpp
// Initialize in a disk-like galaxy configuration
luisa::vector<Particle> host_particles(n_particles);
for (uint i = 0u; i < n_particles; i++) {
    float radius = dist_radius(rng);
    float angle = dist_angle(rng);
    float height = (rng() / float(UINT32_MAX) - 0.5f) * 0.1f;

    // Position in disk
    float3 pos{
        radius * cosf(angle),
        height,
        radius * sinf(angle)};

    // Tangential velocity for orbital motion
    float orbital_speed = sqrtf(G * 1000.0f / radius);
    float3 vel{
        -orbital_speed * sinf(angle),
        0.0f,
        orbital_speed * cosf(angle)};

    host_particles[i] = Particle{
        .position = pos,
        .velocity = vel,
        .mass = dist_mass(rng),
        .pad = {0.0f, 0.0f, 0.0f}};
}
stream << particles_read.copy_from(host_particles.data()) << synchronize();
```

#### Step 3: N-Body Physics Kernel

```cpp
Kernel1D nbody_kernel = [&](BufferVar<Particle> read_buf, 
                            BufferVar<Particle> write_buf) noexcept {
    set_block_size(tile_size);
    Var idx = dispatch_id().x;
    Var p = read_buf.read(idx);

    // Accumulate gravitational forces
    Var force = make_float3(0.0f);

    // Tile loop - process particles in chunks
    for (uint tile = 0u; tile < n_particles; tile += tile_size) {
        for (uint j = 0u; j < tile_size; j++) {
            Var other_idx = tile + j;
            
            // Skip self-interaction to avoid infinity
            if_(other_idx != idx, [&] {
                Var other = read_buf.read(other_idx);
                Var r = other.position - p.position;
                Var dist_sq = dot(r, r) + softening * softening;
                Var dist = sqrt(dist_sq);
                Var f = G * p.mass * other.mass / dist_sq;
                force += f * r / dist;  // F = G*m1*m2/r² * direction
            });
        }
    }

    // Euler integration
    Var new_vel = p.velocity + force / p.mass * dt;
    Var new_pos = p.position + new_vel * dt;

    write_buf.write(idx, Particle{
        .position = new_pos,
        .velocity = new_vel,
        .mass = p.mass,
        .pad = {0.0f, 0.0f, 0.0f}});
};
```

#### Step 4: 3D Rendering Kernel

```cpp
Kernel2D render_kernel = [&](BufferVar<Particle> particles, 
                             ImageFloat image, Float rot_x, Float rot_y) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();
    Var size = dispatch_size().xy();

    // Clear background
    image.write(uv, make_float4(0.02f, 0.02f, 0.05f, 1.0f));

    Var color = make_float3(0.0f);

    // Project particles
    for (uint i = 0u; i < n_particles; i += 64u) {
        Var p = particles.read(i);

        // Rotation around Y axis
        Var cos_y = cos(rot_y);
        Var sin_y = sin(rot_y);
        Var x1 = p.position.x * cos_y - p.position.z * sin_y;
        Var z1 = p.position.x * sin_y + p.position.z * cos_y;
        Var y1 = p.position.y;

        // Rotation around X axis
        Var cos_x = cos(rot_x);
        Var sin_x = sin(rot_x);
        Var y2 = y1 * cos_x - z1 * sin_x;
        Var z2 = y1 * sin_x + z1 * cos_x;

        // Perspective projection
        Var distance = 5.0f + z2;
        Var scale = 2.0f / distance;
        Var screen_x = (x1 * scale + 1.0f) * 0.5f * cast<float>(size.x);
        Var screen_y = (y2 * scale + 1.0f) * 0.5f * cast<float>(size.y);

        // Gaussian glow
        Var dx = cast<float>(uv.x) - screen_x;
        Var dy = cast<float>(uv.y) - screen_y;
        Var d_sq = dx * dx + dy * dy;
        Var intensity = exp(-d_sq / (50.0f * scale));

        // Color based on particle index
        Var particle_color = make_float3(
            0.8f + 0.2f * sin(cast<float>(i)),
            0.6f + 0.3f * cos(cast<float>(i) * 1.3f),
            1.0f);

        color += particle_color * intensity * 0.1f;
    }

    // Motion blur effect
    Var old = image.read(uv).xyz();
    Var final_color = lerp(old, min(color, 1.0f), 0.3f);
    image.write(uv, make_float4(final_color, 1.0f));
};
```

#### Step 5: Main Loop

```cpp
while (!window.should_close()) {
    // Auto-rotate
    rot_y += 0.002f;

    // Physics update (double buffering)
    stream << nbody_shader(particles_read, particles_write).dispatch(n_particles);
    std::swap(particles_read, particles_write);

    // Render every few frames
    if (frame % 3u == 0u) {
        stream << render_shader(particles_read, display, rot_x, rot_y)
                      .dispatch(width, height)
               << swap_chain.present(display);
    }
    frame++;
}
```

#### Key Concepts

1. **Softening**: Add small epsilon to prevent division by zero at close range
2. **Tile-Based Processing**: Process in chunks for cache efficiency
3. **Double Buffering**: Read from one buffer, write to another, then swap
4. **Gaussian Splatting**: Render particles as soft glows rather than points

#### Exercises

1. **Barnes-Hut**: Implement O(n log n) hierarchical approximation
2. **Collision Detection**: Add particle collisions
3. **Different Forces**: Try Coulomb (electrostatic) or Lennard-Jones (molecular)

---

## Fire Particle System

Create a mesmerizing fire simulation using 65,000 GPU particles with physics-based motion, procedural turbulence, and temperature-driven color gradients. Watch as flames dance and flicker with realistic fluid-like behavior.

### Final Result

A captivating campfire-like effect with:
- **White-hot cores** at the base where particles are born
- **Yellow-orange flames** rising and swirling upward
- **Red embers** cooling as they float higher
- **Procedural turbulence** creating realistic flickering motion
- **Interactive wind** (press SPACE) that bends the flames sideways

The particles cycle through their lifecycle endlessly—born hot and bright, rising with buoyancy, cooling as they age, fading to smoke, then respawning to keep the fire alive.

### Key Concepts

**Particle Lifecycle Management**: Each particle tracks its own state:
```cpp
struct FireParticle {
    float3 position;    // Position in 3D space
    float lifetime;     // Seconds remaining before respawn
    float3 velocity;    // Current movement direction
    float temperature;  // 1.0 = white hot, 0.0 = cold
    float size;         // Particle radius for rendering
};
```

This structure enables individual particle behavior while the GPU processes thousands in parallel.

```cpp
struct FireParticle {
    float3 position;
    float lifetime;
    float3 velocity;
    float temperature;
    float size;
};
```

**Procedural Turbulence**: We use a simple 3D noise function to create realistic turbulent motion:

```cpp
Callable noise3d = [](Float3 p) noexcept {
    Var i = floor(p);
    Var f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    // ... trilinear interpolation
};
```

**Temperature-Based Coloring**: The color gradient follows real fire physics:
- White: Hottest (1.0)
- Yellow: Hot (0.75-1.0)
- Orange: Warm (0.5-0.75)
- Red: Cool (0.25-0.5)
- Black: Cold (< 0.25)

### Step-by-Step Implementation

#### Step 1: Particle Update Kernel

```cpp
Kernel1D update_kernel = [&](BufferVar<FireParticle> particles, 
                             Float time, Float wind) noexcept {
    set_block_size(256);
    Var idx = dispatch_id().x;
    Var p = particles.read(idx);

    $if (p.lifetime > 0.0f) {
        // Apply gravity (negative = buoyancy)
        p.velocity.y += gravity * dt;
        
        // Add turbulence
        Var noise_pos = p.position * 2.0f + time;
        p.velocity.x += (noise3d(noise_pos) - 0.5f) * 2.0f * dt;
        
        // Wind effect
        p.velocity.x += wind * dt;
        
        // Update position
        p.position += p.velocity * dt;
        
        // Cool down
        p.temperature -= dt * 0.3f;
        p.lifetime -= dt;
    }
    $else {
        // Respawn with random properties
        Var seed = idx + cast<uint>(time * 1000.0f);
        // ... spawn new particle
    };

    particles.write(idx, p);
};
```

#### Step 2: Rendering with Additive Blending

```cpp
Kernel2D render_kernel = [&](BufferVar<FireParticle> particles, 
                             ImageFloat image) noexcept {
    // For each pixel, accumulate contributions from all particles
    Var color = make_float3(0.0f);
    
    for (uint i = 0u; i < n_particles; i += 256u) {
        Var p = particles.read(i);
        
        // Project to screen space
        Var screen_pos = project(p.position);
        Var dist = length(pixel_pos - screen_pos);
        
        // Gaussian intensity falloff
        Var intensity = exp(-dist_sq / (p.size * size_scale));
        
        // Temperature-based color
        Var fire_color = get_fire_color(p.temperature);
        color += fire_color * intensity;
    }
    
    image.write(uv, make_float4(color, 1.0f));
};
```

#### Exercises

1. **Smoke**: Add a smoke trail that rises from cooling particles
2. **Sparks**: Add smaller, faster particles that shoot upward occasionally
3. **Interactive**: Make the fire react to mouse position as a wind source

---

## Reaction-Diffusion Simulation

Watch mathematical magic unfold as beautiful organic patterns emerge from simple chemical rules. The Gray-Scott model simulates how two chemicals interact to create mesmerizing textures resembling coral reefs, zebra stripes, fingerprints, and more.

### What You'll See

Starting from a tiny seed in the center, watch as intricate patterns **grow and evolve** across the screen:
- **Coral Growth**: Branching, tree-like structures spreading outward
- **Fingerprint Patterns**: Meandering, maze-like lines
- **Leopard Spots**: Isolated dots that grow and stabilize
- **Stripes**: Parallel waves that propagate across the field

Press keys **1-4** to switch between pattern types in real-time and observe how the same equations produce wildly different results!

### The Gray-Scott Model

Two virtual chemicals dance on the grid:
- **U** (inhibitor, shown as blue): The "food" that prevents reaction
- **V** (activator, shown as red/yellow): The "predator" that consumes U and spreads

The mathematical rules that create life-like patterns:
```
du/dt = Du * ∇²u - u*v² + F*(1-u)   // U diffuses and gets consumed
dv/dt = Dv * ∇²v + u*v² - (F+k)*v   // V diffuses, grows, and dies
```

Where:
- **Du, Dv**: How fast each chemical spreads (U spreads faster than V)
- **F (Feed)**: Rate that fresh U is added
- **k (Kill)**: Rate that V is removed
- **u*v²**: The reaction where V consumes U

### Pattern Types

By adjusting feed rate (F) and kill rate (k), we get different patterns:

| Pattern | F | k | Description |
|---------|---|---|-------------|
| Coral | 0.035 | 0.06 | Branching coral-like growth |
| Fingerprint | 0.037 | 0.06 | Meandering lines |
| Spots | 0.03 | 0.062 | Isolated spots |
| Stripes | 0.03 | 0.054 | Parallel stripes |

### Step-by-Step Implementation

#### Step 1: Chemical State Storage

```cpp
// Two buffers for ping-pong rendering
Image<float> u_prev, u_curr;  // Inhibitor
Image<float> v_prev, v_curr;  // Activator
```

#### Step 2: Reaction-Diffusion Kernel

```cpp
Kernel2D rd_step = [&](ImageFloat u_in, ImageFloat v_in,
                       ImageFloat u_out, ImageFloat v_out,
                       Float F, Float k, Float Du, Float Dv) noexcept {
    set_block_size(16, 16, 1);
    Var uv = dispatch_id().xy();

    // Sample current state
    Var u = u_in.read(uv).x;
    Var v = v_in.read(uv).x;

    // Compute Laplacian (5-point stencil)
    Var u_lap = (sample(u_in, uv, {-1,0}) + sample(u_in, uv, {1,0}) +
                 sample(u_in, uv, {0,-1}) + sample(u_in, uv, {0,1}) +
                 u) * 0.2f - u;

    // Reaction term
    Var reaction = u * v * v;

    // Update
    Var u_new = u + dt * (Du * u_lap - reaction + F * (1.0f - u));
    Var v_new = v + dt * (Dv * v_lap + reaction - (F + k) * v);

    u_out.write(uv, make_float4(u_new, 0, 0, 1));
    v_out.write(uv, make_float4(v_new, 0, 0, 1));
};
```

#### Step 3: Visualization

```cpp
Kernel2D visualize = [&](ImageFloat u, ImageFloat v, ImageFloat output) {
    Var uv = dispatch_id().xy();
    Var u_val = u.read(uv).x;
    Var v_val = v.read(uv).x;
    
    // U = blue, V = red/yellow
    Var color = make_float3(v_val * 1.5f,  // Red
                           v_val * u_val * 0.5f,  // Green
                           u_val * 0.8f);  // Blue
    
    output.write(uv, make_float4(color, 1.0f));
};
```

#### Key Concepts

1. **Reaction**: V consumes U when they meet (u*v² term)
2. **Diffusion**: U spreads faster than V (Du > Dv)
3. **Feed/Kill**: Constant injection of U, removal of V
4. **Instability**: Small perturbations grow into patterns

#### Exercises

1. **Multiscale**: Use different F/k values in different regions
2. **3D Extension**: Extend to 3D and render with volume ray marching
3. **Interactive Brush**: Paint V values with mouse to guide pattern growth

---

## Voxel Ray Tracer

Explore a procedurally generated 3D voxel world in real-time! This example renders a Minecraft-style landscape with terrain, trees, and floating islands using ray marching through a voxel grid. Use arrow keys to rotate the view and W/S to zoom.

### What You'll See

A charming voxel scene featuring:
- **Rolling Hills**: Procedurally generated terrain with sine-wave elevation
- **Green Grass Tops**: Different block types for grass, dirt, and stone
- **Trees**: Scattered throughout with wood trunks and leafy canopies
- **Floating Magic Orb**: A mysterious purple sphere hovering above the landscape
- **Interactive Camera**: Rotate with arrow keys, zoom with W/S

### How Ray Voxel Traversal Works

Unlike traditional polygon rendering, we trace rays through a 3D grid:

1. **Ray-Box Intersection**: Calculate where the camera ray enters the voxel grid bounding box
2. **DDA Traversal**: Use the Digital Differential Analyzer algorithm to step through voxels in order along the ray, similar to Bresenham's line algorithm in 3D
3. **Hit Detection**: At each voxel, check if it's solid. If yes, render its color. If no, continue stepping.

This technique, used in games like Minecraft and Teardown, allows rendering millions of voxels efficiently without storing explicit geometry.

### Step-by-Step Implementation

#### Step 1: Voxel Grid Storage

```cpp
static constexpr uint grid_size = 64;
Buffer<uint> voxel_grid = device.create_buffer<uint>(
    grid_size * grid_size * grid_size);

// Voxel types: 0=empty, 1=dirt, 2=grass, 3=stone, etc.
```

#### Step 2: Ray-Box Intersection

```cpp
Callable intersect_box = [](Float3 ray_origin, Float3 ray_dir,
                            Float3 box_min, Float3 box_max) {
    // Slab method for AABB intersection
    Var tmin = (box_min - ray_origin) / ray_dir;
    Var tmax = (box_max - ray_origin) / ray_dir;
    // ... swap if necessary
    return make_float2(entry_t, exit_t);
};
```

#### Step 3: DDA Traversal

```cpp
Callable trace_ray = [](Float3 origin, Float3 dir, BufferVar<uint> voxels) {
    // Find entry point
    Var box_hit = intersect_box(origin, dir, grid_min, grid_max);
    $if (box_hit.x < 0.0f) { return miss_color; };

    // Initialize DDA
    Var t = box_hit.x;
    Var vx = floor(origin.x + dir.x * t);
    // ... calculate step direction and initial distances

    // Traverse
    $for (step, max_steps) {
        Var voxel = read_voxel(voxels, vx, vy, vz);
        $if (voxel > 0u) {
            // Hit! Return voxel color
            return get_voxel_color(voxel);
        };
        
        // Step to next voxel
        $if (next_tx < next_ty & next_tx < next_tz) {
            vx += step_x;
            next_tx += delta_tx;
        }
        // ... similar for y, z
    };
    
    return sky_color;  // Missed everything
};
```

#### Step 4: Camera and Rendering

```cpp
Kernel2D render = [&](ImageFloat image, Float3 cam_pos, 
                      Float2 cam_rot, BufferVar<uint> voxels) {
    Var uv = dispatch_id().xy();
    Var ndc = (make_float2(uv) / resolution) * 2.0f - 1.0f;
    
    // Generate ray direction from camera
    Var ray_dir = normalize(make_float3(
        ndc.x * aspect * fov,
        -ndc.y * fov,
        -1.0f
    ));
    
    // Apply camera rotation
    ray_dir = rotate_y(ray_dir, cam_rot.x);
    ray_dir = rotate_x(ray_dir, cam_rot.y);
    
    // Trace and write color
    Var color = trace_ray(cam_pos, ray_dir, voxels);
    image.write(uv, make_float4(color, 1.0f));
};
```

#### Exercises

1. **Lighting**: Add directional lighting with voxel self-shadowing
2. **Materials**: Add emission for some voxel types (glowing blocks)
3. **Destruction**: Add ability to remove voxels with mouse clicks
4. **LOD**: Implement level-of-detail for distant chunks

---

## Black Hole Renderer

Experience the mind-bending visual effects of Einstein's theory of general relativity! This example renders a Schwarzschild black hole with realistic gravitational lensing, an accretion disk, and relativistic Doppler effects—just like in the movie *Interstellar*.

### What You'll See

- **The Event Horizon**: A perfectly black sphere where light cannot escape
- **Gravitational Lensing**: Stars behind the black hole bend around it, creating a distorted "Einstein ring"
- **The Photon Sphere**: A thin bright ring just outside the event horizon where light orbits the black hole
- **Accretion Disk**: A swirling disk of superheated matter orbiting the black hole
- **Doppler Beaming**: The side of the disk rotating toward you appears brighter (blueshifted), while the receding side appears dimmer (redshifted)

- **Left Mouse Drag**: Orbit around the black hole
- **Right Mouse Drag**: Roll/rotate the view (rotate around viewing direction)
- **Scroll Wheel** or **+/-**: Zoom in/out

Watch how the gravitational lensing distorts the background starfield!

### The Physics

**Gravitational Lensing**: Massive objects bend spacetime, causing light to curve. The bending angle follows:
```
acceleration = -1.5 * GM / r³ * position
```
(This is 2x Newtonian gravity as predicted by general relativity)

**Photon Sphere**: At r = 1.5 × Rs (Schwarzschild radius), light can orbit the black hole in unstable circular paths.

**Accretion Disk Physics**:
- **Temperature**: Hotter near the center (~T ∝ r^(-3/4))
- **Doppler Effect**: Approaching matter appears brighter/bluer
- **Gravitational Redshift**: Light loses energy climbing out of the gravity well

### Step-by-Step Implementation

#### Step 1: Ray Setup with Camera Transformation

```cpp
// Camera position in orbit around black hole
Var cam_pos = make_float3(
    distance * sin(rot_y) * cos(rot_x),
    distance * sin(rot_x),  
    distance * cos(rot_y) * cos(rot_x)
);

// Ray direction through each pixel
Var ray_dir = normalize(forward + ndc.x * right * fov + ndc.y * up * fov);
```

#### Step 2: Gravitational Ray Marching

```cpp
$for (step, max_steps) {
    Var r = length(pos);
    
    // Hit event horizon?
    $if (r < bh_radius) { $break; };
    
    // Gravitational bending (general relativity)
    Var accel = -1.5f * bh_mass / (r * r * r) * pos;
    dir = normalize(dir + accel * dt);
    
    // Step forward
    pos = pos + dir * dt;
};
```

#### Step 3: Accretion Disk Rendering

The accretion disk is a thin sheet in the XZ plane. To render it correctly with proper depth ordering, we detect when the ray crosses the plane and determine whether we're hitting the front (visible) or back (hidden behind black hole) side:

```cpp
// Track previous position to detect plane crossing
Var prev_y = cam_pos.y;

$for (step, max_steps) {
    // ... ray marching loop ...
    
    // Check if we crossed the disk plane (XZ plane at y=0)
    Var crossed_plane = (prev_y > 0.0f & pos.y <= 0.0f) | 
                        (prev_y < 0.0f & pos.y >= 0.0f);
    
    $if (crossed_plane & r > inner_radius & r < outer_radius) {
        // Interpolate to find exact intersection point
        Var t = abs(prev_y) / (abs(prev_y) + abs(pos.y));
        Var intersect_pos = prev_pos * (1.0f - t) + pos * t;
        Var intersect_r = length(intersect_pos);
        
        // Calculate disk color (temperature, Doppler beaming, etc.)
        // ... color calculation code ...
        
        // Determine if this is front or back side
        // Front side: ray moving toward the plane from camera
        Var moving_toward_plane = (cam_pos.y > 0.0f & pos.y < prev_y) | 
                                  (cam_pos.y < 0.0f & pos.y > prev_y);
        
        $if (moving_toward_plane) {
            // Front side - use this sample
            disk_color = local_disk_color;
            hit_disk = true;
        }
        $else {
            // Back side - only use if front wasn't hit
            // (handles edge case of camera inside disk radius)
            $if (!hit_disk) { disk_color = local_disk_color; };
        };
    };
    
    prev_y = pos.y;  // Update for next iteration
    // ... continue ray marching ...
};

#### Step 4: Background Starfield

```cpp
// Procedural stars for lensing demonstration
auto get_star_color = [&](Float3 rd) noexcept {
    Var star_noise = fract(sin(rd.x * 123.45f + ...) * 43758.5453f);
    Var star_brightness = ite(star_noise > 0.995f, star_noise, 0.0f);
    return make_float3(star_brightness);
};
```

### Key Concepts

1. **Geodesic Ray Tracing**: Light follows curved paths in curved spacetime
2. **Relativistic Doppler Effect**: Motion toward/away affects brightness and color
3. **Gravitational Redshift**: Light loses energy escaping gravity
4. **Photon Sphere**: Where light can orbit the black hole
5. **Depth-Ordered Rendering**: The accretion disk must be rendered with proper front/back ordering so the back side (behind the black hole) doesn't incorrectly show through the event horizon

### Exercises

1. **Spinning Black Hole**: Add frame-dragging effects (Kerr metric) for a rotating black hole
2. **Multiple Black Holes**: Simulate a binary black hole system
3. **Time Dilation**: Visualize clock slowing near the event horizon
4. **Gravitational Waves**: Add ripples to the accretion disk from merging black holes

---

## Summary

Congratulations! You've explored eleven fascinating GPU computing tutorials that demonstrate the breadth of what's possible with LuisaCompute:

### Visual Effects & Rendering
1. **Mandelbrot**: Mathematical beauty through iterative complex numbers
2. **Path Tracer**: Physically-based global illumination with ray tracing
3. **Voxel Ray Tracer**: Real-time ray marching through 3D voxel grids
4. **Black Hole**: Gravitational lensing and relativistic ray tracing (Interstellar-style)

### Physics Simulations
5. **MPM**: Material Point Method for fluid/solid simulation
6. **Wave Equation**: Interactive water ripples with caustics rendering
7. **N-Body**: Gravitational galaxy simulation with 4,000+ particles

### Pattern Formation & Cellular Automata
8. **Game of Life**: Classic cellular automata with emergent behavior
9. **Reaction-Diffusion**: Gray-Scott chemical patterns mimicking nature

### Image Processing & Particles
10. **Image Processing**: Multi-pass Gaussian blur and edge detection
11. **Fire Particles**: 65,000 temperature-driven animated particles

### Interactive Features Summary

| Tutorial | Interaction |
|----------|-------------|
| Wave Equation | Click & drag to create ripples |
| N-Body | Mouse drag to rotate, scroll/+/- to zoom |
| Game of Life | Watch evolution, R to reset |
| Fire | SPACE to toggle wind |
| Reaction-Diffusion | 1-4 to switch patterns, R to reset |
| Voxel Ray Tracer | Arrow keys to rotate, W/S to zoom |
| Black Hole | Left drag=Orbit, Right drag=Roll, Scroll/+/-=Zoom |

### What You've Learned

1. **Mandelbrot**: Basic kernel structure, loops, and image output
2. **Path Tracer**: Ray tracing, acceleration structures, sampling
3. **MPM**: Complex simulation with multiple kernels, atomic operations
4. **Game of Life**: Cellular automata, ping-pong buffering
5. **Wave Equation**: PDE solving, interactive simulation
6. **Image Processing**: Multi-pass pipelines, convolution
7. **N-Body**: Particle systems, tile-based optimization
8. **Fire Particles**: Particle systems with lifecycle management and procedural noise
9. **Reaction-Diffusion**: Coupled PDEs and pattern formation
10. **Voxel Ray Tracer**: 3D grid traversal and ray-box intersection
11. **Black Hole**: Gravitational lensing, geodesic ray tracing, relativistic effects

### Performance Tips

1. **Use appropriate block sizes**: 16×16 for 2D kernels, 256 for 1D
2. **Minimize memory transfers**: Keep data on GPU when possible
3. **Batch operations**: Use `CommandList` for multiple dispatches
4. **Profile with tools**: Use Nsight, RenderDoc, or vendor tools

### Next Steps

- Explore the [LuisaCompute tests](https://github.com/LuisaGroup/LuisaCompute/tree/stable/src/tests) for more examples
- Check [LuisaRender](https://github.com/LuisaGroup/LuisaRender) for production rendering
- Try implementing your own simulations!
