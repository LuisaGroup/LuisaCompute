// Material Point Method (MPM) simulation in 2D.
//
// MPM is a hybrid Lagrangian-Eulerian method for simulating deformable materials.
// This implementation follows the "Moving Least Squares MPM" approach with:
// - Particles carrying mass, velocity, and deformation gradient (C)
// - A background Eulerian grid for computing forces
// - Quadratic B-spline interpolation kernels (APIC/MLS)
//
// Physics model:
// - Neo-Hookean elastic material
// - Explicit time integration
// - Boundary conditions with sticky walls
//
// Reference: "The Material Point Method for Simulating Continuum Materials"
// by Jiang et al., 2016

#include <random>
#include <fstream>
#include <chrono>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/logging.h>
#include <luisa/gui/window.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>

int main(int argc, char *argv[]) {

    using namespace luisa;
    using namespace luisa::compute;

    // Helper lambda for squaring values
    auto sqr = [](auto x) noexcept { return x * x; };

    // Initialize compute context
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    // Simulation parameters
    static constexpr uint n_grid = 200u;       // Grid resolution
    static constexpr uint n_steps = 24u;       // Substeps per frame

    static constexpr uint n_particles = n_grid * n_grid / 2u;  // Number of particles
    static constexpr float dx = 1.f / n_grid;  // Grid cell size
    static constexpr float dt = 1e-4f;         // Time step
    static constexpr float p_rho = 1.f;        // Particle density
    static constexpr float p_vol = sqr(dx * .5f);  // Particle volume (dx/2)^2
    static constexpr float p_mass = p_rho * p_vol; // Particle mass
    static constexpr float gravity = 9.8f;     // Gravitational acceleration
    static constexpr uint bound = 3u;          // Boundary thickness (cells)
    static constexpr float E = 400.f;          // Young's modulus (elasticity)

    static constexpr uint resolution = 1024u;  // Display resolution

    // Particle state buffers (Lagrangian)
    Buffer<float2> x = device.create_buffer<float2>(n_particles);  // Positions
    Buffer<float2> v = device.create_buffer<float2>(n_particles);  // Velocities
    Buffer<float2x2> C = device.create_buffer<float2x2>(n_particles);  // Affine momentum (APIC)
    Buffer<float> J = device.create_buffer<float>(n_particles);    // Deformation gradient determinant

    // Grid state buffers (Eulerian)
    Buffer<float> grid_v = device.create_buffer<float>(n_grid * n_grid * 2u);  // Grid velocities (vx, vy)
    Buffer<float> grid_m = device.create_buffer<float>(n_grid * n_grid);       // Grid masses

    // Setup graphics pipeline
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Window window{"MPM88", resolution, resolution};
    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = make_uint2(resolution),
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 2,
        });
    Image<float> display = device.create_image<float>(swap_chain.backend_storage(), make_uint2(resolution));

    // Helper: compute 1D grid index from 2D coordinates with clamping
    auto index = [](UInt2 xy) noexcept {
        auto p = clamp(xy, static_cast<uint2>(0), static_cast<uint2>(n_grid - 1));
        return p.x + p.y * n_grid;
    };

    // Helper: compute outer product of two vectors (a * b^T)
    auto outer_product = [](Float2 a, Float2 b) noexcept {
        return make_float2x2(a[0] * b[0], a[1] * b[0], a[0] * b[1], a[1] * b[1]);
    };

    // Helper: compute matrix trace (sum of diagonal elements)
    auto trace = [](Float2x2 m) noexcept { return m[0][0] + m[1][1]; };

    auto roundtrip_kernel = [](auto &&kernel) {
        auto xir_module = xir::ast_to_xir_translate(kernel.function()->function(), {});
        auto xir_function = xir_module->function_list().front();
        LUISA_ASSERT(xir_function->isa<xir::KernelFunction>(), "Expected XIR function to be a kernel.");
        return xir::XIR2AST::build(static_cast<const xir::KernelFunction *>(xir_function));
    };

    // Kernel: Clear grid velocities and masses
    Kernel2D<Buffer<float>, Buffer<float>> clear_grid_kernel = [&](BufferVar<float> grid_v, BufferVar<float> grid_m) noexcept {
        UInt idx = index(dispatch_id().xy());
        grid_v.write(idx * 2u, 0.f);
        grid_v.write(idx * 2u + 1u, 0.f);
        grid_m.write(idx, 0.f);
    };
    auto clear_grid_ast = roundtrip_kernel(clear_grid_kernel);
    auto clear_grid = device.create<Shader2D<Buffer<float>, Buffer<float>>>(
        clear_grid_ast->function(),
        ShaderOption{.name = luisa::string{"xir2ast_mpm_clear_grid"}});

    // Kernel: Transfer particle data to grid (P2G)
    // Uses quadratic B-spline interpolation weights (APIC)
    Kernel1D<Buffer<float2>, Buffer<float2>, Buffer<float2x2>, Buffer<float>, Buffer<float>, Buffer<float>> point_to_grid_kernel =
        [&](BufferVar<float2> x,
            BufferVar<float2> v,
            BufferVar<float2x2> C,
            BufferVar<float> J,
            BufferVar<float> grid_v,
            BufferVar<float> grid_m) noexcept {
        UInt p = dispatch_id().x;

        // Compute particle position in grid coordinates
        Float2 Xp = x.read(p) / dx;
        Int2 base = make_int2(Xp - 0.5f);
        Float2 fx = Xp - make_float2(base);

        // Quadratic B-spline interpolation weights
        // w[0] = 0.5 * (1.5 - fx)^2
        // w[1] = 0.75 - (fx - 1)^2
        // w[2] = 0.5 * (fx - 0.5)^2
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};

        // Compute stress from Neo-Hookean elasticity
        // stress = -4 * dt * E * volume * (J - 1) / dx^2
        Float stress = -4.f * dt * E * p_vol * (J.read(p) - 1.f) / sqr(dx);

        // Affine momentum from stress and velocity gradient
        // affine = stress * I + mass * C
        Float2x2 affine = make_float2x2(stress, 0.f, 0.f, stress) + p_mass * C.read(p);
        Float2 vp = v.read(p);

        // Scatter to 3x3 neighboring grid cells
        for (uint ii = 0; ii < 9; ii++) {
            int2 offset = make_int2(ii % 3, ii / 3);
            int i = offset.x;
            int j = offset.y;

            // Distance from particle to grid node
            Float2 dpos = (make_float2(offset) - fx) * dx;

            // Quadratic weight for this grid node
            Float weight = w[i].x * w[j].y;

            // Momentum contribution: weight * (mass * velocity + affine * dpos)
            Float2 vadd = weight * (p_mass * vp + affine * dpos);

            // Atomic add to grid (thread-safe accumulation)
            UInt idx = index(base + offset);
            grid_v.atomic(idx * 2u).fetch_add(vadd.x);
            grid_v.atomic(idx * 2u + 1u).fetch_add(vadd.y);
            grid_m.atomic(idx).fetch_add(weight * p_mass);
        }
    };
    auto point_to_grid_ast = roundtrip_kernel(point_to_grid_kernel);
    auto point_to_grid = device.create<Shader1D<Buffer<float2>, Buffer<float2>, Buffer<float2x2>, Buffer<float>, Buffer<float>, Buffer<float>>>(
        point_to_grid_ast->function(),
        ShaderOption{.name = luisa::string{"xir2ast_mpm_point_to_grid"}});

    // Kernel: Grid velocity update (explicit time integration)
    Kernel2D<Buffer<float>, Buffer<float>> simulate_grid_kernel = [&](BufferVar<float> grid_v, BufferVar<float> grid_m) noexcept {
        UInt2 coord = dispatch_id().xy();
        UInt i = index(coord);

        // Read grid velocity and mass
        Float2 v = make_float2(grid_v.read(i * 2u), grid_v.read(i * 2u + 1u));
        Float m = grid_m.read(i);

        // Normalize by mass (if mass > 0)
        v = ite(m > 0.f, v / m, v);

        // Apply gravity
        v.y -= dt * gravity;

        // Boundary conditions: sticky walls at domain boundaries
        // Zero velocity if moving into boundary
        v.x = ite((coord.x < bound & v.x < 0.f) | (coord.x + bound > n_grid & v.x > 0.f), 0.f, v.x);
        v.y = ite((coord.y < bound & v.y < 0.f) | (coord.y + bound > n_grid & v.y > 0.f), 0.f, v.y);

        // Write updated velocity back to grid
        grid_v.write(i * 2u, v.x);
        grid_v.write(i * 2u + 1u, v.y);
    };
    auto simulate_grid_ast = roundtrip_kernel(simulate_grid_kernel);
    auto simulate_grid = device.create<Shader2D<Buffer<float>, Buffer<float>>>(
        simulate_grid_ast->function(),
        ShaderOption{.name = luisa::string{"xir2ast_mpm_simulate_grid"}});

    // Kernel: Transfer grid data back to particles (G2P)
    Kernel1D<Buffer<float2>, Buffer<float2>, Buffer<float2x2>, Buffer<float>, Buffer<float>> grid_to_point_kernel =
        [&](BufferVar<float2> x,
            BufferVar<float2> v,
            BufferVar<float2x2> C,
            BufferVar<float> J,
            BufferVar<float> grid_v) noexcept {
        UInt p = dispatch_id().x;

        // Compute particle position in grid coordinates
        Float2 Xp = x.read(p) / dx;
        Int2 base = make_int2(Xp - 0.5f);
        Float2 fx = Xp - make_float2(base);

        // Same quadratic weights as P2G
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};

        Float2 new_v = def(make_float2(0.f));
        Float2x2 new_C = def(make_float2x2(0.f));

        // Gather from 3x3 neighboring grid cells
        for (uint ii = 0; ii < 9; ii++) {
            int2 offset = make_int2(ii % 3, ii / 3);
            int i = offset.x;
            int j = offset.y;

            Float2 dpos = (make_float2(offset) - fx) * dx;
            Float weight = w[i].x * w[j].y;
            UInt idx = index(base + offset);

            // Read grid velocity
            Float2 g_v = make_float2(grid_v.read(idx * 2u),
                                     grid_v.read(idx * 2u + 1u));

            // Accumulate velocity
            new_v += weight * g_v;

            // Accumulate velocity gradient (APIC)
            // new_C += 4 * weight * outer(g_v, dpos) / dx^2
            new_C = new_C + 4.f * weight * outer_product(g_v, dpos) / sqr(dx);
        }

        // Update particle state
        v.write(p, new_v);
        x.write(p, x.read(p) + new_v * dt);

        // Update deformation gradient: J *= (1 + dt * trace(C))
        J.write(p, J.read(p) * (1.f + dt * trace(new_C)));
        C.write(p, new_C);
    };
    auto grid_to_point_ast = roundtrip_kernel(grid_to_point_kernel);
    auto grid_to_point = device.create<Shader1D<Buffer<float2>, Buffer<float2>, Buffer<float2x2>, Buffer<float>, Buffer<float>>>(
        grid_to_point_ast->function(),
        ShaderOption{.name = luisa::string{"xir2ast_mpm_grid_to_point"}});

    // Single simulation substep
    auto substep = [&](CommandList &cmd_list) noexcept {
        cmd_list << clear_grid(grid_v, grid_m).dispatch(n_grid, n_grid)
                 << point_to_grid(x, v, C, J, grid_v, grid_m).dispatch(n_particles)
                 << simulate_grid(grid_v, grid_m).dispatch(n_grid, n_grid)
                 << grid_to_point(x, v, C, J, grid_v).dispatch(n_particles);
    };

    // Initialize particle state
    auto init = [&](Stream &stream) noexcept {
        luisa::vector<float2> x_init(n_particles);
        std::default_random_engine random{std::random_device{}()};
        std::uniform_real_distribution<float> uniform;

        // Initialize positions in a square block
        for (uint i = 0; i < n_particles; i++) {
            float rx = uniform(random);
            float ry = uniform(random);
            x_init[i] = make_float2(rx * .4f + .2f, ry * .4f + .2f);
        }

        luisa::vector<float2> v_init(n_particles, make_float2(0.f, -1.f));  // Initial downward velocity
        luisa::vector<float> J_init(n_particles, 1.f);  // Initial volume
        luisa::vector<float2x2> C_init(n_particles, make_float2x2(0.f));  // Initial affine momentum

        stream << x.copy_from(x_init.data())
               << v.copy_from(v_init.data())
               << J.copy_from(J_init.data())
               << C.copy_from(C_init.data())
               << synchronize();
    };

    // Kernel: Clear display with background color
    Kernel2D<Image<float>> clear_display_kernel = [&](ImageVar<float> display) noexcept {
        display.write(dispatch_id().xy(), make_float4(.1f, .2f, .3f, 1.f));
    };
    auto clear_display_ast = roundtrip_kernel(clear_display_kernel);
    auto clear_display = device.create<Shader2D<Image<float>>>(
        clear_display_ast->function(),
        ShaderOption{.name = luisa::string{"xir2ast_mpm_clear_display"}});

    // Kernel: Draw particles as 3x3 pixel squares
    Kernel1D<Buffer<float2>, Image<float>> draw_particles_kernel =
        [&](BufferVar<float2> x, ImageVar<float> display) noexcept {
        UInt p = dispatch_id().x;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                Int2 pos = make_int2(x.read(p) * static_cast<float>(resolution)) + make_int2(i, j);
                $if (pos.x >= 0 & pos.x < resolution & pos.y >= 0 & pos.y < resolution) {
                    display.write(make_uint2(cast<uint>(pos.x), resolution - 1u - pos.y),
                                  make_float4(.4f, .6f, .6f, 1.f));
                };
            }
        }
    };
    auto draw_particles_ast = roundtrip_kernel(draw_particles_kernel);
    auto draw_particles = device.create<Shader1D<Buffer<float2>, Image<float>>>(
        draw_particles_ast->function(),
        ShaderOption{.name = luisa::string{"xir2ast_mpm_draw_particles"}});

    // Run simulation
    init(stream);

    while (!window.should_close()) {
        CommandList cmd_list;
        // Run multiple substeps per frame
        for (uint i = 0u; i < n_steps; i++) { substep(cmd_list); }
        cmd_list << clear_display(display).dispatch(resolution, resolution)
                 << draw_particles(x, display).dispatch(n_particles);
        stream << cmd_list.commit() << swap_chain.present(display);
        window.poll_events();
    }
    stream << synchronize();
}
