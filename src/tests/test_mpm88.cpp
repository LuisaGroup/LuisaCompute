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

int main(int argc, char *argv[]) {

    using namespace luisa;
    using namespace luisa::compute;

    auto sqr = [](auto x) noexcept { return x * x; };

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    static constexpr uint n_grid = 128u;
    static constexpr uint n_steps = 50u;

    static constexpr uint n_particles = n_grid * n_grid / 2u;
    static constexpr float dx = 1.f / n_grid;
    static constexpr float dt = 1e-4f;
    static constexpr float p_rho = 1.f;
    static constexpr float p_vol = sqr(dx * .5f);
    static constexpr float p_mass = p_rho * p_vol;
    static constexpr float gravity = 9.8f;
    static constexpr uint bound = 3u;
    static constexpr float E = 400.f;

    static constexpr uint resolution = 512u;

    Buffer<float2> x = device.create_buffer<float2>(n_particles);
    Buffer<float2> v = device.create_buffer<float2>(n_particles);
    Buffer<float2x2> C = device.create_buffer<float2x2>(n_particles);
    Buffer<float> J = device.create_buffer<float>(n_particles);
    Buffer<float> grid_v = device.create_buffer<float>(n_grid * n_grid * 2u);
    Buffer<float> grid_m = device.create_buffer<float>(n_grid * n_grid);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Window window{"MPM88", resolution, resolution};
    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        make_uint2(resolution),
        false, false, 2)};
    Image<float> display = device.create_image<float>(swap_chain.backend_storage(), make_uint2(resolution));

    auto index = [](UInt2 xy) noexcept {
        auto p = clamp(xy, static_cast<uint2>(0), static_cast<uint2>(n_grid - 1));
        return p.x + p.y * n_grid;
    };
    auto outer_product = [](Float2 a, Float2 b) noexcept {
        return make_float2x2(a[0] * b[0], a[1] * b[0], a[0] * b[1], a[1] * b[1]);
    };
    auto trace = [](Float2x2 m) noexcept { return m[0][0] + m[1][1]; };

    Shader2D<> clear_grid = device.compile<2>([&] {
        UInt idx = index(dispatch_id().xy());
        grid_v->write(idx * 2u, 0.f);
        grid_v->write(idx * 2u + 1u, 0.f);
        grid_m->write(idx, 0.f);
    });

    Shader1D<> point_to_grid = device.compile<1>([&] {
        UInt p = dispatch_id().x;
        Float2 Xp = x->read(p) / dx;
        Int2 base = make_int2(Xp - 0.5f);
        Float2 fx = Xp - make_float2(base);
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};
        Float stress = -4.f * dt * E * p_vol * (J->read(p) - 1.f) / sqr(dx);
        Float2x2 affine = make_float2x2(stress, 0.f, 0.f, stress) + p_mass * C->read(p);
        Float2 vp = v->read(p);
        for (uint ii = 0; ii < 9; ii++) {
            int2 offset = make_int2(ii % 3, ii / 3);
            int i = offset.x;
            int j = offset.y;
            Float2 dpos = (make_float2(offset) - fx) * dx;
            Float weight = w[i].x * w[j].y;
            Float2 vadd = weight * (p_mass * vp + affine * dpos);
            UInt idx = index(base + offset);
            grid_v->atomic(idx * 2u).fetch_add(vadd.x);
            grid_v->atomic(idx * 2u + 1u).fetch_add(vadd.y);
            grid_m->atomic(idx).fetch_add(weight * p_mass);
        }
    });

    Shader2D<> simulate_grid = device.compile<2>([&] {
        UInt2 coord = dispatch_id().xy();
        UInt i = index(coord);
        Float2 v = make_float2(grid_v->read(i * 2u), grid_v->read(i * 2u + 1u));
        Float m = grid_m->read(i);
        v = ite(m > 0.f, v / m, v);
        v.y -= dt * gravity;
        v.x = ite((coord.x < bound & v.x < 0.f) | (coord.x + bound > n_grid & v.x > 0.f), 0.f, v.x);
        v.y = ite((coord.y < bound & v.y < 0.f) | (coord.y + bound > n_grid & v.y > 0.f), 0.f, v.y);
        grid_v->write(i * 2u, v.x);
        grid_v->write(i * 2u + 1u, v.y);
    });

    Shader1D<> grid_to_point = device.compile<1>([&] {
        UInt p = dispatch_id().x;
        Float2 Xp = x->read(p) / dx;
        Int2 base = make_int2(Xp - 0.5f);
        Float2 fx = Xp - make_float2(base);
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};
        Float2 new_v = def(make_float2(0.f));
        Float2x2 new_C = def(make_float2x2(0.f));
        for (uint ii = 0; ii < 9; ii++) {
            int2 offset = make_int2(ii % 3, ii / 3);
            int i = offset.x;
            int j = offset.y;
            Float2 dpos = (make_float2(offset) - fx) * dx;
            Float weight = w[i].x * w[j].y;
            UInt idx = index(base + offset);
            Float2 g_v = make_float2(grid_v->read(idx * 2u),
                                   grid_v->read(idx * 2u + 1u));
            new_v += weight * g_v;
            new_C = new_C + 4.f * weight * outer_product(g_v, dpos) / sqr(dx);
        }
        v->write(p, new_v);
        x->write(p, x->read(p) + new_v * dt);
        J->write(p, J->read(p) * (1.f + dt * trace(new_C)));
        C->write(p, new_C);
    });

    auto substep = [&](CommandList &cmd_list) noexcept {
        cmd_list << clear_grid().dispatch(n_grid, n_grid)
                 << point_to_grid().dispatch(n_particles)
                 << simulate_grid().dispatch(n_grid, n_grid)
                 << grid_to_point().dispatch(n_particles);
    };

    auto init = [&](Stream &stream) noexcept {
        luisa::vector<float2> x_init(n_particles);
        std::default_random_engine random{std::random_device{}()};
        std::uniform_real_distribution<float> uniform;
        for (uint i = 0; i < n_particles; i++) {
            float rx = uniform(random);
            float ry = uniform(random);
            x_init[i] = make_float2(rx * .4f + .2f, ry * .4f + .2f);
        }
        luisa::vector<float2> v_init(n_particles, make_float2(0.f, -1.f));
        luisa::vector<float> J_init(n_particles, 1.f);
        luisa::vector<float2x2> C_init(n_particles, make_float2x2(0.f));
        stream << x.copy_from(x_init.data())
               << v.copy_from(v_init.data())
               << J.copy_from(J_init.data())
               << C.copy_from(C_init.data())
               << synchronize();
    };

    Shader2D<> clear_display = device.compile<2>([&] {
        display->write(dispatch_id().xy(), make_float4(.1f, .2f, .3f, 1.f));
    });

    Shader1D<> draw_particles = device.compile<1>([&] {
        UInt p = dispatch_id().x;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                Int2 pos = make_int2(x->read(p) * static_cast<float>(resolution)) + make_int2(i, j);
                $if(pos.x >= 0 & pos.x < resolution & pos.y >= 0 & pos.y < resolution) {
                    display->write(make_uint2(cast<uint>(pos.x), resolution - 1u - pos.y),
                                   make_float4(.4f, .6f, .6f, 1.f));
                };
            }
        }
    });


    init(stream);
    
    while (!window.should_close()) {
        CommandList cmd_list;
        for (uint i = 0u; i < n_steps; i++) { substep(cmd_list); }
        cmd_list << clear_display().dispatch(resolution, resolution)
                 << draw_particles().dispatch(n_particles);
        stream << cmd_list.commit() << swap_chain.present(display);
        window.poll_events();
    }
    stream << synchronize();
}

