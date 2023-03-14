//
// Created by Mike Smith on 2022/5/11.
//

#include <random>
#include <fstream>
#include <chrono>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <runtime/buffer.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <core/logging.h>
#include <gui/window.h>

int main(int argc, char *argv[]) {

    using namespace luisa;
    using namespace luisa::compute;

    auto sqr = [](auto x) noexcept { return x * x; };

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    static constexpr auto n_grid = 128u;
    static constexpr auto n_steps = 50u;

    static constexpr auto n_particles = n_grid * n_grid / 2u;
    static constexpr auto dx = 1.f / n_grid;
    static constexpr auto dt = 1e-4f;
    static constexpr auto p_rho = 1.f;
    static constexpr auto p_vol = sqr(dx * .5f);
    static constexpr auto p_mass = p_rho * p_vol;
    static constexpr auto gravity = 9.8f;
    static constexpr auto bound = 3u;
    static constexpr auto E = 400.f;

    static constexpr auto resolution = 512u;

    auto x = device.create_buffer<float2>(n_particles);
    auto v = device.create_buffer<float2>(n_particles);
    auto C = device.create_buffer<float2x2>(n_particles);
    auto J = device.create_buffer<float>(n_particles);
    auto grid_v = device.create_buffer<float>(n_grid * n_grid * 2u);
    auto grid_m = device.create_buffer<float>(n_grid * n_grid);
    auto display = device.create_image<float>(PixelStorage::BYTE4, make_uint2(resolution));

    auto index = [](auto xy) noexcept {
        using T = vector_expr_element_t<decltype(xy)>;
        auto p = clamp(xy, static_cast<T>(0), static_cast<T>(n_grid - 1));
        return p.x + p.y * n_grid;
    };
    auto outer_product = [](auto a, auto b) noexcept {
        return make_float2x2(a[0] * b[0], a[1] * b[0], a[0] * b[1], a[1] * b[1]);
    };
    auto trace = [](auto m) noexcept { return m[0][0] + m[1][1]; };

    auto clear_grid = device.compile<2>([&] {
        auto idx = index(dispatch_id().xy());
        grid_v->write(idx * 2u, 0.f);
        grid_v->write(idx * 2u + 1u, 0.f);
        grid_m->write(idx, 0.f);
    });

    auto point_to_grid = device.compile<1>([&] {
        auto p = dispatch_id().x;
        auto Xp = x->read(p) / dx;
        auto base = make_int2(Xp - 0.5f);
        auto fx = Xp - make_float2(base);
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};
        auto stress = -4.f * dt * E * p_vol * (J->read(p) - 1.f) / sqr(dx);
        auto affine = make_float2x2(stress, 0.f, 0.f, stress) + p_mass * C->read(p);
        auto vp = v->read(p);
        for (auto ii = 0; ii < 9; ii++) {
            auto offset = make_int2(ii % 3, ii / 3);
            auto i = offset.x;
            auto j = offset.y;
            auto dpos = (make_float2(offset) - fx) * dx;
            auto weight = w[i].x * w[j].y;
            auto vadd = weight * (p_mass * vp + affine * dpos);
            auto idx = index(base + offset);
            grid_v->atomic(idx * 2u).fetch_add(vadd.x);
            grid_v->atomic(idx * 2u + 1u).fetch_add(vadd.y);
            grid_m->atomic(idx).fetch_add(weight * p_mass);
        }
    });

    auto simulate_grid = device.compile<2>([&] {
        auto coord = dispatch_id().xy();
        auto i = index(coord);
        auto v = make_float2(grid_v->read(i * 2u), grid_v->read(i * 2u + 1u));
        auto m = grid_m->read(i);
        v = ite(m > 0.f, v / m, v);
        v.y -= dt * gravity;
        v.x = ite((coord.x < bound & v.x < 0.f) | (coord.x + bound > n_grid & v.x > 0.f), 0.f, v.x);
        v.y = ite((coord.y < bound & v.y < 0.f) | (coord.y + bound > n_grid & v.y > 0.f), 0.f, v.y);
        grid_v->write(i * 2u, v.x);
        grid_v->write(i * 2u + 1u, v.y);
    });

    auto grid_to_point = device.compile<1>([&] {
        auto p = dispatch_id().x;
        auto Xp = x->read(p) / dx;
        auto base = make_int2(Xp - 0.5f);
        auto fx = Xp - make_float2(base);
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};
        auto new_v = def(make_float2(0.f));
        auto new_C = def(make_float2x2(0.f));
        for (auto ii = 0; ii < 9; ii++) {
            auto offset = make_int2(ii % 3, ii / 3);
            auto i = offset.x;
            auto j = offset.y;
            auto dpos = (make_float2(offset) - fx) * dx;
            auto weight = w[i].x * w[j].y;
            auto idx = index(base + offset);
            auto g_v = make_float2(grid_v->read(idx * 2u),
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
        for (auto i = 0; i < n_particles; i++) {
            auto rx = uniform(random);
            auto ry = uniform(random);
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

    auto clear_display = device.compile<2>([&] {
        display->write(dispatch_id().xy(), make_float4(.1f, .2f, .3f, 1.f));
    });

    auto draw_particles = device.compile<1>([&] {
        auto p = dispatch_id().x;
        for (auto i = -1; i <= 1; i++) {
            for (auto j = -1; j <= 1; j++) {
                auto pos = make_int2(x->read(p) * static_cast<float>(resolution)) + make_int2(i, j);
                $if(pos.x >= 0 & pos.x < resolution & pos.y >= 0 & pos.y < resolution) {
                    display->write(make_uint2(cast<uint>(pos.x), resolution - 1u - pos.y),
                                   make_float4(.4f, .6f, .6f, 1.f));
                };
            }
        }
    });

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    init(stream);
    Window window{"MPM88", resolution, resolution, false};
    auto swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        make_uint2(resolution),
        true, false, 2)};
    while (!window.should_close()) {
        CommandList cmd_list;
        for (auto i = 0u; i < n_steps; i++) { substep(cmd_list); }
        cmd_list << clear_display().dispatch(resolution, resolution)
                 << draw_particles().dispatch(n_particles);
        stream << cmd_list.commit() << swap_chain.present(display);
        window.pool_event();
    }
    stream << synchronize();
}
