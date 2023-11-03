/**
 * @file: tests/next/example/mpm3d.cpp
 * @author: sailing-innocent
 * @date: 2023-07-26
 * @brief: the mpm fluid simulation suite, based on the previous MPM3D cases
*/

#include "common/config.h"

#include <random>

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>

#include <luisa/gui/window.h>

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

#include <luisa/gui/framerate.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int mpm3d(Device &device) {

    auto sqr = [](auto x) noexcept { return x * x; };
    static constexpr int n_grid = 64;
    static constexpr uint n_steps = 25u;

    static constexpr uint n_particles = n_grid * n_grid * n_grid / 4u;
    static constexpr float dx = 1.f / n_grid;
    static constexpr float dt = 8e-5f;
    static constexpr float p_rho = 1.f;
    static constexpr float p_vol = (dx * .5f) * (dx * .5f) * (dx * .5f);
    static constexpr float p_mass = p_rho * p_vol;
    static constexpr float gravity = 9.8f;
    static constexpr int bound = 3;
    static constexpr float E = 400.f;

    static constexpr uint resolution = 1024u;

    Buffer<float3> x = device.create_buffer<float3>(n_particles);
    Buffer<float3> v = device.create_buffer<float3>(n_particles);
    Buffer<float3x3> C = device.create_buffer<float3x3>(n_particles);
    Buffer<float> J = device.create_buffer<float>(n_particles);
    Buffer<float4> grid = device.create_buffer<float4>(n_grid * n_grid * n_grid);

    Window window{"MPM3D", resolution, resolution};

    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        make_uint2(resolution),
        false, false, 3)};

    Image<float> display = device.create_image<float>(swap_chain.backend_storage(), make_uint2(resolution));

    auto index = [](UInt3 xyz) noexcept {
        auto p = clamp(xyz, static_cast<uint3>(0), static_cast<uint3>(n_grid - 1));
        return p.x + p.y * n_grid + p.z * n_grid * n_grid;
    };

    auto outer_product = [](Float3 a, Float3 b) noexcept {
        return make_float3x3(
            make_float3(a[0] * b[0], a[1] * b[0], a[2] * b[0]),
            make_float3(a[0] * b[1], a[1] * b[1], a[2] * b[1]),
            make_float3(a[0] * b[2], a[1] * b[2], a[2] * b[2]));
    };

    auto trace = [](Float3x3 m) noexcept { return m[0][0] + m[1][1] + m[2][2]; };

    Shader3D<> clear_grid = device.compile<3>([&] {
        set_block_size(8, 8, 1);
        UInt idx = index(dispatch_id().xyz());
        grid->write(idx, make_float4());
    });

    Shader1D<> point_to_grid = device.compile<1>([&] {
        set_block_size(64, 1, 1);
        UInt p = dispatch_id().x;
        Float3 Xp = x->read(p) / dx;
        Int3 base = make_int3(Xp - 0.5f);
        Float3 fx = Xp - make_float3(base);
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};
        Float stress = -4.f * dt * E * p_vol * (J->read(p) - 1.f) / sqr(dx);
        // TODO: here C runtime read will raise error
        Float3x3 curr_rho = p_mass * C->read(p);
        Float3x3 affine = make_float3x3(
            stress, 0.f, 0.f,
            0.f, stress, 0.f,
            0.f, 0.f, stress);
        Float3 vp = v->read(p);

        for (uint ii = 0; ii < 27; ii++) {
            int3 offset = make_int3(ii % 3, ii / 3 % 3, ii / 3 / 3);
            int i = offset.x;
            int j = offset.y;
            int k = offset.z;
            Float3 dpos = (make_float3(offset) - fx) * dx;
            Float weight = w[i].x * w[j].y * w[k].z;
            Float3 vadd = weight * (p_mass * vp + affine * dpos);
            UInt idx = index(base + offset);
            grid->atomic(idx).x.fetch_add(vadd.x);
            grid->atomic(idx).y.fetch_add(vadd.y);
            grid->atomic(idx).z.fetch_add(vadd.z);
            grid->atomic(idx).w.fetch_add(weight * p_mass);
        }
    });

    Shader3D<> simulate_grid = device.compile<3>([&] {
        set_block_size(8, 8, 1);
        Int3 coord = make_int3(dispatch_id().xyz());
        UInt i = index(coord);
        Float4 v_and_m = grid->read(i);
        Float3 v = v_and_m.xyz();
        Float m = v_and_m.w;
        v = ite(m > 0.f, v / m, v);
        v.y -= dt * gravity;
        v = ite((coord < bound && v < 0.f) || (coord > n_grid - bound && v > 0.f), 0.f, v);
        grid->write(i, make_float4(v, m));
    });

    Shader1D<> grid_to_point = device.compile<1>([&] {
        set_block_size(64, 1, 1);
        UInt p = dispatch_id().x;
        Float3 Xp = x->read(p) / dx;
        Int3 base = make_int3(Xp - 0.5f);
        Float3 fx = Xp - make_float3(base);
        std::array w{0.5f * sqr(1.5f - fx),
                     0.75f - sqr(fx - 1.0f),
                     0.5f * sqr(fx - 0.5f)};
        Float3 new_v = def(make_float3(0.f));
        Float3x3 new_C = def(make_float3x3(0.f));
        for (uint ii = 0; ii < 27; ii++) {
            int3 offset = make_int3(ii % 3, ii / 3 % 3, ii / 3 / 3);
            int i = offset.x;
            int j = offset.y;
            int k = offset.z;
            Float3 dpos = (make_float3(offset) - fx) * dx;
            Float weight = w[i].x * w[j].y * w[k].z;
            UInt idx = index(base + offset);
            Float3 g_v = grid->read(idx).xyz();
            new_v += weight * g_v;
            new_C = new_C + 4.f * weight * outer_product(g_v, dpos) / sqr(dx);
        }
        v->write(p, new_v);
        x->write(p, x->read(p) + new_v * dt);
        J->write(p, J->read(p) * (1.f + dt * trace(new_C)));
        C->write(p, new_C);
    });

    auto substep = [&](CommandList &cmd_list) noexcept {
        cmd_list << clear_grid().dispatch(n_grid, n_grid, n_grid)
                 << point_to_grid().dispatch(n_particles);
        //  << simulate_grid().dispatch(n_grid, n_grid, n_grid)
        //  << grid_to_point().dispatch(n_particles);
    };

    auto init = [&](Stream &stream) noexcept {
        luisa::vector<float3> x_init(n_particles);
        std::default_random_engine random{std::random_device{}()};
        std::uniform_real_distribution<float> uniform;
        for (uint i = 0; i < n_particles; i++) {
            float rx = uniform(random);
            float ry = uniform(random);
            float rz = uniform(random);
            x_init[i] = make_float3(rx * .4f + .2f, ry * .4f + .2f, rz * .4f + .2f);
        }
        luisa::vector<float3> v_init(n_particles, make_float3(0.f));
        luisa::vector<float> J_init(n_particles, 1.f);
        luisa::vector<float3x3> C_init(n_particles, make_float3x3(1.f));
        stream << x.copy_from(x_init.data())
               << v.copy_from(v_init.data())
               << J.copy_from(J_init.data())
               << C.copy_from(C_init.data())
               << synchronize();
    };

    Shader2D<> clear_display = device.compile<2>([&] {
        display->write(dispatch_id().xy(), make_float4(.1f, .2f, .3f, 1.f));
    });

    static constexpr float phi = radians(28);
    static constexpr float theta = radians(32);

    auto T = [&](Float3 a0) noexcept {
        Float3 a = a0 - 0.5f;
        Float c = cos(phi);
        Float s = sin(phi);
        Float C = cos(theta);
        Float S = sin(theta);
        a.x = a.x * c + a.z * s;
        a.z = a.z * c - a.x * s;
        return make_float2(a.x, a.y * C + a.z * S) + 0.5f;
    };

    Shader1D<> draw_particles = device.compile<1>([&] {
        UInt p = dispatch_id().x;
        Float2 basepos = T(x->read(p));
        for (int i = -1; i < 1; i++) {
            for (int j = -1; j < 1; j++) {
                Int2 pos = make_int2(basepos * static_cast<float>(resolution)) + make_int2(i, j);
                $if (pos.x >= 0 & pos.x < resolution & pos.y >= 0 & pos.y < resolution) {
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
    return 0;
}

}// namespace luisa::test

TEST_SUITE("fluid_sim") {
    TEST_CASE("mpm3d") {
        Context context{luisa::test::argv()[0]};

        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::mpm3d(device) == 0);
            }
        }
    }
}