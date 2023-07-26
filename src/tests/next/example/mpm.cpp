/**
 * @file: tests/next/example/mpm.cpp
 * @author: sailing-innocent
 * @date: 2023-07-26
 * @brief: the mpm fluid simulation suite, based on the previous MPM3D and MPM88 cases
*/

#include "common/config.h"

#include <chrono>

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

    // auto index = [](UInt3 xyz) noexcept {
    //     auto p = clamp(xyz, static_cast<uint3>(0), static_cast<uint3>(n_grid -1));
    //     return p.x + p.y * n_grid + p.z * n_grid * n_grid;
    // };

    // auto outer_product = [](Float3 a, Float3 b) noexcept {
    //     return make_float3x3(
    //         make_float3(a[0] * b[0], a[1] * b[0], a[2] * b[0]),
    //         make_float3(a[0] * b[1], a[1] * b[1], a[2] * b[1]),
    //         make_float3(a[0] * b[2], a[1] * b[2], a[2] * b[2]));
    // };

    // auto trace = [](Float3x3 m) noexcept { return m[0][0] + m[1][1] + m[2][2]; };

    // Shader3D<> clear_grid = device.compile<3>([&] {
    //     set_block_size(8, 8, 1);
    //     UInt idx = index(dispatch_id().xyz());
    //     // grid_write(idx, make_float4());
    // });

    // Shader1D<> point_to_grid = device.compile<1>([&] {
    //     set_block_size(64, 1, 1);
    //     UInt p = dispatch_id().x;
    //     Float3 Xp = x->read(p) / dx;
    //     Int3 base = make_int3(Xp - 0.5f);
    //     Float3 fx = Xp - make_float3(base);
    //     std::array w{0.5f * sqr(1.5f - fx),
    //         0.75f - sqr(fx - 1.0f),
    //         0.5f * sqr(fx - 0.5f)};
    //     Float stress = -4.f * dt * E * p_vol * ( J->read(p) - 1.f) / sqr(dx);
    // });

    // auto substep = [&](CommandList &cmd_list) noexcept {
    //     cmd_list << clear_grid().dispatch(n_grid, n_grid, n_grid);
    // };

    // auto init = [&](Stream &stream) noexcept {
    //     luisa::vector<float3> x_init(n_particles);
    // };

    Shader2D<> clear_display = device.compile<2>([&] {
        display->write(dispatch_id().xy(), make_float4(.1f, .2f, .3f, 1.f));
    });

    // init(stream);

    while (!window.should_close()) {
        CommandList cmd_list; 
        cmd_list << clear_display().dispatch(resolution, resolution);
        stream << cmd_list.commit() << swap_chain.present(display);
        window.poll_events();
    }

    stream << synchronize();
    return 0;
}

} // namespace luisa::test


TEST_SUITE("example_sim") {
    TEST_CASE("mpm3d") {
        Context context{luisa::test::argv()[0]};
       
        for (auto i = 0; i < luisa::test::supported_backends_count(); i++) {
            luisa::string device_name = luisa::test::supported_backends()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::mpm3d(device) == 0);
            }
        }
    }
}