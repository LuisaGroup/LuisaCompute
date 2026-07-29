// Regression for HIP ray-tracing scratch reuse across a long sequence of
// mixed-size high-quality BLAS builds.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

// This distribution preserves the resource-size shape of the minimized
// production failure while keeping all geometry synthetic. It deliberately
// crosses many scratch-allocation size classes before and after the largest
// build instead of testing one hand-picked allocation boundary.
constexpr std::array<uint32_t, 348u> primitive_counts{
    10, 68, 20, 54, 226, 420, 444, 124, 136, 6552, 882, 52920, 819, 9828, 924, 56,
    90, 950, 206, 140, 956, 950, 986, 320, 320, 348, 950, 206, 956, 950, 986, 320,
    320, 950, 206, 956, 950, 986, 320, 320, 206, 956, 206, 206, 206, 950, 320, 320,
    956, 482, 950, 206, 956, 950, 986, 320, 320, 950, 206, 956, 986, 320, 320, 950,
    206, 956, 950, 986, 320, 320, 950, 206, 956, 986, 320, 320, 950, 206, 956, 950,
    986, 320, 320, 950, 206, 956, 320, 320, 206, 950, 206, 956, 956, 320, 950, 206,
    956, 956, 320, 206, 320, 950, 206, 956, 206, 206, 206, 950, 320, 320, 956, 206,
    956, 206, 206, 206, 950, 320, 320, 956, 206, 956, 206, 206, 206, 950, 320, 320,
    206, 950, 206, 950, 986, 320, 206, 320, 320, 206, 956, 206, 206, 206, 320, 320,
    956, 206, 950, 206, 950, 320, 320, 950, 320, 950, 950, 206, 956, 950, 986, 320,
    206, 956, 206, 956, 956, 320, 956, 320, 320, 956, 956, 320, 950, 956, 320, 206,
    956, 320, 320, 956, 348, 348, 950, 206, 950, 986, 320, 950, 206, 950, 986, 320,
    950, 206, 950, 986, 320, 950, 950, 950, 950, 950, 950, 950, 950, 956, 16516, 316,
    316, 316, 1092, 628, 1518, 628, 628, 1518, 1008, 2624, 508, 360, 1820, 1752, 13224, 1040,
    13224, 524, 524, 524, 524, 524, 524, 524, 524, 201, 196, 190, 200, 190, 183, 190,
    200, 196, 190, 191, 200, 172, 172, 180, 172, 174, 172, 177, 185, 192, 185, 188,
    185, 185, 5500, 8384, 112704, 112704, 10128, 10128, 10128, 10128, 39200, 26116, 9420, 10638, 26116, 9420,
    10638, 992, 992, 1984, 1984, 2136, 2136, 2136, 2136, 2136, 2136, 9020, 104464, 37680, 42552, 47536,
    104464, 37680, 42552, 7320, 4296, 4448, 474, 386, 7932, 7932, 5152, 696, 170408, 23488, 3260, 92,
    12, 2, 27456, 54912, 9152, 6544, 198, 2816, 1280, 480, 1280, 480, 198, 66, 2816, 4224,
    4224, 1280, 480, 1280, 480, 4224, 15520, 18560, 19520, 21120, 24640, 456, 456, 456, 256, 256,
    24, 96, 6, 4, 6, 6, 6, 22, 48, 10, 16, 2};

}// namespace

void test_hip_rt_scratch_reuse(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP RT scratch-reuse regression on backend '{}'.",
            device.backend_name());
        return;
    }

    auto stream = device.create_stream();
    auto max_primitive_count =
        *std::max_element(
            primitive_counts.cbegin(),
            primitive_counts.cend());

    luisa::vector<float3> host_vertices(
        static_cast<size_t>(max_primitive_count) * 3u);
    luisa::vector<Triangle> host_triangles(max_primitive_count);
    for (auto i = 0u; i < max_primitive_count; i++) {
        auto cell_x = static_cast<float>(i % 64u);
        auto cell_y = static_cast<float>((i / 64u) % 64u);
        auto x = -0.4f + cell_x * (0.8f / 64.0f);
        auto y = -0.4f + cell_y * (0.8f / 64.0f);
        constexpr auto radius = 0.004f;
        host_vertices[i * 3u + 0u] =
            make_float3(x - radius, y - radius, 0.0f);
        host_vertices[i * 3u + 1u] =
            make_float3(x + radius, y - radius, 0.0f);
        host_vertices[i * 3u + 2u] =
            make_float3(x, y + radius, 0.0f);
        host_triangles[i] =
            Triangle{i * 3u, i * 3u + 1u, i * 3u + 2u};
    }
    // Every BLAS prefix must contain a triangle hit by its validation ray.
    host_vertices[0u] = make_float3(-0.5f, -0.5f, 0.0f);
    host_vertices[1u] = make_float3(0.5f, -0.5f, 0.0f);
    host_vertices[2u] = make_float3(0.0f, 0.5f, 0.0f);

    luisa::vector<Buffer<float3>> vertex_buffers;
    luisa::vector<Buffer<Triangle>> triangle_buffers;
    luisa::vector<Mesh> meshes;
    vertex_buffers.reserve(primitive_counts.size());
    triangle_buffers.reserve(primitive_counts.size());
    meshes.reserve(primitive_counts.size());

    AccelOption option{.hint = AccelUsageHint::FAST_TRACE};
    for (auto primitive_count : primitive_counts) {
        auto &vertices = vertex_buffers.emplace_back(
            device.create_buffer<float3>(
                static_cast<size_t>(primitive_count) * 3u));
        auto &triangles = triangle_buffers.emplace_back(
            device.create_buffer<Triangle>(primitive_count));
        auto &mesh = meshes.emplace_back(
            device.create_mesh(vertices, triangles, option));
        stream << vertices.copy_from(luisa::span{
                      host_vertices.data(),
                      static_cast<size_t>(primitive_count) * 3u})
               << triangles.copy_from(luisa::span{
                      host_triangles.data(),
                      static_cast<size_t>(primitive_count)})
               << mesh.build();
    }

    auto accel = device.create_accel(option);
    for (auto i = 0u; i < meshes.size(); i++) {
        accel.emplace_back(
            meshes[i],
            translation(make_float3(
                static_cast<float>(i) * 2.0f,
                0.0f, 0.0f)));
    }
    stream << accel.build() << synchronize();

    Kernel1D trace = [](BufferUInt results,
                        AccelVar accel) noexcept {
        auto index = dispatch_id().x;
        auto ray = make_ray(
            make_float3(
                cast<float>(index) * 2.0f,
                0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f));
        results.write(index, accel.intersect(ray, {})->inst);
    };
    auto shader = device.compile(trace);
    auto results =
        device.create_buffer<uint>(primitive_counts.size());
    luisa::vector<uint> host_results(
        primitive_counts.size());
    stream << shader(results, accel)
                  .dispatch(primitive_counts.size())
           << results.copy_to(luisa::span{host_results})
           << synchronize();

    for (auto i = 0u; i < host_results.size(); i++) {
        expect(host_results[i] == i)
            << luisa::format(
                   "mixed-size BLAS build {} produced instance {}, expected {}",
                   i, host_results[i], i);
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_hip_rt_scratch_reuse(dc->device);
}
