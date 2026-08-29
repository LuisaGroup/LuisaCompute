// Regression for HIP TLAS build inputs large enough to cross runtime pitched-
// copy implementation boundaries.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto grid_width = 1024u;
// Deliberately cross the 2^20-row boundary used by ROCm's internal pitched-
// copy implementation. A power-of-two square only exercises the last row
// before that boundary and cannot prove that the tail remains resident.
constexpr auto grid_height = 1025u;
constexpr auto instance_count = grid_width * grid_height;

}// namespace

void test_hip_large_accel_instances(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping large HIP accel regression on backend '{}'.",
            device.backend_name());
        return;
    }

    auto stream = device.create_stream();
    std::array vertices{
        make_float3(-0.4f, -0.4f, 0.0f),
        make_float3(0.4f, -0.4f, 0.0f),
        make_float3(0.0f, 0.4f, 0.0f)};
    std::array triangles{Triangle{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    AccelOption option{.hint = AccelUsageHint::FAST_BUILD};
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer, option);
    auto accel = device.create_accel(option);
    for (auto instance = 0u; instance < instance_count; instance++) {
        auto x = static_cast<float>(instance % grid_width) * 2.0f;
        auto y = static_cast<float>(instance / grid_width) * 2.0f;
        accel.emplace_back(
            mesh, translation(make_float3(x, y, 0.0f)),
            0xffu, true, instance);
    }

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    constexpr std::array probe_instances{
        0u,
        grid_width - 1u,
        instance_count - grid_width,
        instance_count - 1u};
    auto probes = device.create_buffer<uint>(probe_instances.size());
    auto results = device.create_buffer<uint>(probe_instances.size());
    Kernel1D trace = [](BufferUInt probes, BufferUInt results,
                        AccelVar accel) noexcept {
        auto probe = dispatch_id().x;
        auto instance = probes.read(probe);
        auto x = cast<float>(instance % grid_width) * 2.0f;
        auto y = cast<float>(instance / grid_width) * 2.0f;
        auto ray = make_ray(
            make_float3(x, y, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f));
        results.write(probe, accel.intersect(ray, {})->inst);
    };
    auto shader = device.compile(trace);
    std::array<uint, probe_instances.size()> host_results{};
    stream << probes.copy_from(luisa::span{probe_instances})
           << shader(probes, results, accel).dispatch(probe_instances.size())
           << results.copy_to(luisa::span{host_results})
           << synchronize();

    for (auto i = 0u; i < probe_instances.size(); i++) {
        expect(host_results[i] == probe_instances[i])
            << luisa::format(
                   "large HIP TLAS probe {} hit instance {}, expected {}",
                   i, host_results[i], probe_instances[i]);
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_hip_large_accel_instances(dc->device);
}
