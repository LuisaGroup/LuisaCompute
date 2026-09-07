// Strict Metal4 AIR regression for ShaderOption::enable_extended_accel_limits.
// This test covers static and motion direct traces, generated AIR intrinsic
// suffixes, and executing closest/any-hit results.

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto curve_basis = CurveBasis::PIECEWISE_LINEAR;

[[nodiscard]] std::string read_text(
    const std::filesystem::path &path) {
    std::ifstream stream{path};
    return {std::istreambuf_iterator<char>{stream},
            std::istreambuf_iterator<char>{}};
}

void test_extended_accel_limits(Device &device) {
    log_level_verbose();

    constexpr std::array static_vertices{
        float3{-1.0f, -1.0f, 0.0f},
        float3{1.0f, -1.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f}};
    constexpr std::array motion_vertices{
        float3{-1.0f, -1.0f, 0.0f},
        float3{1.0f, -1.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f},
        float3{-1.0f, -1.0f, -1.0f},
        float3{1.0f, -1.0f, -1.0f},
        float3{0.0f, 1.0f, -1.0f}};
    constexpr std::array triangles{Triangle{0u, 1u, 2u}};

    auto static_vertex_buffer =
        device.create_buffer<float3>(static_vertices.size());
    auto motion_vertex_buffer =
        device.create_buffer<float3>(motion_vertices.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto static_mesh =
        device.create_mesh(static_vertex_buffer, triangle_buffer);
    AccelOption motion_option{};
    motion_option.motion.keyframe_count = 2u;
    motion_option.motion.time_start = 0.0f;
    motion_option.motion.time_end = 1.0f;
    auto motion_mesh = device.create_mesh(
        motion_vertex_buffer, triangle_buffer, motion_option);

    auto static_accel = device.create_accel();
    static_accel.emplace_back(static_mesh);
    auto motion_accel = device.create_accel();
    motion_accel.emplace_back(motion_mesh);

    auto distances_buffer = device.create_buffer<float4>(1u);
    auto any_hits_buffer = device.create_buffer<uint4>(1u);
    Kernel1D trace = [](AccelVar static_accel, AccelVar motion_accel,
                        BufferFloat4 distances,
                        BufferUInt4 any_hits) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto static_hit = static_accel.intersect(ray, {});
        auto motion_hit = motion_accel.intersect_motion(
            ray, 1.0f, {});
        auto static_curve_hit = static_accel.intersect(
            ray, {.curve_bases = {curve_basis}});
        auto motion_curve_hit = motion_accel.intersect_motion(
            ray, 1.0f, {.curve_bases = {curve_basis}});
        auto static_any = static_accel.intersect_any(ray, {});
        auto motion_any = motion_accel.intersect_any_motion(
            ray, 0.5f, {});
        auto static_curve_any = static_accel.intersect_any(
            ray, {.curve_bases = {curve_basis}});
        auto motion_curve_any = motion_accel.intersect_any_motion(
            ray, 0.5f, {.curve_bases = {curve_basis}});
        distances.write(
            0u, make_float4(
                    static_hit->distance(), motion_hit->distance(),
                    static_curve_hit->distance(),
                    motion_curve_hit->distance()));
        any_hits.write(
            0u, make_uint4(
                    ite(static_any, 1u, 0u),
                    ite(motion_any, 1u, 0u),
                    ite(static_curve_any, 1u, 0u),
                    ite(motion_curve_any, 1u, 0u)));
    };

    auto stamp = std::chrono::steady_clock::now()
                     .time_since_epoch()
                     .count();
    auto dump_directory =
        std::filesystem::temp_directory_path() /
        ("luisa-metal4-extended-" + std::to_string(stamp));
    std::error_code filesystem_error;
    std::filesystem::create_directories(
        dump_directory, filesystem_error);
    expect(!filesystem_error)
        << "failed to create the Metal4 LLVM dump directory";
    auto dump_prefix = dump_directory / "extended_accel_limits";

    auto previous_dump = std::getenv("LUISA_DUMP_LLVM_IR");
    auto previous_dump_value =
        previous_dump == nullptr ?
            std::string{} :
            std::string{previous_dump};
    static_cast<void>(setenv("LUISA_DUMP_LLVM_IR", "1", 1));
    ShaderOption shader_option{};
    shader_option.enable_cache = false;
    shader_option.enable_extended_accel_limits = true;
    shader_option.name = dump_prefix.string();
    auto shader = device.compile(trace, shader_option);
    if (previous_dump == nullptr) {
        static_cast<void>(unsetenv("LUISA_DUMP_LLVM_IR"));
    } else {
        static_cast<void>(setenv(
            "LUISA_DUMP_LLVM_IR", previous_dump_value.c_str(), 1));
    }

    auto direct_ir = read_text(
        dump_prefix.string() + ".direct.air.ll");
    auto indirect_ir = read_text(
        dump_prefix.string() + ".indirect.air.ll");
    constexpr auto static_intrinsic =
        "air.intersect.instancing.triangle_data.extended_limits";
    constexpr auto motion_intrinsic =
        "air.intersect.instancing.triangle_data.primitive_motion."
        "instance_motion.extended_limits";
    constexpr auto curve_intrinsic =
        "air.intersect.instancing.triangle_data.curve_data.extended_limits";
    constexpr auto curve_motion_intrinsic =
        "air.intersect.instancing.triangle_data.curve_data.primitive_motion."
        "instance_motion.extended_limits";
    for (auto &&ir : {direct_ir, indirect_ir}) {
        expect(ir.find(static_intrinsic) != std::string::npos)
            << "extended static-trace AIR intrinsic is missing";
        expect(ir.find(motion_intrinsic) != std::string::npos)
            << "extended motion-trace AIR intrinsic is missing";
        expect(ir.find(curve_intrinsic) != std::string::npos)
            << "extended curve-trace AIR intrinsic is missing";
        expect(ir.find(curve_motion_intrinsic) != std::string::npos)
            << "extended curve-motion AIR intrinsic is missing";
    }

    std::array<float4, 1u> distances{};
    std::array<uint4, 1u> any_hits{};
    auto stream = device.create_stream();
    stream << static_vertex_buffer.copy_from(
                  luisa::span{static_vertices})
           << motion_vertex_buffer.copy_from(
                  luisa::span{motion_vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << static_mesh.build()
           << motion_mesh.build()
           << static_accel.build()
           << motion_accel.build()
           << shader(static_accel, motion_accel,
                     distances_buffer, any_hits_buffer)
                  .dispatch(1u)
           << distances_buffer.copy_to(luisa::span{distances})
           << any_hits_buffer.copy_to(luisa::span{any_hits})
           << synchronize();

    constexpr auto epsilon = 1.0e-4f;
    expect(std::abs(distances[0].x - 1.0f) < epsilon)
        << "extended static closest-hit distance is wrong";
    expect(std::abs(distances[0].y - 2.0f) < epsilon)
        << "extended motion closest-hit distance is wrong";
    expect(std::abs(distances[0].z - 1.0f) < epsilon)
        << "extended curve-configured closest-hit distance is wrong";
    expect(std::abs(distances[0].w - 2.0f) < epsilon)
        << "extended curve-motion closest-hit distance is wrong";
    expect(any_hits[0].x == 1u && any_hits[0].y == 1u &&
           any_hits[0].z == 1u && any_hits[0].w == 1u)
        << "one or more extended any-hit traces failed";

    filesystem_error.clear();
    std::filesystem::remove_all(
        dump_directory, filesystem_error);
    expect(!filesystem_error)
        << "failed to remove the Metal4 LLVM dump directory";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_extended_accel_limits(dc->device);
}
