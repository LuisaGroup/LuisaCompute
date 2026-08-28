#include "ios_device_conformance.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string_view>

#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/dsl/struct.h>
#include <luisa/luisa-compute.h>

#include "ios_path_tracing_kernel.h"

struct IOSMetal4RasterVertex {
    luisa::float4 position;
    luisa::float4 color;
};

struct IOSMetal4RasterVarying {
    luisa::float4 position;
    luisa::float4 color;
};

LUISA_STRUCT(IOSMetal4RasterVarying, position, color) {};

namespace luisa::compute::metal {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_ms(
    Clock::time_point begin,
    Clock::time_point end) noexcept {
    return std::chrono::duration<double, std::milli>{end - begin}.count();
}

[[nodiscard]] IOSMetal4ConformanceResult fail(
    IOSMetal4ConformanceResult result,
    luisa::string_view stage,
    luisa::string_view message) noexcept {
    result.success = false;
    result.failed_stage = luisa::string{stage};
    result.error = luisa::string{message};
    return result;
}

struct PathTracingScene {
    luisa::vector<float3> vertices;
    luisa::vector<Triangle> triangles;
    luisa::vector<float4> materials;
};

[[nodiscard]] PathTracingScene make_path_tracing_scene() noexcept {
    PathTracingScene scene;
    scene.vertices.reserve(96u);
    scene.triangles.reserve(32u);
    scene.materials.reserve(32u);

    auto add_triangle = [&](float3 a, float3 b, float3 c,
                            float4 material) noexcept {
        auto base = static_cast<uint32_t>(scene.vertices.size());
        scene.vertices.emplace_back(a);
        scene.vertices.emplace_back(b);
        scene.vertices.emplace_back(c);
        scene.triangles.emplace_back(Triangle{base, base + 1u, base + 2u});
        scene.materials.emplace_back(material);
    };
    auto add_quad = [&](float3 a, float3 b, float3 c, float3 d,
                        float4 material) noexcept {
        add_triangle(a, b, c, material);
        add_triangle(a, c, d, material);
    };

    constexpr auto gray = make_float4(0.72f, 0.70f, 0.66f, 0.0f);
    constexpr auto red = make_float4(0.72f, 0.10f, 0.07f, 0.0f);
    constexpr auto green = make_float4(0.08f, 0.55f, 0.16f, 0.0f);
    constexpr auto blue = make_float4(0.12f, 0.28f, 0.78f, 0.0f);
    constexpr auto light = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    // Open-front Cornell-style room.
    add_quad(
        make_float3(-1.5f, -1.0f, 1.5f),
        make_float3(1.5f, -1.0f, 1.5f),
        make_float3(1.5f, -1.0f, -2.0f),
        make_float3(-1.5f, -1.0f, -2.0f), gray);
    add_quad(
        make_float3(-1.5f, 2.0f, -2.0f),
        make_float3(1.5f, 2.0f, -2.0f),
        make_float3(1.5f, 2.0f, 1.5f),
        make_float3(-1.5f, 2.0f, 1.5f), gray);
    add_quad(
        make_float3(-1.5f, -1.0f, -2.0f),
        make_float3(1.5f, -1.0f, -2.0f),
        make_float3(1.5f, 2.0f, -2.0f),
        make_float3(-1.5f, 2.0f, -2.0f), gray);
    add_quad(
        make_float3(-1.5f, -1.0f, 1.5f),
        make_float3(-1.5f, -1.0f, -2.0f),
        make_float3(-1.5f, 2.0f, -2.0f),
        make_float3(-1.5f, 2.0f, 1.5f), red);
    add_quad(
        make_float3(1.5f, -1.0f, -2.0f),
        make_float3(1.5f, -1.0f, 1.5f),
        make_float3(1.5f, 2.0f, 1.5f),
        make_float3(1.5f, 2.0f, -2.0f), green);

    // A blue box gives the RTX image silhouettes, indirect light, and shadow
    // rays that cannot be reproduced by a single full-screen compute fill.
    constexpr auto box_min = make_float3(-0.58f, -1.0f, -0.88f);
    constexpr auto box_max = make_float3(0.48f, 0.08f, 0.28f);
    add_quad(
        make_float3(box_min.x, box_max.y, box_min.z),
        make_float3(box_max.x, box_max.y, box_min.z),
        make_float3(box_max.x, box_max.y, box_max.z),
        make_float3(box_min.x, box_max.y, box_max.z), blue);
    add_quad(
        make_float3(box_min.x, box_min.y, box_max.z),
        make_float3(box_max.x, box_min.y, box_max.z),
        make_float3(box_max.x, box_max.y, box_max.z),
        make_float3(box_min.x, box_max.y, box_max.z), blue);
    add_quad(
        make_float3(box_max.x, box_min.y, box_min.z),
        make_float3(box_min.x, box_min.y, box_min.z),
        make_float3(box_min.x, box_max.y, box_min.z),
        make_float3(box_max.x, box_max.y, box_min.z), blue);
    add_quad(
        make_float3(box_min.x, box_min.y, box_min.z),
        make_float3(box_min.x, box_min.y, box_max.z),
        make_float3(box_min.x, box_max.y, box_max.z),
        make_float3(box_min.x, box_max.y, box_min.z), blue);
    add_quad(
        make_float3(box_max.x, box_min.y, box_max.z),
        make_float3(box_max.x, box_min.y, box_min.z),
        make_float3(box_max.x, box_max.y, box_min.z),
        make_float3(box_max.x, box_max.y, box_max.z), blue);

    // This rectangle matches the constants in the RTX path-tracing kernel.
    add_quad(
        make_float3(-0.42f, 1.92f, -0.72f),
        make_float3(0.42f, 1.92f, -0.72f),
        make_float3(0.42f, 1.92f, 0.10f),
        make_float3(-0.42f, 1.92f, 0.10f), light);
    return scene;
}

[[nodiscard]] bool run_printer_smoke(
    Device &device,
    Stream &stream,
    IOSMetal4ConformanceResult &result) noexcept {
    struct Capture {
        std::mutex mutex;
        std::condition_variable condition;
        luisa::string message;
    };
    auto capture = std::make_shared<Capture>();
    stream.set_log_callback(
        [capture](luisa::string_view message) noexcept {
            if (!message.starts_with("ios-metal4-air-log ")) { return; }
            {
                std::scoped_lock lock{capture->mutex};
                capture->message = luisa::string{message};
            }
            capture->condition.notify_one();
        });

    Kernel1D printer = []() noexcept {
        device_log("ios-metal4-air-log value={}", 42u);
    };
    ShaderOption option{};
    option.enable_cache = false;
    option.name = "ios_device_printer";
    auto begin = Clock::now();
    auto shader = device.compile(printer, option);
    stream << shader().dispatch(1u) << synchronize();
    auto delivered = [&] {
        std::unique_lock lock{capture->mutex};
        return capture->condition.wait_for(
            lock, std::chrono::seconds{5},
            [&]() noexcept { return !capture->message.empty(); });
    }();
    result.printer_ms = elapsed_ms(begin, Clock::now());
    {
        std::scoped_lock lock{capture->mutex};
        result.printer_message = capture->message;
    }
    return delivered &&
           result.printer_message == "ios-metal4-air-log value=42";
}

[[nodiscard]] bool run_bindless_indirect_smoke(
    Device &device,
    Stream &stream,
    IOSMetal4ConformanceResult &result) noexcept {
    constexpr auto bindless_expected = 0x13579bdfu;
    constexpr auto indirect_count = 8u;
    constexpr auto indirect_kernel_id = 7u;

    auto source = device.create_buffer<uint>(1u);
    auto bindless_output = device.create_buffer<uint>(1u);
    auto indirect_output = device.create_buffer<uint>(indirect_count);
    auto heap = device.create_bindless_array(4u);
    heap.emplace_on_update(0u, source);
    auto indirect = device.create_indirect_dispatch_buffer(1u);

    Kernel1D read_bindless = [](
                                 Var<BindlessArray> bindless,
                                 BufferUInt output) noexcept {
        output.write(0u, bindless.buffer<uint>(0u).read(0u));
    };
    Kernel1D prepare_indirect = [](Var<IndirectDispatchBuffer> dispatches) noexcept {
        dispatches.set_dispatch_count(1u);
        dispatches.set_kernel(
            0u, make_uint3(32u, 1u, 1u),
            make_uint3(indirect_count, 1u, 1u),
            indirect_kernel_id);
    };
    Kernel1D run_indirect = [](BufferUInt output) noexcept {
        set_block_size(32u, 1u, 1u);
        auto lane = dispatch_x();
        output.write(lane, 1000u + kernel_id() + lane);
    };

    ShaderOption option{};
    option.enable_cache = false;
    auto bindless_shader = device.compile(read_bindless, option);
    auto prepare_shader = device.compile(prepare_indirect, option);
    auto indirect_shader = device.compile(run_indirect, option);

    const std::array source_data{bindless_expected};
    std::array<uint, 1u> bindless_result{};
    std::array<uint, indirect_count> indirect_result{};
    auto begin = Clock::now();
    stream << source.copy_from(luisa::span{source_data})
           << heap.update()
           << bindless_shader(heap, bindless_output).dispatch(1u)
           << prepare_shader(indirect).dispatch(1u)
           << indirect_shader(indirect_output).dispatch(indirect)
           << bindless_output.copy_to(luisa::span{bindless_result})
           << indirect_output.copy_to(luisa::span{indirect_result})
           << synchronize();
    result.bindless_indirect_ms = elapsed_ms(begin, Clock::now());
    result.bindless_value = bindless_result[0u];
    for (auto value : indirect_result) {
        result.indirect_checksum += value;
    }
    constexpr auto indirect_expected =
        indirect_count * (1000u + indirect_kernel_id) +
        (indirect_count - 1u) * indirect_count / 2u;
    return result.bindless_value == bindless_expected &&
           result.indirect_checksum == indirect_expected;
}

[[nodiscard]] bool run_raster_smoke(
    Device &device,
    IOSMetal4ConformanceResult &result) noexcept {
    constexpr auto width = 64u;
    constexpr auto height = 64u;

    auto raster = device.extension<RasterExt>();
    if (raster == nullptr) { return false; }

    RasterStageKernel vertex = [](Var<AppData> input) noexcept {
        Var<IOSMetal4RasterVarying> output;
        output.position = make_float4(input.position, 1.0f);
        output.color = input.color;
        return output;
    };
    RasterStageKernel fragment = [](
                                     Var<IOSMetal4RasterVarying> input) noexcept {
        return input.color;
    };
    RasterKernel<decltype(vertex), decltype(fragment)> kernel{
        vertex, fragment};
    MeshFormat mesh_format;
    const VertexAttribute attributes[]{
        {VertexAttributeType::Position, PixelFormat::RGBA32F},
        {VertexAttributeType::Color, PixelFormat::RGBA32F}};
    mesh_format.emplace_vertex_stream(attributes);

    ShaderOption option{};
    option.enable_cache = false;
    option.name = "ios_device_raster";
    auto compile_begin = Clock::now();
    auto shader = device.compile(kernel, mesh_format, option);
    result.raster_compile_ms = elapsed_ms(compile_begin, Clock::now());

    const std::array vertices{
        IOSMetal4RasterVertex{
            .position = {-0.82f, -0.82f, 0.0f, 1.0f},
            .color = {1.0f, 0.0f, 0.0f, 1.0f}},
        IOSMetal4RasterVertex{
            .position = {0.82f, -0.82f, 0.0f, 1.0f},
            .color = {0.0f, 1.0f, 0.0f, 1.0f}},
        IOSMetal4RasterVertex{
            .position = {0.0f, 0.82f, 0.0f, 1.0f},
            .color = {0.0f, 0.0f, 1.0f, 1.0f}}};
    const std::array indices{0u, 1u, 2u};
    auto vertex_buffer = device.create_buffer<IOSMetal4RasterVertex>(
        vertices.size());
    auto index_buffer = device.create_buffer<uint>(indices.size());
    auto render_target = device.create_image<float>(
        PixelStorage::BYTE4, width, height, 1u, false, true);
    auto depth = device.create_depth_buffer(
        DepthFormat::D32, make_uint2(width, height));
    auto stream = device.create_stream(StreamTag::GRAPHICS);

    VertexBufferView vertex_view{vertex_buffer};
    luisa::vector<RasterMesh> meshes;
    meshes.emplace_back(
        luisa::span<const VertexBufferView>{&vertex_view, 1u},
        index_buffer.view(), 1u, 19u);
    RasterState state{};
    state.cull_mode = CullMode::None;
    state.depth_state = DepthState{
        .enable_depth = true,
        .comparison = Comparison::Always,
        .write = true};

    luisa::vector<std::array<uint8_t, 4u>> pixels(width * height);
    auto dispatch_begin = Clock::now();
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << index_buffer.copy_from(luisa::span{indices})
           << depth.clear(1.0f)
           << raster->clear_render_target(
                  render_target.view(), make_float4(0.0f, 0.0f, 0.0f, 1.0f))
           << shader().draw(
                  std::move(meshes), mesh_format,
                  Viewport{0u, 0u, width, height}, state,
                  &depth, render_target)
           << render_target.copy_to(luisa::span{pixels})
           << synchronize();
    result.raster_dispatch_readback_ms =
        elapsed_ms(dispatch_begin, Clock::now());

    for (auto pixel : pixels) {
        if (pixel[0u] > 2u || pixel[1u] > 2u || pixel[2u] > 2u) {
            result.raster_colored_pixels++;
        }
    }
    result.raster_center = pixels[(height / 2u) * width + width / 2u];
    auto center_sum = static_cast<uint32_t>(result.raster_center[0u]) +
                      result.raster_center[1u] +
                      result.raster_center[2u];
    return result.raster_colored_pixels > 900u &&
           result.raster_colored_pixels < 1800u &&
           center_sum > 80u;
}

}// namespace

IOSMetal4ConformanceResult run_ios_metal4_conformance(
    Device &device,
    uint32_t width,
    uint32_t height,
    uint32_t samples_per_pixel) noexcept {
    IOSMetal4ConformanceResult result;
    result.acceleration_structure_path = device.query(
        "metal4_address_driven_acceleration_structures");
    result.motion_blur = device.query("metal_motion_blur");
    result.component_motion = device.query("metal4_component_motion");

    auto compute_stream = device.create_stream(StreamTag::COMPUTE);
    if (!run_printer_smoke(device, compute_stream, result)) {
        return fail(
            std::move(result), "shader logging",
            "Metal LogState callback did not return the expected AIR message");
    }
    if (!run_bindless_indirect_smoke(device, compute_stream, result)) {
        return fail(
            std::move(result), "bindless/indirect",
            "bindless table or GPU-authored MTL4 indirect dispatch mismatch");
    }
    if (!run_raster_smoke(device, result)) {
        return fail(
            std::move(result), "raster",
            "offscreen Metal4 AIR raster draw did not produce the expected triangle");
    }

    auto scene = make_path_tracing_scene();
    auto vertex_buffer = device.create_buffer<float3>(scene.vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(scene.triangles.size());
    auto material_buffer = device.create_buffer<float4>(scene.materials.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh);

    auto build_begin = Clock::now();
    compute_stream << vertex_buffer.copy_from(luisa::span{scene.vertices})
                   << triangle_buffer.copy_from(luisa::span{scene.triangles})
                   << material_buffer.copy_from(luisa::span{scene.materials})
                   << mesh.build()
                   << accel.build()
                   << synchronize();
    result.acceleration_build_ms = elapsed_ms(build_begin, Clock::now());

    auto output = device.create_image<float>(
        PixelStorage::BYTE4, width, height);
    ShaderOption option{};
    option.enable_cache = false;
    option.enable_fast_math = true;
    option.name = "ios_device_rtx_path_tracing";
    auto compile_begin = Clock::now();
    auto shader = device.compile(
        make_ios_rtx_path_tracing_kernel(), option);
    result.path_trace_compile_ms = elapsed_ms(compile_begin, Clock::now());

    result.pixels.resize(static_cast<size_t>(width) * height);
    auto dispatch_begin = Clock::now();
    compute_stream
        << shader(
               output, vertex_buffer, triangle_buffer,
               material_buffer, accel, samples_per_pixel)
               .dispatch(width, height)
        << output.copy_to(luisa::span{result.pixels})
        << synchronize();
    result.path_trace_dispatch_readback_ms =
        elapsed_ms(dispatch_begin, Clock::now());

    uint64_t channel_sum = 0u;
    for (auto pixel : result.pixels) {
        auto rgb_sum = static_cast<uint32_t>(pixel[0u]) +
                       pixel[1u] + pixel[2u];
        channel_sum += rgb_sum;
        if (rgb_sum > 6u) { result.path_trace_nonblack_pixels++; }
        result.path_trace_max_channel = std::max(
            result.path_trace_max_channel,
            std::max(pixel[0u], std::max(pixel[1u], pixel[2u])));
    }
    result.path_trace_mean_luma =
        static_cast<double>(channel_sum) /
        static_cast<double>(result.pixels.size() * 3u * 255u);
    if (result.path_trace_nonblack_pixels < result.pixels.size() / 4u ||
        result.path_trace_max_channel < 32u) {
        return fail(
            std::move(result), "RTX path tracing",
            "hardware ray-traced image is empty or degenerate");
    }

    result.success = true;
    return result;
}

}// namespace luisa::compute::metal
