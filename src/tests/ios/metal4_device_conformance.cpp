#include "metal4_device_conformance.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string_view>

#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/dsl/struct.h>
#include <luisa/luisa-compute.h>

#include "metal4_ios_path_tracing_kernel.h"

struct IOSMetal4RasterVertex {
    luisa::float4 position;
    luisa::float4 color;
};

struct IOSMetal4RasterVarying {
    luisa::float4 position;
    luisa::float4 color;
    luisa::uint base_instance;
};

struct IOSMetal4Bool4 {
    bool x;
    bool y;
    bool z;
    bool w;
};

static_assert(sizeof(IOSMetal4Bool4) == 4u);
static_assert(offsetof(IOSMetal4Bool4, x) == 0u);
static_assert(offsetof(IOSMetal4Bool4, y) == 1u);
static_assert(offsetof(IOSMetal4Bool4, z) == 2u);
static_assert(offsetof(IOSMetal4Bool4, w) == 3u);
static_assert(sizeof(luisa::bool4) == 4u);
static_assert(sizeof(luisa::byte4) == 4u);

LUISA_STRUCT(IOSMetal4RasterVarying, position, color, base_instance) {};
LUISA_STRUCT(IOSMetal4Bool4, x, y, z, w) {};

namespace luisa::compute::metal {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_ms(
    Clock::time_point begin,
    Clock::time_point end) noexcept {
    return std::chrono::duration<double, std::milli>{end - begin}.count();
}

[[nodiscard]] Metal4DeviceConformanceResult fail(
    Metal4DeviceConformanceResult result,
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
    Metal4DeviceConformanceResult &result) noexcept {
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

[[nodiscard]] bool run_compute_abi_smoke(
    Device &device,
    Stream &stream,
    Metal4DeviceConformanceResult &result) noexcept {
    const std::array<IOSMetal4Bool4, 4u> bool_input{
        IOSMetal4Bool4{false, false, false, false},
        IOSMetal4Bool4{true, false, true, false},
        IOSMetal4Bool4{false, true, false, true},
        IOSMetal4Bool4{true, true, true, true}};
    std::array<IOSMetal4Bool4, bool_input.size()> bool_output{};
    std::array<uint, bool_input.size()> bool_masks{};
    const std::array<byte4, 4u> byte_input{
        byte4{1, 2, 3, 4},
        byte4{5, 6, 7, 8},
        byte4{9, 10, 11, 12},
        byte4{13, 14, 15, 16}};
    std::array<byte4, byte_input.size()> byte_output{};

    auto bool_source = device.create_buffer<IOSMetal4Bool4>(bool_input.size());
    auto bool_destination =
        device.create_buffer<IOSMetal4Bool4>(bool_output.size());
    auto mask_destination = device.create_buffer<uint>(bool_masks.size());
    auto byte_source = device.create_buffer<byte4>(byte_input.size());
    auto byte_destination = device.create_buffer<byte4>(byte_output.size());
    auto atomic_counter = device.create_buffer<uint>(1u);
    auto texture = device.create_image<float>(PixelStorage::BYTE4, 2u, 2u);
    auto texture_output = device.create_buffer<float4>(1u);

    Kernel1D layout = [](BufferVar<IOSMetal4Bool4> bools,
                         BufferVar<IOSMetal4Bool4> reversed_bools,
                         BufferUInt masks,
                         BufferVar<byte4> bytes,
                         BufferVar<byte4> reversed_bytes) noexcept {
        auto index = dispatch_x();
        Var flags = bools.read(index);
        Var<IOSMetal4Bool4> reversed;
        reversed.x = flags.w;
        reversed.y = flags.z;
        reversed.z = flags.y;
        reversed.w = flags.x;
        reversed_bools.write(index, reversed);
        masks.write(
            index,
            ite(flags.x, 1u, 0u) |
                ite(flags.y, 2u, 0u) |
                ite(flags.z, 4u, 0u) |
                ite(flags.w, 8u, 0u));
        reversed_bytes.write(index, bytes.read(index).wzyx());
    };
    Kernel1D atomic = [](BufferUInt counter) noexcept {
        counter.atomic(0u).fetch_add(1u);
    };
    Kernel1D read_texture = [](ImageFloat image, BufferFloat4 output) noexcept {
        output.write(0u, image.read(make_uint2(0u)));
    };

    ShaderOption option{};
    option.enable_cache = false;
    auto layout_shader = device.compile(layout, option);
    auto atomic_shader = device.compile(atomic, option);
    auto texture_shader = device.compile(read_texture, option);

    const std::array<uint, 1u> zero{0u};
    const std::array<uint8_t, 16u> texture_pixels{
        255u, 17u, 33u, 255u,
        0u, 255u, 0u, 255u,
        0u, 0u, 255u, 255u,
        255u, 255u, 255u, 255u};
    std::array<uint, 1u> atomic_result{};
    std::array<float4, 1u> texture_result{};
    auto begin = Clock::now();
    stream << bool_source.copy_from(luisa::span{bool_input})
           << byte_source.copy_from(luisa::span{byte_input})
           << atomic_counter.copy_from(luisa::span{zero})
           << texture.copy_from(luisa::span{texture_pixels})
           << layout_shader(
                  bool_source, bool_destination, mask_destination,
                  byte_source, byte_destination)
                  .dispatch(bool_input.size())
           << atomic_shader(atomic_counter).dispatch(64u)
           << texture_shader(texture, texture_output).dispatch(1u)
           << bool_destination.copy_to(luisa::span{bool_output})
           << mask_destination.copy_to(luisa::span{bool_masks})
           << byte_destination.copy_to(luisa::span{byte_output})
           << atomic_counter.copy_to(luisa::span{atomic_result})
           << texture_output.copy_to(luisa::span{texture_result})
           << synchronize();
    result.compute_abi_ms = elapsed_ms(begin, Clock::now());
    result.atomic_value = atomic_result[0u];
    result.texture_read = {
        texture_result[0u].x, texture_result[0u].y,
        texture_result[0u].z, texture_result[0u].w};

    auto valid = result.atomic_value == 64u;
    for (auto i = 0u; i < bool_input.size(); ++i) {
        auto expected_mask = static_cast<uint>(bool_input[i].x) |
                             static_cast<uint>(bool_input[i].y) << 1u |
                             static_cast<uint>(bool_input[i].z) << 2u |
                             static_cast<uint>(bool_input[i].w) << 3u;
        result.abi_layout_checksum += bool_masks[i];
        valid &= bool_masks[i] == expected_mask;
        valid &= bool_output[i].x == bool_input[i].w;
        valid &= bool_output[i].y == bool_input[i].z;
        valid &= bool_output[i].z == bool_input[i].y;
        valid &= bool_output[i].w == bool_input[i].x;
        result.abi_layout_checksum +=
            static_cast<uint8_t>(byte_output[i].x) +
            static_cast<uint8_t>(byte_output[i].y) +
            static_cast<uint8_t>(byte_output[i].z) +
            static_cast<uint8_t>(byte_output[i].w);
        valid &= byte_output[i].x == byte_input[i].w;
        valid &= byte_output[i].y == byte_input[i].z;
        valid &= byte_output[i].z == byte_input[i].y;
        valid &= byte_output[i].w == byte_input[i].x;
    }
    constexpr auto tolerance = 1.0f / 255.0f + 1.0e-6f;
    valid &= std::abs(result.texture_read[0u] - 1.0f) < tolerance;
    valid &= std::abs(result.texture_read[1u] - 17.0f / 255.0f) < tolerance;
    valid &= std::abs(result.texture_read[2u] - 33.0f / 255.0f) < tolerance;
    valid &= std::abs(result.texture_read[3u] - 1.0f) < tolerance;
    return valid && result.abi_layout_checksum == 166u;
}

[[nodiscard]] bool run_timeline_event_smoke(
    Device &device,
    Metal4DeviceConformanceResult &result) noexcept {
    constexpr auto fence = uint64_t{1u} << 63u;
    constexpr auto expected = 0x4c434d34u;
    auto event = device.create_timeline_event();
    auto buffer = device.create_buffer<uint>(1u);
    auto producer = device.create_stream(StreamTag::COMPUTE);
    auto consumer = device.create_stream(StreamTag::COMPUTE);
    const std::array source{expected};
    std::array<uint, 1u> destination{};
    auto begin = Clock::now();
    producer << buffer.copy_from(luisa::span{source})
             << event.signal(fence);
    consumer << event.wait(fence)
             << buffer.copy_to(luisa::span{destination})
             << synchronize();
    producer << synchronize();
    result.timeline_event_ms = elapsed_ms(begin, Clock::now());
    if (event.is_completed(fence) && destination[0u] == expected) {
        result.timeline_value = fence;
        return true;
    }
    return false;
}

[[nodiscard]] bool run_native_include_smoke(
    Device &device,
    Stream &stream,
    Metal4DeviceConformanceResult &result) noexcept {
    ExternalCallable<float2(float2, float2)> get_uv{"get_uv"};
    ExternalCallable<void(float2 &, float2)> offset_uv{"offset_uv"};
    Kernel1D kernel = [&](BufferFloat2 output) noexcept {
        auto index = dispatch_id().x;
        auto coord = make_float2(
            cast<float>(index), cast<float>(index * 2u));
        auto uv = get_uv(coord, make_float2(4.0f, 8.0f));
        constexpr auto delta = make_float2(0.125f, 0.25f);
        offset_uv(uv, delta);
        output.write(index, uv - delta);
    };
    ShaderOption option{};
    option.enable_cache = false;
    option.name = "ios_device_native_include";
    option.native_include = R"(
define <2 x float> @get_uv(<2 x float> %coord, <2 x float> %size) {
entry:
    %half.x = insertelement <2 x float> poison, float 5.000000e-01, i32 0
    %half = insertelement <2 x float> %half.x, float 5.000000e-01, i32 1
    %center = fadd <2 x float> %coord, %half
    %uv = fdiv <2 x float> %center, %size
    ret <2 x float> %uv
}

define void @offset_uv(ptr noundef nonnull align 8 dereferenceable(8) %uv.ptr,
                       <2 x float> noundef %delta) {
entry:
    %uv = load <2 x float>, ptr %uv.ptr, align 8
    %adjusted = fadd <2 x float> %uv, %delta
    store <2 x float> %adjusted, ptr %uv.ptr, align 8
    ret void
}
)";
    auto begin = Clock::now();
    auto shader = device.compile(kernel, option);
    auto output = device.create_buffer<float2>(4u);
    std::array<float2, 4u> values{};
    stream << shader(output).dispatch(values.size())
           << output.copy_to(luisa::span{values})
           << synchronize();
    result.native_include_ms = elapsed_ms(begin, Clock::now());
    auto valid = true;
    for (auto i = 0u; i < values.size(); i++) {
        auto expected = make_float2(
            (static_cast<float>(i) + 0.5f) / 4.0f,
            (static_cast<float>(i * 2u) + 0.5f) / 8.0f);
        valid &= std::abs(values[i].x - expected.x) < 1.0e-6f;
        valid &= std::abs(values[i].y - expected.y) < 1.0e-6f;
        result.native_include_checksum += static_cast<uint32_t>(
            std::lround(values[i].x * 1024.0f));
        result.native_include_checksum += static_cast<uint32_t>(
            std::lround(values[i].y * 1024.0f));
    }
    return valid && result.native_include_checksum == 3840u;
}

[[nodiscard]] float4x4 translation_matrix(
    float x, float y, float z) noexcept {
    auto matrix = make_float4x4(1.0f);
    matrix[3u] = make_float4(x, y, z, 1.0f);
    return matrix;
}

struct MotionProbeResult {
    uint32_t hit_count{};
    double centroid_delta{};
};

[[nodiscard]] bool run_motion_instance_mode(
    Device &device,
    AccelMotionMode mode,
    MotionProbeResult &result) noexcept {
    constexpr auto width = 96u;
    constexpr auto height = 64u;
    constexpr std::array vertices{
        float3{-0.52f, -0.48f, 0.0f},
        float3{0.52f, -0.48f, 0.0f},
        float3{-0.08f, 0.62f, 0.0f},
        float3{-0.52f, -0.48f, 0.0f},
        float3{0.52f, -0.48f, 0.0f},
        float3{0.08f, 0.62f, 0.0f}};
    constexpr std::array triangles{Triangle{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    AccelOption mesh_option{};
    mesh_option.motion.keyframe_count = 2u;
    mesh_option.motion.time_start = 0.0f;
    mesh_option.motion.time_end = 1.0f;
    auto mesh = device.create_mesh(
        vertex_buffer, triangle_buffer, mesh_option);

    AccelMotionOption motion_option{};
    motion_option.mode = mode;
    motion_option.keyframe_count = 2u;
    motion_option.time_start = 0.0f;
    motion_option.time_end = 1.0f;
    auto motion_instance =
        device.create_motion_instance(mesh, motion_option);
    if (mode == AccelMotionMode::MATRIX) {
        const std::array keyframes{
            translation_matrix(-0.68f, 0.0f, 0.0f),
            translation_matrix(0.68f, 0.0f, -0.25f)};
        motion_instance.set_keyframes(luisa::span{keyframes});
    } else {
        const std::array keyframes{
            MotionInstanceTransformSRT{
                .translation = {-0.68f, 0.0f, 0.0f}},
            MotionInstanceTransformSRT{
                .quaternion = {0.0f, 0.0871557f, 0.0f, 0.9961947f},
                .scale = {1.08f, 0.96f, 1.0f},
                .shear = {0.08f, 0.0f, 0.0f},
                .translation = {0.68f, 0.0f, -0.25f}}};
        motion_instance.set_keyframes(luisa::span{keyframes});
    }

    auto accel = device.create_accel();
    accel.emplace_back(
        motion_instance, translation_matrix(0.0f, 0.0f, -0.2f),
        0xffu, true, 73u);
    auto hits_buffer = device.create_buffer<uint>(width * height);
    Kernel2D trace = [](AccelVar accel, BufferUInt hits) noexcept {
        auto coord = dispatch_id().xy();
        auto uv = (make_float2(coord) + 0.5f) /
                  make_float2(width, height);
        auto world = (uv * 2.0f - 1.0f) *
                     make_float2(1.6f, -1.2f);
        auto ray = make_ray(
            make_float3(world, 1.5f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto hit = accel.intersect_motion(ray, uv.y, {});
        hits.write(
            coord.y * 96u + coord.x,
            ite(hit->miss(), 0u, 1u));
    };
    ShaderOption option{};
    option.enable_cache = false;
    option.name = mode == AccelMotionMode::MATRIX ?
                      "ios_device_matrix_motion" :
                      "ios_device_component_motion";
    auto shader = device.compile(trace, option);
    luisa::vector<uint> hits(width * height);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << motion_instance.build()
           << accel.build()
           << shader(accel, hits_buffer).dispatch(width, height)
           << hits_buffer.copy_to(luisa::span{hits})
           << synchronize();

    auto upper_count = uint32_t{0u};
    auto lower_count = uint32_t{0u};
    auto upper_x_sum = uint64_t{0u};
    auto lower_x_sum = uint64_t{0u};
    auto min_x = width;
    auto max_x = 0u;
    for (auto y = 0u; y < height; y++) {
        for (auto x = 0u; x < width; x++) {
            if (hits[y * width + x] == 0u) { continue; }
            result.hit_count++;
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            if (y < height / 2u) {
                upper_count++;
                upper_x_sum += x;
            } else {
                lower_count++;
                lower_x_sum += x;
            }
        }
    }
    if (upper_count == 0u || lower_count == 0u) { return false; }
    auto upper_centroid = static_cast<double>(upper_x_sum) /
                          static_cast<double>(upper_count);
    auto lower_centroid = static_cast<double>(lower_x_sum) /
                          static_cast<double>(lower_count);
    result.centroid_delta = lower_centroid - upper_centroid;
    auto valid = result.hit_count > 200u &&
                 upper_count > 50u && lower_count > 50u &&
                 max_x >= min_x + width / 3u &&
                 result.centroid_delta > static_cast<double>(width) / 16.0;
    LUISA_INFO(
        "Metal4 {} motion geometry: valid={} upper={} lower={} "
        "x_range=[{}, {}] centroid_delta={:.3f}.",
        mode == AccelMotionMode::MATRIX ? "matrix" : "component",
        valid, upper_count, lower_count,
        min_x, max_x, result.centroid_delta);
    return valid;
}

[[nodiscard]] bool run_motion_smoke(
    Device &device,
    Metal4DeviceConformanceResult &result) noexcept {
    if (result.motion_blur != "true") { return false; }
    auto begin = Clock::now();
    MotionProbeResult matrix{};
    result.matrix_motion_valid = run_motion_instance_mode(
        device, AccelMotionMode::MATRIX, matrix);
    result.matrix_motion_hit_count = matrix.hit_count;
    result.matrix_motion_centroid_delta = matrix.centroid_delta;
    LUISA_INFO(
        "Metal4 matrix-motion probe: valid={} hits={} centroid_delta={:.3f}.",
        result.matrix_motion_valid,
        result.matrix_motion_hit_count,
        result.matrix_motion_centroid_delta);
    if (!result.matrix_motion_valid) { return false; }
    if (result.component_motion == "true") {
        result.component_motion_exercised = true;
        MotionProbeResult component{};
        result.component_motion_valid = run_motion_instance_mode(
            device, AccelMotionMode::SRT, component);
        result.component_motion_hit_count = component.hit_count;
        result.component_motion_centroid_delta =
            component.centroid_delta;
        LUISA_INFO(
            "Metal4 component-motion probe: valid={} hits={} "
            "centroid_delta={:.3f}.",
            result.component_motion_valid,
            result.component_motion_hit_count,
            result.component_motion_centroid_delta);
        if (!result.component_motion_valid) { return false; }
    }
    result.motion_instance_ms = elapsed_ms(begin, Clock::now());
    return true;
}

[[nodiscard]] bool run_bindless_indirect_smoke(
    Device &device,
    Stream &stream,
    Metal4DeviceConformanceResult &result) noexcept {
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
    Metal4DeviceConformanceResult &result) noexcept {
    constexpr auto width = 64u;
    constexpr auto height = 64u;

    auto raster = device.extension<RasterExt>();
    if (raster == nullptr) { return false; }

    RasterStageKernel vertex = [](Var<AppData> input) noexcept {
        Var<IOSMetal4RasterVarying> output;
        output.position = make_float4(input.position, 1.0f);
        output.color = input.color;
        output.base_instance = raster_base_instance();
        return output;
    };
    RasterStageKernel fragment = [](
                                     Var<IOSMetal4RasterVarying> input) noexcept {
        $if (input.base_instance != 19u) {
            raster_discard();
        };
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
    auto make_meshes = [&]() noexcept {
        luisa::vector<RasterMesh> meshes;
        meshes.emplace_back(
            luisa::span<const VertexBufferView>{&vertex_view, 1u},
            index_buffer.view(), 1u, 73u, 0, 19u);
        return meshes;
    };
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
                  make_meshes(), mesh_format,
                  Viewport{0u, 0u, width, height}, state,
                  &depth, render_target)
           << render_target.copy_to(luisa::span{pixels})
           << synchronize();

    for (auto pixel : pixels) {
        if (pixel[0u] > 2u || pixel[1u] > 2u || pixel[2u] > 2u) {
            result.raster_colored_pixels++;
        }
    }
    result.raster_center = pixels[(height / 2u) * width + width / 2u];
    auto center_sum = static_cast<uint32_t>(result.raster_center[0u]) +
                      result.raster_center[1u] +
                      result.raster_center[2u];
    if (result.raster_colored_pixels <= 900u ||
        result.raster_colored_pixels >= 1800u ||
        center_sum <= 80u) {
        return false;
    }

    auto stencil_state = [](Comparison comparison,
                            StencilOp pass) noexcept {
        auto face = StencilFaceOp{
            .stencil_fail_op = StencilOp::Keep,
            .depth_fail_op = StencilOp::Keep,
            .pass_op = pass,
            .comparison = comparison};
        RasterState stencil{};
        stencil.cull_mode = CullMode::None;
        stencil.depth_state = DepthState{
            .enable_depth = true,
            .comparison = Comparison::Always,
            .write = true};
        stencil.stencil_state = StencilState{
            .enable_stencil = true,
            .front_face_op = face,
            .back_face_op = face,
            .read_mask = 0xffu,
            .write_mask = 0xffu,
            .reference = 1u};
        return stencil;
    };
    for (auto format : {DepthFormat::D24S8,
                        DepthFormat::D32S8A24}) {
        auto stencil_depth = device.create_depth_buffer(
            format, make_uint2(width, height));
        std::fill(pixels.begin(), pixels.end(),
                  std::array<uint8_t, 4u>{});
        stream << stencil_depth.clear(1.0f)
               << raster->clear_render_target(
                      render_target.view(),
                      make_float4(0.0f, 0.0f, 0.0f, 1.0f))
               << shader().draw(
                      make_meshes(), mesh_format,
                      Viewport{0u, 0u, width, height},
                      stencil_state(Comparison::Always, StencilOp::Replace),
                      &stencil_depth, render_target)
               << raster->clear_render_target(
                      render_target.view(),
                      make_float4(0.0f, 0.0f, 0.0f, 1.0f))
               << shader().draw(
                      make_meshes(), mesh_format,
                      Viewport{0u, 0u, width, height},
                      stencil_state(Comparison::Equal, StencilOp::Keep),
                      &stencil_depth, render_target)
               << render_target.copy_to(luisa::span{pixels})
               << synchronize();
        auto colored = 0u;
        for (auto pixel : pixels) {
            colored += pixel[0u] > 2u || pixel[1u] > 2u ||
                       pixel[2u] > 2u;
        }
        if (colored <= 900u || colored >= 1800u) { return false; }
        result.raster_stencil_colored_pixels += colored;
    }
    result.raster_dispatch_readback_ms =
        elapsed_ms(dispatch_begin, Clock::now());
    return true;
}

}// namespace

Metal4DeviceConformanceResult run_metal4_device_conformance(
    Device &device,
    uint32_t width,
    uint32_t height,
    uint32_t samples_per_pixel) noexcept {
    Metal4DeviceConformanceResult result;
    result.device_name = device.query("device_name");
    result.gpu_family = device.query("metal4_gpu_family");
    result.metal4_runtime = device.query("metal4_runtime");
    result.ray_tracing = device.query("metal_ray_tracing");
    result.acceleration_structure_path = device.query(
        "metal4_address_driven_acceleration_structures");
    result.motion_blur = device.query("metal_motion_blur");
    result.component_motion = device.query("metal4_component_motion");
    if (result.device_name.empty() || result.gpu_family == "unknown" ||
        result.metal4_runtime != "true" || result.ray_tracing != "true" ||
        result.component_motion != result.acceleration_structure_path) {
        return fail(
            std::move(result), "feature guard",
            "Metal4 runtime/device-family/ray-tracing capability mismatch");
    }

    auto compute_stream = device.create_stream(StreamTag::COMPUTE);
    if (!run_printer_smoke(device, compute_stream, result)) {
        return fail(
            std::move(result), "shader logging",
            "Metal LogState callback did not return the expected AIR message");
    }
    if (!run_compute_abi_smoke(device, compute_stream, result)) {
        return fail(
            std::move(result), "compute ABI",
            "bool/i8 layout, atomics, or direct texture I/O mismatch");
    }
    if (!run_native_include_smoke(device, compute_stream, result)) {
        return fail(
            std::move(result), "native include",
            "ExternalCallable LLVM-linkage or mutable-reference ABI mismatch");
    }
    if (!run_timeline_event_smoke(device, result)) {
        return fail(
            std::move(result), "timeline event",
            "cross-stream unsigned Metal4 timeline ordering mismatch");
    }
    if (!run_motion_smoke(device, result)) {
        return fail(
            std::move(result), "motion ray tracing",
            "primitive/matrix/component motion traversal mismatch");
    }
    if (!run_bindless_indirect_smoke(device, compute_stream, result)) {
        return fail(
            std::move(result), "bindless/indirect",
            "bindless table or GPU-authored MTL4 indirect dispatch mismatch");
    }
    if (!run_raster_smoke(device, result)) {
        return fail(
            std::move(result), "raster",
            "Metal4 AIR base-instance or D24S8/D32S8A24 stencil draw mismatch");
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
