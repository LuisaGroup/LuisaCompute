#include <charconv>
#include <filesystem>
#include <fstream>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rhi/resource.h>

#include "metal_air_pipeline.h"
#include "metal_xir_pipeline.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::metal;

namespace {

[[nodiscard]] MetalAIRVersion parse_version(std::string_view text) noexcept {
    MetalAIRVersion version{};
    uint32_t *components[] = {
        &version.major, &version.minor, &version.patch};
    for (auto component = 0u; component < 3u; component++) {
        auto separator = text.find('.');
        auto token = text.substr(0u, separator);
        if (token.empty()) { return {}; }
        auto [end, error] = std::from_chars(
            token.data(), token.data() + token.size(), *components[component]);
        if (error != std::errc{} || end != token.data() + token.size()) {
            return {};
        }
        if (separator == std::string_view::npos) { break; }
        text.remove_prefix(separator + 1u);
    }
    return version;
}

[[nodiscard]] auto make_path_tracing_kernel() noexcept {
    constexpr auto epsilon = 1.e-3f;
    constexpr auto infinity = 1.e4f;

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt sum = def(0u);
        for (auto round = 0u; round < 4u; round++) {
            sum += 0x9e3779b9u;
            v0 += ((v1 << 4u) + 0xa341316cu) ^
                  (v1 + sum) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4u) + 0xad90777du) ^
                  (v0 + sum) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Callable random = [](UInt &state) noexcept {
        state = 1664525u * state + 1013904223u;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    Callable sdf = [](Float3 p) noexcept {
        Float sphere = length(p - make_float3(-0.65f, -0.05f, 0.0f)) - 0.95f;
        Float3 box_delta = abs(p - make_float3(0.85f, -0.35f, -0.25f)) -
                           make_float3(0.62f);
        Float box = length(max(box_delta, 0.0f)) +
                    min(max(max(box_delta.x, box_delta.y), box_delta.z), 0.0f);
        Float floor = p.y + 1.0f;
        return min(min(sphere, box), floor);
    };

    Callable material = [](Float3 p) noexcept {
        Float sphere = length(p - make_float3(-0.65f, -0.05f, 0.0f)) - 0.95f;
        Float3 box_delta = abs(p - make_float3(0.85f, -0.35f, -0.25f)) -
                           make_float3(0.62f);
        Float box = length(max(box_delta, 0.0f)) +
                    min(max(max(box_delta.x, box_delta.y), box_delta.z), 0.0f);
        Float checker = cast<float>(
            (cast<int>(floor(p.x * 1.5f)) +
             cast<int>(floor(p.z * 1.5f))) &
            1);
        Float3 floor_color = lerp(
            make_float3(0.18f, 0.20f, 0.24f),
            make_float3(0.72f, 0.75f, 0.80f), checker);
        return ite(sphere < box & sphere < p.y + 1.0f,
                   make_float3(0.78f, 0.20f, 0.10f),
                   ite(box < p.y + 1.0f,
                       make_float3(0.10f, 0.42f, 0.82f),
                       floor_color));
    };

    Callable ray_march = [&sdf, epsilon, infinity](
                             Float3 origin, Float3 direction) noexcept {
        Float distance = def(0.0f);
        $for (step, 96u) {
            Float d = sdf(origin + distance * direction);
            $if (d < epsilon | distance > 30.0f) { $break; };
            distance += d;
        };
        return ite(distance <= 30.0f, distance, infinity);
    };

    Callable normal = [&sdf](Float3 p) noexcept {
        constexpr auto e = 1.e-3f;
        Float center = sdf(p);
        return normalize(make_float3(
            sdf(p + make_float3(e, 0.0f, 0.0f)) - center,
            sdf(p + make_float3(0.0f, e, 0.0f)) - center,
            sdf(p + make_float3(0.0f, 0.0f, e)) - center));
    };

    Callable cosine_hemisphere = [&random](Float3 n, UInt &state) noexcept {
        Float3 tangent = normalize(ite(
            abs(n.y) < 0.999f,
            cross(make_float3(0.0f, 1.0f, 0.0f), n),
            make_float3(1.0f, 0.0f, 0.0f)));
        Float3 bitangent = cross(n, tangent);
        Float phi = 2.0f * constants::pi * random(state);
        Float r = sqrt(random(state));
        Float z = sqrt(max(0.0f, 1.0f - r * r));
        return normalize(
            tangent * (r * cos(phi)) +
            bitangent * (r * sin(phi)) + n * z);
    };

    Callable environment = [](Float3 direction) noexcept {
        Float horizon = saturate(0.5f * direction.y + 0.5f);
        Float3 sky = lerp(
            make_float3(0.65f, 0.72f, 0.82f),
            make_float3(0.10f, 0.24f, 0.55f), horizon);
        Float3 sun_direction = normalize(make_float3(-0.45f, 0.75f, 0.35f));
        Float sun = pow(max(dot(direction, sun_direction), 0.0f), 512.0f);
        return sky + make_float3(10.0f, 8.0f, 5.0f) * sun;
    };

    return Kernel2D{[=](ImageFloat output, UInt sample_count) noexcept {
        set_block_size(8u, 8u, 1u);
        UInt2 pixel = dispatch_id().xy();
        Float2 resolution = make_float2(dispatch_size().xy());
        Float3 color = def(make_float3(0.0f));
        $for (sample, sample_count) {
            UInt state = tea(
                pixel.x + pixel.y * dispatch_size().x,
                sample + 0x9e3779b9u);
            Float2 jitter = make_float2(random(state), random(state));
            Float2 uv = (make_float2(pixel) + jitter) / resolution;
            Float2 screen = (uv * 2.0f - 1.0f) *
                            make_float2(resolution.x / resolution.y, -1.0f);
            Float3 origin = def(make_float3(0.0f, 0.45f, 4.5f));
            Float3 direction = normalize(make_float3(screen * 0.62f, -1.0f));
            Float3 throughput = def(make_float3(1.0f));
            Float3 radiance = def(make_float3(0.0f));
            $for (depth, 7u) {
                Float distance = ray_march(origin, direction);
                $if (distance >= infinity) {
                    radiance += throughput * environment(direction);
                    $break;
                };
                Float3 p = origin + distance * direction;
                Float3 n = normal(p);
                throughput *= material(p);
                origin = p + n * epsilon;
                direction = cosine_hemisphere(n, state);
                $if (depth >= 3u) {
                    Float survival = clamp(
                        max(max(throughput.x, throughput.y), throughput.z),
                        0.08f, 0.95f);
                    $if (random(state) > survival) { $break; };
                    throughput /= survival;
                };
            };
            color += radiance;
        };
        color /= max(cast<float>(sample_count), 1.0f);
        color = color / (1.0f + color);
        color = sqrt(max(color, 0.0f));
        output.write(pixel, make_float4(color, 1.0f));
    }};
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 4) {
        LUISA_INFO(
            "Usage: {} <output.metallib> [iOS deployment version] [iOS SDK version]",
            argv[0]);
        return 2;
    }
    auto deployment = parse_version(argc >= 3 ? argv[2] : "26.0");
    auto sdk = parse_version(argc >= 4 ? argv[3] : "26.4");
    if (deployment.major == 0u || sdk.major == 0u) {
        LUISA_WARNING("Invalid iOS deployment or SDK version.");
        return 2;
    }

    auto kernel = make_path_tracing_kernel();
    auto option = ShaderOption{
        .enable_cache = false,
        .enable_fast_math = true,
        .enable_debug_info = false,
        .compile_only = true,
        .name = "luisa_ios_path_tracing"};
    auto module = metal_translate_ast_to_xir(
        kernel.function()->function(), option);
    auto target = metal_air_target_for_ios(deployment, sdk);
    auto air = metal_codegen_air(*module, option, target);
    constexpr auto expected_root_argument_size = 32u;
    if (air.library.empty() ||
        air.root_argument_size != expected_root_argument_size) {
        LUISA_WARNING(
            "Unexpected iOS path tracing AIR output: metallib={} bytes, root={} bytes (expected {}).",
            air.library.size(), air.root_argument_size,
            expected_root_argument_size);
        return 1;
    }

    auto output_path = std::filesystem::path{argv[1]};
    std::error_code error;
    if (auto parent = output_path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            LUISA_WARNING("Failed to create '{}': {}.",
                          parent.string(), error.message());
            return 1;
        }
    }
    std::ofstream output{output_path, std::ios::binary};
    output.write(reinterpret_cast<const char *>(air.library.data()),
                 static_cast<std::streamsize>(air.library.size()));
    if (!output) {
        LUISA_WARNING("Failed to write '{}'.", output_path.string());
        return 1;
    }
    LUISA_INFO(
        "Generated iOS AIR path tracer: '{}' ({} bytes, root={} bytes, block=8x8x1, target=iOS {}.{}.{}, SDK {}.{}).",
        output_path.string(), air.library.size(), air.root_argument_size,
        deployment.major, deployment.minor, deployment.patch,
        sdk.major, sdk.minor);
    return 0;
}
