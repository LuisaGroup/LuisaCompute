// Standalone Luisa fallback/SIMD side of the ISPC comparison suite.
// This source is compiled only by run.py and is not part of project CMake.

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr auto sample_count = size_t{7u};
constexpr auto warmup_dispatch_count = uint32_t{2u};
constexpr auto mandelbrot_width = uint32_t{1536u};
constexpr auto mandelbrot_height = uint32_t{1024u};
constexpr auto mandelbrot_dispatch_count = uint32_t{16u};
constexpr auto stream_element_count = uint32_t{8u * 1024u * 1024u};
constexpr auto stream_dispatch_count = uint32_t{32u};
constexpr auto aos_element_count = uint32_t{4u * 1024u * 1024u};
constexpr auto aos_dispatch_count = uint32_t{16u};
constexpr auto gemm_matrix_size = uint32_t{256u};
constexpr auto gemm_dispatch_count = uint32_t{128u};
constexpr auto path_width = uint32_t{640u};
constexpr auto path_height = uint32_t{360u};
constexpr auto path_frame = uint32_t{17u};
constexpr auto path_dispatch_count = uint32_t{8u};

const ShaderOption precise_shader_option{
    .enable_fast_math = false,
};

struct Options {
    std::string_view runtime_directory;
    std::string_view backend;
    uint32_t width{0u};
    uint32_t workers{0u};
    std::string_view workload;
    std::string_view dump_path;
};

[[nodiscard]] bool parse_uint32(
    std::string_view text, uint32_t &value) noexcept {
    auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} &&
           result.ptr == text.data() + text.size();
}

[[nodiscard]] std::optional<Options> parse_options(
    int argc, char *argv[]) noexcept {
    if (argc < 6 || argc > 7) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "luisa_runner")
                  << " <runtime-dir> <fallback|simd> <width> <workers>"
                     " <workload> [dump-path]\n";
        return std::nullopt;
    }
    Options options{
        .runtime_directory = argv[1],
        .backend = argv[2],
        .workload = argv[5],
        .dump_path = argc == 7 ? std::string_view{argv[6]} :
                                 std::string_view{},
    };
    if (!parse_uint32(argv[3], options.width) ||
        !parse_uint32(argv[4], options.workers) ||
        options.workers == 0u) {
        std::cerr << "Invalid width or worker count.\n";
        return std::nullopt;
    }
    if (options.backend == "fallback") {
        if (options.width != 0u) {
            std::cerr << "Fallback width must be zero.\n";
            return std::nullopt;
        }
    } else if (options.backend != "simd" ||
               (options.width != 1u && options.width != 2u &&
                options.width != 4u && options.width != 8u &&
                options.width != 16u)) {
        std::cerr << "SIMD width must be 1, 2, 4, 8, or 16.\n";
        return std::nullopt;
    }
    if (options.workload != "mandelbrot" &&
        options.workload != "masked_stream" &&
        options.workload != "aos_to_soa" &&
        options.workload != "gemm" &&
        options.workload != "path_trace") {
        std::cerr << "Unknown workload '" << options.workload << "'.\n";
        return std::nullopt;
    }
    return options;
}

[[nodiscard]] double median(
    std::array<double, sample_count> values) noexcept {
    std::ranges::sort(values);
    return values[sample_count / 2u];
}

template<typename T>
[[nodiscard]] uint64_t hash_values(luisa::span<T> values) noexcept {
    constexpr auto offset = uint64_t{1469598103934665603ull};
    constexpr auto prime = uint64_t{1099511628211ull};
    auto hash = offset;
    auto bytes = luisa::span{
        reinterpret_cast<const uint8_t *>(values.data()),
        values.size_bytes()};
    for (auto byte : bytes) {
        hash ^= byte;
        hash *= prime;
    }
    return hash;
}

template<typename T>
[[nodiscard]] bool dump_values(
    std::string_view path, luisa::span<T> values) noexcept {
    if (path.empty()) { return true; }
    std::ofstream stream{std::string{path}, std::ios::binary};
    stream.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size_bytes()));
    return stream.good();
}

struct Measurement {
    std::array<double, sample_count> samples{};
    uint64_t item_count{0u};
    uint32_t dispatch_count{0u};
    uint64_t checksum{0u};
    double rate_numerator{0.0};
    double rate_scale{1.0e-6};
    std::string_view rate_unit{"mitems_per_second"};
};

template<typename Dispatch>
[[nodiscard]] auto measure(
    Stream &stream, uint32_t dispatch_count,
    uint64_t item_count, Dispatch &&dispatch) noexcept {
    for (auto i = 0u; i < warmup_dispatch_count; i++) {
        dispatch(stream);
    }
    stream << synchronize();
    Measurement measurement{
        .item_count = item_count,
        .dispatch_count = dispatch_count,
    };
    for (auto sample = size_t{0u}; sample < sample_count; sample++) {
        auto begin = std::chrono::steady_clock::now();
        for (auto i = 0u; i < dispatch_count; i++) {
            dispatch(stream);
        }
        stream << synchronize();
        auto end = std::chrono::steady_clock::now();
        measurement.samples[sample] =
            std::chrono::duration<double>(end - begin).count();
    }
    return measurement;
}

[[nodiscard]] auto make_mandelbrot_kernel() noexcept {
    return Kernel2D{[](BufferUInt output) noexcept {
        auto x = dispatch_id().x;
        auto y = dispatch_id().y;
        auto c_re = -2.0f + cast<float>(x) *
                                (3.0f / static_cast<float>(mandelbrot_width));
        auto c_im = -1.25f + cast<float>(y) *
                                 (2.5f / static_cast<float>(mandelbrot_height));
        auto z_re = def(c_re);
        auto z_im = def(c_im);
        UInt iteration = 0u;
        $while (iteration < 256u &
                z_re * z_re + z_im * z_im <= 4.0f) {
            auto next_re = z_re * z_re - z_im * z_im + c_re;
            auto next_im = 2.0f * z_re * z_im + c_im;
            z_re = next_re;
            z_im = next_im;
            iteration += 1u;
        };
        output.write(y * mandelbrot_width + x, iteration);
    }};
}

[[nodiscard]] auto make_masked_stream_kernel() noexcept {
    return Kernel1D{[](BufferFloat a, BufferFloat b, BufferUInt mask,
                       BufferFloat output) noexcept {
        auto index = dispatch_id().x;
        Float value = 0.0f;
        $if (mask.read(index) != 0u) {
            value = b.read(index) - a.read(index);
        };
        output.write(index, value);
    }};
}

[[nodiscard]] auto make_aos_to_soa_kernel() noexcept {
    return Kernel1D{[](BufferFloat input, BufferFloat output_x,
                       BufferFloat output_y, BufferFloat output_z,
                       BufferFloat output_w) noexcept {
        auto index = dispatch_id().x;
        auto base = index * 4u;
        output_x.write(index, input.read(base + 0u));
        output_y.write(index, input.read(base + 1u));
        output_z.write(index, input.read(base + 2u));
        output_w.write(index, input.read(base + 3u));
    }};
}

[[nodiscard]] auto make_gemm_kernel() noexcept {
    return Kernel2D{[](BufferFloat lhs, BufferFloat rhs,
                       BufferFloat output) noexcept {
        auto column = dispatch_id().x;
        auto row = dispatch_id().y;
        Float sum = 0.0f;
        for (auto inner : dynamic_range(gemm_matrix_size)) {
            sum += lhs.read(row * gemm_matrix_size + inner) *
                   rhs.read(inner * gemm_matrix_size + column);
        }
        output.write(row * gemm_matrix_size + column, sum);
    }};
}

[[nodiscard]] auto make_path_trace_kernel() noexcept {
    Callable normalize_exact = [](Float3 value) noexcept {
        return value * (1.0f / sqrt(dot(value, value)));
    };
    Callable random_float = [](UInt &state) noexcept {
        state = state * 1664525u + 1013904223u;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / 16777216.0f);
    };
    Callable pixel_seed = [](UInt x, UInt y, UInt frame) noexcept {
        UInt state = x * 1973u + y * 9277u + frame * 26699u + 911u;
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    };
    Callable intersect_sphere = [](
                                    Float3 origin, Float3 direction,
                                    Float3 center, Float radius,
                                    UInt material, Float &hit_t,
                                    Float3 &normal,
                                    UInt &hit_material) noexcept {
        auto oc = origin - center;
        auto b = dot(oc, direction);
        auto c = dot(oc, oc) - radius * radius;
        auto discriminant = b * b - c;
        $if (discriminant > 0.0f) {
            auto t = -b - sqrt(discriminant);
            $if (t > 1.0e-4f & t < hit_t) {
                hit_t = t;
                normal = (origin + direction * t - center) / radius;
                hit_material = material;
            };
        };
    };
    return Kernel2D{[=](BufferFloat4 output) noexcept {
        auto x = dispatch_id().x;
        auto y = dispatch_id().y;
        auto aspect = static_cast<float>(path_width) /
                      static_cast<float>(path_height);
        auto px = (2.0f * (cast<float>(x) + 0.5f) /
                       static_cast<float>(path_width) -
                   1.0f) *
                  aspect;
        auto py = 1.0f - 2.0f * (cast<float>(y) + 0.5f) /
                             static_cast<float>(path_height);
        Float3 origin = make_float3(0.0f, 0.6f, 4.5f);
        Float3 direction = normalize_exact(make_float3(px, py, -1.6f));
        Float3 throughput = make_float3(1.0f);
        Float3 radiance = make_float3(0.0f);
        UInt state = pixel_seed(x, y, path_frame);
        for (auto depth : dynamic_range(8u)) {
            Float hit_t = 1.0e30f;
            Float3 normal = make_float3(0.0f);
            UInt material = 0xffffffffu;
            intersect_sphere(
                origin, direction, make_float3(-1.1f, 0.0f, -3.5f),
                1.0f, 0u, hit_t, normal, material);
            intersect_sphere(
                origin, direction, make_float3(1.1f, -0.1f, -4.0f),
                0.9f, 1u, hit_t, normal, material);
            intersect_sphere(
                origin, direction, make_float3(0.0f, 3.4f, -3.0f),
                0.75f, 2u, hit_t, normal, material);
            intersect_sphere(
                origin, direction, make_float3(0.0f, 0.1f, -6.2f),
                1.3f, 3u, hit_t, normal, material);
            $if (abs(direction.y) > 1.0e-6f) {
                auto plane_t = (-1.0f - origin.y) / direction.y;
                $if (plane_t > 1.0e-4f & plane_t < hit_t) {
                    hit_t = plane_t;
                    normal = make_float3(0.0f, 1.0f, 0.0f);
                    material = 4u;
                };
            };
            $if (material == 0xffffffffu) {
                auto blend = 0.5f * (direction.y + 1.0f);
                auto sky = lerp(
                    make_float3(0.08f, 0.10f, 0.15f),
                    make_float3(0.35f, 0.50f, 0.80f), blend);
                radiance += throughput * sky;
                $break;
            };
            $if (material == 2u) {
                radiance += throughput * make_float3(7.0f, 6.5f, 5.5f);
                $break;
            };
            Float3 albedo = make_float3(0.72f);
            $if (material == 0u) {
                albedo = make_float3(0.82f, 0.20f, 0.16f);
            }
            $elif (material == 1u) {
                albedo = make_float3(0.18f, 0.72f, 0.25f);
            }
            $elif (material == 3u) {
                albedo = make_float3(0.20f, 0.30f, 0.82f);
            };
            throughput *= albedo;
            auto hit_position = origin + direction * hit_t;
            auto random_x = 2.0f * random_float(state) - 1.0f;
            auto random_y = 2.0f * random_float(state) - 1.0f;
            auto random_z = 2.0f * random_float(state) - 1.0f;
            Float3 random_direction = make_float3(
                random_x, random_y, random_z);
            auto random_length_squared = length_squared(random_direction);
            $if (random_length_squared < 1.0e-12f) {
                random_direction = normal;
            }
            $else {
                random_direction *= 1.0f / sqrt(random_length_squared);
            };
            $if (dot(random_direction, normal) < 0.0f) {
                random_direction = -random_direction;
            };
            origin = hit_position + normal * 1.0e-3f;
            direction = normalize_exact(normal + random_direction);
            $if (depth >= 3u) {
                auto survival = clamp(
                    max(throughput.x,
                        max(throughput.y, throughput.z)),
                    0.1f, 0.95f);
                $if (random_float(state) >= survival) {
                    $break;
                };
                throughput /= survival;
            };
        }
        output.write(y * path_width + x, make_float4(radiance, 1.0f));
    }};
}

[[nodiscard]] Measurement run_mandelbrot(
    Device &device, Stream &stream, std::string_view dump_path) noexcept {
    auto count = static_cast<size_t>(mandelbrot_width) * mandelbrot_height;
    auto output = device.create_buffer<uint32_t>(count);
    auto shader = device.compile(
        make_mandelbrot_kernel(), precise_shader_option);
    auto result = measure(
        stream, mandelbrot_dispatch_count, count,
        [&](Stream &target) noexcept {
            target << shader(output).dispatch(
                mandelbrot_width, mandelbrot_height);
        });
    luisa::vector<uint32_t> host_output(count);
    stream << output.copy_to(luisa::span{host_output}) << synchronize();
    result.checksum = hash_values(luisa::span{host_output});
    if (!dump_values(dump_path, luisa::span{host_output})) {
        LUISA_ERROR("Failed to write benchmark output '{}'.", dump_path);
    }
    return result;
}

[[nodiscard]] Measurement run_masked_stream(
    Device &device, Stream &stream, std::string_view dump_path) noexcept {
    luisa::vector<float> a(stream_element_count);
    luisa::vector<float> b(stream_element_count);
    luisa::vector<uint32_t> mask(stream_element_count);
    for (auto i = uint32_t{0u}; i < stream_element_count; i++) {
        a[i] = static_cast<float>(static_cast<int32_t>(i % 257u) - 128) *
               (1.0f / 128.0f);
        b[i] = static_cast<float>(static_cast<int32_t>((i * 17u) % 263u) - 131) *
               (1.0f / 64.0f);
        auto bits = i * 747796405u + 2891336453u;
        mask[i] = ((bits >> 29u) & 3u) != 0u ? 1u : 0u;
    }
    auto a_buffer = device.create_buffer<float>(stream_element_count);
    auto b_buffer = device.create_buffer<float>(stream_element_count);
    auto mask_buffer = device.create_buffer<uint32_t>(stream_element_count);
    auto output = device.create_buffer<float>(stream_element_count);
    auto shader = device.compile(
        make_masked_stream_kernel(), precise_shader_option);
    stream << a_buffer.copy_from(luisa::span{a})
           << b_buffer.copy_from(luisa::span{b})
           << mask_buffer.copy_from(luisa::span{mask})
           << synchronize();
    auto result = measure(
        stream, stream_dispatch_count, stream_element_count,
        [&](Stream &target) noexcept {
            target << shader(a_buffer, b_buffer, mask_buffer, output)
                          .dispatch(stream_element_count);
        });
    luisa::vector<float> host_output(stream_element_count);
    stream << output.copy_to(luisa::span{host_output}) << synchronize();
    for (auto i = uint32_t{0u}; i < stream_element_count; i++) {
        auto expected = mask[i] != 0u ? b[i] - a[i] : 0.0f;
        if (host_output[i] != expected) {
            LUISA_ERROR(
                "masked_stream mismatch at {}: expected {}, got {}.",
                i, expected, host_output[i]);
        }
    }
    result.checksum = hash_values(luisa::span{host_output});
    if (!dump_values(dump_path, luisa::span{host_output})) {
        LUISA_ERROR("Failed to write benchmark output '{}'.", dump_path);
    }
    return result;
}

[[nodiscard]] Measurement run_aos_to_soa(
    Device &device, Stream &stream, std::string_view dump_path) noexcept {
    luisa::vector<float> input(
        static_cast<size_t>(aos_element_count) * 4u);
    for (auto i = size_t{0u}; i < input.size(); i++) {
        input[i] = static_cast<float>(
                       static_cast<int32_t>((i * 29u + 7u) % 509u) - 254) *
                   (1.0f / 256.0f);
    }
    auto input_buffer = device.create_buffer<float>(input.size());
    auto output_x = device.create_buffer<float>(aos_element_count);
    auto output_y = device.create_buffer<float>(aos_element_count);
    auto output_z = device.create_buffer<float>(aos_element_count);
    auto output_w = device.create_buffer<float>(aos_element_count);
    auto shader = device.compile(
        make_aos_to_soa_kernel(), precise_shader_option);
    stream << input_buffer.copy_from(luisa::span{input}) << synchronize();
    auto result = measure(
        stream, aos_dispatch_count, aos_element_count,
        [&](Stream &target) noexcept {
            target << shader(
                          input_buffer, output_x, output_y,
                          output_z, output_w)
                          .dispatch(aos_element_count);
        });
    luisa::vector<float> host_output(
        static_cast<size_t>(aos_element_count) * 4u);
    stream << output_x.copy_to(luisa::span{
                  host_output.data(), aos_element_count})
           << output_y.copy_to(luisa::span{
                  host_output.data() + aos_element_count,
                  aos_element_count})
           << output_z.copy_to(luisa::span{
                  host_output.data() + 2u * aos_element_count,
                  aos_element_count})
           << output_w.copy_to(luisa::span{
                  host_output.data() + 3u * aos_element_count,
                  aos_element_count})
           << synchronize();
    for (auto i = uint32_t{0u}; i < aos_element_count; i++) {
        for (auto component = uint32_t{0u}; component < 4u; component++) {
            auto observed = host_output[static_cast<size_t>(component) * aos_element_count + i];
            auto expected = input[static_cast<size_t>(i) * 4u + component];
            if (observed != expected) {
                LUISA_ERROR(
                    "aos_to_soa mismatch at {} component {}.",
                    i, component);
            }
        }
    }
    result.checksum = hash_values(luisa::span{host_output});
    if (!dump_values(dump_path, luisa::span{host_output})) {
        LUISA_ERROR("Failed to write benchmark output '{}'.", dump_path);
    }
    return result;
}

[[nodiscard]] Measurement run_gemm(
    Device &device, Stream &stream, std::string_view dump_path) noexcept {
    auto element_count = static_cast<size_t>(gemm_matrix_size) *
                         gemm_matrix_size;
    luisa::vector<float> lhs(element_count);
    luisa::vector<float> rhs(element_count);
    for (auto i = size_t{0u}; i < element_count; i++) {
        auto lhs_value = static_cast<int32_t>((i * 17u + 3u) % 29u) - 14;
        auto rhs_value = static_cast<int32_t>((i * 11u + 5u) % 31u) - 15;
        lhs[i] = static_cast<float>(lhs_value) * 0.03125f;
        rhs[i] = static_cast<float>(rhs_value) * 0.025f;
    }
    auto lhs_buffer = device.create_buffer<float>(element_count);
    auto rhs_buffer = device.create_buffer<float>(element_count);
    auto output = device.create_buffer<float>(element_count);
    auto shader = device.compile(make_gemm_kernel(), precise_shader_option);
    stream << lhs_buffer.copy_from(luisa::span{lhs})
           << rhs_buffer.copy_from(luisa::span{rhs})
           << synchronize();
    auto result = measure(
        stream, gemm_dispatch_count, element_count,
        [&](Stream &target) noexcept {
            target << shader(lhs_buffer, rhs_buffer, output)
                          .dispatch(gemm_matrix_size, gemm_matrix_size);
        });
    luisa::vector<float> host_output(element_count);
    stream << output.copy_to(luisa::span{host_output}) << synchronize();
    for (auto row = uint32_t{0u}; row < gemm_matrix_size; row++) {
        for (auto column = uint32_t{0u}; column < gemm_matrix_size;
             column++) {
            auto expected = 0.0;
            for (auto inner = uint32_t{0u}; inner < gemm_matrix_size;
                 inner++) {
                expected += static_cast<double>(
                                lhs[row * gemm_matrix_size + inner]) *
                            static_cast<double>(
                                rhs[inner * gemm_matrix_size + column]);
            }
            auto observed = host_output[row * gemm_matrix_size + column];
            auto tolerance = 2.0e-4 + 2.0e-4 * std::abs(expected);
            if (!std::isfinite(observed) ||
                std::abs(static_cast<double>(observed) - expected) >
                    tolerance) {
                LUISA_ERROR("gemm mismatch at ({}, {}).", row, column);
            }
        }
    }
    result.checksum = hash_values(luisa::span{host_output});
    result.rate_numerator =
        2.0 * gemm_matrix_size * gemm_matrix_size * gemm_matrix_size *
        gemm_dispatch_count;
    result.rate_scale = 1.0e-9;
    result.rate_unit = "gflop_per_second";
    if (!dump_values(dump_path, luisa::span{host_output})) {
        LUISA_ERROR("Failed to write benchmark output '{}'.", dump_path);
    }
    return result;
}

[[nodiscard]] Measurement run_path_trace(
    Device &device, Stream &stream, std::string_view dump_path) noexcept {
    auto count = static_cast<size_t>(path_width) * path_height;
    auto output = device.create_buffer<float4>(count);
    auto shader = device.compile(
        make_path_trace_kernel(), precise_shader_option);
    auto result = measure(
        stream, path_dispatch_count, count,
        [&](Stream &target) noexcept {
            target << shader(output).dispatch(path_width, path_height);
        });
    luisa::vector<float4> host_output(count);
    stream << output.copy_to(luisa::span{host_output}) << synchronize();
    for (auto i = size_t{0u}; i < host_output.size(); i++) {
        auto value = host_output[i];
        if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
            !std::isfinite(value.z) || value.w != 1.0f) {
            LUISA_ERROR("path_trace produced invalid output at {}.", i);
        }
    }
    result.checksum = hash_values(luisa::span{host_output});
    if (!dump_values(dump_path, luisa::span{host_output})) {
        LUISA_ERROR("Failed to write benchmark output '{}'.", dump_path);
    }
    return result;
}

}// namespace

int main(int argc, char *argv[]) {
    auto options = parse_options(argc, argv);
    if (!options) { return 1; }
    Context context{options->runtime_directory};
    DeviceConfig config{};
    luisa::unique_ptr<SIMDDeviceConfigExt> simd_config;
    if (options->backend == "simd") {
        simd_config = luisa::make_unique<SIMDDeviceConfigExt>(
            options->width, options->workers);
        config.extension = std::move(simd_config);
    }
    auto device = context.create_device(
        options->backend,
        options->backend == "simd" ? &config : nullptr);
    auto stream = device.create_stream();
    Measurement result{};
    if (options->workload == "mandelbrot") {
        result = run_mandelbrot(device, stream, options->dump_path);
    } else if (options->workload == "masked_stream") {
        result = run_masked_stream(device, stream, options->dump_path);
    } else if (options->workload == "aos_to_soa") {
        result = run_aos_to_soa(device, stream, options->dump_path);
    } else if (options->workload == "gemm") {
        result = run_gemm(device, stream, options->dump_path);
    } else {
        result = run_path_trace(device, stream, options->dump_path);
    }
    auto seconds = median(result.samples);
    auto rate_numerator = result.rate_numerator == 0.0 ?
                              static_cast<double>(result.item_count) *
                                  result.dispatch_count :
                              result.rate_numerator;
    auto rate = rate_numerator / seconds * result.rate_scale;
    std::cout << "simd_ispc_suite"
              << ",implementation=luisa"
              << ",backend=" << options->backend
              << ",width=" << options->width
              << ",workers=" << options->workers
              << ",workload=" << options->workload
              << ",items=" << result.item_count
              << ",dispatches=" << result.dispatch_count
              << ",median_seconds=" << std::setprecision(12) << seconds
              << ",rate_unit=" << result.rate_unit
              << ",median_rate=" << rate
              << ",checksum=" << std::hex << result.checksum << std::dec
              << ",samples_seconds=";
    for (auto i = size_t{0u}; i < result.samples.size(); i++) {
        if (i != 0u) { std::cout << ';'; }
        std::cout << result.samples[i];
    }
    std::cout << '\n';
    return 0;
}
