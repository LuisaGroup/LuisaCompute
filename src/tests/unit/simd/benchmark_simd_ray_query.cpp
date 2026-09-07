// Runtime benchmark for rejection-heavy triangle ray queries.
//
// Non-opaque triangle instances lie along every ray. The handler rejects all
// but the farthest candidate (sixteen by default). This isolates candidate-
// continuation cost while retaining the public DSL and backend boundary. Run
// as:
//   benchmark_simd_ray_query fallback [candidate-count] [structured|explicit]
//                            [counted|capture-free]
//   benchmark_simd_ray_query simd <1|2|4|8|16> [candidate-count]
//                            [structured|explicit] [counted|capture-free]

// The process performs warmup plus seven samples and reports both the median
// and minimum. Running the executable several times remains required on a
// shared host.

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr auto default_candidate_count = 16u;
constexpr auto ray_count = 1u << 18u;
constexpr auto warmup_dispatch_count = 3u;
constexpr auto timed_dispatch_count = 8u;
constexpr auto sample_count = 7u;

[[nodiscard]] uint32_t parse_width(
    int argc, char *argv[], std::string_view backend) noexcept {
    if (backend != "simd") { return 0u; }
    if (argc < 3 || argv[2] == nullptr) {
        std::cerr << "SIMD benchmark requires a width: 1, 2, 4, 8, or 16\n";
        return 0u;
    }
    auto text = std::string_view{argv[2]};
    auto width = uint32_t{0u};
    auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), width);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        (width != 1u && width != 2u && width != 4u &&
         width != 8u && width != 16u)) {
        std::cerr << "Invalid SIMD width '" << text << "'\n";
        return 0u;
    }
    return width;
}

[[nodiscard]] uint32_t parse_candidate_count(
    int argc, char *argv[], std::string_view backend) noexcept {
    auto argument = backend == "simd" ? 3 : 2;
    if (argc <= argument || argv[argument] == nullptr) {
        return default_candidate_count;
    }
    auto text = std::string_view{argv[argument]};
    auto count = uint32_t{0u};
    auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), count);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        count == 0u || count > 256u) {
        std::cerr << "Invalid candidate count '" << text
                  << "' (expected 1..256)\n";
        return 0u;
    }
    return count;
}

[[nodiscard]] bool validate(
    luisa::span<const uint2> values,
    uint32_t candidate_count) noexcept {
    for (auto i = size_t{0u}; i < values.size(); i++) {
        auto expected = make_uint2(
            candidate_count - 1u, candidate_count);
        if (!all(values[i] == expected)) {
            std::cerr << "ray-query benchmark mismatch at " << i
                      << ": expected (" << expected.x << ", "
                      << expected.y << "), got (" << values[i].x
                      << ", " << values[i].y << ")\n";
            return false;
        }
    }
    return true;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argv[1] == nullptr) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "benchmark")
                  << " <fallback|simd> [simd-width] [candidate-count] "
                     "[structured|explicit] [counted|capture-free]\n";
        return 1;
    }
    auto backend = std::string_view{argv[1]};
    if (backend != "fallback" && backend != "simd") {
        std::cerr << "Expected fallback or simd backend\n";
        return 1;
    }
    auto width = parse_width(argc, argv, backend);
    if (backend == "simd" && width == 0u) { return 1; }
    auto candidate_count = parse_candidate_count(argc, argv, backend);
    if (candidate_count == 0u) { return 1; }
    auto query_form_argument = backend == "simd" ? 4 : 3;
    auto query_form = argc > query_form_argument &&
                              argv[query_form_argument] != nullptr ?
                          std::string_view{argv[query_form_argument]} :
                          std::string_view{"structured"};
    if (query_form != "structured" && query_form != "explicit") {
        std::cerr << "Expected structured or explicit query form\n";
        return 1;
    }
    auto explicit_query = query_form == "explicit";
    auto payload_argument = query_form_argument + 1;
    auto payload = argc > payload_argument &&
                           argv[payload_argument] != nullptr ?
                       std::string_view{argv[payload_argument]} :
                       std::string_view{"counted"};
    if (payload != "counted" && payload != "capture-free") {
        std::cerr << "Expected counted or capture-free payload\n";
        return 1;
    }
    auto count_callbacks = payload == "counted";

    Context context{argc > 0 ? argv[0] : ""};
    DeviceConfig config{};
    if (backend == "simd") {
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
    }
    auto device = context.create_device(
        backend, backend == "simd" ? &config : nullptr);
    auto stream = device.create_stream();

    std::array vertices{
        make_float3(-2.0f, -2.0f, 0.0f),
        make_float3(2.0f, -2.0f, 0.0f),
        make_float3(0.0f, 2.0f, 0.0f)};
    std::array triangles{Triangle{0u, 1u, 2u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    for (auto candidate = 0u; candidate < candidate_count; candidate++) {
        accel.emplace_back(
            mesh,
            translation(make_float3(
                0.0f, 0.0f, -2.0f * static_cast<float>(candidate))),
            0xffu, false, candidate);
    }

    auto ray_t_max = 2.0f * static_cast<float>(candidate_count) + 2.0f;
    Kernel1D query = [candidate_count, ray_t_max, explicit_query,
                      count_callbacks](
                         AccelVar scene, BufferUInt2 output) noexcept {
        set_block_size(64u, 1u, 1u);
        auto index = dispatch_x();
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, ray_t_max);
        UInt callbacks = 0u;
        if (explicit_query) {
            auto query = scene.query(ray, AccelTraceOptions{});
            $while (query.proceed()) {
                $if (query.is_surface_candidate()) {
                    auto candidate = query.surface_candidate();
                    if (count_callbacks) { callbacks += 1u; }
                    $if (candidate.hit()->inst ==
                         candidate_count - 1u) {
                        candidate.commit();
                    };
                }
                $else {
                    static_cast<void>(query.procedural_candidate());
                };
            };
            if (count_callbacks) {
                output.write(
                    index,
                    make_uint2(
                        query.committed_hit()->inst, callbacks));
            } else {
                output.write(
                    index,
                    make_uint2(
                        query.committed_hit()->inst,
                        candidate_count));
            }
        } else {
            auto committed = scene.traverse(ray, AccelTraceOptions{})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         if (count_callbacks) {
                                             callbacks += 1u;
                                         }
                                         $if (candidate.hit()->inst ==
                                              candidate_count - 1u) {
                                             candidate.commit();
                                         };
                                     })
                                 .on_procedural_candidate(
                                     [](ProceduralCandidate &) noexcept {})
                                 .trace();
            if (count_callbacks) {
                output.write(
                    index,
                    make_uint2(committed->inst, callbacks));
            } else {
                output.write(
                    index,
                    make_uint2(
                        committed->inst, candidate_count));
            }
        }
    };
    auto shader = device.compile(query);
    auto output = device.create_buffer<uint2>(ray_count);
    luisa::vector<uint2> host_output(ray_count);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << shader(accel, output).dispatch(ray_count)
           << output.copy_to(luisa::span{host_output})
           << synchronize();
    if (!validate(host_output, candidate_count)) { return 2; }

    for (auto i = 0u; i < warmup_dispatch_count; i++) {
        stream << shader(accel, output).dispatch(ray_count);
    }
    stream << synchronize();

    std::array<double, sample_count> samples{};
    for (auto sample = 0u; sample < sample_count; sample++) {
        auto begin = std::chrono::steady_clock::now();
        for (auto dispatch = 0u; dispatch < timed_dispatch_count;
             dispatch++) {
            stream << shader(accel, output).dispatch(ray_count);
        }
        stream << synchronize();
        auto end = std::chrono::steady_clock::now();
        samples[sample] = std::chrono::duration<double>(
                              end - begin)
                              .count();
    }
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    auto median_seconds = sorted[sample_count / 2u];
    auto minimum_seconds = sorted.front();
    auto rays = static_cast<double>(ray_count) * timed_dispatch_count;
    auto candidate_callbacks = rays * candidate_count;
    std::cout << "ray_query_reject_chain"
              << ",backend=" << backend
              << ",width=" << (backend == "simd" ? width : 1u)
              << ",candidates=" << candidate_count
              << ",query_form=" << query_form
              << ",payload=" << payload
              << ",rays_per_sample=" << static_cast<uint64_t>(rays)
              << ",median_seconds=" << median_seconds
              << ",minimum_seconds=" << minimum_seconds
              << ",median_mrays_per_second="
              << rays / median_seconds * 1.0e-6
              << ",best_mrays_per_second="
              << rays / minimum_seconds * 1.0e-6
              << ",median_mcallbacks_per_second="
              << candidate_callbacks / median_seconds * 1.0e-6
              << ",samples_seconds=";
    for (auto i = size_t{0u}; i < samples.size(); i++) {
        if (i != 0u) { std::cout << ';'; }
        std::cout << samples[i];
    }
    std::cout << '\n';
    return 0;
}
