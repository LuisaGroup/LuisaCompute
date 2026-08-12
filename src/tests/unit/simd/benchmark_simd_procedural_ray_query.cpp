// Runtime benchmark for rejection-heavy procedural ray queries.
//
// Run as:
//   benchmark_simd_procedural_ray_query fallback [candidate-count]
//   benchmark_simd_procedural_ray_query simd <1|2|4|8|16> [candidate-count]

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr auto default_candidate_count = 16u;
constexpr auto ray_count = 1u << 16u;
constexpr auto warmup_dispatch_count = 4u;
constexpr auto timed_dispatch_count = 128u;
constexpr auto sample_count = 7u;

[[nodiscard]] uint32_t parse_width(
    int argc, char *argv[], std::string_view backend) noexcept {
    if (backend != "simd") { return 0u; }
    if (argc < 3 || argv[2] == nullptr) { return 0u; }
    auto text = std::string_view{argv[2]};
    auto width = uint32_t{0u};
    auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), width);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        (width != 1u && width != 2u && width != 4u &&
         width != 8u && width != 16u)) {
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
        return 0u;
    }
    return count;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argv[1] == nullptr) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "benchmark")
                  << " <fallback|simd> [simd-width] [candidate-count]\n";
        return 1;
    }
    auto backend = std::string_view{argv[1]};
    if (backend != "fallback" && backend != "simd") {
        std::cerr << "Expected fallback or simd backend\n";
        return 1;
    }
    auto width = parse_width(argc, argv, backend);
    if (backend == "simd" && width == 0u) {
        std::cerr << "SIMD benchmark requires width 1, 2, 4, 8, or 16\n";
        return 1;
    }
    auto candidate_count = parse_candidate_count(argc, argv, backend);
    if (candidate_count == 0u) {
        std::cerr << "Candidate count must be in [1, 256]\n";
        return 1;
    }

    Context context{argc > 0 ? argv[0] : ""};
    DeviceConfig config{};
    if (backend == "simd") {
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
    }
    auto device = context.create_device(
        backend, backend == "simd" ? &config : nullptr);
    auto stream = device.create_stream();

    luisa::vector<AABB> boxes(candidate_count);
    for (auto &box : boxes) {
        box = AABB{
            .packed_min = {-2.0f, -2.0f, -0.1f},
            .packed_max = {2.0f, 2.0f, 0.1f}};
    }
    auto box_buffer = device.create_buffer<AABB>(boxes.size());
    auto procedural = device.create_procedural_primitive(box_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(procedural);

    Kernel1D query = [candidate_count](
                         AccelVar scene, BufferUInt2 output) noexcept {
        set_block_size(64u, 1u, 1u);
        auto index = dispatch_x();
        auto x = (cast<float>(index & 255u) + 0.5f) / 256.0f - 0.5f;
        auto ray = make_ray(
            make_float3(x, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 4.0f);
        UInt callbacks = 0u;
        auto committed = scene.traverse(ray, {})
                             .on_surface_candidate(
                                 [](SurfaceCandidate &) noexcept {})
                             .on_procedural_candidate(
                                 [&](ProceduralCandidate &candidate) noexcept {
                                     callbacks += 1u;
                                     $if (candidate.hit()->prim ==
                                          candidate_count - 1u) {
                                         candidate.commit(0.95f);
                                     };
                                 })
                             .trace();
        output.write(
            index,
            make_uint2(committed->prim, callbacks));
    };
    auto shader = device.compile(query);
    auto output = device.create_buffer<uint2>(ray_count);
    luisa::vector<uint2> host_output(ray_count);
    stream << box_buffer.copy_from(luisa::span{boxes})
           << procedural.build()
           << accel.build()
           << shader(accel, output).dispatch(ray_count)
           << output.copy_to(luisa::span{host_output})
           << synchronize();
    for (auto i = size_t{0u}; i < host_output.size(); i++) {
        auto expected = make_uint2(
            candidate_count - 1u, candidate_count);
        if (!all(host_output[i] == expected)) {
            std::cerr << "procedural benchmark mismatch at " << i
                      << ": got (" << host_output[i].x << ", "
                      << host_output[i].y << "), expected ("
                      << expected.x << ", " << expected.y << ")\n";
            return 2;
        }
    }

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
    auto callbacks = rays * candidate_count;
    std::cout << "procedural_ray_query_reject_chain"
              << ",backend=" << backend
              << ",width=" << (backend == "simd" ? width : 1u)
              << ",candidates=" << candidate_count
              << ",rays_per_sample=" << static_cast<uint64_t>(rays)
              << ",median_seconds=" << median_seconds
              << ",minimum_seconds=" << minimum_seconds
              << ",median_mrays_per_second="
              << rays / median_seconds * 1.0e-6
              << ",best_mrays_per_second="
              << rays / minimum_seconds * 1.0e-6
              << ",median_mcallbacks_per_second="
              << callbacks / median_seconds * 1.0e-6
              << ",samples_seconds=";
    for (auto i = size_t{0u}; i < samples.size(); i++) {
        if (i != 0u) { std::cout << ';'; }
        std::cout << samples[i];
    }
    std::cout << '\n';
    return 0;
}
