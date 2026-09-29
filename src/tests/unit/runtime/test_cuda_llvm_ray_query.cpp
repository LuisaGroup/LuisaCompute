// CUDA ray-query lifecycle regression and reproducible dispatch benchmark.
// Benchmark mode runs the same uncached commit/capture kernel through LLVM or
// AST/NVRTC according to the backend environment. Existing tests cover ray fields;
// this fixture covers real termination, independent captures and repeated
// queries without assuming a candidate visitation order.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

struct Options {
    bool benchmark{};
    bool diagnose_dispatches{};
    uint32_t rays{257u};
    uint32_t warmup{3u};
    uint32_t dispatches{8u};
    uint32_t samples{7u};
};

[[nodiscard]] bool parse_options(int argc, char *argv[], Options &options) noexcept {
    bool explicit_rays = false;
    for (auto i = 2; i < argc; i++) {
        auto argument = luisa::string_view{argv[i]};
        if (argument == "--benchmark") {
            options.benchmark = true;
            continue;
        }
        if (argument == "--diagnose-dispatches") {
            options.diagnose_dispatches = true;
            continue;
        }
        uint32_t *value = nullptr;
        if (argument == "--rays") {
            value = &options.rays;
            explicit_rays = true;
        }
        if (argument == "--warmup") { value = &options.warmup; }
        if (argument == "--dispatches") { value = &options.dispatches; }
        if (argument == "--samples") { value = &options.samples; }
        if (value == nullptr || ++i == argc) { return false; }
        auto text = luisa::string_view{argv[i]};
        auto parsed = std::from_chars(text.data(), text.data() + text.size(), *value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || *value == 0u) { return false; }
    }
    if (options.benchmark && !explicit_rays) { options.rays = 65536u; }
    return options.rays <= 1048576u && options.samples <= 101u &&
           options.warmup <= 1000u && options.dispatches <= 1000u;
}

[[nodiscard]] bool run(Device &device, const Options &options) {
    constexpr auto query_count = 5u;
    auto stream = device.create_stream();
    const std::array vertices{
        make_float3(-2.0f, -2.0f, 0.0f),
        make_float3(2.0f, -2.0f, 0.0f),
        make_float3(0.0f, 2.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    const std::array boxes{
        AABB{.packed_min = {-1.0f, -1.0f, -0.5f}, .packed_max = {1.0f, 1.0f, 0.5f}},
        AABB{.packed_min = {-1.0f, -1.0f, -0.5f}, .packed_max = {1.0f, 1.0f, 0.5f}}};
    auto vertices_buffer = device.create_buffer<float3>(vertices.size());
    auto triangles_buffer = device.create_buffer<Triangle>(triangles.size());
    auto boxes_buffer = device.create_buffer<AABB>(boxes.size());
    auto mesh = device.create_mesh(vertices_buffer, triangles_buffer);
    auto procedural = device.create_procedural_primitive(boxes_buffer);
    auto single_procedural = device.create_procedural_primitive(boxes_buffer.view(0u, 1u));
    auto surface_scene = device.create_accel();
    auto procedural_scene = device.create_accel();
    auto preserve_scene = device.create_accel();
    surface_scene.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    procedural_scene.emplace_back(single_procedural);
    preserve_scene.emplace_back(procedural);

    // Termination without a commit is a separate correctness requirement. The
    // benchmark always commits before terminating so that older AST backends
    // can be compared, with their own exact result checks, on a shared workload.
    Kernel1D kernel = [benchmark = options.benchmark](AccelVar surfaces, AccelVar procedurals, AccelVar preserve_accel,
                                                      BufferUInt4 results, BufferUInt4 captures,
                                                      BufferUInt callback_writes, BufferFloat distances) noexcept {
        set_block_size(64u);
        auto index = dispatch_x();
        auto base = index * query_count;
        auto x = (cast<float>(index % 31u) + 0.5f) / 31.0f - 0.5f;
        auto ray = make_ray(make_float3(x, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
        UInt first_capture = index * 3u + 7u;
        UInt second_capture = index * 5u + 11u;

        UInt surface_reject_count = 0u;
        auto rejected_surface = surfaces.traverse(ray, {})
                                    .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                        surface_reject_count += 1u;
                                        first_capture += 3u;
                                        callback_writes.write(base, index ^ 0x1234u);
                                        if (benchmark) { candidate.commit(); }
                                        candidate.terminate();
                                    })
                                    .trace();
        results.write(base, make_uint4(rejected_surface->hit_type, rejected_surface->inst,
                                       rejected_surface->prim, surface_reject_count));
        UInt saved_first = first_capture;

        UInt surface_commit_count = 0u;
        auto committed_surface = surfaces.traverse(ray, {})
                                     .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                         surface_commit_count += 1u;
                                         second_capture += 5u;
                                         callback_writes.write(base + 1u, index ^ 0x5678u);
                                         candidate.commit();
                                         candidate.terminate();
                                     })
                                     .trace();
        results.write(base + 1u, make_uint4(committed_surface->hit_type, committed_surface->inst,
                                            committed_surface->prim, surface_commit_count));
        UInt saved_second = second_capture;

        UInt procedural_reject_count = 0u;
        auto rejected_procedural = procedurals.traverse(ray, {})
                                       .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                           procedural_reject_count += 1u;
                                           first_capture += 7u;
                                           callback_writes.write(base + 2u, index ^ 0x9abcu);
                                           if (benchmark) { candidate.commit(1.0f); }
                                           candidate.terminate();
                                       })
                                       .trace();
        results.write(base + 2u, make_uint4(rejected_procedural->hit_type, rejected_procedural->inst,
                                            rejected_procedural->prim, procedural_reject_count));

        UInt procedural_commit_count = 0u;
        auto committed_procedural = procedurals.traverse(ray, {})
                                        .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                            procedural_commit_count += 1u;
                                            second_capture += 11u;
                                            callback_writes.write(base + 3u, index ^ 0xdef0u);
                                            candidate.commit(1.0f);
                                            candidate.terminate();
                                        })
                                        .trace();
        results.write(base + 3u, make_uint4(committed_procedural->hit_type, committed_procedural->inst,
                                            committed_procedural->prim, procedural_commit_count));
        captures.write(index, make_uint4(first_capture, second_capture, saved_first, saved_second));

        // Both AABBs overlap the accepted interval. Whichever primitive is
        // visited first commits; the second terminates without a replacement.
        UInt preserve_count = 0u;
        UInt first_primitive = ~0u;
        auto preserved = preserve_accel.traverse(ray, {})
                             .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                 preserve_count += 1u;
                                 $if (preserve_count == 1u) {
                                     first_primitive = candidate.hit()->prim;
                                     candidate.commit(1.25f);
                                     if (benchmark) { candidate.terminate(); }
                                 }
                                 $else {
                                     candidate.terminate();
                                 };
                             })
                             .trace();
        results.write(base + 4u, make_uint4(preserved->hit_type, preserved->inst,
                                            preserved->prim, preserve_count));
        callback_writes.write(base + 4u, first_primitive);
        distances.write(index, preserved->distance());
    };

    auto compile_begin = std::chrono::steady_clock::now();
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    auto compile_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - compile_begin)
                          .count();
    auto results = device.create_buffer<uint4>(options.rays * query_count);
    auto captures = device.create_buffer<uint4>(options.rays);
    auto callback_writes = device.create_buffer<uint>(options.rays * query_count);
    auto distances = device.create_buffer<float>(options.rays);
    luisa::vector<uint4> host_results(options.rays * query_count);
    luisa::vector<uint4> host_captures(options.rays);
    luisa::vector<uint> host_writes(options.rays * query_count);
    luisa::vector<float> host_distances(options.rays);
    auto dispatch = [&] {
        return shader(surface_scene, procedural_scene, preserve_scene, results, captures, callback_writes, distances)
            .dispatch(options.rays);
    };
    auto validate = [&] {
        stream << results.copy_to(luisa::span{host_results})
               << captures.copy_to(luisa::span{host_captures})
               << callback_writes.copy_to(luisa::span{host_writes})
               << distances.copy_to(luisa::span{host_distances}) << synchronize();
        bool correct = true;
        constexpr auto miss = static_cast<uint>(HitType::Miss);
        constexpr auto surface = static_cast<uint>(HitType::Surface);
        constexpr auto procedural_hit = static_cast<uint>(HitType::Procedural);
        for (auto i = 0u; i < options.rays; i++) {
            auto base = i * query_count;
            auto r0 = host_results[base];
            auto r1 = host_results[base + 1u];
            auto r2 = host_results[base + 2u];
            auto r3 = host_results[base + 3u];
            auto r4 = host_results[base + 4u];
            auto expected_capture = make_uint4(i * 3u + 17u, i * 5u + 27u,
                                               i * 3u + 10u, i * 5u + 16u);
            auto valid = r0.x == (options.benchmark ? surface : miss) && r0.w == 1u &&
                         (!options.benchmark || (r0.y == 0u && r0.z == 0u)) &&
                         r1.x == surface && r1.y == 0u && r1.z == 0u && r1.w == 1u &&
                         r2.x == (options.benchmark ? procedural_hit : miss) && r2.w == 1u &&
                         (!options.benchmark || (r2.y == 0u && r2.z == 0u)) &&
                         r3.x == procedural_hit && r3.y == 0u && r3.z == 0u && r3.w == 1u &&
                         r4.x == procedural_hit && r4.y == 0u && r4.w == (options.benchmark ? 1u : 2u) &&
                         r4.z == host_writes[base + 4u] && r4.z < 2u &&
                         std::abs(host_distances[i] - 1.25f) < 1.0e-6f &&
                         all(host_captures[i] == expected_capture) &&
                         host_writes[base] == (i ^ 0x1234u) &&
                         host_writes[base + 1u] == (i ^ 0x5678u) &&
                         host_writes[base + 2u] == (i ^ 0x9abcu) &&
                         host_writes[base + 3u] == (i ^ 0xdef0u);
            if (!valid) {
                LUISA_WARNING("Ray-query lifecycle mismatch at lane {}: surface reject/commit={}/{}, procedural reject/commit/preserve={}/{}/{}, captures={}, preserve t={}.",
                              i, r0, r1, r2, r3, r4, host_captures[i], host_distances[i]);
                correct = false;
                break;
            }
        }
        expect(correct) << "termination, previous committed hit, independent captures and callback writes";
        return correct;
    };

    // This mode changes only host submission and validation. In particular,
    // the kernel, query order, captures and compiler options remain identical.
    auto diagnostic_marker = [](luisa::string_view phase, uint32_t sample,
                                uint32_t dispatch_index, luisa::string_view state) {
        std::cerr << "{\"diagnostic\":\"cuda_ray_query_dispatch\",\"phase\":\"" << phase
                  << "\",\"sample\":" << sample << ",\"dispatch\":" << dispatch_index
                  << ",\"state\":\"" << state << "\"}\n"
                  << std::flush;
    };
    auto diagnosed_dispatch = [&](luisa::string_view phase, uint32_t sample,
                                  uint32_t dispatch_index) {
        diagnostic_marker(phase, sample, dispatch_index, "begin");
        stream << dispatch() << synchronize();
        diagnostic_marker(phase, sample, dispatch_index, "device_complete");
        if (!validate()) {
            diagnostic_marker(phase, sample, dispatch_index, "validation_failed");
            return false;
        }
        diagnostic_marker(phase, sample, dispatch_index, "validated");
        return true;
    };

    if (options.diagnose_dispatches) {
        diagnostic_marker("scene_build", 0u, 0u, "begin");
    }
    auto scene_setup = stream << vertices_buffer.copy_from(luisa::span{vertices})
                              << triangles_buffer.copy_from(luisa::span{triangles})
                              << boxes_buffer.copy_from(luisa::span{boxes})
                              << mesh.build() << procedural.build() << single_procedural.build()
                              << surface_scene.build() << procedural_scene.build() << preserve_scene.build();
    if (options.diagnose_dispatches) {
        std::move(scene_setup) << synchronize();
        diagnostic_marker("scene_build", 0u, 0u, "device_complete");
        if (!diagnosed_dispatch("initial", 0u, 0u)) { return false; }
    } else {
        std::move(scene_setup) << dispatch() << synchronize();
        if (!validate()) { return false; }
    }
    if (!options.benchmark) { return true; }

    if (options.diagnose_dispatches) {
        for (auto i = 0u; i < options.warmup; i++) {
            if (!diagnosed_dispatch("warmup", 0u, i)) { return false; }
        }
        for (auto sample = 0u; sample < options.samples; sample++) {
            for (auto i = 0u; i < options.dispatches; i++) {
                if (!diagnosed_dispatch("sample", sample, i)) { return false; }
            }
        }
        // Per-dispatch synchronization and downloads invalidate throughput
        // measurement. Diagnostic runs deliberately emit no benchmark record.
        diagnostic_marker("complete", 0u, 0u, "validated_no_performance_measurement");
        return true;
    }

    for (auto i = 0u; i < options.warmup; i++) { stream << dispatch(); }
    stream << synchronize();
    luisa::vector<double> samples;
    samples.reserve(options.samples);
    for (auto sample = 0u; sample < options.samples; sample++) {
        auto begin = std::chrono::steady_clock::now();
        for (auto i = 0u; i < options.dispatches; i++) { stream << dispatch(); }
        stream << synchronize();
        samples.emplace_back(std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count());
    }
    if (!validate()) { return false; }
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    auto middle = sorted.size() / 2u;
    auto median = sorted[middle];
    if (sorted.size() % 2u == 0u) { median = (median + sorted[middle - 1u]) * 0.5; }
    auto codegen_env = std::getenv("LUISA_EXPERIMENTAL_LLVM_CODEGEN");
    auto requested_codegen = codegen_env != nullptr && luisa::string_view{codegen_env} == "1" ? "llvm" : "ast";
    auto ray_queries = static_cast<double>(options.rays) * options.dispatches * query_count;
    std::cout << "{\"benchmark\":\"cuda_ray_query_commit_capture\",\"backend\":\"" << device.backend_name()
              << "\",\"requested_codegen\":\"" << requested_codegen
              << "\",\"compile_ms\":" << compile_ms
              << ",\"rays_per_dispatch\":" << options.rays
              << ",\"queries_per_ray\":" << query_count
              << ",\"dispatches_per_sample\":" << options.dispatches
              << ",\"warmup_dispatches\":" << options.warmup
              << ",\"median_seconds\":" << median
              << ",\"minimum_seconds\":" << sorted.front()
              << ",\"median_mqueries_per_second\":" << ray_queries / median * 1.0e-6
              << ",\"samples_seconds\":[";
    for (auto i = size_t{0u}; i < samples.size(); i++) {
        if (i != 0u) { std::cout << ','; }
        std::cout << samples[i];
    }
    std::cout << "]}\n";
    return true;
}

}// namespace

int main(int argc, char *argv[]) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        LUISA_INFO("Usage: {} <backend> [--benchmark] [--diagnose-dispatches] [--rays N] [--warmup N] [--dispatches N] [--samples N]", argv[0]);
        return 2;
    }
    auto dc = luisa::test::create_device(argc, argv);
    return run(dc.device, options) ? 0 : 1;
}
