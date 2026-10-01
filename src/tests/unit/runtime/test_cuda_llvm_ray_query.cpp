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
#include <cstring>
#include <fstream>
#include <iostream>

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/ast/function_builder.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

struct Options {
    bool benchmark{};
    bool diagnose_dispatches{};
    bool large_capture_stress{};
    bool surface_filter_only{};
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
        if (argument == "--large-capture-stress") {
            options.large_capture_stress = true;
            continue;
        }
        if (argument == "--surface-filter-only") {
            options.surface_filter_only = true;
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
    if ((options.benchmark || options.large_capture_stress) && !explicit_rays) { options.rays = 65536u; }
    if (options.large_capture_stress) {
        auto total_records = uint64_t{2u} * options.rays *
                             (uint64_t{options.warmup} + uint64_t{options.dispatches} * options.samples);
        if (total_records > 16777216u) { return false; }
    }
    return !(options.large_capture_stress && (options.benchmark || options.diagnose_dispatches)) &&
           !(options.surface_filter_only && (options.benchmark || options.diagnose_dispatches || options.large_capture_stress)) &&
           options.rays <= 1048576u && options.samples <= 101u &&
           options.warmup <= 1000u && options.dispatches <= 1000u;
}

[[nodiscard]] bool run_large_capture(Device &device, const Options &options) {
    constexpr auto element_count = 512u;
    constexpr auto element_mask = element_count - 1u;
    const std::array boxes{
        AABB{.packed_min = {-1.0f, -1.0f, -0.5f}, .packed_max = {1.0f, 1.0f, 0.5f}},
        AABB{.packed_min = {-1.0f, -1.0f, -0.5f}, .packed_max = {1.0f, 1.0f, 0.5f}},
        AABB{.packed_min = {-1.0f, -1.0f, -0.5f}, .packed_max = {1.0f, 1.0f, 0.5f}},
        AABB{.packed_min = {-1.0f, -1.0f, -0.5f}, .packed_max = {1.0f, 1.0f, 0.5f}}};
    auto stream = device.create_stream();
    auto boxes_buffer = device.create_buffer<AABB>(boxes.size());
    auto primitive = device.create_procedural_primitive(boxes_buffer);
    auto single_primitive = device.create_procedural_primitive(boxes_buffer.view(0u, 1u));
    auto scene = device.create_accel();
    auto single_scene = device.create_accel();
    scene.emplace_back(primitive);
    single_scene.emplace_back(single_primitive);

    Kernel1D kernel = [element_count, element_mask](AccelVar accel, AccelVar single_accel, UInt epoch,
                         BufferUInt4 results, BufferUInt4 checksums) noexcept {
        set_block_size(64u);
        auto lane = dispatch_x();
        auto base = (lane * 17u + epoch * 13u) & element_mask;
        auto seed = lane * 0x9e3779b9u ^ epoch * 0x85ebca6bu;
        // Runtime indexing in both the caller and outlined handlers keeps the
        // entire 2 KiB object addressable across all three traversal calls.
        // The following loop initializes every element before any query.
        // Skip the redundant 512-element default-initializer fold, which
        // exceeds Clang's default expression nesting limit.
        ArrayUInt<element_count> values{static_cast<const Expression *>(
            luisa::compute::detail::FunctionBuilder::current()->local(
                Type::of<std::array<uint, element_count>>()))};
        $for (j, element_count) {
            values[j] = (seed ^ j * 0xc2b2ae35u) & 0x3fffffffu;
        };
        UInt invalid = 0u;
        UInt seen = 0u;
        UInt rejected_count = 0u;
        UInt accepted_count = 0u;
        UInt final_count = 0u;
        UInt selected = ~0u;
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
        auto rejected = accel.traverse(ray, {})
                            .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                auto hit = candidate.hit();
                                rejected_count += 1u;
                                $if ((hit->inst == 0u) & (hit->prim < 4u)) {
                                    seen |= 1u << hit->prim;
                                    auto slot = (base + hit->prim * 37u) & element_mask;
                                    values[slot] |= 0x80000000u;
                                }
                                $else { invalid |= 1u; };
                                // Reject every candidate: all four primitives
                                // must be observed, in any order. OR is also
                                // correct if the BVH repeats a candidate.
                            })
                            .trace();
        auto accepted = accel.traverse(ray, {})
                            .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                auto hit = candidate.hit();
                                accepted_count += 1u;
                                selected = hit->prim;
                                $if ((hit->inst == 0u) & (hit->prim < 4u)) {
                                    auto slot = (base + 160u + hit->prim * 37u) & element_mask;
                                    values[slot] |= 0x40000000u;
                                }
                                $else { invalid |= 2u; };
                                candidate.commit(1.25f);
                                candidate.terminate();
                            })
                            .trace();
        auto final_hit = single_accel.traverse(ray, {})
                             .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                 auto hit = candidate.hit();
                                 final_count += 1u;
                                 $if ((hit->inst == 0u) & (hit->prim == 0u) & (selected < 4u)) {
                                     auto slot = (base + 320u + selected * 7u) & element_mask;
                                     values[slot] ^= 0x13579bdu;
                                 }
                                 $else { invalid |= 4u; };
                                 candidate.commit(1.0f);
                                 candidate.terminate();
                             })
                             .trace();
        $if ((rejected->hit_type != static_cast<uint>(HitType::Miss)) |
             (accepted->hit_type != static_cast<uint>(HitType::Procedural)) |
             (accepted->inst != 0u) | (accepted->prim != selected) |
             (accepted->distance() != 1.25f) |
             (final_hit->hit_type != static_cast<uint>(HitType::Procedural)) |
             (final_hit->inst != 0u) | (final_hit->prim != 0u) |
             (final_hit->distance() != 1.0f) |
             (accepted_count != 1u) | (final_count != 1u)) {
            invalid |= 8u;
        };
        UInt mismatches = 0u;
        UInt sum = 0u;
        UInt weighted_sum = 0u;
        UInt xor_sum = 0u;
        $for (j, element_count) {
            UInt expected = (seed ^ j * 0xc2b2ae35u) & 0x3fffffffu;
            auto offset = (j - base) & element_mask;
            $if ((offset == 0u) | (offset == 37u) | (offset == 74u) | (offset == 111u)) {
                expected |= 0x80000000u;
            };
            $if (offset == 160u + selected * 37u) { expected |= 0x40000000u; };
            $if (offset == 320u + selected * 7u) { expected ^= 0x13579bdu; };
            UInt actual = values[j];
            mismatches += cast<uint>(actual != expected);
            sum += actual;
            weighted_sum += actual * (j + 1u);
            xor_sum ^= actual;
        };
        results.write(lane, make_uint4(invalid | (mismatches << 8u), seen, rejected_count, selected));
        checksums.write(lane, make_uint4(sum, weighted_sum, xor_sum, epoch));
    };
    auto compile_begin = std::chrono::steady_clock::now();
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    auto compile_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - compile_begin)
                          .count();
    auto results = device.create_buffer<uint4>(options.rays);
    auto checksums = device.create_buffer<uint4>(options.rays);
    // Copy every dispatch before the next overwrites the device output, while
    // retaining asynchronous batches. Cap staging memory independently of the
    // requested stress count and validate every saved result after each batch.
    auto batch_capacity = std::min(std::max(options.warmup, options.dispatches),
                                   std::max(1u, 1048576u / options.rays));
    luisa::vector<uint4> host_results(static_cast<size_t>(options.rays) * batch_capacity);
    luisa::vector<uint4> host_checksums(host_results.size());
    stream << boxes_buffer.copy_from(luisa::span{boxes})
           << primitive.build() << single_primitive.build()
           << scene.build() << single_scene.build() << synchronize();
    auto epoch = 0u;
    auto run_dispatches = [&](uint32_t count) {
        for (auto completed = 0u; completed < count;) {
            auto batch = std::min(batch_capacity, count - completed);
            auto first_epoch = epoch;
            for (auto i = 0u; i < batch; i++) {
                auto offset = static_cast<size_t>(i) * options.rays;
                stream << shader(scene, single_scene, epoch++, results, checksums).dispatch(options.rays)
                       << results.copy_to(luisa::span{host_results}.subspan(offset, options.rays))
                       << checksums.copy_to(luisa::span{host_checksums}.subspan(offset, options.rays));
            }
            stream << synchronize();
            bool correct = true;
            for (auto i = 0u; i < batch && correct; i++) {
                auto expected_epoch = first_epoch + i;
                for (auto lane = 0u; lane < options.rays; lane++) {
                    auto index = static_cast<size_t>(i) * options.rays + lane;
                    auto result = host_results[index];
                    auto observed = host_checksums[index];
                    auto base = (lane * 17u + expected_epoch * 13u) & element_mask;
                    auto seed = lane * 0x9e3779b9u ^ expected_epoch * 0x85ebca6bu;
                    auto expected = make_uint4(0u, 0u, 0u, expected_epoch);
                    for (auto j = 0u; j < element_count; j++) {
                        auto value = (seed ^ j * 0xc2b2ae35u) & 0x3fffffffu;
                        auto offset = (j - base) & element_mask;
                        if (offset == 0u || offset == 37u || offset == 74u || offset == 111u) { value |= 0x80000000u; }
                        if (offset == 160u + result.w * 37u) { value |= 0x40000000u; }
                        if (offset == 320u + result.w * 7u) { value ^= 0x13579bdu; }
                        expected.x += value;
                        expected.y += value * (j + 1u);
                        expected.z ^= value;
                    }
                    if (result.x != 0u || result.y != 15u || result.z < 4u || result.w >= 4u || !all(observed == expected)) {
                        LUISA_WARNING("Large ray-query capture mismatch: epoch={} lane={} result={} checksums={} expected={}.",
                                      expected_epoch, lane, result, observed, expected);
                        correct = false;
                        break;
                    }
                }
            }
            expect(correct) << "2 KiB captured array: every element, candidate identity, termination and every dispatch";
            if (!correct) { return false; }
            completed += batch;
        }
        return true;
    };
    if (!run_dispatches(options.warmup)) { return false; }
    for (auto sample = 0u; sample < options.samples; sample++) {
        if (!run_dispatches(options.dispatches)) { return false; }
    }
    std::cout << "{\"regression\":\"cuda_ray_query_large_capture\",\"capture_bytes\":2048"
              << ",\"rays_per_dispatch\":" << options.rays << ",\"validated_dispatches\":" << epoch
              << ",\"compile_ms\":" << compile_ms << "}\n";
    return true;
}

[[nodiscard]] bool run_hardware_payload_boundary(Device &device, Stream &stream,
                                                 const Accel &surfaces, const Accel &procedurals) {
    constexpr auto ray_count = 68u;
    for (auto snapshot_count : {0u, 29u, 30u}) {
        Kernel1D kernel = [snapshot_count](AccelVar surface_accel, AccelVar procedural_accel,
                                           BufferUInt inputs, BufferUInt outputs, BufferUInt4 hits) noexcept {
            set_block_size(64u);
            auto lane = dispatch_x();
            // These are separate DSL locals, not one captured array pointer.
            // Materialize every resource read before constructing either query;
            // the callback must receive the values of these memory snapshots.
            luisa::vector<UInt> snapshots;
            snapshots.reserve(snapshot_count);
            for (auto i = 0u; i < snapshot_count; i++) {
                snapshots.emplace_back(def(inputs.read(lane * snapshot_count + i)));
            }
            ArrayUInt<30u> surface_values;
            ArrayUInt<30u> procedural_values;
            auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                                make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
            auto surface_hit = surface_accel.traverse(ray, {})
                                   .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                       for (auto i = 0u; i < snapshot_count; i++) {
                                           surface_values[i] = snapshots[i] ^ (0x13579bdfu + i * 17u);
                                       }
                                       candidate.commit();
                                   })
                                   .trace();
            auto procedural_hit = procedural_accel.traverse(ray, {})
                                      .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                          for (auto i = 0u; i < snapshot_count; i++) {
                                              procedural_values[i] = snapshots[i] ^ (0x2468ace0u + i * 31u);
                                          }
                                          candidate.commit(1.25f);
                                      })
                                      .trace();
            hits.write(lane * 2u, make_uint4(surface_hit->hit_type, surface_hit->inst,
                                             surface_hit->prim, surface_hit->distance().as<uint>()));
            hits.write(lane * 2u + 1u, make_uint4(procedural_hit->hit_type, procedural_hit->inst,
                                                  procedural_hit->prim, procedural_hit->distance().as<uint>()));
            for (auto i = 0u; i < snapshot_count; i++) {
                outputs.write(lane * (2u * snapshot_count) + i, surface_values[i]);
                outputs.write(lane * (2u * snapshot_count) + snapshot_count + i, procedural_values[i]);
            }
        };
        // Both handlers qualify for hardware results. With no snapshots there
        // are no captures; the module still reserves the public two-word floor.
        // Otherwise each handler captures N independent load snapshots and one
        // array reference: 29 + 2 words fit the direct budget, 30 + 2 do not.
        // Verify the serialized capacity as well as every value below; correlate
        // the AST hash with .opt.rq.xir / LLVM dumps to inspect actual captures.
        LUISA_INFO("Hardware payload boundary: snapshots={}, expected capture words={}, AST hash={:016x}.",
                   snapshot_count, snapshot_count == 0u ? 0u : snapshot_count + 2u, kernel.function()->function().hash());
        auto llvm_env = std::getenv("LUISA_EXPERIMENTAL_LLVM_CODEGEN");
        auto ir_env = std::getenv("LUISA_CUDA_LLVM_OPTIX_IR");
        auto llvm_codegen = llvm_env != nullptr && std::strcmp(llvm_env, "1") == 0;
        auto optix_ir = llvm_codegen && ir_env != nullptr && std::strcmp(ir_env, "1") == 0;
        auto package_stem = luisa::format("test_cuda_llvm_ray_query_hardware_payload_{}_{}", snapshot_count,
                                          std::chrono::steady_clock::now().time_since_epoch().count());
        auto package_leaf = luisa::format("{}{}", package_stem, optix_ir ? ".optixir" : ".ptx");
        auto package_path = luisa::filesystem::absolute(package_leaf.c_str());
        auto package_name = luisa::to_string(package_path);
        auto extensionless_name = luisa::to_string(luisa::filesystem::absolute(package_stem.c_str()));
        auto metadata_path = luisa::filesystem::path{luisa::format("{}.metadata", package_name).c_str()};
        // Explicit shader names use the bytecode store even with cache disabled.
        // Never reuse or remove an artifact that predates this test invocation.
        auto owned_paths_available = !luisa::filesystem::exists(package_path) &&
                                     !luisa::filesystem::exists(metadata_path);
        for (auto extension : {".ptx", ".ptx.metadata", ".optixir", ".optixir.metadata"}) {
            auto candidate = luisa::filesystem::path{luisa::format("{}{}", extensionless_name, extension).c_str()};
            owned_paths_available = owned_paths_available && !luisa::filesystem::exists(candidate);
        }
        expect(owned_paths_available) << "unique payload AOT artifact paths";
        if (!owned_paths_available) { return false; }
        struct ArtifactCleanup {
            const luisa::filesystem::path &package;
            const luisa::filesystem::path &metadata;
            ~ArtifactCleanup() noexcept {
                for (auto path : {&package, &metadata}) {
                    std::error_code error;
                    luisa::filesystem::remove(*path, error);
                    if (error) {
                        LUISA_WARNING("Failed to remove payload AOT artifact '{}': {}.",
                                      luisa::to_string(*path), error.message());
                    }
                }
            }
        } cleanup{package_path, metadata_path};
        // Produce the owned package before any consumer for this kernel exists.
        // compile_only must serialize both files without returning a live shader.
        LUISA_INFO("Hardware payload compile-only producer begins: snapshots={}.", snapshot_count);
        {
            auto producer = device.compile(kernel, ShaderOption{
                                                       .enable_cache = false,
                                                       .compile_only = true,
                                                       .name = package_name});
            auto produced = !producer && producer.compile_ok() &&
                            luisa::filesystem::is_regular_file(package_path) &&
                            luisa::filesystem::is_regular_file(metadata_path) &&
                            luisa::filesystem::file_size(package_path) != 0u &&
                            luisa::filesystem::file_size(metadata_path) != 0u;
            expect(produced) << "compile-only producer writes a package and sidecar without a live shader";
            if (!produced) { return false; }
        }
        LUISA_INFO("Hardware payload compile-only producer ends: snapshots={}.", snapshot_count);
        auto expected_payload_count = !llvm_codegen ? 2u : snapshot_count == 0u ? 2u :
                                                       snapshot_count == 29u    ? 32u :
                                                                                  3u;
        // Observe the serialized public artifact rather than inferring the
        // pipeline ABI from the number of DSL variables alone.
        auto read_payload_count = [&]() noexcept {
            std::ifstream metadata{metadata_path};
            luisa::string token;
            auto value = 0u;
            while (metadata >> token) {
                if (token == "RAY_QUERY_PAYLOAD_COUNT") {
                    metadata >> value;
                    return metadata ? value : 0u;
                }
            }
            return 0u;
        };
        auto serialized_payload_count = read_payload_count();
        LUISA_INFO("Hardware payload artifact: snapshots={}, payload words={}, expected={}.",
                   snapshot_count, serialized_payload_count, expected_payload_count);
        auto correct_payload_count = serialized_payload_count == expected_payload_count;
        expect(correct_payload_count) << "serialized payload count matches the actual capture boundary";
        if (!correct_payload_count) { return false; }
        // Give only our owned files a distinct older timestamp. A named warm
        // compile must read them without rewriting; no timing heuristic is used.
        auto old_timestamp = luisa::filesystem::file_time_type::clock::now() - std::chrono::hours{24};
        luisa::filesystem::last_write_time(package_path, old_timestamp);
        luisa::filesystem::last_write_time(metadata_path, old_timestamp);
        auto package_timestamp = luisa::filesystem::last_write_time(package_path);
        auto metadata_timestamp = luisa::filesystem::last_write_time(metadata_path);
        auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .name = package_name});
        auto warm_shader = device.compile(kernel, ShaderOption{.enable_cache = false, .name = package_name});
        auto reused_artifact = luisa::filesystem::last_write_time(package_path) == package_timestamp &&
                               luisa::filesystem::last_write_time(metadata_path) == metadata_timestamp &&
                               read_payload_count() == expected_payload_count;
        expect(reused_artifact) << "warm compile must adopt the cached payload count without regenerating artifacts";
        if (!reused_artifact) { return false; }
        auto loaded_shader = device.load_shader<1, Accel, Accel, Buffer<uint>, Buffer<uint>, Buffer<uint4>>(package_name);
        expect(static_cast<bool>(loaded_shader)) << "payload AOT shader loads its serialized OptiX ABI";
        if (!loaded_shader) { return false; }
        auto extensionless_shader = device.load_shader<1, Accel, Accel, Buffer<uint>, Buffer<uint>, Buffer<uint4>>(extensionless_name);
        expect(static_cast<bool>(extensionless_shader)) << "extensionless AOT load discovers the actual PTX or OptiX IR package";
        if (!extensionless_shader) { return false; }
        auto inputs = device.create_buffer<uint>(std::max(1u, ray_count * snapshot_count));
        auto outputs = device.create_buffer<uint>(std::max(1u, ray_count * snapshot_count * 2u));
        auto hits = device.create_buffer<uint4>(ray_count * 2u);
        luisa::vector<uint> host_inputs(inputs.size());
        luisa::vector<uint> host_outputs(outputs.size());
        luisa::vector<uint4> host_hits(hits.size());
        for (auto epoch = 0u; epoch < 4u; epoch++) {
            for (auto lane = 0u; lane < ray_count; lane++) {
                for (auto i = 0u; i < snapshot_count; i++) {
                    host_inputs[lane * snapshot_count + i] =
                        lane * 0x9e3779b9u ^ i * 0x85ebca6bu ^ (epoch + 1u) * 0xc2b2ae35u;
                }
            }
            const auto &active_shader = epoch == 0u ? shader : epoch == 1u ? warm_shader :
                                                           epoch == 2u     ? loaded_shader :
                                                                             extensionless_shader;
            stream << inputs.copy_from(luisa::span{host_inputs})
                   << active_shader(surfaces, procedurals, inputs, outputs, hits).dispatch(ray_count);
            if (snapshot_count != 0u) { stream << outputs.copy_to(luisa::span{host_outputs}); }
            stream << hits.copy_to(luisa::span{host_hits}) << synchronize();
            bool correct = true;
            for (auto lane = 0u; lane < ray_count && correct; lane++) {
                auto expected_surface = make_uint4(static_cast<uint>(HitType::Surface), 0u, 0u, 0x3f800000u);
                auto expected_procedural = make_uint4(static_cast<uint>(HitType::Procedural), 0u, 0u, 0x3fa00000u);
                if (!all(host_hits[lane * 2u] == expected_surface) ||
                    !all(host_hits[lane * 2u + 1u] == expected_procedural)) {
                    LUISA_WARNING("Hardware payload capture hit mismatch: snapshots={} epoch={} lane={} surface={} procedural={}.",
                                  snapshot_count, epoch, lane, host_hits[lane * 2u], host_hits[lane * 2u + 1u]);
                    correct = false;
                    break;
                }
                for (auto i = 0u; i < snapshot_count; i++) {
                    auto value = host_inputs[lane * snapshot_count + i];
                    auto expected_surface_value = value ^ (0x13579bdfu + i * 17u);
                    auto expected_procedural_value = value ^ (0x2468ace0u + i * 31u);
                    auto actual_surface = host_outputs[lane * (2u * snapshot_count) + i];
                    auto actual_procedural = host_outputs[lane * (2u * snapshot_count) + snapshot_count + i];
                    if (actual_surface != expected_surface_value || actual_procedural != expected_procedural_value) {
                        LUISA_WARNING("Hardware payload capture value mismatch: snapshots={} epoch={} lane={} word={} surface={}/{} procedural={}/{}.",
                                      snapshot_count, epoch, lane, i, actual_surface, expected_surface_value,
                                      actual_procedural, expected_procedural_value);
                        correct = false;
                        break;
                    }
                }
            }
            expect(correct) << "every hardware-result capture reaches AH and IS through cold/warm JIT and AOT";
            if (!correct) { return false; }
        }
    }
    return true;
}

[[nodiscard]] bool run_payload_capture_boundary(Device &device, Stream &stream,
                                                const Accel &surfaces, const Accel &procedurals) {
    constexpr auto ray_count = 257u;
    for (auto snapshot_count : {0u, 27u, 30u}) {
        Kernel1D kernel = [snapshot_count](AccelVar surface_accel, AccelVar procedural_accel,
                                           BufferUInt inputs, BufferUInt outputs, BufferUInt4 hits) noexcept {
            set_block_size(64u);
            auto lane = dispatch_x();
            // These are separate DSL locals, not one captured array pointer.
            // Materialize every resource read before constructing either query;
            // the callback must receive the values of these memory snapshots.
            luisa::vector<UInt> snapshots;
            snapshots.reserve(snapshot_count);
            for (auto i = 0u; i < snapshot_count; i++) {
                snapshots.emplace_back(def(inputs.read(lane * snapshot_count + i)));
            }
            ArrayUInt<30u> surface_values;
            ArrayUInt<30u> procedural_values;
            ArrayUInt<30u> filter_values;
            auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                                make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
            auto surface_hit = surface_accel.traverse(ray, {})
                                   .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                       for (auto i = 0u; i < snapshot_count; i++) {
                                           surface_values[i] = snapshots[i] ^ (0x13579bdfu + i * 17u);
                                       }
                                       candidate.commit();
                                       candidate.terminate();
                                   })
                                   .trace();
            auto procedural_hit = procedural_accel.traverse(ray, {})
                                      .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                          for (auto i = 0u; i < snapshot_count; i++) {
                                              procedural_values[i] = snapshots[i] ^ (0x2468ace0u + i * 31u);
                                          }
                                          candidate.commit(1.25f);
                                          candidate.terminate();
                                      })
                                      .trace();
            // One module deliberately mixes full handlers with a surface-only
            // filter. Its independent memory snapshots also exercise both
            // direct payload and context fallback with the same output oracle.
            auto filter_hit = surface_accel.traverse(ray, {})
                                  .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                      for (auto i = 0u; i < snapshot_count; i++) {
                                          filter_values[i] = snapshots[i] ^ (0xa5a55a5au + i * 13u);
                                      }
                                      candidate.commit();
                                  })
                                  .trace();
            hits.write(lane * 3u, make_uint4(surface_hit->hit_type, surface_hit->inst,
                                             surface_hit->prim, surface_hit->distance().as<uint>()));
            hits.write(lane * 3u + 1u, make_uint4(procedural_hit->hit_type, procedural_hit->inst,
                                                  procedural_hit->prim, procedural_hit->distance().as<uint>()));
            hits.write(lane * 3u + 2u, make_uint4(filter_hit->hit_type, filter_hit->inst,
                                                filter_hit->prim, filter_hit->distance().as<uint>()));
            for (auto i = 0u; i < snapshot_count; i++) {
                outputs.write(lane * (3u * snapshot_count) + i, surface_values[i]);
                outputs.write(lane * (3u * snapshot_count) + snapshot_count + i, procedural_values[i]);
                outputs.write(lane * (3u * snapshot_count) + 2u * snapshot_count + i, filter_values[i]);
            }
        };
        // No snapshots means no output-array reference is captured: only the
        // three protocol words remain. Otherwise N scalars plus one 64-bit
        // reference use direct payloads up to 29 capture words, then fallback.
        // Correlate this hash with .opt.rq.xir and LUISA_DUMP_LLVM_IR to verify
        // the actual capture types and direct/fallback codegen selection.
        LUISA_INFO("Ray-query payload boundary fixture: snapshots={}, expected capture words={}, AST hash={:016x}.",
                   snapshot_count, snapshot_count == 0u ? 0u : snapshot_count + 2u, kernel.function()->function().hash());
        auto llvm_env = std::getenv("LUISA_EXPERIMENTAL_LLVM_CODEGEN");
        auto ir_env = std::getenv("LUISA_CUDA_LLVM_OPTIX_IR");
        auto llvm_codegen = llvm_env != nullptr && std::strcmp(llvm_env, "1") == 0;
        auto optix_ir = llvm_codegen && ir_env != nullptr && std::strcmp(ir_env, "1") == 0;
        auto package_stem = luisa::format("test_cuda_llvm_ray_query_payload_{}_{}", snapshot_count,
                                          std::chrono::steady_clock::now().time_since_epoch().count());
        auto package_leaf = luisa::format("{}{}", package_stem, optix_ir ? ".optixir" : ".ptx");
        auto package_path = luisa::filesystem::absolute(package_leaf.c_str());
        auto package_name = luisa::to_string(package_path);
        auto extensionless_name = luisa::to_string(luisa::filesystem::absolute(package_stem.c_str()));
        auto metadata_path = luisa::filesystem::path{luisa::format("{}.metadata", package_name).c_str()};
        // Explicit shader names use the bytecode store even with cache disabled.
        // Never reuse or remove an artifact that predates this test invocation.
        auto owned_paths_available = !luisa::filesystem::exists(package_path) &&
                                     !luisa::filesystem::exists(metadata_path);
        for (auto extension : {".ptx", ".ptx.metadata", ".optixir", ".optixir.metadata"}) {
            auto candidate = luisa::filesystem::path{luisa::format("{}{}", extensionless_name, extension).c_str()};
            owned_paths_available = owned_paths_available && !luisa::filesystem::exists(candidate);
        }
        expect(owned_paths_available) << "unique payload AOT artifact paths";
        if (!owned_paths_available) { return false; }
        struct ArtifactCleanup {
            const luisa::filesystem::path &package;
            const luisa::filesystem::path &metadata;
            ~ArtifactCleanup() noexcept {
                for (auto path : {&package, &metadata}) {
                    std::error_code error;
                    luisa::filesystem::remove(*path, error);
                    if (error) {
                        LUISA_WARNING("Failed to remove payload AOT artifact '{}': {}.",
                                      luisa::to_string(*path), error.message());
                    }
                }
            }
        } cleanup{package_path, metadata_path};
        // Produce the owned package before any consumer for this kernel exists.
        // compile_only must serialize both files without returning a live shader.
        LUISA_INFO("Ray-query compile-only producer begins: snapshots={}.", snapshot_count);
        {
            auto producer = device.compile(kernel, ShaderOption{
                                                       .enable_cache = false,
                                                       .compile_only = true,
                                                       .name = package_name});
            auto produced = !producer && producer.compile_ok() &&
                            luisa::filesystem::is_regular_file(package_path) &&
                            luisa::filesystem::is_regular_file(metadata_path) &&
                            luisa::filesystem::file_size(package_path) != 0u &&
                            luisa::filesystem::file_size(metadata_path) != 0u;
            expect(produced) << "compile-only producer writes a package and sidecar without a live shader";
            if (!produced) { return false; }
        }
        LUISA_INFO("Ray-query compile-only producer ends: snapshots={}.", snapshot_count);
        auto expected_payload_count = !llvm_codegen ? 2u : snapshot_count == 0u ? 3u : snapshot_count == 27u ? 32u : 5u;
        // Observe the serialized public artifact rather than inferring the
        // pipeline ABI from the number of DSL variables alone.
        auto read_payload_count = [&]() noexcept {
            std::ifstream metadata{metadata_path};
            luisa::string token;
            auto value = 0u;
            while (metadata >> token) {
                if (token == "RAY_QUERY_PAYLOAD_COUNT") {
                    metadata >> value;
                    return metadata ? value : 0u;
                }
            }
            return 0u;
        };
        auto serialized_payload_count = read_payload_count();
        LUISA_INFO("Ray-query payload artifact: snapshots={}, payload words={}, expected={}.",
                   snapshot_count, serialized_payload_count, expected_payload_count);
        auto correct_payload_count = serialized_payload_count == expected_payload_count;
        expect(correct_payload_count) << "serialized payload count matches the actual capture boundary";
        if (!correct_payload_count) { return false; }
        // Give only our owned files a distinct older timestamp. A named warm
        // compile must read them without rewriting; no timing heuristic is used.
        auto old_timestamp = luisa::filesystem::file_time_type::clock::now() - std::chrono::hours{24};
        luisa::filesystem::last_write_time(package_path, old_timestamp);
        luisa::filesystem::last_write_time(metadata_path, old_timestamp);
        auto package_timestamp = luisa::filesystem::last_write_time(package_path);
        auto metadata_timestamp = luisa::filesystem::last_write_time(metadata_path);
        auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .name = package_name});
        auto warm_shader = device.compile(kernel, ShaderOption{.enable_cache = false, .name = package_name});
        auto reused_artifact = luisa::filesystem::last_write_time(package_path) == package_timestamp &&
                               luisa::filesystem::last_write_time(metadata_path) == metadata_timestamp &&
                               read_payload_count() == expected_payload_count;
        expect(reused_artifact) << "warm compile must adopt the cached payload count without regenerating artifacts";
        if (!reused_artifact) { return false; }
        auto loaded_shader = device.load_shader<1, Accel, Accel, Buffer<uint>, Buffer<uint>, Buffer<uint4>>(package_name);
        expect(static_cast<bool>(loaded_shader)) << "payload AOT shader loads its serialized OptiX ABI";
        if (!loaded_shader) { return false; }
        auto extensionless_shader = device.load_shader<1, Accel, Accel, Buffer<uint>, Buffer<uint>, Buffer<uint4>>(extensionless_name);
        expect(static_cast<bool>(extensionless_shader)) << "extensionless AOT load discovers the actual PTX or OptiX IR package";
        if (!extensionless_shader) { return false; }
        auto inputs = device.create_buffer<uint>(std::max(1u, ray_count * snapshot_count));
        auto outputs = device.create_buffer<uint>(std::max(1u, ray_count * snapshot_count * 3u));
        auto hits = device.create_buffer<uint4>(ray_count * 3u);
        luisa::vector<uint> host_inputs(inputs.size());
        luisa::vector<uint> host_outputs(outputs.size());
        luisa::vector<uint4> host_hits(hits.size());
        for (auto epoch = 0u; epoch < 4u; epoch++) {
            for (auto lane = 0u; lane < ray_count; lane++) {
                for (auto i = 0u; i < snapshot_count; i++) {
                    host_inputs[lane * snapshot_count + i] =
                        lane * 0x9e3779b9u ^ i * 0x85ebca6bu ^ (epoch + 1u) * 0xc2b2ae35u;
                }
            }
            const auto &active_shader = epoch == 0u ? shader : epoch == 1u ? warm_shader :
                                            epoch == 2u ? loaded_shader : extensionless_shader;
            stream << inputs.copy_from(luisa::span{host_inputs})
                   << active_shader(surfaces, procedurals, inputs, outputs, hits).dispatch(ray_count);
            if (snapshot_count != 0u) { stream << outputs.copy_to(luisa::span{host_outputs}); }
            stream << hits.copy_to(luisa::span{host_hits}) << synchronize();
            bool correct = true;
            for (auto lane = 0u; lane < ray_count && correct; lane++) {
                auto expected_surface = make_uint4(static_cast<uint>(HitType::Surface), 0u, 0u, 0x3f800000u);
                auto expected_procedural = make_uint4(static_cast<uint>(HitType::Procedural), 0u, 0u, 0x3fa00000u);
                if (!all(host_hits[lane * 3u] == expected_surface) ||
                    !all(host_hits[lane * 3u + 1u] == expected_procedural) ||
                    !all(host_hits[lane * 3u + 2u] == expected_surface)) {
                    LUISA_WARNING("Payload capture hit mismatch: snapshots={} epoch={} lane={} surface={} procedural={} filter={}.",
                                  snapshot_count, epoch, lane, host_hits[lane * 3u], host_hits[lane * 3u + 1u], host_hits[lane * 3u + 2u]);
                    correct = false;
                    break;
                }
                for (auto i = 0u; i < snapshot_count; i++) {
                    auto value = host_inputs[lane * snapshot_count + i];
                    auto expected_surface_value = value ^ (0x13579bdfu + i * 17u);
                    auto expected_procedural_value = value ^ (0x2468ace0u + i * 31u);
                    auto actual_surface = host_outputs[lane * (3u * snapshot_count) + i];
                    auto actual_procedural = host_outputs[lane * (3u * snapshot_count) + snapshot_count + i];
                    auto expected_filter_value = value ^ (0xa5a55a5au + i * 13u);
                    auto actual_filter = host_outputs[lane * (3u * snapshot_count) + 2u * snapshot_count + i];
                    if (actual_surface != expected_surface_value || actual_procedural != expected_procedural_value ||
                        actual_filter != expected_filter_value) {
                        LUISA_WARNING("Payload capture value mismatch: snapshots={} epoch={} lane={} word={} surface={}/{} procedural={}/{} filter={}/{}.",
                                      snapshot_count, epoch, lane, i, actual_surface, expected_surface_value,
                                      actual_procedural, expected_procedural_value, actual_filter, expected_filter_value);
                        correct = false;
                        break;
                    }
                }
            }
            expect(correct) << "every captured snapshot reaches mixed full/filter handlers through cold/warm JIT and AOT";
            if (!correct) { return false; }
        }
    }
    return true;
}

[[nodiscard]] bool run_resource_capture_queries(Device &device, Stream &stream,
                                                const Accel &surface_scene, const Accel &procedural_scene) {
    constexpr auto ray_count = 68u;
    for (auto snapshot_count : {0u, 30u}) {
        Callable query = [snapshot_count](BufferUInt input, UInt salt, AccelVar accel,
                                          BufferUInt common, UInt slot, BufferUInt output, UInt lane) noexcept {
            // Keep ordinary callable forwarding visible to XIR origin analysis.
            // input has two kernel roots; common, output and accel each have one.
            luisa::compute::detail::FunctionBuilder::current()->mark_noinline();
            luisa::vector<UInt> snapshots;
            snapshots.reserve(snapshot_count);
            for (auto i = 0u; i < snapshot_count; i++) {
                snapshots.emplace_back(def(input.read((lane + i) % input.size())));
            }
            UInt stamp = 0u;
            auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                                make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
            auto write_resources = [&]() noexcept {
                auto input_size = input.size();
                auto common_size = common.size();
                auto output_size = output.size();
                UInt checksum = 0u;
                for (auto i = 0u; i < snapshot_count; i++) { checksum ^= snapshots[i] * (i * 2u + 1u); }
                $if (input_size != 0u & common_size != 0u & lane < output_size / 2u) {
                    auto value = (input.read(lane % input_size) ^
                                  common.read(lane % common_size) ^ salt ^ checksum) +
                                 input_size * 3u + common_size * 5u + output_size * 7u;
                    output.write(slot * (output_size / 2u) + lane, value);
                    stamp = input_size ^ (common_size << 8u) ^ (output_size << 16u) ^ salt ^ checksum;
                };
            };
            auto hit = accel.traverse(ray, {})
                           .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                               write_resources();
                               candidate.commit();
                           })
                           .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                               write_resources();
                               candidate.commit(1.0f);
                           })
                           .trace();
            return make_uint4(hit->hit_type, hit->inst, hit->prim, stamp);
        };
        Kernel1D kernel = [&query](BufferUInt input_a, UInt salt_a, AccelVar accel,
                                   BufferUInt common, UInt salt_b, BufferUInt input_b,
                                   BufferUInt4 hits, UInt marker, BufferUInt output) noexcept {
            auto lane = dispatch_x();
            hits.write(lane * 2u, query(input_a, salt_a ^ marker, accel, common, 0u, output, lane));
            hits.write(lane * 2u + 1u, query(input_b, salt_b ^ marker, accel, common, 1u, output, lane));
        };
        // Thirty independently loaded snapshots exceed the 29-word capture limit
        // even after resource rematerialization. Correlate this hash with XIR/PTX
        // dumps to verify both the compact payload and context fallback mappings.
        LUISA_INFO("Ray-query resource capture fixture: snapshots={}, AST hash={:016x}.", snapshot_count, kernel.function()->function().hash());
        auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
        constexpr auto a_offset = 5u, a_size = 97u, b_offset = 11u, b_size = 113u;
        constexpr auto common_offset = 7u, common_size = 43u, output_offset = 9u;
        constexpr auto output_size = ray_count * 2u;
        constexpr auto sentinel = 0xdeadbeefu, salt_a = 0x13579bdfu, salt_b = 0x2468ace0u, marker = 0x5a5aa5a5u;
        luisa::vector<uint> host_a(a_offset + a_size + 3u);
        luisa::vector<uint> host_b(b_offset + b_size + 5u);
        luisa::vector<uint> host_common(common_offset + common_size + 7u);
        luisa::vector<uint> host_output(output_offset + output_size + 11u, sentinel);
        luisa::vector<uint4> host_hits(output_size);
        for (auto i = 0u; i < host_a.size(); i++) { host_a[i] = i * 17u + 101u; }
        for (auto i = 0u; i < host_b.size(); i++) { host_b[i] = i * 31u + 503u; }
        for (auto i = 0u; i < host_common.size(); i++) { host_common[i] = i * 43u + 1009u; }
        auto input_a = device.create_buffer<uint>(host_a.size());
        auto input_b = device.create_buffer<uint>(host_b.size());
        auto common = device.create_buffer<uint>(host_common.size());
        auto output = device.create_buffer<uint>(host_output.size());
        auto hits = device.create_buffer<uint4>(host_hits.size());
        for (auto use_procedural : {false, true}) {
            const auto &scene = use_procedural ? procedural_scene : surface_scene;
            std::fill(host_output.begin(), host_output.end(), sentinel);
            stream << input_a.copy_from(luisa::span{host_a})
                   << input_b.copy_from(luisa::span{host_b})
                   << common.copy_from(luisa::span{host_common})
                   << output.copy_from(luisa::span{host_output})
                   << shader(input_a.view(a_offset, a_size), salt_a, scene,
                             common.view(common_offset, common_size), salt_b, input_b.view(b_offset, b_size),
                             hits, marker, output.view(output_offset, output_size))
                          .dispatch(ray_count)
                   << output.copy_to(luisa::span{host_output})
                   << hits.copy_to(luisa::span{host_hits}) << synchronize();
            auto correct = true;
            for (auto slot = 0u; slot < 2u; slot++) {
                auto size = slot == 0u ? a_size : b_size;
                auto offset = slot == 0u ? a_offset : b_offset;
                const auto &input = slot == 0u ? host_a : host_b;
                auto salt = (slot == 0u ? salt_a : salt_b) ^ marker;
                for (auto lane = 0u; lane < ray_count; lane++) {
                    auto checksum = 0u;
                    for (auto i = 0u; i < snapshot_count; i++) { checksum ^= input[offset + (lane + i) % size] * (i * 2u + 1u); }
                    auto expected = (input[offset + lane % size] ^ host_common[common_offset + lane % common_size] ^ salt ^ checksum) +
                                    size * 3u + common_size * 5u + output_size * 7u;
                    auto expected_type = use_procedural ? HitType::Procedural : HitType::Surface;
                    auto expected_hit = make_uint4(static_cast<uint>(expected_type), 0u, 0u,
                                                   size ^ (common_size << 8u) ^ (output_size << 16u) ^ salt ^ checksum);
                    auto actual = host_output[output_offset + slot * ray_count + lane];
                    auto hit = host_hits[lane * 2u + slot];
                    if (actual != expected || !all(hit == expected_hit)) {
                        LUISA_WARNING("Resource capture mismatch: snapshots={} procedural={} slot={} lane={} value={}/{} hit={}/{}.",
                                      snapshot_count, use_procedural, slot, lane, actual, expected, hit, expected_hit);
                        correct = false;
                    }
                }
            }
            for (auto i = 0u; i < host_output.size(); i++) {
                if (i < output_offset || i >= output_offset + output_size) {
                    correct = correct && host_output[i] == sentinel;
                }
            }
            expect(correct) << "resource captures preserve descriptor size/offset, unique roots, conflicting roots and interleaved scalar captures";
            if (!correct) { return false; }
        }
    }
    return true;
}

[[nodiscard]] bool run_surface_filter_termination(Device &device, Stream &stream, const Mesh &mesh) {
    constexpr auto ray_count = 68u;
    Kernel1D kernel = [](AccelVar scene, BufferUInt4 hits,
                         BufferUInt4 captures, BufferFloat4 distances) noexcept {
        auto lane = dispatch_x();
        UInt accept_mask = lane & 3u;
        Float x = (lane % 17u).cast<float>() * (1.0f / 32.0f) - 0.25f;
        auto ray = make_ray(make_float3(x, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
        for (auto any : {false, true}) {
            UInt calls = 0u;
            UInt accepted = 0u;
            UInt seen = 0u;
            UInt invalid = 0u;
            UInt stamp = lane ^ 0x1234u;
            auto filter = [&](auto &candidate) noexcept {
                // Any callback after an accepted termination is an error,
                // regardless of the backend's candidate visitation order.
                $if (accepted != 0u) { invalid |= 1u; };
                calls += 1u;
                auto h = candidate.hit();
                $if (h.inst > 1u | h.prim != 0u) {
                    invalid |= 2u;
                }
                $else {
                    UInt bit = 1u << h.inst;
                    seen |= bit;
                    Float expected_t = ite(h.inst == 0u, 0.75f, 1.0f);
                    $if (!(abs(h.bary.x - (0.25f + x * 0.25f)) <= 1e-5f &
                           abs(h.bary.y - 0.5f) <= 1e-5f &
                           abs(h.committed_ray_t - expected_t) <= 1e-5f)) {
                        invalid |= 4u;
                    };
                    $if ((accept_mask & bit) != 0u) {
                        accepted += 1u;
                        stamp = (lane ^ 0x5678u) + bit * 17u;
                        candidate.commit();
                        candidate.terminate();
                    };
                };
            };
            auto hit = any ? scene.traverse_any(ray, {}).on_surface_candidate(filter).trace() :
                             scene.traverse(ray, {}).on_surface_candidate(filter).trace();
            auto index = lane * 2u + static_cast<uint>(any);
            hits.write(index, make_uint4(hit->hit_type, hit->inst, hit->prim, stamp));
            captures.write(index, make_uint4(calls, accepted, seen, invalid));
            distances.write(index, make_float4(hit->bary, hit->committed_ray_t, x));
        }
    };
    LUISA_INFO("Surface filter termination fixture: AST hash={:016x}.", kernel.function()->function().hash());
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    auto scene = device.create_accel();
    scene.emplace_back(mesh, translation(make_float3(0.0f, 0.0f, 0.25f)), 0xffu, false);
    scene.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    auto hits = device.create_buffer<uint4>(ray_count * 2u);
    auto captures = device.create_buffer<uint4>(ray_count * 2u);
    auto distances = device.create_buffer<float4>(ray_count * 2u);
    luisa::vector<uint4> host_hits(hits.size());
    luisa::vector<uint4> host_captures(captures.size());
    luisa::vector<float4> host_distances(distances.size());
    stream << scene.build()
           << shader(scene, hits, captures, distances).dispatch(ray_count)
           << hits.copy_to(host_hits.data())
           << captures.copy_to(host_captures.data())
           << distances.copy_to(host_distances.data()) << synchronize();
    auto correct = true;
    for (auto lane = 0u; lane < ray_count; lane++) {
        auto accept_mask = lane & 3u;
        auto expected_x = static_cast<float>(lane % 17u) / 32.0f - 0.25f;
        for (auto any = 0u; any < 2u; any++) {
            auto index = lane * 2u + any;
            auto h = host_hits[index];
            auto c = host_captures[index];
            auto d = host_distances[index];
            auto valid = c.w == 0u && (c.z & ~3u) == 0u && d.w == expected_x;
            if (accept_mask == 0u) {
                // Rejected candidates may be repeated after BVH splitting.
                valid = valid && h.x == static_cast<uint>(HitType::Miss) &&
                        c.x >= 2u && c.y == 0u && c.z == 3u && h.w == (lane ^ 0x1234u);
            } else {
                auto valid_instance = h.y < 2u && (accept_mask & (1u << h.y)) != 0u;
                valid = valid && h.x == static_cast<uint>(HitType::Surface) && valid_instance &&
                        h.z == 0u && c.y == 1u && c.x >= (c.z == 3u ? 2u : 1u);
                if (valid_instance) {
                    auto bit = 1u << h.y;
                    valid = valid && (c.z & bit) != 0u &&
                            h.w == (lane ^ 0x5678u) + bit * 17u &&
                            std::abs(d.x - (0.25f + expected_x * 0.25f)) <= 1e-5f &&
                            std::abs(d.y - 0.5f) <= 1e-5f &&
                            std::abs(d.z - (h.y == 0u ? 0.75f : 1.0f)) <= 1e-5f;
                    if (accept_mask == 3u) { valid = valid && c.x == 1u && c.z == bit; }
                }
            }
            if (!valid) {
                LUISA_WARNING("Surface filter termination mismatch: lane={}, any={}, mask={}, hit={}, captures={}, bary/t={}.",
                              lane, any, accept_mask, h, c, d);
                correct = false;
            }
        }
    }
    expect(correct) << "accepted surface termination stops callbacks and preserves hit fields and captured writes";
    return correct;
}

[[nodiscard]] bool run_zero_capture_filter_queries(Device &device, Stream &stream, const Mesh &mesh) {
    constexpr auto ray_count = 17u;
    Kernel1D kernel = [](AccelVar scene, BufferUInt4 hits, BufferFloat4 distances) noexcept {
        auto lane = dispatch_id().x;
        Float x = lane.cast<float>() * (1.0f / 32.0f) - 0.25f;
        auto ray = make_ray(make_float3(x, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
        for (auto query = 0u; query < 4u; query++) {
            // The host loop records two equivalent callbacks followed by two
            // callbacks with a different literal. Nothing is captured in XIR.
            auto wanted_instance = query < 2u ? 0u : 1u;
            auto filter = [wanted_instance](auto &candidate) noexcept {
                auto h = candidate.hit();
                $if (h.inst == wanted_instance) { candidate.commit(); };
            };
            auto hit = (query & 1u) != 0u ? scene.traverse_any(ray, {}).on_surface_candidate(filter).trace() :
                                          scene.traverse(ray, {}).on_surface_candidate(filter).trace();
            auto index = lane * 4u + query;
            hits.write(index, make_uint4(hit->hit_type, hit->inst, hit->prim, query));
            distances.write(index, make_float4(hit->bary, hit->committed_ray_t, x));
        }
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    auto scene = device.create_accel();
    scene.emplace_back(mesh, translation(make_float3(0.0f, 0.0f, 0.25f)), 0xffu, false);
    scene.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    auto hits = device.create_buffer<uint4>(ray_count * 4u);
    auto distances = device.create_buffer<float4>(ray_count * 4u);
    luisa::vector<uint4> host_hits(hits.size());
    luisa::vector<float4> host_distances(distances.size());
    stream << scene.build()
           << shader(scene, hits, distances).dispatch(ray_count)
           << hits.copy_to(luisa::span{host_hits})
           << distances.copy_to(luisa::span{host_distances})
           << synchronize();
    auto correct = true;
    for (auto lane = 0u; lane < ray_count; lane++) {
        for (auto query = 0u; query < 4u; query++) {
            auto index = lane * 4u + query;
            auto h = host_hits[index];
            auto d = host_distances[index];
            auto wanted_instance = query < 2u ? 0u : 1u;
            auto expected_x = static_cast<float>(lane) / 32.0f - 0.25f;
            auto valid = h.x == static_cast<uint>(HitType::Surface) && h.y == wanted_instance &&
                         h.z == 0u && h.w == query &&
                         std::abs(d.x - (0.25f + expected_x * 0.25f)) <= 1e-5f &&
                         std::abs(d.y - 0.5f) <= 1e-5f &&
                         std::abs(d.z - (wanted_instance == 0u ? 0.75f : 1.0f)) <= 1e-5f &&
                         d.w == expected_x;
            if (!valid) {
                LUISA_WARNING("Zero-capture filter mismatch: lane={}, query={}, hit=({}, {}, {}), bary/t=({}, {}, {}).",
                              lane, query, h.x, h.y, h.z, d.x, d.y, d.z);
                correct = false;
            }
        }
    }
    expect(correct) << "shared zero-capture callbacks preserve closest/any queries and distinct filter constants";
    return correct;
}

[[nodiscard]] bool run_surface_filter_queries(Device &device, Stream &stream, const Mesh &mesh) {
    constexpr auto ray_count = 68u;
    Kernel1D kernel = [](AccelVar scene, UInt opaque_mask, BufferUInt4 hits,
                        BufferUInt4 captures, BufferFloat4 distances) noexcept {
        auto lane = dispatch_id().x;
        UInt accept_mask = lane & 3u;
        Float x = (lane % 17u).cast<float>() * (1.0f / 32.0f) - 0.25f;
        auto ray = make_ray(make_float3(x, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
        for (auto any : {false, true}) {
            // Initialize captures before constructing the query: traversal is
            // recorded when its DSL expression is created. OR tolerates a
            // backend visiting a split primitive more than once.
            UInt seen = 0u;
            UInt accepted = 0u;
            UInt invalid = 0u;
            UInt stamp = lane & 255u;
            auto filter = [&](auto &candidate) noexcept {
                auto h = candidate.hit();
                $if (h.inst > 1u | h.prim != 0u) {
                    invalid |= 1u;
                }
                $else {
                    UInt bit = 1u << h.inst;
                    seen |= bit;
                    stamp |= bit << 8u;
                    $if ((opaque_mask & bit) != 0u |
                         abs(h.bary.x - (0.25f + x * 0.25f)) > 1e-5f |
                         abs(h.bary.y - 0.5f) > 1e-5f) {
                        invalid |= 2u;
                    };
                    $if ((accept_mask & bit) != 0u) {
                        accepted |= bit;
                        candidate.commit();
                    };
                };
            };
            auto hit = any ? scene.traverse_any(ray, {}).on_surface_candidate(filter).trace() :
                             scene.traverse(ray, {}).on_surface_candidate(filter).trace();
            auto index = lane * 2u + static_cast<uint>(any);
            hits.write(index, make_uint4(hit->hit_type, hit->inst, hit->prim, accept_mask));
            captures.write(index, make_uint4(seen, accepted, invalid, stamp));
            distances.write(index, make_float4(hit->bary, hit->committed_ray_t, x));
        }
    };
    LUISA_INFO("Surface filter fixture: AST hash={:016x}.", kernel.function()->function().hash());
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    auto hits = device.create_buffer<uint4>(ray_count * 2u);
    auto captures = device.create_buffer<uint4>(ray_count * 2u);
    auto distances = device.create_buffer<float4>(ray_count * 2u);
    luisa::vector<uint4> host_hits(hits.size());
    luisa::vector<uint4> host_captures(captures.size());
    luisa::vector<float4> host_distances(distances.size());
    for (auto opaque_mask = 0u; opaque_mask < 4u; opaque_mask++) {
        auto scene = device.create_accel();
        scene.emplace_back(mesh, translation(make_float3(0.0f, 0.0f, 0.25f)), 0xffu, (opaque_mask & 1u) != 0u);
        scene.emplace_back(mesh, make_float4x4(1.0f), 0xffu, (opaque_mask & 2u) != 0u);
        stream << scene.build()
               << shader(scene, opaque_mask, hits, captures, distances).dispatch(ray_count)
               << hits.copy_to(host_hits.data())
               << captures.copy_to(host_captures.data())
               << distances.copy_to(host_distances.data())
               << synchronize();
        auto correct = true;
        for (auto lane = 0u; lane < ray_count; lane++) {
            auto accept_mask = lane & 3u;
            auto eligible_mask = accept_mask | opaque_mask;
            for (auto any = 0u; any < 2u; any++) {
                auto index = lane * 2u + any;
                auto h = host_hits[index];
                auto c = host_captures[index];
                auto d = host_distances[index];
                auto expected_x = static_cast<float>(lane % 17u) / 32.0f - 0.25f;
                auto valid = c.z == 0u && (c.x & ~3u) == 0u && (c.x & opaque_mask) == 0u &&
                             c.y == (c.x & accept_mask) && c.w == ((lane & 255u) | (c.x << 8u)) &&
                             h.w == accept_mask && d.w == expected_x;
                if (eligible_mask == 0u) {
                    valid = valid && h.x == static_cast<uint>(HitType::Miss) && c.x == 3u;
                } else {
                    auto expected_closest = (eligible_mask & 1u) != 0u ? 0u : 1u;
                    auto valid_instance = h.y < 2u && (eligible_mask & (1u << h.y)) != 0u;
                    valid = valid && h.x == static_cast<uint>(HitType::Surface) && valid_instance && h.z == 0u;
                    if (valid_instance) {
                        auto bit = 1u << h.y;
                        valid = valid && (any != 0u || h.y == expected_closest) &&
                                ((opaque_mask & bit) != 0u || (c.y & bit) != 0u) &&
                                std::abs(d.x - (0.25f + expected_x * 0.25f)) <= 1e-5f &&
                                std::abs(d.y - 0.5f) <= 1e-5f &&
                                std::abs(d.z - (h.y == 0u ? 0.75f : 1.0f)) <= 1e-5f;
                    }
                    // Rejecting the nearer surface cannot hide it from a
                    // closest query that eventually commits the farther one.
                    if (any == 0u && opaque_mask == 0u && accept_mask == 2u) {
                        valid = valid && c.x == 3u;
                    }
                }
                if (!valid) {
                    LUISA_WARNING("Surface filter mismatch: opaque={}, lane={}, any={}, hit=({}, {}, {}), captures=({}, {}, {}, {}), bary/t=({}, {}, {}).",
                                  opaque_mask, lane, any, h.x, h.y, h.z, c.x, c.y, c.z, c.w, d.x, d.y, d.z);
                    correct = false;
                }
            }
        }
        expect(correct) << "surface filtering preserves closest/any hits, opaque traversal and captured writes";
        if (!correct) { return false; }
    }
    return run_surface_filter_termination(device, stream, mesh) &&
           run_zero_capture_filter_queries(device, stream, mesh);
}

[[nodiscard]] bool run(Device &device, const Options &options) {
    // static constexpr so the DSL kernel lambdas below can reference it without
    // an explicit capture (automatic-storage locals are not implicitly capturable).
    static constexpr auto query_count = 5u;
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
    if (options.surface_filter_only) {
        stream << vertices_buffer.copy_from(luisa::span{vertices})
               << triangles_buffer.copy_from(luisa::span{triangles})
               << mesh.build() << synchronize();
        return run_surface_filter_queries(device, stream, mesh);
    }

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
        // visited first commits; the second attempts a farther hit, then
        // terminates without replacing the first primitive's committed record.
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
                                     // Within the original ray, but beyond the
                                     // committed 1.25 bound: retain every field
                                     // of the first candidate, including prim.
                                     if (!benchmark) { candidate.commit(2.0f); }
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
    if (!options.benchmark) {
        if (!run_surface_filter_queries(device, stream, mesh)) { return false; }
        if (!run_resource_capture_queries(device, stream, surface_scene, procedural_scene)) { return false; }
        if (!run_hardware_payload_boundary(device, stream, surface_scene, procedural_scene)) { return false; }
        return run_payload_capture_boundary(device, stream, surface_scene, procedural_scene);
    }

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
    auto requested_codegen = codegen_env != nullptr && std::strcmp(codegen_env, "1") == 0 ? "llvm" : "ast";
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
        LUISA_INFO("Usage: {} <backend> [--benchmark] [--diagnose-dispatches] [--large-capture-stress] [--surface-filter-only] [--rays N] [--warmup N] [--dispatches N] [--samples N]", argv[0]);
        return 2;
    }
    auto dc = luisa::test::create_device(argc, argv);
    if (options.large_capture_stress) { return run_large_capture(dc.device, options) ? 0 : 1; }
    if (!run(dc.device, options)) { return 1; }
    if (!options.benchmark && !options.surface_filter_only) {
        Options regression_options;
        regression_options.warmup = 1u;
        regression_options.dispatches = 2u;
        regression_options.samples = 2u;
        if (!run_large_capture(dc.device, regression_options)) { return 1; }
    }
    return 0;
}
