// Strict Metal AIR integration tests for explicit ray-query state machines.
//
// The triangle case is intentionally outlineable and compares the native
// stateful loop against the Metal intersection-function-table pipeline. The
// procedural case is intentionally not outlineable yet and proves that the
// pipeline-enabled compiler retains the complete stateful loop atomically.

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <cstddef>
#include <cmath>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

struct PipelinePayloadBool4 {
    bool x;
    bool y;
    bool z;
    bool w;
};

struct PipelinePayloadConfig {
    PipelinePayloadBool4 flags;
    byte4 bytes;
    uint bias;
};

static_assert(sizeof(PipelinePayloadBool4) == 4u);
static_assert(offsetof(PipelinePayloadBool4, w) == 3u);
static_assert(sizeof(byte4) == 4u);
static_assert(sizeof(PipelinePayloadConfig) == 12u);

LUISA_STRUCT(PipelinePayloadBool4, x, y, z, w) {};
LUISA_STRUCT(PipelinePayloadConfig, flags, bytes, bias) {};

namespace {

constexpr auto surface_hit = static_cast<uint>(HitType::Surface);
constexpr auto procedural_hit = static_cast<uint>(HitType::Procedural);

struct UIntRun {
    Buffer<uint> result;
    Buffer<uint> scratch;
    std::array<uint, 4u> host_result{};
    std::array<uint, 3u> host_scratch{};
};

[[nodiscard]] UIntRun make_uint_run(Device &device) {
    return {
        device.create_buffer<uint>(4u),
        device.create_buffer<uint>(3u)};
}

void test_outlineable_triangle_state_machine(Device &device) {
    constexpr std::array vertices{
        float3{-1.0f, -1.0f, 0.0f},
        float3{1.0f, -1.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f},
        float3{-1.0f, -1.0f, -1.0f},
        float3{1.0f, -1.0f, -1.0f},
        float3{0.0f, 1.0f, -1.0f},
        float3{-1.0f, -1.0f, -2.0f},
        float3{1.0f, -1.0f, -2.0f},
        float3{0.0f, 1.0f, -2.0f}};
    constexpr std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u},
        Triangle{6u, 7u, 8u}};
    constexpr std::array selection{0u, 0u, 1u};
    constexpr std::array weights{11u, 23u, 37u};
    constexpr PipelinePayloadConfig payload_config{
        .flags = {true, false, true, true},
        .bytes = byte4{3, 5, 7, 11},
        .bias = 19u};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto selection_buffer = device.create_buffer<uint>(selection.size());
    auto weight_buffer = device.create_buffer<uint>(weights.size());
    auto heap = device.create_bindless_array(1u);
    heap.emplace_on_update(0u, weight_buffer);
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);

    Kernel1D state_machine = [](
                                 AccelVar scene,
                                 BufferUInt result,
                                 BufferUInt scratch,
                                 BufferUInt selection,
                                 BufferUInt weights,
                                 BindlessVar heap,
                                 UInt heap_slot,
                                 Var<PipelinePayloadConfig> config) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto query = scene.query(ray, {});
        UInt candidate_count = 0u;
        UInt checksum = 0u;
        $while (query.proceed()) {
            $if (query.is_surface_candidate()) {
                auto candidate = query.surface_candidate();
                auto hit = candidate.hit();
                UInt folded = 0u;
                $for (step, 0u, 6u) {
                    $if (step == 1u) { $continue; };
                    folded += (step + 1u) * (hit.prim + 1u);
                    $if (step == hit.prim + 2u) { $break; };
                };

                UInt branch = 0u;
                $switch (hit.prim) {
                    $case (0u) { branch = 101u; };
                    $case (1u) { branch = 211u; };
                    $default { branch = 307u; };
                };

                auto mirrored = 2u - hit.prim;
                auto resource_value =
                    weights.read(hit.prim) +
                    heap.buffer<uint>(heap_slot).read(mirrored);
                auto flags = config.flags;
                auto bytes = config.bytes;
                auto payload_bias =
                    config.bias +
                    cast<uint>(bytes.x) + cast<uint>(bytes.y) +
                    cast<uint>(bytes.z) + cast<uint>(bytes.w) +
                    ite(flags.x, 1u, 0u) + ite(flags.y, 2u, 0u) +
                    ite(flags.z, 4u, 0u) + ite(flags.w, 8u, 0u);
                auto candidate_checksum =
                    resource_value + folded + branch + payload_bias;
                scratch.write(hit.prim, candidate_checksum);
                checksum += candidate_checksum;
                candidate_count += 1u;

                $if (flags.w & (selection.read(hit.prim) != 0u) &
                     (resource_value == 48u)) {
                    candidate.commit();
                };
            }
            $else {
                // This empty arm is deliberate: it keeps the explicit
                // software-state-machine shape while remaining triangle-only
                // and therefore eligible for IFT outlining.
            };
        };
        auto committed = query.committed_hit();
        result.write(0u, candidate_count);
        result.write(1u, committed.hit_type);
        result.write(2u, committed.prim);
        result.write(3u, checksum);
    };

    auto pipeline_shader = device.compile(
        state_machine,
        ShaderOption{
            .enable_cache = false,
            .enable_ray_query_pipeline = true,
            .force_ray_query_pipeline = true});
    auto stateful_shader = device.compile(
        state_machine,
        ShaderOption{
            .enable_cache = false,
            .enable_ray_query_pipeline = false});

    auto indirect_dispatch = device.create_indirect_dispatch_buffer(1u);
    Kernel1D prepare_indirect = [](
                                    Var<IndirectDispatchBuffer> commands) noexcept {
        commands.set_dispatch_count(1u);
        commands.set_kernel(0u, make_uint3(1u), make_uint3(1u), 0u);
    };
    auto prepare_indirect_shader = device.compile(prepare_indirect);

    std::array runs{
        make_uint_run(device), make_uint_run(device),
        make_uint_run(device), make_uint_run(device)};
    auto stream = device.create_stream();
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << selection_buffer.copy_from(luisa::span{selection})
           << weight_buffer.copy_from(luisa::span{weights})
           << heap.update()
           << mesh.build()
           << accel.build()
           << prepare_indirect_shader(indirect_dispatch).dispatch(1u)
           << pipeline_shader(
                  accel, runs[0].result, runs[0].scratch,
                  selection_buffer, weight_buffer, heap, 0u,
                  payload_config)
                  .dispatch(1u)
           << pipeline_shader(
                  accel, runs[1].result, runs[1].scratch,
                  selection_buffer, weight_buffer, heap, 0u,
                  payload_config)
                  .dispatch(indirect_dispatch)
           << stateful_shader(
                  accel, runs[2].result, runs[2].scratch,
                  selection_buffer, weight_buffer, heap, 0u,
                  payload_config)
                  .dispatch(1u)
           << stateful_shader(
                  accel, runs[3].result, runs[3].scratch,
                  selection_buffer, weight_buffer, heap, 0u,
                  payload_config)
                  .dispatch(indirect_dispatch);
    for (auto &run : runs) {
        stream << run.result.copy_to(luisa::span{run.host_result})
               << run.scratch.copy_to(luisa::span{run.host_scratch});
    }
    stream << synchronize();

    constexpr std::array expected_result{3u, surface_hit, 2u, 994u};
    constexpr std::array expected_scratch{211u, 331u, 452u};
    for (auto i = 0u; i < runs.size(); i++) {
        expect(runs[i].host_result == expected_result)
            << luisa::format(
                   "triangle state-machine result mismatch in mode {}",
                   i);
        expect(runs[i].host_scratch == expected_scratch)
            << luisa::format(
                   "triangle state-machine resource capture mismatch in mode {}",
                   i);
        expect(runs[i].host_result == runs[0].host_result)
            << "pipeline/stateful or direct/indirect result mismatch";
        expect(runs[i].host_scratch == runs[0].host_scratch)
            << "pipeline/stateful or direct/indirect scratch mismatch";
    }
}

struct ProceduralRun {
    Buffer<uint> result;
    Buffer<uint> scratch;
    Buffer<float4> ray_data;
    std::array<uint, 4u> host_result{};
    std::array<uint, 2u> host_scratch{};
    std::array<float4, 4u> host_ray_data{};
};

[[nodiscard]] ProceduralRun make_procedural_run(Device &device) {
    return {
        device.create_buffer<uint>(4u),
        device.create_buffer<uint>(2u),
        device.create_buffer<float4>(4u)};
}

void test_retained_procedural_state_machine(Device &device) {
    std::array<AABB, 2u> bounds{};
    bounds[0].packed_min = {-0.5f, -0.5f, -0.1f};
    bounds[0].packed_max = {0.5f, 0.5f, 0.1f};
    bounds[1].packed_min = {-0.5f, -0.5f, -1.1f};
    bounds[1].packed_max = {0.5f, 0.5f, -0.9f};
    constexpr std::array distances{0.95f, 1.95f};
    constexpr std::array selection{0u, 1u};
    constexpr std::array weights{13u, 29u};

    auto aabb_buffer = device.create_buffer<AABB>(bounds.size());
    auto distance_buffer = device.create_buffer<float>(distances.size());
    auto selection_buffer = device.create_buffer<uint>(selection.size());
    auto weight_buffer = device.create_buffer<uint>(weights.size());
    auto heap = device.create_bindless_array(1u);
    heap.emplace_on_update(0u, weight_buffer);
    auto primitive = device.create_procedural_primitive(aabb_buffer.view());
    auto accel = device.create_accel();
    accel.emplace_back(primitive);

    Callable trace_procedural = [](
                                    AccelVar scene,
                                    BufferUInt scratch,
                                    BufferFloat4 ray_data,
                                    BufferFloat distances,
                                    BufferUInt selection,
                                    BindlessVar heap,
                                    UInt &candidate_count,
                                    UInt &checksum) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto query = scene.query(ray, {});
        $while (query.proceed()) {
            $if (query.is_procedural_candidate()) {
                auto candidate = query.procedural_candidate();
                auto hit = candidate.hit();
                auto world_ray = candidate.ray();
                auto object_ray = candidate.object_ray();

                UInt folded = 0u;
                $for (step, 0u, 5u) {
                    $if (((step + hit.prim) & 1u) == 0u) {
                        folded += step + 1u;
                    }
                    $else {
                        folded += (step + 1u) * 2u;
                    };
                    $if (step == hit.prim + 2u) { $break; };
                };
                $switch (hit.prim) {
                    $case (0u) { folded += 17u; };
                    $default { folded += 31u; };
                };

                auto resource_value =
                    heap.buffer<uint>(0u).read(hit.prim);
                scratch.write(hit.prim, folded + resource_value);
                checksum += folded + resource_value;
                candidate_count += 1u;
                ray_data.write(
                    hit.prim * 2u,
                    make_float4(world_ray->origin(), world_ray->t_min()));
                ray_data.write(
                    hit.prim * 2u + 1u,
                    make_float4(object_ray->direction(), object_ray->t_max()));

                $if ((selection.read(hit.prim) != 0u) &
                     (distances.read(hit.prim) > 0.0f)) {
                    candidate.commit(distances.read(hit.prim));
                };
            }
            $else {
                // The test scene has no triangle geometry.
            };
        };
        return query.committed_hit();
    };
    trace_procedural.function_builder()->set_name(
        "metal4_retained_procedural_state_machine_callable");

    Kernel1D state_machine = [&trace_procedural](
                                 AccelVar scene,
                                 BufferUInt result,
                                 BufferUInt scratch,
                                 BufferFloat4 ray_data,
                                 BufferFloat distances,
                                 BufferUInt selection,
                                 BindlessVar heap) noexcept {
        UInt candidate_count = 0u;
        UInt checksum = 0u;
        auto committed = trace_procedural(
            scene, scratch, ray_data, distances, selection, heap,
            candidate_count, checksum);
        result.write(0u, candidate_count);
        result.write(1u, committed.hit_type);
        result.write(2u, committed.prim);
        result.write(3u, checksum);
    };

    auto pipeline_enabled_shader = device.compile(
        state_machine,
        ShaderOption{
            .enable_cache = false,
            .enable_ray_query_pipeline = true,
            .force_ray_query_pipeline = true});
    auto explicitly_stateful_shader = device.compile(
        state_machine,
        ShaderOption{
            .enable_cache = false,
            .enable_ray_query_pipeline = false});

    auto indirect_dispatch = device.create_indirect_dispatch_buffer(1u);
    Kernel1D prepare_indirect = [](
                                    Var<IndirectDispatchBuffer> commands) noexcept {
        commands.set_dispatch_count(1u);
        commands.set_kernel(0u, make_uint3(1u), make_uint3(1u), 0u);
    };
    auto prepare_indirect_shader = device.compile(prepare_indirect);
    std::array runs{
        make_procedural_run(device), make_procedural_run(device),
        make_procedural_run(device), make_procedural_run(device)};

    auto stream = device.create_stream();
    stream << aabb_buffer.copy_from(luisa::span{bounds})
           << distance_buffer.copy_from(luisa::span{distances})
           << selection_buffer.copy_from(luisa::span{selection})
           << weight_buffer.copy_from(luisa::span{weights})
           << heap.update()
           << primitive.build()
           << accel.build()
           << prepare_indirect_shader(indirect_dispatch).dispatch(1u)
           << pipeline_enabled_shader(
                  accel, runs[0].result, runs[0].scratch,
                  runs[0].ray_data, distance_buffer,
                  selection_buffer, heap)
                  .dispatch(1u)
           << pipeline_enabled_shader(
                  accel, runs[1].result, runs[1].scratch,
                  runs[1].ray_data, distance_buffer,
                  selection_buffer, heap)
                  .dispatch(indirect_dispatch)
           << explicitly_stateful_shader(
                  accel, runs[2].result, runs[2].scratch,
                  runs[2].ray_data, distance_buffer,
                  selection_buffer, heap)
                  .dispatch(1u)
           << explicitly_stateful_shader(
                  accel, runs[3].result, runs[3].scratch,
                  runs[3].ray_data, distance_buffer,
                  selection_buffer, heap)
                  .dispatch(indirect_dispatch);
    for (auto &run : runs) {
        stream << run.result.copy_to(luisa::span{run.host_result})
               << run.scratch.copy_to(luisa::span{run.host_scratch})
               << run.ray_data.copy_to(luisa::span{run.host_ray_data});
    }
    stream << synchronize();

    constexpr std::array expected_result{2u, procedural_hit, 1u, 112u};
    constexpr std::array expected_scratch{38u, 74u};
    constexpr auto epsilon = 1.0e-5f;
    for (auto i = 0u; i < runs.size(); i++) {
        expect(runs[i].host_result == expected_result)
            << luisa::format(
                   "procedural state-machine result mismatch in mode {}",
                   i);
        expect(runs[i].host_scratch == expected_scratch)
            << luisa::format(
                   "procedural state-machine resource mismatch in mode {}",
                   i);
        expect(runs[i].host_result == runs[0].host_result)
            << "procedural pipeline-option/direct/indirect result mismatch";
        expect(runs[i].host_scratch == runs[0].host_scratch)
            << "procedural pipeline-option/direct/indirect scratch mismatch";
        for (auto j = 0u; j < runs[i].host_ray_data.size(); j++) {
            expect(all(abs(
                           runs[i].host_ray_data[j] -
                           runs[0].host_ray_data[j]) < epsilon))
                << "procedural world/object ray payload mismatch";
        }
    }

    expect(all(abs(
                   runs[0].host_ray_data[0] -
                   make_float4(0.0f, 0.0f, 1.0f, 0.0f)) < epsilon))
        << "unexpected first procedural world ray";
    expect(all(abs(
                   runs[0].host_ray_data[1].xyz() -
                   make_float3(0.0f, 0.0f, -1.0f)) < epsilon) &&
           (std::abs(runs[0].host_ray_data[1].w - 10.0f) < epsilon ||
            std::abs(runs[0].host_ray_data[1].w - 1.95f) < epsilon))
        << "unexpected first procedural object ray";
    expect(all(abs(
                   runs[0].host_ray_data[3] -
                   make_float4(0.0f, 0.0f, -1.0f, 10.0f)) < epsilon))
        << "unexpected committed procedural object ray";
}

struct MixedPipelineRun {
    Buffer<uint> result;
    Buffer<float4> rays;
    std::array<uint, 8u> host_result{};
    std::array<float4, 4u> host_rays{};
};

[[nodiscard]] MixedPipelineRun make_mixed_pipeline_run(Device &device) {
    return {
        device.create_buffer<uint>(8u),
        device.create_buffer<float4>(4u)};
}

void test_outlineable_mixed_procedural_and_surface_pipeline(Device &device) {
    // Instance 0: opaque triangle at z = -3.0 (t = 4.0) -> auto-commits in hardware without surface callback.
    // Instance 1: non-opaque triangle scaled by 2 and translated to z = -1.5 (t = 2.5) -> enters surface callback.
    // Instance 2: procedural primitive translated by (1, -2, 0) * scaling(2.0) with two overlapping AABBs around world origin x=0, y=0:
    //   object space x in [-0.5, -0.5].. wait: world (0, 0, z) -> object ((0 - 1)/2, (0 - (-2))/2, z/2) = (-0.5, 1.0, z/2).
    constexpr std::array vertices{
        float3{-2.0f, -2.0f, 0.0f},
        float3{2.0f, -2.0f, 0.0f},
        float3{0.0f, 2.0f, 0.0f}};
    constexpr std::array triangles{Triangle{0u, 1u, 2u}};
    std::array<AABB, 2u> bounds{};
    // Both object-space AABBs enclose (-0.5, 1.0, z_obj):
    bounds[0].packed_min = {-1.0f, 0.5f, -0.4f};
    bounds[0].packed_max = {0.0f, 1.5f, -0.1f};
    bounds[1].packed_min = {-1.0f, 0.5f, -1.2f};
    bounds[1].packed_max = {0.0f, 1.5f, -0.9f};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto aabb_buffer = device.create_buffer<AABB>(bounds.size());
    auto opaque_mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto cutout_mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto proc_prim = device.create_procedural_primitive(aabb_buffer.view());

    auto accel = device.create_accel();
    accel.emplace_back(
        opaque_mesh, translation(make_float3(0.0f, 0.0f, -3.0f)), 0x01u, true);
    accel.emplace_back(
        cutout_mesh,
        translation(make_float3(0.0f, 0.0f, -1.5f)) * scaling(2.0f),
        0x02u, false);
    accel.emplace_back(
        proc_prim,
        translation(make_float3(1.0f, -2.0f, 0.0f)) * scaling(2.0f),
        0x04u, false);

    // Value-only Callable exercising deduplicate-ray-query-handlers when called twice!
    Callable trace_shadow_any = [](AccelVar scene, Var<Ray> shadow_ray) noexcept {
        UInt count = 0u;
        auto hit = scene.traverse_any(shadow_ray, {.visibility_mask = 0x04u})
                       .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                           count += 1u;
                           c.commit(c.ray()->t_min() + 0.25f);
                           c.terminate();
                       })
                       .trace();
        return ite(hit->miss(), 0u, 10u + count);
    };

    Kernel1D mixed_kernel = [&trace_shadow_any](
                                AccelVar scene,
                                BufferUInt out_u,
                                BufferFloat4 out_f,
                                Float nan_input) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.1f, 10.0f);
        UInt surf_calls = 0u;
        UInt proc_calls = 0u;
        Float4 captured_surf_obj_ray = make_float4(-99.0f);
        Float4 captured_proc_obj_orig = make_float4(-99.0f);
        Float4 captured_proc_obj_dir = make_float4(-99.0f);
        Float4 captured_proc_world_after_commit = make_float4(-99.0f);

        // First query: reject cutout_mesh (inst 1) and reject both procedural AABBs -> must hit opaque_mesh (inst 0) at t=4.0 with surf_calls == 1 (opaque inst 0 never enters callback!).
        UInt opaque_surf_calls = 0u;
        auto opaque_only_hit =
            scene.traverse(ray, {})
                .on_surface_candidate([&](SurfaceCandidate &c) noexcept {
                    opaque_surf_calls += 1u;
                    auto obj_r = c.object_ray();
                    captured_surf_obj_ray = make_float4(
                        obj_r->origin(), obj_r->t_max());
                    // Reject inst 1 so traversal must reach opaque inst 0.
                })
                .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                    // Test out-of-bounds and NaN commit rejection:
                    c.commit(nan_input);
                    c.commit(-1.0f);
                    c.commit(99.0f);
                })
                .trace();

        // Second query: mixed procedural + surface where procedural prim 0 commits twice (1.8 then 1.4) and observes contracted t_max!
        auto mixed_hit =
            scene.traverse(ray, {})
                .on_surface_candidate([&](SurfaceCandidate &c) noexcept {
                    surf_calls += 1u;
                    c.commit();
                })
                .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                    proc_calls += 1u;
                    auto h = c.hit();
                    $if (h.prim == 0u) {
                        auto obj_r = c.object_ray();
                        captured_proc_obj_orig = make_float4(
                            obj_r->origin(), obj_r->t_min());
                        captured_proc_obj_dir = make_float4(
                            obj_r->direction(), obj_r->t_max());
                        c.commit(1.8f);
                        c.commit(nan_input);// must be ignored, preserving 1.8f
                        c.commit(2.2f);     // > contracted 1.8f -> must be ignored!
                        c.commit(1.4f);     // <= 1.8f -> tightens commit to 1.4f!
                        auto wr = c.ray();
                        captured_proc_world_after_commit = make_float4(
                            wr->origin(), wr->t_max());
                    }
                    $else {
                        c.commit(3.1f);
                    };
                })
                .trace();

        // Two calls to the same value-capturing Callable containing an outlineable procedural query:
        auto s0 = trace_shadow_any(scene, ray);
        auto s1 = trace_shadow_any(
            scene,
            make_ray(make_float3(50.0f, 50.0f, 1.0f),
                     make_float3(0.0f, 0.0f, -1.0f), 0.1f, 10.0f));

        out_u.write(0u, opaque_surf_calls);
        out_u.write(1u, opaque_only_hit.hit_type);
        out_u.write(2u, opaque_only_hit.inst);
        out_u.write(3u, mixed_hit.hit_type);
        out_u.write(4u, mixed_hit.inst);
        out_u.write(5u, mixed_hit.prim);
        out_u.write(6u, s0);
        out_u.write(
            7u,
            s1 + ite(surf_calls > 99u, 100u, 0u) +
                ite(proc_calls == 0u, 100u, 0u));

        out_f.write(0u, captured_surf_obj_ray);
        out_f.write(1u, captured_proc_obj_orig);
        out_f.write(2u, captured_proc_obj_dir);
        out_f.write(3u, make_float4(
                            captured_proc_world_after_commit.w,
                            opaque_only_hit->distance(),
                            mixed_hit->distance(),
                            0.0f));
    };

    auto pipeline_shader = device.compile(
        mixed_kernel,
        ShaderOption{
            .enable_cache = false,
            .enable_ray_query_pipeline = true,
            .force_ray_query_pipeline = true});
    auto stateful_shader = device.compile(
        mixed_kernel,
        ShaderOption{
            .enable_cache = false,
            .enable_ray_query_pipeline = false});

    auto indirect_dispatch = device.create_indirect_dispatch_buffer(1u);
    Kernel1D prepare_indirect = [](Var<IndirectDispatchBuffer> commands) noexcept {
        commands.set_dispatch_count(1u);
        commands.set_kernel(0u, make_uint3(1u), make_uint3(1u), 0u);
    };
    auto prepare_indirect_shader = device.compile(prepare_indirect);

    std::array runs{
        make_mixed_pipeline_run(device),
        make_mixed_pipeline_run(device),
        make_mixed_pipeline_run(device),
        make_mixed_pipeline_run(device)};
    const float quiet_nan = std::numeric_limits<float>::quiet_NaN();

    auto stream = device.create_stream();
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << aabb_buffer.copy_from(luisa::span{bounds})
           << opaque_mesh.build()
           << cutout_mesh.build()
           << proc_prim.build()
           << accel.build()
           << prepare_indirect_shader(indirect_dispatch).dispatch(1u)
           << pipeline_shader(accel, runs[0].result, runs[0].rays, quiet_nan).dispatch(1u)
           << pipeline_shader(accel, runs[1].result, runs[1].rays, quiet_nan).dispatch(indirect_dispatch)
           << stateful_shader(accel, runs[2].result, runs[2].rays, quiet_nan).dispatch(1u)
           << stateful_shader(accel, runs[3].result, runs[3].rays, quiet_nan).dispatch(indirect_dispatch);
    for (auto &run : runs) {
        stream << run.result.copy_to(luisa::span{run.host_result})
               << run.rays.copy_to(luisa::span{run.host_rays});
    }
    stream << synchronize();

    constexpr std::array<uint, 8u> expected_u{
        1u, surface_hit, 0u, procedural_hit, 2u, 0u, 11u, 0u};
    constexpr auto eps = 1.0e-4f;
    for (auto i = 0u; i < runs.size(); i++) {
        expect(runs[i].host_result == expected_u)
            << luisa::format("mixed pipeline uint mismatch in mode {}", i);
        // Cutout mesh transform: translate(0,0,-1.5) * scale(2) -> inverse maps (0,0,1) to (0, 0, 1.25)
        expect(std::abs(runs[i].host_rays[0].x) < eps &&
               std::abs(runs[i].host_rays[0].y) < eps &&
               std::abs(runs[i].host_rays[0].z - 1.25f) < eps)
            << luisa::format("surface object_ray origin mismatch in mode {}", i);
        // Procedural transform: translate(1,-2,0) * scale(2) -> inverse maps origin (0,0,1) to (-0.5, 1.0, 0.5) and dir (0,0,-1) to (0,0,-0.5)
        expect(all(abs(runs[i].host_rays[1] - make_float4(-0.5f, 1.0f, 0.5f, 0.1f)) < eps))
            << luisa::format("procedural object_ray origin/t_min mismatch in mode {}", i);
        expect(std::abs(runs[i].host_rays[2].x) < eps &&
               std::abs(runs[i].host_rays[2].y) < eps &&
               std::abs(runs[i].host_rays[2].z + 0.5f) < eps)
            << luisa::format("procedural object_ray direction mismatch in mode {}", i);
        expect(all(abs(runs[i].host_rays[3] - make_float4(1.4f, 4.0f, 1.4f, 0.0f)) < eps))
            << luisa::format("intra-callback contracted t_max or committed distance mismatch in mode {}", i);
    }
}

void test_per_query_curve_and_procedural_coexistence(Device &device) {
    constexpr auto basis = CurveBasis::PIECEWISE_LINEAR;
    constexpr std::array control_points{
        float4{-1.0f, 0.0f, 0.0f, 0.2f},
        float4{1.0f, 0.0f, 0.0f, 0.2f}};
    constexpr std::array segments{0u};
    std::array<AABB, 1u> bounds{};
    bounds[0].packed_min = {-0.5f, -0.5f, -2.1f};
    bounds[0].packed_max = {0.5f, 0.5f, -1.9f};

    auto cp_buf = device.create_buffer<float4>(control_points.size());
    auto seg_buf = device.create_buffer<uint>(segments.size());
    auto aabb_buf = device.create_buffer<AABB>(bounds.size());
    auto curve = device.create_curve(basis, cp_buf, seg_buf);
    auto proc = device.create_procedural_primitive(aabb_buf.view());
    auto curve_accel = device.create_accel();
    curve_accel.emplace_back(curve, make_float4x4(1.0f), 0xffu, false);
    auto proc_accel = device.create_accel();
    proc_accel.emplace_back(proc);

    Kernel1D coexist_kernel = [](AccelVar c_accel, AccelVar p_accel,
                                 BufferUInt out_u, BufferFloat2 out_f) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        // Query 0 uses curves -> retained on stateful intersection_query by loop_filter.
        auto c_hit = c_accel.traverse(ray, {.curve_bases = {basis}})
                         .on_surface_candidate([](SurfaceCandidate &c) noexcept {
                             c.commit();
                         })
                         .trace();
        // Query 1 is procedural -> outlined to 2-slot IFT in the same shader!
        auto p_hit = p_accel.traverse(ray, {})
                         .on_procedural_candidate([](ProceduralCandidate &c) noexcept {
                             c.commit(2.75f);
                         })
                         .trace();
        out_u.write(0u, c_hit.hit_type);
        out_u.write(1u, p_hit.hit_type);
        out_f.write(0u, make_float2(c_hit->distance(), p_hit->distance()));
    };

    auto shader = device.compile(
        coexist_kernel,
        ShaderOption{
            .enable_cache = false,
            .enable_ray_query_pipeline = true,
            .force_ray_query_pipeline = true});
    auto out_u = device.create_buffer<uint>(2u);
    auto out_f = device.create_buffer<float2>(1u);
    std::array<uint, 2u> host_u{};
    std::array<float2, 1u> host_f{};
    auto stream = device.create_stream();
    stream << cp_buf.copy_from(luisa::span{control_points})
           << seg_buf.copy_from(luisa::span{segments})
           << aabb_buf.copy_from(luisa::span{bounds})
           << curve.build()
           << proc.build()
           << curve_accel.build()
           << proc_accel.build()
           << shader(curve_accel, proc_accel, out_u, out_f).dispatch(1u)
           << out_u.copy_to(luisa::span{host_u})
           << out_f.copy_to(luisa::span{host_f})
           << synchronize();

    expect(host_u[0] == surface_hit && host_u[1] == procedural_hit)
        << "coexisting curve + procedural ray queries must both commit expected hit types";
    expect(std::abs(host_f[0].x - 0.8f) < 1.0e-4f &&
           std::abs(host_f[0].y - 2.75f) < 1.0e-4f)
        << "coexisting curve + procedural ray queries returned unexpected distances";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_outlineable_triangle_state_machine(dc->device);
    test_retained_procedural_state_machine(dc->device);
    test_outlineable_mixed_procedural_and_surface_pipeline(dc->device);
    test_per_query_curve_and_procedural_coexistence(dc->device);
}
