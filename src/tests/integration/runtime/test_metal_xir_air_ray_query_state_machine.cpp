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
            .enable_ray_query_pipeline = true});
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

    Kernel1D state_machine = [](
                                 AccelVar scene,
                                 BufferUInt result,
                                 BufferUInt scratch,
                                 BufferFloat4 ray_data,
                                 BufferFloat distances,
                                 BufferUInt selection,
                                 BindlessVar heap) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto query = scene.query(ray, {});
        UInt candidate_count = 0u;
        UInt checksum = 0u;
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
        auto committed = query.committed_hit();
        result.write(0u, candidate_count);
        result.write(1u, committed.hit_type);
        result.write(2u, committed.prim);
        result.write(3u, checksum);
    };

    auto pipeline_enabled_shader = device.compile(
        state_machine,
        ShaderOption{
            .enable_cache = false,
            .enable_ray_query_pipeline = true});
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
                   runs[0].host_ray_data[1] -
                   make_float4(0.0f, 0.0f, -1.0f, 10.0f)) < epsilon))
        << "unexpected first procedural object ray";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_outlineable_triangle_state_machine(dc->device);
    test_retained_procedural_state_machine(dc->device);
}
