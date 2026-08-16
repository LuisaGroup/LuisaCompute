// Test for HIP XIR RayQueryPipelineInst code generation.
// This test covers:
// - lowering a DSL ray-query loop to the XIR pipeline instruction
// - surface and procedural candidate handler dispatch
// - query-object state propagation into outlined handlers
// - multiple mutable reference captures shared by both handlers
// - committed world-ray distance after surface and procedural commits

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void expect_result(uint index, const uint4 &actual0, const uint4 &actual1,
                   const uint4 &expected0, const uint4 &expected1) {
    auto check = [index](luisa::string_view field, uint actual,
                         uint expected) noexcept {
        expect(actual == expected)
            << luisa::format("ray {} {}: got {}, expected {}",
                             index, field, actual, expected);
    };
    check("hit_type", actual0.x, expected0.x);
    check("committed_inst", actual0.y, expected0.y);
    check("committed_prim", actual0.z, expected0.z);
    check("callback_mask", actual0.w, expected0.w);
    check("score", actual1.x, expected1.x);
    check("surface_inst", actual1.y, expected1.y);
    check("procedural_inst", actual1.z, expected1.z);
    check("callback_count", actual1.w, expected1.w);
}

void test_hip_ray_query_pipeline(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific ray-query pipeline test on backend '{}'.",
            device.backend_name());
        return;
    }

    auto stream = device.create_stream();

    // Primitive zero in each BLAS is isolated at x=-2 / x=+2. Primitive one
    // in each BLAS overlaps at x=0, with the triangle strictly nearer than the
    // procedural commit distance. Its surface handler rejects the candidate,
    // so both handlers must observe and mutate the same captured state before
    // the procedural hit is committed.
    const std::array vertices{
        make_float3(-2.5f, -0.5f, 0.0f),
        make_float3(-1.5f, -0.5f, 0.0f),
        make_float3(-2.0f, 0.5f, 0.0f),
        make_float3(-0.5f, -0.5f, 0.5f),
        make_float3(0.5f, -0.5f, 0.5f),
        make_float3(0.0f, 0.5f, 0.5f)};
    const std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};
    const std::array aabbs{
        AABB{.packed_min = {1.5f, -0.5f, -2.1f},
             .packed_max = {2.5f, 0.5f, -1.9f}},
        AABB{.packed_min = {-0.5f, -0.5f, -2.1f},
             .packed_max = {0.5f, 0.5f, -1.9f}}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto procedural = device.create_procedural_primitive(aabb_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    accel.emplace_back(procedural);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << aabb_buffer.copy_from(luisa::span{aabbs})
           << mesh.build()
           << procedural.build()
           << accel.build()
           << synchronize();

    auto result0 = device.create_buffer<uint4>(3u);
    auto result1 = device.create_buffer<uint4>(3u);
    auto result2 = device.create_buffer<float>(3u);

    Callable dispatch_width = []() noexcept {
        return dispatch_size().x;
    };
    Callable forwarded_dispatch_width = [&]() noexcept {
        return dispatch_width();
    };

    Kernel1D trace = [&](AccelVar accel, BufferUInt4 result0,
                         BufferUInt4 result1,
                         BufferFloat result2,
                         Float procedural_distance) noexcept {
        auto index = dispatch_id().x;
        auto origin_x = ite(index == 0u, -2.0f,
                            ite(index == 1u, 2.0f, 0.0f));
        auto origin = make_float3(origin_x, 0.0f, 1.0f);
        auto ray = make_ray(origin, make_float3(0.0f, 0.0f, -1.0f));

        // All five variables are local allocas captured by reference by both
        // outlined callbacks. Reads after trace() prove mutations survive the
        // RayQueryPipelineInst call rather than merely reaching the callback.
        UInt callback_mask = 0u;
        UInt callback_count = 0u;
        UInt score = 5u + index;
        UInt surface_inst = ~0u;
        UInt procedural_inst = ~0u;
        Float committed_ray_t_max = -1.0f;

        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     callback_mask = callback_mask | 1u;
                                     callback_count += 1u;
                                     // Keep an implicit Callable ABI input
                                     // observably live through the outlined
                                     // handler. The HIP environment projection
                                     // must retain it through any forwarding
                                     // Callable chain rather than considering
                                     // only explicit captures.
                                     score += 10u + hit.prim +
                                              forwarded_dispatch_width();
                                     surface_inst = hit.inst;
                                     $if (index != 2u) {
                                         candidate.commit();
                                         committed_ray_t_max = candidate.ray()->t_max();
                                     };
                                 })
                             .on_procedural_candidate(
                                 [&](ProceduralCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     callback_mask = callback_mask | 2u;
                                     callback_count += 1u;
                                     score += 20u + hit.prim +
                                              forwarded_dispatch_width();
                                     procedural_inst = hit.inst;
                                     candidate.commit(procedural_distance);
                                     committed_ray_t_max = candidate.ray()->t_max();
                                 })
                             .trace();

        result0.write(index, make_uint4(
                                 committed->hit_type,
                                 committed->inst,
                                 committed->prim,
                                 callback_mask));
        result1.write(index, make_uint4(
                                 score,
                                 surface_inst,
                                 procedural_inst,
                                 callback_count));
        result2.write(index, committed_ray_t_max);
    };

    // This shader naturally needs more than 128 VGPRs on gfx12. Capping it
    // exercises HIP's ShaderOption::max_registers propagation through the
    // linked ray-query call graph as well as correctness under the resulting
    // register allocation/spilling decisions.
    // This is a compiler-pipeline regression, so bypass the persistent cache:
    // every invocation must exercise callback projection and traversal-plan
    // selection rather than merely reloading an earlier code object.
    auto shader = device.compile(
        trace, ShaderOption{.enable_cache = false,
                            .max_registers = 128u});
    std::array<uint4, 3u> host_result0{};
    std::array<uint4, 3u> host_result1{};
    std::array<float, 3u> host_result2{};
    stream << shader(accel, result0, result1, result2, 3.0f).dispatch(3u)
           << result0.copy_to(luisa::span{host_result0})
           << result1.copy_to(luisa::span{host_result1})
           << result2.copy_to(luisa::span{host_result2})
           << synchronize();

    constexpr auto surface = static_cast<uint>(HitType::Surface);
    constexpr auto procedural_hit = static_cast<uint>(HitType::Procedural);
    expect_result(0u, host_result0[0], host_result1[0],
                  make_uint4(surface, 0u, 0u, 1u),
                  make_uint4(18u, 0u, ~0u, 1u));
    expect_result(1u, host_result0[1], host_result1[1],
                  make_uint4(procedural_hit, 1u, 0u, 2u),
                  make_uint4(29u, ~0u, 1u, 1u));
    expect_result(2u, host_result0[2], host_result1[2],
                  make_uint4(procedural_hit, 1u, 1u, 3u),
                  make_uint4(45u, 0u, 1u, 2u));
    expect(std::abs(host_result2[0] - 1.0f) < 1.0e-5f);
    expect(std::abs(host_result2[1] - 3.0f) < 1.0e-5f);
    expect(std::abs(host_result2[2] - 3.0f) < 1.0e-5f);
}

void test_hip_effect_only_native_enumeration(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP native effect-only RayQuery test on backend '{}'.",
            device.backend_name());
        return;
    }

    // Five overlapping candidates exercise both callback domains. The first
    // four callbacks form the observable prefix; terminate() must stop before
    // the fifth without committing any query candidate.
    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.75f),
        make_float3(1.0f, -1.0f, 0.75f),
        make_float3(0.0f, 1.0f, 0.75f),
        make_float3(-1.0f, -1.0f, 0.50f),
        make_float3(1.0f, -1.0f, 0.50f),
        make_float3(0.0f, 1.0f, 0.50f),
        make_float3(-1.0f, -1.0f, 0.25f),
        make_float3(1.0f, -1.0f, 0.25f),
        make_float3(0.0f, 1.0f, 0.25f)};
    const std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u},
        Triangle{6u, 7u, 8u}};
    const std::array aabbs{
        AABB{.packed_min = {-0.5f, -0.5f, 0.60f},
             .packed_max = {0.5f, 0.5f, 0.65f}},
        AABB{.packed_min = {-0.5f, -0.5f, 0.10f},
             .packed_max = {0.5f, 0.5f, 0.15f}}};

    auto stream = device.create_stream();
    auto vertex_buffer =
        device.create_buffer<float3>(vertices.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto procedural = device.create_procedural_primitive(aabb_buffer);
    auto nonopaque_accel = device.create_accel();
    nonopaque_accel.emplace_back(
        mesh, make_float4x4(1.0f), 0xffu, false);
    nonopaque_accel.emplace_back(procedural);
    auto opaque_accel = device.create_accel();
    opaque_accel.emplace_back(
        mesh, make_float4x4(1.0f), 0xffu, true);
    opaque_accel.emplace_back(procedural);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << aabb_buffer.copy_from(luisa::span{aabbs})
           << mesh.build()
           << procedural.build()
           << nonopaque_accel.build()
           << opaque_accel.build()
           << synchronize();

    auto make_trace = [](bool observe_post_state) noexcept {
        return Kernel1D{[observe_post_state](
                            AccelVar accel,
                            BufferUInt4 metadata,
                            BufferUInt sequence,
                            BufferUInt post_state) noexcept {
            UInt callback_count = 0u;
            UInt surface_count = 0u;
            UInt procedural_count = 0u;
            UInt checksum = 0u;
            const auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 2.0f);
            const auto committed =
                accel.traverse(ray, {})
                    .on_surface_candidate(
                        [&](SurfaceCandidate &candidate) noexcept {
                            const auto hit = candidate.hit();
                            const auto code =
                                0x10000000u |
                                (hit->inst << 16u) |
                                hit->prim;
                            sequence.write(callback_count, code);
                            checksum = checksum * 16777619u ^ code;
                            callback_count += 1u;
                            surface_count += 1u;
                            $if (callback_count == 4u) {
                                candidate.terminate();
                            };
                        })
                    .on_procedural_candidate(
                        [&](ProceduralCandidate &candidate) noexcept {
                            const auto hit = candidate.hit();
                            const auto code =
                                0x20000000u |
                                (hit->inst << 16u) |
                                hit->prim;
                            sequence.write(callback_count, code);
                            checksum = checksum * 16777619u ^ code;
                            callback_count += 1u;
                            procedural_count += 1u;
                            $if (callback_count == 4u) {
                                candidate.terminate();
                            };
                        })
                    .trace();
            metadata.write(
                0u, make_uint4(
                        callback_count, surface_count,
                        procedural_count, checksum));
            if (observe_post_state) {
                // This read deliberately makes the otherwise identical query
                // ineligible for effect-only lowering, providing the exact
                // iterative oracle for candidate effects and ordering.
                post_state.write(0u, committed->hit_type);
            } else {
                static_cast<void>(committed);
            }
        }};
    };

    auto native_shader = device.compile(
        make_trace(false), ShaderOption{.enable_cache = false});
    auto exact_shader = device.compile(
        make_trace(true), ShaderOption{.enable_cache = false});
    auto native_metadata = device.create_buffer<uint4>(1u);
    auto exact_metadata = device.create_buffer<uint4>(1u);
    auto native_sequence = device.create_buffer<uint>(8u);
    auto exact_sequence = device.create_buffer<uint>(8u);
    auto post_state = device.create_buffer<uint>(1u);

    auto compare = [&](Accel &accel, bool expect_four_callbacks) noexcept {
        constexpr std::array<uint, 8u> sentinel{
            ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
        std::array<uint4, 1u> host_native_metadata{};
        std::array<uint4, 1u> host_exact_metadata{};
        std::array<uint, 8u> host_native_sequence{};
        std::array<uint, 8u> host_exact_sequence{};
        stream << native_sequence.copy_from(luisa::span{sentinel})
               << exact_sequence.copy_from(luisa::span{sentinel})
               << native_shader(
                      accel, native_metadata, native_sequence,
                      post_state)
                      .dispatch(1u)
               << exact_shader(
                      accel, exact_metadata, exact_sequence,
                      post_state)
                      .dispatch(1u)
               << native_metadata.copy_to(
                      luisa::span{host_native_metadata})
               << exact_metadata.copy_to(
                      luisa::span{host_exact_metadata})
               << native_sequence.copy_to(
                      luisa::span{host_native_sequence})
               << exact_sequence.copy_to(
                      luisa::span{host_exact_sequence})
               << synchronize();
        const auto native_summary = host_native_metadata[0];
        const auto exact_summary = host_exact_metadata[0];
        expect(native_summary.x == exact_summary.x &&
               native_summary.y == exact_summary.y &&
               native_summary.z == exact_summary.z &&
               native_summary.w == exact_summary.w)
            << "native effect-only callback summary differs from exact "
               "RayQueryAll";
        for (auto i = 0u; i < host_native_sequence.size(); ++i) {
            expect(host_native_sequence[i] == host_exact_sequence[i])
                << luisa::format(
                       "effect-only callback {} differs: native=0x{:08x}, "
                       "exact=0x{:08x}",
                       i, host_native_sequence[i],
                       host_exact_sequence[i]);
        }
        if (expect_four_callbacks) {
            expect(host_native_metadata[0].x == 4u)
                << "effect-only terminate boundary did not retain exactly "
                   "four callbacks";
        } else {
            // The opaque mesh bypasses the surface handler. This case also
            // proves that a nonzero accel certificate enters the exact path
            // before any callback side effect is executed.
            expect(host_native_metadata[0].y == 0u)
                << "opaque instance unexpectedly reached the surface "
                   "candidate handler";
        }
    };
    compare(nonopaque_accel, true);
    compare(opaque_accel, false);

    // RayQueryAny has a distinct, formally smaller post-state when only the
    // final hit kind is read. Exercise all three terminal transitions on the
    // same mixed surface/procedural scene: commit -> hit, exhaust -> miss, and
    // explicit terminate without commit -> miss. Callback counters remain
    // externally observable and prove that the quotient neither skips nor
    // replays candidate effects.
    Kernel1D terminal_predicate = [](
                                        AccelVar accel,
                                        BufferUInt4 result) noexcept {
        const auto mode = dispatch_x();
        UInt callback_count = 0u;
        UInt surface_count = 0u;
        UInt procedural_count = 0u;
        const auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 2.0f);
        const auto hit =
            accel.traverse_any(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        callback_count += 1u;
                        surface_count += 1u;
                        $if (mode == 0u) {
                            candidate.commit();
                        }
                        $elif (mode == 2u) {
                            candidate.terminate();
                        };
                    })
                .on_procedural_candidate(
                    [&](ProceduralCandidate &candidate) noexcept {
                        callback_count += 1u;
                        procedural_count += 1u;
                        $if (mode == 0u) {
                            candidate.commit(0.5f);
                        }
                        $elif (mode == 2u) {
                            candidate.terminate();
                        };
                    })
                .trace();
        result.write(
            mode, make_uint4(
                      hit->hit_type, callback_count,
                      surface_count, procedural_count));
    };
    Kernel1D opaque_terminal_predicate = [](
                                               AccelVar accel,
                                               BufferUInt result) noexcept {
        const auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 2.0f);
        const auto hit = accel.traverse_any(ray, {}).trace();
        result.write(0u, hit->hit_type);
    };
    auto terminal_shader = device.compile(
        terminal_predicate, ShaderOption{.enable_cache = false});
    auto opaque_terminal_shader = device.compile(
        opaque_terminal_predicate,
        ShaderOption{.enable_cache = false});
    auto terminal_result = device.create_buffer<uint4>(3u);
    auto opaque_terminal_result = device.create_buffer<uint>(1u);
    std::array<uint4, 3u> host_terminal_result{};
    std::array<uint, 1u> host_opaque_terminal_result{};
    stream << terminal_shader(nonopaque_accel, terminal_result)
                  .dispatch(3u)
           << opaque_terminal_shader(
                  opaque_accel, opaque_terminal_result)
                  .dispatch(1u)
           << terminal_result.copy_to(
                  luisa::span{host_terminal_result})
           << opaque_terminal_result.copy_to(
                  luisa::span{host_opaque_terminal_result})
           << synchronize();

    constexpr auto miss = static_cast<uint>(HitType::Miss);
    constexpr auto surface = static_cast<uint>(HitType::Surface);
    constexpr auto procedural_kind =
        static_cast<uint>(HitType::Procedural);
    expect((host_terminal_result[0].x == surface ||
            host_terminal_result[0].x == procedural_kind) &&
           host_terminal_result[0].y == 1u)
        << "RayQueryAny terminal commit did not publish exactly one hit";
    expect(host_terminal_result[1].x == miss &&
           host_terminal_result[1].y ==
               triangles.size() + aabbs.size())
        << "RayQueryAny terminal predicate did not exhaust every rejected "
           "candidate exactly once";
    expect(host_terminal_result[2].x == miss &&
           host_terminal_result[2].y == 1u)
        << "RayQueryAny explicit terminate did not retain a miss after one "
           "candidate";
    expect(host_opaque_terminal_result[0] == surface)
        << "opaque surface did not auto-commit in the terminal predicate "
           "quotient";
}

void test_hip_ray_query_paired_triangle_resume(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific paired-triangle ray-query test on backend '{}'.",
            device.backend_name());
        return;
    }
    auto amdgpu_arch = device.query("amdgpu_arch");
    if (amdgpu_arch != "gfx1200" && amdgpu_arch != "gfx1201") {
        LUISA_INFO(
            "Skipping gfx12 compact ray-query state test on AMDGPU architecture '{}'.",
            amdgpu_arch);
        return;
    }

    auto stream = device.create_stream();

    // The first two primitives share an edge and are deliberately ordered so
    // that triangle slot zero is farther away than slot one. HIPRT packs them
    // into one triangle pair (a third primitive is required to enable pairing),
    // which exercises the buffered-hit and leaf-resume state in the gfx12 path.
    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(-1.0f, 1.0f, 0.0f),
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(1.0f, 0.0f, 1.0f),
        make_float3(-1.0f, -1.0f, -0.5f),
        make_float3(-1.0f, 1.0f, -0.5f),
        make_float3(1.0f, 0.0f, -0.5f)};
    const std::array triangles{
        Triangle{0u, 1u, 2u}, // far:  t = 1.0
        Triangle{1u, 0u, 3u}, // near: t = 0.5
        Triangle{4u, 5u, 6u}};// later standalone hit: t = 1.5

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    accel.emplace_back(mesh, translation(4.0f, 0.0f, 0.0f), 0xffu, true);
    accel.emplace_back(
        mesh,
        translation(8.0f, 0.0f, 0.0f) * scaling(1.0f, 1.0f, 2.0f),
        0x5au, false);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    constexpr auto case_count = 8u;
    auto metadata = device.create_buffer<uint4>(case_count);
    auto callback_order = device.create_buffer<uint2>(case_count);
    auto committed_detail = device.create_buffer<float4>(case_count);
    auto callback_detail = device.create_buffer<float4>(case_count);
    auto callback_ray_origin_tmin = device.create_buffer<float4>(case_count);
    auto callback_ray_direction_tmax = device.create_buffer<float4>(case_count);

    Kernel1D trace_all = [](AccelVar accel,
                            BufferUInt4 metadata,
                            BufferUInt2 callback_order,
                            BufferFloat4 committed_detail,
                            BufferFloat4 callback_detail,
                            BufferFloat4 callback_ray_origin_tmin,
                            BufferFloat4 callback_ray_direction_tmax) noexcept {
        auto index = dispatch_id().x;
        auto transformed_case = index == 4u;
        auto origin_x = ite(index == 3u, 4.0f,
                            ite(transformed_case, 8.0f, 0.0f));
        auto origin_z = ite(transformed_case, 2.0f, 1.0f);
        auto ray = make_ray(make_float3(origin_x, 0.0f, origin_z),
                            make_float3(0.0f, 0.0f, -1.0f),
                            ite(transformed_case, 0.25f, 0.0f),
                            ite(transformed_case, 4.0f, 2.0f));
        AccelTraceOptions options{
            .visibility_mask = ite(transformed_case, 0x5au, 0xffu)};

        UInt callback_count = 0u;
        UInt first_prim = ~0u;
        UInt second_prim = ~0u;
        Float first_t = -1.0f;
        Float second_t = -1.0f;
        Float first_tmax = -1.0f;
        Float second_tmax = -1.0f;
        Float3 callback_ray_origin = make_float3(-1.0f);
        Float3 callback_ray_direction = make_float3(-1.0f);
        Float callback_ray_tmin = -1.0f;
        Float callback_ray_tmax = -1.0f;
        Float final_callback_tmax = -1.0f;

        auto committed = accel.traverse(ray, options)
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto callback_index = callback_count;
                                     auto callback_ray = candidate.ray();
                                     callback_ray_origin = callback_ray->origin();
                                     callback_ray_direction = callback_ray->direction();
                                     callback_ray_tmin = callback_ray->t_min();
                                     callback_ray_tmax = callback_ray->t_max();
                                     $if (callback_index == 0u) {
                                         first_prim = hit->prim;
                                         first_t = hit->distance();
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_prim = hit->prim;
                                             second_t = hit->distance();
                                         };
                                     };
                                     callback_count += 1u;

                                     // Case zero rejects the near triangle and
                                     // commits the buffered far triangle. Case
                                     // one commits the near hit and must suppress
                                     // the buffered far hit. Case two rejects the
                                     // pair and commits a later standalone leaf,
                                     // proving traversal resumes after pending.
                                     // The opaque case must bypass this callback.
                                     $if (index == 0u) {
                                         $if (hit->prim == 0u) {
                                             candidate.commit();
                                         };
                                     }
                                     $else {
                                         $if ((index == 2u) | transformed_case) {
                                             $if (hit->prim == 2u) {
                                                 candidate.commit();
                                             };
                                         }
                                         $else {
                                             candidate.commit();
                                         };
                                     };

                                     auto tmax = candidate.ray()->t_max();
                                     final_callback_tmax = tmax;
                                     $if (callback_index == 0u) {
                                         first_tmax = tmax;
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_tmax = tmax;
                                         };
                                     };
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();

        metadata.write(index, make_uint4(
                                  committed->hit_type, committed->inst,
                                  committed->prim, callback_count));
        callback_order.write(index, make_uint2(first_prim, second_prim));
        committed_detail.write(index, make_float4(
                                          committed->distance(), committed->bary.x,
                                          committed->bary.y, final_callback_tmax));
        callback_detail.write(index, make_float4(
                                         first_t, second_t,
                                         first_tmax, second_tmax));
        callback_ray_origin_tmin.write(
            index, make_float4(callback_ray_origin, callback_ray_tmin));
        callback_ray_direction_tmax.write(
            index, make_float4(callback_ray_direction, callback_ray_tmax));
    };

    Kernel1D trace_any = [](AccelVar accel,
                            BufferUInt4 metadata,
                            BufferUInt2 callback_order,
                            BufferFloat4 committed_detail,
                            BufferFloat4 callback_detail) noexcept {
        auto index = dispatch_id().x;
        auto output_index = index + 5u;
        auto origin_x = ite(index == 2u, 4.0f, 0.0f);
        auto ray = make_ray(make_float3(origin_x, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);

        UInt callback_count = 0u;
        UInt first_prim = ~0u;
        UInt second_prim = ~0u;
        Float first_t = -1.0f;
        Float second_t = -1.0f;
        Float first_tmax = -1.0f;
        Float second_tmax = -1.0f;

        auto committed = accel.traverse_any(ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto callback_index = callback_count;
                                     $if (callback_index == 0u) {
                                         first_prim = hit->prim;
                                         first_t = hit->distance();
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_prim = hit->prim;
                                             second_t = hit->distance();
                                         };
                                     };
                                     callback_count += 1u;

                                     // Reject near/commit far, explicitly
                                     // terminate on near without a commit, or
                                     // diagnose an illegal opaque callback.
                                     $if (index == 0u) {
                                         $if (hit->prim == 0u) {
                                             candidate.commit();
                                         };
                                     }
                                     $else {
                                         $if (index == 1u) {
                                             candidate.terminate();
                                         }
                                         $else {
                                             candidate.commit();
                                         };
                                     };

                                     auto tmax = candidate.ray()->t_max();
                                     $if (callback_index == 0u) {
                                         first_tmax = tmax;
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_tmax = tmax;
                                         };
                                     };
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();

        metadata.write(output_index, make_uint4(
                                         committed->hit_type, committed->inst,
                                         committed->prim, callback_count));
        callback_order.write(output_index,
                             make_uint2(first_prim, second_prim));
        committed_detail.write(output_index, make_float4(
                                                 committed->distance(), committed->bary.x,
                                                 committed->bary.y, 0.0f));
        callback_detail.write(output_index, make_float4(
                                                first_t, second_t,
                                                first_tmax, second_tmax));
    };

    auto all_shader = device.compile(trace_all);
    auto any_shader = device.compile(trace_any);
    std::array<uint4, case_count> host_metadata{};
    std::array<uint2, case_count> host_callback_order{};
    std::array<float4, case_count> host_committed_detail{};
    std::array<float4, case_count> host_callback_detail{};
    std::array<float4, case_count> host_callback_ray_origin_tmin{};
    std::array<float4, case_count> host_callback_ray_direction_tmax{};
    stream << all_shader(accel, metadata, callback_order,
                         committed_detail, callback_detail,
                         callback_ray_origin_tmin, callback_ray_direction_tmax)
                  .dispatch(5u)
           << any_shader(accel, metadata, callback_order,
                         committed_detail, callback_detail)
                  .dispatch(3u)
           << metadata.copy_to(luisa::span{host_metadata})
           << callback_order.copy_to(luisa::span{host_callback_order})
           << committed_detail.copy_to(luisa::span{host_committed_detail})
           << callback_detail.copy_to(luisa::span{host_callback_detail})
           << callback_ray_origin_tmin.copy_to(
                  luisa::span{host_callback_ray_origin_tmin})
           << callback_ray_direction_tmax.copy_to(
                  luisa::span{host_callback_ray_direction_tmax})
           << synchronize();

    constexpr auto surface = static_cast<uint>(HitType::Surface);
    constexpr auto miss = static_cast<uint>(HitType::Miss);
    const std::array expected_metadata{
        make_uint4(surface, 0u, 0u, 2u),
        make_uint4(surface, 0u, 1u, 1u),
        make_uint4(surface, 0u, 2u, 3u),
        make_uint4(surface, 1u, 1u, 0u),
        make_uint4(surface, 2u, 2u, 3u),
        make_uint4(surface, 0u, 0u, 2u),
        make_uint4(miss, ~0u, ~0u, 1u),
        make_uint4(surface, 1u, 1u, 0u)};
    const std::array expected_callback_order{
        make_uint2(1u, 0u),
        make_uint2(1u, ~0u),
        make_uint2(1u, 0u),
        make_uint2(~0u, ~0u),
        make_uint2(1u, 0u),
        make_uint2(1u, 0u),
        make_uint2(1u, ~0u),
        make_uint2(~0u, ~0u)};
    const std::array expected_committed_detail{
        make_float4(1.0f, 0.25f, 0.5f, 0.0f),
        make_float4(0.5f, 0.25f, 0.5f, 0.0f),
        make_float4(1.5f, 0.25f, 0.5f, 0.0f),
        make_float4(0.5f, 0.25f, 0.5f, 0.0f),
        make_float4(3.0f, 0.25f, 0.5f, 3.0f),
        make_float4(1.0f, 0.25f, 0.5f, 0.0f),
        make_float4(2.0f, 0.0f, 0.0f, 0.0f),
        make_float4(0.5f, 0.25f, 0.5f, 0.0f)};
    const std::array expected_callback_detail{
        make_float4(0.5f, 1.0f, 2.0f, 1.0f),
        make_float4(0.5f, -1.0f, 0.5f, -1.0f),
        make_float4(0.5f, 1.0f, 2.0f, 2.0f),
        make_float4(-1.0f),
        make_float4(1.0f, 2.0f, 4.0f, 4.0f),
        make_float4(0.5f, 1.0f, 2.0f, 1.0f),
        make_float4(0.5f, -1.0f, 2.0f, -1.0f),
        make_float4(-1.0f)};

    auto check_uint = [](uint case_index, luisa::string_view field,
                         uint actual, uint expected) noexcept {
        expect(actual == expected)
            << luisa::format("paired-triangle case {} {}: got {}, expected {}",
                             case_index, field, actual, expected);
    };
    auto check_float = [](uint case_index, luisa::string_view field,
                          float actual, float expected) noexcept {
        expect(std::abs(actual - expected) < 1.0e-5f)
            << luisa::format("paired-triangle case {} {}: got {}, expected {}",
                             case_index, field, actual, expected);
    };
    for (auto i = 0u; i < case_count; i++) {
        auto actual_meta = host_metadata[i];
        auto expected_meta = expected_metadata[i];
        check_uint(i, "hit_type", actual_meta.x, expected_meta.x);
        check_uint(i, "instance", actual_meta.y, expected_meta.y);
        check_uint(i, "primitive", actual_meta.z, expected_meta.z);
        check_uint(i, "callback_count", actual_meta.w, expected_meta.w);

        auto actual_order = host_callback_order[i];
        auto expected_order = expected_callback_order[i];
        check_uint(i, "first_callback_primitive",
                   actual_order.x, expected_order.x);
        check_uint(i, "second_callback_primitive",
                   actual_order.y, expected_order.y);

        // Miss payload distance/barycentrics are not part of the semantic
        // contract; only validate committed detail for actual hits.
        if (actual_meta.x != miss) {
            auto actual_committed = host_committed_detail[i];
            auto expected_committed = expected_committed_detail[i];
            check_float(i, "committed_distance",
                        actual_committed.x, expected_committed.x);
            check_float(i, "committed_bary_u",
                        actual_committed.y, expected_committed.y);
            check_float(i, "committed_bary_v",
                        actual_committed.z, expected_committed.z);
        }

        auto actual_callback = host_callback_detail[i];
        auto expected_callback = expected_callback_detail[i];
        check_float(i, "first_callback_distance",
                    actual_callback.x, expected_callback.x);
        check_float(i, "second_callback_distance",
                    actual_callback.y, expected_callback.y);
        check_float(i, "first_callback_tmax",
                    actual_callback.z, expected_callback.z);
        check_float(i, "second_callback_tmax",
                    actual_callback.w, expected_callback.w);
    }

    // The traversal ray is transformed into BLAS space, but callback accessors
    // must keep exposing the original world-space ray across all three yields.
    constexpr auto transformed_case = 4u;
    const auto transformed_origin_tmin =
        host_callback_ray_origin_tmin[transformed_case];
    const auto transformed_direction_tmax =
        host_callback_ray_direction_tmax[transformed_case];
    check_float(transformed_case, "world_origin_x",
                transformed_origin_tmin.x, 8.0f);
    check_float(transformed_case, "world_origin_y",
                transformed_origin_tmin.y, 0.0f);
    check_float(transformed_case, "world_origin_z",
                transformed_origin_tmin.z, 2.0f);
    check_float(transformed_case, "world_tmin",
                transformed_origin_tmin.w, 0.25f);
    check_float(transformed_case, "world_direction_x",
                transformed_direction_tmax.x, 0.0f);
    check_float(transformed_case, "world_direction_y",
                transformed_direction_tmax.y, 0.0f);
    check_float(transformed_case, "world_direction_z",
                transformed_direction_tmax.z, -1.0f);
    check_float(transformed_case, "world_tmax_before_commit",
                transformed_direction_tmax.w, 4.0f);
    check_float(transformed_case, "world_tmax_after_commit",
                host_committed_detail[transformed_case].w, 3.0f);

    // Opacity is mutable device state, not immutable TLAS metadata. First
    // exercise the terminal AnyHit quotient itself: the near member changes
    // its instance to opaque and rejects. The buffered far member must observe
    // that store, bypass its callback, and auto-commit. Only the final hit kind
    // is read, which also proves that this scenario takes the native terminal
    // route instead of passing accidentally through the exact frontier.
    auto terminal_opacity_result = device.create_buffer<uint4>(1u);
    Kernel1D mutate_opacity_during_terminal_trace = [](
                                                        AccelVar accel,
                                                        BufferUInt4 result) noexcept {
        UInt callback_count = 0u;
        UInt first_primitive = ~0u;
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 2.0f);
        auto committed =
            accel.traverse_any(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        auto hit = candidate.hit();
                        $if (callback_count == 0u) {
                            first_primitive = hit->prim;
                        };
                        callback_count += 1u;
                        accel.set_instance_opaque(hit->inst, true);
                    })
                .on_procedural_candidate(
                    [](ProceduralCandidate &) noexcept {})
                .trace();
        result.write(
            0u,
            make_uint4(
                committed->hit_type, callback_count,
                first_primitive, 0u));
    };
    auto mutate_terminal_opacity_shader = device.compile(
        mutate_opacity_during_terminal_trace,
        ShaderOption{.enable_cache = false});
    std::array<uint4, 1u> host_terminal_opacity_result{};
    stream << mutate_terminal_opacity_shader(
                  accel, terminal_opacity_result)
                  .dispatch(1u)
           << terminal_opacity_result.copy_to(
                  luisa::span{host_terminal_opacity_result})
           << synchronize();
    expect(host_terminal_opacity_result[0].x == surface);
    expect(host_terminal_opacity_result[0].y == 1u);
    expect(host_terminal_opacity_result[0].z == 1u);

    // Restore the initial non-opaque state before checking the exact query
    // below. This host update is intentionally synchronized: the two tests
    // prove distinct state transitions and must not rely on queue ordering to
    // hide a stale opacity flag.
    accel.set_opaque_on_update(0u, false);
    stream << accel.build(AccelBuildRequest::PREFER_UPDATE)
           << synchronize();

    // The near member of the hardware triangle pair is initially non-opaque
    // and reaches
    // the callback, which changes the same instance to opaque without
    // committing. The farther member must then observe that store, bypass its
    // callback, and auto-commit. This guards the exact boundary of the native
    // stable-opacity specialization: any reachable opacity write must select
    // the per-candidate-load variant for the whole kernel.
    auto opacity_result = device.create_buffer<uint4>(1u);
    Kernel1D mutate_opacity_during_trace = [](
                                               AccelVar accel,
                                               BufferUInt4 result) noexcept {
        UInt callback_count = 0u;
        UInt first_primitive = ~0u;
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 2.0f);
        auto committed =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        auto hit = candidate.hit();
                        $if (callback_count == 0u) {
                            first_primitive = hit->prim;
                        };
                        callback_count += 1u;
                        accel.set_instance_opaque(hit->inst, true);
                    })
                .on_procedural_candidate(
                    [](ProceduralCandidate &) noexcept {})
                .trace();
        result.write(
            0u,
            make_uint4(
                committed->hit_type, committed->prim,
                callback_count, first_primitive));
    };
    auto mutate_opacity_shader = device.compile(
        mutate_opacity_during_trace,
        ShaderOption{.enable_cache = false});
    std::array<uint4, 1u> host_opacity_result{};
    stream << mutate_opacity_shader(accel, opacity_result).dispatch(1u)
           << opacity_result.copy_to(luisa::span{host_opacity_result})
           << synchronize();
    expect(host_opacity_result[0].x == surface);
    expect(host_opacity_result[0].y == 0u);
    expect(host_opacity_result[0].z == 1u);
    expect(host_opacity_result[0].w == 1u);
}

void test_ray_query_near_surface_resume(Device &device) {
    if (device.backend_name() != "hip" &&
        device.backend_name() != "fallback") {
        LUISA_INFO(
            "Skipping near-surface ray-query test on backend '{}'.",
            device.backend_name());
        return;
    }

    constexpr auto case_count = 6u;
    constexpr std::array separations{
        0x1p-20f,
        0x1p-16f,
        0x1p-12f,
        1.0e-3f,
        1.0e-2f,
        1.0e-1f};
    constexpr auto direction =
        make_float3(-0.300768f, -0.520893f, 0.798636f);
    constexpr auto instance_offset =
        make_float3(18.0f, 28.0f, 5.0f);

    // Each pair consists of a source triangle and an otherwise identical
    // blocker translated a small positive distance along the ray. The query
    // begins exactly on the source, rejects that exact primitive, and must
    // still expose and commit the blocker. This is the semantic invariant
    // required by explicit primitive self-exclusion: candidate rejection may
    // not advance past any other in-range candidate, including a triangle
    // buffered in the same hardware leaf packet.
    std::array<float3, case_count * 6u> vertices{};
    std::array<Triangle, case_count * 2u> triangles{};
    for (auto i = 0u; i < case_count; i++) {
        const auto center_x = static_cast<float>(i) * 4.0f;
        const std::array source{
            make_float3(center_x - 1.0f, -1.0f, 0.0f),
            make_float3(center_x + 1.0f, -1.0f, 0.0f),
            make_float3(center_x, 1.0f, 0.0f)};
        const auto vertex_base = i * 6u;
        for (auto j = 0u; j < 3u; j++) {
            vertices[vertex_base + j] = source[j];
            vertices[vertex_base + 3u + j] =
                source[j] + separations[i] * direction;
        }
        triangles[i * 2u] = Triangle{
            vertex_base, vertex_base + 1u, vertex_base + 2u};
        triangles[i * 2u + 1u] = Triangle{
            vertex_base + 3u,
            vertex_base + 4u,
            vertex_base + 5u};
    }

    auto stream = device.create_stream();
    auto vertex_buffer =
        device.create_buffer<float3>(vertices.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto separation_buffer =
        device.create_buffer<float>(separations.size());
    auto mesh = device.create_mesh(
        vertex_buffer,
        triangle_buffer,
        AccelOption{.allow_update = true});
    auto accel = device.create_accel();
    accel.emplace_back(
        mesh,
        translation(
            instance_offset.x,
            instance_offset.y,
            instance_offset.z),
        0xffu,
        false);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << separation_buffer.copy_from(luisa::span{separations})
           << mesh.build()
           << accel.build()
           << synchronize();

    auto metadata = device.create_buffer<uint4>(case_count);
    auto callback_order =
        device.create_buffer<uint4>(case_count);
    auto detail = device.create_buffer<float4>(case_count);
    Kernel1D trace = [](
                         AccelVar accel,
                         BufferFloat separations,
                         BufferUInt4 metadata,
                         BufferUInt4 callback_order,
                         BufferFloat4 detail) noexcept {
        const auto index = dispatch_x();
        const auto source_primitive = index * 2u;
        const auto origin =
            make_float3(
                18.0f + cast<float>(index) * 4.0f,
                28.0f - 0.25f,
                5.0f);
        const auto direction =
            make_float3(
                -0.300768f,
                -0.520893f,
                0.798636f);
        const auto ray = make_ray(
            origin, direction, 0.0f, 1.0f);
        UInt callback_count = 0u;
        UInt first_primitive = ~0u;
        UInt second_primitive = ~0u;
        Float first_t = -1.0f;
        Float second_t = -1.0f;
        UInt4 callbacks = make_uint4(~0u);
        auto committed =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        const auto hit = candidate.hit();
                        $if (callback_count == 0u) {
                            first_primitive = hit->prim;
                            first_t = hit->distance();
                            callbacks.x = hit->prim;
                        }
                        $else {
                            $if (callback_count == 1u) {
                                second_primitive = hit->prim;
                                second_t = hit->distance();
                                callbacks.y = hit->prim;
                            }
                            $else {
                                $if (callback_count == 2u) {
                                    callbacks.z = hit->prim;
                                }
                                $else {
                                    $if (callback_count == 3u) {
                                        callbacks.w = hit->prim;
                                    };
                                };
                            };
                        };
                        callback_count += 1u;
                        $if (hit->prim != source_primitive) {
                            candidate.commit();
                        };
                    })
                .on_procedural_candidate(
                    [](ProceduralCandidate &) noexcept {})
                .trace();
        metadata.write(
            index,
            make_uint4(
                committed->hit_type,
                committed->inst,
                committed->prim,
                callback_count));
        callback_order.write(index, callbacks);
        detail.write(
            index,
            make_float4(
                committed->distance(),
                separations.read(index),
                first_t,
                second_t));
    };

    auto shader = device.compile(
        trace, ShaderOption{.enable_cache = false});
    std::array<uint4, case_count> host_metadata{};
    std::array<uint4, case_count> host_callback_order{};
    std::array<float4, case_count> host_detail{};
    stream << shader(
                  accel,
                  separation_buffer,
                  metadata,
                  callback_order,
                  detail)
                  .dispatch(case_count)
           << metadata.copy_to(luisa::span{host_metadata})
           << callback_order.copy_to(
                  luisa::span{host_callback_order})
           << detail.copy_to(luisa::span{host_detail})
           << synchronize();

    constexpr auto surface =
        static_cast<uint>(HitType::Surface);
    for (auto i = 0u; i < case_count; i++) {
        const auto expected_primitive = i * 2u + 1u;
        const auto actual = host_metadata[i];
        expect(actual.x == surface)
            << luisa::format(
                   "near-surface case {} ({}) missed blocker at distance {}",
                   i, device.backend_name(), separations[i]);
        expect(actual.y == 0u)
            << luisa::format(
                   "near-surface case {} ({}) committed instance {}, expected 0",
                   i, device.backend_name(), actual.y);
        expect(actual.z == expected_primitive)
            << luisa::format(
                   "near-surface case {} ({}) committed primitive {}, expected {}",
                   i, device.backend_name(), actual.z,
                   expected_primitive);
        expect(actual.w >= 1u && actual.w <= 2u)
            << luisa::format(
                   "near-surface case {} ({}) observed {} callbacks; "
                   "order [{}, {}, {}, {}], first t {}, second t {}",
                   i, device.backend_name(), actual.w,
                   host_callback_order[i].x,
                   host_callback_order[i].y,
                   host_callback_order[i].z,
                   host_callback_order[i].w,
                   host_detail[i].z, host_detail[i].w);
        const auto tolerance =
            max(2.0e-6f, separations[i] * 2.0e-3f);
        expect(std::abs(host_detail[i].x - separations[i]) <=
               tolerance)
            << luisa::format(
                   "near-surface case {} ({}) committed t {}, expected {}",
                   i, device.backend_name(), host_detail[i].x,
                   separations[i]);
    }
}

void test_ray_query_coincident_surface_resume(Device &device) {
    if (device.backend_name() != "hip" &&
        device.backend_name() != "fallback") {
        LUISA_INFO(
            "Skipping coincident-surface ray-query test on backend '{}'.",
            device.backend_name());
        return;
    }

    // Rejecting one candidate must not discard a distinct candidate at the
    // exact same ray parameter. This is the traversal invariant required by
    // identity-based self exclusion for intentionally coincident geometry:
    // the callback order is unspecified, but both primitives remain separate
    // candidates and the second one must still be eligible for commit.
    constexpr std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    constexpr std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};

    auto stream = device.create_stream();
    auto vertex_buffer =
        device.create_buffer<float3>(vertices.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(
        vertex_buffer,
        triangle_buffer,
        AccelOption{.allow_update = true});
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    auto result_buffer = device.create_buffer<uint4>(1u);
    auto distance_buffer = device.create_buffer<float>(1u);
    Kernel1D trace = [](
                         AccelVar accel,
                         BufferUInt4 result,
                         BufferFloat distance) noexcept {
        const auto ray = make_ray(
            make_float3(0.0f, -0.25f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f,
            2.0f);
        UInt callback_count = 0u;
        UInt first_primitive = ~0u;
        UInt second_primitive = ~0u;
        auto committed =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        const auto hit = candidate.hit();
                        $if (callback_count == 0u) {
                            first_primitive = hit->prim;
                        }
                        $else {
                            second_primitive = hit->prim;
                            candidate.commit();
                            candidate.terminate();
                        };
                        callback_count += 1u;
                    })
                .on_procedural_candidate(
                    [](ProceduralCandidate &) noexcept {})
                .trace();
        result.write(
            0u,
            make_uint4(
                committed->hit_type,
                committed->prim,
                callback_count,
                select(
                    0u,
                    1u,
                    first_primitive == second_primitive)));
        distance.write(0u, committed->distance());
    };

    auto shader = device.compile(
        trace, ShaderOption{.enable_cache = false});
    std::array<uint4, 1u> result{};
    std::array<float, 1u> distance{};
    stream << shader(
                  accel, result_buffer, distance_buffer)
                  .dispatch(1u)
           << result_buffer.copy_to(luisa::span{result})
           << distance_buffer.copy_to(luisa::span{distance})
           << synchronize();

    constexpr auto surface =
        static_cast<uint>(HitType::Surface);
    expect(result[0].x == surface)
        << luisa::format(
               "coincident query ({}) did not commit the sibling",
               device.backend_name());
    expect(result[0].y < 2u);
    expect(result[0].z == 2u)
        << luisa::format(
               "coincident query ({}) observed {} callbacks",
               device.backend_name(), result[0].z);
    expect(result[0].w == 0u);
    expect(std::abs(distance[0] - 1.0f) < 1.0e-6f);
}

void test_ray_query_reconstructed_coincident_surface(Device &device) {
    if (device.backend_name() != "hip" &&
        device.backend_name() != "fallback") {
        LUISA_INFO(
            "Skipping reconstructed coincident-surface test on backend '{}'.",
            device.backend_name());
        return;
    }

    constexpr std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    constexpr std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};

    auto stream = device.create_stream();
    auto vertex_buffer =
        device.create_buffer<float3>(vertices.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(
        vertex_buffer,
        triangle_buffer,
        AccelOption{.allow_update = true});
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    auto result_buffer = device.create_buffer<uint4>(1u);
    auto detail_buffer = device.create_buffer<float4>(1u);
    auto callback_tmin_buffer = device.create_buffer<float>(1u);
    Kernel1D trace = [](
                         AccelVar accel,
                         BufferUInt4 result,
                         BufferFloat4 detail,
                         BufferFloat callback_tmin_output) noexcept {
        const auto primary_ray = make_ray(
            make_float3(0.125f, -0.25f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f,
            2.0f);
        auto primary =
            accel.traverse(primary_ray, {})
                .on_surface_candidate(
                    [](SurfaceCandidate &candidate) noexcept {
                        candidate.commit();
                    })
                .on_procedural_candidate(
                    [](ProceduralCandidate &) noexcept {})
                .trace();
        const auto p0 =
            make_float3(-1.0f, -1.0f, 0.0f);
        const auto p1 =
            make_float3(1.0f, -1.0f, 0.0f);
        const auto p2 =
            make_float3(0.0f, 1.0f, 0.0f);
        const auto origin =
            p0 +
            primary->bary.x * (p1 - p0) +
            primary->bary.y * (p2 - p0);
        const auto shadow_ray = make_ray(
            origin,
            make_float3(0.0f, 0.0f, 1.0f),
            0.0f,
            1.0f);
        UInt callback_count = 0u;
        Float callback_tmin = -1.0f;
        auto sibling =
            accel.traverse(shadow_ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        const auto hit = candidate.hit();
                        callback_tmin = candidate.ray()->t_min();
                        callback_count += 1u;
                        $if (hit->prim != primary->prim) {
                            candidate.commit();
                            candidate.terminate();
                        };
                    })
                .on_procedural_candidate(
                    [](ProceduralCandidate &) noexcept {})
                .trace();
        result.write(
            0u,
            make_uint4(
                primary->hit_type,
                primary->prim,
                sibling->hit_type,
                callback_count));
        detail.write(
            0u,
            make_float4(
                primary->bary,
                origin.z,
                sibling->distance()));
        callback_tmin_output.write(0u, callback_tmin);
    };

    auto shader = device.compile(
        trace, ShaderOption{.enable_cache = false});
    std::array<uint4, 1u> result{};
    std::array<float4, 1u> detail{};
    std::array<float, 1u> callback_tmin{};
    stream << shader(
                  accel, result_buffer, detail_buffer,
                  callback_tmin_buffer)
                  .dispatch(1u)
           << result_buffer.copy_to(luisa::span{result})
           << detail_buffer.copy_to(luisa::span{detail})
           << callback_tmin_buffer.copy_to(
                  luisa::span{callback_tmin})
           << synchronize();

    constexpr auto surface =
        static_cast<uint>(HitType::Surface);
    expect(result[0].x == surface);
    expect(result[0].y < 2u);
    expect(result[0].z == surface)
        << luisa::format(
               "reconstructed coincident query ({}) missed sibling; "
               "callbacks {}, source {}, bary ({}, {}), origin z {}",
               device.backend_name(), result[0].w, result[0].y,
               detail[0].x, detail[0].y, detail[0].z);
    expect(result[0].w >= 1u);
    expect(std::abs(detail[0].z) < 1.0e-7f);
    expect(std::abs(detail[0].w) < 1.0e-7f);
    expect(callback_tmin[0] == 0.0f)
        << luisa::format(
               "reconstructed coincident query ({}) exposed internal tmin {}",
               device.backend_name(), callback_tmin[0]);
}

void test_hip_ray_query_any_automatic_termination(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific ANY ray-query termination test on backend '{}'.",
            device.backend_name());
        return;
    }
    auto amdgpu_arch = device.query("amdgpu_arch");
    if (amdgpu_arch != "gfx1200" && amdgpu_arch != "gfx1201") {
        LUISA_INFO(
            "Skipping gfx12 compact ray-query termination test on AMDGPU architecture '{}'.",
            amdgpu_arch);
        return;
    }

    auto stream = device.create_stream();
    const std::array aabbs{
        AABB{.packed_min = {-1.0f, -1.0f, 0.5f},
             .packed_max = {1.0f, 1.0f, 1.5f}},
        AABB{.packed_min = {-1.0f, -1.0f, 0.5f},
             .packed_max = {1.0f, 1.0f, 1.5f}}};
    auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
    auto procedural = device.create_procedural_primitive(aabb_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(procedural);
    stream << aabb_buffer.copy_from(luisa::span{aabbs})
           << procedural.build()
           << accel.build()
           << synchronize();

    auto metadata = device.create_buffer<uint4>(4u);
    auto callback_order = device.create_buffer<uint2>(4u);
    auto detail = device.create_buffer<float2>(4u);

    Kernel1D trace_all = [](AccelVar accel, BufferUInt4 metadata,
                            BufferUInt2 callback_order,
                            BufferFloat2 detail) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 2.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 3.0f);
        UInt callback_count = 0u;
        UInt first_prim = ~0u;
        UInt second_prim = ~0u;
        Float final_tmax = -1.0f;
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [](SurfaceCandidate &) noexcept {})
                             .on_procedural_candidate(
                                 [&](ProceduralCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto callback_index = callback_count;
                                     $if (callback_index == 0u) {
                                         first_prim = hit->prim;
                                         candidate.commit(1.5f);
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_prim = hit->prim;
                                         };
                                         candidate.commit(1.0f);
                                     };
                                     callback_count += 1u;
                                     final_tmax = candidate.ray()->t_max();
                                 })
                             .trace();
        metadata.write(0u, make_uint4(
                               committed->hit_type, committed->inst,
                               committed->prim, callback_count));
        callback_order.write(0u, make_uint2(first_prim, second_prim));
        detail.write(0u, make_float2(committed->distance(), final_tmax));
    };

    Kernel1D trace_any = [](AccelVar accel, BufferUInt4 metadata,
                            BufferUInt2 callback_order,
                            BufferFloat2 detail) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 2.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 3.0f);
        UInt callback_count = 0u;
        UInt first_prim = ~0u;
        UInt second_prim = ~0u;
        Float final_tmax = -1.0f;
        auto committed = accel.traverse_any(ray, {})
                             .on_surface_candidate(
                                 [](SurfaceCandidate &) noexcept {})
                             .on_procedural_candidate(
                                 [&](ProceduralCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto callback_index = callback_count;
                                     $if (callback_index == 0u) {
                                         first_prim = hit->prim;
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_prim = hit->prim;
                                         };
                                     };
                                     callback_count += 1u;
                                     candidate.commit(1.5f);
                                     final_tmax = candidate.ray()->t_max();
                                 })
                             .trace();
        metadata.write(1u, make_uint4(
                               committed->hit_type, committed->inst,
                               committed->prim, callback_count));
        callback_order.write(1u, make_uint2(first_prim, second_prim));
        detail.write(1u, make_float2(committed->distance(), final_tmax));
    };

    Kernel1D trace_compact_transactions = [](
                                                  AccelVar accel,
                                                  BufferUInt4 metadata,
                                                  BufferUInt2 callback_order,
                                                  BufferFloat2 detail) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 2.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 3.0f);
        UInt all_count = 0u;
        UInt all_first = ~0u;
        UInt all_second = ~0u;
        auto all = accel.traverse(ray, {})
                       .on_surface_candidate(
                           [](SurfaceCandidate &) noexcept {})
                       .on_procedural_candidate(
                           [&](ProceduralCandidate &candidate) noexcept {
                               auto hit = candidate.hit();
                               $if (all_count == 0u) {
                                   all_first = hit->prim;
                                   candidate.commit(1.5f);
                               }
                               $else {
                                   all_second = hit->prim;
                                   candidate.commit(1.0f);
                                   candidate.terminate();
                               };
                               all_count += 1u;
                           })
                       .trace();
        metadata.write(2u, make_uint4(
                               all->hit_type, all->inst,
                               all->prim, all_count));
        callback_order.write(2u, make_uint2(all_first, all_second));
        detail.write(2u, make_float2(all->distance(), 0.0f));

        UInt any_count = 0u;
        UInt any_first = ~0u;
        auto any = accel.traverse_any(ray, {})
                       .on_surface_candidate(
                           [](SurfaceCandidate &) noexcept {})
                       .on_procedural_candidate(
                           [&](ProceduralCandidate &candidate) noexcept {
                               any_first = candidate.hit()->prim;
                               any_count += 1u;
                               candidate.commit(1.5f);
                           })
                       .trace();
        metadata.write(3u, make_uint4(
                               any->hit_type, any->inst,
                               any->prim, any_count));
        callback_order.write(3u, make_uint2(any_first, ~0u));
        detail.write(3u, make_float2(any->distance(), 0.0f));
    };

    auto all_shader = device.compile(trace_all);
    auto any_shader = device.compile(trace_any);
    auto compact_shader = device.compile(
        trace_compact_transactions,
        ShaderOption{.enable_cache = false});
    std::array<uint4, 4u> host_metadata{};
    std::array<uint2, 4u> host_callback_order{};
    std::array<float2, 4u> host_detail{};
    stream << all_shader(accel, metadata, callback_order, detail).dispatch(1u)
           << any_shader(accel, metadata, callback_order, detail).dispatch(1u)
           << compact_shader(accel, metadata, callback_order, detail)
                  .dispatch(1u)
           << metadata.copy_to(luisa::span{host_metadata})
           << callback_order.copy_to(luisa::span{host_callback_order})
           << detail.copy_to(luisa::span{host_detail})
           << synchronize();

    constexpr auto procedural_hit = static_cast<uint>(HitType::Procedural);
    expect(host_metadata[0].x == procedural_hit);
    expect(host_metadata[0].y == 0u);
    expect(host_metadata[0].w == 2u)
        << "ALL query must continue to the second procedural candidate";
    expect(host_callback_order[0].x < 2u);
    expect(host_callback_order[0].y < 2u);
    expect(host_callback_order[0].x != host_callback_order[0].y);
    expect(host_metadata[0].z == host_callback_order[0].y)
        << "ALL query must retain the second, closer procedural commit";
    expect(std::abs(host_detail[0].x - 1.0f) < 1.0e-5f);
    expect(std::abs(host_detail[0].y - 1.0f) < 1.0e-5f);

    expect(host_metadata[1].x == procedural_hit);
    expect(host_metadata[1].y == 0u);
    expect(host_metadata[1].w == 1u)
        << "ANY commit must terminate before the second eligible candidate";
    expect(host_callback_order[1].x < 2u);
    expect(host_callback_order[1].y == ~0u);
    expect(host_metadata[1].z == host_callback_order[1].x);
    expect(std::abs(host_detail[1].x - 1.5f) < 1.0e-5f);
    expect(std::abs(host_detail[1].y - 1.5f) < 1.0e-5f);

    // These two handlers never observe committed state or the world ray.
    // They therefore exercise the compact scalar action ABI, including a
    // procedural distance return, explicit termination, and RayQueryAny's
    // implicit termination on commit.
    expect(host_metadata[2].x == procedural_hit);
    expect(host_metadata[2].y == 0u);
    expect(host_metadata[2].w == 2u);
    expect(host_callback_order[2].x < 2u);
    expect(host_callback_order[2].y < 2u);
    expect(host_callback_order[2].x != host_callback_order[2].y);
    expect(host_metadata[2].z == host_callback_order[2].y);
    expect(std::abs(host_detail[2].x - 1.0f) < 1.0e-5f);

    expect(host_metadata[3].x == procedural_hit);
    expect(host_metadata[3].y == 0u);
    expect(host_metadata[3].w == 1u);
    expect(host_callback_order[3].x < 2u);
    expect(host_callback_order[3].y == ~0u);
    expect(host_metadata[3].z == host_callback_order[3].x);
    expect(std::abs(host_detail[3].x - 1.5f) < 1.0e-5f);
}

void test_hip_ray_query_reentrant_handler_trace(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific reentrant ray-query handler test on backend '{}'.",
            device.backend_name());
        return;
    }

    // A synchronous candidate handler is allowed to issue an ordinary trace.
    // The outer query's gfx12 frontier lives in lane-local LDS, so executing a
    // second hardware traversal there would overwrite it. The backend must
    // detect this capability through the handler call graph and select its
    // reentrant software-stack path before code generation.
    const std::array vertices{
        make_float3(-0.5f, -0.5f, 0.0f),
        make_float3(0.5f, -0.5f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f),
        make_float3(1.5f, -0.5f, 0.0f),
        make_float3(2.5f, -0.5f, 0.0f),
        make_float3(2.0f, 0.5f, 0.0f)};
    const std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};

    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    auto result = device.create_buffer<uint2>(1u);
    Kernel1D trace = [](AccelVar accel, BufferUInt2 result) noexcept {
        auto outer_ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f));
        UInt nested_hit = 0u;
        auto committed = accel.traverse(outer_ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto nested_ray = make_ray(
                                         make_float3(2.0f, 0.0f, 1.0f),
                                         make_float3(0.0f, 0.0f, -1.0f));
                                     nested_hit = ite(
                                         accel.intersect_any(nested_ray, {}),
                                         1u, 0u);
                                     candidate.commit();
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();
        result.write(0u, make_uint2(
                             committed->hit_type, nested_hit));
    };

    auto shader = device.compile(trace);
    std::array<uint2, 1u> host_result{};
    stream << shader(accel, result).dispatch(1u)
           << result.copy_to(luisa::span{host_result})
           << synchronize();
    expect(host_result[0].x == static_cast<uint>(HitType::Surface));
    expect(host_result[0].y == 1u);
}

void test_hip_ray_query_short_stack_backtracking(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific short-stack ray-query test on backend '{}'.",
            device.backend_name());
        return;
    }
    auto amdgpu_arch = device.query("amdgpu_arch");
    if (amdgpu_arch != "gfx1200" && amdgpu_arch != "gfx1201") {
        LUISA_INFO(
            "Skipping gfx12 short-stack ray-query test on AMDGPU architecture '{}'.",
            amdgpu_arch);
        return;
    }

    // GFX12's ds_bvh_stack instruction uses a deliberately bounded LDS stack
    // in the HIP backend. The effectful query below must use exact parent-link
    // traversal and observe every candidate exactly once. A second query must
    // still find a deliberately late accepted candidate in both a deep BLAS
    // and a deep TLAS.
    constexpr auto blas_primitive_count = 1024u;
    constexpr auto tlas_instance_count = 256u;
    const std::array base_triangle{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};

    std::vector<float3> blas_vertices;
    std::vector<Triangle> blas_triangles;
    blas_vertices.reserve(blas_primitive_count * 3u);
    blas_triangles.reserve(blas_primitive_count);
    for (auto i = 0u; i < blas_primitive_count; i++) {
        const auto vertex_base = i * 3u;
        blas_vertices.insert(blas_vertices.end(),
                             base_triangle.begin(), base_triangle.end());
        blas_triangles.emplace_back(
            vertex_base, vertex_base + 1u, vertex_base + 2u);
    }

    auto stream = device.create_stream();
    auto blas_vertex_buffer =
        device.create_buffer<float3>(blas_vertices.size());
    auto blas_triangle_buffer =
        device.create_buffer<Triangle>(blas_triangles.size());
    auto deep_mesh =
        device.create_mesh(blas_vertex_buffer, blas_triangle_buffer);
    auto deep_blas_accel = device.create_accel();
    deep_blas_accel.emplace_back(
        deep_mesh, make_float4x4(1.0f), 0xffu, false);

    auto single_vertex_buffer =
        device.create_buffer<float3>(base_triangle.size());
    const std::array single_triangle{Triangle{0u, 1u, 2u}};
    auto single_triangle_buffer =
        device.create_buffer<Triangle>(single_triangle.size());
    auto single_mesh =
        device.create_mesh(single_vertex_buffer, single_triangle_buffer);
    auto deep_tlas_accel = device.create_accel();
    for (auto i = 0u; i < tlas_instance_count; i++) {
        deep_tlas_accel.emplace_back(
            single_mesh, make_float4x4(1.0f), 0xffu, false);
    }

    stream << blas_vertex_buffer.copy_from(luisa::span{blas_vertices})
           << blas_triangle_buffer.copy_from(luisa::span{blas_triangles})
           << single_vertex_buffer.copy_from(luisa::span{base_triangle})
           << single_triangle_buffer.copy_from(luisa::span{single_triangle})
           << deep_mesh.build()
           << single_mesh.build()
           << deep_blas_accel.build()
           << deep_tlas_accel.build()
           << synchronize();

    auto blas_visits = device.create_buffer<uint>(blas_primitive_count);
    auto tlas_visits = device.create_buffer<uint>(tlas_instance_count);
    auto metadata = device.create_buffer<uint2>(2u);
    auto late_commit_result = device.create_buffer<uint4>(2u);
    std::vector<uint> blas_zeros(blas_primitive_count, 0u);
    std::vector<uint> tlas_zeros(tlas_instance_count, 0u);

    Kernel1D trace_all = [](AccelVar accel, BufferUInt visits,
                            BufferUInt2 metadata, UInt key_by_instance,
                            UInt output_index) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        UInt callback_count = 0u;
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto key = ite(
                                         key_by_instance != 0u,
                                         hit->inst, hit->prim);
                                     visits.write(key, visits.read(key) + 1u);
                                     callback_count += 1u;
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();
        metadata.write(output_index,
                       make_uint2(committed->hit_type, callback_count));
    };

    auto shader = device.compile(trace_all);
    Kernel1D trace_late_commit = [](
                                     AccelVar accel,
                                     BufferUInt4 result,
                                     UInt key_by_instance,
                                     UInt target,
                                     UInt output_index) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto key = ite(
                                         key_by_instance != 0u,
                                         hit->inst, hit->prim);
                                     $if (key == target) {
                                         candidate.commit();
                                     };
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();
        result.write(output_index,
                     make_uint4(committed->hit_type,
                                committed->inst,
                                committed->prim, 0u));
    };
    auto late_commit_shader = device.compile(trace_late_commit);
    std::array<uint2, 2u> host_metadata{};
    std::array<uint4, 2u> host_late_commit_result{};
    std::vector<uint> host_blas_visits(blas_primitive_count);
    std::vector<uint> host_tlas_visits(tlas_instance_count);
    stream << blas_visits.copy_from(luisa::span{blas_zeros})
           << tlas_visits.copy_from(luisa::span{tlas_zeros})
           << shader(deep_blas_accel, blas_visits, metadata, 0u, 0u)
                  .dispatch(1u)
           << shader(deep_tlas_accel, tlas_visits, metadata, 1u, 1u)
                  .dispatch(1u)
           << late_commit_shader(deep_blas_accel, late_commit_result, 0u,
                                 blas_primitive_count - 1u, 0u)
                  .dispatch(1u)
           << late_commit_shader(deep_tlas_accel, late_commit_result, 1u,
                                 tlas_instance_count - 1u, 1u)
                  .dispatch(1u)
           << metadata.copy_to(luisa::span{host_metadata})
           << late_commit_result.copy_to(
                  luisa::span{host_late_commit_result})
           << blas_visits.copy_to(luisa::span{host_blas_visits})
           << tlas_visits.copy_to(luisa::span{host_tlas_visits})
           << synchronize();

    constexpr auto miss = static_cast<uint>(HitType::Miss);
    expect(host_metadata[0].x == miss);
    expect(host_metadata[0].y == blas_primitive_count);
    expect(host_metadata[1].x == miss);
    expect(host_metadata[1].y == tlas_instance_count);
    constexpr auto surface = static_cast<uint>(HitType::Surface);
    expect(host_late_commit_result[0].x == surface);
    expect(host_late_commit_result[0].y == 0u);
    expect(host_late_commit_result[0].z == blas_primitive_count - 1u);
    expect(host_late_commit_result[1].x == surface);
    expect(host_late_commit_result[1].y == tlas_instance_count - 1u);
    expect(host_late_commit_result[1].z == 0u);
    for (auto i = 0u; i < blas_primitive_count; i++) {
        expect(host_blas_visits[i] == 1u)
            << luisa::format(
                   "BLAS primitive {} visited {} times after short-stack overflow",
                   i, host_blas_visits[i]);
    }
    for (auto i = 0u; i < tlas_instance_count; i++) {
        expect(host_tlas_visits[i] == 1u)
            << luisa::format(
                   "TLAS instance {} visited {} times after short-stack overflow",
                   i, host_tlas_visits[i]);
    }
}

void test_hip_ray_query_large_mixed_dispatch(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific large mixed ray-query test on backend '{}'.",
            device.backend_name());
        return;
    }
    auto amdgpu_arch = device.query("amdgpu_arch");
    if (amdgpu_arch != "gfx1200" && amdgpu_arch != "gfx1201") {
        LUISA_INFO(
            "Skipping gfx12 mixed ray-query dispatch test on architecture '{}'.",
            amdgpu_arch);
        return;
    }

    constexpr auto resolution = make_uint2(1024u, 1024u);
    constexpr auto element_count =
        static_cast<size_t>(resolution.x) * resolution.y;
    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};

    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    auto input_0 = device.create_buffer<uint>(1u);
    auto input_1 = device.create_buffer<uint>(1u);
    auto input_2 = device.create_buffer<uint>(1u);
    auto input_3 = device.create_buffer<uint>(1u);
    auto input_4 = device.create_buffer<uint>(1u);
    auto input_5 = device.create_buffer<uint>(1u);
    auto input_6 = device.create_buffer<uint>(1u);
    auto input_7 = device.create_buffer<uint>(1u);
    auto input_8 = device.create_buffer<uint>(1u);
    const std::array<uint, 9u> input_values{
        1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << input_0.copy_from(input_values.data() + 0u)
           << input_1.copy_from(input_values.data() + 1u)
           << input_2.copy_from(input_values.data() + 2u)
           << input_3.copy_from(input_values.data() + 3u)
           << input_4.copy_from(input_values.data() + 4u)
           << input_5.copy_from(input_values.data() + 5u)
           << input_6.copy_from(input_values.data() + 6u)
           << input_7.copy_from(input_values.data() + 7u)
           << input_8.copy_from(input_values.data() + 8u)
           << mesh.build()
           << accel.build()
           << synchronize();

    auto result = device.create_buffer<uint>(element_count);
    Kernel2D trace = [](AccelVar accel, BufferUInt result,
                        BufferUInt input_0, BufferUInt input_1,
                        BufferUInt input_2, BufferUInt input_3,
                        BufferUInt input_4, BufferUInt input_5,
                        BufferUInt input_6, BufferUInt input_7,
                        BufferUInt input_8) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel = dispatch_id().xy();
        auto index = pixel.x + pixel.y * dispatch_size().x;
        auto origin_x = ite((pixel.x & 1u) == 0u, 0.0f, 4.0f);
        auto ray = make_ray(
            make_float3(origin_x, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f));
        UInt callback_checksum = 0u;
        auto closest =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        callback_checksum =
                            input_0.read(0u) ^ input_1.read(0u) ^
                            input_2.read(0u) ^ input_3.read(0u) ^
                            input_4.read(0u) ^ input_5.read(0u) ^
                            input_6.read(0u) ^ input_7.read(0u) ^
                            input_8.read(0u) ^
                            cast<uint>(candidate.ray()->t_max() > 0.0f);
                        candidate.commit();
                    })
                .trace();
        auto any = accel.traverse_any(ray, {})
                       .on_surface_candidate(
                           [](SurfaceCandidate &candidate) noexcept {
                               candidate.commit();
                           })
                       .trace();
        auto encoded = ite(closest->miss(), 0u, 1u) |
                       ite(any->miss(), 0u, 2u) |
                       (callback_checksum << 2u);
        result.write(index, encoded);
    };
    auto shader = device.compile(
        trace, ShaderOption{.enable_cache = false});
    std::vector<uint> host_result(element_count);
    stream << shader(accel, result,
                     input_0, input_1, input_2,
                     input_3, input_4, input_5,
                     input_6, input_7, input_8)
                  .dispatch(resolution)
           << result.copy_to(host_result.data())
           << synchronize();

    auto mismatch_count = size_t{0u};
    auto first_mismatch = size_t{0u};
    for (auto i = size_t{0u}; i < host_result.size(); ++i) {
        const auto x = i % resolution.x;
        const auto expected = (x & 1u) == 0u ? 3u : 0u;
        if (host_result[i] != expected) {
            if (mismatch_count == 0u) { first_mismatch = i; }
            ++mismatch_count;
        }
    }
    expect(mismatch_count == 0u)
        << luisa::format(
               "large mixed ray-query dispatch produced {} mismatch(es); "
               "first index = {}, value = {}",
               mismatch_count, first_mismatch,
               host_result[first_mismatch]);
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP ray-query pipeline captures and commits"_test = [&] {
        test_hip_ray_query_pipeline(dc->device);
    };
    "HIP ray-query effect-only native enumeration"_test = [&] {
        test_hip_effect_only_native_enumeration(dc->device);
    };
    "HIP ray-query paired-triangle resume state"_test = [&] {
        test_hip_ray_query_paired_triangle_resume(dc->device);
    };
    "ray-query resumes after exact source exclusion"_test = [&] {
        test_ray_query_near_surface_resume(dc->device);
    };
    "ray-query resumes at a coincident sibling"_test = [&] {
        test_ray_query_coincident_surface_resume(dc->device);
    };
    "ray-query sees reconstructed coincident sibling"_test = [&] {
        test_ray_query_reconstructed_coincident_surface(dc->device);
    };
    "HIP ray-query ANY commit terminates automatically"_test = [&] {
        test_hip_ray_query_any_automatic_termination(dc->device);
    };
    "HIP ray-query handler traces are reentrant"_test = [&] {
        test_hip_ray_query_reentrant_handler_trace(dc->device);
    };
    "HIP ray-query short-stack backtracking is exact"_test = [&] {
        test_hip_ray_query_short_stack_backtracking(dc->device);
    };
    "HIP ray-query large mixed dispatch is exact"_test = [&] {
        test_hip_ray_query_large_mixed_dispatch(dc->device);
    };
}
