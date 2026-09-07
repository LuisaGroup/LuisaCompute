// Test for BLAS/TLAS lifetime coupling between Mesh (BLAS) and Accel (TLAS).
// This test covers:
// - two meshes sharing one triangle buffer in two separate accels, built in one batch
// - BLAS recreate (FORCE_BUILD) while referenced by a TLAS, then TLAS rebuild and trace
// - destroying a referenced mesh, replacing the instance with a new mesh, and rebuilding
// - teardown order with the mesh destroyed before the accel
// - explicit instance replacement racing a pending BLAS-recreate refresh entry
// - repeated in-place BLAS updates interleaved with TLAS update builds
// - motion-enabled BLAS recreate / partial updates preserving untouched instances (vk)

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <optional>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A triangle in the z=0 plane, centered at x, large enough for the test rays.
[[nodiscard]] auto make_triangle_vertices(float x) noexcept {
    return std::array{
        make_float3(x - 0.5f, -0.5f, 0.0f),
        make_float3(x + 0.5f, -0.5f, 0.0f),
        make_float3(x, 0.5f, 0.0f)};
}

// Rays shot straight down (-z) at x = -1 and x = +1.
[[nodiscard]] auto make_ray_origins() noexcept {
    return std::array{make_float3(-1.0f, 0.0f, 1.0f),
                      make_float3(+1.0f, 0.0f, 1.0f)};
}

struct TraceContext {
    Buffer<float3> origin_buffer;
    Buffer<uint> result_buffer;
    Shader1D<Accel, Buffer<float3>, Buffer<uint>> shader;
};

TraceContext make_trace_context(Device &device) {
    TraceContext ctx;
    ctx.origin_buffer = device.create_buffer<float3>(2u);
    ctx.result_buffer = device.create_buffer<uint>(2u);
    auto origins = make_ray_origins();
    Stream stream = device.create_stream();
    stream << ctx.origin_buffer.copy_from(luisa::span{origins}) << synchronize();
    Kernel1D trace_kernel = [](AccelVar accel,
                               BufferFloat3 origins,
                               BufferUInt results) noexcept {
        auto i = dispatch_id().x;
        auto ray = make_ray(origins.read(i), make_float3(0.0f, 0.0f, -1.0f));
        auto hit = accel.intersect(ray, {});
        results.write(i, ite(hit->miss(), 0u, 1u));
    };
    ctx.shader = device.compile(trace_kernel);
    return ctx;
}

// Trace the two test rays against `accel` and return per-ray hit flags.
[[nodiscard]] std::array<uint, 2u> trace_hits(Device &device, Stream &stream,
                                              const TraceContext &ctx, Accel &accel) {
    std::array<uint, 2u> hits{};
    stream << ctx.shader(accel, ctx.origin_buffer, ctx.result_buffer).dispatch(2u)
           << ctx.result_buffer.copy_to(luisa::span{hits})
           << synchronize();
    return hits;
}

void expect_hits(const std::array<uint, 2u> &hits, uint expect_neg_x, uint expect_pos_x,
                 luisa::string_view phase) {
    expect(hits[0] == expect_neg_x) << luisa::format(
        "{}: ray at x=-1 hit={}, expected {}", phase, hits[0], expect_neg_x);
    expect(hits[1] == expect_pos_x) << luisa::format(
        "{}: ray at x=+1 hit={}, expected {}", phase, hits[1], expect_pos_x);
}

// Two meshes sharing one triangle buffer, each wrapped in its own accel, all
// built in a single batched stream submission (the "UV-space BLAS" pattern of
// the AO baker: world mesh + UV mesh share the index buffer).
void test_shared_index_buffer_two_accels(Device &device) {
    Stream stream = device.create_stream();
    auto ctx = make_trace_context(device);

    auto vertices_a = make_triangle_vertices(-1.0f);
    auto vertices_b = make_triangle_vertices(+1.0f);
    const std::array triangles{Triangle{0u, 1u, 2u}};

    auto vb_a = device.create_buffer<float3>(vertices_a.size());
    auto vb_b = device.create_buffer<float3>(vertices_b.size());
    auto tb = device.create_buffer<Triangle>(triangles.size());
    auto mesh_a = device.create_mesh(vb_a, tb);
    auto mesh_b = device.create_mesh(vb_b, tb);
    Accel accel_a = device.create_accel({});
    Accel accel_b = device.create_accel({});
    accel_a.emplace_back(mesh_a, make_float4x4(1.0f));
    accel_b.emplace_back(mesh_b, make_float4x4(1.0f));

    // One batch: uploads, both BLAS builds, both TLAS builds, no sync in between.
    stream << vb_a.copy_from(luisa::span{vertices_a})
           << vb_b.copy_from(luisa::span{vertices_b})
           << tb.copy_from(luisa::span{triangles})
           << mesh_a.build()
           << accel_a.build()
           << mesh_b.build()
           << accel_b.build();

    auto hits_a = trace_hits(device, stream, ctx, accel_a);
    expect_hits(hits_a, 1u, 0u, "shared-index accel_a");
    auto hits_b = trace_hits(device, stream, ctx, accel_b);
    expect_hits(hits_b, 0u, 1u, "shared-index accel_b");
}

// Recreating a BLAS (FORCE_BUILD) moves it to a new device address; the TLAS
// must pick up the new address on its next build. Trace results must reflect
// the new geometry, not the stale pre-rebuild BLAS.
void test_blas_recreate_refreshes_tlas(Device &device) {
    Stream stream = device.create_stream();
    auto ctx = make_trace_context(device);

    auto vertices = make_triangle_vertices(-1.0f);
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vb = device.create_buffer<float3>(vertices.size());
    auto tb = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vb, tb);
    Accel accel = device.create_accel({});
    accel.emplace_back(mesh, make_float4x4(1.0f));
    stream << vb.copy_from(luisa::span{vertices})
           << tb.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build();
    expect_hits(trace_hits(device, stream, ctx, accel), 1u, 0u, "recreate: initial");

    // Move the triangle to x=+1 and recreate the BLAS, then rebuild the TLAS.
    vertices = make_triangle_vertices(+1.0f);
    stream << vb.copy_from(luisa::span{vertices})
           << mesh.build(AccelBuildRequest::FORCE_BUILD)
           << accel.build(AccelBuildRequest::FORCE_BUILD);
    expect_hits(trace_hits(device, stream, ctx, accel), 0u, 1u, "recreate: moved");

    // Move it back with the default PREFER_UPDATE request (no allow_update in
    // AccelOption, so this is another full recreate).
    vertices = make_triangle_vertices(-1.0f);
    stream << vb.copy_from(luisa::span{vertices})
           << mesh.build()
           << accel.build();
    expect_hits(trace_hits(device, stream, ctx, accel), 1u, 0u, "recreate: moved back");
}

// Destroying a mesh while its accel still references it, then replacing the
// instance with a new mesh and rebuilding the accel. The backend must not
// dereference the destroyed BLAS bookkeeping when folding the replacement into
// the TLAS, and the rebuilt TLAS must trace the new mesh exactly.
void test_mesh_destroyed_then_instance_replaced(Device &device) {
    Stream stream = device.create_stream();
    auto ctx = make_trace_context(device);

    auto vertices_a = make_triangle_vertices(-1.0f);
    auto vertices_b = make_triangle_vertices(+1.0f);
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vb_a = device.create_buffer<float3>(vertices_a.size());
    auto vb_b = device.create_buffer<float3>(vertices_b.size());
    auto tb = device.create_buffer<Triangle>(triangles.size());
    stream << vb_a.copy_from(luisa::span{vertices_a})
           << vb_b.copy_from(luisa::span{vertices_b})
           << tb.copy_from(luisa::span{triangles});

    Accel accel = device.create_accel({});
    std::optional<Mesh> mesh_a;
    mesh_a.emplace(device.create_mesh(vb_a, tb));
    accel.emplace_back(*mesh_a, make_float4x4(1.0f));
    stream << mesh_a->build() << accel.build();
    expect_hits(trace_hits(device, stream, ctx, accel), 1u, 0u, "replace: initial");

    // Recreate the BLAS once so the TLAS has stale instance bookkeeping for it,
    // then destroy the mesh before touching the accel again.
    stream << mesh_a->build(AccelBuildRequest::FORCE_BUILD) << synchronize();
    mesh_a.reset();

    // Replace instance 0 with a fresh mesh and rebuild the accel.
    auto mesh_b = device.create_mesh(vb_b, tb);
    stream << mesh_b.build();
    accel.set(0u, mesh_b, make_float4x4(1.0f));
    stream << accel.build(AccelBuildRequest::FORCE_BUILD);
    expect_hits(trace_hits(device, stream, ctx, accel), 0u, 1u, "replace: swapped");
}

// Teardown order: the mesh goes out of scope before the accel that references
// it. Destroying both must not crash, and the device must stay usable.
void test_mesh_destroyed_before_accel_teardown(Device &device) {
    Stream stream = device.create_stream();
    auto ctx = make_trace_context(device);

    auto vertices = make_triangle_vertices(-1.0f);
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vb = device.create_buffer<float3>(vertices.size());
    auto tb = device.create_buffer<Triangle>(triangles.size());
    stream << vb.copy_from(luisa::span{vertices})
           << tb.copy_from(luisa::span{triangles});

    Accel accel = device.create_accel({});
    {
        auto mesh = device.create_mesh(vb, tb);
        accel.emplace_back(mesh, make_float4x4(1.0f));
        stream << mesh.build() << accel.build();
        expect_hits(trace_hits(device, stream, ctx, accel), 1u, 0u, "teardown: initial");
        // mesh destroyed here, while the accel still references it
    }
    accel = Accel{};// destroy the accel after the mesh

    // The device must remain usable: build a fresh pair and trace.
    auto mesh2 = device.create_mesh(vb, tb);
    Accel accel2 = device.create_accel({});
    accel2.emplace_back(mesh2, make_float4x4(1.0f));
    stream << mesh2.build() << accel2.build();
    expect_hits(trace_hits(device, stream, ctx, accel2), 1u, 0u, "teardown: fresh pair");
}

// A BLAS recreate queues a per-TLAS mesh-refresh entry for the recreated mesh.
// An explicit instance replacement for the same slot in the next accel build
// must win over that pending refresh entry — otherwise the TLAS silently keeps
// tracing the old mesh (and a later destroy of the "replaced" mesh escalates
// the stale slot into a dangling BLAS reference).
void test_blas_recreate_then_set_new_mesh(Device &device) {
    Stream stream = device.create_stream();
    auto ctx = make_trace_context(device);

    auto vertices_a = make_triangle_vertices(-1.0f);
    auto vertices_b = make_triangle_vertices(+1.0f);
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vb_a = device.create_buffer<float3>(vertices_a.size());
    auto vb_b = device.create_buffer<float3>(vertices_b.size());
    auto tb = device.create_buffer<Triangle>(triangles.size());
    auto mesh_a = device.create_mesh(vb_a, tb);
    auto mesh_b = device.create_mesh(vb_b, tb);
    Accel accel = device.create_accel({});
    accel.emplace_back(mesh_a, make_float4x4(1.0f));
    stream << vb_a.copy_from(luisa::span{vertices_a})
           << vb_b.copy_from(luisa::span{vertices_b})
           << tb.copy_from(luisa::span{triangles})
           << mesh_a.build()
           << mesh_b.build()
           << accel.build();
    expect_hits(trace_hits(device, stream, ctx, accel), 1u, 0u, "override: initial");

    // Recreate mesh_a's BLAS without rebuilding the accel in between: the
    // backend now carries a pending mesh-refresh entry for slot 0.
    stream << mesh_a.build(AccelBuildRequest::FORCE_BUILD) << synchronize();

    // Replace instance 0 with mesh_b in the very next accel build. The explicit
    // replacement must take precedence over the pending refresh of mesh_a.
    accel.set(0u, mesh_b, make_float4x4(1.0f));
    stream << accel.build(AccelBuildRequest::FORCE_BUILD);
    expect_hits(trace_hits(device, stream, ctx, accel), 0u, 1u, "override: replaced");

    // Once the refresh entry is consumed, repeating the replacement is a no-op
    // and must keep tracing mesh_b.
    accel.set(0u, mesh_b, make_float4x4(1.0f));
    stream << accel.build(AccelBuildRequest::FORCE_BUILD);
    expect_hits(trace_hits(device, stream, ctx, accel), 0u, 1u, "override: steady");
}

// Repeated in-place BLAS updates (allow_update + PREFER_UPDATE on both levels)
// interleaved with TLAS update builds. Every TLAS build must observe the BLAS
// content produced by the same batch; a missing per-referenced-BLAS barrier
// lets the TLAS build pick up stale geometry from a previous iteration.
void test_inplace_blas_update_tlas_sync_stress(Device &device) {
    static constexpr uint iteration_count = 32u;
    Stream stream = device.create_stream();
    auto ctx = make_trace_context(device);

    const std::array triangles{Triangle{0u, 1u, 2u}};
    AccelOption option{};
    option.allow_update = true;
    auto vertices = make_triangle_vertices(-1.0f);
    auto vb = device.create_buffer<float3>(vertices.size());
    auto tb = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vb, tb, option);
    Accel accel = device.create_accel(option);
    accel.emplace_back(mesh, make_float4x4(1.0f));
    stream << vb.copy_from(luisa::span{vertices})
           << tb.copy_from(luisa::span{triangles})
           << mesh.build(AccelBuildRequest::FORCE_BUILD)
           << accel.build(AccelBuildRequest::FORCE_BUILD);
    expect_hits(trace_hits(device, stream, ctx, accel), 1u, 0u, "in-place stress: initial");

    for (auto i = 0u; i < iteration_count; i++) {
        // Alternate the triangle between x=+1 and x=-1 using in-place updates
        // only: neither level is recreated and both build-time bookkeeping
        // lists stay empty, so a missing barrier between the BLAS build and
        // the TLAS build shows up as the previous iteration's hit pattern.
        auto target_x = (i & 1u) == 0u ? 1.0f : -1.0f;
        vertices = make_triangle_vertices(target_x);
        stream << vb.copy_from(luisa::span{vertices})
               << mesh.build(AccelBuildRequest::PREFER_UPDATE)
               << accel.build(AccelBuildRequest::PREFER_UPDATE);
        auto hits = trace_hits(device, stream, ctx, accel);
        expect_hits(hits, target_x < 0.0f ? 1u : 0u, target_x > 0.0f ? 1u : 0u,
                    luisa::format("in-place stress: iteration {}", i));
    }
}

// Whether the backend supports motion-enabled acceleration structures
// (VK_NV_ray_tracing_motion_blur on Vulkan).
[[nodiscard]] bool motion_blur_supported(Device &device) noexcept {
    if (device.backend_name() != "vk") { return false; }
    return device.query("motion_blur") == "true";
}

struct MotionTraceContext {
    Buffer<float3> origin_buffer;
    Buffer<uint> result_buffer;
    Shader1D<Accel, Buffer<float3>, Buffer<uint>, float> shader;
};

MotionTraceContext make_motion_trace_context(Device &device) {
    MotionTraceContext ctx;
    ctx.origin_buffer = device.create_buffer<float3>(2u);
    ctx.result_buffer = device.create_buffer<uint>(2u);
    auto origins = make_ray_origins();
    Stream stream = device.create_stream();
    stream << ctx.origin_buffer.copy_from(luisa::span{origins}) << synchronize();
    Kernel1D trace_kernel = [](AccelVar accel,
                               BufferFloat3 origins,
                               BufferUInt results,
                               Float time) noexcept {
        auto i = dispatch_id().x;
        auto ray = make_ray(origins.read(i), make_float3(0.0f, 0.0f, -1.0f));
        auto hit = accel.intersect_motion(ray, time, {});
        results.write(i, ite(hit->miss(), 0u, 1u));
    };
    ctx.shader = device.compile(trace_kernel);
    return ctx;
}

[[nodiscard]] std::array<uint, 2u> trace_motion_hits(Stream &stream,
                                                     const MotionTraceContext &ctx,
                                                     Accel &accel, float time) {
    std::array<uint, 2u> hits{};
    stream << ctx.shader(accel, ctx.origin_buffer, ctx.result_buffer, time).dispatch(2u)
           << ctx.result_buffer.copy_to(luisa::span{hits})
           << synchronize();
    return hits;
}

// Two-instance accel used by the motion-path tests below.
// Note on observability: vk compute-shader ray queries currently lower
// intersect_motion to a plain non-motion trace (the time argument is dropped
// by the HLSL fallback), so the tests assert structural hit/miss integrity of
// the TLAS instance table rather than time-varying geometry. A vertex-motion
// BLAS child never reports a hit under that degraded trace, which is itself a
// stable observable used below.
struct MotionTwoMeshScene {
    Buffer<float3> motion_vertices;
    Buffer<float3> static_vertices;
    Buffer<Triangle> triangles;
    Mesh motion_mesh;
    Mesh static_mesh;
    Accel accel;
};

// Slot 0: a vertex-motion BLAS (two identical keyframes) at x=-1 — misses under
// the degraded ray-query trace. Slot 1: a static mesh at x=+1 — hits.
MotionTwoMeshScene make_motion_two_mesh_scene(Device &device, Stream &stream) {
    auto keyframe = make_triangle_vertices(-1.0f);
    std::array<float3, 6u> motion_verts{};
    std::copy(keyframe.begin(), keyframe.end(), motion_verts.begin());
    std::copy(keyframe.begin(), keyframe.end(), motion_verts.begin() + 3);
    auto static_verts = make_triangle_vertices(+1.0f);
    const std::array tris{Triangle{0u, 1u, 2u}};

    AccelOption motion_option{};
    motion_option.motion.keyframe_count = 2u;
    motion_option.motion.time_start = 0.0f;
    motion_option.motion.time_end = 1.0f;

    MotionTwoMeshScene scene;
    scene.motion_vertices = device.create_buffer<float3>(motion_verts.size());
    scene.static_vertices = device.create_buffer<float3>(static_verts.size());
    scene.triangles = device.create_buffer<Triangle>(tris.size());
    scene.motion_mesh = device.create_mesh(scene.motion_vertices, scene.triangles, motion_option);
    scene.static_mesh = device.create_mesh(scene.static_vertices, scene.triangles);
    scene.accel = device.create_accel({});
    scene.accel.emplace_back(scene.motion_mesh, make_float4x4(1.0f));
    scene.accel.emplace_back(scene.static_mesh, make_float4x4(1.0f));
    stream << scene.motion_vertices.copy_from(luisa::span{motion_verts})
           << scene.static_vertices.copy_from(luisa::span{static_verts})
           << scene.triangles.copy_from(luisa::span{tris})
           << scene.motion_mesh.build()
           << scene.static_mesh.build()
           << scene.accel.build();
    return scene;
}

// A motion-enabled BLAS recreate queues a mesh-refresh entry; the next TLAS
// build must fold it in without disturbing the other instances. The vk motion
// path used to rebuild the instance buffers from the current modification list
// only, so this build (empty modifications, one pending refresh) dropped the
// refresh, zeroed the whole instance buffer, and every ray missed afterwards —
// including the static mesh that was never touched.
void test_motion_blas_recreate_preserves_tlas(Device &device) {
    Stream stream = device.create_stream();
    auto ctx = make_motion_trace_context(device);
    auto scene = make_motion_two_mesh_scene(device, stream);
    // The vertex-motion BLAS (x=-1) misses under the degraded trace; the
    // static mesh (x=+1) hits.
    expect_hits(trace_motion_hits(stream, ctx, scene.accel, 0.5f), 0u, 1u,
                "motion recreate: initial");

    stream << scene.motion_mesh.build(AccelBuildRequest::FORCE_BUILD)
           << synchronize();
    stream << scene.accel.build(AccelBuildRequest::FORCE_BUILD);
    expect_hits(trace_motion_hits(stream, ctx, scene.accel, 0.5f), 0u, 1u,
                "motion recreate: rebuilt");
}

// Scene with an SRT motion instance (identity keyframes, so the pose is
// time-independent and hits under the degraded trace) at slot 0 wrapping the
// mesh at x=-1, plus a plain static mesh at x=+1 at slot 1.
struct MotionInstanceScene {
    Buffer<float3> vertices_a;
    Buffer<float3> vertices_b;
    Buffer<Triangle> triangles;
    Mesh mesh_a;
    Mesh mesh_b;
    MotionInstance instance_a;
    Accel accel;
};

MotionInstanceScene make_motion_instance_scene(Device &device, Stream &stream) {
    auto verts_a = make_triangle_vertices(-1.0f);
    auto verts_b = make_triangle_vertices(+1.0f);
    const std::array tris{Triangle{0u, 1u, 2u}};

    AccelMotionOption motion_option{};
    motion_option.mode = AccelMotionMode::SRT;
    motion_option.keyframe_count = 2u;

    MotionInstanceScene scene;
    scene.vertices_a = device.create_buffer<float3>(verts_a.size());
    scene.vertices_b = device.create_buffer<float3>(verts_b.size());
    scene.triangles = device.create_buffer<Triangle>(tris.size());
    scene.mesh_a = device.create_mesh(scene.vertices_a, scene.triangles);
    scene.mesh_b = device.create_mesh(scene.vertices_b, scene.triangles);
    scene.instance_a = device.create_motion_instance(scene.mesh_a, motion_option);
    const std::array keyframes{MotionInstanceTransformSRT{}, MotionInstanceTransformSRT{}};
    scene.instance_a.set_keyframes(luisa::span{keyframes});
    scene.accel = device.create_accel({});
    scene.accel.emplace_back(scene.instance_a, make_float4x4(1.0f));
    scene.accel.emplace_back(scene.mesh_b, make_float4x4(1.0f));
    stream << scene.vertices_a.copy_from(luisa::span{verts_a})
           << scene.vertices_b.copy_from(luisa::span{verts_b})
           << scene.triangles.copy_from(luisa::span{tris})
           << scene.mesh_a.build()
           << scene.mesh_b.build()
           << scene.instance_a.build()
           << scene.accel.build();
    return scene;
}

// A partial update that re-sets only the motion instance must preserve the
// untouched static instance; the vk motion path used to refill the instance
// buffers from the modification list alone, zeroing every untouched slot.
// Expected hit patterns assume the current degraded vk ray-query trace (the
// time operand of intersect_motion is dropped on the compute path): an
// SRT-encoded instance never reports a hit, a static instance does. If the vk
// ray-query codegen later learns to trace motion properly, the expectations
// for slot 0 flip from miss to hit and must be updated here.
void test_motion_partial_update_preserves_instances(Device &device) {
    Stream stream = device.create_stream();
    auto ctx = make_motion_trace_context(device);
    auto scene = make_motion_instance_scene(device, stream);
    // slot 0: SRT motion instance (misses under the degraded trace);
    // slot 1: static mesh (hits).
    expect_hits(trace_motion_hits(stream, ctx, scene.accel, 0.5f), 0u, 1u,
                "motion partial update: initial");

    // Re-set slot 0 with the same motion instance: the modification list covers
    // slot 0 only, so slot 1 must be preserved from the previous build.
    scene.accel.set_motion_instance(0u, scene.instance_a);
    stream << scene.accel.build(AccelBuildRequest::FORCE_BUILD);
    expect_hits(trace_motion_hits(stream, ctx, scene.accel, 0.5f), 0u, 1u,
                "motion partial update: re-set motion instance");

    // A transform-only update of slot 1 must keep the TLAS motion-capable
    // (slot 0 stays SRT-encoded and still misses) while applying the new
    // transform (slot 1 moves away and misses). Before the fix this build
    // flipped the TLAS to the non-motion path, silently reinterpreting slot 0
    // as a static instance.
    scene.accel.set_transform_on_update(1u, translation(0.0f, 2.0f, 0.0f));
    stream << scene.accel.build(AccelBuildRequest::PREFER_UPDATE);
    expect_hits(trace_motion_hits(stream, ctx, scene.accel, 0.5f), 0u, 0u,
                "motion partial update: transform-only");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "shared_index_buffer_two_accels"_test = [&] {
        test_shared_index_buffer_two_accels(dc->device);
    };
    "blas_recreate_refreshes_tlas"_test = [&] {
        test_blas_recreate_refreshes_tlas(dc->device);
    };
    "mesh_destroyed_then_instance_replaced"_test = [&] {
        test_mesh_destroyed_then_instance_replaced(dc->device);
    };
    "mesh_destroyed_before_accel_teardown"_test = [&] {
        test_mesh_destroyed_before_accel_teardown(dc->device);
    };
    "blas_recreate_then_set_new_mesh"_test = [&] {
        test_blas_recreate_then_set_new_mesh(dc->device);
    };
    "inplace_blas_update_tlas_sync_stress"_test = [&] {
        test_inplace_blas_update_tlas_sync_stress(dc->device);
    };
    "motion_blas_recreate_preserves_tlas"_test = [&] {
        if (!motion_blur_supported(dc->device)) {
            LUISA_INFO("Skipping motion_blas_recreate_preserves_tlas: motion blur not supported.");
            return;
        }
        test_motion_blas_recreate_preserves_tlas(dc->device);
    };
    "motion_partial_update_preserves_instances"_test = [&] {
        if (!motion_blur_supported(dc->device)) {
            LUISA_INFO("Skipping motion_partial_update_preserves_instances: motion blur not supported.");
            return;
        }
        test_motion_partial_update_preserves_instances(dc->device);
    };
}
