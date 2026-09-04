// Test for BLAS/TLAS lifetime coupling between Mesh (BLAS) and Accel (TLAS).
// This test covers:
// - two meshes sharing one triangle buffer in two separate accels, built in one batch
// - BLAS recreate (FORCE_BUILD) while referenced by a TLAS, then TLAS rebuild and trace
// - destroying a referenced mesh, replacing the instance with a new mesh, and rebuilding
// - teardown order with the mesh destroyed before the accel

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

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
}
