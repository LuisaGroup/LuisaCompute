// TLAS multi-instance intersection correctness test.
// This test covers the known DX12 backend bug where, after emplacing two (or
// more) meshes into an Accel and building the TLAS, rays that should hit the
// second instance report the correct TLAS instance slot (inst == 1) but
// primitive indices coming from the FIRST emplace's bottom-level acceleration
// structure (BLAS). The repro is minimal:
//   - mesh A: subdivided plane (4x4 grid, 16 vertices, 18 triangles)
//   - mesh B: unit quad (4 vertices, 2 triangles)
//   - accel entry 0 = mesh A at origin, entry 1 = mesh B at (10, 0, 0)
//   - trace rays at both regions and verify inst/prim against the expected
//     ranges (entry 0: prim < 18, entry 1: prim < 2).
// Several trigger variants are exercised: a plain full Accel::build(), builds
// split across synchronize points, compaction-enabled builds, multiple
// consecutive builds, and a three-mesh scene.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto grid_side = 4u;               // 4x4 vertices
constexpr auto grid_tri_count = 18u;         // 9 quads * 2
constexpr auto quad_tri_count = 2u;          // 1 quad * 2
constexpr auto region_grid = 5u;             // rays per axis
constexpr auto rays_per_region = region_grid * region_grid;
constexpr auto two_region_rays = rays_per_region * 2u;

[[nodiscard]] auto make_grid_vertices() noexcept {
    std::array<float3, grid_side * grid_side> vertices{};
    for (auto j = 0u; j < grid_side; j++) {
        for (auto i = 0u; i < grid_side; i++) {
            auto x = -2.0f + 4.0f * static_cast<float>(i) / (grid_side - 1u);
            auto y = -2.0f + 4.0f * static_cast<float>(j) / (grid_side - 1u);
            vertices[j * grid_side + i] = make_float3(x, y, 0.0f);
        }
    }
    return vertices;
}

[[nodiscard]] auto make_grid_triangles() noexcept {
    std::array<Triangle, grid_tri_count> triangles{};
    auto t = 0u;
    for (auto j = 0u; j + 1u < grid_side; j++) {
        for (auto i = 0u; i + 1u < grid_side; i++) {
            auto a = j * grid_side + i;
            auto b = a + 1u;
            auto c = a + grid_side + 1u;
            auto d = a + grid_side;
            triangles[t++] = Triangle{a, b, c};
            triangles[t++] = Triangle{a, c, d};
        }
    }
    return triangles;
}

[[nodiscard]] auto unit_quad_vertices() noexcept {
    return std::array{
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(1.0f, 1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
}

[[nodiscard]] auto unit_quad_triangles() noexcept {
    return std::array{
        Triangle{0u, 1u, 2u},
        Triangle{0u, 2u, 3u}};
}

struct VariantResult {
    luisa::string name;
    bool ok = true;
    luisa::string detail;
};

// Builds per-region ray specifications: origins, expected instance and the
// maximum primitive index allowed for that instance.
struct RaySetup {
    std::vector<float3> origins;
    std::vector<uint> expected_inst;
    std::vector<uint> max_prim;
};

[[nodiscard]] auto make_two_region_setup() noexcept {
    RaySetup setup;
    auto append_region = [&](float x0, float x1, float y0, float y1,
                             uint inst, uint max_primitive) {
        for (auto j = 0u; j < region_grid; j++) {
            for (auto i = 0u; i < region_grid; i++) {
                auto x = x0 + (x1 - x0) * static_cast<float>(i) / (region_grid - 1u);
                auto y = y0 + (y1 - y0) * static_cast<float>(j) / (region_grid - 1u);
                setup.origins.emplace_back(x, y, 1.0f);
                setup.expected_inst.emplace_back(inst);
                setup.max_prim.emplace_back(max_primitive);
            }
        }
    };
    append_region(-1.5f, 1.5f, -1.5f, 1.5f, 0u, grid_tri_count);
    append_region(10.2f, 10.8f, 0.2f, 0.8f, 1u, quad_tri_count);
    return setup;
}

// Checks a result buffer against the ray setup; returns a failure summary.
[[nodiscard]] auto verify_results(luisa::span<const float3> origins,
                                  luisa::span<const uint> expected_inst,
                                  luisa::span<const uint> max_prim,
                                  luisa::span<const uint4> results,
                                  luisa::string_view label) noexcept {
    VariantResult r;
    r.name = label;
    luisa::string detail;
    auto fail_count = 0u;
    for (auto i = 0u; i < origins.size(); i++) {
        auto const &hit = results[i];
        auto miss = hit.z == 0u;
        auto inst_ok = !miss && hit.x == expected_inst[i];
        auto prim_ok = !miss && hit.y < max_prim[i];
        if (miss || !inst_ok || !prim_ok) {
            r.ok = false;
            fail_count++;
            detail += luisa::format(
                "  ray {} @ ({:.2f},{:.2f}): inst={} prim={} miss={} "
                "(exp inst={} prim<{})\n",
                i, origins[i].x, origins[i].y, hit.x, hit.y, miss,
                expected_inst[i], max_prim[i]);
        }
    }
    if (!r.ok) {
        r.detail = luisa::format("[{}] {} / {} rays violated inst/prim:\n{}",
                                 label, fail_count, origins.size(), detail);
    } else {
        LUISA_INFO("[{}] all {} rays passed.", label, origins.size());
    }
    return r;
}

}// namespace

void test_accel_tlas_instances(Device &device) {
    log_level_verbose();

    auto stream = device.create_stream();

    auto grid_vertices = make_grid_vertices();
    auto grid_triangles = make_grid_triangles();
    auto quad_vertices = unit_quad_vertices();
    auto quad_triangles = unit_quad_triangles();
    auto grid_vb = device.create_buffer<float3>(grid_vertices.size());
    auto grid_ib = device.create_buffer<Triangle>(grid_triangles.size());
    auto quad_vb = device.create_buffer<float3>(quad_vertices.size());
    auto quad_ib = device.create_buffer<Triangle>(quad_triangles.size());
    stream << grid_vb.copy_from(luisa::span{grid_vertices})
           << grid_ib.copy_from(luisa::span{grid_triangles})
           << quad_vb.copy_from(luisa::span{quad_vertices})
           << quad_ib.copy_from(luisa::span{quad_triangles})
           << synchronize();

    Kernel1D trace = [](BufferFloat3 origins,
                        BufferUInt4 results,
                        AccelVar accel) noexcept {
        auto i = dispatch_id().x;
        auto origin = origins.read(i);
        auto ray = make_ray(origin, make_float3(0.0f, 0.0f, -1.0f));
        auto hit = accel.intersect(ray, {});
        auto is_miss = hit->inst == ~0u;
        results.write(i, make_uint4(
                             ite(is_miss, ~0u, hit->inst),
                             ite(is_miss, ~0u, hit->prim),
                             ite(is_miss, 0u, 1u),
                             0u));
    };
    auto trace_shader = device.compile(trace);

    auto setup = make_two_region_setup();
    auto origin_buffer = device.create_buffer<float3>(setup.origins.size());
    auto result_buffer = device.create_buffer<uint4>(setup.origins.size());
    stream << origin_buffer.copy_from(luisa::span{setup.origins}) << synchronize();

    auto run_trace = [&](Accel &accel, luisa::string_view label) {
        std::array<uint4, two_region_rays> host_results{};
        stream << trace_shader(origin_buffer, result_buffer, accel)
                      .dispatch(setup.origins.size())
               << result_buffer.copy_to(luisa::span{host_results})
               << synchronize();
        auto r = verify_results(setup.origins, setup.expected_inst,
                                setup.max_prim, luisa::span{host_results}, label);
        if (!r.ok) { LUISA_INFO("{}", r.detail); }
        expect(r.ok) << r.detail;
        return r.ok;
    };

    auto make_meshes = [&](bool compaction) {
        AccelOption mesh_option{
            .hint = AccelUsageHint::FAST_TRACE,
            .allow_compaction = compaction,
            .allow_update = true};
        auto mesh_a = device.create_mesh(grid_vb, grid_ib, mesh_option);
        auto mesh_b = device.create_mesh(quad_vb, quad_ib, mesh_option);
        return std::pair{std::move(mesh_a), std::move(mesh_b)};
    };
    auto emplace_ab = [&](Accel &accel, Mesh &mesh_a, Mesh &mesh_b) {
        accel.emplace_back(mesh_a, make_float4x4(1.0f), 0xffu, true);
        accel.emplace_back(mesh_b, translation(make_float3(10.0f, 0.0f, 0.0f)),
                           0xffu, true);
    };

    constexpr auto build_req = AccelBuildRequest::FORCE_BUILD;

    // ---- variant 1: plain full build, everything in one chain --------------
    {
        auto [mesh_a, mesh_b] = make_meshes(false);
        AccelOption accel_option{.hint = AccelUsageHint::FAST_TRACE,
                                 .allow_compaction = false,
                                 .allow_update = true};
        auto accel = device.create_accel(accel_option);
        emplace_ab(accel, mesh_a, mesh_b);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << accel.build(build_req);
        run_trace(accel, "v1 plain full build");
    }

    // ---- variant 2: builds split by synchronize points ---------------------
    {
        auto [mesh_a, mesh_b] = make_meshes(false);
        auto accel = device.create_accel();
        emplace_ab(accel, mesh_a, mesh_b);
        stream << mesh_a.build(build_req) << synchronize();
        stream << mesh_b.build(build_req) << synchronize();
        stream << accel.build(build_req) << synchronize();
        run_trace(accel, "v2 split by synchronize");
    }

    // ---- variant 3: compaction enabled -------------------------------------
    {
        auto [mesh_a, mesh_b] = make_meshes(true);
        AccelOption accel_option{.hint = AccelUsageHint::FAST_TRACE,
                                 .allow_compaction = true,
                                 .allow_update = true};
        auto accel = device.create_accel(accel_option);
        emplace_ab(accel, mesh_a, mesh_b);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << accel.build(build_req);
        run_trace(accel, "v3 compaction initial");
        // A second build (and a refit) after compaction kicked in.
        stream << accel.build(build_req);
        run_trace(accel, "v3 compaction second build");
        stream << accel.build(AccelBuildRequest::PREFER_UPDATE);
        run_trace(accel, "v3 compaction prefer-update");
    }

    // ---- variant 4: two consecutive full builds ----------------------------
    {
        auto [mesh_a, mesh_b] = make_meshes(false);
        auto accel = device.create_accel();
        emplace_ab(accel, mesh_a, mesh_b);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << accel.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace(accel, "v4 two consecutive builds");
    }

    // ---- variant 5: three meshes (middle one expected to break too) --------
    {
        auto mesh_a = device.create_mesh(grid_vb, grid_ib);
        auto mesh_b = device.create_mesh(quad_vb, quad_ib);
        auto mesh_c = device.create_mesh(quad_vb, quad_ib);
        auto accel = device.create_accel();
        accel.emplace_back(mesh_a, make_float4x4(1.0f), 0xffu, true);
        accel.emplace_back(mesh_b, translation(make_float3(10.0f, 0.0f, 0.0f)),
                           0xffu, true);
        accel.emplace_back(mesh_c, translation(make_float3(10.0f, 3.0f, 0.0f)),
                           0xffu, true);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << mesh_c.build(build_req)
               << accel.build(build_req);
        // The existing 2-region setup also probes entries 0 and 1; entry 2 is
        // probed separately below.
        run_trace(accel, "v5 three meshes entries 0/1");
        std::array<float3, rays_per_region> c_origins{};
        std::array<uint, rays_per_region> c_inst{}, c_prim{};
        for (auto j = 0u; j < region_grid; j++) {
            for (auto i = 0u; i < region_grid; i++) {
                auto idx = j * region_grid + i;
                c_origins[idx] = make_float3(
                    10.2f + 0.6f * static_cast<float>(i) / (region_grid - 1u),
                    3.2f + 0.6f * static_cast<float>(j) / (region_grid - 1u),
                    1.0f);
                c_inst[idx] = 2u;
                c_prim[idx] = quad_tri_count;
            }
        }
        std::array<uint4, rays_per_region> c_results{};
        auto c_origin_buffer = device.create_buffer<float3>(c_origins.size());
        auto c_result_buffer = device.create_buffer<uint4>(c_results.size());
        stream << c_origin_buffer.copy_from(luisa::span{c_origins}) << synchronize();
        stream << trace_shader(c_origin_buffer, c_result_buffer, accel)
                      .dispatch(c_origins.size())
               << c_result_buffer.copy_to(luisa::span{c_results})
               << synchronize();
        auto r = verify_results(luisa::span{c_origins}, luisa::span{c_inst},
                                luisa::span{c_prim}, luisa::span{c_results},
                                "v5 three meshes entry 2");
        if (!r.ok) { LUISA_INFO("{}", r.detail); }
        expect(r.ok) << r.detail;
    }
    // ---- variant 6: path-tracer-like refit loop ----------------------------
    {
        auto [mesh_a, mesh_b] = make_meshes(false);
        auto accel = device.create_accel();
        emplace_ab(accel, mesh_a, mesh_b);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace(accel, "v6 loop frame 0");
        for (auto f = 1u; f <= 24u; f++) {
            auto dx = 10.0f + 0.01f * std::sin(static_cast<float>(f) * 0.3f);
            accel.set_transform_on_update(1u,
                                          translation(make_float3(dx, 0.0f, 0.0f)));
            auto req = (f % 2u == 0u) ? AccelBuildRequest::PREFER_UPDATE
                                      : build_req;
            stream << accel.build(req);
            run_trace(accel, luisa::format("v6 loop frame {}", f));
        }
    }

    // ---- variant 7: two accels sharing both meshes -------------------------
    {
        auto mesh_a = device.create_mesh(grid_vb, grid_ib);
        auto mesh_b = device.create_mesh(quad_vb, quad_ib);
        auto accel1 = device.create_accel();
        auto accel2 = device.create_accel();
        emplace_ab(accel1, mesh_a, mesh_b);
        emplace_ab(accel2, mesh_a, mesh_b);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << accel1.build(build_req)
               << accel2.build(build_req);
        run_trace(accel1, "v7 shared meshes accel1");
        run_trace(accel2, "v7 shared meshes accel2");
        for (auto f = 0u; f < 12u; f++) {
            auto dx = 10.0f + 0.01f * std::cos(static_cast<float>(f));
            accel1.set_transform_on_update(1u,
                                           translation(make_float3(dx, 0.0f, 0.0f)));
            accel2.set_transform_on_update(0u,
                                           make_float4x4(1.0f));
            stream << accel1.build(AccelBuildRequest::PREFER_UPDATE)
                   << accel2.build(AccelBuildRequest::PREFER_UPDATE);
            run_trace(accel1, luisa::format("v7 shared refit accel1 f{}", f));
            run_trace(accel2, luisa::format("v7 shared refit accel2 f{}", f));
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_accel_tlas_instances(dc->device);
}