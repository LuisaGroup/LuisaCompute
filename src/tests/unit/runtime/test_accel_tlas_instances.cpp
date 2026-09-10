// TLAS multi-instance intersection correctness test.
// This test covers the known DX12 backend bug where, after emplacing two (or
// more) meshes into an Accel and building the TLAS, rays that should hit the
// second instance report the correct TLAS instance slot (inst == 1) but
// primitive indices coming from the FIRST emplace's bottom-level acceleration
// structure (BLAS). The repro is minimal:
// - mesh A: subdivided plane (4x4 grid, 16 vertices, 18 triangles)
// - mesh B: unit quad (4 vertices, 2 triangles)
// - accel entry 0 = mesh A at origin, entry 1 = mesh B at (10, 0, 0)
// - trace rays at both regions and verify inst/prim against the expected
// ranges (entry 0: prim < 18, entry 1: prim < 2).
// Several trigger variants are exercised: a plain full Accel::build(), builds
// split across synchronize points, compaction-enabled builds, multiple
// consecutive builds, and a three-mesh scene.
//
// Variants v9 onward stress the TLAS refresh-entry lifecycle hardened in
// commits 2909242c0 (dx) and 4c0d416f8 (vk): pending BLAS-address refresh
// entries were keyed by TLAS instance index and stored pooled MeshHandle
// pointers, so a stale entry surviving handle destruction/reuse could rebind
// an instance to the wrong BLAS ("correct TLAS slot, wrong BLAS"), and entries
// for slots removed by shrink could resurrect as phantom instances. The
// backend now stores the stable BottomAccel/Blas (the only value read from the
// handle), erases entries referencing a destroyed BLAS, and drops out-of-range
// entries on shrink. The variants verify: shrink 3->1 with entries pending
// (v9), destroy-with-pending followed by MeshHandle-pool reuse and replacement
// (v10), middle-slot kept vs dropped entries on 3->2 shrink (v11),
// destroy/replace churn across cycles (v12), single-shot entry consumption
// across repeated builds (v13), teardown ordering with pending entries (v14),
// and two accels sharing meshes with pending entries + destruction (v15).
// The refresh bookkeeping lives per-slot inside the TLAS instance array
// (queue/consume/drop are O(1) array accesses with no hash map or per-entry
// allocation).

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
  // Three-region setup: grid at the origin (inst 0), quad at (10, 0) (inst 1)
  // and quad at (10, 3) (inst 2).
  [[nodiscard]] auto make_three_region_setup() noexcept {
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
      append_region(10.2f, 10.8f, 3.2f, 3.8f, 2u, quad_tri_count);
      return setup;
  }
  // Marks rays in [first, last) as expected-miss (~0u instance) so the
  // verifier asserts those regions do not intersect any instance.
  [[nodiscard]] auto make_miss_expectation(RaySetup setup,
                                           size_t first, size_t last) noexcept {
      for (auto i = first; i < last && i < setup.expected_inst.size(); i++) {
          setup.expected_inst[i] = ~0u;
      }
      return setup;
  }

// Checks a result buffer against the ray setup; returns a failure summary.
  // An expected instance of ~0u means the ray is expected to MISS.
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
          auto expect_miss = expected_inst[i] == ~0u;
          auto inst_ok = expect_miss ? miss : (!miss && hit.x == expected_inst[i]);
          auto prim_ok = expect_miss ? miss : (!miss && hit.y < max_prim[i]);
          if (miss != expect_miss || !inst_ok || !prim_ok) {
              r.ok = false;
              fail_count++;
              if (expect_miss) {
                  detail += luisa::format(
                      "  ray {} @ ({:.2f},{:.2f}): inst={} prim={} miss={} "
                      "(exp miss)\n",
                      i, origins[i].x, origins[i].y, hit.x, hit.y, miss);
              } else {
                  detail += luisa::format(
                      "  ray {} @ ({:.2f},{:.2f}): inst={} prim={} miss={} "
                      "(exp inst={} prim<{})\n",
                      i, origins[i].x, origins[i].y, hit.x, hit.y, miss,
                      expected_inst[i], max_prim[i]);
              }
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
    // Generic runner: creates its own origin/result buffers sized to the
    // RaySetup so arbitrary region layouts (including expected-miss regions)
    // can be verified.
    auto run_trace_setup = [&](Accel &accel, RaySetup const &ray_setup,
                               luisa::string_view label) {
        std::vector<uint4> host_results(ray_setup.origins.size());
        auto local_origin = device.create_buffer<float3>(ray_setup.origins.size());
        auto local_result = device.create_buffer<uint4>(ray_setup.origins.size());
        stream << local_origin.copy_from(luisa::span{ray_setup.origins})
               << synchronize();
        stream << trace_shader(local_origin, local_result, accel)
                      .dispatch(ray_setup.origins.size())
               << local_result.copy_to(luisa::span{host_results})
               << synchronize();
        auto r = verify_results(ray_setup.origins, ray_setup.expected_inst,
                                ray_setup.max_prim, luisa::span{host_results},
                                label);
        if (!r.ok) { LUISA_INFO("{}", r.detail); }
        expect(r.ok) << r.detail;
        return r.ok;
    };
    // Creates a grid (grid == true) or quad mesh; BLAS compaction enabled via
    // `compaction` so both backends deterministically queue refresh entries on
    // a FORCE rebuild (vk on any recreate with a live accel, dx on compact).
    auto make_mesh = [&](bool grid, bool compaction) {
        AccelOption mesh_option{.hint = AccelUsageHint::FAST_TRACE,
                                .allow_compaction = compaction,
                                .allow_update = true};
        return grid ? device.create_mesh(grid_vb, grid_ib, mesh_option)
                    : device.create_mesh(quad_vb, quad_ib, mesh_option);
    };
    auto fast_trace_accel_option = [] {
        return AccelOption{.hint = AccelUsageHint::FAST_TRACE,
                           .allow_compaction = false,
                           .allow_update = true};
    };
    auto setup3 = make_three_region_setup();

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
    // ---- variant 8: BLAS recreate queues refresh entries, accel shrinks -----
    // Recreating a BLAS queues a per-slot refresh (vk queues on every FORCE
    // recreate; dx on growth/compaction). Shrinking the accel afterwards
    // destroys the pooled handles of the removed slots: the refresh entries
    // must not survive into the next build and rebind a recycled slot to a
    // stale BLAS. This covers the hazard fixed for dx (stale setMap on shrink)
    // and now hardened identically on vk.
    {
        AccelOption mesh_option{.hint = AccelUsageHint::FAST_TRACE,
                                .allow_compaction = true,
                                .allow_update = true};
        auto mesh_a = device.create_mesh(grid_vb, grid_ib, mesh_option);
        auto mesh_b = device.create_mesh(quad_vb, quad_ib, mesh_option);
        auto mesh_c = device.create_mesh(quad_vb, quad_ib, mesh_option);
        AccelOption accel_option{.hint = AccelUsageHint::FAST_TRACE,
                                 .allow_compaction = false,
                                 .allow_update = true};
        auto accel = device.create_accel(accel_option);
        accel.emplace_back(mesh_a, make_float4x4(1.0f), 0xffu, true);
        accel.emplace_back(mesh_b, translation(make_float3(10.0f, 0.0f, 0.0f)),
                           0xffu, true);
        accel.emplace_back(mesh_c, translation(make_float3(0.0f, 5.0f, 0.0f)),
                           0xffu, true);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << mesh_c.build(build_req)
               << accel.build(build_req) << synchronize();
        // Recreate every BLAS: queues refresh entries for slots 0..2 (vk
        // unconditionally on recreate; dx via compaction at the flush point).
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << mesh_c.build(build_req)
               << accel.build(build_req) << synchronize();
        // Shrink from 3 to 1 instance while refresh entries are still queued.
        accel.pop_back();
        accel.pop_back();
        accel.set_transform_on_update(0u, make_float4x4(1.0f));
        stream << accel.build(build_req) << synchronize();
        // Re-grow slot 1 with a fresh mesh and refit transform-only: slot 1
        // must keep the freshly-assigned quad, not a stale recycled BLAS.
        accel.emplace_back(mesh_b, translation(make_float3(10.0f, 0.0f, 0.0f)),
                           0xffu, true);
        accel.set_transform_on_update(1u,
                                      translation(make_float3(10.0f, 0.0f, 0.0f)));
        stream << mesh_b.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace(accel, "v8 recreate+shrink+regrow");
    }

    // ---- variant 9: pending refreshes + shrink 3->1 + trace-before-regrow --
    // Queues refresh entries for every slot (vk queues on any FORCE recreate;
    // dx queues via compaction), then shrinks 3 -> 1 BEFORE any TLAS build
    // consumes them. The out-of-range entries must be dropped (vk commit
    // 4c0d416f8, dx commit 48a388433) so removed slots can never be rebound
    // to a recycled BLAS or resurrect as phantom instances.
    {
        auto accel = device.create_accel(fast_trace_accel_option());
        auto mesh_a = make_mesh(true, true);
        auto mesh_b = make_mesh(false, true);
        auto mesh_c = make_mesh(false, true);
        accel.emplace_back(mesh_a, make_float4x4(1.0f), 0xffu, true);
        accel.emplace_back(mesh_b, translation(make_float3(10.0f, 0.0f, 0.0f)),
                           0xffu, true);
        accel.emplace_back(mesh_c, translation(make_float3(10.0f, 3.0f, 0.0f)),
                           0xffu, true);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << mesh_c.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace_setup(accel, setup3, "v9 initial");
        // Queue refresh entries for all three slots; do NOT build the TLAS.
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << mesh_c.build(build_req) << synchronize();
        accel.pop_back();
        accel.pop_back();
        stream << accel.build(build_req) << synchronize();
        // Region 0 (grid) must still hit; regions 1/2 must MISS: a surviving
        // stale entry would have rebound/rewritten a removed slot.
        auto shrunken = make_miss_expectation(setup3, rays_per_region,
                                              setup3.origins.size());
        run_trace_setup(accel, shrunken, "v9 shrunk 3->1 (regions 1,2 miss)");
        // Regrow both slots and re-verify the full scene.
        accel.emplace_back(mesh_b, translation(make_float3(10.0f, 0.0f, 0.0f)),
                           0xffu, true);
        accel.emplace_back(mesh_c, translation(make_float3(10.0f, 3.0f, 0.0f)),
                           0xffu, true);
        stream << accel.build(build_req) << synchronize();
        run_trace_setup(accel, setup3, "v9 regrown");
        // Transform-only refit loop; each FORCE rebuild re-queues entries that
        // must be consumed by the following build and never outlive it.
        for (auto f = 0u; f < 6u; f++) {
            accel.set_transform_on_update(0u, make_float4x4(1.0f));
            accel.set_transform_on_update(
                1u, translation(make_float3(10.0f, 0.0f, 0.0f)));
            accel.set_transform_on_update(
                2u, translation(make_float3(10.0f, 3.0f, 0.0f)));
            auto req = (f % 2u == 0u) ? AccelBuildRequest::PREFER_UPDATE
                                      : build_req;
            stream << accel.build(req) << synchronize();
        }
        run_trace_setup(accel, setup3, "v9 refit loop");
    }

    // ---- variant 10: destroy meshes with pending entries + pool reuse ------
    // Queue refresh entries for both slots, then destroy the meshes BEFORE the
    // TLAS build consumes the entries. The BLAS destructor must erase entries
    // pointing to the destroyed BLAS (~Blas / ~BottomAccel). Fresh meshes are
    // created immediately after so the MeshHandle pool recycles the freed
    // slots: if any stale entry survived it would now alias a different mesh
    // and a later build would rebind an instance to the wrong BLAS.
    {
        auto accel = device.create_accel(fast_trace_accel_option());
        auto mesh_a = make_mesh(true, true);
        auto mesh_b = make_mesh(false, true);
        emplace_ab(accel, mesh_a, mesh_b);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace(accel, "v10 initial");
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req) << synchronize();
        // Destroy both meshes while their refresh entries are still pending.
        mesh_a = Mesh{};
        mesh_b = Mesh{};
        // Replacement meshes reuse the just-freed MeshHandle pool slots.
        auto mesh_c = make_mesh(true, true);
        auto mesh_d = make_mesh(false, true);
        accel.set_mesh(0u, mesh_c);
        accel.set_mesh(1u, mesh_d);
        stream << mesh_c.build(build_req)
               << mesh_d.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace(accel, "v10 replaced after destroy");
        // Queue fresh entries and let a transform-only build consume them.
        stream << mesh_c.build(build_req)
               << mesh_d.build(build_req) << synchronize();
        accel.set_transform_on_update(0u, make_float4x4(1.0f));
        accel.set_transform_on_update(1u,
                                      translation(make_float3(10.0f, 0.0f, 0.0f)));
        stream << accel.build(build_req) << synchronize();
        run_trace(accel, "v10 refit after recreate");
    }

    // ---- variant 11: middle-slot kept vs dropped on 3->2 shrink ------------
    // Queue refresh entries for slot 0 and slot 2 only (slot 1 untouched),
    // then shrink 3 -> 2. The out-of-range entry for slot 2 must be dropped
    // while the in-range entry for slot 0 must survive and be consumed by the
    // following transform-only refit.
    {
        auto accel = device.create_accel(fast_trace_accel_option());
        auto mesh_a = make_mesh(true, true);
        auto mesh_b = make_mesh(false, true);
        auto mesh_c = make_mesh(false, true);
        accel.emplace_back(mesh_a, make_float4x4(1.0f), 0xffu, true);
        accel.emplace_back(mesh_b, translation(make_float3(10.0f, 0.0f, 0.0f)),
                           0xffu, true);
        accel.emplace_back(mesh_c, translation(make_float3(10.0f, 3.0f, 0.0f)),
                           0xffu, true);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << mesh_c.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace_setup(accel, setup3, "v11 initial");
        // Queue refresh entries for slot 0 and slot 2 only.
        stream << mesh_a.build(build_req)
               << mesh_c.build(build_req) << synchronize();
        accel.pop_back();
        stream << accel.build(build_req) << synchronize();
        auto shrunken = make_miss_expectation(setup3, 2u * rays_per_region,
                                              setup3.origins.size());
        run_trace_setup(accel, shrunken, "v11 shrunk 3->2 (region 2 miss)");
        // Transform-only refit of the kept slots: slot 0's pending refresh is
        // consumed here; slot 1 keeps its untouched state.
        accel.set_transform_on_update(0u, make_float4x4(1.0f));
        accel.set_transform_on_update(1u,
                                      translation(make_float3(10.0f, 0.0f, 0.0f)));
        stream << accel.build(AccelBuildRequest::PREFER_UPDATE) << synchronize();
        run_trace_setup(accel, shrunken, "v11 refit kept slots");
        // Regrow slot 2 and confirm the full scene.
        accel.emplace_back(mesh_c, translation(make_float3(10.0f, 3.0f, 0.0f)),
                           0xffu, true);
        stream << accel.build(build_req) << synchronize();
        run_trace_setup(accel, setup3, "v11 regrown");
    }

    // ---- variant 12: destroy/replace churn with pool reuse (3 slots) -------
    // Repeats the v10 pattern across multiple cycles: queue entries, destroy
    // every referenced mesh, replace with fresh meshes (which recycle the
    // pooled handles), rebuild and trace. Sustained correctness proves no
    // stale entry can rebind a slot to the BLAS of a different mesh.
    {
        auto accel = device.create_accel(fast_trace_accel_option());
        auto mesh_a = make_mesh(true, true);
        auto mesh_b = make_mesh(false, true);
        auto mesh_c = make_mesh(false, true);
        accel.emplace_back(mesh_a, make_float4x4(1.0f), 0xffu, true);
        accel.emplace_back(mesh_b, translation(make_float3(10.0f, 0.0f, 0.0f)),
                           0xffu, true);
        accel.emplace_back(mesh_c, translation(make_float3(10.0f, 3.0f, 0.0f)),
                           0xffu, true);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << mesh_c.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace_setup(accel, setup3, "v12 cycle0 initial");
        for (auto cycle = 1u; cycle <= 2u; cycle++) {
            // Queue refresh entries for every slot, then destroy all meshes.
            stream << mesh_a.build(build_req)
                   << mesh_b.build(build_req)
                   << mesh_c.build(build_req) << synchronize();
            mesh_a = Mesh{};
            mesh_b = Mesh{};
            mesh_c = Mesh{};
            // Replace with fresh meshes; pool recycles the freed handles.
            mesh_a = make_mesh(true, true);
            mesh_b = make_mesh(false, true);
            mesh_c = make_mesh(false, true);
            accel.set_mesh(0u, mesh_a);
            accel.set_mesh(1u, mesh_b);
            accel.set_mesh(2u, mesh_c);
            stream << mesh_a.build(build_req)
                   << mesh_b.build(build_req)
                   << mesh_c.build(build_req)
                   << accel.build(build_req) << synchronize();
            run_trace_setup(accel, setup3,
                            luisa::format("v12 cycle{} replaced", cycle));
        }
    }

    // ---- variant 13: entries consumed exactly once / repeated builds -------
    // Each frame re-queues a refresh entry for slot 0 (FORCE rebuild), then
    // builds the TLAS twice. The first build must consume the entry; the
    // second must not re-apply anything (a leftover entry would be harmless
    // here only because the BLAS address is unchanged — the trace still
    // verifies the instance table and hit inst/prim stay correct).
    {
        auto accel = device.create_accel(fast_trace_accel_option());
        auto mesh_a = make_mesh(true, true);
        auto mesh_b = make_mesh(false, true);
        emplace_ab(accel, mesh_a, mesh_b);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << accel.build(build_req) << synchronize();
        run_trace(accel, "v13 initial");
        for (auto f = 0u; f < 6u; f++) {
            stream << mesh_a.build(build_req) << synchronize();
            auto first_req = (f % 2u == 0u) ? build_req
                                            : AccelBuildRequest::PREFER_UPDATE;
            auto second_req = (f % 2u == 0u) ? AccelBuildRequest::PREFER_UPDATE
                                             : build_req;
            stream << accel.build(first_req) << synchronize();
            stream << accel.build(second_req) << synchronize();
            run_trace(accel, luisa::format("v13 frame {}", f));
        }
    }

    // ---- variant 14: teardown orders with pending entries ------------------
    // Pending refresh entries must never outlive either their owning BLAS or
    // their owning TLAS; both destruction orders must be crash-free.
    {
        // 14a: destroy the TLAS (with pending entries) before the meshes.
        {
            auto accel = device.create_accel(fast_trace_accel_option());
            auto mesh_a = make_mesh(true, true);
            auto mesh_b = make_mesh(false, true);
            emplace_ab(accel, mesh_a, mesh_b);
            stream << mesh_a.build(build_req)
                   << mesh_b.build(build_req)
                   << accel.build(build_req) << synchronize();
            stream << mesh_a.build(build_req)
                   << mesh_b.build(build_req) << synchronize();
            accel = Accel{};// destroys the TLAS with refresh entries pending
            stream << synchronize();
        }// meshes destroyed here
        // 14b: destroy the meshes (with pending entries) before the TLAS.
        {
            auto accel = device.create_accel(fast_trace_accel_option());
            auto mesh_a = make_mesh(true, true);
            auto mesh_b = make_mesh(false, true);
            emplace_ab(accel, mesh_a, mesh_b);
            stream << mesh_a.build(build_req)
                   << mesh_b.build(build_req)
                   << accel.build(build_req) << synchronize();
            stream << mesh_a.build(build_req)
                   << mesh_b.build(build_req) << synchronize();
            mesh_a = Mesh{};
            mesh_b = Mesh{};
            stream << synchronize();
            accel = Accel{};// destroyed after its meshes
            stream << synchronize();
        }
    }

    // ---- variant 15: shared meshes, pending entries, destroy + replace -----
    // Two accels share both meshes. A BLAS recreate queues a refresh entry in
    // EVERY referencing accel; destroying the shared meshes must erase the
    // entries in all of them, and replacing the instances must keep both
    // accels consistent (correct TLAS slot, correct BLAS).
    {
        auto accel1 = device.create_accel(fast_trace_accel_option());
        auto accel2 = device.create_accel(fast_trace_accel_option());
        auto mesh_a = make_mesh(true, true);
        auto mesh_b = make_mesh(false, true);
        emplace_ab(accel1, mesh_a, mesh_b);
        emplace_ab(accel2, mesh_a, mesh_b);
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req)
               << accel1.build(build_req)
               << accel2.build(build_req) << synchronize();
        run_trace(accel1, "v15 shared initial accel1");
        run_trace(accel2, "v15 shared initial accel2");
        // Queue refresh entries in both accels, then destroy shared meshes.
        stream << mesh_a.build(build_req)
               << mesh_b.build(build_req) << synchronize();
        mesh_a = Mesh{};
        mesh_b = Mesh{};
        auto mesh_c = make_mesh(true, true);
        auto mesh_d = make_mesh(false, true);
        accel1.set_mesh(0u, mesh_c);
        accel1.set_mesh(1u, mesh_d);
        accel2.set_mesh(0u, mesh_c);
        accel2.set_mesh(1u, mesh_d);
        stream << mesh_c.build(build_req)
               << mesh_d.build(build_req)
               << accel1.build(build_req)
               << accel2.build(build_req) << synchronize();
        run_trace(accel1, "v15 replaced accel1");
        run_trace(accel2, "v15 replaced accel2");
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_accel_tlas_instances(dc->device);
}