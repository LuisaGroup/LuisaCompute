// Cluster (RTX Mega Geometry) acceleration structure test.
//
// Builds the same three triangles twice: once as a VK_NV_cluster_acceleration_structure
// cluster (mega-geometry) BLAS from an array of ClusterBuildInfo records, and once as
// an ordinary triangle Mesh. The identical ray set is then traced through both
// acceleration structures with the trace-closest ray-query path, and the committed hit
// distance plus the hit/miss decision are compared ray by ray - against each other and
// against an analytically computed t. That cross-check is what proves the cluster path
// produces correct geometry instead of merely "not crashing".
//
// The two clusters deliberately do *not* overlap in XY and do *not* declare the same
// vertex/triangle counts: cluster 0 is the square x, y in [-1, 1] in the plane z = 0,
// cluster 1 is a triangle at z = -2 whose footprint is x in [2, 4]. Both properties are
// load bearing - they turn "a cluster record points at the wrong vertex block" from an
// invisible change of t into a changed hit/miss decision (see the two notes below).
//
// Two submission patterns are exercised, because the cluster build takes raw device
// addresses that the command graph cannot see (documented on
// Device::create_mesh(ClusterBuildInfo array) in <luisa/runtime/rtx/mesh.h>):
//   1. the uploads in their own, earlier flush (the safe pattern), and
//   2. uploads + cluster build + TLAS build in one flush with command reordering turned
//      off through CommandReorderExt, i.e. strict submission order. Pattern 2 is what a
//      user writes naturally and is only deterministic because reordering is off; the
//      env-var equivalent (LUISA_DISABLE_COMMAND_REORDER=1) forces the same thing for
//      the whole process, the extension does it for one device without touching the
//      process environment.
//
// The test never fails on unsupported hardware: a non-Vulkan backend, or a Vulkan
// device that does not report VK_NV_cluster_acceleration_structure, prints an
// unmistakable "SKIPPED:" line and exits with skip_exit_code (77). CMakeLists.txt
// registers 77 as CTest's SKIP_RETURN_CODE, so CI records the run as *skipped* instead
// of silently passing it.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/command_reorder_ext.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Exit status of a run that could not exercise the cluster path (no vk backend, or a
// Vulkan device without VK_NV_cluster_acceleration_structure). Non-zero so it cannot be
// mistaken for a pass in CI, but registered as CTest's SKIP_RETURN_CODE so it is not a
// failure either. 77 is the automake "skipped" convention and is already used for this
// purpose in src/ext/llvm-downgrade/test/CMakeLists.txt.
constexpr int skip_exit_code = 77;

namespace test_rtx_cluster_detail {

//--------------------------------------------------------------- geometry ---
// Two clusters with disjoint XY footprints and different sizes:
//
//   cluster 0: the quad (-1,-1,0) (1,-1,0) (1,1,0) (-1,1,0), 4 vertices, 2 triangles,
//              i.e. the square x, y in [-1, 1] in the plane z = 0, cut by the diagonal
//              y = x into (0,1,2) [below the diagonal] and (0,2,3) [above it];
//   cluster 1: the right triangle (2,-1,-2) (4,-1,-2) (2,1,-2), 3 vertices, 1 triangle,
//              interior = { x > 2, y > -1, x + y < 3 } in the plane z = -2.
//
// What the ray table below buys from that layout:
//   * x = 0.25 can only be satisfied by cluster 0, x = 3 only by cluster 1;
//   * x = 1.5 is the gap between the two footprints, so a downward ray starting between
//     the two planes (z = -1) must miss *both* - the sharpest probe that each record
//     reads its own vertex block instead of a shifted or shared one;
//   * the two records declare 4/2 and 3/1 vertices/triangles, so they are not
//     interchangeable and a permutation of the two records changes the covered
//     geometry instead of only which cluster owns it.
//
// What a ray-only oracle can and cannot see (F2):
// A ray query reports only "hit at t" (the DSL does not expose the per-cluster
// ClusterIDNV), so it constrains the *union* of geometry the acceleration structure
// covers. A permutation of two interchangeable records - same vertex count, same index
// list - leaves that union unchanged and is therefore invisible to any ray set, however
// dense; the previous version of this test used two congruent quads and could not see a
// swap at all. The asymmetric layout above closes most of that gap, because the records
// are no longer interchangeable. Still out of reach: winding/orientation (every ray here
// hits the front face) and per-cluster flags/opacity, which the test leaves at their
// documented defaults. Do not read the two-cluster setup as full per-cluster validation.
constexpr uint cluster_count = 2u;

// Per-cluster geometry sizes carried by the ClusterBuildInfo records. They differ on
// purpose (see the note above); the vertex/index blocks keep a uniform stride so the
// per-cluster device address is a simple multiple, exactly like a real producer would
// lay out a cluster page.
constexpr std::array<uint, cluster_count> cluster_vertex_counts{4u, 3u};
constexpr std::array<uint, cluster_count> cluster_triangle_counts{2u, 1u};

constexpr uint max_vertices_per_cluster = 4u;
constexpr uint max_triangles_per_cluster = 2u;
constexpr uint max_indices_per_cluster = max_triangles_per_cluster * 3u;// 6

// Number of vertices the reference triangle mesh and the vertex buffer span: one
// 4-slot block per cluster.
constexpr uint vertex_slots_per_cluster = max_vertices_per_cluster;// 4
constexpr uint vertex_slot_count = cluster_count * vertex_slots_per_cluster;// 8

// Triangles covered by the whole test geometry, i.e. the sum of the per-cluster counts.
// Derived from cluster_triangle_counts so it can never drift from the records.
constexpr uint mesh_triangle_count = [] {
    uint n = 0u;
    for (auto t : cluster_triangle_counts) { n += t; }
    return n;
}();

// Positions are stored as three *separate* floats per vertex (12 B per vertex): the
// cluster build declares VK_FORMAT_R32G32B32_SFLOAT with vertexBufferStride = 0, i.e.
// tightly packed positions and nothing else. luisa::float3 cannot be used for that -
// Vector<float, 3> is alignas(16), so sizeof(float3) == 16 and a Buffer<float3> is
// 16-byte strided. Using it here makes the driver read every vertex but the first one
// at the wrong offset (the first run of this test did exactly that and every ray
// missed).
constexpr size_t position_floats_per_vertex = 3u;
constexpr size_t position_bytes_per_vertex = position_floats_per_vertex * sizeof(float);// 12 B

// `size` rounded up to a multiple of `alignment`. Same arithmetic as luisa::align, but
// constexpr so it can size the arrays below - and self-checking, unlike the hand-written
// 16u of the previous version: growing max_indices_per_cluster past a 16-byte block
// automatically widens the block instead of silently overlapping the next cluster.
constexpr size_t round_up(size_t size, size_t alignment) noexcept {
    return size + (alignment - size % alignment) % alignment;
}

// Byte strides of the per-cluster blocks inside the shared vertex / index buffer.
// ClusterBuildInfo carries *device addresses* for index_buffer / vertex_buffer, so the
// per-cluster address is base + (cluster index * these strides). Both blocks are
// multiples of 16 bytes, so every cluster base is 16-byte aligned.
constexpr size_t vertex_block_stride_bytes = vertex_slots_per_cluster * position_bytes_per_vertex;// 48 B
constexpr size_t index_block_stride_bytes = round_up(max_indices_per_cluster * sizeof(ushort), 16u);// 16 B
constexpr size_t index_element_count =
    cluster_count * index_block_stride_bytes / sizeof(ushort);// 16 elements

constexpr size_t position_float_count = vertex_slot_count * position_floats_per_vertex;// 24 floats

constexpr std::array<float, position_float_count> cluster_vertex_positions{
    // block 0 -> cluster 0, the quad in the plane z = 0 (x, y in [-1, 1])
    -1.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f,
    // block 1 -> cluster 1, the triangle in the plane z = -2 (x in [2, 4])
    2.0f, -1.0f, -2.0f,
    4.0f, -1.0f, -2.0f,
    2.0f, 1.0f, -2.0f,
    // Unused 4th slot of block 1: cluster 1 declares only 3 vertices, so a correct build
    // never reads it. It exists so that a record with a *wrong* vertex_count (or a
    // record pointed at the neighbouring block) stays inside the allocation and fails
    // with a changed hit/miss table instead of an out-of-bounds device read. Kept far
    // away from every ray in the table below.
    1000.0f, 1000.0f, 1000.0f};

// 16-bit indices, relative to the owning cluster's vertexBuffer device address, so both
// clusters index their own block from 0. The trailing zeros of each block only pad it out
// to index_block_stride_bytes; a cluster that declared more triangles than it owns would
// read them, which is deliberate (see the note on the unused vertex slot).
constexpr std::array<ushort, index_element_count> cluster_indices{
    0u, 1u, 2u, 0u, 2u, 3u, 0u, 0u,// cluster 0: triangles (0,1,2) and (0,2,3) + padding
    0u, 1u, 2u, 0u, 0u, 0u, 0u, 0u};// cluster 1: triangle (0,1,2) + padding

// The very same triangles expressed as one plain triangle mesh: indices into
// cluster_vertex_positions (cluster 1's block starts at vertex slot 4), so the reference
// BLAS traces identical geometry.
constexpr std::array<uint, mesh_triangle_count * 3u> mesh_indices{
    0u, 1u, 2u, 0u, 2u, 3u,// cluster 0's quad
    4u, 5u, 6u};           // cluster 1's triangle

// Rays. t_min = 0 and t_max = 1e30 in the kernel, so a hit is only ever reported for
// t > 0 and a plane *behind* the origin never hits. All rays are kept off the quad
// edges, off the shared 0-2 diagonal (y = x) of cluster 0 and off cluster 1's
// hypotenuse (x + y = 3), so no assertion depends on a tie-break between two
// neighbouring triangles or on a driver's edge rule.
struct ray_spec {
    float3 origin;
    float3 direction;
    bool expect_hit;
    float expected_t;// analytically expected committed t when expect_hit
};

constexpr std::array<ray_spec, 12u> rays{{
    // 0: straight down through cluster 0's lower-right triangle; x = 0.25 is outside
    // cluster 1's footprint (x in [2, 4]), so only cluster 0 can satisfy it.
    //   z = 0 at t = 1 -> (0.25, 0, 0); y = 0 < x = 0.25 -> triangle (0,1,2).
    {{0.25f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}, true, 1.0f},
    // 1: same plane, but the *other* triangle of the quad (above the diagonal y = x),
    // so a build that only ever reads the first triangle is caught.
    //   z = 0 at t = 1 -> (-0.5, 0.5, 0); y = 0.5 > x = -0.5 -> triangle (0,2,3).
    {{-0.5f, 0.5f, 1.0f}, {0.0f, 0.0f, -1.0f}, true, 1.0f},
    // 2: straight down through cluster 1's triangle; x = 3 is outside cluster 0's
    // footprint, so only cluster 1 can satisfy it, and only because it sits at z = -2.
    //   z = -2 at t = 3 -> (3, -0.5, -2); 3 > 2, -0.5 > -1, 3 - 0.5 < 3 -> inside.
    //   A cluster 0 that covers x = 3 would hit at t = 1 instead.
    {{3.0f, -0.5f, 1.0f}, {0.0f, 0.0f, -1.0f}, true, 3.0f},
    // 3: the gap between the two footprints, starting between the two planes. Cluster 0
    // is behind the origin (t < 0) and cluster 1 does not reach x = 1.5, so this must
    // miss - if either record reads a shifted vertex block, it can still miss, but if
    // cluster 1 reads cluster 0's block *at the wrong plane* or the block stride is
    // dropped, this ray turns into a hit.
    {{1.5f, 0.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, false, 0.0f},
    // 4: between the planes, just outside cluster 1's hypotenuse: at (3, 0.5) the value
    // x + y = 3.5 exceeds 3, i.e. ~0.35 outside the edge -> miss.
    {{3.0f, 0.5f, -1.0f}, {0.0f, 0.0f, -1.0f}, false, 0.0f},
    // 5: vertically through the *centre of cluster 0's footprint*, but from between the
    // planes: cluster 0's plane is behind the origin and cluster 1's footprint starts at
    // x = 2 -> miss. Distance and footprint are independent facts, not one fact.
    {{0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, false, 0.0f},
    // 6: just outside cluster 0's +X edge (x = 1.05 > 1) while crossing both planes
    // -> miss. Pins the quad's X *extent*, which a t-only oracle never constrains.
    {{1.05f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}, false, 0.0f},
    // 7: just outside cluster 0's +Y edge (y = 1.05 > 1) -> miss. Pins the Y extent.
    {{0.0f, 1.05f, 1.0f}, {0.0f, 0.0f, -1.0f}, false, 0.0f},
    // 8: between the planes, just outside cluster 1's +X edge (x = 4.05 > 4) -> miss.
    // Pins cluster 1's own extent (the far corner of its vertex block).
    {{4.05f, -0.5f, -1.0f}, {0.0f, 0.0f, -1.0f}, false, 0.0f},
    // 9: oblique. Because dz = -1, z = 0 is reached at t = 1 and z = -2 at t = 3, so
    // this ray pierces cluster 0 at (0.5, -0.5, 0) and cluster 1 at (2.5, -0.5, -2):
    //   (0.5, -0.5, 0):  -0.5 > -1, 0.5 < 1, -0.5 < 0.5  -> inside (0,1,2)
    //   (2.5, -0.5, -2): 2.5 > 2, -0.5 > -1, 2.0 < 3      -> inside cluster 1
    // The closest hit is t = 1; a broken closest-hit test reports t = 3 (delta 2).
    {{-0.5f, -0.5f, 1.0f}, {1.0f, 0.0f, -1.0f}, true, 1.0f},
    // 10: oblique on cluster 1. |d| = sqrt(1.0625) ~ 1.031, so a t that had been
    // normalised would read 3.09 instead of 3: this pins "committed t is the raw ray
    // parameter", the same role the oblique ray played before.
    //   z = -2 at t = 3 -> (2.75, -0.5, -2); 2.75 > 2, -0.5 > -1, 2.25 < 3 -> inside.
    {{2.0f, -0.5f, 1.0f}, {0.25f, 0.0f, -1.0f}, true, 3.0f},
    // 11: points away from both quads, and both planes are behind the origin
    // (negative t) -> miss.
    {{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, false, 0.0f},
}};

constexpr size_t ray_count = rays.size();
constexpr float ray_t_max = 1.0e30f;

// Tightly packed positions, i.e. the stride the reference triangle mesh has to be told
// about explicitly (Buffer<float> itself only knows sizeof(float)).
constexpr size_t position_stride_bytes = position_bytes_per_vertex;

// Relative tolerance for "the cluster t and the triangle t agree". The two paths may
// pick a different triangle along a shared edge and may evaluate the same plane with a
// different rounding order, but they must agree on the committed distance to well below
// a thousandth of a unit.
constexpr float relative_tolerance = 1.0e-3f;

// The published BLAS address is aligned up to
// VkPhysicalDeviceClusterAccelerationStructurePropertiesNV::clusterBottomLevelByteAlignment
// by the backend (256 B on the RTX 4060), so 16 B is a conservative floor: it still
// catches a published address that is not an address at all (a size, a handle, a
// truncated 32-bit value) without failing on a driver that reports a smaller alignment.
constexpr uint64_t cluster_address_alignment = 16u;

[[nodiscard]] bool t_matches(float a, float b) noexcept {
    auto scale = std::max(1.0f, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= relative_tolerance * scale;
}

// What one traced ray set reports, in the order of `rays`.
struct traced_rays {
    std::array<uint, ray_count> hits{};
    std::array<float, ray_count> distances{};
};

// Result of the whole test: the number of failed checks plus whether the cluster path
// could not be exercised at all (which main() turns into skip_exit_code, not into a
// pass - see F13 in REVIEW_F4.md).
struct test_outcome {
    uint failures{0u};
    bool skipped{false};
};

[[nodiscard]] test_outcome run_rtx_cluster_test(Device &device) noexcept {

    log_level_info();

    //-------------------------------------------------------- 1. skip gate ----
    // Cluster acceleration structures are a Vulkan-only feature (VK_NV_-
    // cluster_acceleration_structure); there is no DX/CUDA/Metal counterpart in
    // this tree yet, so every other backend logs a skip and passes.
    if (auto backend = device.backend_name(); backend != "vk") {
        LUISA_WARNING(
            "SKIPPED: [test_rtx_cluster] VK_NV_cluster_acceleration_structure is "
            "Vulkan-only, but the test device backend is '{}'. Nothing was validated.",
            backend);
        return {0u, true};
    }
    auto cluster_as = device.query("vk_cluster_acceleration_structure");
    if (cluster_as != "1") {
        LUISA_WARNING(
            "SKIPPED: [test_rtx_cluster] this Vulkan device reports "
            "vk_cluster_acceleration_structure = '{}'. The cluster "
            "(mega-geometry) path needs VK_NV_cluster_acceleration_structure "
            "together with ray tracing and VK_KHR_buffer_device_address, i.e. an "
            "NVIDIA RTX GPU on a driver that exposes the extension. Nothing was "
            "validated.",
            cluster_as);
        return {0u, true};
    }
    LUISA_INFO(
        "[test_rtx_cluster] backend={} vk_cluster_acceleration_structure={}",
        device.backend_name(), cluster_as);

    //----------------------------------------------------------- 2. inputs ----
    Stream stream = device.create_stream();

    // Tightly packed positions (3 floats per vertex) - see the comment on
    // position_bytes_per_vertex: a Buffer<float3> would be 16-byte strided and
    // every cluster vertex but the first would be read at the wrong offset.
    Buffer<float> vertex_buffer = device.create_buffer<float>(position_float_count);
    Buffer<ushort> index_buffer = device.create_buffer<ushort>(index_element_count);
    Buffer<ClusterBuildInfo> cluster_buffer = device.create_buffer<ClusterBuildInfo>(cluster_count);

    // A cluster build record carries raw device addresses instead of buffer
    // handles, so a 0 here means VK_KHR_buffer_device_address was not enabled
    // for these buffers. That is a backend bug, not a reason to skip.
    auto vertex_base = vertex_buffer.device_address();
    auto index_base = index_buffer.device_address();
    if (vertex_base == 0u || index_base == 0u) {
        LUISA_ERROR(
            "Cluster build records need buffer device addresses, but "
            "Buffer::device_address() returned vertex_base=0x{:x} index_base=0x{:x}.",
            vertex_base, index_base);
    }

    //------------------------------------------------- 3. cluster records -----
    std::array<ClusterBuildInfo, cluster_count> records{};
    for (auto i = 0u; i < cluster_count; i++) {
        auto index_address = index_base + static_cast<uint64_t>(i) * index_block_stride_bytes;
        auto vertex_address = vertex_base + static_cast<uint64_t>(i) * vertex_block_stride_bytes;
        if (index_address == 0u || vertex_address == 0u) {
            LUISA_ERROR(
                "Cluster {} has a null index/vertex address (index=0x{:x} vertex=0x{:x}); "
                "cluster build records cannot be filled.",
                i, index_address, vertex_address);
        }
        records[i] = make_cluster_build_info(
            i, cluster_triangle_counts[i], cluster_vertex_counts[i],
            index_address, vertex_address,
            ClusterIndexFormat::UInt16,// 16-bit indices, like cluster_indices
            /*base_geometry_index=*/0u,
            ClusterGeometryFlags::Opaque);
        LUISA_INFO(
            "[test_rtx_cluster] cluster {}: triangles={} vertices={} index_type={} "
            "index_stride={} vertex_stride={} index_va=0x{:x} vertex_va=0x{:x}",
            i, records[i].triangle_count(), records[i].vertex_count(),
            static_cast<uint>(records[i].index_type()),
            records[i].index_buffer_stride, records[i].vertex_buffer_stride,
            index_address, vertex_address);
    }

    uint failures = 0u;
    auto check = [&failures](bool condition, luisa::string message) {
        if (!condition) { failures++; }
        boost::ut::expect(condition) << message;
    };

    // The three uploads. Kept in one place because both submission patterns below need
    // exactly the same host data on the device.
    auto upload_host_data = [&] {
        stream << vertex_buffer.copy_from(luisa::span{cluster_vertex_positions})
               << index_buffer.copy_from(luisa::span{cluster_indices})
               << cluster_buffer.copy_from(luisa::span{records});
    };

    //------------------------------------ 4. reference geometry, unclustered ----
    // The same vertex buffer, read with the same tightly packed 12-byte stride the
    // cluster records declare, so both acceleration structures trace bit-identical
    // positions. Built once; every scenario below compares against it.
    Buffer<Triangle> mesh_triangles = device.create_buffer<Triangle>(mesh_triangle_count);
    Mesh triangle_mesh = device.create_mesh(vertex_buffer, position_stride_bytes, mesh_triangles);
    Accel triangle_accel = device.create_accel();
    // One instance, identity transform, so both acceleration structures have exactly the
    // same instance layout and the committed t values are directly comparable.
    triangle_accel.emplace_back(triangle_mesh);
    stream << mesh_triangles.copy_from(luisa::span{mesh_indices})
           << triangle_mesh.build()
           << triangle_accel.build()
           << synchronize();

    //------------------------------------------------------- 5. ray query -----
    // One ray per dispatch element. trace-closest is accel.intersect(), the same form
    // test_rtx.cpp uses; on a miss it reports inst == 0xffffffff, prim == 0 and
    // committed_ray_t == 0, so both the decision and the distance are well-defined.
    Kernel1D trace_kernel = [](AccelVar accel,
                               BufferFloat3 ray_origins,
                               BufferFloat3 ray_directions,
                               BufferFloat distances,
                               BufferUInt hit_flags) noexcept {
        UInt i = dispatch_id().x;
        Var<Ray> ray = make_ray(ray_origins.read(i), ray_directions.read(i), 0.0f, ray_t_max);
        Var<SurfaceHit> hit = accel.intersect(ray, {});
        distances.write(i, hit->distance());
        hit_flags.write(i, ite(hit->miss(), 0u, 1u));
    };
    auto trace_shader = device.compile(trace_kernel);

    Buffer<float3> ray_origins = device.create_buffer<float3>(ray_count);
    Buffer<float3> ray_directions = device.create_buffer<float3>(ray_count);
    std::array<float3, ray_count> host_origins{};
    std::array<float3, ray_count> host_directions{};
    for (size_t i = 0u; i < ray_count; i++) {
        host_origins[i] = rays[i].origin;
        host_directions[i] = rays[i].direction;
    }

    // Trace the whole ray table through one acceleration structure.
    auto trace_rays = [&](const Accel &accel) {
        Buffer<float> distances = device.create_buffer<float>(ray_count);
        Buffer<uint> hits = device.create_buffer<uint>(ray_count);
        traced_rays traced;
        stream << ray_origins.copy_from(luisa::span{host_origins})
               << ray_directions.copy_from(luisa::span{host_directions})
               << trace_shader(accel, ray_origins, ray_directions, distances, hits)
                      .dispatch(ray_count)
               << distances.copy_to(luisa::span{traced.distances})
               << hits.copy_to(luisa::span{traced.hits})
               << synchronize();
        return traced;
    };

    auto distance_or_nan = [](uint hit, float t) noexcept {
        return hit == 0u ? std::numeric_limits<float>::quiet_NaN() : t;
    };

    // The independent oracle: the hand-computed hit/miss decision and t of every ray.
    auto check_against_oracle = [&](const char *label,
                                    const traced_rays &traced) {
        for (size_t i = 0u; i < ray_count; i++) {
            auto const &r = rays[i];
            auto t = distance_or_nan(traced.hits[i], traced.distances[i]);
            LUISA_INFO(
                "[test_rtx_cluster] {} ray {}: origin=({}, {}, {}) direction=({}, {}, {}) "
                "expected_hit={} expected_t={} | hit={} t={}",
                label, i, r.origin.x, r.origin.y, r.origin.z,
                r.direction.x, r.direction.y, r.direction.z,
                r.expect_hit, r.expected_t, traced.hits[i], t);
            if (r.expect_hit) {
                check(traced.hits[i] != 0u,
                      luisa::format("{} ray {}: the cluster BLAS must hit the geometry this ray "
                                    "can only reach through one cluster, but it reported a miss",
                                    label, i));
                if (traced.hits[i] != 0u) {
                    check(t_matches(traced.distances[i], r.expected_t),
                          luisa::format("{} ray {}: t={} does not match the analytic t={}",
                                        label, i, traced.distances[i], r.expected_t));
                }
            } else {
                check(traced.hits[i] == 0u,
                      luisa::format("{} ray {}: must miss every cluster, but it hit at t={}",
                                    label, i, t));
            }
        }
    };

    // "Wrong but consistent" is the failure mode the oracle cannot see: if the cluster
    // BLAS and the oracle shared a bug the analytic t would still match. Comparing the
    // cluster BLAS against a plain triangle BLAS built from the same vertices is an
    // independent second opinion for the distance, and the hit/miss arm of the oracle is
    // what pins the footprint.
    auto check_against_reference = [&](const char *label,
                                       const traced_rays &cluster_traced,
                                       const traced_rays &triangle_traced) {
        for (size_t i = 0u; i < ray_count; i++) {
            auto cluster_t = distance_or_nan(cluster_traced.hits[i], cluster_traced.distances[i]);
            auto triangle_t = distance_or_nan(triangle_traced.hits[i], triangle_traced.distances[i]);
            LUISA_INFO(
                "[test_rtx_cluster] {} ray {}: cluster_hit={} cluster_t={} | "
                "triangle_hit={} triangle_t={}",
                label, i, cluster_traced.hits[i], cluster_t,
                triangle_traced.hits[i], triangle_t);
            auto both_hit = cluster_traced.hits[i] != 0u && triangle_traced.hits[i] != 0u;
            auto both_miss = cluster_traced.hits[i] == 0u && triangle_traced.hits[i] == 0u;
            if (rays[i].expect_hit) {
                // Both arms are asserted separately, so a both_hit == false cannot
                // silently skip the distance comparison.
                check(cluster_traced.hits[i] != 0u,
                      luisa::format("{} ray {}: the cluster BLAS must hit the quad, but it "
                                    "reported a miss",
                                    label, i));
                check(triangle_traced.hits[i] != 0u,
                      luisa::format("{} ray {}: the reference triangle BLAS must hit the quad, "
                                    "but it reported a miss",
                                    label, i));
                if (both_hit) {
                    check(t_matches(cluster_traced.distances[i], triangle_traced.distances[i]),
                          luisa::format("{} ray {}: cluster t={} and triangle t={} disagree "
                                        "(tolerance {} relative)",
                                        label, i, cluster_traced.distances[i],
                                        triangle_traced.distances[i], relative_tolerance));
                }
            } else {
                check(both_miss,
                      luisa::format("{} ray {}: must miss every quad, but cluster_hit={} (t={}) "
                                    "and triangle_hit={} (t={})",
                                    label, i, cluster_traced.hits[i], cluster_t,
                                    triangle_traced.hits[i], triangle_t));
            }
        }
    };

    //--------------------------------- 6. scenario 1: uploads in their own flush ----
    // Flush 1 puts the vertex data, the index data *and* the build records on the device
    // before anything reads them. Flush 2 records the cluster build itself.
    //
    // Why this split: nothing *guarantees* the relative order of an upload and a cluster
    // build that shares its batch. The command-reorder pass sees only the records buffer
    // as a dependency of a cluster build - the vertex and index buffers are reachable
    // only through the raw device addresses stored inside those records, which the pass
    // cannot follow (see the "TODO(cluster)" note in
    // src/backends/common/command_reorder_visitor.h). For *this particular* submission
    // the order happens to be preserved anyway (each copy lands in layer 0 of its own
    // buffer and the build consequently in layer 1), but that is an accident of layer
    // arithmetic, not a guarantee: one more write to the vertex buffer, or a dispatch
    // that fills it from a buffer with a longer history, is enough to lose it. The split
    // flush is the pattern the public header documents, so the test uses it here and
    // exercises the single-flush pattern separately in scenario 2, where command
    // reordering is switched off and strict submission order makes it deterministic.
    upload_host_data();
    stream << synchronize();

    Mesh cluster_mesh = device.create_mesh(cluster_buffer, {});
    stream << cluster_mesh.build() << synchronize();

    //--------------------------------------------------- 7. mesh sanity -------
    // Host-side bookkeeping only, *not* device-side validation: the backend publishes
    // the pool address while the build command is being *recorded*, and cluster_count()
    // is arithmetic over the size of the buffer the test itself uploaded, so both would
    // also be right if the GPU had produced an empty BLAS. The evidence that the device
    // really built the geometry is the ray table below; these two checks only catch a
    // mis-wired binding (wrong buffer, wrong offset, non-address value).
    auto cluster_address = cluster_mesh.cluster_address();
    auto reported_cluster_count = cluster_mesh.cluster_count();
    LUISA_INFO("[test_rtx_cluster] cluster BLAS address=0x{:x} cluster_count={}",
               cluster_address, reported_cluster_count);
    check(cluster_address != 0u,
          luisa::format("mesh.cluster_address() must be non-zero once a build has been "
                        "recorded (host-cached pool address), got 0x{:x}",
                        cluster_address));
    check(cluster_address % cluster_address_alignment == 0u,
          luisa::format("mesh.cluster_address()=0x{:x} is not a {}-byte aligned device "
                        "address; the backend aligns it to the driver's "
                        "clusterBottomLevelByteAlignment",
                        cluster_address, cluster_address_alignment));
    check(reported_cluster_count == static_cast<uint>(records.size()),
          luisa::format("mesh.cluster_count()={} must equal the {} ClusterBuildInfo records "
                        "in the mesh's cluster buffer",
                        reported_cluster_count, records.size()));
    check(cluster_mesh.mode() == Mesh::Mode::Cluster,
          luisa::format("mesh.mode()={}, expected Mesh::Mode::Cluster ({})",
                        static_cast<uint>(cluster_mesh.mode()),
                        static_cast<uint>(Mesh::Mode::Cluster)));

    Accel cluster_accel = device.create_accel();
    // Emplaced before the build so this scenario also covers "mesh build and TLAS build
    // in the same batch": the TLAS pre-build bakes the instance's
    // accelerationStructureReference from the mesh's device address, i.e. it depends on
    // the cluster build having completed.
    cluster_accel.emplace_back(cluster_mesh);
    stream << cluster_accel.build() << synchronize();

    auto cluster_traced = trace_rays(cluster_accel);
    auto triangle_traced = trace_rays(triangle_accel);
    check_against_oracle("split-flush", cluster_traced);
    check_against_oracle("split-flush/reference", triangle_traced);
    check_against_reference("split-flush", cluster_traced, triangle_traced);

    //----------------- 8. scenario 2: the natural single-flush pattern, no reordering ----
    // A user writing the obvious code uploads and builds in one batch. That is only
    // deterministic when the backend submits in strict order, which is exactly what
    // CommandReorderExt::set_command_reorder_enabled(false) selects (and what
    // LUISA_DISABLE_COMMAND_REORDER=1 forces for the whole process - the extension is
    // the in-process equivalent, no environment trickery and no second device).
    auto *reorder_ext = device.extension<CommandReorderExt>();
    if (reorder_ext == nullptr) {
        LUISA_WARNING(
            "[test_rtx_cluster] backend '{}' exposes no CommandReorderExt; skipping the "
            "single-flush scenario. The split-flush scenario above already validated the "
            "cluster geometry.",
            device.backend_name());
    } else {
        reorder_ext->set_command_reorder_enabled(false);
        LUISA_INFO("[test_rtx_cluster] command reorder enabled={} for the single-flush scenario",
                   reorder_ext->command_reorder_enabled());

        Mesh single_flush_mesh = device.create_mesh(cluster_buffer, {});
        Accel single_flush_accel = device.create_accel();
        single_flush_accel.emplace_back(single_flush_mesh);
        // One flush: uploads, cluster build, TLAS build. Strict order = the copies run
        // before the build that reads them through the records' raw addresses, and the
        // cluster build runs before the TLAS build that bakes its address.
        upload_host_data();
        stream << single_flush_mesh.build()
               << single_flush_accel.build()
               << synchronize();

        LUISA_INFO("[test_rtx_cluster] single-flush cluster BLAS address=0x{:x}",
                   single_flush_mesh.cluster_address());
        check(single_flush_mesh.cluster_address() != 0u,
              luisa::format("single-flush mesh.cluster_address() must be non-zero, got 0x{:x}",
                            single_flush_mesh.cluster_address()));
        check(single_flush_mesh.cluster_count() == reported_cluster_count,
              luisa::format("the second cluster mesh reports cluster_count()={}, the first {}",
                            single_flush_mesh.cluster_count(), reported_cluster_count));

        auto single_flush_traced = trace_rays(single_flush_accel);
        check_against_oracle("single-flush", single_flush_traced);
        check_against_reference("single-flush", single_flush_traced, triangle_traced);

        // Leave the device as we found it for whatever runs next in this process.
        reorder_ext->set_command_reorder_enabled(true);
    }

    LUISA_INFO("[test_rtx_cluster] {} rays traced through the cluster and reference "
               "acceleration structures in both submission patterns, {} failed check(s).",
               ray_count, failures);
    return {failures, false};
}

}// namespace test_rtx_cluster_detail

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        // Pre-existing convention of the other tests in this directory (see
        // test_rtx.cpp): a missing backend argument prints the usage and reports
        // success, because it is a misuse, not a failed run.
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto outcome = test_rtx_cluster_detail::run_rtx_cluster_test(dc->device);
    if (outcome.skipped) {
        LUISA_WARNING("SKIPPED: [test_rtx_cluster] the cluster acceleration-structure path "
                      "is unavailable on this device; exiting with {} so CI records a skip "
                      "rather than a pass.",
                      skip_exit_code);
        return skip_exit_code;
    }
    // Boost.UT's own exit path is the primary one: its runner destructor reports the
    // summary at process exit and calls std::exit(-1) (status 255, 127 through a POSIX
    // shell) when any expect() failed, so a real failure is non-zero even before this
    // return value is reached. This mapping is the belt-and-braces path for the case
    // where that destructor does not fire; it is never 0 on failure either way.
    return outcome.failures == 0u ? 0 : 1;
}
