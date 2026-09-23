// Two-level software LBVH (BLAS + TLAS) for LuisaCompute - the demo.
//
// The acceleration structure is built *by hand* inside Luisa kernels and lives
// in plain `Buffer`s: no Luisa RTX mesh/accel API is involved.  This file only
// generates the scene, drives the build and the traversal, and cross-checks the
// result; the LBVH itself lives in
//
//   lbvh/lbvh_common.h   - GPU-side layout (nodes, primitives, rays, hits) and
//                          the ray/AABB + ray/triangle tests
//   lbvh/lbvh_storage.h  - storage shared by all trees and the tree-agnostic
//   lbvh/lbvh_storage.cpp  build stages: Morton codes, LSD radix sort, Karras
//                          radix tree, plus the structural self-check
//   lbvh/blas.h          - a BLAS (LBVH over the triangles of one mesh): the
//   lbvh/blas.cpp          triangle AABB kernel, the build and the traversal
//   lbvh/tlas.h          - a TLAS (LBVH over instances): the instance AABB
//   lbvh/tlas.cpp          kernel, the build, and the two-level traversal
//   lbvh/software_lbvh.h - the facade owning storage, builders and the
//   lbvh/software_lbvh.cpp  compiled traversal kernel
//   lbvh/scene.h         - the host-side meshes of the demo scene
//
// The LBVH exposes the same acceleration-structure interface as the hardware
// backends (src/backends/vk/{blas,tlas}.cpp,
// src/backends/dx/Resource/{Bottom,Top}Accel.cpp): the caller first queries the
// maximum required buffer sizes on the host, then creates the BLAS/TLAS
// resources, then pre-builds them (which reserves storage and returns the build
// scratch) and finally records the builds.  Keeping that shape is what will let
// this LBVH be plugged in as a fallback implementation for devices without
// hardware ray tracing.
//
// The build follows the algorithm of
//
//   Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees,
//   and k-d Trees", High Performance Graphics 2012,
//
// as implemented by Mirco Werner's VkLBVH reference
// (https://github.com/MircoWerner/VkLBVH).
//
// Traversal is a plain software BVH walk (slab test + two-sided
// Moller-Trumbore).  As a cross-check the very same scene is handed to the
// Luisa RTX API (`Mesh` + `Accel`), the same rays are traced through both
// implementations, and the sampled hits are compared.
//
// Both implementations read the *same* GPU buffers, so they also have to agree
// on their element layout: a vertex is a `float3` (16-byte stride, x/y/z in the
// first 12 bytes) and a triangle is three `uint`s (12 bytes, no padding, which
// is what the RTX backends use as a `uint32` index buffer view).  The backends
// configure the hardware geometry from exactly those strides (see the
// self-check before the rays are traced).
//
// Usage: example_software_lbvh <backend>

#include "lbvh/scene.h"
#include "lbvh/software_lbvh.h"

#include <luisa/luisa-compute.h>

#include <algorithm>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::example::lbvh;

int main(int argc, char *argv[]) {
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    if (argc < 2 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        LUISA_INFO("Usage: {} <backend>.", executable);
        return 1;
    }

    Context context{executable};
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();

    // -----------------------------------------------------------------------
    // Scene: three meshes, nine instances
    // -----------------------------------------------------------------------
    HostMesh sphere = make_uv_sphere(0.5f, 24u, 12u);
    HostMesh box = make_box(make_float3(0.45f));
    HostMesh torus = make_torus(0.35f, 0.15f, 24u, 12u);

    luisa::vector<float3> host_vertices;
    luisa::vector<Triangle> host_triangles;
    struct MeshRange {
        uint triangle_offset;
        uint triangle_count;
        float3 lo;
        float3 hi;
    };
    luisa::vector<MeshRange> mesh_ranges;
    for (auto *mesh : {&sphere, &box, &torus}) {
        MeshRange range{static_cast<uint>(host_triangles.size()),
                        static_cast<uint>(mesh->triangles.size()),
                        mesh->lo, mesh->hi};
        auto vertex_offset = static_cast<uint>(host_vertices.size());
        for (auto v : mesh->vertices) { host_vertices.emplace_back(v); }
        for (auto t : mesh->triangles) {
            host_triangles.emplace_back(Triangle{t.i0 + vertex_offset,
                                                 t.i1 + vertex_offset,
                                                 t.i2 + vertex_offset});
        }
        mesh_ranges.emplace_back(range);
    }

    Buffer<float3> vertices = device.create_buffer<float3>(host_vertices.size());
    Buffer<Triangle> triangles = device.create_buffer<Triangle>(host_triangles.size());
    stream << vertices.copy_from(luisa::span{host_vertices})
           << triangles.copy_from(luisa::span{host_triangles})
           << synchronize();
    LUISA_INFO("scene: {} vertices, {} triangles, {} meshes",
               host_vertices.size(), host_triangles.size(), mesh_ranges.size());

    struct InstanceSpec {
        uint mesh;
        float4x4 to_world;
    };
    luisa::vector<InstanceSpec> instance_specs;
    auto add_instance = [&](uint mesh, float3 position, float3 axis, float angle, float scale = 1.0f) noexcept {
        instance_specs.emplace_back(InstanceSpec{
            mesh,
            translation(position) * rotation(axis, radians(angle)) * scaling(scale)});
    };
    add_instance(0u, make_float3(-1.30f, 0.60f, 0.00f), make_float3(0.0f, 1.0f, 0.0f), 0.0f);
    add_instance(1u, make_float3(0.00f, 0.60f, 0.00f), make_float3(0.4f, 1.0f, 0.2f), 35.0f);
    add_instance(2u, make_float3(1.30f, 0.60f, 0.00f), make_float3(1.0f, 0.3f, 0.0f), 25.0f);
    add_instance(0u, make_float3(-1.30f, -0.70f, 0.30f), make_float3(0.2f, 1.0f, 0.5f), 60.0f, 0.8f);
    add_instance(1u, make_float3(0.00f, -0.70f, -0.25f), make_float3(1.0f, 0.5f, 0.7f), 15.0f, 0.9f);
    add_instance(2u, make_float3(1.30f, -0.70f, 0.20f), make_float3(0.7f, 1.0f, 0.1f), 70.0f, 1.1f);
    add_instance(0u, make_float3(-0.65f, 1.55f, -0.20f), make_float3(1.0f, 0.2f, 1.0f), 40.0f, 0.7f);
    add_instance(2u, make_float3(0.65f, 1.55f, 0.25f), make_float3(0.3f, 0.9f, 1.0f), 55.0f, 0.75f);
    add_instance(1u, make_float3(0.00f, 0.00f, -1.30f), make_float3(1.0f, 1.0f, 0.0f), 45.0f, 1.2f);

    // -----------------------------------------------------------------------
    // Build the software BLASes and the software TLAS (everything in kernels).
    // The build follows the same order as the hardware backends: query the
    // maximum required acceleration-structure storage on the host, create the
    // resources, pre-build (reserve) them, then record the builds.
    // -----------------------------------------------------------------------
    Clock clock;
    clock.tic();
    // Host-side size query before anything is allocated on the device, the
    // software counterpart of the driver's prebuild-info query.
    auto sizes = SoftwareLbvh::estimate(host_triangles.size(), instance_specs.size(),
                                        mesh_ranges.size());
    LUISA_INFO("software LBVH storage estimate: {} primitives, {} nodes, {:.1f} MiB",
               sizes.primitive_capacity, sizes.node_capacity,
               static_cast<double>(sizes.total_bytes()) / (1024.0 * 1024.0));
    SoftwareLbvh lbvh{device, sizes};
    luisa::vector<Blas> blases;
    blases.reserve(mesh_ranges.size());
    size_t blas_nodes = 0u;
    size_t blas_scratch = 0u;
    for (auto &&range : mesh_ranges) {
        // create (create_mesh) -> pre_build (size query + reservation) -> build
        auto blas = lbvh.create_blas(AccelOption{}, range.triangle_offset,
                                     range.triangle_count, range.lo, range.hi);
        blas_scratch += lbvh.pre_build_blas(blas);
        lbvh.build_blas(stream, blas, vertices, triangles);
        blas_nodes += blas.node_count();
        blases.emplace_back(blas);
    }
    luisa::vector<InstanceDesc> instance_descs;
    instance_descs.reserve(instance_specs.size());
    for (auto &&spec : instance_specs) {
        instance_descs.emplace_back(InstanceDesc{spec.to_world, spec.mesh});
    }
    auto tlas = lbvh.create_accel(AccelOption{},
                                  static_cast<uint>(instance_descs.size()));
    blas_scratch += lbvh.pre_build_accel(stream, tlas,
                                         luisa::span{blases}, luisa::span{instance_descs});
    lbvh.build_accel(stream, tlas);
    stream << synchronize();
    auto build_ms = clock.toc();
    LUISA_INFO("software LBVH built in {:.3f} ms: {} BLAS ({} nodes), TLAS with {} instances ({} nodes), scratch {:.1f} KiB",
               build_ms, lbvh.blas_count(), blas_nodes, tlas.instance_count(), tlas.node_count(),
               static_cast<double>(blas_scratch) / 1024.0);

    // structural self-check of every tree that was just built
    {
        size_t problems = 0u;
        for (auto i = 0u; i < blases.size(); i++) {
            problems += lbvh.validate_tree(stream, blases[i].node_offset(), blases[i].triangle_count());
        }
        problems += lbvh.validate_tree(stream, tlas.node_offset(), tlas.instance_count());
        LUISA_INFO("structural self-check: {} BLAS + 1 TLAS, {} problem(s)", blases.size(), problems);
        LUISA_ASSERT(problems == 0u, "the software LBVH is malformed.");
    }

    // -----------------------------------------------------------------------
    // The same scene through the Luisa RTX API, as a reference
    // -----------------------------------------------------------------------
    luisa::vector<Mesh> meshes;
    meshes.reserve(mesh_ranges.size());
    for (auto &&range : mesh_ranges) {
        meshes.emplace_back(device.create_mesh(
            vertices, triangles.view(range.triangle_offset, range.triangle_count)));
    }
    Accel accel = device.create_accel();
    for (auto &&spec : instance_specs) {
        accel.emplace_back(meshes[spec.mesh], spec.to_world);
    }
    for (auto &&mesh : meshes) { stream << mesh.build(); }
    stream << accel.build() << synchronize();

    // -----------------------------------------------------------------------
    // Shared vertex / triangle layout
    // -----------------------------------------------------------------------
    // The software BLASes and the RTX meshes consume the very same buffers, so
    // they must agree on how the elements are laid out.  A vertex is a `float3`
    // (16-byte stride, x/y/z in the first 12 bytes) and a triangle is three
    // `uint`s (12 bytes, no padding); the RTX backends configure the hardware
    // geometry from exactly these strides (a 3-float vertex format with the
    // element stride reported by the buffer, and `uint32` indices with a
    // 12-byte stride), so both paths resolve the same primitives.
    static_assert(sizeof(Triangle) == 3u * sizeof(uint));
    static_assert(sizeof(float3) == 16u && alignof(float3) == 16u);
    size_t layout_problems = 0u;
    for (auto i = 0u; i < meshes.size(); i++) {
        auto &&mesh = meshes[i];
        auto &&range = mesh_ranges[i];
        auto &&blas = blases[i];
        // vertices: the mesh covers the whole shared vertex buffer, and the
        // stride it reports is the device-side stride of `float3`, which is the
        // one the software kernels use to read (and the host one to upload).
        layout_problems += mesh.vertex_stride() != vertices.stride() ||
                           mesh.vertex_stride() != sizeof(float3) ||
                           mesh.vertex_buffer<float3>().size() != host_vertices.size();
        // triangles: the mesh covers exactly the triangles of the BLAS, at the
        // very same byte offset of the shared triangle buffer.
        layout_problems += mesh.triangle_count() != blas.triangle_count() ||
                           mesh.triangle_count() != range.triangle_count ||
                           mesh.triangle_buffer().offset() != blas.triangle_offset() ||
                           mesh.triangle_buffer().stride() != sizeof(Triangle);
    }
    // Read the shared buffers back through the DSL: were the device-side layout
    // of `float3` / `Triangle` (or its stride) any different from the host-side
    // one, these values would be scrambled.
    auto vertex_count = static_cast<uint>(host_vertices.size());
    auto triangle_count = static_cast<uint>(host_triangles.size());
    Buffer<float> vertex_readback = device.create_buffer<float>(vertex_count * 3u);
    Buffer<uint> triangle_readback = device.create_buffer<uint>(triangle_count * 3u);
    Kernel1D read_layout = [](BufferVar<float3> vertices, BufferVar<Triangle> triangles,
                              BufferVar<float> vertex_out, BufferVar<uint> triangle_out,
                              UInt vertex_count, UInt triangle_count) noexcept {
        set_block_size(256u);
        UInt index = dispatch_id().x;
        $if (index < vertex_count) {
            auto v = vertices.read(index);
            vertex_out.write(index * 3u + 0u, v.x);
            vertex_out.write(index * 3u + 1u, v.y);
            vertex_out.write(index * 3u + 2u, v.z);
        };
        $if (index < triangle_count) {
            auto t = triangles.read(index);
            triangle_out.write(index * 3u + 0u, t.i0);
            triangle_out.write(index * 3u + 1u, t.i1);
            triangle_out.write(index * 3u + 2u, t.i2);
        };
    };
    luisa::vector<float> host_vertex_readback(vertex_count * 3u);
    luisa::vector<uint> host_triangle_readback(triangle_count * 3u);
    auto read_layout_shader = device.compile(read_layout);
    stream << read_layout_shader(vertices, triangles, vertex_readback, triangle_readback,
                                 vertex_count, triangle_count)
                  .dispatch(std::max(vertex_count, triangle_count))
           << vertex_readback.copy_to(luisa::span{host_vertex_readback})
           << triangle_readback.copy_to(luisa::span{host_triangle_readback})
           << synchronize();
    for (auto i = 0u; i < vertex_count; i++) {
        layout_problems += host_vertex_readback[i * 3u + 0u] != host_vertices[i].x ||
                           host_vertex_readback[i * 3u + 1u] != host_vertices[i].y ||
                           host_vertex_readback[i * 3u + 2u] != host_vertices[i].z;
    }
    for (auto i = 0u; i < triangle_count; i++) {
        layout_problems += host_triangle_readback[i * 3u + 0u] != host_triangles[i].i0 ||
                           host_triangle_readback[i * 3u + 1u] != host_triangles[i].i1 ||
                           host_triangle_readback[i * 3u + 2u] != host_triangles[i].i2;
    }
    LUISA_INFO("shared layout: {} vertices x {}-byte stride, {} triangles x {}-byte stride, {} problem(s)",
               vertex_count, vertices.stride(), triangle_count, triangles.stride(),
               layout_problems);
    LUISA_ASSERT(layout_problems == 0u,
                 "the software LBVH and the RTX meshes disagree on the vertex/triangle layout.");

    // -----------------------------------------------------------------------
    // Rays
    // -----------------------------------------------------------------------
    constexpr uint width = 512u;
    constexpr uint height = 512u;
    constexpr uint ray_count = width * height;
    auto eye = make_float3(0.0f, 0.0f, 5.2f);
    auto target = make_float3(0.0f, 0.15f, 0.0f);
    auto forward = normalize(target - eye);
    auto right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
    auto up = cross(right, forward);
    auto half_height = tan(radians(22.5f));
    auto half_width = half_height * static_cast<float>(width) / static_cast<float>(height);

    Kernel1D generate_rays = [](BufferVar<LbvhRay> rays, UInt w, UInt h,
                                Float3 origin, Float3 forward, Float3 right, Float3 up,
                                Float half_w, Float half_h) noexcept {
        set_block_size(256u);
        UInt index = dispatch_id().x;
        $if (index < w * h) {
            // PCG-style hash, so both traversals see bit-identical rays
            auto state = index * 747796405u + 2891336453u;
            state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
            state = (state >> 22u) ^ state;
            auto jitter_x = cast<float>(state & 0xFFFFu) * (1.0f / 65536.0f);
            auto jitter_y = cast<float>((state >> 16u) & 0xFFFFu) * (1.0f / 65536.0f);
            auto uv = (make_float2(cast<float>(index % w), cast<float>(index / w)) +
                       make_float2(jitter_x, jitter_y)) /
                      make_float2(cast<float>(w), cast<float>(h));
            auto direction = normalize(forward + (uv.x * 2.0f - 1.0f) * half_w * right +
                                       (1.0f - uv.y * 2.0f) * half_h * up);
            Var<LbvhRay> ray;
            ray.origin = origin;
            ray.direction = direction;
            ray.t_min = 1.0e-3f;
            ray.t_max = 1.0e30f;
            rays.write(index, ray);
        };
    };
    Kernel1D trace_rtx = [](AccelVar accel, BufferVar<LbvhRay> rays,
                            BufferVar<LbvhHit> hits, UInt count) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if (index < count) {
            auto r = rays.read(index);
            auto ray = make_ray(r.origin, r.direction, r.t_min, r.t_max);
            auto hit = accel.intersect(ray, {});
            Var<LbvhHit> result;
            result.inst = invalid_node;
            result.prim = invalid_node;
            result.bary = make_float2(0.0f);
            result.t = r.t_max;
            $if (!hit->miss()) {
                result.inst = hit.inst;
                result.prim = hit.prim;
                result.bary = hit.bary;
                result.t = hit->distance();
            };
            hits.write(index, result);
        };
    };
    auto generate_rays_shader = device.compile(generate_rays);
    auto trace_rtx_shader = device.compile(trace_rtx);

    Buffer<LbvhRay> rays = device.create_buffer<LbvhRay>(ray_count);
    Buffer<LbvhHit> software_hits = device.create_buffer<LbvhHit>(ray_count);
    Buffer<LbvhHit> rtx_hits = device.create_buffer<LbvhHit>(ray_count);
    stream << generate_rays_shader(rays, width, height, eye, forward, right, up,
                                   half_width, half_height)
                  .dispatch(ray_count)
           << synchronize();

    clock.tic();
    lbvh.trace_software(stream, vertices, triangles, rays, software_hits, tlas, ray_count);
    stream << synchronize();
    auto software_ms = clock.toc();
    clock.tic();
    stream << trace_rtx_shader(accel, rays, rtx_hits, ray_count).dispatch(ray_count)
           << synchronize();
    auto rtx_ms = clock.toc();
    LUISA_INFO("traced {} rays: software LBVH {:.3f} ms, Luisa RTX {:.3f} ms",
               ray_count, software_ms, rtx_ms);

    // -----------------------------------------------------------------------
    // Compare the sampled hits
    // -----------------------------------------------------------------------
    luisa::vector<LbvhHit> host_software(ray_count);
    luisa::vector<LbvhHit> host_rtx(ray_count);
    stream << software_hits.copy_to(luisa::span{host_software})
           << rtx_hits.copy_to(luisa::span{host_rtx})
           << synchronize();

    size_t software_hit_count = 0u;
    size_t rtx_hit_count = 0u;
    size_t miss_mismatch = 0u;
    size_t instance_mismatch = 0u;
    size_t prim_mismatch = 0u;
    size_t distance_mismatch = 0u;
    size_t bary_mismatch = 0u;
    float max_relative_distance_error = 0.0f;
    float max_barycentric_error = 0.0f;
    // The hardware traversal uses its own triangle intersection, so the
    // distance/barycentrics agree only up to rounding.  Observed worst case on
    // vk/cuda/dx is ~1e-5 relative distance and ~8e-4 barycentric error.
    constexpr float distance_tolerance = 1.0e-3f;
    constexpr float barycentric_tolerance = 5.0e-3f;
    for (auto i = 0u; i < ray_count; i++) {
        auto a = host_software[i];
        auto b = host_rtx[i];
        auto a_miss = a.inst == invalid_node;
        auto b_miss = b.inst == invalid_node;
        if (!a_miss) { software_hit_count++; }
        if (!b_miss) { rtx_hit_count++; }
        if (a_miss != b_miss) {
            miss_mismatch++;
            continue;
        }
        if (a_miss) { continue; }
        if (a.inst != b.inst) { instance_mismatch++; }
        if (a.prim != b.prim) { prim_mismatch++; }
        auto distance_error = std::abs(a.t - b.t) / std::max(1.0f, std::abs(b.t));
        auto barycentric_error = std::max(std::abs(a.bary.x - b.bary.x),
                                          std::abs(a.bary.y - b.bary.y));
        max_relative_distance_error = std::max(max_relative_distance_error, distance_error);
        max_barycentric_error = std::max(max_barycentric_error, barycentric_error);
        if (distance_error > distance_tolerance) { distance_mismatch++; }
        if (barycentric_error > barycentric_tolerance) { bary_mismatch++; }
    }
    LUISA_INFO("hits: software {}/{} ({:.2f}%), Luisa RTX {}/{} ({:.2f}%)",
               software_hit_count, ray_count,
               100.0 * static_cast<double>(software_hit_count) / static_cast<double>(ray_count),
               rtx_hit_count, ray_count,
               100.0 * static_cast<double>(rtx_hit_count) / static_cast<double>(ray_count));
    LUISA_INFO("mismatch: hit/miss {}, instance {}, primitive {}, distance {} (tol {}), barycentric {} (tol {})",
               miss_mismatch, instance_mismatch, prim_mismatch, distance_mismatch,
               distance_tolerance, bary_mismatch, barycentric_tolerance);
    LUISA_INFO("max relative distance error {:.3e}, max barycentric error {:.3e}",
               max_relative_distance_error, max_barycentric_error);

    if (miss_mismatch != 0u || instance_mismatch != 0u || prim_mismatch != 0u ||
        distance_mismatch != 0u || bary_mismatch != 0u) {
        LUISA_ERROR("software LBVH traversal does not match the Luisa RTX reference.");
        return 1;
    }
    LUISA_INFO("software LBVH traversal matches the Luisa RTX reference.");
    return 0;
}
