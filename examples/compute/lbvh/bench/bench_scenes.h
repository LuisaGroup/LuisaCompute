// The adversarial scene catalogue of the software-LBVH benchmark.
//
// Every generator here is host-side, deterministic and data-parallel-friendly:
// the same (scene, triangles, instances, seed) gives the same geometry on every
// backend, and a scene split into K meshes (`--instances K`) contains exactly
// the same triangles as the one-mesh scene, so the multi-BLAS runs stay
// comparable with the single-BLAS ones.
//
// A BLAS has *one* tree and *one* radix sort, so the default shape of every
// scene is one BLAS with N triangles plus a one-instance TLAS; the only
// exception is `instance-chain`, whose whole point is the multi-BLAS /
// multi-instance path.
//
// Each scene also carries the ray distribution that makes its worst case
// *reachable*: a degenerate tree is only expensive if the rays actually walk it.

#pragma once

#include "../lbvh_common.h"
#include "../scene.h"

#include <cstddef>
#include <cstdint>

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// One mesh of a benchmark scene: a triangle range of the shared triangle soup
// plus the object-space bounds of its vertices, i.e. the volume the BLAS maps
// onto the unit cube for its Morton codes.
struct MeshRange {
    uint triangle_offset;
    uint triangle_count;
    float3 lo;
    float3 hi;
};

// One TLAS instance: which mesh of the scene, and its object->world transform.
struct InstanceSpec {
    uint mesh;
    float4x4 to_world;
};

// How the rays of a scene are distributed.  The mode is part of the scene
// because a worst case that the rays cannot reach is not a measurement.
enum struct BenchRayMode : uint {
    CAMERA = 0u,// outward frustum aimed at the scene AABB (the demo's distribution)
    BLOB = 1u,  // origins on a shell around the degenerate region, through it
    AXIS = 2u,  // rays nearly parallel to the chain axis (maximum overlap)
};

// Host-side ray parameters of a scene (the camera basis, the degenerate region
// and the chain axis), all in world space.
struct BenchRaySetup {
    BenchRayMode mode{BenchRayMode::CAMERA};
    float3 eye{};    // camera position (CAMERA) / chain start (AXIS)
    float3 forward{};// camera direction (CAMERA) / chain axis (AXIS)
    float3 right{};
    float3 up{};
    float half_w{1.0f};// frustum half-size (CAMERA) / disc radius (AXIS)
    float half_h{1.0f};
    float3 blob_center{};// degenerate region (BLOB)
    float blob_radius{1.0f};
    float3 axis{};// chain axis (AXIS / BLOB)
};

struct BenchScene {
    const char *name;         // CLI selector
    const char *stress;       // "build" / "traversal" / "both"
    const char *worst_case;   // one line: which LBVH worst case this triggers
    const char *dispatch_note;// why the default sizes keep one dispatch short
    size_t default_triangles;
    size_t default_instances;
    size_t default_rays;
    luisa::vector<float3> vertices;
    luisa::vector<Triangle> triangles;
    luisa::vector<MeshRange> meshes;
    luisa::vector<InstanceSpec> instances;
    BenchRaySetup rays;
    float3 scene_lo{1.0e30f};
    float3 scene_hi{-1.0e30f};

    [[nodiscard]] size_t primitive_count() const noexcept {
        return triangles.size() + instances.size();
    }
};

// The catalogue without geometry: name, stress class, worst case and the
// default sizes (what `--list` prints).
struct BenchSceneInfo {
    const char *name;
    const char *stress;
    const char *worst_case;
    // Measured cost of one dispatch at the default sizes on the reference
    // machine (RTX 4060); it is what the defaults are tuned against, because a
    // single multi-second submission makes the driver reset the device.
    const char *dispatch_note;
    size_t default_triangles;
    size_t default_instances;
    size_t default_rays;
};

[[nodiscard]] luisa::span<const BenchSceneInfo> bench_scene_catalogue() noexcept;

// Materializes one catalogue scene.  `triangles` / `instances` / `rays` are the
// effective counts (0 means "the scene's default").  Returns false and sets
// `error` when the request cannot be honoured (unknown name, empty geometry).
[[nodiscard]] bool make_bench_scene(const char *name, size_t triangles, size_t instances,
                                    size_t rays, uint64_t seed, BenchScene &scene,
                                    luisa::string &error) noexcept;

// Materializes a scene from an OBJ file: one BLAS over the mesh, `instances`
// copies of it laid out in a small deterministic grid (the "real asset" path).
// `max_triangles` (0 = all) stops the loader early, which is what a pre-flight
// uses to measure a bounded *probe* of an asset that may be arbitrarily large.
[[nodiscard]] bool make_obj_scene(const char *path, size_t instances, size_t rays,
                                  uint64_t seed, BenchScene &scene, luisa::string &error,
                                  size_t max_triangles = 0u) noexcept;

// Tiny hand-written OBJ loader: `v` and `f` only, ignoring materials, normals,
// groups and smoothing; faces with more than three vertices are fan
// triangulated, negative (relative) indices are resolved, and degenerate
// triangles are dropped.  Nothing is bundled with the repository: the caller
// passes any OBJ it likes (Sponza, the Stanford dragon, ...).
[[nodiscard]] bool load_obj(const char *path, HostMesh &mesh, luisa::string &error,
                            size_t max_triangles = 0u) noexcept;

// The ray generator, one thread per ray.  It is a kernel (not host code) so the
// three backends see bit-identical rays; the hash is a PCG one, as in the demo,
// and every mode derives its jitter from it.
[[nodiscard]] inline auto make_ray_kernel() noexcept {
    return Kernel1D{[](BufferVar<LbvhRay> rays, UInt count, UInt grid_w, UInt seed, UInt mode,
                       Float3 eye, Float3 forward, Float3 right, Float3 up,
                       Float half_w, Float half_h, Float3 blob_center, Float blob_radius,
                       Float3 axis) noexcept {
        set_block_size(256u);
        UInt index = dispatch_id().x;
        $if (index < count) {
            // PCG hash: a fixed, backend-independent stream of 32-bit words.
            auto state = def(seed * 747796405u + 2891336453u + index * 2891336453u);
            auto next_u32 = [&state]() noexcept {
                state = state * 747796405u + 2891336453u;
                auto word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                return (word >> 22u) ^ word;
            };
            auto next_float = [&next_u32]() noexcept {
                return cast<float>(next_u32() >> 8u) * (1.0f / 16777216.0f);
            };
            auto next_symmetric = [&next_float]() noexcept { return next_float() * 2.0f - 1.0f; };
            // uniform direction on the unit sphere
            auto next_direction = [&]() noexcept {
                auto z = next_symmetric();
                auto phi = next_float() * 6.283185307179586f;
                auto r = sqrt(max(0.0f, 1.0f - z * z));
                return make_float3(r * cos(phi), r * sin(phi), z);
            };
            Var<LbvhRay> ray;
            ray.t_min = 1.0e-3f;
            ray.t_max = 1.0e30f;
            $if (mode == static_cast<uint>(BenchRayMode::CAMERA)) {
                // outward frustum with jittered sub-pixel positions, like the demo
                auto x = index % grid_w;
                auto y = index / grid_w;
                auto uv = (make_float2(cast<float>(x), cast<float>(y)) +
                           make_float2(next_float(), next_float())) /
                          make_float2(cast<float>(grid_w), cast<float>(grid_w));
                ray.origin = eye;
                ray.direction = normalize(forward +
                                          (uv.x * 2.0f - 1.0f) * half_w * right +
                                          (1.0f - uv.y * 2.0f) * half_h * up);
            }
            $elif (mode == static_cast<uint>(BenchRayMode::BLOB)) {
                // The origin sits on a shell around the degenerate region (its
                // radius travels in half_w) and the direction points at a random
                // point *inside* it: the ray has to cross the whole blob, so
                // almost nothing can be culled.
                auto origin = blob_center + next_direction() * half_w;
                auto target = blob_center + next_direction() * (blob_radius * next_float());
                ray.origin = origin;
                ray.direction = normalize(target - origin);
            }
            $else {
                // Rays parallel to the chain axis, spread over a disc and only
                // slightly jittered in direction: they overlap the whole chain.
                ray.origin = eye + next_symmetric() * half_w * right +
                             next_symmetric() * half_h * up;
                ray.direction = normalize(axis + 0.02f * (next_symmetric() * right +
                                                          next_symmetric() * up));
            };
            rays.write(index, ray);
        };
    }};
}

}// namespace luisa::example::lbvh
