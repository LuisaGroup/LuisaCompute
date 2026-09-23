// The bottom level of the two-level software LBVH: an LBVH over the triangles
// of one mesh (bottom level acceleration structure, BLAS).
//
// The interface deliberately mirrors the acceleration-structure interface of
// the hardware backends (src/backends/vk/blas.cpp and
// src/backends/dx/Resource/BottomAccel.cpp) so this LBVH can later be used as
// a fallback for devices without hardware ray tracing:
//
//   * `create()` plays the role of `DeviceInterface::create_mesh()`: it takes an
//     `AccelOption` and the geometry and produces the `Blas` handle; it performs
//     no GPU work.
//   * `estimate()` / `LbvhBuildSizes` is the host-side build-size query, the
//     software counterpart of `vkGetAccelerationStructureBuildSizesKHR()` /
//     `GetRaytracingAccelerationStructurePrebuildInfo()`: it reports the maximum
//     sizes of the acceleration-structure (node) and scratch (primitive/key)
//     buffers before anything is allocated.
//   * `pre_build()` plays the role of the backend pre-build
//     (`Blas::_pre_build()` / `BottomAccel::PreProcessStates()`): it reserves
//     the tree's slice of the shared storage from the estimate and returns the
//     scratch bytes the build needs.
//   * `build()` plays the role of the backend build (`vkCmdBuildAccelerationStructuresKHR()`
//     / `BuildRaytracingAccelerationStructure()`): it records the LBVH build
//     kernels into the stream.
//
// The stage order the backends use is therefore preserved exactly:
// estimate -> create -> pre_build -> build.  The generated acceleration
// structure lives in the shared `LbvhStorage` (the software analogue of the
// backend's acceleration-structure buffer) and its build scratch is the shared
// primitive/key buffers.
//
// Geometry layout: the build and the traversal read the very same vertex and
// triangle buffers a Luisa RTX `Mesh` would be created from, and they assume the
// same element layout the backends configure the hardware geometry with:
//
//   * a vertex is a `float3`, i.e. a 16-byte element whose x/y/z occupy the
//     first 12 bytes (the backends use a 3-float vertex format with that stride);
//   * a triangle is three `uint`s (12 bytes, no padding), which is exactly a
//     32-bit index buffer holding three indices per primitive.
//
// The vertex indices of a triangle are *global* ids into the shared vertex
// buffer; the BLAS's own base is the `triangle_offset` this mesh starts at.

#pragma once

#include "lbvh_common.h"
#include "lbvh_storage.h"

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// A bottom-level acceleration structure: an LBVH over the triangles of one mesh.
class Blas {

public:
    Blas() noexcept = default;

    // ---- host-side build-size query (backend build-size query analogue) ----
    // Maximum acceleration-structure (node) and scratch (primitive/key) buffer
    // sizes of an LBVH over `triangle_count` triangles.
    [[nodiscard]] static LbvhBuildSizes estimate(uint triangle_count) noexcept {
        return lbvh_build_sizes(triangle_count);
    }

    // ---- resource state (backend-like accessors) ----
    [[nodiscard]] const AccelOption &option() const noexcept { return _option; }
    [[nodiscard]] const LbvhBuildSizes &sizes() const noexcept { return _sizes; }
    [[nodiscard]] bool is_created() const noexcept { return _sizes.primitive_count != 0u; }
    [[nodiscard]] bool is_pre_built() const noexcept { return _pre_built; }
    [[nodiscard]] uint node_offset() const noexcept { return _node_offset; }
    [[nodiscard]] uint prim_offset() const noexcept { return _prim_offset; }
    [[nodiscard]] uint triangle_offset() const noexcept { return _triangle_offset; }
    [[nodiscard]] uint triangle_count() const noexcept { return _triangle_count; }
    [[nodiscard]] uint node_count() const noexcept { return static_cast<uint>(_sizes.node_count); }
    [[nodiscard]] float3 object_space_min() const noexcept { return _lo; }
    [[nodiscard]] float3 object_space_max() const noexcept { return _hi; }
    // GPU-side view of this BLAS.
    [[nodiscard]] LbvhBlas record() const noexcept {
        return LbvhBlas{_node_offset, _triangle_offset, _triangle_count};
    }

private:
    friend class BlasBuilder;
    AccelOption _option;
    LbvhBuildSizes _sizes;
    uint _node_offset{};
    uint _prim_offset{};
    uint _triangle_offset{};
    uint _triangle_count{};
    float3 _lo{};
    float3 _hi{};
    bool _pre_built{false};
};

// Builds one `Blas`, following the backend phases: `create()` records the
// geometry and the option, `pre_build()` reserves the tree in the shared
// storage, `build()` records the triangle AABB kernel and the shared tree build.
class BlasBuilder {

public:
    explicit BlasBuilder(Device &device) noexcept;

    // Create the BLAS resource: records the LBVH over the triangles
    // [triangle_offset, triangle_offset + triangle_count) of `triangles`, whose
    // vertex indices are global indices into `vertices`.  `object_min` /
    // `object_max` are the object-space bounds of the mesh (they determine the
    // unit cube the Morton codes are computed in).  No GPU work is done here.
    [[nodiscard]] Blas create(const AccelOption &option, uint triangle_offset,
                              uint triangle_count, float3 object_min,
                              float3 object_max) noexcept;

    // Host-side pre-build: reserve the tree's acceleration-structure (node) and
    // scratch (primitive) ranges in the shared storage.  Returns the number of
    // scratch bytes the build of `blas` needs, mirroring the scratch size the
    // backends return from their pre-build.
    [[nodiscard]] size_t pre_build(LbvhStorage &storage, Blas &blas) noexcept;

    // Record the build: triangle AABBs -> Morton codes -> 4 x 8-bit LSD radix
    // sort -> Karras radix tree.  `request` mirrors the RTX build request; the
    // software LBVH has no in-place update path and always rebuilds the tree.
    // A non-null `timings` asks for the per-stage times of this build to be
    // accumulated into it (see `LbvhBuildTimings`).
    void build(Stream &stream, LbvhStorage &storage, const Blas &blas,
               const Buffer<float3> &vertices, const Buffer<Triangle> &triangles,
               AccelBuildRequest request,
               LbvhBuildTimings *timings = nullptr) noexcept;

private:
    Shader1D<Buffer<Triangle>, Buffer<float3>, Buffer<LbvhPrim>, uint, uint, uint> _prim_kernel;
};

// Software traversal of one BLAS by a single ray given in the *object space* of
// the instance that owns it: `origin` / `direction` are transformed by the
// caller, but the direction is deliberately not renormalized, so the ray
// parameter keeps its world-space meaning and `t_min` / the current best
// distance of `best` can be compared directly.  `nodes`, `vertices` and
// `triangles` are the shared buffers of the scene.
//
// `best` is the running closest hit; it must already be initialized (miss
// marker, `bary`, and the current upper distance bound in `t`), and is updated
// in place with the closest hit inside this BLAS, `instance` being copied into
// `LbvhHit::inst`.
void blas_traversal(Var<LbvhHit> &best, UInt instance, const Var<LbvhBlas> &blas,
                    Float3 origin, Float3 direction, Float t_min,
                    const BufferVar<LbvhNode> &nodes,
                    const BufferVar<float3> &vertices,
                    const BufferVar<Triangle> &triangles) noexcept;

}// namespace luisa::example::lbvh
