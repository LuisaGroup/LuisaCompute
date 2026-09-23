// The top level of the two-level software LBVH: an LBVH over instances (top
// level acceleration structure, TLAS), where every instance references a `Blas`
// and an object->world transform.
//
// As on the bottom level (blas.h), the interface mirrors the acceleration-
// structure interface of the hardware backends (src/backends/vk/tlas.cpp and
// src/backends/dx/Resource/TopAccel.cpp):
//
//   * `create()` plays the role of `DeviceInterface::create_accel()`: it takes
//     an `AccelOption` and the instance count and produces the `Tlas` handle.
//   * `estimate()` is the host-side build-size query (the software counterpart
//     of the driver's prebuild-info query).
//   * `pre_build()` uploads the BLAS table / instance records and reserves the
//     tree in the shared storage, returning the build scratch size.
//   * `build()` records the instance AABB kernel and the shared tree build.
//
// The stage order is therefore the backend one: estimate -> create -> pre_build
// -> build.

#pragma once

#include "blas.h"

#include <utility>

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// A top-level acceleration structure: an LBVH over instances.
class Tlas {

public:
    Tlas() noexcept = default;

    // ---- host-side build-size query (backend build-size query analogue) ----
    // Maximum acceleration-structure (node) and scratch (instance AABB +
    // Morton key) buffer sizes of an LBVH over `instance_count` instances.
    [[nodiscard]] static LbvhBuildSizes estimate(uint instance_count) noexcept {
        return lbvh_build_sizes(instance_count);
    }

    // ---- resource state (backend-like accessors) ----
    [[nodiscard]] const AccelOption &option() const noexcept { return _option; }
    [[nodiscard]] const LbvhBuildSizes &sizes() const noexcept { return _sizes; }
    [[nodiscard]] bool is_created() const noexcept { return _sizes.primitive_count != 0u; }
    [[nodiscard]] bool is_pre_built() const noexcept { return _pre_built; }
    [[nodiscard]] uint node_offset() const noexcept { return _node_offset; }
    [[nodiscard]] uint prim_offset() const noexcept { return _prim_offset; }
    [[nodiscard]] uint instance_count() const noexcept { return _instance_count; }
    [[nodiscard]] uint node_count() const noexcept { return static_cast<uint>(_sizes.node_count); }

private:
    friend class TlasBuilder;
    AccelOption _option;
    LbvhBuildSizes _sizes;
    uint _node_offset{};
    uint _prim_offset{};
    uint _instance_count{};
    // World-space volume the Morton codes of the build are normalized with; kept
    // here (like the backend keeps its prebuild info in the resource) so that
    // `build()` consumes what `pre_build()` computed.
    float3 _volume_lo{1.0e30f};
    float3 _volume_hi{-1.0e30f};
    bool _pre_built{false};
};

// One TLAS instance: the object->world transform of a BLAS reference.
struct InstanceDesc {
    float4x4 to_world;
    uint blas;
};

// Builds one `Tlas`, following the backend phases: `create()` records the
// option, `pre_build()` uploads the instance data and reserves the tree, and
// `build()` records the instance AABB kernel and the shared tree build.
class TlasBuilder {

public:
    explicit TlasBuilder(Device &device) noexcept;

    // Create the TLAS resource for `instance_count` instances; the instances
    // themselves are supplied to `pre_build()`.  No GPU work is done here.
    [[nodiscard]] Tlas create(const AccelOption &option, uint instance_count) noexcept;

    // Host-side pre-build: `blases` must contain the BLAS referenced by every
    // instance.  Uploads the BLAS table and the instance records (both are
    // shared with the traversal), computes the world-space volume of the scene
    // for the Morton codes, and reserves the tree in the shared storage.
    // Returns the number of scratch bytes the build needs.
    [[nodiscard]] size_t pre_build(Stream &stream, LbvhStorage &storage, Tlas &tlas,
                                   luisa::span<const Blas> blases,
                                   luisa::span<const InstanceDesc> instances) noexcept;

    // Record the build: world-space instance AABBs -> Morton codes -> 4 x 8-bit
    // LSD radix sort -> Karras radix tree.  `request` mirrors the RTX build
    // request; the software LBVH has no in-place update path and always rebuilds
    // the tree.
    void build(Stream &stream, LbvhStorage &storage, const Tlas &tlas,
               AccelBuildRequest request) noexcept;

private:
    // Instance records (world->object rows, object->world rows, BLAS index).
    void upload_instances(Stream &stream, LbvhStorage &storage,
                          luisa::span<const InstanceDesc> instances) noexcept;
    // World-space bounds of all instances, used for the Morton codes.
    [[nodiscard]] std::pair<float3, float3> instance_volume(
        luisa::span<const Blas> blases,
        luisa::span<const InstanceDesc> instances) const noexcept;

    Shader1D<Buffer<LbvhNode>, Buffer<LbvhBlas>, Buffer<LbvhInstance>, Buffer<LbvhPrim>, uint, uint>
        _prim_kernel;
};

// Two-level software traversal: walks the TLAS and descends into the BLAS
// referenced by every instance leaf it reaches.  Returns the closest hit of the
// whole scene, with `inst == invalid_node` on a miss.
[[nodiscard]] Var<LbvhHit> tlas_traversal(const Var<LbvhRay> &ray, UInt tlas_node_offset,
                                          const BufferVar<LbvhNode> &nodes,
                                          const BufferVar<LbvhBlas> &blas_table,
                                          const BufferVar<LbvhInstance> &instances,
                                          const BufferVar<float3> &vertices,
                                          const BufferVar<Triangle> &triangles) noexcept;

}// namespace luisa::example::lbvh
