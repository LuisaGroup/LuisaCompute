// The bottom level of the fallback: an LBVH over the triangles of one mesh,
// laid out as one *region* of the shared acceleration buffer
// (fallback_rtx_layout.h).
//
// This is the port of examples/compute/lbvh/blas.{h,cpp}, with one addition the
// fallback needs and the example did not: the region carries the mesh's
// *geometry*.  A traversal of this module is given a single descriptor - the
// region - and reads the vertices and the triangle indices through it, so the
// build copies one uint4 per vertex and one per triangle into the region.
//
// The geometry arrives the way the backends hand it to their own mesh build:
//
//   * a vertex is a `float3`, i.e. `vertex_stride` bytes whose first 12 are
//     x/y/z (`vertex_stride` is 12 for a tightly packed mesh and 16 for the
//     DSL's `float3`, both accepted);
//   * a triangle is three `uint` (12 bytes, no padding), which is exactly a
//     32-bit index buffer holding three indices per primitive.
//
// Anything else fails closed here instead of being read as if it had the shape
// above.

#pragma once

#include "fallback_rtx.h"
#include "fallback_rtx_layout.h"
#include "fallback_rtx_storage.h"

namespace lc::fallback_rtx {

// The host-side state of one fallback BLAS.
struct FallbackBlas {
    AccelOption option;
    // Where the tree lives; `region.base == 0u` until the first build.
    FallbackRtxRegion region;
    // This BLAS' record inside the shared blas directory, as its *entry index*,
    // which is what an instance record's reserved lane carries.  Valid once
    // `built`.
    uint directory_entry{};
    uint triangle_count{};
    uint vertex_count{};
    bool built{false};
};

class FallbackBlasBuilder {

public:
    explicit FallbackBlasBuilder(FallbackRtxStorage &storage) noexcept;

    // Record the build of `blas` for the geometry of a `MeshBuildCommand`:
    // copy the geometry into a fresh region, one AABB per triangle, then the
    // shared Morton -> sort -> radix-tree stages.
    void build(CommandList &commands, FallbackBlas &blas,
               const FallbackRtxDevice::MeshGeometry &geometry) noexcept;

private:
    FallbackRtxStorage *_storage{nullptr};
    // One thread per triangle: read the three vertices through the raw-word
    // views, copy them and the triangle into the region, emit the primitive AABB
    // and reduce the scene volume the Morton codes are normalized with.
    Shader1D<Buffer<uint>, Buffer<uint>, Buffer<uint4>, Buffer<FallbackRtxPrim>,
             Buffer<uint>, uint, uint, uint, uint, uint, uint>
        _prim_kernel;
};

}// namespace lc::fallback_rtx
