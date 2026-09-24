// The software two-level LBVH of one scene.
//
// `SoftwareLbvh` owns the storage shared by all trees, the two builders (bottom
// level in blas.h / blas.cpp, top level in tlas.h / tlas.cpp) and the compiled
// two-level traversal kernel (software_lbvh.cpp), and is therefore the
// user-facing entry point of the example.  Its interface follows the
// acceleration-structure part of `DeviceInterface`, so that the LBVH can later
// be plugged in as a fallback for devices without hardware ray tracing:
//
//   estimate()          - host-side scene size query, before any allocation
//   create_blas()       - create a BLAS resource   (create_mesh analogue)
//   create_accel()      - create a TLAS resource   (create_accel analogue)
//   pre_build_blas()    - host pre-build: reserve the tree, return the scratch
//   pre_build_accel()   - host pre-build: upload instances, register the heap,
//                         reserve the tree
//   build_blas()        - record the bottom-level build kernels
//   build_accel()       - record the top-level build kernels
//   trace_software()    - software two-level traversal (through the TLAS heap)
//
// Usage (the order the hardware backends use: estimate -> create -> pre_build
// -> build):
//
//   auto sizes = SoftwareLbvh::estimate(max_triangles, max_instances, max_blas);
//   SoftwareLbvh lbvh{device, sizes};
//   Blas blas = lbvh.create_blas({}, offset, count, lo, hi);
//   lbvh.pre_build_blas(blas);
//   lbvh.build_blas(stream, blas, vertices, triangles);
//   Tlas tlas = lbvh.create_accel({}, instance_count);
//   lbvh.pre_build_accel(stream, tlas, luisa::span{blases}, luisa::span{instances});
//   lbvh.build_accel(stream, tlas);
//   lbvh.trace_software(stream, vertices, triangles, rays, hits, tlas, ray_count);

#pragma once

#include "tlas.h"

#include <cstddef>

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

class SoftwareLbvh {

public:
    // Host-side scene size query (see `LbvhStorage::estimate`): the maximum
    // sizes of every shared buffer a scene with these capacities needs,
    // computed before any device allocation.
    using Sizes = LbvhStorage::Sizes;
    [[nodiscard]] static Sizes estimate(size_t max_triangles, size_t max_instances,
                                        size_t max_blas) noexcept {
        return LbvhStorage::estimate(max_triangles, max_instances, max_blas);
    }
    // The compaction vocabulary lives on the storage; re-export it so a caller
    // names `SoftwareLbvh::CompactionPolicy` / `CompactResult` directly.
    using CompactionPolicy = LbvhStorage::CompactionPolicy;
    using CompactResult = LbvhStorage::CompactResult;

    SoftwareLbvh(Device &device, const Sizes &sizes) noexcept;
    SoftwareLbvh(Device &device, size_t max_triangles, size_t max_instances,
                 size_t max_blas) noexcept
        : SoftwareLbvh{device, estimate(max_triangles, max_instances, max_blas)} {}

    // ---- create (host side; mirrors create_mesh / create_accel) ----------
    [[nodiscard]] Blas create_blas(const AccelOption &option, uint triangle_offset,
                                   uint triangle_count, float3 object_min,
                                   float3 object_max) noexcept;
    [[nodiscard]] Tlas create_accel(const AccelOption &option, uint instance_count) noexcept;

    // ---- pre_build (host side; mirrors the backends' pre-build) ----------
    // Reserve the BLAS's slice of the shared storage; returns the scratch bytes
    // its build needs.
    [[nodiscard]] size_t pre_build_blas(Blas &blas) noexcept;
    // Upload the BLAS table and the instance records of `tlas`, then reserve its
    // slice of the shared storage; returns the scratch bytes its build needs.
    [[nodiscard]] size_t pre_build_accel(Stream &stream, Tlas &tlas,
                                         luisa::span<const Blas> blases,
                                         luisa::span<const InstanceDesc> instances) noexcept;

    // ---- build (records the GPU work; mirrors the backends' build) -------
    // The optional `timings` asks the build to separate its stages by a
    // synchronisation and to accumulate their host-observed times into it; the
    // default (null) keeps the plain recorded build (see `LbvhBuildTimings`).
    void build_blas(Stream &stream, const Blas &blas, const Buffer<float3> &vertices,
                    const Buffer<Triangle> &triangles,
                    AccelBuildRequest request = AccelBuildRequest::PREFER_UPDATE,
                    LbvhBuildTimings *timings = nullptr) noexcept;
    void build_accel(Stream &stream, const Tlas &tlas,
                     AccelBuildRequest request = AccelBuildRequest::PREFER_UPDATE,
                     LbvhBuildTimings *timings = nullptr) noexcept;

    // ---- compaction (the software analogue of an RTX compacted copy) ------
    // The scene must be fully built, and every BLAS and the TLAS must have been
    // created with `AccelOption::allow_compaction`: the flag is the caller's
    // intent (a hint, never a semantic change, exactly as on the hardware
    // backends), this call is the action that honours it.
    //
    // It queries the used node count on the device, reads it back with a hard
    // synchronise, allocates a node buffer of exactly that size, records the copy
    // into it, re-registers every bindless heap view onto the dense buffer, and
    // retires the loose node buffer plus the temporary buffers through a
    // *completion* callback of the same command list - so nothing the GPU still
    // reads is destroyed early.  A compacted storage has no spare capacity: a
    // later full build of the same storage must re-reserve (the `as_built` policy
    // keeps the indices valid, `subtree_contiguous` does not).
    //
    // `release_scratch` (opt-in) also retires the build scratch (the primitive
    // AABBs, both Morton-key buffers, the block AABBs and the plan) through the
    // same callback; after that the storage can only be traversed/validated, so a
    // rebuild needs a new `LbvhStorage`.
    //
    // `CompactResult::nodes` hands the dense buffer over; once the call returns
    // the storage owns it and the field is moved-from - read `nodes()` instead.
    [[nodiscard]] CompactResult compact(Stream &stream, Tlas &tlas,
                                        luisa::span<const Blas> blases,
                                        CompactionPolicy policy = CompactionPolicy::as_built,
                                        bool release_scratch = false) noexcept;

    // ---- traversal -------------------------------------------------------
    // `ray_offset` and `ray_stride` (additive, defaults 0 and 1) let a caller
    // trace a *strided slice* of the ray buffer: it walks the indices
    // `ray_offset, ray_offset + ray_stride, ...`.  That is what a caller that
    // must keep every single device submission short does - it splits the rays
    // into slices instead of issuing one dispatch over all of them - and striding
    // (rather than a contiguous range) keeps every slice representative of the
    // whole ray set.  The defaults trace the whole range contiguously, i.e. the
    // behaviour is unchanged.
    void trace_software(Stream &stream, const Buffer<float3> &vertices,
                        const Buffer<Triangle> &triangles, const Buffer<LbvhRay> &rays,
                        const Buffer<LbvhHit> &hits, const Tlas &tlas, uint ray_count,
                        uint ray_offset = 0u, uint ray_stride = 1u) noexcept;

    // Structural self-check of one built tree; returns the number of problems
    // found (see LbvhStorage::validate_tree).
    [[nodiscard]] size_t validate_tree(Stream &stream, uint node_base, uint count) noexcept {
        return _storage.validate_tree(stream, node_base, count);
    }

    // Device-side self-check of a built TLAS' bindless heap (lbvh_common.h's "The
    // bindless heap of a TLAS"): every `LbvhBlas` record of the table must name
    // its node region by the slot the layout fixes (2 + its index), and the root
    // node read through that slot must be bit-identical to the root node read
    // directly from the shared node buffer - which is exactly the resolution a
    // traversal performs.  The TLAS' own region (`heap_tlas_slot`) is checked the
    // same way.  Returns the number of problems found.
    [[nodiscard]] size_t validate_heap(Stream &stream, const Tlas &tlas,
                                       uint blas_count) noexcept;

    // ---- introspection ---------------------------------------------------
    [[nodiscard]] const Buffer<LbvhNode> &nodes() const noexcept { return _storage.nodes(); }
    [[nodiscard]] const Buffer<LbvhBlas> &blas_table() const noexcept { return _storage.blas_table(); }
    [[nodiscard]] const Buffer<LbvhInstance> &instances() const noexcept { return _storage.instances(); }
    [[nodiscard]] const LbvhStorage::Sizes &sizes() const noexcept { return _storage.sizes(); }
    [[nodiscard]] size_t primitive_count() const noexcept { return _storage.primitive_count(); }
    [[nodiscard]] size_t node_count() const noexcept { return _storage.node_count(); }
    // Node slots the *live* node buffer holds (the reserved capacity before a
    // `compact()`, exactly the kept count afterwards).
    [[nodiscard]] size_t node_capacity() const noexcept { return _storage.node_capacity(); }
    [[nodiscard]] size_t tree_count() const noexcept { return _storage.tree_count(); }
    // False once the build scratch has been released (see the `release_scratch`
    // argument of `compact`): the storage can then only be traversed/validated.
    [[nodiscard]] bool buildable() const noexcept { return _storage.buildable(); }
    [[nodiscard]] size_t blas_count() const noexcept { return _blas_count; }

private:
    LbvhStorage _storage;
    BlasBuilder _blas_builder;
    TlasBuilder _tlas_builder;
    Shader1D<BindlessArray, Buffer<LbvhBlas>, Buffer<LbvhInstance>, Buffer<float3>,
             Buffer<Triangle>, Buffer<LbvhRay>, Buffer<LbvhHit>, uint, uint, uint, uint>
        _trace_kernel;
    Shader1D<BindlessArray, Buffer<LbvhBlas>, Buffer<LbvhNode>, Buffer<uint>, uint, uint>
        _heap_check_kernel;
    Buffer<uint> _heap_problems;// one element: the mismatches of `validate_heap`
    size_t _blas_count{0u};
};

}// namespace luisa::example::lbvh
