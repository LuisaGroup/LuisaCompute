// Fallback (software) ray tracing: the CUDA device-side traversal.
//
// This header is one of the three copies of the fallback acceleration-structure
// layout (see src/backends/common/rtx/fallback_rtx_layout.h, which is the single
// source of truth, and the HLSL copy used by the DX and Vulkan backends).  It
// cannot include that header: it is embedded into every generated shader source
// (cuda_builtin/cuda_device_fallback_rtx.h -> cuda_builtin_embedded.cpp) and
// compiled by NVRTC, which sees neither the host headers nor the DSL.
//
// The code generators (cuda_codegen_ast.cpp / cuda_codegen_xir.cpp) only append
// this text to a shader that is compiled in *fallback mode* (see
// CUDADevice::use_fallback_rtx()).  When the fallback is off this file is not
// part of any generated source, so the hardware path stays byte-for-byte what it
// was.
//
// ---------------------------------------------------------------------------
// The ABI, restated (see fallback_rtx_layout.h for the authoritative version)
// ---------------------------------------------------------------------------
//
// One *acceleration buffer* (`Buffer<uint4>`) holds every tree, one *instance
// buffer* (`Buffer<uint4>`) holds the TLAS instance records.  An `accel` shader
// argument carries the device addresses of the two regions in `LCAccel`:
//
//   accel.handle    = address of the acceleration-buffer region of this tree
//   accel.instances = address of the instance buffer (the TLAS records)
//
// A region starts with a 4 x uint4 header whose first lane is the region's own
// absolute uint4 offset `base`; the descriptor handed to a shader starts at the
// region, so every other absolute handle `h` is read as `accel[h - base]`.
//
//   u4 0 : (base,            node_base,        node_count,  prim_count)
//   u4 1 : (blas_table_base, blas_count,       index_base,  index_count)
//   u4 2 : (vertex_base,     vertex_count,     root,        flags)
//   u4 3 : reserved
//
//   node    : 2 x u4, (lo.x, lo.y, lo.z, left), (hi.x, hi.y, hi.z, right)
//   blas    : 2 x u4, (blas_base, node_base, index_base, vertex_base),
//                     (triangle_count, flags, reserved, reserved)
//   index   : 1 x u4, (i0, i1, i2, 0)
//   vertex  : 1 x u4, (x, y, z, 0)
//   inst    : 8 x u4, three to_object rows, three to_world rows,
//                     (blas_index, visibility, user_id, flags), reserved
//
// Every float of the ABI travels as the *bit pattern* of its lane (the buffers
// are uint4), so the loaders below bit-cast; handles are never floats, which is
// what keeps `invalid_offset` alive across the round trip.
//
// A node is a leaf when its left handle is `invalid_offset`; `right` is then the
// triangle index inside the BLAS (or the instance index, in a TLAS).  A TLAS
// leaf names its BLAS through the instance record's `blas_index`, which indexes
// the BLAS table; the table record names the BLAS region, whose header is
// re-read exactly like the top-level region (a region is self-describing, which
// is why it starts with its own base).
//
// The walk below is the same two-level blind-push depth-first walk as
// examples/compute/lbvh/{blas,tlas}.cpp: a node is loaded once, when popped, and
// is tested there, and the instance transform of a TLAS leaf is applied to the
// ray without renormalizing the direction, which keeps `t` the world-space ray
// parameter (this is what makes the software result match the hardware one).
//
// Limitations, reported rather than hidden:
//   * `opaque == false` instances are *hits* like opaque ones: the fallback has
//     no any-hit shading, so there is nothing to call.  The instance flag is
//     recorded by the build (instance_flag_opaque) and only affects nothing
//     else; see RAY_TRACING_SET_INSTANCE_OPACITY.
//   * ray queries, motion blur, curves and procedural primitives are not
//     implemented; the code generators refuse to emit them in fallback mode
//     instead of calling into a missing symbol.
//   * `flip_facing` / `disable_face_culling` do not change the intersection
//     test: triangles are two-sided here, exactly like the hardware geometry the
//     CUDA backend builds (`GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING`).

#pragma once

// ---------------------------------------------------------------------------
// Header slot indices, in uints (four per uint4) - restated from
// fallback_rtx_layout.h.
// ---------------------------------------------------------------------------
inline constexpr lc_uint lc_fallback_header_u4 = 4u;
inline constexpr lc_uint lc_fallback_node_u4 = 2u;
inline constexpr lc_uint lc_fallback_blas_record_u4 = 2u;
inline constexpr lc_uint lc_fallback_instance_u4 = 8u;
// Instance-record lane block of the to_object rows / the misc word.
inline constexpr lc_uint lc_fallback_i_to_object = 0u;
inline constexpr lc_uint lc_fallback_i_to_world = 3u;
inline constexpr lc_uint lc_fallback_i_misc = 6u;
// Lanes of the misc word.
inline constexpr lc_uint lc_fallback_im_blas_index = 0u;
inline constexpr lc_uint lc_fallback_im_visibility = 1u;
inline constexpr lc_uint lc_fallback_im_user_id = 2u;
inline constexpr lc_uint lc_fallback_im_flags = 3u;
// The left handle of a leaf node, and the "no hit" marker of a traversal.
inline constexpr lc_uint lc_fallback_invalid_offset = 0xFFFFFFFFu;
// Depth of the software traversal stack.  A Morton-code radix tree over the
// primitives of one tree is far shallower than this (the LBVH example uses the
// same bound).
inline constexpr lc_uint lc_fallback_stack_size = 64u;

static_assert(lc_fallback_node_u4 * 16u == 32u, "a fallback node is one 32-byte sector");
static_assert(lc_fallback_instance_u4 * 16u == 128u, "a fallback instance record is 128 bytes");
static_assert(lc_fallback_header_u4 * 16u == 64u, "a fallback region header is 64 bytes");

// ---------------------------------------------------------------------------
// Lane loaders: the ABI stores floats as the bit pattern of a uint lane.
// ---------------------------------------------------------------------------
[[nodiscard]] __device__ inline lc_float lc_fallback_as_float(lc_uint u) noexcept {
    return __uint_as_float(u);
}

[[nodiscard]] __device__ inline lc_uint lc_fallback_as_uint(lc_float f) noexcept {
    return __float_as_uint(f);
}

[[nodiscard]] __device__ inline lc_float2 lc_fallback_float2_of(lc_uint4 v) noexcept {
    return lc_make_float2(__uint_as_float(v.x), __uint_as_float(v.y));
}

[[nodiscard]] __device__ inline lc_float3 lc_fallback_float3_of(lc_uint4 v) noexcept {
    return lc_make_float3(__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z));
}

[[nodiscard]] __device__ inline lc_float4 lc_fallback_float4_of(lc_uint4 v) noexcept {
    return lc_make_float4(__uint_as_float(v.x), __uint_as_float(v.y),
                          __uint_as_float(v.z), __uint_as_float(v.w));
}

[[nodiscard]] __device__ inline lc_uint4 lc_fallback_uint4_of(lc_float4 v) noexcept {
    return lc_make_uint4(__float_as_uint(v.x), __float_as_uint(v.y),
                         __float_as_uint(v.z), __float_as_uint(v.w));
}

// ---------------------------------------------------------------------------
// The fallback `accel` argument.
//
// A fallback TLAS does not hand a traversal an absolute uint4 offset any more:
// it owns a *bindless heap* whose slots hold the region buffers, and the blas
// table carries the *slot* of the tree it references (fallback_rtx_layout.h).
// The kernel argument therefore carries the heap and the slot of the TLAS' own
// region instead of the address of one shared buffer.
//
// The layout is 32 bytes and mirrors the host-side `FallbackAccelArgument` of
// cuda_shader_native.cpp byte for byte.
// ---------------------------------------------------------------------------
struct alignas(16u) LCFallbackAccel {
    unsigned long long heap_slots;   // device address of the heap's `LCBindlessSlot` array
    unsigned long long heap_capacity;// slots of that array
    unsigned long long instances;    // device address of the instance buffer slice
    lc_uint region_slot;             // heap slot of this TLAS' own region
    lc_uint pad;
};

static_assert(sizeof(LCFallbackAccel) == 32u, "the fallback accel argument is 32 bytes");

[[nodiscard]] __device__ inline LCBindlessArray lc_fallback_heap(LCFallbackAccel accel) noexcept {
    return LCBindlessArray{reinterpret_cast<const LCBindlessSlot *>(accel.heap_slots),
                           static_cast<size_t>(accel.heap_capacity)};
}

// One uint4 of the region a heap slot names.  A heap entry is a *view that
// starts at the region*, so every index below is region-relative.
[[nodiscard]] __device__ inline lc_uint4 lc_fallback_region_read(
    LCFallbackAccel accel, lc_uint slot, lc_uint index) noexcept {
    return lc_bindless_buffer_read<lc_uint4>(lc_fallback_heap(accel), slot, index);
}

// ---------------------------------------------------------------------------
// A view of one region: the heap it lives in (and its slot) plus the header
// fields a walk needs.  `at(h)` is the *only* place that knows about the base
// subtraction: a node handle is an absolute uint4 offset, while a heap entry
// starts at element 0 of the region.
// ---------------------------------------------------------------------------
struct LCFallbackRegion {

    LCFallbackAccel accel;// the argument that names the heap
    lc_uint slot;         // the heap slot of this region
    lc_uint base;         // absolute uint4 offset of element 0 of the region
    lc_uint node_base;    // absolute uint4 offset of the node array
    lc_uint blas_table_base;// absolute uint4 offset of the blas table (TLAS only)
    lc_uint index_base;   // absolute uint4 offset of the index array (BLAS only)
    lc_uint vertex_base;  // absolute uint4 offset of the vertex array (BLAS only)
    lc_uint root;         // absolute uint4 handle of the root node
    lc_uint flags;        // region flags (region_flag_tlas for a TLAS)

    [[nodiscard]] __device__ inline lc_uint4 at(lc_uint handle) const noexcept {
        return lc_fallback_region_read(accel, slot, handle - base);
    }
};

// Read the header of the region the heap slot `slot` names.
[[nodiscard]] __device__ inline LCFallbackRegion lc_fallback_region(
    LCFallbackAccel accel, lc_uint slot) noexcept {
    // u4 0 : (base, node_base, node_count, prim_count)
    // u4 1 : (blas_table_base, blas_count, index_base, index_count)
    // u4 2 : (vertex_base, vertex_count, root, flags)
    auto h0 = lc_fallback_region_read(accel, slot, 0u);
    auto h1 = lc_fallback_region_read(accel, slot, 1u);
    auto h2 = lc_fallback_region_read(accel, slot, 2u);
    LCFallbackRegion region;
    region.accel = accel;
    region.slot = slot;
    region.base = h0.x;
    region.node_base = h0.y;
    region.blas_table_base = h1.x;
    region.index_base = h1.z;
    region.vertex_base = h2.x;
    region.root = h2.z;
    region.flags = h2.w;
    return region;
}

// ---------------------------------------------------------------------------
// Ray intersection helpers (the same maths as examples/compute/lbvh).
// ---------------------------------------------------------------------------

// Reciprocal of one lane that never produces an infinity from a zero direction
// (the same clamp the LBVH example uses).
[[nodiscard]] __device__ inline lc_float lc_fallback_reciprocal_lane(lc_float x) noexcept {
    return 1.0f / (fabsf(x) < 1.0e-20f ? 1.0e-20f : x);
}

// Reciprocal that never produces a NaN from a zero direction.
[[nodiscard]] __device__ inline lc_float3 lc_fallback_safe_reciprocal(lc_float3 d) noexcept {
    return lc_make_float3(lc_fallback_reciprocal_lane(d.x),
                          lc_fallback_reciprocal_lane(d.y),
                          lc_fallback_reciprocal_lane(d.z));
}

// Slab test of an AABB, bounded by the current best hit.
[[nodiscard]] __device__ inline bool lc_fallback_aabb_test(
    lc_float3 lo, lc_float3 hi, lc_float3 origin, lc_float3 inv_dir,
    lc_float t_min, lc_float t_max) noexcept {
    auto t0 = (lo - origin) * inv_dir;
    auto t1 = (hi - origin) * inv_dir;
    auto near_t = lc_min(t0, t1);
    auto far_t = lc_max(t0, t1);
    auto t_near = lc_max(lc_max(near_t.x, near_t.y), lc_max(near_t.z, t_min));
    auto t_far = lc_min(lc_min(far_t.x, far_t.y), lc_min(far_t.z, t_max));
    return t_near <= t_far;
}

// Two-sided Moller-Trumbore.  Returns `true` and fills `t` / `bary` on a hit;
// `bary` follows the Luisa convention (`w0 = 1 - u - v`, `w1 = u`, `w2 = v`),
// which is what the hardware path's `_optix_get_triangle_barycentrics` reports
// too.  The transaction is not kept when it is not closer than `t_max`.
[[nodiscard]] __device__ inline bool lc_fallback_triangle_test(
    lc_float3 v0, lc_float3 v1, lc_float3 v2,
    lc_float3 origin, lc_float3 dir,
    lc_float t_min, lc_float t_max,
    lc_float &t_out, lc_float2 &bary_out) noexcept {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pv = lc_cross(dir, e2);
    auto det = lc_dot(e1, pv);
    auto inv_det = 1.0f / det;
    auto tv = origin - v0;
    auto u = lc_dot(tv, pv) * inv_det;
    auto qv = lc_cross(tv, e1);
    auto v = lc_dot(dir, qv) * inv_det;
    auto t = lc_dot(e2, qv) * inv_det;
    auto hit = (fabsf(det) > 1.0e-12f) & (u >= 0.0f) & (v >= 0.0f) &
               (u + v <= 1.0f) & (t >= t_min) & (t <= t_max);
    if (!hit) { return false; }
    t_out = t;
    bary_out = lc_make_float2(u, v);
    return true;
}

// ---------------------------------------------------------------------------
// The bottom-level walk: blind-push depth first over the nodes of one BLAS.
//
// `desc` points at the BLAS region (element 0 is its header), `instance` is the
// instance index a hit found here is reported with, and the ray is already in
// the object space of that instance.  The best hit (`t_best` / `inst_best` /
// `prim_best` / `bary_best`, in world space) is written *only* where a triangle
// is closer than the current best, and the walk returns whether it improved the
// hit at all.
//
// The return value is the whole point of this contract: the TLAS walk has to
// commit a descent iff it improved the hit.  Two instances of the same mesh can
// be hit on the *same* local triangle index - a lattice of repeated instances
// does that constantly - so a caller that compares the primitive index instead
// silently keeps the farther instance (this is the DirectX workstream's report,
// see `_LCFbWalkBlas` in src/backends/common/hlsl/builtin/
// fallback_rtx_header.bytes, whose semantics this copy matches).
// ---------------------------------------------------------------------------
[[nodiscard]] __device__ inline bool lc_fallback_walk_blas(
    LCFallbackAccel accel, lc_uint slot, lc_uint instance,
    lc_float3 origin, lc_float3 direction, lc_float t_min,
    lc_float &t_best, lc_uint &inst_best, lc_uint &prim_best, lc_float2 &bary_best) noexcept {

    auto improved = false;
    auto region = lc_fallback_region(accel, slot);
    auto inv_dir = lc_fallback_safe_reciprocal(direction);
    lc_uint stack[lc_fallback_stack_size];
    lc_uint size = 1u;
    stack[0u] = region.root;
    while (size != 0u) {
        auto handle = stack[--size];
        // One 32-byte node, read as its two uint4 planes.
        auto node_lo = region.at(handle);
        auto node_hi = region.at(handle + 1u);
        auto lo = lc_fallback_float3_of(node_lo);
        auto hi = lc_fallback_float3_of(node_hi);
        if (!lc_fallback_aabb_test(lo, hi, origin, inv_dir, t_min, t_best)) { continue; }
        auto left = node_lo.w;
        if (left == lc_fallback_invalid_offset) {
            // ---- triangle leaf: `right` is the local triangle index ----
            auto prim = node_hi.w;
            auto index = region.at(region.index_base + prim);
            auto v0 = lc_fallback_float3_of(region.at(region.vertex_base + index.x));
            auto v1 = lc_fallback_float3_of(region.at(region.vertex_base + index.y));
            auto v2 = lc_fallback_float3_of(region.at(region.vertex_base + index.z));
            auto t = t_best;
            auto bary = bary_best;
            if (lc_fallback_triangle_test(v0, v1, v2, origin, direction,
                                          t_min, t_best, t, bary)) {
                t_best = t;
                bary_best = bary;
                prim_best = prim;
                inst_best = instance;
                improved = true;
            }
        } else {
            // ---- internal node: push both children (blind push) ----
            if (size + 2u < lc_fallback_stack_size) {
                stack[size++] = left;
                stack[size++] = node_hi.w;
            }
        }
    }
    return improved;
}

// ---------------------------------------------------------------------------
// The top-level walk: the same blind-push walk over the instances of a TLAS.
// A leaf is an instance; it is descended into through the BLAS it references.
// ---------------------------------------------------------------------------
__device__ inline void lc_fallback_walk_tlas(
    LCFallbackAccel accel, lc_uint mask,
    lc_float3 origin, lc_float3 direction, lc_float t_min,
    lc_float &t_best, lc_uint &inst_best, lc_uint &prim_best, lc_float2 &bary_best) noexcept {

    // The TLAS' own region is the heap slot the argument carries; every index
    // below is relative to it (a heap entry is a view that starts at the
    // region).
    auto region = lc_fallback_region(accel, accel.region_slot);
    auto instances = reinterpret_cast<const lc_uint4 *>(accel.instances);
    auto inv_dir = lc_fallback_safe_reciprocal(direction);
    auto o4 = lc_make_float4(origin.x, origin.y, origin.z, 1.0f);
    auto d4 = lc_make_float4(direction.x, direction.y, direction.z, 0.0f);
    lc_uint stack[lc_fallback_stack_size];
    lc_uint size = 1u;
    stack[0u] = region.root;
    while (size != 0u) {
        auto handle = stack[--size];
        auto node_lo = region.at(handle);
        auto node_hi = region.at(handle + 1u);
        auto lo = lc_fallback_float3_of(node_lo);
        auto hi = lc_fallback_float3_of(node_hi);
        if (!lc_fallback_aabb_test(lo, hi, origin, inv_dir, t_min, t_best)) { continue; }
        auto left = node_lo.w;
        if (left == lc_fallback_invalid_offset) {
            // ---- instance leaf ----
            auto instance = node_hi.w;
            auto record = instances + lc_fallback_instance_u4 * instance;
            auto misc = record[lc_fallback_i_misc];
            // Visibility: an instance is culled when its mask and the ray's
            // share no bit, exactly like the hardware's instance visibility.
            if ((mask & misc.y) == 0u) { continue; }
            // The BLAS table record of this instance names the tree it refers
            // to by the *bindless slot* of its region, which the build copied
            // into the record's metadata lane (fallback_rtx_layout.h).
            auto blas_index = misc.x;
            auto row = region.blas_table_base +
                       lc_fallback_blas_record_u4 * blas_index;
            auto metadata = region.at(row + 1u);
            auto blas_slot = metadata.z;
            // Slot 0 is the *null* slot: the build never registers a region
            // there, so a row whose slot lane is 0 is an instance the caller
            // never gave a mesh.  Skip it instead of reading a slot that holds
            // no buffer.
            if (blas_slot == 0u) { continue; }
            // The referenced BLAS region is resolved through the heap; the
            // descent then indexes that region relatively, exactly like the
            // TLAS walk indexes its own.  There is no backwards descriptor
            // arithmetic any more: a heap entry is a view that starts at the
            // region.
            // World space -> object space: the three to_object rows as an
            // explicit row-major affine.  The direction is *not* renormalized,
            // so the triangle test keeps the world-space ray parameter.
            auto to_object_0 = lc_fallback_float4_of(record[lc_fallback_i_to_object + 0u]);
            auto to_object_1 = lc_fallback_float4_of(record[lc_fallback_i_to_object + 1u]);
            auto to_object_2 = lc_fallback_float4_of(record[lc_fallback_i_to_object + 2u]);
            auto object_origin = lc_make_float3(lc_dot(o4, to_object_0),
                                                lc_dot(o4, to_object_1),
                                                lc_dot(o4, to_object_2));
            auto object_dir = lc_make_float3(lc_dot(d4, to_object_0),
                                             lc_dot(d4, to_object_1),
                                             lc_dot(d4, to_object_2));
            // The descent reports whether it improved the hit, and writes the
            // hit it found into the copies handed to it, so the commit below is
            // exactly "it is closer than what we had".  Comparing the primitive
            // index instead would be wrong: two instances of the same mesh can
            // be hit on the same local triangle, and the nearer hit would then
            // be dropped in favour of the farther one.
            auto t_instance = t_best;
            auto inst_instance = inst_best;
            auto prim_instance = prim_best;
            auto bary_instance = bary_best;
            if (lc_fallback_walk_blas(accel, blas_slot, instance, object_origin, object_dir,
                                      t_min, t_instance, inst_instance, prim_instance, bary_instance)) {
                t_best = t_instance;
                inst_best = inst_instance;
                prim_best = prim_instance;
                bary_best = bary_instance;
            }
        } else {
            if (size + 2u < lc_fallback_stack_size) {
                stack[size++] = left;
                stack[size++] = node_hi.w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The ABI entry points the code generators emit.
// ---------------------------------------------------------------------------

// Closest hit of `ray` against `mask`-visible instances.  On a miss the hit's
// instance index is `invalid_offset` (the marker the DSL's `SurfaceHit::miss()`
// tests), its primitive index is `invalid_offset`, its barycentrics are zero and
// its `t` is the ray's `t_max`.
[[nodiscard]] __device__ inline LCTriangleHit lc_fallback_trace_closest(
    LCFallbackAccel accel, LCRay ray, lc_uint mask) noexcept {
    LCTriangleHit hit{lc_fallback_invalid_offset, lc_fallback_invalid_offset,
                      lc_make_float2(0.0f, 0.0f), ray.m3};
    if (accel.heap_slots == 0ull || accel.instances == 0ull) { return hit; }
    auto origin = lc_make_float3(ray.m0[0], ray.m0[1], ray.m0[2]);
    auto direction = lc_make_float3(ray.m2[0], ray.m2[1], ray.m2[2]);
    auto t_best = ray.m3;
    auto inst_best = lc_fallback_invalid_offset;
    auto prim_best = lc_fallback_invalid_offset;
    auto bary_best = lc_make_float2(0.0f, 0.0f);
    lc_fallback_walk_tlas(accel, mask, origin, direction, ray.m1,
                          t_best, inst_best, prim_best, bary_best);
    if (inst_best != lc_fallback_invalid_offset) {
        hit.m0 = inst_best;
        hit.m1 = prim_best;
        hit.m2 = bary_best;
        hit.m3 = t_best;
    }
    return hit;
}

// Whether *any* triangle of a `mask`-visible instance is hit by `ray`.
[[nodiscard]] __device__ inline bool lc_fallback_trace_any(
    LCFallbackAccel accel, LCRay ray, lc_uint mask) noexcept {
    if (accel.heap_slots == 0ull || accel.instances == 0ull) { return false; }
    auto origin = lc_make_float3(ray.m0[0], ray.m0[1], ray.m0[2]);
    auto direction = lc_make_float3(ray.m2[0], ray.m2[1], ray.m2[2]);
    auto t_best = ray.m3;
    auto inst_best = lc_fallback_invalid_offset;
    auto prim_best = lc_fallback_invalid_offset;
    auto bary_best = lc_make_float2(0.0f, 0.0f);
    lc_fallback_walk_tlas(accel, mask, origin, direction, ray.m1,
                          t_best, inst_best, prim_best, bary_best);
    return inst_best != lc_fallback_invalid_offset;
}

// ---------------------------------------------------------------------------
// Instance accessors (RAY_TRACING_INSTANCE_*) and their setters
// (RAY_TRACING_SET_INSTANCE_*).  They read and write the instance buffer, i.e.
// the records the build uploaded; the row convention of
// `lc_accel_instance_transform` is reproduced exactly, so a shader that reads
// back what it wrote sees the same matrix on both paths.
// ---------------------------------------------------------------------------

[[nodiscard]] __device__ inline lc_float4x4 lc_fallback_instance_transform(
    LCFallbackAccel accel, lc_uint instance_id) noexcept {
    auto record = reinterpret_cast<const lc_uint4 *>(accel.instances) +
                  lc_fallback_instance_u4 * instance_id;
    // to_world rows, as the rows of the object->world matrix.
    auto w0 = lc_fallback_float4_of(record[lc_fallback_i_to_world + 0u]);
    auto w1 = lc_fallback_float4_of(record[lc_fallback_i_to_world + 1u]);
    auto w2 = lc_fallback_float4_of(record[lc_fallback_i_to_world + 2u]);
    // Identical to `lc_accel_instance_transform`, which unpacks the same three
    // rows from the OptiX instance record.
    return lc_make_float4x4(
        w0.x, w1.x, w2.x, 0.0f,
        w0.y, w1.y, w2.y, 0.0f,
        w0.z, w1.z, w2.z, 0.0f,
        w0.w, w1.w, w2.w, 1.0f);
}

[[nodiscard]] __device__ inline lc_uint lc_fallback_instance_user_id(
    LCFallbackAccel accel, lc_uint instance_id) noexcept {
    auto record = reinterpret_cast<const lc_uint4 *>(accel.instances) +
                  lc_fallback_instance_u4 * instance_id;
    return record[lc_fallback_i_misc][lc_fallback_im_user_id];
}

[[nodiscard]] __device__ inline lc_uint lc_fallback_instance_visibility(
    LCFallbackAccel accel, lc_uint instance_id) noexcept {
    auto record = reinterpret_cast<const lc_uint4 *>(accel.instances) +
                  lc_fallback_instance_u4 * instance_id;
    return record[lc_fallback_i_misc][lc_fallback_im_visibility];
}

__device__ inline void lc_fallback_set_instance_transform(
    LCFallbackAccel accel, lc_uint instance_id, lc_float4x4 m) noexcept {
    auto record = reinterpret_cast<lc_uint4 *>(accel.instances) +
                  lc_fallback_instance_u4 * instance_id;
    // to_world: the rows of the object->world matrix, the same twelve floats
    // `AccelBuildCommand::Modification::set_transform` packs.
    auto w0 = lc_make_float4(m[0][0], m[1][0], m[2][0], m[3][0]);
    auto w1 = lc_make_float4(m[0][1], m[1][1], m[2][1], m[3][1]);
    auto w2 = lc_make_float4(m[0][2], m[1][2], m[2][2], m[3][2]);
    record[lc_fallback_i_to_world + 0u] = lc_fallback_uint4_of(w0);
    record[lc_fallback_i_to_world + 1u] = lc_fallback_uint4_of(w1);
    record[lc_fallback_i_to_world + 2u] = lc_fallback_uint4_of(w2);
    // to_object: the traversal transforms the ray with these rows, so they have
    // to follow the transform the caller just set.  The affine inverse of a
    // rotation/scale/translation is computed explicitly (3x3 inverse + the
    // negated transformed translation); a singular matrix leaves them as they
    // were, which culls nothing but also invents no geometry.
    auto a00 = w0.x, a01 = w0.y, a02 = w0.z, t0 = w0.w;
    auto a10 = w1.x, a11 = w1.y, a12 = w1.z, t1 = w1.w;
    auto a20 = w2.x, a21 = w2.y, a22 = w2.z, t2 = w2.w;
    auto c00 = a11 * a22 - a12 * a21;
    auto c01 = a12 * a20 - a10 * a22;
    auto c02 = a10 * a21 - a11 * a20;
    auto det = a00 * c00 + a01 * c01 + a02 * c02;
    if (fabsf(det) > 1.0e-20f) {
        auto inv_det = 1.0f / det;
        // rows of the inverse linear part
        auto i00 = c00 * inv_det;
        auto i01 = (a02 * a21 - a01 * a22) * inv_det;
        auto i02 = (a01 * a12 - a02 * a11) * inv_det;
        auto i10 = c01 * inv_det;
        auto i11 = (a00 * a22 - a02 * a20) * inv_det;
        auto i12 = (a02 * a10 - a00 * a12) * inv_det;
        auto i20 = c02 * inv_det;
        auto i21 = (a01 * a20 - a00 * a21) * inv_det;
        auto i22 = (a00 * a11 - a01 * a10) * inv_det;
        auto o0 = lc_make_float4(i00, i01, i02, -(i00 * t0 + i01 * t1 + i02 * t2));
        auto o1 = lc_make_float4(i10, i11, i12, -(i10 * t0 + i11 * t1 + i12 * t2));
        auto o2 = lc_make_float4(i20, i21, i22, -(i20 * t0 + i21 * t1 + i22 * t2));
        record[lc_fallback_i_to_object + 0u] = lc_fallback_uint4_of(o0);
        record[lc_fallback_i_to_object + 1u] = lc_fallback_uint4_of(o1);
        record[lc_fallback_i_to_object + 2u] = lc_fallback_uint4_of(o2);
    }
}

__device__ inline void lc_fallback_set_instance_visibility(
    LCFallbackAccel accel, lc_uint instance_id, lc_uint mask) noexcept {
    auto record = reinterpret_cast<lc_uint4 *>(accel.instances) +
                  lc_fallback_instance_u4 * instance_id;
    // The hardware masks the visibility to 8 bits (`lc_accel_set_instance_visibility`).
    record[lc_fallback_i_misc][lc_fallback_im_visibility] = mask & 0xffu;
}

__device__ inline void lc_fallback_set_instance_opacity(
    LCFallbackAccel accel, lc_uint instance_id, bool opaque) noexcept {
    auto record = reinterpret_cast<lc_uint4 *>(accel.instances) +
                  lc_fallback_instance_u4 * instance_id;
    auto flags = record[lc_fallback_i_misc][lc_fallback_im_flags];
    // `LC_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING` marks a triangle mesh
    // (procedural primitives ignore the opaque flag), exactly like the hardware
    // setter.  The fallback has no any-hit shading, so this only records the
    // flag: a non-opaque instance is still a hit (documented limitation above).
    if ((flags & (1u << 0u)) != 0u) {
        flags &= ~((1u << 2u) | (1u << 3u));
        flags |= opaque ? (1u << 2u) : (1u << 3u);
        record[lc_fallback_i_misc][lc_fallback_im_flags] = flags;
    }
}

__device__ inline void lc_fallback_set_instance_user_id(
    LCFallbackAccel accel, lc_uint instance_id, lc_uint user_id) noexcept {
    auto record = reinterpret_cast<lc_uint4 *>(accel.instances) +
                  lc_fallback_instance_u4 * instance_id;
    record[lc_fallback_i_misc][lc_fallback_im_user_id] = user_id;
}
