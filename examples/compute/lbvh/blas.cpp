// Bottom-level build (triangle AABBs -> `LbvhStorage::build_tree`) and the
// bottom-level software traversal.
//
// The three build phases mirror the hardware backends: `create()` = create_mesh,
// `pre_build()` = the backend pre-build (size query + buffer reservation) and
// `build()` = the backend build command.

#include "blas.h"

namespace luisa::example::lbvh {

BlasBuilder::BlasBuilder(Device &device) noexcept
    : _prim_kernel{device.compile(Kernel1D{
          // AABB of the triangles of one BLAS (indices are global vertex ids).
          [](BufferVar<Triangle> triangles, BufferVar<float3> vertices, BufferVar<LbvhPrim> prims,
             UInt prim_base, UInt triangle_base, UInt count) noexcept {
              set_block_size(sort_block_size);
              UInt i = dispatch_id().x;
              $if (i < count) {
                  auto tri = triangles.read(triangle_base + i);
                  auto v0 = vertices.read(tri.i0);
                  auto v1 = vertices.read(tri.i1);
                  auto v2 = vertices.read(tri.i2);
                  Var<LbvhPrim> prim;
                  // xyz = the AABB, w = the id bit-cast (see `LbvhPrim`): the record
                  // stays one 32-byte sector, and the random prims[slot] read of
                  // the leaf pass never needs a second one.
                  prim.lo = make_float4(min(min(v0, v1), v2), pack_handle(i));
                  prim.hi = make_float4(max(max(v0, v1), v2), 0.0f);
                  prims.write(prim_base + i, prim);
              };
          }})} {}

Blas BlasBuilder::create(const AccelOption &option, uint triangle_offset,
                         uint triangle_count, float3 object_min,
                         float3 object_max) noexcept {
    // Mirror create_mesh(): validate the option, then record the geometry.
    if (option.motion.is_enabled()) {
        LUISA_WARNING("The software LBVH has no primitive motion blur; "
                      "the motion option of this BLAS is ignored.");
    }
    if (option.allow_compaction) {
        LUISA_WARNING("The software LBVH does not compact its trees; "
                      "allow_compaction of this BLAS is ignored.");
    }
    Blas blas;
    blas._option = option;
    blas._sizes = Blas::estimate(triangle_count);
    blas._triangle_offset = triangle_offset;
    blas._triangle_count = triangle_count;
    blas._lo = object_min;
    blas._hi = object_max;
    return blas;
}

size_t BlasBuilder::pre_build(LbvhStorage &storage, Blas &blas) noexcept {
    LUISA_ASSERT(blas.is_created(), "pre_build() on a BLAS that was not created.");
    // Reserve the acceleration structure (nodes) and the build scratch
    // (primitive AABBs, Morton keys) of this tree in the shared buffers,
    // exactly like the backend pre-build allocates the acceleration-structure
    // buffer after querying the driver for the required sizes.
    auto range = storage.allocate(blas._sizes.primitive_count);
    blas._node_offset = range.node_base;
    blas._prim_offset = range.prim_base;
    blas._plan_offset = range.plan_base;
    blas._pre_built = true;
    // What the backend pre-build returns: the scratch size of the build.
    return blas._sizes.scratch_bytes;
}

void BlasBuilder::build(Stream &stream, LbvhStorage &storage, const Blas &blas,
                        const Buffer<float3> &vertices, const Buffer<Triangle> &triangles,
                        AccelBuildRequest request, LbvhBuildTimings *timings) noexcept {
    LUISA_ASSERT(blas.is_pre_built(), "build() on a BLAS that was not pre-built.");
    // The RTX request (PREFER_UPDATE / FORCE_BUILD) selects in-place update on
    // the hardware backends; the software LBVH has no update path and always
    // rebuilds the tree, so the request only documents the caller's intent.
    (void)request;
    Clock clock;
    if (timings != nullptr) { clock.tic(); }
    stream << _prim_kernel(triangles, vertices, storage.prims(),
                           blas.prim_offset(), blas.triangle_offset(),
                           blas.triangle_count())
                  .dispatch(blas.triangle_count());
    if (timings != nullptr) {
        stream << synchronize();
        timings->prim_ms += clock.toc();
    }
    LbvhStorage::TreeRange range;
    range.prim_base = blas.prim_offset();
    range.node_base = blas.node_offset();
    range.plan_base = blas.plan_offset();
    range.count = blas.triangle_count();
    storage.build_tree(stream, range, blas.object_space_min(), blas.object_space_max(),
                       timings);
}

// The bottom-level walk.
//
// This is the plain (blind-push) depth-first walk: a node is loaded once, when it
// is popped, and it is tested there - so every node the walk visits costs exactly
// one load and one slab test, and a child the ray does not enter still costs a
// push, a pop and that test.
//
// Two cheaper-looking variants were implemented and measured against it and both
// *regressed* the scenes this benchmark exists for, so neither is in the code:
// testing a child before pushing it (which costs one node load per child to learn
// that it is not entered, and a zero-culling tree pays that for almost every
// child), and pushing only one child while descending into the other directly
// (which halves the stack traffic but lengthens the per-visit dependency chain;
// it lost 10-15% on coincident, bimodal and sliver-soup and only won on the
// scenes whose walk is short).  See bench/README.md for the numbers.
void blas_traversal(Var<LbvhHit> &best, UInt instance, const Var<LbvhBlas> &blas,
                    Float3 origin, Float3 direction, Float t_min,
                    const BindlessVar &heap,
                    const BufferVar<float3> &vertices,
                    const BufferVar<Triangle> &triangles) noexcept {
    // The BLAS node region is a *bindless heap entry* (lbvh_common.h): the
    // record carries the slot, and a heap entry is a view that starts at the
    // region, so a node handle - an absolute index in the shared node buffer -
    // is read at `handle - node_offset`.  The root of a tree is its region's
    // first node, i.e. `node_offset` itself, and the subtraction is
    // region-local and never negative.
    auto nodes = heap.buffer<LbvhNode>(blas.heap_slot);
    auto base = blas.node_offset;
    auto inv_dir = safe_reciprocal(direction);
    Local<uint> stack{traversal_stack_size};
    stack[0u] = base;
    auto size = def(1u);
    $while (size > 0u) {
        size = size - 1u;
        // one 32-byte node record: the two AABB planes and the two handles (see
        // `LbvhNode`); the handles are only read once the AABB test has passed,
        // and a leaf's second handle is its primitive id.
        auto node = nodes.read(stack[size] - base);
        $if (aabb_test(aabb_lo(node), aabb_hi(node), origin, inv_dir, t_min, best.t)) {
            auto node_left = child_left(node);
            $if (node_left == invalid_node) {
                auto node_prim = child_right(node);
                auto tri = triangles.read(blas.triangle_offset + node_prim);
                auto v0 = vertices.read(tri.i0);
                auto v1 = vertices.read(tri.i1);
                auto v2 = vertices.read(tri.i2);
                auto result = triangle_test(v0, v1, v2, origin, direction, t_min, best.t);
                $if (result.x >= 0.0f) {
                    best.t = result.x;
                    best.bary = make_float2(result.y, result.z);
                    best.prim = node_prim;
                    best.inst = instance;
                };
            }
            $else {
                $if (size + 2u < traversal_stack_size) {
                    stack[size] = node_left;
                    size = size + 1u;
                    stack[size] = child_right(node);
                    size = size + 1u;
                };
            };
        };
    };
}

}// namespace luisa::example::lbvh
