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
                  prim.id = i;// local triangle index inside this BLAS
                  prim.lo = min(min(v0, v1), v2);
                  prim.hi = max(max(v0, v1), v2);
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
    blas._pre_built = true;
    // What the backend pre-build returns: the scratch size of the build.
    return blas._sizes.scratch_bytes;
}

void BlasBuilder::build(Stream &stream, LbvhStorage &storage, const Blas &blas,
                        const Buffer<float3> &vertices, const Buffer<Triangle> &triangles,
                        AccelBuildRequest request) noexcept {
    LUISA_ASSERT(blas.is_pre_built(), "build() on a BLAS that was not pre-built.");
    // The RTX request (PREFER_UPDATE / FORCE_BUILD) selects in-place update on
    // the hardware backends; the software LBVH has no update path and always
    // rebuilds the tree, so the request only documents the caller's intent.
    (void)request;
    stream << _prim_kernel(triangles, vertices, storage.prims(),
                           blas.prim_offset(), blas.triangle_offset(),
                           blas.triangle_count())
                  .dispatch(blas.triangle_count());
    LbvhStorage::TreeRange range;
    range.prim_base = blas.prim_offset();
    range.node_base = blas.node_offset();
    range.count = blas.triangle_count();
    storage.build_tree(stream, range, blas.object_space_min(), blas.object_space_max());
}

void blas_traversal(Var<LbvhHit> &best, UInt instance, const Var<LbvhBlas> &blas,
                    Float3 origin, Float3 direction, Float t_min,
                    const BufferVar<LbvhNode> &nodes,
                    const BufferVar<float3> &vertices,
                    const BufferVar<Triangle> &triangles) noexcept {
    auto inv_dir = safe_reciprocal(direction);
    Local<uint> stack{traversal_stack_size};
    stack[0u] = blas.node_offset;
    auto size = def(1u);
    $while (size > 0u) {
        size = size - 1u;
        auto node = nodes.read(stack[size]);
        auto node_lo = node.lo;
        auto node_hi = node.hi;
        auto node_left = node.left;
        auto node_right = node.right;
        auto node_prim = node.prim;
        $if (aabb_test(node_lo, node_hi, origin, inv_dir, t_min, best.t)) {
            $if (node_left == invalid_node) {
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
                    stack[size] = node_right;
                    size = size + 1u;
                };
            };
        };
    };
}

}// namespace luisa::example::lbvh
