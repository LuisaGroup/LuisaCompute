// Top-level build (instance AABBs -> `LbvhStorage::build_tree`) and the
// two-level software traversal that descends from an instance into its BLAS.

#include "tlas.h"

namespace luisa::example::lbvh {

TlasBuilder::TlasBuilder(Device &device) noexcept
    : _prim_kernel{device.compile(Kernel1D{
          // World-space AABB of every TLAS instance, from its BLAS root AABB.
          [](BufferVar<LbvhNode> nodes, BufferVar<LbvhBlas> blas_table,
             BufferVar<LbvhInstance> instances, BufferVar<LbvhPrim> prims,
             UInt prim_base, UInt count) noexcept {
              set_block_size(sort_block_size);
              UInt i = dispatch_id().x;
              $if (i < count) {
                  auto instance = instances.read(i);
                  auto blas = blas_table.read(instance.blas);
                  auto root = nodes.read(blas.node_offset);
                  auto lo = root.lo;
                  auto hi = root.hi;
                  auto w0 = instance.to_world_0;
                  auto w1 = instance.to_world_1;
                  auto w2 = instance.to_world_2;
                  auto world_lo = def(make_float3(1.0e30f));
                  auto world_hi = def(make_float3(-1.0e30f));
                  $for (c, 8u) {
                      auto corner = make_float3(select(lo.x, hi.x, (c & 1u) != 0u),
                                                select(lo.y, hi.y, (c & 2u) != 0u),
                                                select(lo.z, hi.z, (c & 4u) != 0u));
                      auto p = make_float4(corner, 1.0f);
                      auto world = make_float3(dot(p, w0), dot(p, w1), dot(p, w2));
                      world_lo = min(world_lo, world);
                      world_hi = max(world_hi, world);
                  };
                  Var<LbvhPrim> prim;
                  prim.id = i;// instance index
                  prim.lo = world_lo;
                  prim.hi = world_hi;
                  prims.write(prim_base + i, prim);
              };
          }})} {}

Tlas TlasBuilder::create(const AccelOption &option, uint instance_count) noexcept {
    // Mirror create_accel(): validate the option, then record the size.
    if (option.motion.is_enabled()) {
        LUISA_WARNING("The software LBVH has no instance motion blur; "
                      "the motion option of this TLAS is ignored.");
    }
    if (option.allow_compaction) {
        LUISA_WARNING("The software LBVH does not compact its trees; "
                      "allow_compaction of this TLAS is ignored.");
    }
    LUISA_ASSERT(instance_count > 0u, "a TLAS needs at least one instance.");
    Tlas tlas;
    tlas._option = option;
    tlas._sizes = Tlas::estimate(instance_count);
    tlas._instance_count = instance_count;
    return tlas;
}

size_t TlasBuilder::pre_build(Stream &stream, LbvhStorage &storage, Tlas &tlas,
                              luisa::span<const Blas> blases,
                              luisa::span<const InstanceDesc> instances) noexcept {
    LUISA_ASSERT(tlas.is_created(), "pre_build() on a TLAS that was not created.");
    LUISA_ASSERT(!instances.empty(), "a TLAS needs at least one instance.");
    LUISA_ASSERT(instances.size() == tlas.instance_count(),
                 "a TLAS build got {} instances but was created for {}.",
                 instances.size(), tlas.instance_count());
    // The BLAS table comes first: both the instance AABB kernel of build() and
    // the two-level traversal read it.
    luisa::vector<LbvhBlas> table;
    table.reserve(blases.size());
    for (auto &&blas : blases) { table.emplace_back(blas.record()); }
    storage.upload_blas_table(stream, luisa::span{table});
    upload_instances(stream, storage, instances);

    // World-space volume of the scene, used to normalize the Morton codes.
    auto [volume_lo, volume_hi] = instance_volume(blases, instances);
    // Reserve the acceleration structure (nodes) and the build scratch of this
    // tree, exactly like the backend pre-build allocates its buffers.
    auto range = storage.allocate(tlas._sizes.primitive_count);
    tlas._node_offset = range.node_base;
    tlas._prim_offset = range.prim_base;
    tlas._volume_lo = volume_lo;
    tlas._volume_hi = volume_hi;
    tlas._pre_built = true;
    // What the backend pre-build returns: the scratch size of the build.
    return tlas._sizes.scratch_bytes;
}

void TlasBuilder::build(Stream &stream, LbvhStorage &storage, const Tlas &tlas,
                        AccelBuildRequest request) noexcept {
    LUISA_ASSERT(tlas.is_pre_built(), "build() on a TLAS that was not pre-built.");
    // The software LBVH always rebuilds the whole tree; there is no in-place
    // update path, so the request only documents the caller's intent.
    (void)request;
    LbvhStorage::TreeRange range;
    range.prim_base = tlas.prim_offset();
    range.node_base = tlas.node_offset();
    range.count = tlas.instance_count();
    stream << _prim_kernel(storage.nodes(), storage.blas_table(), storage.instances(),
                           storage.prims(), range.prim_base, range.count)
                  .dispatch(range.count);
    storage.build_tree(stream, range, tlas._volume_lo, tlas._volume_hi);
}

void TlasBuilder::upload_instances(Stream &stream, LbvhStorage &storage,
                                   luisa::span<const InstanceDesc> instances) noexcept {
    luisa::vector<LbvhInstance> records;
    records.reserve(instances.size());
    for (auto &&instance : instances) {
        auto to_world = instance.to_world;
        auto to_object = inverse(to_world);
        records.emplace_back(LbvhInstance{
            matrix_row(to_object, 0u), matrix_row(to_object, 1u), matrix_row(to_object, 2u),
            matrix_row(to_world, 0u), matrix_row(to_world, 1u), matrix_row(to_world, 2u),
            instance.blas});
    }
    storage.upload_instances(stream, luisa::span{records});
}

std::pair<float3, float3> TlasBuilder::instance_volume(
    luisa::span<const Blas> blases, luisa::span<const InstanceDesc> instances) const noexcept {
    auto lo = make_float3(1.0e30f);
    auto hi = make_float3(-1.0e30f);
    for (auto &&instance : instances) {
        LUISA_ASSERT(instance.blas < blases.size(),
                     "a TLAS instance references an unknown BLAS.");
        auto blas_lo = blases[instance.blas].object_space_min();
        auto blas_hi = blases[instance.blas].object_space_max();
        for (auto c = 0u; c < 8u; c++) {
            auto corner = make_float3(select(blas_lo.x, blas_hi.x, (c & 1u) != 0u),
                                      select(blas_lo.y, blas_hi.y, (c & 2u) != 0u),
                                      select(blas_lo.z, blas_hi.z, (c & 4u) != 0u));
            auto p = instance.to_world * make_float4(corner, 1.0f);
            lo = min(lo, p.xyz());
            hi = max(hi, p.xyz());
        }
    }
    return {lo, hi};
}

Var<LbvhHit> tlas_traversal(const Var<LbvhRay> &ray, UInt tlas_node_offset,
                            const BufferVar<LbvhNode> &nodes,
                            const BufferVar<LbvhBlas> &blas_table,
                            const BufferVar<LbvhInstance> &instances,
                            const BufferVar<float3> &vertices,
                            const BufferVar<Triangle> &triangles) noexcept {
    auto origin = ray.origin;
    auto direction = ray.direction;
    auto t_min = ray.t_min;
    auto inv_dir = safe_reciprocal(direction);
    Var<LbvhHit> best;
    best.inst = invalid_node;
    best.prim = invalid_node;
    best.bary = make_float2(0.0f);
    best.t = ray.t_max;
    Local<uint> stack{traversal_stack_size};
    // ---- top level: instances ----
    stack[0u] = tlas_node_offset;
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
                // ---- instance leaf: descend into its BLAS ----
                auto instance = instances.read(node_prim);
                auto blas = blas_table.read(instance.blas);
                auto o4 = make_float4(origin, 1.0f);
                auto d4 = make_float4(direction, 0.0f);
                auto object_origin = make_float3(dot(o4, instance.to_object_0),
                                                 dot(o4, instance.to_object_1),
                                                 dot(o4, instance.to_object_2));
                // not renormalized: keeps the world-space ray parameter
                auto object_dir = make_float3(dot(d4, instance.to_object_0),
                                              dot(d4, instance.to_object_1),
                                              dot(d4, instance.to_object_2));
                blas_traversal(best, node_prim, blas, object_origin, object_dir, t_min,
                               nodes, vertices, triangles);
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
    return best;
}

}// namespace luisa::example::lbvh
