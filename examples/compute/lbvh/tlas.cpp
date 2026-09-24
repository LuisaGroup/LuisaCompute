// Top-level build (instance AABBs -> `LbvhStorage::build_tree`) and the
// two-level software traversal that descends from an instance into its BLAS.

#include "tlas.h"

namespace luisa::example::lbvh {

TlasBuilder::TlasBuilder(Device &device) noexcept
    : _device{&device},
      _prim_kernel{device.compile(Kernel1D{
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
                  auto lo = aabb_lo(root);
                  auto hi = aabb_hi(root);
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
                  // xyz = the world-space AABB, w = the instance id bit-cast (see
                  // `LbvhPrim`): two float4, one 32-byte sector.
                  prim.lo = make_float4(world_lo, pack_handle(i));
                  prim.hi = make_float4(world_hi, 0.0f);
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
    LUISA_ASSERT(!blases.empty(), "a TLAS needs at least one BLAS.");
    LUISA_ASSERT(instances.size() == tlas.instance_count(),
                 "a TLAS build got {} instances but was created for {}.",
                 instances.size(), tlas.instance_count());
    // The bindless heap this TLAS resolves its BLAS regions through
    // (lbvh_common.h): slot 0 is the null slot, slot 1 the TLAS' own region and
    // slot 2 + i the node region of BLAS i.  It is created here, where the BLAS
    // count is known.
    ensure_heap(tlas, blases.size());
    // The BLAS table comes first: both the instance AABB kernel of build() and
    // the two-level traversal read it.  Each record names the node region of its
    // BLAS by the bindless slot the heap reserves for it.
    luisa::vector<LbvhBlas> table;
    table.reserve(blases.size());
    for (auto i = 0u; i < blases.size(); i++) {
        table.emplace_back(blases[i].record(heap_first_blas_slot + i));
    }
    storage.upload_blas_table(stream, luisa::span{table});
    upload_instances(stream, storage, instances);

    // World-space volume of the scene, used to normalize the Morton codes.
    auto [volume_lo, volume_hi] = instance_volume(blases, instances);
    // Reserve the acceleration structure (nodes) and the build scratch of this
    // tree, exactly like the backend pre-build allocates its buffers.
    auto range = storage.allocate(tlas._sizes.primitive_count);
    tlas._node_offset = range.node_base;
    tlas._prim_offset = range.prim_base;
    tlas._plan_offset = range.plan_base;
    tlas._volume_lo = volume_lo;
    tlas._volume_hi = volume_hi;
    tlas._pre_built = true;
    // Register the node regions in the heap: a heap entry is a view of the
    // shared node buffer that *starts at the region*, so it is exactly the
    // `node_offset` / `node_count` slice of the tree (lbvh_common.h).  The TLAS'
    // own region is slot 1, and every BLAS is registered at its slot; the update
    // is a command of its own and is ordered before the build (and hence before
    // any traversal) by the stream.
    auto &nodes = storage.nodes();
    for (auto i = 0u; i < blases.size(); i++) {
        tlas._accel_heap.emplace_on_update(
            heap_first_blas_slot + i,
            nodes.view(blases[i].node_offset(), blases[i].node_count()));
    }
    tlas._accel_heap.emplace_on_update(
        heap_tlas_slot, nodes.view(tlas._node_offset, tlas.node_count()));
    stream << tlas._accel_heap.update();
    // What the backend pre-build returns: the scratch size of the build.
    return tlas._sizes.scratch_bytes;
}

void TlasBuilder::ensure_heap(Tlas &tlas, size_t blas_count) noexcept {
    // Slot 0 is the null slot, slot 1 the TLAS' own region and slot 2 + i BLAS i
    // (lbvh_common.h).
    auto needed = blas_count + heap_first_blas_slot;
    if (tlas._accel_heap && tlas._accel_heap.size() >= needed) { return; }
    if (tlas._accel_heap) {
        // A bindless array has a fixed slot count, so a TLAS that grew gets a new
        // one.  The heap it outgrew is *retired*, not destroyed: a command that
        // is still in flight may read it.
        tlas._retired_heaps.emplace_back(std::move(tlas._accel_heap));
    }
    tlas._accel_heap = _device->create_bindless_array(needed);
}

void TlasBuilder::build(Stream &stream, LbvhStorage &storage, const Tlas &tlas,
                        AccelBuildRequest request, LbvhBuildTimings *timings) noexcept {
    LUISA_ASSERT(tlas.is_pre_built(), "build() on a TLAS that was not pre-built.");
    // The software LBVH always rebuilds the whole tree; there is no in-place
    // update path, so the request only documents the caller's intent.
    (void)request;
    LbvhStorage::TreeRange range;
    range.prim_base = tlas.prim_offset();
    range.node_base = tlas.node_offset();
    range.plan_base = tlas.plan_offset();
    range.count = tlas.instance_count();
    Clock clock;
    if (timings != nullptr) { clock.tic(); }
    stream << _prim_kernel(storage.nodes(), storage.blas_table(), storage.instances(),
                           storage.prims(), range.prim_base, range.count)
                  .dispatch(range.count);
    if (timings != nullptr) {
        stream << synchronize();
        timings->prim_ms += clock.toc();
    }
    storage.build_tree(stream, range, tlas._volume_lo, tlas._volume_hi, timings);
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

Var<LbvhHit> tlas_traversal(const Var<LbvhRay> &ray, const BindlessVar &heap,
                            UInt tlas_node_offset,
                            const BufferVar<LbvhBlas> &blas_table,
                            const BufferVar<LbvhInstance> &instances,
                            const BufferVar<float3> &vertices,
                            const BufferVar<Triangle> &triangles) noexcept {
    // The TLAS' own node region is heap slot 1 (lbvh_common.h).  A heap entry is
    // a view that starts at the region, so a node handle - an absolute index in
    // the shared node buffer - is read at `handle - tlas_node_offset`.
    auto nodes = heap.buffer<LbvhNode>(heap_tlas_slot);
    auto base = tlas_node_offset;
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
    // ---- top level: instances; the same blind-push walk as `blas_traversal` ----
    stack[0u] = base;
    auto size = def(1u);
    $while (size > 0u) {
        size = size - 1u;
        auto node = nodes.read(stack[size] - base);
        $if (aabb_test(aabb_lo(node), aabb_hi(node), origin, inv_dir, t_min, best.t)) {
            auto node_left = child_left(node);
            $if (node_left == invalid_node) {
                // ---- instance leaf: descend into its BLAS ----
                auto node_prim = child_right(node);
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
                               heap, vertices, triangles);
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
    return best;
}

}// namespace luisa::example::lbvh
