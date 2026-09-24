// Implementation of the tree-agnostic LBVH build stages (Morton codes, LSD
// radix sort, Karras radix tree) plus the shared GPU storage.

#include "lbvh_storage.h"

#include <algorithm>

namespace luisa::example::lbvh {

LbvhStorage::Sizes LbvhStorage::estimate(size_t max_triangles, size_t max_instances,
                                         size_t max_blas) noexcept {
    LUISA_ASSERT(max_triangles + max_instances > 0u,
                 "software LBVH storage needs at least one primitive.");
    Sizes sizes;
    sizes.primitive_capacity = max_triangles + max_instances;
    sizes.node_capacity = sizes.primitive_capacity * 2u;
    sizes.blas_capacity = max_blas;
    sizes.instance_capacity = max_instances;
    sizes.primitive_bytes = sizes.primitive_capacity * sizeof(LbvhPrim);
    sizes.key_bytes = sizes.primitive_capacity * sizeof(LbvhKey);
    sizes.node_bytes = sizes.node_capacity * sizeof(LbvhNode);
    // Block AABBs of the two-level reduction (see `node_reduction_block`): one
    // per `node_reduction_block` slots of the shared node buffer (leaves live at
    // their own global slot, so no per-tree rounding is involved).
    sizes.block_capacity = (sizes.node_capacity + node_reduction_block - 1u) /
                           node_reduction_block;
    sizes.block_bytes = sizes.block_capacity * sizeof(LbvhNode);
    // One plan record per internal node; a scene has at most one internal node per
    // primitive (a tree of `count` primitives has `count - 1` of them).
    sizes.plan_capacity = sizes.primitive_capacity;
    sizes.plan_bytes = sizes.plan_capacity * sizeof(uint4);
    sizes.blas_table_bytes = sizes.blas_capacity * sizeof(LbvhBlas);
    sizes.instance_bytes = sizes.instance_capacity * sizeof(LbvhInstance);
    // Worst case of the parallel sort's per-block histogram/scan scratch (see
    // LbvhRadixSort::scratch_bytes_for): the size query has to report everything
    // the build will allocate, exactly like the scratch size of a backend build.
    sizes.sort_scratch_bytes = LbvhRadixSort::scratch_bytes_for(sizes.primitive_capacity);
    return sizes;
}

LbvhStorage::LbvhStorage(Device &device, const Sizes &sizes) noexcept
    : _sizes{sizes},
      _prims{device.create_buffer<LbvhPrim>(_sizes.primitive_capacity)},
      _keys_a{device.create_buffer<LbvhKey>(_sizes.primitive_capacity)},
      _keys_b{device.create_buffer<LbvhKey>(_sizes.primitive_capacity)},
      _nodes{device.create_buffer<LbvhNode>(_sizes.node_capacity)},
      _blocks{device.create_buffer<LbvhNode>(_sizes.block_capacity)},
      _plan{device.create_buffer<uint4>(_sizes.plan_capacity)},
      _blas_table{device.create_buffer<LbvhBlas>(_sizes.blas_capacity)},
      _instances{device.create_buffer<LbvhInstance>(_sizes.instance_capacity)},
      _sort{device, _sizes.primitive_capacity},
      _warp_size{device.compute_warp_size()},

      // 30-bit Morton code of every primitive.
      _morton_kernel{device.compile(Kernel1D{
          [](BufferVar<LbvhPrim> prims, BufferVar<LbvhKey> keys, UInt prim_base, UInt count,
             Float3 scene_min, Float3 scene_inv_extent) noexcept {
              set_block_size(sort_block_size);
              // "expandBits" of VkLBVH / Karras 2012: spread 10 bits to 30.
              auto expand_bits = [](UInt v) noexcept {
                  v = (v * 0x00010001u) & 0xFF0000FFu;
                  v = (v * 0x00000101u) & 0x0F00F00Fu;
                  v = (v * 0x00000011u) & 0xC30C30C3u;
                  v = (v * 0x00000005u) & 0x49249249u;
                  return v;
              };
              UInt i = dispatch_id().x;
              $if (i < count) {
                  auto prim = prims.read(prim_base + i);
                  auto center = (prim_lo(prim) + prim_hi(prim)) * 0.5f;
                  auto mapped = clamp((center - scene_min) * scene_inv_extent,
                                      make_float3(0.0f), make_float3(1.0f)) *
                                1024.0f;
                  auto x = cast<uint>(min(mapped.x, 1023.0f));
                  auto y = cast<uint>(min(mapped.y, 1023.0f));
                  auto z = cast<uint>(min(mapped.z, 1023.0f));
                  Var<LbvhKey> key;
                  key.code = expand_bits(x) * 4u + expand_bits(y) * 2u + expand_bits(z);
                  key.slot = i;
                  keys.write(prim_base + i, key);
              };
          }})},

      // Radix-tree construction, pass 1 of 4: the leaves.  Leaf `i` is the
      // primitive at sorted position `i`, so this pass is the *only* place that
      // still chases the random `prims[slot]` read; it writes the leaf AABB into
      // the node array, where pass 3 reads it back contiguously.
      _leaf_kernel{device.compile(Kernel1D{
          [](BufferVar<LbvhKey> keys, BufferVar<LbvhPrim> prims, BufferVar<LbvhNode> nodes,
             UInt prim_base, UInt node_base, UInt count) noexcept {
              set_block_size(sort_block_size);
              UInt i = dispatch_id().x;
              // leaves: [node_base + count - 1, node_base + 2 * count - 2]
              $if (i < count) {
                  auto slot = keys.read(prim_base + i).slot;
                  auto prim = prims.read(prim_base + slot);
                  Var<LbvhNode> leaf;
                  leaf.packed_lo = pack_node_plane(prim_lo(prim), invalid_node);
                  leaf.packed_hi = pack_node_plane(prim_hi(prim), prim_id(prim));
                  nodes.write(node_base + count - 1u + i, leaf);
              };
          }})},

      // Radix-tree construction, pass 2 of 4: the *structure* of the internal
      // nodes - their leaf range and their two child handles.
      //
      // One lane per internal node, not one warp.  `determine_range()` and
      // `find_split()` are chains of ~3 * log2(range) *dependent* key reads: every
      // step's address depends on the previous comparison, so the chain cannot be
      // overlapped with itself.  With a whole warp per node all `warp_lane_count()`
      // lanes walked the very same chain, i.e. the same answer was computed 32
      // times and the only latency-hiding available was other warps.  Measured on
      // a 1 M-primitive tree (cuda) the searches were 3.7 ms of the 4.6 ms node
      // stage - the dominant cost of the whole build - while the reduction they
      // were fused with was under 1 ms.  With one lane per node the lanes of a
      // warp walk 32 *independent* chains, so one warp keeps 32 loads in flight
      // instead of one and the pass needs no work-group cooperation at all (hence
      // a plain 1D dispatch).  The result is published as one `uint4` per node,
      // (first, last, child_a, child_b), so the reduction pass reads the whole
      // plan of a node back in a single 16-byte load.
      _plan_kernel{device.compile(Kernel1D{
          [](BufferVar<LbvhKey> keys, BufferVar<uint4> plans,
             UInt prim_base, UInt node_base, UInt plan_base, UInt count) noexcept {
              set_block_size(sort_block_size);
// -------- the two searches of Karras 2012, shared by the plan pass --------
              // VkLBVH's delta(): the number of leading bits shared by two keys
              // (i.e. clz of their xor).  Equal Morton codes are ordered by their
              // sorted slot, which keeps the keys strictly increasing and the
              // tree well formed; such pairs rank above every pair of distinct
              // codes, hence the +32.
              Callable delta = [](BufferVar<LbvhKey> keys, UInt base, UInt n,
                                  Int i, UInt code_i, Int j) noexcept {
                  Int result = def(-1);
                  $if (j >= 0 & j < cast<int>(n)) {
                      auto code_j = keys.read(base + cast<uint>(j)).code;
                      $if (code_i == code_j) {
                          result = 32 + cast<int>(clz(cast<uint>(i) ^ cast<uint>(j)));
                      }
                      $else {
                          result = cast<int>(clz(code_i ^ code_j));
                      };
                  };
                  return result;
              };
              // determineRange(): the leaf range covered by internal node `idx`.
              Callable determine_range = [&delta](BufferVar<LbvhKey> keys, UInt base,
                                                  UInt n, Int idx) noexcept {
                  auto code = keys.read(base + cast<uint>(idx)).code;
                  auto delta_left = delta(keys, base, n, idx, code, idx - 1);
                  auto delta_right = delta(keys, base, n, idx, code, idx + 1);
                  auto direction = select(-1, 1, delta_right >= delta_left);
                  auto delta_min = min(delta_left, delta_right);
                  auto l_max = def(2);
                  $while (delta(keys, base, n, idx, code, idx + l_max * direction) > delta_min &
                          l_max < cast<int>(n)) {
                      l_max = l_max * 2;
                  };
                  auto l = def(0);
                  auto stride = l_max / 2;
                  $while (stride > 0) {
                      $if (delta(keys, base, n, idx, code, idx + (l + stride) * direction) > delta_min) {
                          l = l + stride;
                      };
                      stride = stride / 2;
                  };
                  auto other = idx + l * direction;
                  return make_uint2(cast<uint>(min(idx, other)), cast<uint>(max(idx, other)));
              };
              // findSplit(): where a range is split between the two children.
              Callable find_split = [&delta](BufferVar<LbvhKey> keys, UInt base, UInt n,
                                             UInt first, UInt last) noexcept {
                  auto first_code = keys.read(base + first).code;
                  auto common_prefix = delta(keys, base, n, cast<int>(first), first_code, cast<int>(last));
                  auto split = def(cast<int>(first));
                  auto stride = cast<int>(last - first);
                  $loop {
                      stride = (stride + 1) / 2;
                      auto next = split + stride;
                      $if (next < cast<int>(last)) {
                          $if (delta(keys, base, n, cast<int>(first), first_code, next) > common_prefix) {
                              split = next;
                          };
                      };
                      $if (stride <= 1) { $break; };
                  };
                  return cast<uint>(split);
              };

              // internal nodes: [node_base, node_base + count - 2]
              UInt i = dispatch_id().x;
              $if (i + 1u < count) {
                  auto range = determine_range(keys, prim_base, count, cast<int>(i));
                  auto split = find_split(keys, prim_base, count, range.x, range.y);
                  // a child covering a single element is a leaf; child pointers
                  // are absolute indices into the shared node buffer, so every
                  // tree can reference its own range.
                  auto child_a = select(node_base + split,
                                        node_base + count - 1u + split,
                                        split == range.x);
                  auto child_b = select(node_base + split + 1u,
                                        node_base + count - 1u + split + 1u,
                                        split + 1u == range.y);
                  plans.write(plan_base + i,
                              make_uint4(range.x, range.y, child_a, child_b));
              };
          }})},

      // Radix-tree construction, pass 3 of 4: the block AABBs of the two-level
      // reduction (see `node_reduction_block` in lbvh_common.h).  Block `b` is
      // the AABB of the leaf slots `[b * node_reduction_block, (b + 1) * ...)`,
      // i.e. it is indexed by a slot of the shared node buffer, so this pass
      // reads every leaf of the tree exactly once and turns the
      // O(sum of leaf depths) traffic of the direct reduction into O(count) here
      // plus O(count / block_size) in the internal-node pass.  One warp per
      // block, the lanes walking the block together like the reduction does.
      //
      // The span is clamped to the tree's own leaves: the first and the last
      // block of the span may also hold leaves of another tree (or a run of the
      // previous tree's internal nodes), and only the part inside this tree's
      // span is this tree's.  Those partial blocks are never read back - a block
      // the internal-node pass reads is strictly inside a node's range, hence
      // entirely inside the tree's leaf span.
      _block_kernel{device.compile(Kernel1D{
          [](BufferVar<LbvhNode> nodes, BufferVar<LbvhNode> blocks,
             UInt leaf_begin, UInt leaf_end, UInt block_begin) noexcept {
              set_block_size(sort_block_size);
              auto lane_count = warp_lane_count();
              UInt lane = warp_lane_id();
              UInt block = block_begin +
                           (block_id().x * sort_block_size + thread_x()) / lane_count;
              // The lanes of the warp all hold the same `block`, so the whole
              // warp is active inside the guard and the lanes that run past the
              // block contribute the identity to the reduction.
              auto block_first = max(block * node_reduction_block, leaf_begin);
              auto block_last = min((block + 1u) * node_reduction_block, leaf_end);
              $if (block_first < block_last) {
                  auto lo = def(make_float3(1.0e30f));
                  auto hi = def(make_float3(-1.0e30f));
                  auto j = def(block_first + lane);
                  $while (j < block_last) {
                      auto leaf = nodes.read(j);
                      lo = min(lo, aabb_lo(leaf));
                      hi = max(hi, aabb_hi(leaf));
                      j = j + lane_count;
                  };
                  lo = warp_active_min(lo);
                  hi = warp_active_max(hi);
                  $if (lane == 0u) {
                      // The handle lanes of a block are unused; only the AABB
                      // planes are ever read back (lbvh_common.h).
                      Var<LbvhNode> bounds;
                      bounds.packed_lo = make_float4(lo, 0.0f);
                      bounds.packed_hi = make_float4(hi, 0.0f);
                      blocks.write(block, bounds);
                  };
              };
          }})},

      // Radix-tree construction, pass 4 of 4: the AABBs of the internal nodes,
      // one warp per node (the structure was computed by `_plan_kernel`).
      //
      // Internal node `i`'s AABB is the union of the leaf AABBs of its range
      // [range.x, range.y], and the leaf nodes of a tree are contiguous at
      // `node_base + count - 1`: the reduction therefore streams that array
      // instead of doing one random `prims` read per range slot.  It never
      // touches `keys` at all - `_plan_kernel` does, and it stays close to `i`.
      //
      // The range is covered by three disjoint pieces (see `node_reduction_block`
      // in lbvh_common.h): the leaves from `range.x` to the next block boundary,
      // the whole blocks strictly inside the range, and the leaves from the last
      // block boundary to `range.y`.  A node whose range stays inside one block
      // - which is the overwhelming majority, the mean range of a 1 M-primitive
      // tree is ~20 leaves - takes the first loop only and therefore costs
      // exactly what the direct reduction cost.  The union is bit-identical to
      // the direct reduction: min/max are exact, and every element of the range
      // is read exactly once.
      _build_kernel{device.compile(Kernel2D{
          [](BufferVar<LbvhNode> nodes, BufferVar<LbvhNode> blocks, BufferVar<uint4> plans,
             UInt node_base, UInt plan_base, UInt count, UInt row_stride) noexcept {
              set_block_size(sort_block_size, 1u);
              // One *warp* per internal node, the lanes cooperating on the
              // node's leaf range.  A warp costs `warp_lane_count()` threads per
              // node, which is a grid of `count * warp_size / sort_block_size`
              // work-groups - more than the 65535 per dimension DirectX 12 allows,
              // so the grid is 2D and the rows are laid out `row_stride` threads
              // apart.
              auto lane_count = warp_lane_count();
              UInt lane = warp_lane_id();
              UInt linear = block_id().y * row_stride +
                            block_id().x * sort_block_size + thread_id().x;
              UInt i = linear / lane_count;
              $if (i + 1u < count) {
                  auto plan = plans.read(plan_base + i);
                  auto child_a = plan.z;
                  auto child_b = plan.w;
                  // The leaf slots of the range in the *shared* node buffer: a
                  // block is indexed by such a slot, because the leaves of a tree
                  // live at their own global node index (see `node_reduction_block`).
                  auto leaf_base = node_base + count - 1u;
                  auto first = leaf_base + plan.x;
                  auto last = leaf_base + plan.y;
                  auto lo = def(make_float3(1.0e30f));
                  auto hi = def(make_float3(-1.0e30f));
                  auto first_block = first / node_reduction_block;
                  auto last_block = last / node_reduction_block;
                  // `left_end` is exclusive; `right_begin` is empty when the range
                  // stays inside one block (the first loop then covers everything).
                  auto left_end = min(last + 1u, (first_block + 1u) * node_reduction_block);
                  auto right_begin = max(left_end, last_block * node_reduction_block);
                  // The lanes walk each piece *together* - lane `l` takes
                  // `begin + l, + lane_count, ...` - so that one warp instruction
                  // reads `lane_count` consecutive records, which is one contiguous
                  // run the memory system can coalesce.  Lanes whose first index is
                  // past the end contribute the identity to the reduction.
                  // Two records per lane per iteration was measured too - it is the
                  // obvious way to add memory-level parallelism to this loop, but on
                  // all three backends it made the stage *slower* (4.32 -> 5.14 ms on
                  // `uniform`, 1 M primitives, cuda): the clamp that keeps the second
                  // load inside the range and the extra min/max per iteration cost
                  // more than the second outstanding load buys.
                  auto j = def(first + lane);
                  $while (j < left_end) {
                      auto leaf = nodes.read(j);
                      lo = min(lo, aabb_lo(leaf));
                      hi = max(hi, aabb_hi(leaf));
                      j = j + lane_count;
                  };
                  j = right_begin + lane;
                  $while (j <= last) {
                      auto leaf = nodes.read(j);
                      lo = min(lo, aabb_lo(leaf));
                      hi = max(hi, aabb_hi(leaf));
                      j = j + lane_count;
                  };
                  j = first_block + 1u + lane;
                  $while (j < last_block) {
                      auto block = blocks.read(j);
                      lo = min(lo, aabb_lo(block));
                      hi = max(hi, aabb_hi(block));
                      j = j + lane_count;
                  };
                  lo = warp_active_min(lo);
                  hi = warp_active_max(hi);
                  $if (lane == 0u) {
                      Var<LbvhNode> node;
                      node.packed_lo = pack_node_plane(lo, child_a);
                      node.packed_hi = pack_node_plane(hi, child_b);
                      nodes.write(node_base + i, node);
                  };
              };
          }})} {}         

LbvhStorage::TreeRange LbvhStorage::allocate(size_t prim_count) noexcept {
    LUISA_ASSERT(prim_count > 0u, "an LBVH needs at least one primitive.");
    LUISA_ASSERT(_prim_count + prim_count <= _sizes.primitive_capacity,
                 "software LBVH primitive capacity exceeded.");
    TreeRange range;
    range.prim_base = static_cast<uint>(_prim_count);
    range.node_base = static_cast<uint>(_node_count);
    range.plan_base = static_cast<uint>(_plan_count);
    range.count = static_cast<uint>(prim_count);
    LUISA_ASSERT(_plan_count + range.internal_count() <= _sizes.plan_capacity,
                 "software LBVH node-plan capacity exceeded.");
    _prim_count += prim_count;
    _node_count += range.node_count();
    _plan_count += range.internal_count();
    return range;
}

void LbvhStorage::upload_blas_table(Stream &stream, luisa::span<const LbvhBlas> table) noexcept {
    LUISA_ASSERT(table.size() <= _sizes.blas_capacity, "software LBVH BLAS capacity exceeded.");
    stream << _blas_table.view(0u, table.size()).copy_from(table);
}

void LbvhStorage::upload_instances(Stream &stream,
                                   luisa::span<const LbvhInstance> instances) noexcept {
    LUISA_ASSERT(instances.size() <= _sizes.instance_capacity,
                 "software LBVH instance capacity exceeded.");
    stream << _instances.view(0u, instances.size()).copy_from(instances);
}

void LbvhStorage::build_tree(Stream &stream, const TreeRange &range,
                             float3 lo, float3 hi, LbvhBuildTimings *timings) noexcept {
    // The range is built by the caller (the two builders copy it out of their
    // resource's accessors), so check it here instead of trusting it: a range
    // that lost a field would silently build a tree out of another tree's plan.
    LUISA_ASSERT(range.count > 0u &&
                     range.node_base + range.node_count() <= _node_count &&
                     range.prim_base + range.count <= _prim_count &&
                     range.plan_base + range.internal_count() <= _plan_count,
                 "build_tree() got a range that was not reserved by allocate().");
    auto extent = max(hi - lo, make_float3(1.0e-8f));
    auto inv_extent = make_float3(1.0f) / extent;
    // One code path for both callers: without `timings` the stages are simply
    // recorded back to back (no fence, exactly as before), with it every stage
    // is followed by a synchronisation so its time can be attributed.  Every
    // radix-tree pass after the leaves is *one* stage: each one reads only what
    // the previous one wrote, which the stream order guarantees without a fence,
    // so a single `node_ms` covers all four of them.
    Clock clock;
    if (timings != nullptr) { clock.tic(); }
    stream << _morton_kernel(_prims, _keys_a, range.prim_base, range.count, lo, inv_extent)
                  .dispatch(range.count);
    if (timings != nullptr) {
        stream << synchronize();
        timings->morton_ms += clock.toc();
        clock.tic();
    }
    // 4 x 8 bit LSD radix sort: keys_a -> keys_b -> ... -> keys_a.  `LbvhRadixSort`
    // owns this stage now (lbvh_sort.h): it is one work-group below
    // `automatic_min_count` elements - the old implementation, byte for byte -
    // and a block-parallel sort above it, and it leaves the sorted keys in
    // keys_a either way.
    _sort.sort(stream, _keys_a, _keys_b, range.prim_base, range.count);
    if (timings != nullptr) {
        stream << synchronize();
        timings->sort_ms += clock.toc();
        clock.tic();
    }
      // Radix tree: the leaves (one random `prims` read per leaf, then the leaf
      // node is written), the structure of the internal nodes (the searches of
      // Karras 2012, one lane per node), the block AABBs of the two-level
      // reduction (`node_reduction_block`), and the AABBs of the internal nodes
      // (one warp per node, the lanes splitting the node's leaf range).
      stream << _leaf_kernel(_keys_a, _prims, _nodes, range.prim_base, range.node_base, range.count)
                    .dispatch(range.count);
      if (range.count > 1u) {
          // Pass 2: the structure of the internal nodes, one *lane* per node.
          // The searches are chains of dependent key reads, so a whole warp per
          // node would compute the same chain `warp_lane_count()` times over;
          // with one lane per node a warp instead keeps 32 independent chains - 32
          // loads - in flight (see `_plan_kernel`).  The plan only needs the
          // sorted keys, so it is recorded here purely to keep the four passes of
          // the tree in reading order.
          stream << _plan_kernel(_keys_a, _plan, range.prim_base, range.node_base,
                                 range.plan_base, range.count)
                        .dispatch(range.count);
          // Pass 3: the block AABBs the reduction of pass 4 reads back.  The tree's
          // leaves occupy the *global* node slots [leaf_begin, leaf_end), so the
          // blocks a node's range can reference are the blocks of that span (a
          // block the reduction reads is strictly inside a range, hence entirely
          // inside the span - the two partial blocks at the ends of the span are
          // never read).  A tree that stays inside one block needs no block at all:
          // every one of its ranges lies inside block 0, and the reduction then
          // takes the same single-loop path it took before this pass existed.
          auto leaf_begin = range.node_base + range.count - 1u;
          auto leaf_end = leaf_begin + range.count;
          auto block_begin = leaf_begin / node_reduction_block;
          auto block_end = (leaf_end - 1u) / node_reduction_block;
          if (block_end > block_begin) {
              auto block_count = static_cast<size_t>(block_end - block_begin) + 1u;
              auto threads = block_count * _warp_size;
              auto groups = (threads + sort_block_size - 1u) / sort_block_size;
              stream << _block_kernel(_nodes, _blocks, leaf_begin, leaf_end, block_begin)
                            .dispatch(static_cast<uint>(groups * sort_block_size));
          }
          // Pass 4: the AABBs of the internal nodes, one warp per node, i.e.
          // `count * warp_size` threads, laid out over a 2D grid whose every
          // dimension stays inside the 65535 work-groups DirectX 12 allows per
          // dimension (`rows` of `row_stride` threads each).
        auto threads = static_cast<size_t>(range.count) * _warp_size;
        auto groups = (threads + sort_block_size - 1u) / sort_block_size;
        auto groups_x = std::min<size_t>(groups, max_build_dispatch_groups);
        auto rows = (groups + groups_x - 1u) / groups_x;
        auto row_stride = static_cast<uint>(groups_x * sort_block_size);
        stream << _build_kernel(_nodes, _blocks, _plan, range.node_base, range.plan_base,
                                range.count, row_stride)
                      .dispatch(groups_x * sort_block_size, static_cast<uint>(rows));
    }
    if (timings != nullptr) {
        stream << synchronize();
        timings->node_ms += clock.toc();
    }
}

size_t LbvhStorage::validate_tree(Stream &stream, uint node_base, uint count) noexcept {
    LUISA_ASSERT(count > 0u, "validate_tree() needs at least one primitive.");
    auto node_count = count * 2u - 1u;
    luisa::vector<LbvhNode> nodes(node_count);
    stream << _nodes.view(node_base, node_count).copy_to(luisa::span{nodes})
           << synchronize();
    // Both handles of a node must address a node *of this tree*; an inconsistent
    // (node_base, count) - a caller's bug, not a malformed tree - must produce a
    // problem report instead of reading past the vector.
    auto addressable = [&](uint child) noexcept {
        return child >= node_base && child < node_base + node_count;
    };
    luisa::vector<uint> visits(node_count, 0u);
    luisa::vector<uint> stack{node_base};// the root
    size_t problems = 0u;
    size_t leaves = 0u;
    while (!stack.empty()) {
        auto index = stack.back();
        stack.pop_back();
        if (!addressable(index)) {
            problems++;// dangling child pointer
            continue;
        }
        auto slot = index - node_base;
        if (visits[slot]++ != 0u) {
            problems++;// node reached twice (cycle or shared child)
            continue;
        }
        auto node = nodes[slot];
        auto left = host_child_left(node);
        if (left == invalid_node) {
            leaves++;
            // the second handle of a leaf carries the primitive id, not a child
            if (host_child_right(node) >= count) { problems++; }
            continue;
        }
        auto right = host_child_right(node);
        if (!addressable(left) || !addressable(right)) {
            problems++;// dangling child pointer
            continue;
        }
        auto a = nodes[left - node_base];
        auto b = nodes[right - node_base];
        auto lo = min(host_aabb_lo(a), host_aabb_lo(b));
        auto hi = max(host_aabb_hi(a), host_aabb_hi(b));
        auto node_lo = host_aabb_lo(node);
        auto node_hi = host_aabb_hi(node);
        auto dlo = abs(lo - node_lo);
        auto dhi = abs(hi - node_hi);
        auto error = std::max(std::max(dlo.x, dlo.y),
                              std::max(dlo.z, std::max(dhi.x, std::max(dhi.y, dhi.z))));
        auto scale = std::max(1.0f, std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z)));
        if (error > 1.0e-4f * scale) { problems++; }
        stack.push_back(left);
        stack.push_back(right);
    }
    for (auto v : visits) {
        if (v != 1u) { problems++; }
    }
    if (leaves != count) { problems++; }
    return problems;
}

}// namespace luisa::example::lbvh
