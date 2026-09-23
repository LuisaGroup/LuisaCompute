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
    sizes.blas_table_bytes = sizes.blas_capacity * sizeof(LbvhBlas);
    sizes.instance_bytes = sizes.instance_capacity * sizeof(LbvhInstance);
    return sizes;
}

LbvhStorage::LbvhStorage(Device &device, const Sizes &sizes) noexcept
    : _sizes{sizes},
      _prims{device.create_buffer<LbvhPrim>(_sizes.primitive_capacity)},
      _keys_a{device.create_buffer<LbvhKey>(_sizes.primitive_capacity)},
      _keys_b{device.create_buffer<LbvhKey>(_sizes.primitive_capacity)},
      _nodes{device.create_buffer<LbvhNode>(_sizes.node_capacity)},
      _blas_table{device.create_buffer<LbvhBlas>(_sizes.blas_capacity)},
      _instances{device.create_buffer<LbvhInstance>(_sizes.instance_capacity)},
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
                  auto center = (prim.lo + prim.hi) * 0.5f;
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

      // One LSD radix-sort pass over 8 bits.  The source/destination buffers are
      // shader arguments, so a single shader can ping-pong.
      _sort_kernel{device.compile(Kernel1D{
          [](BufferVar<LbvhKey> keys_in, BufferVar<LbvhKey> keys_out,
             UInt base, UInt count, UInt shift) noexcept {
              constexpr uint words = sort_block_size / 32u;
              set_block_size(sort_block_size);
              Shared<uint> histogram{sort_radix_bins};
              Shared<uint> offsets{sort_radix_bins};
              Shared<uint> flags{sort_radix_bins * words};
              UInt tid = thread_x();
              UInt word = tid / 32u;
              UInt bit = 1u << (tid % 32u);
              // per-bin histogram
              histogram[tid] = 0u;
              sync_block();
              UInt i = tid;
              $while (i < count) {
                  auto code = keys_in.read(base + i).code;
                  histogram.atomic((code >> shift) & (sort_radix_bins - 1u)).fetch_add(1u);
                  i = i + sort_block_size;
              };
              sync_block();
              // exclusive scan of the 256 bins (one thread is plenty here)
              $if (tid == 0u) {
                  UInt sum = def(0u);
                  $for (b, sort_radix_bins) {
                      auto c = histogram[b];
                      offsets[b] = sum;
                      sum = sum + c;
                  };
              };
              sync_block();
              // stable scatter, chunk by chunk: every lane only accumulates the
              // lanes before it inside its own chunk, and the shared cursor is
              // advanced once per bin and chunk.  Stability is what makes the
              // four LSD passes add up to a full sort.
              UInt chunks = (count + sort_block_size - 1u) / sort_block_size;
              $for (chunk, chunks) {
                  $for (k, words) { flags[tid * words + k] = 0u; };
                  sync_block();
                  UInt index = chunk * sort_block_size + tid;
                  Bool valid = index < count;
                  UInt bin = def(0u);
                  UInt bin_offset = def(0u);
                  UInt code = def(0u);
                  UInt slot = def(0u);
                  $if (valid) {
                      auto key = keys_in.read(base + index);
                      code = key.code;
                      slot = key.slot;
                      bin = (code >> shift) & (sort_radix_bins - 1u);
                      bin_offset = offsets[bin];
                      flags.atomic(bin * words + word).fetch_add(bit);
                  };
                  sync_block();
                  $if (valid) {
                      UInt prefix = def(0u);
                      UInt total = def(0u);
                      $for (w, words) {
                          auto bits = flags[bin * words + w];
                          auto bin_count = popcount(bits);
                          total = total + bin_count;
                          prefix = prefix + select(0u, bin_count, w < word);
                          prefix = prefix + select(0u, popcount(bits & (bit - 1u)), w == word);
                      };
                      Var<LbvhKey> sorted;
                      sorted.code = code;
                      sorted.slot = slot;
                      keys_out.write(base + bin_offset + prefix, sorted);
                      // the last element of a bin advances the shared cursor
                      $if (prefix == total - 1u) {
                          offsets.atomic(bin).fetch_add(total);
                      };
                  };
                  sync_block();
              };
          }})},

      // Radix-tree construction, pass 1 of 2: the leaves.  Leaf `i` is the
      // primitive at sorted position `i`, so this pass is the *only* place that
      // still chases the random `prims[slot]` read; it writes the leaf AABB into
      // the node array, where pass 2 reads it back contiguously.
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
                  leaf.lo = prim.lo;
                  leaf.hi = prim.hi;
                  leaf.left = invalid_node;
                  leaf.right = invalid_node;
                  leaf.prim = prim.id;
                  nodes.write(node_base + count - 1u + i, leaf);
              };
          }})},

      // Radix-tree construction, pass 2 of 2: the internal nodes.
      //
      // Internal node `i`'s AABB is the union of the leaf AABBs of its range
      // [range.x, range.y], and the leaf nodes of a tree are contiguous at
      // `node_base + count - 1`: the reduction therefore streams that array
      // instead of doing one random `keys` read plus one random `prims` read per
      // range slot.  The reduction iterates sum(leaf depth) times, which is what
      // makes it the hottest loop of the whole build, and it no longer touches
      // `keys` at all - the two searches do, and they stay close to `i`.
      _build_kernel{device.compile(Kernel2D{
          [](BufferVar<LbvhKey> keys, BufferVar<LbvhNode> nodes,
             UInt prim_base, UInt node_base, UInt count, UInt row_stride) noexcept {
              set_block_size(sort_block_size, 1u);
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

              // One *warp* per internal node, the lanes cooperating on the
              // node's leaf range (see the reduction below).  A warp costs
              // `warp_lane_count()` threads per node, which is a grid of
              // `count * warp_size / sort_block_size` work-groups - more than the
              // 65535 per dimension DirectX 12 allows, so the grid is 2D and the
              // rows are laid out `row_stride` threads apart.
              auto lane_count = warp_lane_count();
              UInt lane = warp_lane_id();
              UInt linear = block_id().y * row_stride +
                            block_id().x * sort_block_size + thread_id().x;
              UInt i = linear / lane_count;
              // internal nodes: [node_base, node_base + count - 2]
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
                  // The AABB of the node: the union of the leaf AABBs of
                  // [range.x, range.y], which pass 1 left as a contiguous run
                  // of nodes at `leaf_base`.  The number of reduction steps is
                  // sum over leaves of their depth (~count * mean_depth), so
                  // this loop is the hottest part of the build - but the range
                  // is extremely unbalanced: the root reduces the whole array
                  // while a node at the bottom reduces two elements.  One thread
                  // per node therefore leaves the few top nodes (which own most
                  // of the work) running alone, which measured as ~95% of the
                  // whole build.  Giving each node a whole warp instead splits
                  // its range over `warp_lane_count()` lanes; the narrow nodes
                  // at the bottom (the vast majority) end up reading a
                  // contiguous run of nodes per warp instruction.
                  auto leaf_base = node_base + count - 1u;
                  auto length = range.y - range.x + 1u;
                  // elements per lane; lanes whose slice is empty contribute the
                  // identity and are dropped by the warp reduction
                  auto chunk = (length + lane_count - 1u) / lane_count;
                  auto begin = range.x + lane * chunk;
                  auto end = min(begin + chunk, range.y + 1u);
                  auto lo = def(make_float3(1.0e30f));
                  auto hi = def(make_float3(-1.0e30f));
                  $for (j, begin, end) {
                      auto leaf = nodes.read(leaf_base + j);
                      lo = min(lo, leaf.lo);
                      hi = max(hi, leaf.hi);
                  };
                  lo = warp_active_min(lo);
                  hi = warp_active_max(hi);
                  $if (lane == 0u) {
                      Var<LbvhNode> node;
                      node.lo = lo;
                      node.hi = hi;
                      node.left = child_a;
                      node.right = child_b;
                      node.prim = 0u;
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
    range.count = static_cast<uint>(prim_count);
    _prim_count += prim_count;
    _node_count += range.node_count();
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
    auto extent = max(hi - lo, make_float3(1.0e-8f));
    auto inv_extent = make_float3(1.0f) / extent;
    // One code path for both callers: without `timings` the stages are simply
    // recorded back to back (no fence, exactly as before), with it every stage
    // is followed by a synchronisation so its time can be attributed.  The two
    // radix-tree passes are *one* stage: the second reads the leaf nodes the
    // first wrote, which the stream order guarantees without a fence, so a
    // single `node_ms` covers both (as it covered the single fused pass).
    Clock clock;
    if (timings != nullptr) { clock.tic(); }
    stream << _morton_kernel(_prims, _keys_a, range.prim_base, range.count, lo, inv_extent)
                  .dispatch(range.count);
    if (timings != nullptr) {
        stream << synchronize();
        timings->morton_ms += clock.toc();
        clock.tic();
    }
    // 4 x 8 bit LSD radix sort: keys_a -> keys_b -> ... -> keys_a
    stream << _sort_kernel(_keys_a, _keys_b, range.prim_base, range.count, 0u).dispatch(sort_block_size)
           << _sort_kernel(_keys_b, _keys_a, range.prim_base, range.count, 8u).dispatch(sort_block_size)
           << _sort_kernel(_keys_a, _keys_b, range.prim_base, range.count, 16u).dispatch(sort_block_size)
           << _sort_kernel(_keys_b, _keys_a, range.prim_base, range.count, 24u).dispatch(sort_block_size);
    if (timings != nullptr) {
        stream << synchronize();
        timings->sort_ms += clock.toc();
        clock.tic();
    }
    // Radix tree: the leaves first (one random `prims` read per leaf, then the
    // leaf node is written), then the internal nodes (one warp per node, the
    // lanes splitting the node's leaf range).
    stream << _leaf_kernel(_keys_a, _prims, _nodes, range.prim_base, range.node_base, range.count)
                  .dispatch(range.count);
    if (range.count > 1u) {
        // One warp per internal node, i.e. `count * warp_size` threads, laid out
        // over a 2D grid whose every dimension stays inside the 65535 work-groups
        // DirectX 12 allows per dimension (`rows` of `row_stride` threads each).
        auto threads = static_cast<size_t>(range.count) * _warp_size;
        auto groups = (threads + sort_block_size - 1u) / sort_block_size;
        auto groups_x = std::min<size_t>(groups, max_build_dispatch_groups);
        auto rows = (groups + groups_x - 1u) / groups_x;
        auto row_stride = static_cast<uint>(groups_x * sort_block_size);
        stream << _build_kernel(_keys_a, _nodes, range.prim_base, range.node_base,
                                range.count, row_stride)
                      .dispatch(groups_x * sort_block_size, static_cast<uint>(rows));
    }
    if (timings != nullptr) {
        stream << synchronize();
        timings->node_ms += clock.toc();
    }
}

size_t LbvhStorage::validate_tree(Stream &stream, uint node_base, uint count) noexcept {
    auto node_count = count * 2u - 1u;
    luisa::vector<LbvhNode> nodes(node_count);
    stream << _nodes.view(node_base, node_count).copy_to(luisa::span{nodes})
           << synchronize();
    luisa::vector<uint> visits(node_count, 0u);
    luisa::vector<uint> stack{node_base};// the root
    size_t problems = 0u;
    size_t leaves = 0u;
    while (!stack.empty()) {
        auto index = stack.back();
        stack.pop_back();
        if (index < node_base || index >= node_base + node_count) {
            problems++;// dangling child pointer
            continue;
        }
        index -= node_base;
        if (visits[index]++ != 0u) {
            problems++;// node reached twice (cycle or shared child)
            continue;
        }
        auto node = nodes[index];
        if (node.left == invalid_node) {
            leaves++;
            if (node.right != invalid_node || node.prim >= count) { problems++; }
            continue;
        }
        if (node.right == invalid_node) { problems++; }
        auto a = nodes[node.left - node_base];
        auto b = nodes[node.right - node_base];
        auto lo = min(a.lo, b.lo);
        auto hi = max(a.hi, b.hi);
        auto dlo = abs(lo - node.lo);
        auto dhi = abs(hi - node.hi);
        auto error = std::max(std::max(dlo.x, dlo.y),
                              std::max(dlo.z, std::max(dhi.x, std::max(dhi.y, dhi.z))));
        auto scale = std::max(1.0f, std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z)));
        if (error > 1.0e-4f * scale) { problems++; }
        stack.push_back(node.left);
        stack.push_back(node.right);
    }
    for (auto v : visits) {
        if (v != 1u) { problems++; }
    }
    if (leaves != count) { problems++; }
    return problems;
}

}// namespace luisa::example::lbvh
