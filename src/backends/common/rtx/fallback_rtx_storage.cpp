// Implementation of the shared storage: the region planner, the grow-on-demand
// buffers and the tree-agnostic build stages (Morton codes, LSD radix sort,
// Karras radix tree), plus the host-side validator the fallback's debug entry
// points use.
//
// Ported from examples/compute/lbvh/lbvh_storage.cpp.  Three things differ, and
// each one is called out where it happens:
//
//   * the AABB planes and the handles live in the two uint4 of a node, so the
//     packing is integer (`pack_node_plane`) and a handle is an absolute uint4
//     offset rather than a node index;
//   * the scene volume of a tree is not a host uniform: the host never sees the
//     geometry, so the build reduces it on the device (`_volume_kernel`) and the
//     Morton kernel reads it from the reduction slot;
//   * every allocation goes through `detail::GrowableBuffer`, i.e. it appends.

#include "fallback_rtx_storage.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>

#include <algorithm>
#include <bit>
#include <cstdint>

namespace lc::fallback_rtx {

namespace {

// Initial capacities of the shared buffers.  The API knows no scene size up
// front (a BLAS gets its geometry at build time and a TLAS its instances then
// too), so the storage starts small and doubles on demand; these numbers only
// decide when the first few growths happen.
constexpr size_t kInitialAccelU4 = 1024u;   // 16 KiB: a few small meshes
constexpr size_t kInitialInstanceU4 = 1024u;// 128 instances
constexpr size_t kInitialPrims = 1024u;     // triangles + instances
constexpr size_t kInitialReductionSlots = 8u;
constexpr size_t kInitialDirectoryEntries = 16u;

// The block size of the tiny one-thread kernels (the header, the reduction
// identity, the volume, one directory record): a full warp is dispatched and
// lane 0 does the work, which keeps every backend on a well-formed group.
constexpr uint kSingleThreadBlock = 32u;

}// namespace

FallbackRtxStorage::FallbackRtxStorage(Device &device) noexcept
    : _accel{device, kInitialAccelU4},
      _instances{device, kInitialInstanceU4},
      _prims{device, kInitialPrims},
      _keys_a{device, kInitialPrims},
      _keys_b{device, kInitialPrims},
      _reduce{device, kInitialReductionSlots * reduce_slot_uints},
      _blas_directory{device, kInitialDirectoryEntries * blas_record_u4},
      _sort{device, kInitialPrims},
      _device{&device},
      _warp_size{device.compute_warp_size()},

      // The identity of the min/max reduction of one tree, written before the
      // primitive kernel of that tree accumulates into it.
      _reset_kernel{device.compile(Kernel1D{
          [](BufferVar<uint> reduce, UInt offset) noexcept {
              set_block_size(kSingleThreadBlock);
              $if (thread_x() == 0u) {
                  $for (i, 3u) { reduce.write(offset + i, 0xffffffffu); };
                  $for (i, reduce_slot_uints - 3u) { reduce.write(offset + 3u + i, 0u); };
              };
          }})},

      // 30-bit Morton code of every primitive, normalized with the *device*
      // volume of this tree (a BLAS' geometry and a TLAS' world AABB are both
      // only known on the device).
      _morton_kernel{device.compile(Kernel1D{
          [](BufferVar<FallbackRtxPrim> prims, BufferVar<FallbackRtxKey> keys,
             BufferVar<uint> reduce, UInt prim_base, UInt count,
             UInt reduce_offset) noexcept {
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
                  auto scene_min = make_float3(
                      reduce.read(reduce_offset + reduce_volume_lo + 0u).bitcast<float>(),
                      reduce.read(reduce_offset + reduce_volume_lo + 1u).bitcast<float>(),
                      reduce.read(reduce_offset + reduce_volume_lo + 2u).bitcast<float>());
                  auto scene_inv_extent = make_float3(
                      reduce.read(reduce_offset + reduce_volume_inv + 0u).bitcast<float>(),
                      reduce.read(reduce_offset + reduce_volume_inv + 1u).bitcast<float>(),
                      reduce.read(reduce_offset + reduce_volume_inv + 2u).bitcast<float>());
                  auto prim = prims.read(prim_base + i);
                  auto center = (prim.lo + prim.hi) * 0.5f;
                  auto mapped = clamp((center - scene_min) * scene_inv_extent,
                                      make_float3(0.0f), make_float3(1.0f)) *
                                1024.0f;
                  auto x = cast<uint>(min(mapped.x, 1023.0f));
                  auto y = cast<uint>(min(mapped.y, 1023.0f));
                  auto z = cast<uint>(min(mapped.z, 1023.0f));
                  Var<FallbackRtxKey> key;
                  key.code = expand_bits(x) * 4u + expand_bits(y) * 2u + expand_bits(z);
                  key.slot = i;
                  keys.write(prim_base + i, key);
              };
          }})},

      // The scene bounds of one tree: decode the reduced keys and turn them into
      // the (min, reciprocal extent) pair the Morton kernel maps the unit cube
      // with.  A degenerate (single-point) tree gets a minimum extent so the
      // reciprocal stays finite.
      _volume_kernel{device.compile(Kernel1D{
          [](BufferVar<uint> reduce, UInt offset) noexcept {
              set_block_size(kSingleThreadBlock);
              $if (thread_x() == 0u) {
                  auto lo = make_float3(
                      unorderable_key(reduce.read(offset + reduce_min_key + 0u)),
                      unorderable_key(reduce.read(offset + reduce_min_key + 1u)),
                      unorderable_key(reduce.read(offset + reduce_min_key + 2u)));
                  auto hi = make_float3(
                      unorderable_key(reduce.read(offset + reduce_max_key + 0u)),
                      unorderable_key(reduce.read(offset + reduce_max_key + 1u)),
                      unorderable_key(reduce.read(offset + reduce_max_key + 2u)));
                  auto inv_extent = 1.0f / max(hi - lo, make_float3(1.0e-8f));
                  reduce.write(offset + reduce_volume_lo + 0u, lo.x.bitcast<uint>());
                  reduce.write(offset + reduce_volume_lo + 1u, lo.y.bitcast<uint>());
                  reduce.write(offset + reduce_volume_lo + 2u, lo.z.bitcast<uint>());
                  reduce.write(offset + reduce_volume_inv + 0u, inv_extent.x.bitcast<uint>());
                  reduce.write(offset + reduce_volume_inv + 1u, inv_extent.y.bitcast<uint>());
                  reduce.write(offset + reduce_volume_inv + 2u, inv_extent.z.bitcast<uint>());
              };
          }})},

      // Radix-tree construction, pass 1 of 2: the leaves.  Leaf `i` is the
      // primitive at sorted position `i`, so this pass is the *only* place that
      // still chases the random `prims[slot]` read; it writes the leaf AABB into
      // the node array, where pass 2 reads it back contiguously.
      _leaf_kernel{device.compile(Kernel1D{
          [](BufferVar<FallbackRtxKey> keys, BufferVar<FallbackRtxPrim> prims,
             BufferVar<uint4> accel, UInt prim_base, UInt node_base, UInt count) noexcept {
              set_block_size(sort_block_size);
              UInt i = dispatch_id().x;
              // leaves: [node_base + count - 1, node_base + 2 * count - 2]
              $if (i < count) {
                  auto slot = keys.read(prim_base + i).slot;
                  auto prim = prims.read(prim_base + slot);
                  // A leaf is the node whose left handle is `invalid_offset`; its
                  // right handle is the primitive id (the local triangle index of
                  // a BLAS, the instance index of a TLAS).
                  auto node = node_base + (count - 1u + i) * node_u4;
                  accel.write(node, pack_node_plane(prim.lo, invalid_offset));
                  accel.write(node + 1u, pack_node_plane(prim.hi, prim.id));
              };
          }})},

      // Radix-tree construction, pass 2 of 2: the internal nodes.
      //
      // Internal node `i`'s AABB is the union of the leaf AABBs of its range
      // [range.x, range.y], and the leaf nodes of a tree are contiguous at
      // `node_base + count - 1`: the reduction therefore streams that array
      // instead of doing one random `keys` read plus one random `prims` read per
      // range slot, and it no longer touches `keys` at all - the two searches do,
      // and they stay close to `i`.
      _build_kernel{device.compile(Kernel2D{
          [](BufferVar<FallbackRtxKey> keys, BufferVar<uint4> accel, UInt prim_base,
             UInt node_base, UInt count, UInt row_stride) noexcept {
              set_block_size(sort_block_size, 1u);
              // VkLBVH's delta(): the number of leading bits shared by two keys
              // (i.e. clz of their xor).  Equal Morton codes are ordered by their
              // sorted slot, which keeps the keys strictly increasing and the
              // tree well formed; such pairs rank above every pair of distinct
              // codes, hence the +32.
              Callable delta = [](BufferVar<FallbackRtxKey> keys, UInt base, UInt n,
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
              Callable determine_range = [&delta](BufferVar<FallbackRtxKey> keys, UInt base,
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
              Callable find_split = [&delta](BufferVar<FallbackRtxKey> keys, UInt base, UInt n,
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
              // `warp_lane_count()` threads per node, so the grid is a 2D fold of
              // `count * warp_size / sort_block_size` groups at the 65535 groups
              // per dimension DirectX 12 allows.
              auto lane_count = warp_lane_count();
              UInt lane = warp_lane_id();
              UInt linear = block_id().y * row_stride +
                            block_id().x * sort_block_size + thread_id().x;
              UInt i = linear / lane_count;
              // internal nodes: [node_base, node_base + 2 * (count - 2)]
              $if (i + 1u < count) {
                  auto range = determine_range(keys, prim_base, count, cast<int>(i));
                  auto split = find_split(keys, prim_base, count, range.x, range.y);
                  // A child covering a single element is a leaf; both handles are
                  // the *absolute* uint4 offset of the child node, which is what
                  // lets a traversal walk a region with a descriptor that starts
                  // at the region (fallback_rtx_layout.h).
                  auto leaf_base = node_base + (count - 1u) * node_u4;
                  auto child_a = select(node_base + split * node_u4,
                                        leaf_base + split * node_u4, split == range.x);
                  auto child_b = select(node_base + (split + 1u) * node_u4,
                                        leaf_base + (split + 1u) * node_u4,
                                        split + 1u == range.y);
                  // The AABB of the node: the union of the leaf AABBs of
                  // [range.x, range.y], which pass 1 left as a contiguous run of
                  // nodes at `leaf_base`.  The number of reduction steps is
                  // sum over leaves of their depth (~count * mean_depth), and the
                  // range is extremely unbalanced: one thread per node would leave
                  // the few top nodes - which own most of the work - running
                  // alone, so every node gets a whole warp.
                  //
                  // The lanes walk the range *together* - lane `l` takes
                  // `range.x + l, + lane_count, ...` - instead of each lane owning a
                  // contiguous slice: a per-lane slice would put the lanes of one
                  // warp instruction `length / lane_count` records apart, i.e. 32
                  // unrelated cache lines per instruction, while the strided walk
                  // reads `lane_count` consecutive records, which is one contiguous
                  // run the memory system can coalesce.  Lanes whose first index is
                  // past the end contribute the identity and are dropped by the
                  // reduction.
                  auto lo = def(make_float3(1.0e30f));
                  auto hi = def(make_float3(-1.0e30f));
                  auto j = def(range.x + lane);
                  $while (j <= range.y) {
                      auto plane_lo = accel.read(leaf_base + j * node_u4);
                      auto plane_hi = accel.read(leaf_base + j * node_u4 + 1u);
                      lo = min(lo, node_aabb_lo(plane_lo));
                      hi = max(hi, node_aabb_hi(plane_hi));
                      j = j + lane_count;
                  };
                  lo = warp_active_min(lo);
                  hi = warp_active_max(hi);
                  $if (lane == 0u) {
                      auto node = node_base + i * node_u4;
                      accel.write(node, pack_node_plane(lo, child_a));
                      accel.write(node + 1u, pack_node_plane(hi, child_b));
                  };
              };
          }})},

      // The region header.  The planner knows every field when the region is
      // reserved, so this is one lane copying three uint4 of host scalars; the
      // reserved fourth uint4 is zeroed so a validator cannot mistake garbage
      // for a field a later version of the ABI might add.
      _header_kernel{device.compile(Kernel1D{
          [](BufferVar<uint4> accel, UInt4 h0, UInt4 h1, UInt4 h2) noexcept {
              set_block_size(kSingleThreadBlock);
              $if (thread_x() == 0u) {
                  auto base = h0.x;
                  accel.write(base + 0u, h0);
                  accel.write(base + 1u, h1);
                  accel.write(base + 2u, h2);
                  accel.write(base + 3u, make_uint4(0u, 0u, 0u, 0u));
              };
          }})},

      // One blas-table record of the shared directory, from the planner's own
      // numbers (see `append_blas_directory`).
      _directory_kernel{device.compile(Kernel1D{
          [](BufferVar<uint4> directory, UInt4 r0, UInt4 r1, UInt entry) noexcept {
              set_block_size(kSingleThreadBlock);
              $if (thread_x() == 0u) {
                  directory.write(entry + 0u, r0);
                  directory.write(entry + 1u, r1);
              };
          }})} {}

// ---------------------------------------------------------------------------
// Region planner
// ---------------------------------------------------------------------------

FallbackRtxRegion FallbackRtxStorage::plan_blas(CommandList &commands,
                                                size_t triangle_count,
                                                size_t vertex_count) noexcept {
    LUISA_ASSERT(triangle_count > 0u, "a fallback BLAS needs at least one triangle.");
    FallbackRtxRegion region;
    region.prim_count = static_cast<uint>(triangle_count);
    region.node_count = static_cast<uint>(triangle_count * 2u - 1u);
    region.index_count = region.prim_count;
    region.vertex_count = static_cast<uint>(vertex_count);
    region.base = static_cast<uint>(_accel_used);
    region.node_base = region.base + header_u4;
    region.index_base = region.node_base + node_u4 * region.node_count;
    region.vertex_base = region.index_base + region.index_count;
    region.prim_offset = static_cast<uint>(_prim_used);
    region.reduce_offset = static_cast<uint>(_reduce_used);
    _accel_used += region.region_u4();
    _prim_used += triangle_count;
    _reduce_used += reduce_slot_uints;
    // The regions are handed out in call order and never moved, so the planner
    // is deterministic: the same sequence of builds produces the same offsets.
    _accel.reserve(_accel_used, commands);
    _prims.reserve(_prim_used, commands);
    _keys_a.reserve(_prim_used, commands);
    _keys_b.reserve(_prim_used, commands);
    _reduce.reserve(_reduce_used, commands);
    _sort.reserve(_prim_used);
    return region;
}

FallbackRtxRegion FallbackRtxStorage::plan_tlas(CommandList &commands,
                                                size_t instance_count) noexcept {
    LUISA_ASSERT(instance_count > 0u, "a fallback TLAS needs at least one instance.");
    FallbackRtxRegion region;
    region.prim_count = static_cast<uint>(instance_count);
    region.node_count = static_cast<uint>(instance_count * 2u - 1u);
    region.blas_count = region.prim_count;
    region.base = static_cast<uint>(_accel_used);
    region.node_base = region.base + header_u4;
    region.blas_table_base = region.node_base + node_u4 * region.node_count;
    region.prim_offset = static_cast<uint>(_prim_used);
    region.reduce_offset = static_cast<uint>(_reduce_used);
    region.instance_offset = static_cast<uint>(_instance_used);
    _accel_used += region.region_u4();
    _prim_used += instance_count;
    _reduce_used += reduce_slot_uints;
    _instance_used += instance_count * instance_u4;
    _accel.reserve(_accel_used, commands);
    _instances.reserve(_instance_used, commands);
    _prims.reserve(_prim_used, commands);
    _keys_a.reserve(_prim_used, commands);
    _keys_b.reserve(_prim_used, commands);
    _reduce.reserve(_reduce_used, commands);
    _sort.reserve(_prim_used);
    return region;
}

uint FallbackRtxStorage::append_blas_directory(CommandList &commands, uint4 record_0,
                                               uint4 record_1) noexcept {
    auto entry = static_cast<uint>(_blas_directory_used);
    _blas_directory_used++;
    _blas_directory.reserve(_blas_directory_used * blas_record_u4, commands);
    commands << _directory_kernel(_blas_directory.buffer(), record_0, record_1,
                                  entry * blas_record_u4)
                    .dispatch(kSingleThreadBlock);
    return entry;
}

void FallbackRtxStorage::write_region_header(CommandList &commands,
                                             const FallbackRtxRegion &region,
                                             uint flags) noexcept {
    auto h0 = make_uint4(region.base, region.node_base, region.node_count, region.prim_count);
    auto h1 = make_uint4(region.blas_table_base, region.blas_count,
                         region.index_base, region.index_count);
    // The root of a tree is its node 0: the header carries it explicitly so a
    // traversal starts from a field instead of from a layout assumption.
    auto h2 = make_uint4(region.vertex_base, region.vertex_count, region.node_base, flags);
    commands << _header_kernel(_accel.buffer(), h0, h1, h2).dispatch(kSingleThreadBlock);
}

void FallbackRtxStorage::reset_reduction(CommandList &commands,
                                         const FallbackRtxRegion &region) noexcept {
    commands << _reset_kernel(_reduce.buffer(), region.reduce_offset)
                    .dispatch(kSingleThreadBlock);
}

// ---------------------------------------------------------------------------
// Build stages
// ---------------------------------------------------------------------------

void FallbackRtxStorage::build_tree(CommandList &commands,
                                    const FallbackRtxRegion &region) noexcept {
    // The scene volume first: the primitive kernel of the caller has filled the
    // reduction slot, and the Morton codes need the unit-cube mapping of it.
    commands << _volume_kernel(_reduce.buffer(), region.reduce_offset)
                    .dispatch(kSingleThreadBlock);
    commands << _morton_kernel(_prims.buffer(), _keys_a.buffer(), _reduce.buffer(),
                               region.prim_offset, region.prim_count, region.reduce_offset)
                    .dispatch(region.prim_count);
    // 4 x 8 bit LSD radix sort into `keys_a` (see fallback_rtx_sort.h).
    _sort.sort(commands, _keys_a.buffer(), _keys_b.buffer(),
               region.prim_offset, region.prim_count);
    // Radix tree: the leaves first (one random `prims` read per leaf, then the
    // leaf node is written), then the internal nodes (one warp per node, the
    // lanes splitting the node's leaf range).
    commands << _leaf_kernel(_keys_a.buffer(), _prims.buffer(), _accel.buffer(),
                             region.prim_offset, region.node_base, region.prim_count)
                    .dispatch(region.prim_count);
    if (region.prim_count > 1u) {
        // One warp per internal node, i.e. `count * warp_size` threads, laid out
        // over a 2D grid whose every dimension stays inside the 65535 work-groups
        // DirectX 12 allows per dimension (`rows` of `row_stride` threads each).
        auto threads = static_cast<size_t>(region.prim_count) * _warp_size;
        auto groups = (threads + sort_block_size - 1u) / sort_block_size;
        auto groups_x = std::min<size_t>(groups, max_build_dispatch_groups);
        auto rows = (groups + groups_x - 1u) / groups_x;
        auto row_stride = static_cast<uint>(groups_x * sort_block_size);
        commands << _build_kernel(_keys_a.buffer(), _accel.buffer(), region.prim_offset,
                                  region.node_base, region.prim_count, row_stride)
                        .dispatch(static_cast<uint>(groups_x * sort_block_size),
                                  static_cast<uint>(rows));
    }
}

// ---------------------------------------------------------------------------
// Host-side structural check
// ---------------------------------------------------------------------------

namespace {

// One lane of a downloaded uint4 buffer, addressed the way the ABI addresses it:
// a *uint* index into the acceleration buffer.  The header slots, the instance
// lanes and the blas-table lanes are all defined that way.
[[nodiscard]] uint load_u32(luisa::span<const uint4> buffer, size_t index) noexcept {
    auto value = buffer[index / 4u];
    switch (index % 4u) {
        case 0u: return value.x;
        case 1u: return value.y;
        case 2u: return value.z;
        default: return value.w;
    }
}

// Header slot `k` is the uint at index `base * 4 + k` (the region is four uint4).
[[nodiscard]] uint header_slot(luisa::span<const uint4> accel, uint header_base, uint slot) noexcept {
    return load_u32(accel, static_cast<size_t>(header_base) * 4u + slot);
}

[[nodiscard]] float4 u4_as_float4(const uint4 &v) noexcept {
    return make_float4(std::bit_cast<float>(v.x), std::bit_cast<float>(v.y),
                       std::bit_cast<float>(v.z), std::bit_cast<float>(v.w));
}

[[nodiscard]] float3 u4_xyz_as_float3(const uint4 &v) noexcept {
    return make_float3(std::bit_cast<float>(v.x), std::bit_cast<float>(v.y),
                       std::bit_cast<float>(v.z));
}

[[nodiscard]] float max_abs_diff(float3 a, float3 b) noexcept {
    return max(max(abs(a.x - b.x), abs(a.y - b.y)), abs(a.z - b.z));
}

[[nodiscard]] float max_component(float3 v) noexcept {
    return max(max(v.x, v.y), v.z);
}

// Problems are counted, not thrown, and only the first few are printed: a
// structurally broken tree produces one problem per node, and a thousand
// identical warnings help nobody.  `operator()` is the gate - it counts one
// problem and returns true while the warning budget lasts - so the message at
// the call site stays a literal format string, which is what the project's
// logging requires.
class ProblemReport {

public:
    static constexpr size_t max_reported = 16u;

    [[nodiscard]] bool operator()() noexcept {
        _count++;
        if (_reported < max_reported) {
            _reported++;
            return true;
        }
        return false;
    }

    [[nodiscard]] size_t count() const noexcept { return _count; }
    [[nodiscard]] size_t reported() const noexcept { return _reported; }

private:
    size_t _count{0u};
    size_t _reported{0u};
};

}// namespace

size_t FallbackRtxStorage::validate_tree(const HostView &view, uint header_base,
                                         bool expect_tlas) noexcept {
    auto accel = view.accel;
    ProblemReport problem;
    // The header is the first thing that can be wrong; everything below reads it.
    if (static_cast<size_t>(header_base) + header_u4 > accel.size()) {
        if (problem()) {
            LUISA_WARNING("[fallback-rtx] the region header at u4 {} is outside the {} u4 "
                          "that were downloaded.",
                          header_base, accel.size());
        }
        return problem.count();
    }
    auto base = header_slot(accel, header_base, h_base);
    auto node_base = header_slot(accel, header_base, h_node_base);
    auto node_count = header_slot(accel, header_base, h_node_count);
    auto prim_count = header_slot(accel, header_base, h_prim_count);
    auto table_base = header_slot(accel, header_base, h_blas_table_base);
    auto blas_count = header_slot(accel, header_base, h_blas_count);
    auto index_base = header_slot(accel, header_base, h_index_base);
    auto index_count = header_slot(accel, header_base, h_index_count);
    auto vertex_base = header_slot(accel, header_base, h_vertex_base);
    auto vertex_count = header_slot(accel, header_base, h_vertex_count);
    auto root = header_slot(accel, header_base, h_root);
    auto flags = header_slot(accel, header_base, h_flags);
    auto is_tlas = (flags & region_flag_tlas) != 0u;

    // ---- the header must describe this region -------------------------------
    if (base != header_base && problem()) {
        LUISA_WARNING("[fallback-rtx] the header at u4 {} claims to be the region at u4 {}.",
                      header_base, base);
    }
    if (prim_count == 0u && problem()) {
        LUISA_WARNING("[fallback-rtx] the region at u4 {} has no primitive.", header_base);
    }
    if (node_count != prim_count * 2u - 1u && problem()) {
        LUISA_WARNING("[fallback-rtx] the region at u4 {} has {} nodes for {} primitives.",
                      header_base, node_count, prim_count);
    }
    if (node_base != base + header_u4 && problem()) {
        LUISA_WARNING("[fallback-rtx] the region at u4 {} puts its node array at {} "
                      "instead of {}.",
                      header_base, node_base, base + header_u4);
    }
    if (is_tlas != expect_tlas && problem()) {
        LUISA_WARNING("[fallback-rtx] the region at u4 {} is {}.",
                      header_base, is_tlas ? "a TLAS" : "a BLAS");
    }
    auto end = static_cast<size_t>(node_base) + static_cast<size_t>(node_u4) * node_count;
    if (is_tlas) {
        if ((blas_count == 0u || index_count != 0u || vertex_count != 0u) && problem()) {
            LUISA_WARNING("[fallback-rtx] the TLAS at u4 {} carries {} index / {} vertex "
                          "records.",
                          header_base, index_count, vertex_count);
        }
        if (table_base != end && problem()) {
            LUISA_WARNING("[fallback-rtx] the TLAS at u4 {} puts its blas table at {} "
                          "instead of {}.",
                          header_base, table_base, end);
        }
        end += static_cast<size_t>(blas_record_u4) * blas_count;
        if (blas_count != prim_count && problem()) {
            LUISA_WARNING("[fallback-rtx] the TLAS at u4 {} has {} table records for {} "
                          "instances.",
                          header_base, blas_count, prim_count);
        }
    } else {
        if ((blas_count != 0u || table_base != 0u) && problem()) {
            LUISA_WARNING("[fallback-rtx] the BLAS at u4 {} carries a blas table of {} "
                          "records.",
                          header_base, blas_count);
        }
        if (index_count != prim_count && problem()) {
            LUISA_WARNING("[fallback-rtx] the BLAS at u4 {} has {} triangle records for {} "
                          "triangles.",
                          header_base, index_count, prim_count);
        }
        if (index_base != end && problem()) {
            LUISA_WARNING("[fallback-rtx] the BLAS at u4 {} puts its index array at {} "
                          "instead of {}.",
                          header_base, index_base, end);
        }
        end += index_count;
        if (vertex_base != end && problem()) {
            LUISA_WARNING("[fallback-rtx] the BLAS at u4 {} puts its vertex array at {} "
                          "instead of {}.",
                          header_base, vertex_base, end);
        }
        end += vertex_count;
    }
    if (end > accel.size() && problem()) {
        LUISA_WARNING("[fallback-rtx] the region at u4 {} ends at u4 {} but only {} u4 were "
                      "downloaded.",
                      header_base, end, accel.size());
    }
    if (root != node_base && problem()) {
        LUISA_WARNING("[fallback-rtx] the region at u4 {} reports its root as {} instead of "
                      "its node array {}.",
                      header_base, root, node_base);
    }

    // ---- the tree: reachable from the root exactly once ---------------------
    // A header whose node array runs past the downloaded buffer must not turn the
    // walk into an out-of-bounds read: it is clipped to what was downloaded, and
    // the region-size check above has already counted the mismatch.
    auto node_end = static_cast<size_t>(node_base) + static_cast<size_t>(node_u4) * node_count;
    auto walk_end = std::min(node_end, accel.size());
    // The arrays the walk reads by computed index are clipped for the same
    // reason.  Every term is a uint4 offset (the unit `accel` is indexed in).
    auto index_in_range =
        static_cast<size_t>(index_base) + prim_count <= accel.size();
    auto table_in_range =
        static_cast<size_t>(table_base) +
            static_cast<size_t>(blas_record_u4) * blas_count <=
        accel.size();
    auto is_node_offset = [&](size_t offset) noexcept {
        return offset >= node_base && offset + node_u4 <= walk_end &&
               (offset - node_base) % node_u4 == 0u;
    };
    luisa::vector<uint32_t> visits(node_count, 0u);
    luisa::vector<size_t> stack;
    stack.reserve(64u);
    stack.push_back(root);
    size_t leaves = 0u;
    while (!stack.empty()) {
        auto offset = stack.back();
        stack.pop_back();
        if (!is_node_offset(offset)) {
            if (problem()) {
                LUISA_WARNING("[fallback-rtx] a child handle {} is not the uint4 offset of a "
                              "node of [{}, {}).",
                              offset, node_base, node_end);
            }
            continue;
        }
        auto index = (offset - static_cast<size_t>(node_base)) / node_u4;
        if (visits[index]++ != 0u) {
            if (problem()) {
                LUISA_WARNING("[fallback-rtx] node {} is reached more than once.", index);
            }
            continue;
        }
        auto plane_lo = accel[offset];
        auto plane_hi = accel[offset + 1u];
        auto left = plane_lo.w;
        auto right = plane_hi.w;
        auto lo = u4_xyz_as_float3(plane_lo);
        auto hi = u4_xyz_as_float3(plane_hi);
        if ((lo.x > hi.x || lo.y > hi.y || lo.z > hi.z) && problem()) {
            LUISA_WARNING("[fallback-rtx] node {} has an inverted AABB.", index);
        }
        if (left == invalid_offset) {
            // A leaf's right handle is the primitive id inside *this* tree.
            leaves++;
            if (right >= prim_count) {
                if (problem()) {
                    LUISA_WARNING("[fallback-rtx] leaf {} carries primitive id {}, but the "
                                  "tree has {} primitives.",
                                  index, right, prim_count);
                }
            } else if (!expect_tlas && index_in_range) {
                // A BLAS leaf names a triangle, whose three indices must address
                // the vertices copied into this region.
                auto triangle = static_cast<size_t>(index_base) * 4u +
                                static_cast<size_t>(right) * 4u;
                for (auto k = 0u; k < 3u; k++) {
                    auto vertex = load_u32(accel, triangle + k);
                    if (vertex >= vertex_count && problem()) {
                        LUISA_WARNING("[fallback-rtx] triangle {} of the BLAS at u4 {} "
                                      "indexes vertex {}, but the region holds {} vertices.",
                                      right, header_base, vertex, vertex_count);
                    }
                }
            }
            continue;
        }
        // An internal node's AABB must be the union of its children's.
        if (!is_node_offset(left) && problem()) {
            LUISA_WARNING("[fallback-rtx] node {} points at the left child {}.", index, left);
        }
        if (!is_node_offset(right) && problem()) {
            LUISA_WARNING("[fallback-rtx] node {} points at the right child {}.", index, right);
        }
        if (is_node_offset(left) && is_node_offset(right)) {
            auto a_lo = u4_xyz_as_float3(accel[left]);
            auto a_hi = u4_xyz_as_float3(accel[left + 1u]);
            auto b_lo = u4_xyz_as_float3(accel[right]);
            auto b_hi = u4_xyz_as_float3(accel[right + 1u]);
            auto union_lo = min(a_lo, b_lo);
            auto union_hi = max(a_hi, b_hi);
            // The union is the same min/max chain the build kernel runs, so only
            // a fused multiply-add on one side can move it; a relative tolerance
            // covers that and nothing else.
            auto tolerance = 1.0e-4f * std::max(1.0f, max_component(union_hi - union_lo));
            if ((max_abs_diff(union_lo, lo) > tolerance ||
                 max_abs_diff(union_hi, hi) > tolerance) &&
                problem()) {
                LUISA_WARNING("[fallback-rtx] node {} is not the union of its children.", index);
            }
        }
        stack.push_back(left);
        stack.push_back(right);
    }
    auto unreachable = 0u;
    for (auto i = 0u; i < node_count; i++) {
        if (visits[i] == 0u) {
            unreachable++;
            if (problem()) {
                LUISA_WARNING("[fallback-rtx] node {} is not reachable from the root.", i);
            }
        }
    }
    if (unreachable != 0u && problem()) {
        LUISA_WARNING("[fallback-rtx] {} of the {} nodes of the tree at u4 {} are unreachable.",
                      unreachable, node_count, header_base);
    }
    if (leaves != prim_count && problem()) {
        LUISA_WARNING("[fallback-rtx] the tree at u4 {} has {} leaves for {} primitives.",
                      header_base, leaves, prim_count);
    }

    // ---- TLAS: the blas table and the instances -----------------------------
    if (!expect_tlas) { return problem.count(); }
    if (!table_in_range && problem()) {
        LUISA_WARNING("[fallback-rtx] the blas table of the TLAS at u4 {} runs past the "
                      "downloaded buffer and was not checked row by row.",
                      header_base);
    }
    for (auto row = 0u; table_in_range && row < blas_count; row++) {
        auto record = static_cast<size_t>(table_base) * 4u + static_cast<size_t>(row) * 8u;
        auto blas_base = load_u32(accel, record + 0u);
        auto blas_node_base = load_u32(accel, record + 1u);
        auto blas_index_base = load_u32(accel, record + 2u);
        auto blas_vertex_base = load_u32(accel, record + 3u);
        auto blas_triangle_count = load_u32(accel, record + 4u);
        if (blas_base == 0u) {
            // The null reference: an instance the caller never gave a mesh, or one
            // whose blas directory entry is still the builder's sentinel.  No real
            // region starts at 0 (the planner reserves the four uint4 at the front
            // of the buffer), so this is unambiguous, and a traversal has to skip
            // such a row (fallback_rtx_layout.h).
            if (problem()) {
                LUISA_WARNING("[fallback-rtx] blas-table row {} is a null reference: instance {} "
                              "was never given a mesh.",
                              row, row);
            }
            continue;
        }
        if (static_cast<size_t>(blas_base) + header_u4 > accel.size() ||
            header_slot(accel, blas_base, h_base) != blas_base) {
            if (problem()) {
                LUISA_WARNING("[fallback-rtx] blas-table row {} does not point at a region "
                              "(base {}).",
                              row, blas_base);
            }
            continue;
        }
        if ((header_slot(accel, blas_base, h_node_base) != blas_node_base ||
             header_slot(accel, blas_base, h_index_base) != blas_index_base ||
             header_slot(accel, blas_base, h_vertex_base) != blas_vertex_base ||
             header_slot(accel, blas_base, h_prim_count) != blas_triangle_count) &&
            problem()) {
            LUISA_WARNING("[fallback-rtx] blas-table row {} disagrees with the region at "
                          "u4 {}.",
                          row, blas_base);
        }
        // The referenced tree has to be well formed on its own; this is the same
        // check one level down, so validating a TLAS covers the whole scene.
        if (validate_tree(view, blas_base, false) != 0u && problem()) {
            LUISA_WARNING("[fallback-rtx] blas-table row {} points at a malformed BLAS region "
                          "(u4 {}).",
                          row, blas_base);
        }
    }
    auto root_aabb_in_range = static_cast<size_t>(root) + 2u <= accel.size();
    auto root_lo = root_aabb_in_range ? u4_xyz_as_float3(accel[root]) : make_float3(1.0e30f);
    auto root_hi = root_aabb_in_range ? u4_xyz_as_float3(accel[root + 1u]) : make_float3(-1.0e30f);
    auto root_tolerance = 1.0e-4f * std::max(1.0f, max_component(root_hi - root_lo));
    for (auto instance = 0u; root_aabb_in_range && instance < prim_count; instance++) {
        auto first = static_cast<size_t>(view.instance_base_u4) +
                     static_cast<size_t>(instance) * instance_u4;
        if (first + instance_u4 > view.instances.size()) {
            if (problem()) {
                LUISA_WARNING("[fallback-rtx] instance {} is outside the {} u4 of the "
                              "downloaded instance buffer (slice base {}).",
                              instance, view.instances.size(), view.instance_base_u4);
            }
            break;
        }
        auto misc = view.instances[first + i_misc];
        auto reserved = view.instances[first + i_misc + 1u];
        auto blas_index = misc.x;
        // The build keeps one blas-table record per instance and points the
        // instance at its own record, so a traversal may index the table with
        // either the instance index or `blas_index` (fallback_rtx_layout.h).
        if (blas_index != instance && problem()) {
            LUISA_WARNING("[fallback-rtx] instance {} carries blas_index {}; the build keeps "
                          "them equal.",
                          instance, blas_index);
        }
        if (blas_index >= blas_count) {
            if (problem()) {
                LUISA_WARNING("[fallback-rtx] instance {} carries blas_index {}, but the "
                              "table has {} records.",
                              instance, blas_index, blas_count);
            }
            continue;
        }
        if (reserved.x >= view.blas_directory_entries && problem()) {
            LUISA_WARNING("[fallback-rtx] instance {} references blas directory entry {}, but "
                          "the directory holds {}.",
                          instance, reserved.x, view.blas_directory_entries);
        }
        // The instance's world AABB - the referenced BLAS' root AABB transformed
        // by the instance - has to be inside the TLAS root AABB the build made
        // out of it, which cross-checks the table against the tree.
        auto record = static_cast<size_t>(table_base) * 4u +
                      static_cast<size_t>(blas_index) * 8u;
        auto blas_node_base = load_u32(accel, record + 1u);
        if (static_cast<size_t>(blas_node_base) + 2u > accel.size()) {
            if (problem()) {
                LUISA_WARNING("[fallback-rtx] instance {} resolves to the node array at u4 "
                              "{}, which is outside the downloaded buffer.",
                              instance, blas_node_base);
            }
            continue;
        }
        auto blas_lo = u4_xyz_as_float3(accel[blas_node_base]);
        auto blas_hi = u4_xyz_as_float3(accel[blas_node_base + 1u]);
        auto w0 = u4_as_float4(view.instances[first + i_to_world + 0u]);
        auto w1 = u4_as_float4(view.instances[first + i_to_world + 1u]);
        auto w2 = u4_as_float4(view.instances[first + i_to_world + 2u]);
        auto world_lo = make_float3(1.0e30f);
        auto world_hi = make_float3(-1.0e30f);
        for (auto c = 0u; c < 8u; c++) {
            auto corner = make_float3((c & 1u) != 0u ? blas_hi.x : blas_lo.x,
                                      (c & 2u) != 0u ? blas_hi.y : blas_lo.y,
                                      (c & 4u) != 0u ? blas_hi.z : blas_lo.z);
            auto p = make_float4(corner, 1.0f);
            auto world = make_float3(dot(p, w0), dot(p, w1), dot(p, w2));
            world_lo = min(world_lo, world);
            world_hi = max(world_hi, world);
        }
        auto outside = (world_lo.x < root_lo.x - root_tolerance) ||
                       (world_lo.y < root_lo.y - root_tolerance) ||
                       (world_lo.z < root_lo.z - root_tolerance) ||
                       (world_hi.x > root_hi.x + root_tolerance) ||
                       (world_hi.y > root_hi.y + root_tolerance) ||
                       (world_hi.z > root_hi.z + root_tolerance);
        if (outside && problem()) {
            LUISA_WARNING("[fallback-rtx] instance {} is outside the TLAS root AABB.", instance);
        }
    }
    return problem.count();
}

}// namespace lc::fallback_rtx
