// Stable LSD radix sort of the fallback LBVH's `(Morton code, primitive slot)`
// keys - the second stage of `FallbackRtxStorage::build_tree()`.
//
// Ported from examples/compute/lbvh/lbvh_sort.{h,cpp} with the *algorithm*
// untouched: the same stable 4 x 8-bit LSD passes, the same single-block /
// multi-block split, the same `automatic_min_count` threshold and the same two
// scatter variants, so a tree built here is byte-identical to one built by the
// example for the same input.
//
// Only the namespace and the *surface* changed:
//
//   * the build is recorded into a `CommandList` instead of a `Stream`, because
//     there is no stream here - a backend splices the list this library returns
//     into its own stream at the position of the build command
//     (fallback_rtx.h), and a `Stream` cannot be handed across that boundary;
//   * the benchmark-only entry points (`sort_passes`, the batched/unbatched
//     switch, the size query) are gone: nothing in the fallback uses them;
//   * the scratch is *growable*: the storage plans its region sizes at build
//     time (the API knows no scene size up front), so the sort has to survive a
//     capacity that is only discovered later.  Growth replaces the scratch
//     buffers and *retires* the previous ones, because a recorded command may
//     still be in flight (see `FallbackRtxStorage`).
//
// The sort is a stable 4 x 8-bit LSD sort over the full 32-bit key.  Stability
// is what makes four LSD passes add up to a correct sort, and it is also what
// makes two runs bit-identical.
//
//   * `Method::single_block` is one work-group over the whole range, the
//     shared-memory flag machine of `LbvhStorage`, 3 barriers per 256 elements.
//
//   * `Method::multi_block` cuts the range into independent blocks of
//     `block_size * items` elements and runs the same algorithm on every block
//     in parallel.  The only new information a block needs is where "its" digits
//     go in the final output, which one pass obtains with four dispatches:
//
//         position(element) = digit_base[digit]
//                          + counts of that digit in the blocks before it
//                          + its rank inside its block.
//
//       1. `_hist_pass`    - per-block digit histogram into a global
//                            `blocks x 256` matrix,
//       2. `_scan_partial` - each of <= 64 scan groups exclusive-scans the
//                            matrix of its own block range and publishes its
//                            per-digit total,
//       3. `_scan_groups`  - exclusive scan of those totals, first over the
//                            groups and then over the digits, which yields both
//                            the per-group offsets and every digit's global base,
//       4. `_scatter_*`    - per-block ranked stable scatter, which starts its
//                            per-digit cursors at that sum.
//
//     Two scatter variants are compiled: variant 0 uses the shared-memory flag
//     machine per tile (lowest risk, 3 barriers per tile), variant 1 ranks a
//     whole warp sub-chunk with warp ballots and needs only 3 barriers per
//     *block*.  Both consume exactly the same offsets, so a change of variant
//     cannot change the result - only how it is computed.
//
// The result always ends in `keys_a`: four passes ping-pong
// keys_a -> keys_b -> keys_a -> keys_b -> keys_a.

#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/dsl/local.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include <cstddef>

// ---------------------------------------------------------------------------
// The GPU-side key of both build levels.
//
// These structs have to be declared at global scope: LUISA_STRUCT opens
// namespace luisa::compute to specialize the DSL types for the struct.  The key
// type lives with the sort because it is the sort's input (the storage's Morton
// and leaf kernels include this header and use it).
// ---------------------------------------------------------------------------

// One (Morton code, primitive slot) pair; `slot` indexes the primitive array of
// the tree being built.
struct FallbackRtxKey {
    luisa::uint code;
    luisa::uint slot;
};
LUISA_STRUCT(FallbackRtxKey, code, slot) {};

namespace lc::fallback_rtx {

using namespace luisa;
using namespace luisa::compute;

// Work-group size of the sort (one single work-group sorts a whole tree).
inline constexpr uint sort_block_size = 256u;
inline constexpr uint sort_radix_bins = 256u;
// Upper bound of one grid dimension of any build dispatch.  The radix-tree
// construction spends one warp on every internal node, i.e.
// `primitive_count * warp_size` threads, which for any tree of a few hundred
// thousand primitives is a grid larger than the 65535 work-groups per dimension
// DirectX 12 allows for a single Dispatch().  The grids of this module are
// therefore two-dimensional and folded at this many work-groups per dimension.
inline constexpr uint max_build_dispatch_groups = 65535u;

class FallbackRtxSort {

public:
    // `single_block` is the reference implementation, `multi_block` the
    // parallel one, `automatic` picks between them from `count` only.
    enum class Method { single_block,
                        multi_block,
                        automatic };

    // Work-group size, digit count and lane width of every kernel below.  The
    // warp width is *pinned* (`set_warp_size`) because the warp-scope ranking of
    // variant 1 is defined in terms of `warp_active_bit_or`, which a wider
    // wave/subgroup would widen with it.
    static constexpr uint block_size = sort_block_size;// 256
    static constexpr uint bins = sort_radix_bins;      // 256 (8-bit digits)
    static constexpr uint radix_bits = 8u;
    static constexpr uint warp_lanes = 32u;
    static constexpr uint warps_per_block = block_size / warp_lanes;// 8
    // Elements per so-far-independent block: `block_size * items`, i.e. one pass
    // over a block costs `items` tiles (variant 0) or `items` elements per lane
    // (variant 1).
    static constexpr uint smallest_items = 4u;
    static constexpr uint largest_items = 16u;
    static constexpr uint default_items = 8u;
    // Upper bound of the number of scan groups of the matrix scan; a group is one
    // work-group of 256 threads that walks its own range of blocks, so the
    // sequential part of the scan is `ceil(blocks / scan_group_limit)` steps.
    static constexpr uint scan_group_limit = 64u;
    // `Method::automatic` sorts fewer than this many elements with one work-group
    // (the parallel path's extra dispatches are not paid back below it).
    static constexpr uint automatic_min_count = 8192u;

    // `capacity` is the maximum element count of any tree that will be sorted;
    // it sizes the scratch.  The storage starts from a guess and grows through
    // `reserve()` when the planner discovers a larger tree, so this constructor
    // number is not a limit.
    FallbackRtxSort(Device &device, size_t capacity) noexcept;

    // Make room for `capacity` elements: recreates the scratch buffers when they
    // are too small and retires the old ones (never destroys them - a command
    // already recorded in a stream may still read them).
    void reserve(size_t capacity) noexcept;

    [[nodiscard]] size_t capacity() const noexcept { return _capacity; }
    // Device memory the parallel path's scratch currently occupies.
    [[nodiscard]] size_t scratch_bytes() const noexcept;

    // Stable LSD sort of `keys_a[base .. base + count)`, four 8-bit passes,
    // result left in `keys_a`.  `keys_b` is scratch of the same size; both
    // buffers must hold at least `base + count` elements and are never touched
    // outside that range.  `count <= 1` is a no-op (and does not touch anything),
    // `base` may be non-zero (several trees share one buffer), and two runs of
    // the same method on the same input are bit-identical by construction.
    void sort(CommandList &commands, const Buffer<FallbackRtxKey> &keys_a,
              const Buffer<FallbackRtxKey> &keys_b, uint base, uint count,
              Method method = Method::automatic) noexcept;

    // Measurement knobs; they only change the implementation of the same bytes.
    // `items` is snapped to the power-of-two values this class compiles for
    // (4, 8, 16) and `variant` is 0 (shared-memory flag machine per tile) or 1
    // (warp-ballot sub-chunks).
    void set_items(uint items) noexcept;
    void set_variant(uint variant) noexcept;
    [[nodiscard]] uint items() const noexcept { return _items; }
    [[nodiscard]] uint variant() const noexcept { return _variant; }

    // Which implementation a call with `method` on `count` elements will use.
    [[nodiscard]] static Method resolve_method(Method method, uint count) noexcept;

    // Elements of one independent block of the parallel path.
    [[nodiscard]] uint chunk_size() const noexcept { return block_size * _items; }

private:
    void encode_multi_block_pass(CommandList &commands,
                                 const Buffer<FallbackRtxKey> &keys_in,
                                 const Buffer<FallbackRtxKey> &keys_out,
                                 uint base, uint count, uint shift) noexcept;

    Device *_device{nullptr};
    size_t _capacity{1u};
    uint _items{default_items};
    uint _variant{0u};
    size_t _max_blocks{1u};
    size_t _max_groups{1u};
    Buffer<uint> _matrix;    // _max_blocks x bins: per-block histogram, then its exclusive scan
    Buffer<uint> _groups;    // _max_groups x bins: per-group totals, then their exclusive scan
    Buffer<uint> _digit_base;// bins: where each digit's output range starts
    // Scratch replaced by a growth; kept alive on purpose (see `reserve`).
    luisa::vector<Buffer<uint>> _retired;
    // Reference: one work-group sorts the whole range (buffer args, so one shader
    // can ping-pong).
    Shader1D<Buffer<FallbackRtxKey>, Buffer<FallbackRtxKey>, uint, uint, uint> _single_pass;
    // Per-block digit histogram -> `_matrix`.
    Shader2D<Buffer<FallbackRtxKey>, Buffer<uint>, uint, uint, uint, uint, uint> _hist_pass;
    // Exclusive scan of the `(block, digit)` matrix over each scan group.
    Shader1D<Buffer<uint>, Buffer<uint>, uint, uint> _scan_partial;
    // Exclusive scan of the scan groups' per-digit totals, in place, plus the
    // exclusive scan of the per-digit totals of the whole range.
    Shader1D<Buffer<uint>, Buffer<uint>, uint> _scan_groups;
    // Ranked stable scatter of one block, `items` tiles of `block_size` elements.
    Shader2D<Buffer<FallbackRtxKey>, Buffer<FallbackRtxKey>, Buffer<uint>, Buffer<uint>,
             Buffer<uint>, uint, uint, uint, uint, uint, uint>
        _scatter_tile;
    // Ranked stable scatter of one block, one warp per sub-chunk, `items`
    // elements per lane; one shader per `items`.
    luisa::vector<Shader2D<Buffer<FallbackRtxKey>, Buffer<FallbackRtxKey>, Buffer<uint>,
                           Buffer<uint>, Buffer<uint>, uint, uint, uint, uint, uint>>
        _scatter_warp;
};

}// namespace lc::fallback_rtx
