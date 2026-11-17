// Stable LSD radix sort of the software LBVH's `(Morton code, primitive slot)`
// keys - the second stage of `LbvhStorage::build_tree()` and, measured, the
// dominant one (42-62% of a whole build).
//
// The sort is a stable 4 x 8-bit LSD sort over the full 32-bit key, exactly what
// the single-work-group `_sort_kernel` of `LbvhStorage` does.  It is the only
// stage of the build whose *output order* is part of the acceleration
// structure's semantics: stability is what makes four LSD passes add up to a
// correct sort, and it is also what makes two runs bit-identical.  Every method
// below therefore produces the *same* bytes, not merely a sorted array; the
// benchmark (`bench/sort_bench.cpp`) checks that against a host reference and
// against `Method::single_block` element by element.
//
//   * `Method::single_block` is that reference implementation: one work-group
//     walks the range in `block_size`-element tiles, histogramming and
//     scattering with a per-digit bit-flag machine in shared memory.  Its cost
//     is a chain of ~3 barriers per 256 elements of *one* work-group, so the
//     device is idle except in one block - which is why the build's sort takes
//     13.45 ms for 2^20 keys (measured, cuda, release).
//
//   * `Method::multi_block` cuts the range into independent blocks of
//     `block_size * items` elements and runs the same algorithm on every block in
//     parallel.  The only new information a block needs is where "its" digits go
//     in the final output:
//
//         position(element) = digit_base[digit]                  (the digit's own
//                                                               output range)
//                          + counts of that digit in the blocks before it
//                          + its rank inside its block.
//
//     One pass is therefore four dispatches:
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
//     A pass over the whole range therefore costs 24 bytes per element (one
//     histogram read, one scatter read, one scatter write) and is 16 dispatches
//     for the four passes.
//
//     Two scatter variants are compiled and measured (`variant`):
//       variant 0 uses the same shared-memory flag machine per tile as the
//       reference (lowest risk, 3 barriers per tile), variant 1 ranks a whole
//       warp sub-chunk with warp ballots and needs only 3 barriers per *block*
//       (the onesweep shape).
//
// The result always ends in `keys_a`: four passes ping-pong
// keys_a -> keys_b -> keys_a -> keys_b -> keys_a, so an odd number of passes
// ends in `keys_b` (only used by the benchmark's per-pass breakdown).
//
// All three dials (`items`, `variant`, `batched`) only change *how* the same
// bytes are produced; the default settings are the ones the benchmark measured
// as best, and `Method::automatic` depends on `count` alone (see
// `automatic_min_count`), never on hidden state.

#pragma once

#include "lbvh_common.h"

#include <luisa/dsl/local.h>
#include <luisa/runtime/command_list.h>

#include <cstddef>

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

class LbvhRadixSort {

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
    // (variant 1).  The smallest value bounds the scratch allocation.
    static constexpr uint smallest_items = 4u;
    static constexpr uint largest_items = 16u;
    static constexpr uint default_items = 8u;
    // Upper bound of the number of scan groups of the matrix scan; a group is one
    // work-group of 256 threads that walks its own range of blocks, so the
    // sequential part of the scan is `ceil(blocks / scan_group_limit)` steps.
    static constexpr uint scan_group_limit = 64u;
    // `Method::automatic` sorts fewer than this many elements with one work-group
    // (measured: the parallel path's extra dispatches are not paid back below it).
    static constexpr uint automatic_min_count = 8192u;

    // `capacity` is the maximum element count of any tree that will be sorted;
    // it sizes the scratch, which is allocated once here.
    LbvhRadixSort(Device &device, size_t capacity) noexcept;

    // Extra device memory the parallel path needs for a tree of the capacity
    // given to the constructor.  The single-work-group path needs none, but the
    // scratch is allocated unconditionally so that any call - including the
    // first `automatic` one - can use either method without an allocation.
    [[nodiscard]] size_t scratch_bytes() const noexcept { return scratch_bytes_for(_capacity); }
    // The same number without an instance: the storage size query
    // (`LbvhStorage::estimate`) has to report it *before* anything is allocated,
    // exactly like the scratch size of a backend build.  It is the worst case
    // over every `items` setting, so changing that knob cannot invalidate an
    // estimate the caller has already used to size its buffers.
    [[nodiscard]] static size_t scratch_bytes_for(size_t capacity) noexcept;
    [[nodiscard]] size_t capacity() const noexcept { return _capacity; }

    // Stable LSD sort of `keys_a[base .. base + count)`, four 8-bit passes,
    // result left in `keys_a`.  `keys_b` is scratch of the same size; both
    // buffers must hold at least `base + count` elements and are never touched
    // outside that range.  `count <= 1` is a no-op (and does not touch anything),
    // `base` may be non-zero (several trees share one buffer), and two runs of
    // the same method on the same input are bit-identical by construction.
    void sort(Stream &stream, const Buffer<LbvhKey> &keys_a, const Buffer<LbvhKey> &keys_b,
              uint base, uint count, Method method = Method::automatic) noexcept;

    // The first `pass_count` (1..4) passes of the same sort.  Only the benchmark
    // uses this: the difference between consecutive pass counts is the cost of
    // one pass.  The result of an odd pass count ends in `keys_b`.
    void sort_passes(Stream &stream, const Buffer<LbvhKey> &keys_a, const Buffer<LbvhKey> &keys_b,
                     uint base, uint count, uint pass_count,
                     Method method = Method::automatic) noexcept;

    // Measurement knobs; they only change the implementation of the same bytes.
    // `items` is snapped to the power-of-two values this class compiles for
    // (4, 8, 16) and `variant` is 0 (shared-memory flag machine per tile, the
    // fastest above ~64K elements) or 1 (warp-ballot sub-chunks, the fastest
    // below it).  The defaults - 8 elements and variant 0 - are what the
    // benchmark measured as the best overall configuration.
    void set_items(uint items) noexcept;
    void set_variant(uint variant) noexcept;
    void set_batched(bool batched) noexcept;
    [[nodiscard]] uint items() const noexcept { return _items; }
    [[nodiscard]] uint variant() const noexcept { return _variant; }
    [[nodiscard]] bool batched() const noexcept { return _batched; }

    // Which implementation a call with `method` on `count` elements will use.
    [[nodiscard]] static Method resolve_method(Method method, uint count) noexcept;

    // Elements of one independent block of the parallel path.
    [[nodiscard]] uint chunk_size() const noexcept { return block_size * _items; }

private:
    void encode_multi_block_pass(CommandList &commands,
                                 const Buffer<LbvhKey> &keys_in, const Buffer<LbvhKey> &keys_out,
                                 uint base, uint count, uint shift) noexcept;

    size_t _capacity{1u};
    uint _items{default_items};
    uint _variant{0u};
    bool _batched{true};
    size_t _max_blocks{1u};
    size_t _max_groups{1u};
    Buffer<uint> _matrix;    // _max_blocks x bins: per-block histogram, then its exclusive scan
    Buffer<uint> _groups;    // _max_groups x bins: per-group totals, then their exclusive scan
    Buffer<uint> _digit_base;// bins: where each digit's output range starts
    // Reference: one work-group sorts the whole range (buffer args, so one shader
    // can ping-pong).
    Shader1D<Buffer<LbvhKey>, Buffer<LbvhKey>, uint, uint, uint> _single_pass;
    // Per-block digit histogram -> `_matrix`.
    Shader2D<Buffer<LbvhKey>, Buffer<uint>, uint, uint, uint, uint, uint> _hist_pass;
    // Exclusive scan of the `(block, digit)` matrix over each scan group.
    Shader1D<Buffer<uint>, Buffer<uint>, uint, uint> _scan_partial;
    // Exclusive scan of the scan groups' per-digit totals, in place, plus the
    // exclusive scan of the per-digit totals of the whole range.
    Shader1D<Buffer<uint>, Buffer<uint>, uint> _scan_groups;
    // Ranked stable scatter of one block, `items` tiles of `block_size` elements.
    Shader2D<Buffer<LbvhKey>, Buffer<LbvhKey>, Buffer<uint>, Buffer<uint>, Buffer<uint>,
             uint, uint, uint, uint, uint, uint>
        _scatter_tile;
    // Ranked stable scatter of one block, one warp per sub-chunk, `items`
    // elements per lane; one shader per `items`.
    luisa::vector<Shader2D<Buffer<LbvhKey>, Buffer<LbvhKey>, Buffer<uint>, Buffer<uint>,
                           Buffer<uint>, uint, uint, uint, uint, uint>>
        _scatter_warp;
};

}// namespace luisa::example::lbvh
