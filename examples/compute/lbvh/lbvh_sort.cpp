// Implementation of the two LSD radix sorts of `LbvhRadixSort` (see lbvh_sort.h).
//
// Everything here is one of two algorithms over the same bytes:
//
//   single_block  the reference: one work-group over the whole range, the
//                 shared-memory flag machine of `LbvhStorage` per 256 elements;
//   multi_block   the same ranking, but every block of `block_size * items`
//                 elements runs on its own work-group with the block's output
//                 offsets precomputed by a per-block histogram and a scan of the
//                 `(block, digit)` matrix.
//
// The multi-block path is four dispatches per pass, and both scatter variants
// below consume exactly the same `(digit_base, matrix, groups)` offsets, so a
// change of variant cannot change the result - only how it is computed.

#include "lbvh_sort.h"

#include <algorithm>
#include <utility>

namespace luisa::example::lbvh {

namespace {

constexpr uint kBlock = LbvhRadixSort::block_size;     // 256 threads per work-group
constexpr uint kBins = LbvhRadixSort::bins;            // 256 digits (8 bits)
constexpr uint kRadixBits = LbvhRadixSort::radix_bits; // 8
constexpr uint kWords = kBlock / 32u;                  // 8 x 32-bit ballots = 256 lanes
constexpr uint kWarps = LbvhRadixSort::warps_per_block;// 8 warps
constexpr uint kWarpLanes = LbvhRadixSort::warp_lanes; // 32
// One spare digit per warp: the lanes whose element lies outside the range are
// parked in digit `kBins` (see the scatter), so they can be ranked by the warp
// ballot without ever contributing to a real digit.  Keeping that spare slot in
// the same shared array keeps the per-warp slices apart.
constexpr uint kWarpDigits = kBins + 1u;
// Upper bound of one grid dimension; a wider grid is folded into a second
// dimension, which is the only shape DirectX 12 accepts above 65535 groups.
constexpr uint kMaxGroupsX = max_build_dispatch_groups;
// Longest per-thread walk of the matrix scan (`_scan_partial`).
constexpr uint kScanWalk = 256u;

[[nodiscard]] constexpr uint ceil_div(uint x, uint y) noexcept { return (x + y - 1u) / y; }

[[nodiscard]] size_t maximum_blocks(size_t capacity) noexcept {
    constexpr auto smallest_chunk =
        static_cast<size_t>(kBlock) * LbvhRadixSort::smallest_items;
    return (std::max<size_t>(capacity, 1u) + smallest_chunk - 1u) / smallest_chunk;
}

[[nodiscard]] size_t maximum_groups(size_t blocks) noexcept {
    return std::max<size_t>(std::min<size_t>(blocks, LbvhRadixSort::scan_group_limit), 1u);
}

// A power of two in [smallest_items, largest_items]; the warp-scatter variant is
// compiled for each of those, so any other value is snapped to one of them.
[[nodiscard]] uint snap_items(uint items) noexcept {
    auto value = LbvhRadixSort::smallest_items;
    while (value < LbvhRadixSort::largest_items && value < items) { value *= 2u; }
    return value;
}

}// namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

LbvhRadixSort::LbvhRadixSort(Device &device, size_t capacity) noexcept
    : _capacity{std::max<size_t>(capacity, 1u)},
      _max_blocks{maximum_blocks(_capacity)},
      _max_groups{maximum_groups(_max_blocks)},
      _matrix{device.create_buffer<uint>(_max_blocks * kBins)},
      _groups{device.create_buffer<uint>(_max_groups * kBins)},
      _digit_base{device.create_buffer<uint>(kBins)} {

    // ---- reference: one work-group, four 8-bit passes -----------------------
    // A verbatim behavioural copy of `LbvhStorage`'s sort: the same tile walk,
    // the same shared-memory flag machine, so it is the baseline every other
    // implementation is compared against.
    _single_pass = device.compile(Kernel1D{
        [](BufferVar<LbvhKey> keys_in, BufferVar<LbvhKey> keys_out,
           UInt base, UInt count, UInt shift) noexcept {
            set_block_size(kBlock);
            Shared<uint> histogram{kBins};
            Shared<uint> offsets{kBins};
            Shared<uint> flags{kBins * kWords};
            UInt tid = thread_x();
            UInt word = tid / 32u;
            UInt bit = 1u << (tid % 32u);
            // per-digit histogram
            histogram[tid] = 0u;
            sync_block();
            UInt i = tid;
            $while (i < count) {
                auto code = keys_in.read(base + i).code;
                histogram.atomic((code >> shift) & (kBins - 1u)).fetch_add(1u);
                i = i + kBlock;
            };
            sync_block();
            // exclusive scan of the 256 digits (one thread is plenty here)
            $if (tid == 0u) {
                UInt sum = def(0u);
                $for (b, kBins) {
                    auto c = histogram[b];
                    offsets[b] = sum;
                    sum = sum + c;
                };
            };
            sync_block();
            // stable scatter, chunk by chunk: every lane only accumulates the
            // lanes before it inside its own chunk, and the shared cursor is
            // advanced once per digit and chunk.
            UInt chunks = (count + kBlock - 1u) / kBlock;
            $for (chunk, chunks) {
                $for (k, kWords) { flags[tid * kWords + k] = 0u; };
                sync_block();
                UInt index = chunk * kBlock + tid;
                Bool valid = index < count;
                UInt bin = def(0u);
                UInt bin_offset = def(0u);
                UInt code = def(0u);
                UInt slot = def(0u);
                $if (valid) {
                    auto key = keys_in.read(base + index);
                    code = key.code;
                    slot = key.slot;
                    bin = (code >> shift) & (kBins - 1u);
                    bin_offset = offsets[bin];
                    flags.atomic(bin * kWords + word).fetch_add(bit);
                };
                sync_block();
                $if (valid) {
                    UInt prefix = def(0u);
                    UInt total = def(0u);
                    $for (w, kWords) {
                        auto bits = flags[bin * kWords + w];
                        auto bin_count = popcount(bits);
                        total = total + bin_count;
                        prefix = prefix + select(0u, bin_count, w < word);
                        prefix = prefix + select(0u, popcount(bits & (bit - 1u)), w == word);
                    };
                    Var<LbvhKey> sorted;
                    sorted.code = code;
                    sorted.slot = slot;
                    keys_out.write(base + bin_offset + prefix, sorted);
                    // the last element of a digit advances the shared cursor
                    $if (prefix == total - 1u) {
                        offsets.atomic(bin).fetch_add(total);
                    };
                };
                sync_block();
            };
        }});

    // ---- parallel: per-block digit histogram --------------------------------
    // One work-group per block of `items * block_size` elements.  The histogram
    // is privatized per warp (contention-free shared atomics) and reduced to the
    // global `(block, digit)` matrix by 256 threads, one digit each.
    _hist_pass = device.compile(Kernel2D{
        [](BufferVar<LbvhKey> keys_in, BufferVar<uint> matrix,
           UInt base, UInt count, UInt shift, UInt grid_x, UInt items) noexcept {
            set_block_size(kBlock);
            Shared<uint> local_hist{kWarps * kBins};
            UInt tid = thread_x();
            UInt warp = tid / kWarpLanes;
            $for (i, ceil_div(kWarps * kBins, kBlock)) {
                auto slot = i * kBlock + tid;
                $if (slot < kWarps * kBins) { local_hist[slot] = 0u; };
            };
            sync_block();
            UInt block = block_id().y * grid_x + block_id().x;
            UInt first = block * (items * kBlock);
            $for (t, items) {
                UInt index = first + t * kBlock + tid;
                $if (index < count) {
                    auto code = keys_in.read(base + index).code;
                    local_hist.atomic(warp * kBins + ((code >> shift) & (kBins - 1u)))
                        .fetch_add(1u);
                };
            };
            sync_block();
            $if (tid < kBins) {
                UInt sum = def(0u);
                $for (w, kWarps) { sum = sum + local_hist[w * kBins + tid]; };
                matrix.write(block * kBins + tid, sum);
            };
        }});

    // ---- parallel: exclusive scan of the (block, digit) matrix --------------
    // One work-group per scan group, one thread per digit, walking the blocks of
    // the group in order.  `_groups` receives the group's per-digit total, which
    // the next kernel turns into the group's own output offset.
    _scan_partial = device.compile(Kernel1D{
        [](BufferVar<uint> matrix, BufferVar<uint> groups,
           UInt blocks, UInt blocks_per_group) noexcept {
            set_block_size(kBins);
            UInt digit = thread_x();
            UInt begin = block_id().x * blocks_per_group;
            UInt end = min(begin + blocks_per_group, blocks);
            UInt sum = def(0u);
            $for (block, begin, end) {
                auto value = matrix.read(block * kBins + digit);
                matrix.write(block * kBins + digit, sum);
                sum = sum + value;
            };
            groups.write(block_id().x * kBins + digit, sum);
        }});

    // ---- parallel: exclusive scan of the scan groups' totals, plus the base
    // ---- of every digit's output range --------------------------------------
    // One work-group.  First every digit's total over the whole range is turned
    // into the digit's output offset (an exclusive scan over the 256 digits), then
    // the group totals are scanned in place.  A block's digit `d` starts at
    // `_digit_base[d] + <counts of d in the preceding blocks>`: the first term is
    // what makes the digit's whole output range contiguous, the second what keeps
    // the blocks of that digit in input order.
    _scan_groups = device.compile(Kernel1D{
        [](BufferVar<uint> groups, BufferVar<uint> digit_base, UInt group_count) noexcept {
            set_block_size(kBins);
            Shared<uint> total{kBins};
            UInt digit = thread_x();
            UInt sum = def(0u);
            $for (group, group_count) {
                auto value = groups.read(group * kBins + digit);
                groups.write(group * kBins + digit, sum);
                sum = sum + value;
            };
            total[digit] = sum;
            sync_block();
            // one sequential 256-entry scan is cheaper than a block scan here
            $if (digit == 0u) {
                UInt prefix = def(0u);
                $for (d, kBins) {
                    auto value = total[d];
                    digit_base.write(d, prefix);
                    prefix = prefix + value;
                };
            };
        }});

    // ---- parallel: ranked stable scatter, `items` tiles per block -----------
    // The shared flag machine of the reference, with the block's per-digit
    // output offset as the starting cursor instead of the zero of a scan over
    // the whole range.  Stability is unchanged: the lanes of a tile keep their
    // order, the tiles of a block are walked in order, and a block's digits start
    // where every preceding block left off.
    _scatter_tile = device.compile(Kernel2D{
        [](BufferVar<LbvhKey> keys_in, BufferVar<LbvhKey> keys_out,
           BufferVar<uint> matrix, BufferVar<uint> groups, BufferVar<uint> digit_base,
           UInt base, UInt count, UInt shift, UInt grid_x, UInt items,
           UInt blocks_per_group) noexcept {
            set_block_size(kBlock);
            Shared<uint> offsets{kBins};
            Shared<uint> flags{kBins * kWords};
            UInt tid = thread_x();
            UInt word = tid / 32u;
            UInt bit = 1u << (tid % 32u);
            UInt block = block_id().y * grid_x + block_id().x;
            UInt first = block * (items * kBlock);
            // where this block's digit `tid` starts in the output: the digit's
            // global base, plus the digit counts of the preceding blocks
            // (its own scan group first, then the groups before it)
            offsets[tid] = digit_base.read(tid) +
                           matrix.read(block * kBins + tid) +
                           groups.read((block / blocks_per_group) * kBins + tid);
            sync_block();
            $for (t, items) {
                $for (k, kWords) { flags[tid * kWords + k] = 0u; };
                sync_block();
                UInt index = first + t * kBlock + tid;
                Bool valid = index < count;
                UInt bin = def(0u);
                UInt bin_offset = def(0u);
                UInt code = def(0u);
                UInt slot = def(0u);
                $if (valid) {
                    auto key = keys_in.read(base + index);
                    code = key.code;
                    slot = key.slot;
                    bin = (code >> shift) & (kBins - 1u);
                    bin_offset = offsets[bin];
                    flags.atomic(bin * kWords + word).fetch_add(bit);
                };
                sync_block();
                $if (valid) {
                    UInt prefix = def(0u);
                    UInt total = def(0u);
                    $for (w, kWords) {
                        auto bits = flags[bin * kWords + w];
                        auto bin_count = popcount(bits);
                        total = total + bin_count;
                        prefix = prefix + select(0u, bin_count, w < word);
                        prefix = prefix + select(0u, popcount(bits & (bit - 1u)), w == word);
                    };
                    Var<LbvhKey> sorted;
                    sorted.code = code;
                    sorted.slot = slot;
                    keys_out.write(base + bin_offset + prefix, sorted);
                    $if (prefix == total - 1u) {
                        offsets.atomic(bin).fetch_add(total);
                    };
                };
                sync_block();
            };
        }});

    // ---- parallel: ranked stable scatter, one warp per sub-chunk ------------
    // Same ranking, but a whole warp sub-chunk (`items` elements per lane) is
    // ranked before the next barrier: the digit mask of a lane is its warp
    // ballot, built with one bitwise warp reduction per digit bit, and the
    // per-warp per-digit cursors are one shared array.  A lane's rank inside its
    // warp is therefore the number of preceding lanes of the same digit plus the
    // warp's running count, and the cross-warp order comes from the exclusive
    // scan of `_warp_hist` - still exactly the input order, so the output is
    // byte-identical to the reference.
    auto make_scatter_warp = [&device](uint items) noexcept {
        return device.compile(Kernel2D{
            [items](BufferVar<LbvhKey> keys_in, BufferVar<LbvhKey> keys_out,
                    BufferVar<uint> matrix, BufferVar<uint> groups, BufferVar<uint> digit_base,
                    UInt base, UInt count, UInt shift, UInt grid_x,
                    UInt blocks_per_group) noexcept {
                set_block_size(kBlock);
                set_warp_size(static_cast<uint8_t>(kWarpLanes));
                Shared<uint> warp_hist{kWarps * kWarpDigits};
                UInt tid = thread_x();
                UInt lane = warp_lane_id();
                UInt warp = tid / kWarpLanes;
                UInt block = block_id().y * grid_x + block_id().x;
                UInt first = block * (items * kBlock);
                // the warp's own, contiguous sub-chunk: warp `w` owns
                // [first + w * items * 32, first + (w + 1) * items * 32)
                UInt warp_first = first + warp * (items * kWarpLanes);
                $for (i, ceil_div(kWarps * kWarpDigits, kBlock)) {
                    auto slot = i * kBlock + tid;
                    $if (slot < kWarps * kWarpDigits) { warp_hist[slot] = 0u; };
                };
                sync_block();
                Local<uint> rank{items};
                // the keys are kept in registers: the scatter must not read the
                // block's range a second time (the ranking already paid for it),
                // which is what makes this variant bandwidth-competitive
                Local<uint> code_of{items};
                Local<uint> slot_of{items};
                $for (i, items) {
                    UInt index = warp_first + i * kWarpLanes + lane;
                    UInt code = def(0u);
                    UInt slot = def(0u);
                    // lanes past the end are ranked in the spare digit `kBins`
                    UInt bin = def(kBins);
                    $if (index < count) {
                        auto key = keys_in.read(base + index);
                        code = key.code;
                        slot = key.slot;
                        bin = (code >> shift) & (kBins - 1u);
                    };
                    code_of[i] = code;
                    slot_of[i] = slot;
                    // the lanes with the same digit: one warp-wide bit-or per
                    // digit bit, `matched` shrinks to exactly the equal digits
                    UInt matched = def(0xffffffffu);
                    $for (b, kRadixBits) {
                        auto x = (bin >> b) & 1u;
                        auto y = warp_active_bit_or(x << lane);
                        matched = matched & (y ^ select(0u, 0xffffffffu, x == 0u));
                    };
                    UInt prefix = popcount(matched & ((1u << lane) - 1u));
                    UInt total = popcount(matched);
                    auto warp_pre = warp_hist[warp * kWarpDigits + bin];
                    // the first lane of a digit publishes the warp's new cursor;
                    // every lane of the warp reads the old value above, which is
                    // what keeps the ranks deterministic
                    $if (prefix == 0u) { warp_hist[warp * kWarpDigits + bin] = warp_pre + total; };
                    rank[i] = warp_pre + prefix;
                };
                sync_block();
                // per-warp exclusive scan of the digit counts, offset by the
                // block's own output offset
                UInt running = def(0u);
                running = digit_base.read(tid) +
                          matrix.read(block * kBins + tid) +
                          groups.read((block / blocks_per_group) * kBins + tid);
                $for (w, kWarps) {
                    auto count_w = warp_hist[w * kWarpDigits + tid];
                    warp_hist[w * kWarpDigits + tid] = running;
                    running = running + count_w;
                };
                sync_block();
                $for (i, items) {
                    UInt index = warp_first + i * kWarpLanes + lane;
                    $if (index < count) {
                        auto code = code_of[i];
                        auto bin = (code >> shift) & (kBins - 1u);
                        auto pos = rank[i] + warp_hist[warp * kWarpDigits + bin];
                        Var<LbvhKey> key;
                        key.code = code;
                        key.slot = slot_of[i];
                        keys_out.write(base + pos, key);
                    };
                };
            }});
    };
    for (auto items = smallest_items; items <= largest_items; items *= 2u) {
        _scatter_warp.emplace_back(make_scatter_warp(items));
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

size_t LbvhRadixSort::scratch_bytes_for(size_t capacity) noexcept {
    auto blocks = maximum_blocks(capacity);
    auto groups = maximum_groups(blocks);
    return (blocks + groups + 1u) * kBins * sizeof(uint);
}

void LbvhRadixSort::set_items(uint items) noexcept {
    _items = snap_items(std::clamp(items, smallest_items, largest_items));
}

void LbvhRadixSort::set_variant(uint variant) noexcept {
    LUISA_ASSERT(variant < 2u, "unknown multi-block scatter variant {}.",
                 static_cast<int>(variant));
    _variant = variant;
}

void LbvhRadixSort::set_batched(bool batched) noexcept { _batched = batched; }

LbvhRadixSort::Method LbvhRadixSort::resolve_method(Method method, uint count) noexcept {
    switch (method) {
        case Method::single_block: return Method::single_block;
        case Method::multi_block: return Method::multi_block;
        case Method::automatic: break;
    }
    return count < automatic_min_count ? Method::single_block : Method::multi_block;
}

void LbvhRadixSort::encode_multi_block_pass(CommandList &commands,
                                            const Buffer<LbvhKey> &keys_in,
                                            const Buffer<LbvhKey> &keys_out,
                                            uint base, uint count, uint shift) noexcept {
    auto chunk = chunk_size();
    auto blocks = ceil_div(count, chunk);
    // a grid wider than the per-dimension limit of DirectX 12 is folded into a
    // second dimension
    auto grid_x = std::min(blocks, kMaxGroupsX);
    auto rows = ceil_div(blocks, grid_x);
    auto blocks_per_group = std::clamp(ceil_div(blocks, scan_group_limit), 1u, kScanWalk);
    auto groups = ceil_div(blocks, blocks_per_group);
    commands << _hist_pass(keys_in, _matrix, base, count, shift, grid_x, _items)
                    .dispatch(grid_x * kBlock, rows);
    commands << _scan_partial(_matrix, _groups, blocks, blocks_per_group)
                    .dispatch(groups * kBins);
    commands << _scan_groups(_groups, _digit_base, groups).dispatch(kBins);
    if (_variant == 0u) {
        commands << _scatter_tile(keys_in, keys_out, _matrix, _groups, _digit_base, base,
                                  count, shift, grid_x, _items, blocks_per_group)
                        .dispatch(grid_x * kBlock, rows);
    } else {
        auto variant = 0u;
        for (auto items = smallest_items; items < _items; items *= 2u) { variant++; }
        commands << _scatter_warp[variant](keys_in, keys_out, _matrix, _groups,
                                           _digit_base, base, count, shift, grid_x,
                                           blocks_per_group)
                        .dispatch(grid_x * kBlock, rows);
    }
}

void LbvhRadixSort::sort_passes(Stream &stream, const Buffer<LbvhKey> &keys_a,
                                const Buffer<LbvhKey> &keys_b, uint base, uint count,
                                uint pass_count, Method method) noexcept {
    LUISA_ASSERT(static_cast<size_t>(base) + count <= _capacity,
                 "radix sort range [{}, {}) exceeds the capacity {}.",
                 base, base + count, _capacity);
    LUISA_ASSERT(pass_count >= 1u && pass_count <= 4u,
                 "the sort has four passes, not {}.", pass_count);
    if (count <= 1u) { return; }
    LUISA_ASSERT(resolve_method(method, count) != Method::multi_block || chunk_size() > 0u,
                 "empty multi-block chunk.");
    auto resolved = resolve_method(method, count);
    auto commands = CommandList::create(4u * pass_count + 4u);
    // Recording the whole sort into one command list keeps it a *single*
    // submission (the four passes are one dependent chain anyway); with batching
    // turned off every pass is submitted on its own, which is what the library
    // used to do and what the benchmark measures as `single_unbatched`.
    const Buffer<LbvhKey> *in = &keys_a;
    const Buffer<LbvhKey> *out = &keys_b;
    for (auto pass = 0u; pass < pass_count; pass++) {
        auto shift = pass * kRadixBits;
        if (resolved == Method::single_block) {
            auto encoder = _single_pass(*in, *out, base, count, shift).dispatch(kBlock);
            if (_batched) {
                commands << std::move(encoder);
            } else {
                stream << std::move(encoder);
            }
        } else {
            encode_multi_block_pass(commands, *in, *out, base, count, shift);
        }
        std::swap(in, out);
    }
    if (_batched) { stream << commands.commit(); }
}

void LbvhRadixSort::sort(Stream &stream, const Buffer<LbvhKey> &keys_a,
                         const Buffer<LbvhKey> &keys_b, uint base, uint count,
                         Method method) noexcept {
    sort_passes(stream, keys_a, keys_b, base, count, 4u, method);
}

}// namespace luisa::example::lbvh
