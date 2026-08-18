// GDeflate GPU compression/decompression kernels written in LuisaCompute DSL.
//
// This is a self-contained port of the Microsoft DirectStorage GDeflate GPU
// decompression shader (D:/DirectStorage/GDeflate/shaders/GDeflate.hlsl) with
// the cross-lane primitives implemented via native warp/subgroup intrinsics
// (see luisa/dsl/builtin.h).  Each tile is decoded cooperatively by one 32-lane
// warp, using per-warp slices of a shared scratch array for the code-length
// array (g_buf), the symbol table (g_lut) and the histogram scratch (g_tmp).
//
// All three DEFLATE block types are supported: uncompressed (BTYPE=00),
// fixed-Huffman (BTYPE=01) and dynamic-Huffman (BTYPE=10).  The fixed/dynamic
// paths are direct ports of the DecoderPair/SymbolTable code from the HLSL on
// top of the same BitReader and warp-primitive infrastructure provided here.

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/byte_buffer.h>

#include "gdeflate.h"

namespace luisa::example::gdeflate {

using namespace luisa;
using namespace luisa::compute;

// ============================================================================
// Low level bit primitives (HLSL -> Luisa DSL)
// ============================================================================

inline Callable<uint(uint)> c_mask = [](Var<uint> n) noexcept {
    return select(0xffffffffu, (1u << n) - 1u, n < 32u);
};

inline Callable<uint(uint, uint, uint, uint)> c_extract = [](Var<uint> data, Var<uint> pos, Var<uint> n, Var<uint> base) noexcept {
    return ((data >> pos) & c_mask(n)) + base;
};

// firstbitlow -> count trailing zeros; returns 32 for an empty mask.
inline Callable<uint(uint)> c_firstbitlow = [](Var<uint> m) noexcept {
    return select(32u, ctz(m), m != 0u);
};

// firstbithigh -> 31 - clz; returns 32 for an empty mask.
inline Callable<uint(uint)> c_firstbithigh = [](Var<uint> m) noexcept {
    return select(32u, 31u - clz(m), m != 0u);
};

// ============================================================================
// Cross-lane communication using warp intrinsics (native subgroup ops)
//
// The decompressor runs one 32-lane warp per tile (see make_decompress_kernel,
// which pins set_warp_size(32u)).  Every Shared<uint> scratch array used by the
// old portable fallback is replaced by the equivalent warp intrinsic:
//   lane_vote      -> warp_active_bit_mask(p).x   (WaveActiveBallot)
//   lane_broadcast -> warp_read_lane(value, src)  (WaveReadLaneAt)
//   lane_scan      -> warp_prefix_sum(value)      (WavePrefixSum, if needed)
// Warp collectives need no sync_block(): they are guaranteed to complete within
// the warp without barriers.
// ============================================================================

// Return a 32-bit ballot mask: bit i set iff predicate is true in lane i.
inline auto lane_vote(Bool p) noexcept -> UInt {
    return warp_active_bit_mask(p).x;
}

// Broadcast value from lane src_idx to all lanes.
inline auto lane_broadcast(Var<uint> value, Var<uint> src_idx) noexcept -> UInt {
    return warp_read_lane(value, src_idx);
}

// Exclusive prefix sum over the warp (HLSL WavePrefixSum).
inline auto lane_scan(UInt value) noexcept -> UInt {
    return warp_prefix_sum(value);
}

// Inclusive prefix sum within each 16-lane segment (HLSL scan16).
inline UInt scan16_inclusive(UInt value, UInt tid) noexcept {
    UInt incl = warp_prefix_sum(value) + value; // inclusive over all 32 lanes
    UInt base = warp_read_lane(incl, 15u);      // sum of lanes 0..15
    return select(incl, incl - base, tid >= 16u);
}

// Lane-match information for a 4-bit value (HLSL match()): for every lane,
// `below` is the number of lanes with a smaller index holding the same value,
// `first` is the index of the first such lane, and `cnt` is the group size.
// This replaces a 32-iteration warp_read_lane loop with 16 ballots (one per
// possible 4-bit length), which is significantly cheaper on every backend.
inline void c_lane_match_info(UInt value, UInt tid, UInt &below, UInt &first, UInt &cnt) noexcept {
    below = 0u;
    first = 32u;
    cnt = 0u;
    $for (L, 16u) {
        Bool eq = value == L;
        UInt m = lane_vote(eq); // all lanes execute the ballot uniformly
        $if (eq) {
            below = popcount(m & c_mask(tid));
            first = c_firstbitlow(m);
            cnt = popcount(m);
        };
    };
}

// ============================================================================
// GDeflate / DEFLATE64 constants (must match GDeflate.hlsl TranslateSymbol)
// ============================================================================

inline constexpr uint32_t kMaxSymbols = 320u;       // 288 litlen + 32 distance
inline constexpr uint32_t kDistanceCodesBase = 288u;

// Order in which the 19 precode code lengths are stored in the dynamic header.
inline Constant<uint> k_lane4id{3u, 17u, 15u, 13u, 11u, 9u, 7u, 5u, 4u, 6u, 8u, 10u, 12u, 14u, 16u, 18u,
                                0u, 1u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

// base/xlen for code-length repeat symbols 0..3 (HLSL base[4]/xlen[4]).
inline Constant<uint> k_base4{1u, 3u, 3u, 11u};
inline Constant<uint> k_xlen4{0u, 2u, 3u, 7u};

// DEFLATE64 length/distance tables (EXACTLY as in GDeflate.hlsl
// TranslateSymbol; do NOT replace with classic-DEFLATE tables).
inline Constant<uint> k_base_dist{1u, 2u, 3u, 4u, 5u, 7u, 9u, 13u, 17u, 25u, 33u, 49u, 65u, 97u, 129u, 193u,
                                  257u, 385u, 513u, 769u, 1025u, 1537u, 2049u, 3073u, 4097u, 6145u, 8193u, 12289u, 16385u, 24577u, 32769u, 49153u};
inline Constant<uint> k_extra_dist{0u, 0u, 0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u, 6u,
                                   7u, 7u, 8u, 8u, 9u, 9u, 10u, 10u, 11u, 11u, 12u, 12u, 13u, 13u, 14u, 14u};
inline Constant<uint> k_base_length{0u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 13u, 15u, 17u, 19u, 23u, 27u,
                                    31u, 35u, 43u, 51u, 59u, 67u, 83u, 99u, 115u, 131u, 163u, 195u, 227u, 3u, 0u};
inline Constant<uint> k_extra_length{0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u,
                                     2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 16u, 0u};

// ============================================================================
// Output byte helpers
//
// Expr<ByteBuffer> has no atomic operations, so the decompressor writes the
// output through a BufferVar<uint> and performs byte accesses with
// InterlockedOr-style word atomics, exactly like the HLSL StoreByte/ReadByte.
// ============================================================================

inline void store_byte(BufferVar<uint> &output, UInt offset, UInt data) noexcept {
    output.atomic(offset >> 2u).fetch_or((data & 0xffu) << ((offset & 3u) << 3u));
}

inline UInt read_output_byte(BufferVar<uint> &output, UInt offset) noexcept {
    return (output.read(offset >> 2u) >> ((offset & 3u) << 3u)) & 0xffu;
}

// ============================================================================
// Tile stream parser (matches tilestream.hlsl)
// ============================================================================

inline Callable<TileStream(ByteBuffer, uint)> c_tilestream_construct = [](ByteBufferVar input, Var<uint> stream_in_pos) noexcept {
    Var<TileStream> ts;
    ts.word1 = input.read<uint>(stream_in_pos);
    ts.word2 = input.read<uint>(stream_in_pos + 4u);
    ts.num_tiles = (ts.word1 >> 16u) & 0xffffu;
    return ts;
};

inline Callable<uint(TileStream)> c_tilestream_last_tile_size = [](Var<TileStream> ts) noexcept {
    UInt last = (ts.word2 >> 2u) & ((1u << 18u) - 1u);
    return select(kDefaultTileSize, last, last != 0u);
};

inline Callable<TileParams(ByteBuffer, uint, uint, uint, TileStream)> c_tilestream_get_params = [](ByteBufferVar input, Var<uint> stream_in_pos, Var<uint> stream_out_pos, Var<uint> tile_idx, Var<TileStream> ts) noexcept {
    Var<TileParams> params{};
    UInt tile_table_pos = stream_in_pos + kStreamHeaderSize;

    params.in_pos = select(0u, input.read<uint>(tile_table_pos + tile_idx * 4u), tile_idx > 0u);

    UInt next_pos = select(0u, input.read<uint>(tile_table_pos + (tile_idx + 1u) * 4u), tile_idx < ts.num_tiles - 1u);
    params.in_size = select(next_pos - params.in_pos, input.read<uint>(tile_table_pos), tile_idx < ts.num_tiles - 1u);

    params.out_pos = stream_out_pos + tile_idx * kDefaultTileSize;
    // Non-last tiles are full-size; the last tile uses the header's size.
    params.out_size = select(c_tilestream_last_tile_size(ts), kDefaultTileSize, tile_idx < ts.num_tiles - 1u);

    UInt data_start = tile_table_pos + ts.num_tiles * 4u;
    params.in_pos = params.in_pos + data_start;
    return params;
};

// ============================================================================
// Swizzled bit reader (matches GDeflate.hlsl BitReader)
// ============================================================================

struct BitReader {
    static constexpr uint k_width = 32u;
    UInt base;
    UInt cnt;
    ULong buf;
};

inline void bitreader_init(BitReader &br, const ByteBufferVar &input,
                           UInt tile_in_pos, UInt tid) noexcept {
    br.cnt = BitReader::k_width;
    UInt word = input.read<uint>(tile_in_pos + tid * 4u);
    br.buf = static_cast<ULong>(word);
    br.base = tile_in_pos + BitReader::k_width * 4u;
}

inline void bitreader_refill(BitReader &br, const ByteBufferVar &input,
                             UInt tid, Bool p) noexcept {
    p = p & (br.cnt < BitReader::k_width);
    // Exclusive prefix popcount over lanes is a single warp instruction and
    // replaces the ballot + popcount(ballot & mask(tid)) pair.
    UInt offset = warp_prefix_count_bits(p) * 4u;
    $if (p) {
        br.buf = br.buf | (static_cast<ULong>(input.read<uint>(br.base + offset)) << br.cnt);
        br.cnt = br.cnt + BitReader::k_width;
    };
    br.base = br.base + warp_active_count_bits(p) * 4u;
}

inline void bitreader_eat(BitReader &br, const ByteBufferVar &input,
                          UInt tid, UInt n, Bool p) noexcept {
    $if (p) {
        br.buf = br.buf >> n;
        br.cnt = br.cnt - n;
    };
    bitreader_refill(br, input, tid, p);
}

inline UInt bitreader_read(BitReader &br, const ByteBufferVar &input,
                           UInt tid, UInt n, Bool p) noexcept {
    UInt bits = select(0u, static_cast<UInt>(br.buf) & c_mask(n), p);
    bitreader_eat(br, input, tid, n, p);
    return bits;
}

inline UInt bitreader_peek(BitReader &br, UInt n) noexcept {
    return static_cast<UInt>(br.buf) & c_mask(n);
}

inline UInt bitreader_peek(BitReader &br) noexcept {
    return static_cast<UInt>(br.buf);
}

// ============================================================================
// DEFLATE block decoders
// ============================================================================

// Uncompressed DEFLATE block: copy LEN raw bytes from the bitstream.
//
// When the destination is 4-byte aligned, each 32-byte round covers whole
// words, so the round is written with 4 warp shuffles (assemble the word) plus
// 8 plain global stores instead of 32 atomic byte-ORs.  dst is warp-uniform
// and advances by 32 per round, so the alignment test is uniform and loop-
// invariant.  Unaligned destinations (e.g. after a compressed block) fall back
// to the atomic byte stores, which remain correct for any offset.
inline UInt uncompressed_block(BitReader &br, const ByteBufferVar &input, BufferVar<uint> &output,
                               UInt tid, UInt dst, UInt size) noexcept {
    UInt nrounds = size / 32u;
    Bool aligned = (dst & 3u) == 0u;

    $while (nrounds != 0u) {
        UInt byte = bitreader_read(br, input, tid, 8u, true);
        $if (aligned) {
            // Assemble the 4-byte word of this lane's 4-lane group, then have
            // the group leader issue one plain word store.
            UInt word = 0u;
            $for (j, 4u) {
                UInt src = (tid & ~3u) + j;
                word = word | (warp_read_lane(byte, src) << (j * 8u));
            };
            $if ((tid & 3u) == 0u) {
                output.write(dst / 4u + tid / 4u, word);
            };
        } $else {
            store_byte(output, dst + tid, byte);
        };
        dst = dst + 32u;
        nrounds = nrounds - 1u;
    };

    UInt rem = size % 32u;
    $if (rem != 0u) {
        // bitreader_read returns 0 for lanes >= rem, so byte is already zeroed.
        UInt byte = bitreader_read(br, input, tid, 8u, tid < rem);
        $if (tid < rem) {
            store_byte(output, dst + tid, byte);
        };
        dst = dst + rem;
    };

    return dst;
}

// ============================================================================
// Fast path: whole tile is one uncompressed DEFLATE block (BFINAL=1, BTYPE=00).
//
// For this layout the swizzle is the classic GDeflate one: packet p of stream
// s lives at word p*32+s.  Stream 0 carries BFINAL(1)+BTYPE(2)+LEN(16)=19 bits
// followed by bytes 0, 32, 64, ...; every other stream s carries bytes
// s, s+32, s+64, ...  Byte r*32+s (r-th round) sits at bit r*8 (+19 for s=0)
// of stream s.
//
// Each lane reads ONE word per 4 rounds (its own stream's packet) and unpacks
// 4 bytes from it, instead of the BitReader's per-round 64-bit refill scans.
// The 4 output bytes are written with the same 4-lane word-assembly used by
// uncompressed_block, so input traffic drops 4x while output stays coalesced.
// ============================================================================
inline void fast_uncompressed_tile(const ByteBufferVar &input, BufferVar<uint> &output,
                                   UInt in_pos, UInt out_pos, UInt len, UInt tid) noexcept {
    UInt nrounds = len / 32u;
    UInt qgroups = (nrounds + 3u) / 4u;
    $for (q, qgroups) {
        // Packet q of this lane's stream (stream == lane).  Lane 0 may need the
        // next packet too because its data is shifted by the 19-bit header.
        UInt word = input.read<uint>(in_pos + (q * 32u + tid) * 4u);
        $for (j, 4u) {
            UInt round = q * 4u + j;
            $if (round < nrounds) {
                UInt byte;
                $if (tid == 0u) {
                    // Stream 0's bytes start 19 bits into the stream, so they
                    // are NOT byte-aligned in the packet: bit = 19 + round*8.
                    // A byte can even straddle two packets (shift 27), so read
                    // the containing packet and, when needed, the next one.
                    UInt bit = 19u + round * 8u;
                    UInt p = bit >> 5u;
                    UInt shift = bit & 31u;
                    UInt w0 = input.read<uint>(in_pos + (p * 32u) * 4u);
                    byte = (w0 >> shift) & 0xffu;
                    $if (shift > 24u) {
                        UInt w1 = input.read<uint>(in_pos + ((p + 1u) * 32u) * 4u);
                        byte = byte | ((w1 << (32u - shift)) & 0xffu);
                    };
                } $else {
                    byte = (word >> (j * 8u)) & 0xffu;
                };
                // Assemble the 4-byte word of this lane's 4-lane group, then
                // have the group leader issue one plain word store.
                UInt word_out = 0u;
                $for (t, 4u) {
                    UInt src = (tid & ~3u) + t;
                    word_out = word_out | (warp_read_lane(byte, src) << (t * 8u));
                };
                $if ((tid & 3u) == 0u) {
                    output.write(out_pos / 4u + round * 8u + tid / 4u, word_out);
                };
            };
        };
    };
    // Remainder: fewer than 32 bytes left; one byte per lane, atomic store.
    UInt rem = len % 32u;
    $if (rem != 0u) {
        UInt b = nrounds * 32u + tid;
        $if (tid < rem) {
            UInt bit;
            UInt w;
            $if (tid == 0u) {
                bit = 19u + (b / 32u) * 8u;
                UInt p = bit >> 5u;
                UInt shift = bit & 31u;
                w = input.read<uint>(in_pos + (p * 32u) * 4u);
                UInt byte0 = (w >> shift) & 0xffu;
                $if (shift > 24u) {
                    UInt w1 = input.read<uint>(in_pos + ((p + 1u) * 32u) * 4u);
                    byte0 = byte0 | ((w1 << (32u - shift)) & 0xffu);
                };
                w = byte0;
            } $else {
                bit = (b / 32u) * 8u;
                UInt p = bit >> 5u;
                w = input.read<uint>(in_pos + (p * 32u + tid) * 4u);
                w = (w >> (bit & 31u)) & 0xffu;
            };
            store_byte(output, out_pos + b, w);
        };
    };
}

// ============================================================================
// Shared-memory scratch helpers (per-warp slices of one Shared<uint> array)
//
// Layout inside a 416-word slice:
//   g_tmp = slice + [0, 32)     histogram / running-offset scratch
//   g_buf = slice + [32, 96)    4-bit code lengths (64 words = 512 nibbles)
//   g_lut = slice + [96, 416)   symbol table (320 words)
//
// These are plain C++ helpers (NOT DSL Callables), so they receive the
// Shared<uint> by reference and the caller's per-warp `slice` base.
// ============================================================================

inline void c_scratch_clear(Shared<uint> &g_buf, UInt slice, UInt tid) noexcept {
    g_buf[slice + 32u + tid] = 0u;
    g_buf[slice + 32u + 32u + tid] = 0u;
}

inline UInt c_get4b(Shared<uint> &g_buf, UInt slice, UInt i) noexcept {
    return (g_buf[slice + 32u + (i / 8u)] >> ((i % 8u) * 4u)) & 15u;
}

inline void c_set4b(Shared<uint> &g_buf, UInt slice, UInt nibbles, UInt n, UInt i) noexcept {
    // Expand the nibble into 8 copies, keep the low n*4 bits (HLSL set4b).
    nibbles = nibbles | (nibbles << 4u);
    nibbles = nibbles | (nibbles << 8u);
    nibbles = nibbles | (nibbles << 16u);
    UInt nm = n * 4u;
    nibbles = nibbles & select(0xffffffffu, c_mask(nm), nm < 32u);
    UInt base = i / 8u;
    UInt shift = i % 8u;
    g_buf.atomic(slice + 32u + base).fetch_or(nibbles << (shift * 4u));
    $if (shift + n > 8u) {
        UInt rshift = (8u - shift) * 4u;
        g_buf.atomic(slice + 32u + base + 1u).fetch_or(select(0u, nibbles >> rshift, rshift < 32u));
    };
}

// Build a histogram of in-register code lengths.  Returns the count of length
// (tid & 15) in the low 16 lanes (HLSL GetHistogram).
inline UInt c_get_histogram(Shared<uint> &g_tmp, UInt slice, UInt cnt, UInt len, UInt tid) noexcept {
    g_tmp[slice + tid] = 0u;
    $if (len != 0u & tid < cnt) {
        g_tmp.atomic(slice + len).fetch_add(1u);
    };
    return g_tmp[slice + (tid & 15u)];
}

// Read the 19 precode code lengths (HLSL ReadLenCodes).
inline UInt c_read_len_codes(BitReader &br, const ByteBufferVar &input, UInt hclen, UInt tid) noexcept {
    UInt len = bitreader_read(br, input, tid, 3u, tid < hclen);
    len = warp_read_lane(len, k_lane4id[tid]);
    return select(0u, len & 15u, tid < 19u);
}

// Update the literal/length and distance histograms for a run of `n` code
// lengths starting at position `i` (HLSL UpdateHistograms; signed arithmetic).
inline void c_update_histograms(Shared<uint> &g_tmp, UInt slice, UInt len, UInt i, UInt n, UInt hlit) noexcept {
    Int cnt = max(min(static_cast<Int>(hlit) - static_cast<Int>(i), static_cast<Int>(n)), Int(0));
    $if (cnt != 0) {
        g_tmp.atomic(slice + len).fetch_add(static_cast<UInt>(cnt));
    };
    cnt = max(min(static_cast<Int>(i) + static_cast<Int>(n) - static_cast<Int>(hlit), static_cast<Int>(n)), Int(0));
    $if (cnt != 0) {
        g_tmp.atomic(slice + 16u + len).fetch_add(static_cast<UInt>(cnt));
    };
}

// In-register pair of canonical Huffman decoders (HLSL DecoderPair).  Each
// 32-lane register holds both the literal/length decoder (lanes 0..15) and the
// distance decoder (lanes 16..31).
struct DecoderPair {
    UInt base_codes;
    UInt offsets;

    UInt offset(UInt i) const noexcept {
        return warp_read_lane(offsets, i);
    }

    // Build both decoders in parallel from a histogram of code lengths.
    void init(UInt counts, UInt maxlen, UInt tid) noexcept {
        offsets = scan16_inclusive(counts, tid);
        UInt lane = tid & 15u;
        UInt base_code = 0u;
        $for (i, 1u, maxlen) {
            UInt count = warp_read_lane(counts, (tid & 16u) + i);
            $if (lane >= i) {
                base_code = base_code + (count << (lane - i));
            };
        };
        // Left-align and fill in sentinel values (avoid shift-by-32 for lane 0).
        UInt tmp = select(0u, base_code << (32u - lane), lane != 0u);
        base_codes = select(0xffffffffu, tmp, !(tmp < base_code | lane >= maxlen));
    }

    // Maps a code to its length (base selects the decoder: 0 or 16).  The
    // warp_read_lane calls are kept outside the per-lane $if branches so every
    // lane executes them uniformly (only the lane index varies per lane).
    UInt len4code(UInt code, UInt base) const noexcept {
        UInt len = 1u;
        UInt b7 = warp_read_lane(base_codes, 7u + base);
        $if (code >= b7) { len = 8u; };
        UInt b3 = warp_read_lane(base_codes, len + 3u + base);
        $if (code >= b3) { len = len + 4u; };
        UInt b1 = warp_read_lane(base_codes, len + 1u + base);
        $if (code >= b1) { len = len + 2u; };
        UInt b0 = warp_read_lane(base_codes, len + base);
        $if (code >= b0) { len = len + 1u; };
        return len;
    }

    // Maps a code and its length to a symbol-table offset (base 0 or 16).
    UInt id4code(UInt code, UInt len, UInt base) const noexcept {
        UInt i = len + base - 1u;
        return warp_read_lane(offsets, i) + ((code - warp_read_lane(base_codes, i)) >> (32u - len));
    }

    // Decode one Huffman symbol from the left-aligned reversed code bits.
    UInt decode(Shared<uint> &g_lut, UInt slice, UInt bits, UInt &len_out, Bool isdist) const noexcept {
        UInt code = reverse(bits);
        UInt base = select(0u, 16u, isdist);
        len_out = len4code(code, base);
        return g_lut[slice + 96u + id4code(code, len_out, base) + select(0u, kDistanceCodesBase, isdist)];
    }
};

// Build the symbol table from the code-length array (HLSL SymbolTable::init).
inline void c_symbol_table_init(Shared<uint> &g_tmp, Shared<uint> &g_buf, Shared<uint> &g_lut,
                                UInt slice, UInt hlit, UInt offsets, UInt tid) noexcept {
    // g_tmp[tid + 1] = offsets (leaves g_tmp[0] and g_tmp[32] untouched).
    $if (tid != 15u & tid != 31u) {
        g_tmp[slice + tid + 1u] = offsets;
    };
    // 8 unconditional iterations scatter literals 0..255.
    $for (i, 8u) {
        UInt sym = i * 32u + tid;
        UInt len = c_get4b(g_buf, slice, sym);
        UInt below, first, cnt;
        c_lane_match_info(len, tid, below, first, cnt);
        $if (len != 0u) {
            g_lut[slice + 96u + g_tmp[slice + len] + below] = sym;
        };
        // First lane of each equal-length group advances the running offset.
        $if (tid == first) {
            g_tmp[slice + len] = g_tmp[slice + len] + cnt;
        };
    };
    // Bounds-checked last literal iteration (symbols 256..287).
    UInt sym = 8u * 32u + tid;
    UInt len = select(0u, c_get4b(g_buf, slice, sym), sym < hlit);
    {
        UInt below, first, cnt;
        c_lane_match_info(len, tid, below, first, cnt);
        $if (len != 0u) {
            g_lut[slice + 96u + g_tmp[slice + len] + below] = sym;
        };
    }
    // Distance codes (symbols 288..319 in the LUT).
    len = c_get4b(g_buf, slice, tid + hlit);
    {
        UInt below, first, cnt;
        c_lane_match_info(len, tid, below, first, cnt);
        $if (len != 0u) {
            g_lut[slice + 96u + kDistanceCodesBase + g_tmp[slice + 16u + len] + below] = tid;
        };
    }
}

// Initialize the fixed-Huffman code lengths and return their histogram.
inline UInt c_fixed_code_lengths(Shared<uint> &g_buf, UInt slice, UInt tid) noexcept {
    g_buf[slice + 32u + tid] = select(0x99999999u, 0x88888888u, tid < 18u);
    g_buf[slice + 64u + tid] = select(0x55555555u, select(0x88888888u, 0x77777777u, tid < 3u), tid < 4u);
    // Return the histogram: lane7->24, lane8->152, lane9->112, lane21->32.
    UInt counts = 0u;
    $if (tid == 7u) { counts = 24u; };
    $if (tid == 8u) { counts = 152u; };
    $if (tid == 9u) { counts = 112u; };
    $if (tid == 21u) { counts = 32u; };
    return counts;
}

// Unpack the dynamic-Huffman code lengths into g_buf and return a histogram of
// literal/length lengths (lanes 0..15) and distance lengths (lanes 16..31).
inline UInt c_unpack_code_lengths(BitReader &br, const ByteBufferVar &input,
                                  Shared<uint> &g_tmp, Shared<uint> &g_buf, Shared<uint> &g_lut,
                                  UInt slice, UInt hlit, UInt hdist, UInt hclen, UInt tid, UInt dst) noexcept {
    UInt len = c_read_len_codes(br, input, hclen, tid);
    UInt cnts = c_get_histogram(g_tmp, slice, 19u, len, tid);
    DecoderPair dec;
    dec.init(cnts, 7u, tid);
    // Scatter the precode symbols.
    {
        UInt below, first, cnt;
        c_lane_match_info(len, tid, below, first, cnt);
        $if (len != 0u) {
            g_lut[slice + 96u + dec.offset(len - 1u) + below] = tid;
        };
    }
    c_scratch_clear(g_buf, slice, tid);
    g_tmp[slice + tid] = 0u;
    UInt count = hlit + hdist;
    UInt base_offset = 0u;
    UInt lastlen = 0xFFFFFFFFu;
    $loop {
        UInt bits = bitreader_peek(br, 14u);
        UInt sym_len;
        UInt sym = dec.decode(g_lut, slice, bits, sym_len, false);
        UInt idx = select(0u, sym - 15u, sym > 15u);
        UInt n = k_base4[idx] + ((bits >> sym_len) & c_mask(k_xlen4[idx]));
        // Scan back for the nearest lane holding a valid (non-16) symbol.
        UInt lane = c_firstbithigh(lane_vote(sym != 16u) & c_mask(tid));
        UInt codelen = sym;
        $if (sym > 16u) { codelen = 0u; };
        // c_firstbithigh returns 32 for an empty mask; never pass it to
        // warp_read_lane.
        UInt safe_lane = select(0u, lane, lane != 32u);
        UInt prevlen = warp_read_lane(codelen, safe_lane);
        $if (sym == 16u) {
            $if (lane == 32u) {
                codelen = lastlen;
            } $else {
                codelen = prevlen;
            };
        };
        lastlen = lane_broadcast(codelen, 31u);
        base_offset = lane_scan(n) + base_offset;
        $if (base_offset < count & codelen != 0u) {
            c_update_histograms(g_tmp, slice, codelen, base_offset, n, hlit);
            c_set4b(g_buf, slice, codelen, n, base_offset);
        };
        bitreader_eat(br, input, tid, sym_len + select(0u, k_xlen4[idx], idx != 0u), base_offset < count);
        base_offset = lane_broadcast(base_offset + n, 31u);
        $if (!warp_active_all(base_offset < count)) { $break; };
    };
    return g_tmp[slice + tid];
}

// Translate a decoded symbol into its value (literal length, match length or
// distance) and consume the extra bits (HLSL TranslateSymbol).
inline UInt c_translate_symbol(BitReader &br, const ByteBufferVar &input,
                               UInt sym, UInt sym_len, UInt bits, Bool isdist, UInt tid, Bool p) noexcept {
    UInt base = 1u;
    UInt n = 0u;
    $if (isdist) {
        base = k_base_dist[sym];
        n = k_extra_dist[sym];
    } $else {
        $if (sym >= 256u) {
            base = k_base_length[sym - 256u];
            n = k_extra_length[sym - 256u];
        } $else {
            base = 1u;
            n = 0u;
        };
    };
    bitreader_eat(br, input, tid, sym_len + n, isdist | p);
    return base + ((bits >> sym_len) & c_mask(n));
}

// Write the current round's outputs (HLSL WriteOutput): one byte per literal
// lane and the full copy run for every copy lane, using all 32 lanes.
inline void c_write_output(BufferVar<uint> &output, UInt dst, UInt offset, UInt dist, UInt length,
                           UInt byte, Bool iscopy, UInt tid) noexcept {
    dst = dst + offset;
    // Output literals.
    $if (!iscopy & length != 0u) {
        store_byte(output, dst, byte);
    };
    // Fill in copy destinations, one copy (lane) at a time, all lanes together.
    UInt mask = lane_vote(iscopy);
    $while (mask != 0u) {
        UInt lane = c_firstbitlow(mask);
        UInt off = max(lane_broadcast(dist, lane), 1u); // dist 0 is invalid; avoid % 0
        UInt len = lane_broadcast(length, lane);
        UInt out = lane_broadcast(dst, lane);
        $for (i, tid, len, 32u) {
            UInt data = read_output_byte(output, out + (i % off) - off);
            store_byte(output, i + out, data);
        };
        mask = mask & (mask - 1u);
    };
}

// Decode one compressed (fixed or dynamic Huffman) block; returns the updated
// destination byte offset (HLSL CompressedBlock).
inline UInt c_compressed_block(BitReader &br, const ByteBufferVar &input, BufferVar<uint> &output,
                               Shared<uint> &g_tmp, Shared<uint> &g_buf, Shared<uint> &g_lut,
                               UInt slice, UInt hlit, UInt counts, UInt dst, UInt tid) noexcept {
    DecoderPair dec;
    dec.init(counts, 15u, tid);
    c_symbol_table_init(g_tmp, g_buf, g_lut, slice, hlit, dec.offsets, tid);

    // Initial round - no copy processing yet.
    UInt sym_len;
    UInt sym = dec.decode(g_lut, slice, bitreader_peek(br, 31u), sym_len, false);
    UInt eob = lane_vote(sym == 256u);
    Bool oob = (eob & c_mask(tid)) != 0u;
    UInt value = c_translate_symbol(br, input, sym, sym_len, bitreader_peek(br), false, tid, !oob);
    UInt length = select(0u, value, !oob);
    Bool iscopy = sym > 256u;
    UInt byte = sym;
    UInt offset = lane_scan(length);

    $while (eob == 0u) {
        sym = dec.decode(g_lut, slice, bitreader_peek(br, 31u), sym_len, iscopy);
        eob = lane_vote(sym == 256u);
        oob = (eob & c_mask(tid)) != 0u;
        value = c_translate_symbol(br, input, sym, sym_len, bitreader_peek(br), iscopy, tid, !oob);
        c_write_output(output, dst, offset, value, length, byte, iscopy, tid);
        dst = dst + lane_broadcast(offset + length, 31u);
        length = select(0u, value, !(iscopy | oob));
        offset = lane_scan(length);
        iscopy = sym > 256u;
        byte = sym;
    };

    // One last round of copy processing.
    sym = dec.decode(g_lut, slice, bitreader_peek(br, 31u), sym_len, true);
    iscopy = iscopy & !oob;
    UInt dist = c_translate_symbol(br, input, sym, sym_len, bitreader_peek(br), iscopy, tid, false);
    c_write_output(output, dst, offset, dist, length, byte, iscopy, tid);
    return dst + lane_broadcast(offset + length, 31u);
}

// ============================================================================
// Tile decompression
// ============================================================================

inline void decompress_tile(const ByteBufferVar &input, BufferVar<uint> &output,
                            UInt tid, UInt warp_in_block, UInt in_pos, UInt out_pos, UInt out_size) noexcept {
    // One 32-lane warp decodes one tile.  Each warp owns a private slice of the
    // shared scratch array (416 words = 32 g_tmp + 64 g_buf + 320 g_lut).  The
    // warps are independent and may diverge, so no sync_block() may be used.
    Shared<uint> scratch{4u * (32u + 64u + 320u)};
    UInt slice = warp_in_block * (32u + 64u + 320u);

    // Fast path: the tile is a single uncompressed DEFLATE block.  Random and
    // otherwise incompressible tiles hit this branch every time; it bypasses the
    // BitReader entirely (direct p*32+s packet addressing, 4x fewer input reads
    // and no per-round warp refill scans).  Stream 0 packet 0 holds the block
    // header, so BFINAL/BTYPE/LEN are read straight from the first word.
    UInt word0 = input.read<uint>(in_pos);
    $if (((word0 & 1u) != 0u) & ((word0 >> 1u) & 3u) == 0u) {
        UInt len = min((word0 >> 3u) & 0xffffu, out_size);
        // The round loop overwrites whole words, but the tail bytes are written
        // with byte-OR atomics, so zero the tail words (and any words beyond
        // LEN for invalid streams) first.  For a full tile this is skipped
        // entirely, avoiding a second full write pass over the output.
        UInt rem = len % 32u;
        $if (rem != 0u | len < out_size) {
            UInt start_word = (len / 32u) * 8u;
            UInt clear_words = (out_size + 3u) / 4u;
            UInt clear_iters = (clear_words - start_word + 31u) / 32u;
            $for (iter, clear_iters) {
                UInt idx = start_word + tid + iter * 32u;
                $if (idx < clear_words) {
                    output.write(out_pos / 4u + idx, 0u);
                };
            };
        };
        fast_uncompressed_tile(input, output, in_pos, out_pos, len, tid);
    } $else {
        // General path (multi-block tiles, Huffman blocks): clear the whole
        // output tile first to avoid garbage around unaligned tails.
        UInt clear_words = (out_size + 3u) / 4u;
        UInt clear_iters = (clear_words + 31u) / 32u;
        $for (iter, clear_iters) {
            UInt idx = tid + iter * 32u;
            $if (idx < clear_words) {
                output.write(out_pos / 4u + idx, 0u);
            };
        };

        BitReader br;
        bitreader_init(br, input, in_pos, tid);
        UInt dst = out_pos;

        Bool done = false;
        $while (!done) {
            // Read block header from lane 0 and broadcast.
            UInt header = lane_broadcast(bitreader_peek(br), 0u);

            done = c_extract(header, 0u, 1u, 0u) != 0u;
            UInt btype = c_extract(header, 1u, 2u, 0u);

            bitreader_eat(br, input, tid, 3u, tid == 0u);

            $switch (btype) {
                $case (0u) { // Uncompressed block (GDeflate omits the NLEN field)
                    UInt len = lane_broadcast(bitreader_read(br, input, tid, 16u, tid == 0u), 0u);
                    dst = uncompressed_block(br, input, output, tid, dst, len);
                };
                $case (1u) { // Fixed Huffman
                    UInt counts = c_fixed_code_lengths(scratch, slice, tid);
                    dst = c_compressed_block(br, input, output, scratch, scratch, scratch, slice, 288u, counts, dst, tid);
                };
                $case (2u) { // Dynamic Huffman
                    UInt hlit = c_extract(header, 3u, 5u, 257u);
                    UInt hdist = c_extract(header, 8u, 5u, 1u);
                    bitreader_eat(br, input, tid, 14u, tid == 0u);
                    UInt counts = c_unpack_code_lengths(br, input, scratch, scratch, scratch, slice, hlit, hdist,
                                                        c_extract(header, 13u, 4u, 4u), tid, dst);
                    dst = c_compressed_block(br, input, output, scratch, scratch, scratch, slice, hlit, counts, dst, tid);
                };
                $default { // Invalid block type
                };
            };
        };
    };
}

// ============================================================================
// Decompression kernel entry point
// ============================================================================

inline auto make_decompress_kernel() noexcept {
        return [](ByteBufferVar input, BufferVar<uint> output,
                  UInt stream_in_pos, UInt stream_out_pos, UInt num_tiles) noexcept {
        // One 32-lane warp decompresses one tile.  128 threads = 4 warps, so each
        // block handles 4 tiles.  Pin the warp size to 32 so this mapping is exact
        // on every backend (GDeflate's bit-stream layout assumes 32 lanes/tile).
        constexpr uint kBlockThreads = 128u;
        constexpr uint kWarpThreads = 32u;
        constexpr uint kWarpsPerBlock = kBlockThreads / kWarpThreads; // 4
        set_block_size(kBlockThreads, 1u, 1u);
        set_warp_size(static_cast<uint8_t>(kWarpThreads));

        UInt tid = warp_lane_id();                 // lane within the warp: 0..31
        UInt warp_in_block = thread_x() / warp_lane_count(); // 0..kWarpsPerBlock-1
        UInt tile_idx = block_id().x * kWarpsPerBlock + warp_in_block;

        $if (tile_idx < num_tiles) {
            Var<TileStream> ts = c_tilestream_construct(input, stream_in_pos);
            Var<TileParams> params = c_tilestream_get_params(input, stream_in_pos, stream_out_pos, tile_idx, ts);
            decompress_tile(input, output, tid, warp_in_block, params.in_pos, params.out_pos, params.out_size);
        };
    };
}

// ============================================================================
// Compression kernel helpers
//
// The compressor emits a GDeflate-compatible tile stream where every tile is
// stored as an uncompressed DEFLATE block.  This is a valid GDeflate stream
// and exercises the same BitReader path used by real compressed tiles.
//
// The bit assignment follows libdeflate: groups of DEFLATE bits are assigned
// round-robin to the 32 streams.  For an uncompressed block:
//   stream 0 : BFINAL(1) + BTYPE(2) + LEN(16) + byte 0 + byte 32 + byte 64 + ...
//   stream k : byte k + byte (k+32) + byte (k+64) + ...   (k >= 1)
//
// Each stream accumulates bits in a 64-bit buffer and flushes 32-bit packets
// to the output when the buffer fills.  This single-thread-per-tile compressor
// simulates that flush scheduling exactly.
// ============================================================================

// Read one input byte from the tile.
inline UInt read_input_byte(const ByteBufferVar &input, UInt in_pos, UInt byte_idx) noexcept {
    UInt word_off = (byte_idx / 4u) * 4u;
    UInt shift = (byte_idx % 4u) * 8u;
    return (input.read<uint>(in_pos + word_off) >> shift) & 0xffu;
}

// Worst-case compressed data size for a tile of `tile_size` bytes.
inline UInt compressed_tile_data_size(UInt tile_size) noexcept {
    // 32 initial packets (one per stream) + ceil(total_bits / 32) - 32 refill packets.
    UInt total_bits = 40u + tile_size * 8u;
    UInt num_words = (total_bits + 31u) / 32u;
    return max(128u, num_words * 4u);
}

// Write the 8-byte TileStream header and tile pointer table on thread 0.
inline void write_tilestream_header(ByteBufferVar &output, UInt out_pos,
                                    UInt num_tiles, UInt last_tile_size,
                                    UInt full_tile_data_size) noexcept {
    // Header word1: id=4, magic=0xFB, numTiles
    UInt word1 = (4u) | (0xFBu << 8u) | (num_tiles << 16u);
    // Header word2: lastTileSize in bits [2..19]
    UInt word2 = last_tile_size << 2u;
    output.write(out_pos, word1);
    output.write(out_pos + 4u, word2);

    UInt table_pos = out_pos + kStreamHeaderSize;
    UInt last_tile_data_size = compressed_tile_data_size(last_tile_size);
    output.write(table_pos, last_tile_data_size);
    $for (i, 1u, num_tiles) {
        output.write(table_pos + i * 4u, i * full_tile_data_size);
    };
}

// Compress one tile using a single thread.  This faithfully reproduces the
// libdeflate GDeflate flush order so the HLSL-style BitReader can decode it.
inline void compress_tile_single_thread(const ByteBufferVar &input, ByteBufferVar &output,
                                        UInt in_pos, UInt out_pos, UInt tile_size) noexcept {
    $array<uint, 32u> write_ptr;
    $array<uint, 32u> next_ptr;
    $array<ulong, 32u> bitbuf;
    $array<uint, 32u> bitcount;
    $array<uint, 32u> remaining_bits;

    // Count how many bits each stream will carry so we only reserve a packet
    // when the stream actually has more bits to flush later.
    $for (s, 32u) { remaining_bits[s] = 0u; };
    UInt rem_count = tile_size;
    $while (rem_count != 0u) {
        UInt block_len = min(rem_count, 65535u);
        remaining_bits[0u] = remaining_bits[0u] + 19u; // BFINAL + BTYPE + LEN
        UInt b = 0u;
        $while (b < block_len) {
            UInt s = b % 32u;
            remaining_bits[s] = remaining_bits[s] + 8u;
            b = b + 1u;
        };
        rem_count = rem_count - block_len;
    };

    UInt next = out_pos + 128u;
    $for (s, 32u) {
        write_ptr[s] = out_pos + s * 4u;
        next_ptr[s] = 0xFFFFFFFFu;
        bitbuf[s] = 0u;
        bitcount[s] = 0u;
        output.write(out_pos + s * 4u, 0u);
    };

    UInt idx = 0u;

    // Helper lambda: add n bits from `bits` to the current stream.
    auto add_bits = [&](UInt bits, UInt n) noexcept {
        // Reserve the next packet only if this add triggers a flush and more
        // bits (leftover from this add or future adds) will be written to it.
        $if (bitcount[idx] + n >= 32u & next_ptr[idx] == 0xFFFFFFFFu) {
            UInt after_flush_bits = bitcount[idx] + n - 32u;
            UInt after_add_remaining = remaining_bits[idx] - n;
            $if (after_flush_bits != 0u | after_add_remaining != 0u) {
                next_ptr[idx] = next;
                next = next + 4u;
            };
        };

        bitbuf[idx] = bitbuf[idx] | (static_cast<ULong>(bits) << bitcount[idx]);
        bitcount[idx] = bitcount[idx] + n;
        remaining_bits[idx] = remaining_bits[idx] - n;

        // Flush a full 32-bit packet if available.
        $if (bitcount[idx] >= 32u) {
            output.write(write_ptr[idx], static_cast<UInt>(bitbuf[idx]));
            bitbuf[idx] = bitbuf[idx] >> 32u;
            bitcount[idx] = bitcount[idx] - 32u;
            write_ptr[idx] = next_ptr[idx];
            next_ptr[idx] = 0xFFFFFFFFu;
        };
    };

    auto advance = [&]() noexcept {
        idx = (idx + 1u) % 32u;
    };

    // DEFLATE uncompressed blocks are limited to 65535 bytes, so tiles larger
    // than that are split into multiple uncompressed blocks.  Each new block
    // resets the stream index so its header (BFINAL/BTYPE/LEN) starts on stream 0.
    UInt remaining = tile_size;
    $while (remaining != 0u) {
        UInt block_len = min(remaining, 65535u);
        Bool is_last = remaining == block_len;

        // Block header.
        add_bits(select(0u, 1u, is_last), 1u); // BFINAL (1 on the last block)
        add_bits(0u, 2u);                       // BTYPE=00
        add_bits(block_len, 16u);               // LEN

        // Data bytes for this block.
        UInt byte_idx = tile_size - remaining;
        UInt block_end = byte_idx + block_len;
        $while (byte_idx < block_end) {
            UInt b = read_input_byte(input, in_pos, byte_idx);
            add_bits(b, 8u);
            advance();
            byte_idx = byte_idx + 1u;
        };

        remaining = remaining - block_len;
        idx = 0u; // next block header starts on stream 0
    };

    // Flush any remaining bits.
    $for (s, 32u) {
        $if (bitcount[s] != 0u) {
            output.write(write_ptr[s], static_cast<UInt>(bitbuf[s]));
        };
    };
}

// Parallel compressor for a single uncompressed-block tile (tile_size <= 65535,
// which is always true for kDefaultTileSize = 32768).  The 32 GDeflate streams
// are independent, so one 32-lane warp compresses one tile with one lane per
// stream.  Stream 0 additionally packs the 19-bit block header; its bytes are
// bit-packed at 8-bit intervals after the header, so lane 0 uses a 64-bit
// accumulator.  All other streams are byte-aligned and pack 4 bytes per word.
// The output layout is the classic packet p of stream s at word p*32+s, which
// is exactly what fast_uncompressed_tile and the BitReader expect.
inline void compress_tile_warp(const ByteBufferVar &input, ByteBufferVar &output,
                               UInt in_pos, UInt out_pos, UInt tile_size, UInt tid) noexcept {
    UInt nbytes = (tile_size + 31u) / 32u; // bytes per stream (ceil)
    $if (tid == 0u) {
        // Stream 0: BFINAL=1, BTYPE=00, LEN(tile_size) = 19 header bits, then
        // bytes 0, 32, 64, ... at 8-bit intervals.
        ULong acc = static_cast<ULong>(1u | (tile_size << 3u));
        UInt nbits = 19u;
        UInt p = 0u;
        $for (k, nbytes) {
            UInt b = read_input_byte(input, in_pos, k * 32u);
            acc = acc | (static_cast<ULong>(b) << nbits);
            nbits = nbits + 8u;
            $if (nbits >= 32u) {
                output.write(out_pos + p * 128u, static_cast<UInt>(acc));
                acc = acc >> 32u;
                nbits = nbits - 32u;
                p = p + 1u;
            };
        };
        $if (nbits != 0u) {
            output.write(out_pos + p * 128u, static_cast<UInt>(acc));
        };
    } $else {
        // Streams 1..31: bytes s, s+32, ... packed 4 per word.
        UInt nwords = (nbytes + 3u) / 4u;
        $for (p, nwords) {
            UInt w = 0u;
            $for (j, 4u) {
                UInt k = p * 4u + j;
                $if (k < nbytes) {
                    UInt byte_idx = k * 32u + tid;
                    $if (byte_idx < tile_size) {
                        UInt b = read_input_byte(input, in_pos, byte_idx);
                        w = w | (b << (j * 8u));
                    };
                };
            };
            output.write(out_pos + (p * 32u + tid) * 4u, w);
        };
    };
}

inline auto make_compress_kernel() noexcept {
    return [](ByteBufferVar input, ByteBufferVar output,
              UInt stream_in_pos, UInt stream_out_pos,
              UInt num_tiles, UInt last_tile_size) noexcept {
        // Dispatch 32 threads per tile; each warp compresses one tile, one lane
        // per GDeflate bit-stream.  A larger block size packs more tiles per
        // block and reduces launch overhead (total thread count is unchanged).
        set_block_size(128u, 1u, 1u);

        UInt tile_idx = dispatch_id().x / 32u;
        UInt tid = dispatch_id().x % 32u;
        UInt full_tile_data_size = compressed_tile_data_size(kDefaultTileSize);

        $if (tile_idx == 0u & tid == 0u) {
            write_tilestream_header(output, stream_out_pos, num_tiles, last_tile_size, full_tile_data_size);
        };

        $if (tile_idx < num_tiles) {
            UInt tile_size = select(last_tile_size, kDefaultTileSize, tile_idx < num_tiles - 1u);
            UInt in_pos = stream_in_pos + tile_idx * kDefaultTileSize;
            UInt data_start = stream_out_pos + kStreamHeaderSize + num_tiles * 4u;
            UInt out_pos = data_start + tile_idx * full_tile_data_size;
            $if (tile_size <= 65535u) {
                compress_tile_warp(input, output, in_pos, out_pos, tile_size, tid);
            } $else {
                // Tiles larger than one DEFLATE uncompressed block need the
                // multi-block single-thread writer.
                $if (tid == 0u) {
                    compress_tile_single_thread(input, output, in_pos, out_pos, tile_size);
                };
            };
        };
    };
}

}// namespace luisa::example::gdeflate
