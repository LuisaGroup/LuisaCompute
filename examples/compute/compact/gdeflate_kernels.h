// GDeflate GPU compression/decompression kernels written in LuisaCompute DSL.
//
// This is a self-contained port of the Microsoft DirectStorage GDeflate GPU
// decompression shader (D:/DirectStorage/GDeflate/shaders/GDeflate.hlsl) with
// the cross-lane primitives implemented via native warp/subgroup intrinsics
// (see luisa/dsl/builtin.h).  Each tile is decoded cooperatively by one 32-lane
// warp, so no shared memory scratch arrays are needed.
//
// For clarity and reliability this first version focuses on uncompressed
// DEFLATE blocks (BTYPE=00).  Compressed DEFLATE blocks (fixed/dynamic
// Huffman) can be added by porting the DecoderPair/SymbolTable code from the
// HLSL on top of the same BitReader and warp-primitive infrastructure provided
// here.

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
    params.out_size = select(kDefaultTileSize, c_tilestream_last_tile_size(ts), tile_idx < ts.num_tiles - 1u);

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
    UInt ballot = lane_vote(p);
    UInt offset = popcount(ballot & c_mask(tid)) * 4u;
    $if (p) {
        br.buf = br.buf | (static_cast<ULong>(input.read<uint>(br.base + offset)) << br.cnt);
        br.cnt = br.cnt + BitReader::k_width;
    };
    br.base = br.base + popcount(ballot) * 4u;
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
inline UInt uncompressed_block(BitReader &br, const ByteBufferVar &input, ByteBufferVar &output,
                               UInt tid, UInt dst, UInt size) noexcept {
    UInt nrounds = size / 32u;

    $while (nrounds != 0u) {
        UInt byte = bitreader_read(br, input, tid, 8u, true);
        // Each group of 4 lanes writes one 32-bit word.  All lanes participate in
        // the shuffles so the source lanes are active; the index is clamped so the
        // boundary lanes 29..31 never read an out-of-range lane.
        UInt b0 = byte;
        UInt b1 = warp_read_lane(byte, min(tid + 1u, 31u));
        UInt b2 = warp_read_lane(byte, min(tid + 2u, 31u));
        UInt b3 = warp_read_lane(byte, min(tid + 3u, 31u));
        $if ((tid & 3u) == 0u) {
            UInt word = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
            output.write(dst + (tid / 4u) * 4u, word);
        };
        dst = dst + 32u;
        nrounds = nrounds - 1u;
    };

    UInt rem = size % 32u;
    $if (rem != 0u) {
        // bitreader_read returns 0 for lanes >= rem, so byte is already zeroed.
        UInt byte = bitreader_read(br, input, tid, 8u, tid < rem);

        UInt full_words = rem / 4u;
        UInt partial = rem % 4u;

        UInt b0 = byte;
        UInt b1 = warp_read_lane(byte, min(tid + 1u, 31u));
        UInt b2 = warp_read_lane(byte, min(tid + 2u, 31u));
        UInt b3 = warp_read_lane(byte, min(tid + 3u, 31u));

        $if ((tid & 3u) == 0u & (tid / 4u) < full_words) {
            UInt word = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
            output.write(dst + (tid / 4u) * 4u, word);
        };

        // Last partial word (if any) is written by the first lane after the full words.
        $if (partial != 0u & tid == full_words * 4u) {
            UInt word = b0;
            $if (partial > 1u) { word = word | (b1 << 8u); };
            $if (partial > 2u) { word = word | (b2 << 16u); };
            $if (partial > 3u) { word = word | (b3 << 24u); };
            output.write(dst + full_words * 4u, word);
        };

        dst = dst + rem;
    };

    return dst;
}

// ============================================================================
// Tile decompression
// ============================================================================

inline void decompress_tile(const ByteBufferVar &input, ByteBufferVar &output,
                            UInt tid, UInt in_pos, UInt out_pos, UInt out_size) noexcept {
    BitReader br;
    bitreader_init(br, input, in_pos, tid);

    UInt dst = out_pos;
    // Clear output tile to avoid garbage around unaligned tail.
    UInt clear_words = (out_size + 3u) / 4u;
    UInt clear_iters = (clear_words + 31u) / 32u;
    $for (iter, clear_iters) {
        UInt idx = tid + iter * 32u;
        $if (idx < clear_words) {
            output.write(out_pos + idx * 4u, 0u);
        };
    };

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
            $case (1u) { // Fixed Huffman (not implemented in this first version)
            };
            $case (2u) { // Dynamic Huffman (not implemented in this first version)
            };
            $default { // Invalid block type
            };
        };
    };
}

// ============================================================================
// Decompression kernel entry point
// ============================================================================

inline auto make_decompress_kernel() noexcept {
        return [](ByteBufferVar input, ByteBufferVar output,
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
            decompress_tile(input, output, tid, params.in_pos, params.out_pos, params.out_size);
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

inline auto make_compress_kernel() noexcept {
    return [](ByteBufferVar input, ByteBufferVar output,
              UInt stream_in_pos, UInt stream_out_pos,
              UInt num_tiles, UInt last_tile_size) noexcept {
        // Dispatch num_tiles threads; each thread compresses one tile.  A larger
        // block size packs more independent tiles per block and reduces launch
        // overhead (total thread count is unchanged).
        set_block_size(128u, 1u, 1u);

        UInt tile_idx = dispatch_id().x;
        UInt full_tile_data_size = compressed_tile_data_size(kDefaultTileSize);

        $if (tile_idx == 0u) {
            write_tilestream_header(output, stream_out_pos, num_tiles, last_tile_size, full_tile_data_size);
        };

        $if (tile_idx < num_tiles) {
            UInt tile_size = select(last_tile_size, kDefaultTileSize, tile_idx < num_tiles - 1u);
            UInt in_pos = stream_in_pos + tile_idx * kDefaultTileSize;
            UInt data_start = stream_out_pos + kStreamHeaderSize + num_tiles * 4u;
            UInt out_pos = data_start + tile_idx * full_tile_data_size;
            compress_tile_single_thread(input, output, in_pos, out_pos, tile_size);
        };
    };
}

}// namespace luisa::example::gdeflate
