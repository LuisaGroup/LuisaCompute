// GDeflate GPU compression/decompression example for LuisaCompute.
//
// This header defines the common tile-stream layout and a small host-side
// runtime class that drives the DSL compression/decompression kernels.
//
// The on-disk layout is compatible with the Microsoft DirectStorage GDeflate
// tile stream (see D:/DirectStorage/GDeflate/shaders/tilestream.hlsl):
//
//   [TileStream header]        8 bytes
//   [Tile pointer table]       numTiles * 4 bytes
//   [Tile data]                concatenated compressed tiles
//
// Each tile is a GDeflate-swizzled DEFLATE stream.  32 GPU lanes cooperate to
// decode one tile; bits are interleaved across the 32 sub-streams so that each
// lane owns every 32nd bit.

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/byte_buffer.h>

#include <algorithm>
#include <cstdint>

// DSL-visible structs must be registered at global scope.
struct TileStream {
    uint32_t word1;
    uint32_t word2;
    uint32_t num_tiles;
};
LUISA_STRUCT(TileStream, word1, word2, num_tiles) {};

struct TileParams {
    uint32_t in_pos;
    uint32_t in_size;
    uint32_t out_pos;
    uint32_t out_size;
};
LUISA_STRUCT(TileParams, in_pos, in_size, out_pos, out_size) {};

namespace luisa::example::gdeflate {

using namespace luisa;
using namespace luisa::compute;

// GDeflate constants.
inline constexpr uint32_t kDefaultTileSize = 32u * 1024u; // 32 KiB per tile (fits in one DEFLATE uncompressed block)
inline constexpr uint32_t kMaxTiles = 65535u;
inline constexpr uint32_t kStreamHeaderSize = 8u;
inline constexpr uint32_t kNumBitstreams = 32u;

// Host-side helper: build a tile stream header on the CPU (only metadata,
// pointer table is filled by the GPU compressor).
inline uint32_t compute_num_tiles(uint64_t uncompressed_size) noexcept {
    return static_cast<uint32_t>((uncompressed_size + kDefaultTileSize - 1u) / kDefaultTileSize);
}

// Worst-case compressed size for an uncompressed input.  The compressor emits
// uncompressed DEFLATE blocks, so each 64 KiB tile becomes at most
// ceil((40 + tile_size*8) / 1024) * 128 bytes plus stream metadata.
inline uint32_t compress_bound(uint64_t uncompressed_size) noexcept {
    uint32_t num_tiles = compute_num_tiles(uncompressed_size);
    uint64_t full_tile_bits = 40u + static_cast<uint64_t>(kDefaultTileSize) * 8u;
    uint64_t full_tile_data_size = ((full_tile_bits + 1023u) / 1024u) * 128u;
    uint64_t data_bound = num_tiles * full_tile_data_size;
    uint64_t meta_bound = kStreamHeaderSize + num_tiles * sizeof(uint32_t);
    return static_cast<uint32_t>(data_bound + meta_bound);
}

// Compressed data size for a single tile of `tile_size` bytes (matches the GPU
// writer's exact packet layout; see compressed_tile_data_size in the kernels).
inline uint32_t compressed_tile_size(uint32_t tile_size) noexcept {
    uint32_t nbytes = (tile_size + 31u) / 32u;          // bytes per stream (ceil)
    uint32_t stream0_words = (19u + nbytes * 8u + 31u) / 32u;
    uint32_t stream_words = (nbytes + 3u) / 4u;         // packets for streams 1..31
    uint32_t total_words = stream0_words + 31u * stream_words;
    return std::max<uint32_t>(128u, total_words * 4u);
}

// Standalone runtime class that owns the compiled kernels and scratch buffers.
class GDeflateCodec {
public:
    GDeflateCodec(Device &device, Stream &stream);

    // Compress `input_size` bytes from `input` into `output`.
    // Returns the exact number of bytes written to `output`.
    // `output` must be at least compress_bound(input_size) bytes.
    uint32_t compress(const ByteBuffer &input, ByteBuffer &output, uint32_t input_size);

    // Decompress a GDeflate tile stream from `input` into `output`.
    // `output_size` is the expected uncompressed size (used to derive tile count).
    void decompress(const ByteBuffer &input, ByteBuffer &output, uint32_t output_size);

    // Allocate an output buffer large enough for compress(input_size).
    [[nodiscard]] ByteBuffer allocate_compressed(uint32_t input_size) const;
    [[nodiscard]] ByteBuffer allocate_uncompressed(uint32_t output_size) const;

private:
    Device &_device;
    Stream &_stream;

    // Compiled DSL shaders.
    Shader1D<ByteBuffer, ByteBuffer, uint, uint, uint, uint> _compress_shader;
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint, uint, uint> _decompress_shader;
};

}// namespace luisa::example::gdeflate
