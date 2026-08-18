// GDeflate GPU compression/decompression runtime implementation.

#include "gdeflate.h"
#include "gdeflate_kernels.h"

#include <algorithm>
#include <cstring>

namespace luisa::example::gdeflate {

// GDeflate stream header layout (matches the GPU writer):
//   word1: id=4 [7:0], magic=0xFB [15:8], numTiles [31:16]
//   word2: lastTileSize (bytes) [19:2]
inline constexpr uint32_t kGDeflateHeaderId = 4u;
inline constexpr uint32_t kGDeflateHeaderMagic = 0xFBu;

GDeflateCodec::GDeflateCodec(Device &device, Stream &stream)
    : _device{device}, _stream{stream} {
    auto compress_kernel = make_compress_kernel();
    auto decompress_kernel = make_decompress_kernel();
    _compress_shader = device.compile<1>(compress_kernel);
    _decompress_shader = device.compile<1>(decompress_kernel);
}

uint32_t GDeflateCodec::compress(const ByteBuffer &input, ByteBuffer &output, uint32_t input_size) {
    uint32_t num_tiles = compute_num_tiles(input_size);
    if (num_tiles > kMaxTiles) {
        LUISA_ERROR("GDeflate compress: {} bytes needs {} tiles, exceeding the 16-bit limit of {}",
                    input_size, num_tiles, kMaxTiles);
        return 0u;
    }
    if (input_size > input.size_bytes()) {
        LUISA_ERROR("GDeflate compress: input_size {} exceeds buffer size {}",
                    input_size, input.size_bytes());
        return 0u;
    }
    // For an empty input emit a valid 8-byte empty stream (no tiles).
    uint32_t last_tile_size = (num_tiles == 0u) ? 0u
                                                : (input_size - (num_tiles - 1u) * kDefaultTileSize);
    uint32_t output_size = compress_bound(input_size);
    uint32_t dispatch_units = std::max(num_tiles, 1u); // 1 unit still writes the header

    // The compressed stream is written at the start of the output buffer.
    // 256 threads per tile: each thread emits one output-word slot.
    _stream << _compress_shader(input, output, 0u, 0u, num_tiles, last_tile_size).dispatch(dispatch_units * 256u)
            << synchronize();

    return output_size;
}

void GDeflateCodec::decompress(const ByteBuffer &input, ByteBuffer &output, uint32_t output_size) {
    uint32_t expected_tiles = compute_num_tiles(output_size);
    if (expected_tiles > kMaxTiles) {
        LUISA_ERROR("GDeflate decompress: output_size {} implies {} tiles, exceeding the 16-bit limit of {}",
                    output_size, expected_tiles, kMaxTiles);
        return;
    }
    if (input.size_bytes() < kStreamHeaderSize) {
        LUISA_ERROR("GDeflate decompress: input buffer too small ({} bytes < {})",
                    input.size_bytes(), kStreamHeaderSize);
        return;
    }

    // Validate the stream header on the host so a wrong-format stream is
      // Validate the stream header on the host so a wrong-format stream is
      // rejected before any kernel work, and clamp the dispatch to the tile count
      // actually present in the stream (a wrong output_size must not read the
      // pointer table out of bounds).
      uint64_t hdr = 0u;
      _stream << input.view(0u, 8u).copy_to(&hdr) << synchronize();
      uint32_t word1 = static_cast<uint32_t>(hdr);
      uint32_t word2 = static_cast<uint32_t>(hdr >> 32u);
      if ((word1 & 0xffu) != kGDeflateHeaderId || ((word1 >> 8u) & 0xffu) != kGDeflateHeaderMagic) {
          LUISA_ERROR("GDeflate decompress: invalid stream header (word1 = 0x{:08x})", word1);
          return;
      }
      uint32_t stream_tiles = (word1 >> 16u) & 0xffffu;
      if (stream_tiles != expected_tiles) {
          LUISA_WARNING("GDeflate decompress: output_size {} implies {} tiles but the stream has {}; clamping",
                        output_size, expected_tiles, stream_tiles);
      }
      uint32_t tiles = std::min(expected_tiles, stream_tiles);
      if (tiles == 0u) { return; } // empty stream: nothing to decode
      // Input tile stream starts at offset 0.  The decompressor needs atomic byte
    // writes, so the output is bound as a uint view of the byte buffer.
    _stream << _decompress_shader(input, output.view().as<uint>(), 0u, 0u, tiles,
                                  static_cast<uint32_t>(input.size_bytes()))
                    .dispatch(tiles * 32u)
            << synchronize();
}

ByteBuffer GDeflateCodec::allocate_compressed(uint32_t input_size) const {
    return _device.create_byte_buffer(std::max<uint64_t>(compress_bound(input_size), 8u));
}

ByteBuffer GDeflateCodec::allocate_uncompressed(uint32_t output_size) const {
    // Pad to at least one byte so a zero-length buffer is still a valid resource.
    return _device.create_byte_buffer(std::max<uint64_t>(output_size, 1u));
}

}// namespace luisa::example::gdeflate
