// GDeflate GPU compression/decompression runtime implementation.

#include "gdeflate.h"
#include "gdeflate_kernels.h"

namespace luisa::example::gdeflate {

GDeflateCodec::GDeflateCodec(Device &device, Stream &stream)
    : _device{device}, _stream{stream} {
    auto compress_kernel = make_compress_kernel();
    auto decompress_kernel = make_decompress_kernel();
    _compress_shader = device.compile<1>(compress_kernel);
    _decompress_shader = device.compile<1>(decompress_kernel);
}

uint32_t GDeflateCodec::compress(const ByteBuffer &input, ByteBuffer &output, uint32_t input_size) {
    uint32_t num_tiles = compute_num_tiles(input_size);
    uint32_t last_tile_size = input_size - (num_tiles - 1u) * kDefaultTileSize;
    uint32_t output_size = compress_bound(input_size);

    // The compressed stream is written at the start of the output buffer.
    // 32 threads per tile: one warp compresses each tile, one lane per stream.
    _stream << _compress_shader(input, output, 0u, 0u, num_tiles, last_tile_size).dispatch(num_tiles * 32u)
            << synchronize();

    return output_size;
}

void GDeflateCodec::decompress(const ByteBuffer &input, ByteBuffer &output, uint32_t output_size) {
    uint32_t num_tiles = compute_num_tiles(output_size);

    // Input tile stream starts at offset 0.  The decompressor needs atomic byte
    // writes, so the output is bound as a uint view of the byte buffer.
    _stream << _decompress_shader(input, output.view().as<uint>(), 0u, 0u, num_tiles).dispatch(num_tiles * 32u)
            << synchronize();
}

ByteBuffer GDeflateCodec::allocate_compressed(uint32_t input_size) const {
    return _device.create_byte_buffer(compress_bound(input_size));
}

ByteBuffer GDeflateCodec::allocate_uncompressed(uint32_t output_size) const {
    return _device.create_byte_buffer(output_size);
}

}// namespace luisa::example::gdeflate
