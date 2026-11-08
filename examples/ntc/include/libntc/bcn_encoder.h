/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * Pure GPU BCn (BC1, BC3, BC4, BC5, BC7) Block Compression Compute Shaders in LuisaCompute DSL.
 * Zero-copy direct VRAM resident transcoding.
 */

#pragma once

#include <libntc/common.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <filesystem>
#include <vector>

namespace luisa::ntc {

class BCnEncoder {
public:
    Device &device;
    Stream &stream;

private:
    // Pure GPU DSL Block Compression Shaders
    // BC1: takes float buffer [H, W, 16], outputs Buffer<uint> (2 uints per 4x4 block)
    Shader2D<Buffer<float>, Buffer<uint>, uint, uint, uint> _compressBC1Kernel;
    // BC3: outputs Buffer<uint> (4 uints per 4x4 block: BC4 Alpha + BC1 RGB)
    Shader2D<Buffer<float>, Buffer<uint>, uint, uint, uint, uint> _compressBC3Kernel;
    // BC4: outputs Buffer<uint> (2 uints per 4x4 block: single channel)
    Shader2D<Buffer<float>, Buffer<uint>, uint, uint, uint> _compressBC4Kernel;
    // BC5: outputs Buffer<uint> (4 uints per 4x4 block: two BC4 channels)
    Shader2D<Buffer<float>, Buffer<uint>, uint, uint, uint, uint> _compressBC5Kernel;
    // BC7: outputs Buffer<uint> (4 uints per 4x4 block: high-quality Mode 6)
    Shader2D<Buffer<float>, Buffer<uint>, uint, uint, uint, uint> _compressBC7Kernel;

    bool _compiled{false};

    void _compile_shaders();

public:
    BCnEncoder(Device &dev, Stream &st);

    // Compress input GPU float buffer [H, W, num_channels] directly into GPU memory
    // Returns Buffer<uint> containing raw BCn blocks
    Buffer<uint> encode_to_gpu_buffer(const Buffer<float> &input_float_buffer,
                                      uint32_t width, uint32_t height,
                                      uint32_t channel_offset, uint32_t num_channels,
                                      BlockCompressedFormat format);

    // Transcode and write standard DirectDraw Surface (.dds) file
    bool encode_and_save_dds(const Buffer<float> &input_float_buffer,
                             uint32_t width, uint32_t height,
                             uint32_t channel_offset, uint32_t num_channels,
                             BlockCompressedFormat format,
                             const std::filesystem::path &out_path);
};

} // namespace luisa::ntc
