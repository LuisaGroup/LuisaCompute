/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * High-performance GPU Inference Engine: On-Load Decompression & On-Sample Evaluation.
 */

#pragma once

#include <libntc/common.h>
#include "ntex_container.h"
#include "feature_grid.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace luisa::ntc {

struct DecompressedTexture {
    std::string name;
    uint32_t width{0};
    uint32_t height{0};
    uint32_t channels{0};
    ChannelFormat format{ChannelFormat::UNORM8};
    ColorSpace colorSpace{ColorSpace::Linear};
    BlockCompressedFormat bcFormat{BlockCompressedFormat::None};
    // Host pixels (RGBA float or UNORM8 bytes)
    std::vector<uint8_t> pixelBytes;
    // GPU image if resident
    Image<float> deviceImage;
};

class InferEngine {
public:
    Device &device;
    Stream &stream;

private:
    // Uploaded weights buffer (all layers)
    Buffer<int> _weightBuffer;
    Buffer<float> _scaleBuffer;
    Buffer<int> _biasBuffer;

    // Uploaded latent grid buffer (all mips and layers)
    Buffer<uint> _latentBuffer;
    Buffer<uint> _latentMipOffsets;
    Buffer<uint> _latentMipDims; // (width, height) per mip

    // Texture set metadata buffer
    Buffer<TextureSetConstants> _constantsBuffer;

    // Compiled On-Load Decompression Shader
    // args: (weights, scales, biases, latents, mip_offsets, mip_dims, out_buffer, neuralLod, positionLod, positionScale, width, height)
    Shader2D<Buffer<int>, Buffer<float>, Buffer<int>, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<float>, uint, float, float, uint, uint> _decompressKernel;

    bool _initialized{false};
    uint32_t _numNeuralMips{0};
    uint32_t _numLatentLayers{0};
    uint32_t _numFeatures{16};
    uint32_t _gridScale{4};
    uint32_t _containerWidth{0};
    uint32_t _containerHeight{0};
    std::vector<ColorMipDesc> _colorMips;
    TextureSetConstants _constants{};

public:
    InferEngine(Device &dev, Stream &st);

    // Initialize from loaded NTEX container
    bool load_container(const NtexContainer &container);

    // On-Load Decompression: decompress all channels for a specific color mip
    // Returns flattened float buffer on GPU: [height, width, NTC_MLP_OUTPUT_CHANNELS]
    Buffer<float> decompress_mip_device(uint32_t color_mip, uint32_t width, uint32_t height);

    // On-Load Decompression to host images
    std::vector<DecompressedTexture> decompress_all_textures(const NtexContainer &container, uint32_t color_mip = 0);

    // Benchmark decompression throughput (in Mtexels/s)
    double benchmark_throughput(uint32_t color_mip, uint32_t width, uint32_t height, uint32_t iterations = 50);

    [[nodiscard]] bool is_initialized() const noexcept { return _initialized; }
    [[nodiscard]] const TextureSetConstants &get_constants() const noexcept { return _constants; }
};

} // namespace luisa::ntc
