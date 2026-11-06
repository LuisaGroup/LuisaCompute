/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * Feature Grid: Multi-resolution latent grid storage, LOD conversion, and sampling.
 */

#pragma once

#include <libntc/common.h>
#include <vector>

namespace luisa::ntc {

class FeatureGridMath {
public:
    static constexpr int BytesPerLatentPixel = 2; // 16 bits = BGRA4
    static constexpr int FeaturesPerLayer = 4;

    static int lod_to_neural_lod(int lod, int grid_scale, int neural_lods = NTC_MAX_NEURAL_MIPS) noexcept {
        static constexpr int shared_mips[] = { 2, 3, 3, 3, 4, 4 };
        int gs = std::clamp(grid_scale, 1, 6);
        int shared = shared_mips[gs - 1];
        int neural_lod = (lod - shared + 2) & ~1;
        return std::clamp(neural_lod, 0, std::max(0, neural_lods - 2));
    }

    static int get_grid_dimension(int image_dim, int neural_lod, int grid_scale) noexcept {
        int scaled = std::max(1, image_dim / std::max(1, grid_scale));
        return std::max(1, scaled >> neural_lod);
    }

    static int get_num_layers(int num_features) noexcept {
        return (num_features + FeaturesPerLayer - 1) / FeaturesPerLayer;
    }

    static int calculate_num_neural_mips(int width, int height, int grid_scale) noexcept {
        int min_dim = std::min(width, height);
        int min_grid = std::max(1, min_dim / std::max(1, grid_scale));
        int mips = static_cast<int>(std::floor(std::log2(static_cast<float>(min_grid)))) + 1;
        return std::max(2, mips) & ~1;
    }

    static float calculate_latent_bits_per_pixel(int grid_scale, int num_features) noexcept {
        int num_layers = get_num_layers(num_features);
        return static_cast<float>(num_layers) / static_cast<float>(grid_scale * grid_scale) * static_cast<float>(BytesPerLatentPixel) * 10.0f;
    }
};

struct LatentMipLevel {
    uint32_t width{0};
    uint32_t height{0};
    uint32_t numLayers{0};
    // Flattened BGRA4 uint16 pixels: [layer][y * width + x]
    std::vector<std::vector<uint16_t>> layerData;
};

class FeatureGridStorage {
public:
    int32_t gridSizeScale{4};
    int32_t numFeatures{16};
    std::vector<LatentMipLevel> mipLevels;

public:
    FeatureGridStorage() = default;

    void initialize(int image_width, int image_height, int num_features, int grid_scale = 4);

    [[nodiscard]] size_t get_num_mips() const noexcept { return mipLevels.size(); }
    [[nodiscard]] size_t get_num_layers() const noexcept {
        return mipLevels.empty() ? 0 : mipLevels[0].numLayers;
    }

    // BGRA4 quantization
    static uint16_t pack_bgra4(float b, float g, float r, float a) noexcept;
    static void unpack_bgra4(uint16_t packed, float &b, float &g, float &r, float &a) noexcept;

    // CPU reference sample
    void sample_bilinear(int neural_lod, float u, float v, float *out_features) const noexcept;
};

} // namespace luisa::ntc
