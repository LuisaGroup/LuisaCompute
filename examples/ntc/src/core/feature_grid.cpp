/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * Feature Grid implementation.
 */

#include "feature_grid.h"
#include <cmath>
#include <algorithm>

namespace luisa::ntc {

void FeatureGridStorage::initialize(int image_width, int image_height, int num_features, int grid_scale) {
    gridSizeScale = grid_scale;
    numFeatures = num_features;
    int num_mips = FeatureGridMath::calculate_num_neural_mips(image_width, image_height, grid_scale);
    int num_layers = FeatureGridMath::get_num_layers(num_features);

    mipLevels.resize(num_mips);
    for (int mip = 0; mip < num_mips; ++mip) {
        auto &lvl = mipLevels[mip];
        lvl.width = FeatureGridMath::get_grid_dimension(image_width, mip, grid_scale);
        lvl.height = FeatureGridMath::get_grid_dimension(image_height, mip, grid_scale);
        lvl.numLayers = num_layers;
        lvl.layerData.resize(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            lvl.layerData[l].assign(static_cast<size_t>(lvl.width) * lvl.height, 0);
        }
    }
}

uint16_t FeatureGridStorage::pack_bgra4(float f0, float f1, float f2, float f3) noexcept {
    auto q = [](float v) -> uint32_t {
        float c = std::clamp(v, 0.0f, 1.0f);
        return static_cast<uint32_t>(std::round(c * 15.0f));
    };
    uint32_t b0 = q(f0);
    uint32_t b1 = q(f1);
    uint32_t b2 = q(f2);
    uint32_t b3 = q(f3);
    return static_cast<uint16_t>(b0 | (b1 << 4) | (b2 << 8) | (b3 << 12));
}

void FeatureGridStorage::unpack_bgra4(uint16_t packed, float &f0, float &f1, float &f2, float &f3) noexcept {
    f0 = static_cast<float>(packed & 0xF) * (1.0f / 15.0f);
    f1 = static_cast<float>((packed >> 4) & 0xF) * (1.0f / 15.0f);
    f2 = static_cast<float>((packed >> 8) & 0xF) * (1.0f / 15.0f);
    f3 = static_cast<float>((packed >> 12) & 0xF) * (1.0f / 15.0f);
}

void FeatureGridStorage::sample_bilinear(int neural_lod, float u, float v, float *out_features) const noexcept {
    if (neural_lod < 0 || neural_lod >= static_cast<int>(mipLevels.size())) {
        std::fill_n(out_features, numFeatures, 0.0f);
        return;
    }

    const auto &lvl = mipLevels[neural_lod];
    float x = u * static_cast<float>(lvl.width) - 0.5f;
    float y = v * static_cast<float>(lvl.height) - 0.5f;

    int x_b = static_cast<int>(std::floor(x));
    int y_b = static_cast<int>(std::floor(y));

    float dx = x - static_cast<float>(x_b);
    float dy = y - static_cast<float>(y_b);
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    int x0 = (x_b % static_cast<int>(lvl.width) + lvl.width) % lvl.width;
    int y0 = (y_b % static_cast<int>(lvl.height) + lvl.height) % lvl.height;
    int x1 = (x0 + 1) % lvl.width;
    int y1 = (y0 + 1) % lvl.height;

    size_t idx00 = static_cast<size_t>(y0) * lvl.width + x0;
    size_t idx10 = static_cast<size_t>(y0) * lvl.width + x1;
    size_t idx01 = static_cast<size_t>(y1) * lvl.width + x0;
    size_t idx11 = static_cast<size_t>(y1) * lvl.width + x1;

    for (uint32_t l = 0; l < lvl.numLayers; ++l) {
        float c00[4], c10[4], c01[4], c11[4];
        unpack_bgra4(lvl.layerData[l][idx00], c00[0], c00[1], c00[2], c00[3]);
        unpack_bgra4(lvl.layerData[l][idx10], c10[0], c10[1], c10[2], c10[3]);
        unpack_bgra4(lvl.layerData[l][idx01], c01[0], c01[1], c01[2], c01[3]);
        unpack_bgra4(lvl.layerData[l][idx11], c11[0], c11[1], c11[2], c11[3]);

        for (int k = 0; k < 4; ++k) {
            uint32_t f_idx = l * 4 + k;
            if (f_idx < static_cast<uint32_t>(numFeatures)) {
                float val = c00[k] * w00 + c10[k] * w10 + c01[k] * w01 + c11[k] * w11;
                // Scale from [0, 1] to [-1, 1]
                out_features[f_idx] = val * 2.0f - 1.0f;
            }
        }
    }
}

} // namespace luisa::ntc
