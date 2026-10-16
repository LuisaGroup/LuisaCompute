// Procedural Texture Generator -- Gaussian noise (Box-Muller over white noise).
//
// Two decorrelated per-pixel uniform hashes feed the Box-Muller transform to
// produce a standard-normal value, then the color is 0.5 + sigma * z clamped to
// [0, 1] and written as grayscale. The histogram is Gaussian, centered at 0.5.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct GaussianNoiseParams {
    int seed = 0;          // hash seed (slider 0..255)
    float sigma = 0.18f;   // std-dev of the Gaussian (slider 0.02..0.5)
};

class GaussianNoise final : public ProceduralTexture {
public:
    explicit GaussianNoise(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "gaussian_noise"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

    // Additionally require the image mean to lie in [0.4, 0.6]: the Gaussian is
    // centered at 0.5.
    [[nodiscard]] bool validate(const luisa::vector<float4> &pixels, uint2 resolution) override;

private:
    GaussianNoiseParams _params;
    using ShaderType = Shader2D<Image<float>, uint, float>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_gaussian_noise(Device &device);

}// namespace procedural_tex
