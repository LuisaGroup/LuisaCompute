// Procedural Texture Generator -- value noise (bilinear-interpolated lattice hash).
//
// Uses the shared make_value_noise_2d() callable. The pixel coordinate is mapped
// to lattice units (scale cells across the image); the noise returns ~[0, 1] and
// is written directly as grayscale.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct ValueNoiseParams {
    float scale = 8.0f;// lattice cells across the image (slider 1..64)
    int seed = 0;      // hash seed (slider 0..255)
};

class ValueNoise final : public ProceduralTexture {
public:
    explicit ValueNoise(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "value_noise"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

private:
    ValueNoiseParams _params;
    // freq = scale / resolution, computed host-side (parameter-derived normalization).
    using ShaderType = Shader2D<Image<float>, float, uint>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_value_noise(Device &device);

}// namespace procedural_tex
