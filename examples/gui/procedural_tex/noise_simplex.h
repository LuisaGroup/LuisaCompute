// Procedural Texture Generator -- 2D simplex noise (Perlin 2001).
//
// Uses the shared make_simplex_2d() callable (output roughly [-1, 1]); remapped to
// [0, 1] via v = n * 0.5 + 0.5 and clamped before writing as grayscale.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct SimplexNoiseParams {
    float scale = 8.0f;// simplex-lattice cells across the image (slider 1..64)
    int seed = 0;      // hash seed (slider 0..255)
};

class SimplexNoise final : public ProceduralTexture {
public:
    explicit SimplexNoise(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "simplex_noise"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

private:
    SimplexNoiseParams _params;
    using ShaderType = Shader2D<Image<float>, float, uint>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_simplex_noise(Device &device);

}// namespace procedural_tex
