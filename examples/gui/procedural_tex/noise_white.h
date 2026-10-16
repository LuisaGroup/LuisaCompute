// Procedural Texture Generator -- white noise (per-pixel integer-lattice hash).
//
// Each pixel is mapped to a coarse "block" (cells across the image) and the
// block's coordinates are hashed to a single [0, 1) value written as grayscale.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct WhiteNoiseParams {
    int seed = 0;  // hash seed (slider 0..255)
    int scale = 16;// cells across the image (slider 1..128)
};

class WhiteNoise final : public ProceduralTexture {
public:
    explicit WhiteNoise(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "white_noise"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

    // Additionally require the grayscale range to span ~0.5: white noise must be
    // high-contrast.
    [[nodiscard]] bool validate(const luisa::vector<float4> &pixels, uint2 resolution) override;

private:
    WhiteNoiseParams _params;
    using ShaderType = Shader2D<Image<float>, uint, uint>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_white_noise(Device &device);

}// namespace procedural_tex
