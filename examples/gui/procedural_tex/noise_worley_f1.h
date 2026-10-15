// Procedural Texture Generator -- Worley (cellular) noise, F1 metric.
//
// Per-pixel distance to the nearest feature point of the cellular lattice,
// via the shared make_worley_2d() callable; the F1 distance is written as
// grayscale.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct WorleyF1Params {
    float scale = 8.0f;  // cells across the image (slider 2..64)
    float jitter = 1.0f; // feature-point jitter within the cell (slider 0..1)
    int seed = 0;        // hash seed (slider 0..255)
};

class WorleyF1 final : public ProceduralTexture {
public:
    explicit WorleyF1(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "worley_f1"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

    // Additionally require a wide grayscale range: F1 spans almost the full
    // [0, 1] because it is the remapped min distance over cells across the image.
    [[nodiscard]] bool validate(const luisa::vector<float4> &pixels, uint2 resolution) override;

private:
    WorleyF1Params _params;
    using ShaderType = Shader2D<Image<float>, float, float, uint>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_worley_f1(Device &device);

}// namespace procedural_tex
