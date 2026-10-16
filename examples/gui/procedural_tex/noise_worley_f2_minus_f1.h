// Procedural Texture Generator -- Worley noise, F2 - F1 metric.
//
// The classic caustics / foam-border cellular metric: the gap between the
// second and the first nearest feature-point distance is small exactly along
// cell boundaries, so (f2 - f1) * contrast, clamped to [0, 1], lights up
// bright borders between dark cells.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct WorleyF2MinusF1Params {
    float scale = 8.0f;    // cells across the image (slider 2..64)
    float jitter = 1.0f;   // feature-point jitter within the cell (slider 0..1)
    float contrast = 4.0f; // (f2 - f1) multiplier (slider 1..16)
    int seed = 0;          // hash seed (slider 0..255)
};

class WorleyF2MinusF1 final : public ProceduralTexture {
public:
    explicit WorleyF2MinusF1(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "worley_f2_minus_f1"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

    // Additionally require both a dark interior and a lit border: the output
    // must contain values close to 0 and values close to 1.
    [[nodiscard]] bool validate(const luisa::vector<float4> &pixels, uint2 resolution) override;

private:
    WorleyF2MinusF1Params _params;
    using ShaderType = Shader2D<Image<float>, float, float, float, uint>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_worley_f2_minus_f1(Device &device);

}// namespace procedural_tex
