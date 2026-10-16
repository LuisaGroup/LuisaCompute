// Procedural Texture Generator -- fractal Worley (F1 cellular noise over octaves).
//
// Each octave adds a jittered-cell Euclidean F1 distance with the usual fBm
// frequency/amplitude progression; the unsigned sum is normalized host-side
// (v = sum * norm). A classic base for stone and hammered-metal looks.
//
// NOTE: the shared make_worley_2d() callable scans a 3x3 cell neighborhood per
// call, so 5-8 octaves per pixel is already heavy -- the search window is kept
// at 3x3 deliberately.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct FractalWorleyParams {
    float scale = 6.0f;      // base cell count across the image (slider 1..64)
    int octaves = 5;         // octave count (slider 1..8)
    float lacunarity = 2.0f; // per-octave frequency multiplier (slider 1.5..4)
    float gain = 0.5f;       // per-octave amplitude multiplier (slider 0.1..0.9)
    float jitter = 1.0f;     // feature-point jitter (slider 0..1)
    int seed = 0;            // hash seed (slider 0..255)
};

class FractalWorley final : public ProceduralTexture {
public:
    explicit FractalWorley(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "fractal_worley"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

private:
    FractalWorleyParams _params;
    // norm compensates host-side for the shared callable's internal 1.2 distance
    // scale so default parameters still produce a full-range texture.
    using ShaderType = Shader2D<Image<float>, float, uint, uint, float, float, float, float>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_fractal_worley(Device &device);

}// namespace procedural_tex
