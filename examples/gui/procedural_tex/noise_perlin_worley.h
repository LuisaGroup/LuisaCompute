// Procedural Texture Generator -- Perlin fBm modulated by fractal Worley.
//
// The canonical real-time volumetric-cloud base (Horizon Zero Dawn style):
// low-frequency Perlin fBm provides the soft density field, an inverted fractal
// Worley F1 carves cell-shaped holes into it. `worley_weight` controls how deep
// the Worley cuts (0 = pure Perlin fBm, 1 = full Worley erosion).
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct PerlinWorleyParams {
    float scale = 4.0f;          // lattice cells across the image (slider 1..32)
    int octaves = 5;             // octave count for both fractals (slider 1..8)
    float lacunarity = 2.0f;     // per-octave frequency multiplier (slider 1.5..4)
    float gain = 0.5f;           // per-octave amplitude multiplier (slider 0.1..0.9)
    float jitter = 1.0f;         // Worley feature-point jitter (slider 0..1)
    float worley_weight = 0.7f;  // Worley erosion strength (slider 0..1)
    int seed = 0;                // hash seed (slider 0..255)
};

class PerlinWorley final : public ProceduralTexture {
public:
    explicit PerlinWorley(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "perlin_worley"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

private:
    PerlinWorleyParams _params;
    // The two normalization factors (Perlin amplitude sum, Worley amplitude sum
    // including the shared callable's internal 1.2 distance scale) are computed
    // host-side.
    using ShaderType = Shader2D<Image<float>, float, uint, uint, float, float, float, float, float, float>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_perlin_worley(Device &device);

}// namespace procedural_tex
