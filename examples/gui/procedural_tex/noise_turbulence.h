// Procedural Texture Generator -- Billow / creased fBm ("turbulence").
//
// Same fractal accumulation as fBm but each octave contributes abs(perlin), so
// the ridges of the gradient noise become creases. Unsigned accumulation:
// v = sum * norm with the host-computed gain normalization. With `invert` the
// result is flipped (1 - v), which reads as puffy billow clouds instead of
// creased rock.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct TurbulenceParams {
    float scale = 6.0f;      // lattice cells across the image (slider 1..64)
    int octaves = 6;         // octave count (slider 1..8)
    float lacunarity = 2.0f; // per-octave frequency multiplier (slider 1.5..4)
    float gain = 0.5f;       // per-octave amplitude multiplier (slider 0.1..0.9)
    int seed = 0;            // hash seed (slider 0..255)
    bool invert = true;      // flip the billow (checkbox)
};

class Turbulence final : public ProceduralTexture {
public:
    explicit Turbulence(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "turbulence"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

private:
    TurbulenceParams _params;
    // `invert` is passed as a runtime float (0/1) so toggling it only re-dispatches.
    using ShaderType = Shader2D<Image<float>, float, uint, uint, float, float, float, float>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_turbulence(Device &device);

}// namespace procedural_tex
