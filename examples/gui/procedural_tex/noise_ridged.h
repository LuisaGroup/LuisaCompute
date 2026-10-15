// Procedural Texture Generator -- Musgrave ridged multifractal.
//
// Folded-noise fBm variant that produces sharp creases / vein networks: each
// octave contributes (1 - |perlin|)^2 (the fold turns zero-crossings into
// ridges) and the running ridge intensity acts as the weight of the next
// octave, so detail concentrates in the valleys between ridges.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct RidgedMultifractalParams {
    float scale = 6.0f;      // base frequency in lattice units across the image (1..64)
    int octaves = 6;         // octave count (1..8)
    float lacunarity = 2.0f; // frequency multiplier per octave (1.5..4)
    float gain = 0.5f;       // amplitude multiplier per octave (0.1..0.9)
    int seed = 0;            // hash seed (0..255)
};

class RidgedMultifractal final : public ProceduralTexture {
public:
    explicit RidgedMultifractal(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "ridged_multifractal"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

    // Additionally require the ridge structure to survive to the output:
    // high contrast AND a mean in the mid range (a collapsed/uniform texture
    // fails one of the two).
    [[nodiscard]] bool validate(const luisa::vector<float4> &pixels, uint2 resolution) override;

private:
    RidgedMultifractalParams _params;
    using ShaderType = Shader2D<Image<float>, uint, float, float, float, float, uint>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_ridged_multifractal(Device &device);

}// namespace procedural_tex
