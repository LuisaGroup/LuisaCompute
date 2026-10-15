// Procedural Texture Generator -- Quilez-style domain-warped fBm.
//
// A Perlin fBm is evaluated at a position displaced by two other fBm fields
// (a two-stage warp: q = fbm(p), r = fbm(p + w*q), color = fbm(p + w*r)). The
// result reads as organic marble / fluid smoke. The warp strength, octave count,
// lacunarity and gain are kernel arguments; the fBm helper is an inline Callable
// with its own runtime $for octave loop, called five times per pixel.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct DomainWarpFbmParams {
    float scale = 4.0f;      // lattice cells across the image (slider 1..32)
    float warp = 4.0f;       // warp strength (slider 0..8)
    int octaves = 5;         // per-fBm octave count (slider 1..8)
    float lacunarity = 2.0f; // per-octave frequency multiplier (slider 1.5..4)
    float gain = 0.5f;       // per-octave amplitude multiplier (slider 0.1..0.9)
    int seed = 0;            // hash seed (slider 0..255)
};

class DomainWarpFbm final : public ProceduralTexture {
public:
    explicit DomainWarpFbm(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "domain_warp"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

private:
    DomainWarpFbmParams _params;
    using ShaderType = Shader2D<Image<float>, float, float, uint, uint, float, float, float>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_domain_warp(Device &device);

}// namespace procedural_tex
