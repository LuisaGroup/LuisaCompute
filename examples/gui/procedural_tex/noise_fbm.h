// Procedural Texture Generator -- fractal Brownian motion (fBm) over improved
// Perlin noise: the default "clouds" look.
//
// Per octave: sum += perlin(p * f, seed + o * K) * a, with f *= lacunarity and
// a *= gain. The octave loop runs on the device ($for over the runtime octave
// count). The signed sum is normalized to ~[-1, 1] by the host-computed
// 1 / sum_{o<octaves} gain^o and then remapped to grayscale (v = sum * 0.5 + 0.5).
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct FbmCloudsParams {
    float scale = 6.0f;      // lattice cells across the image (slider 1..64)
    int octaves = 6;         // octave count (slider 1..8)
    float lacunarity = 2.0f; // per-octave frequency multiplier (slider 1.5..4)
    float gain = 0.5f;       // per-octave amplitude multiplier (slider 0.1..0.9)
    int seed = 0;            // hash seed (slider 0..255)
};

class FbmClouds final : public ProceduralTexture {
public:
    explicit FbmClouds(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "fbm"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

private:
    FbmCloudsParams _params;
    // Shader args: freq = scale / resolution and the amplitude normalization
    // (1 / sum_{o<octaves} gain^o) are parameter-derived and computed host-side.
    using ShaderType = Shader2D<Image<float>, float, uint, uint, float, float, float>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_fbm(Device &device);

}// namespace procedural_tex
