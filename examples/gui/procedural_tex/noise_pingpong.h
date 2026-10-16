// Procedural Texture Generator -- FastNoiseLite-style ping-pong fractal.
//
// Each octave's Perlin value is mapped to [0, 1], pushed outwards by `strength`
// and then folded through a triangle wave ("ping-pong"), producing sediment-like
// bands. The folds are summed with the usual fBm amplitude falloff and normalized
// host-side (unsigned mapping: v = sum * norm).
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct PingPongParams {
    float scale = 6.0f;      // lattice cells across the image (slider 1..64)
    int octaves = 6;         // octave count (slider 1..8)
    float lacunarity = 2.0f; // per-octave frequency multiplier (slider 1.5..4)
    float gain = 0.5f;       // per-octave amplitude multiplier (slider 0.1..0.9)
    float strength = 2.6f;   // ping-pong fold strength (slider 0.5..4); 2.0 would coincide with inverted turbulence
    int seed = 0;            // hash seed (slider 0..255)
};

class PingPong final : public ProceduralTexture {
public:
    explicit PingPong(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "pingpong"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

private:
    PingPongParams _params;
    using ShaderType = Shader2D<Image<float>, float, uint, uint, float, float, float, float>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_pingpong(Device &device);

}// namespace procedural_tex
