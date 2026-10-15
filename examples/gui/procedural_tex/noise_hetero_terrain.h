// Procedural Texture Generator -- Musgrave heterogeneous terrain.
//
// fBm whose per-octave contribution is weighted by the accumulated height:
// early (large-scale) noise decides where later detail is allowed to appear,
// which produces mountains next to flat plains instead of uniform bumpy fBm.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct HeteroTerrainParams {
    float scale = 6.0f;        // base frequency in lattice units across the image (1..64)
    int octaves = 6;           // octave count (1..8)
    float lacunarity = 2.0f;   // frequency multiplier per octave (1.5..4)
    float offset = 0.3f;       // signal offset before weighting (0..1)
    float roughness = 0.5f;    // H; amplitude ratio = lacunarity^-H (0..1)
    int seed = 0;              // hash seed (0..255)
};

class HeteroTerrain final : public ProceduralTexture {
public:
    explicit HeteroTerrain(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "hetero_terrain"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

    // Additionally require a wide grayscale range and a mid-range mean: the
    // weight recursion collapses detail on plains, but the output must still
    // span both bright mountain regions and darker flat regions.
    [[nodiscard]] bool validate(const luisa::vector<float4> &pixels, uint2 resolution) override;

private:
    HeteroTerrainParams _params;
    using ShaderType = Shader2D<Image<float>, uint, float, float, float, float, float, uint>;
    ShaderType _shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_hetero_terrain(Device &device);

}// namespace procedural_tex
