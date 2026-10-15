// Procedural Texture Generator -- Musgrave heterogeneous terrain implementation.
//
// Kernel captures the shared make_perlin_2d() callable (which itself captures
// make_hash01() and make_fade_quintic()); compilation happens once in the ctor
// and the per-pixel fBm loop uses the DSL `$for` construct to run the octave
// counter on the device.
//
// Heterogeneous-terrain recursion (Musgrave, "Texturing and Modeling: A
// Procedural Approach", 1998), exactly as in the algorithm spec:
//   signal = perlin(p*freq, seed+o*101)*0.5+0.5
//   signal = (signal + offset) * weight
//   value += signal * amp
//   weight = clamp(signal, 0, 1)
//   freq  *= lacunarity; amp *= amp_ratio  (amp_ratio = lacunarity^-H)
// The normalization constant norm = 1 / sum_{o<octaves} amp_ratio^o is
// precomputed host-side and passed as a kernel arg so the accumulated value
// maps approximately to [0, 1].

#include "noise_hetero_terrain.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

HeteroTerrain::HeteroTerrain(Device &device) {
    auto perlin = make_perlin_2d();
    Kernel2D kernel = [&](ImageVar<float> image, UInt octaves, Float scale, Float lacunarity,
                          Float offset, Float amp_ratio, Float norm, UInt seed) noexcept {
        auto coord = dispatch_id().xy();
        Float2 uv = make_float2(cast<float>(coord.x) + 0.5f,
                                 cast<float>(coord.y) + 0.5f) /
                    make_float2(cast<float>(image.size().x), cast<float>(image.size().y));
        // Mutable DSL accumulators (def(...)).
        Float freq = def(scale);
        Float amp = def(1.0f);
        Float value = def(0.0f);
        Float weight = def(1.0f);
        // Device-side octave loop.
        $for (o, octaves) {
            UInt sd = seed + cast<uint>(o) * 101u;
            Float n = perlin(uv * freq, sd) * 0.5f + 0.5f;
            Float sig = (n + offset) * weight;
            value = value + sig * amp;
            weight = clamp(sig, 0.0f, 1.0f);
            freq = freq * lacunarity;
            amp = amp * amp_ratio;
        };
        Float g = clamp(value * norm, 0.0f, 1.0f);
        image.write(coord, make_float4(g, g, g, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool HeteroTerrain::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderInt("octaves", &_params.octaves, 1, 8);
    changed |= ImGui::SliderFloat("lacunarity", &_params.lacunarity, 1.5f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("offset", &_params.offset, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("roughness", &_params.roughness, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void HeteroTerrain::generate(Image<float> &image, Stream &stream) {
    // Clamp params into their documented ranges (defensive; the sliders already do).
    const auto octaves = static_cast<uint>(std::clamp(_params.octaves, 1, 8));
    const auto scale = std::clamp(_params.scale, 1.0f, 64.0f);
    const auto lacunarity = std::clamp(_params.lacunarity, 1.5f, 4.0f);
    const auto offset = std::clamp(_params.offset, 0.0f, 1.0f);
    const auto roughness = std::clamp(_params.roughness, 0.0f, 1.0f);
    const auto seed = static_cast<uint>(std::clamp(_params.seed, 0, 255));
    // H -> amp ratio.
    const auto amp_ratio = std::pow(lacunarity, -roughness);
    // norm = 1 / sum_{o<octaves} amp_ratio^o.
    double s = 0.0;
    double a = 1.0;
    for (uint o = 0u; o < octaves; ++o) {
        s += a;
        a *= amp_ratio;
    }
    const auto norm = (s > 0.0) ? static_cast<float>(1.0 / s) : 1.0f;
    stream << _shader(image, octaves, scale, lacunarity, offset, amp_ratio, norm, seed)
                  .dispatch(image.size());
}

bool HeteroTerrain::validate(const luisa::vector<float4> &pixels, uint2 resolution) {
    // Generic invariants (finite, [0, 1], spatially varying).
    if (!ProceduralTexture::validate(pixels, resolution)) {
        return false;
    }
    // At the default parameters the recursion (perlin*0.5+0.5 + offset) * weight
    // has a wide, mostly-bright spread; require both a broad range and a mid
    // mean. Pure constant textures and heavily saturated extremes fail one of
    // the two bounds.
    auto min_v = 1e30f;
    auto max_v = -1e30f;
    double sum = 0.0;
    for (auto &p : pixels) {
        min_v = std::min(min_v, p.x);
        max_v = std::max(max_v, p.x);
        sum += p.x;
    }
    const auto mean = static_cast<float>(sum / static_cast<double>(pixels.size()));
    if (max_v - min_v < 0.2f) {
        LUISA_WARNING("[hetero_terrain] output range too narrow (min {} max {})", min_v, max_v);
        return false;
    }
    if (mean < 0.05f || mean > 0.95f) {
        LUISA_WARNING("[hetero_terrain] output collapsed (mean {})", mean);
        return false;
    }
    return true;
}

luisa::unique_ptr<ProceduralTexture> create_hetero_terrain(Device &device) {
    return luisa::make_unique<HeteroTerrain>(device);
}

}// namespace procedural_tex
