// Procedural Texture Generator -- Musgrave ridged multifractal implementation.
//
// Kernel captures the shared make_perlin_2d() callable; the octave loop is a
// device-side `$for` over the runtime `octaves` argument (per-octave seeds are
// derived inside the loop from the loop counter, decorrelating the octaves).
//
// Per octave (exactly as in the algorithm spec):
//   n = 1 - |perlin(p*freq, seed + o*101)|   // fold: creases where noise = 0
//   n = n * n                                // sharpen the ridges
//   n = n * weight                           // previous octave's ridge intensity
//   weight = clamp(n * 2, 0, 1)
//   sum += n * amp
//   amp *= gain; freq *= lacunarity
// norm = 1 / sum_{o<octaves} gain^o is computed host-side and passed in.

#include "noise_ridged.h"

#include <algorithm>

#include "noise_common.h"

namespace procedural_tex {

RidgedMultifractal::RidgedMultifractal(Device &device) {
    auto perlin = make_perlin_2d();
    Kernel2D kernel = [&](ImageVar<float> image, UInt octaves, Float scale, Float lacunarity,
                          Float gain, Float norm, UInt seed) noexcept {
        auto coord = dispatch_id().xy();
        Float2 uv = make_float2(cast<float>(coord.x) + 0.5f,
                                 cast<float>(coord.y) + 0.5f) /
                    make_float2(cast<float>(image.size().x), cast<float>(image.size().y));
        Float freq = def(scale);
        Float amp = def(1.0f);
        Float weight = def(1.0f);
        Float sum = def(0.0f);
        $for (o, octaves) {
            UInt sd = seed + cast<uint>(o) * 101u;
            Float n = 1.0f - abs(perlin(uv * freq, sd));
            n = n * n;
            n = n * weight;
            weight = clamp(n * 2.0f, 0.0f, 1.0f);
            sum = sum + n * amp;
            amp = amp * gain;
            freq = freq * lacunarity;
        };
        Float g = clamp(sum * norm, 0.0f, 1.0f);
        image.write(coord, make_float4(g, g, g, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool RidgedMultifractal::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderInt("octaves", &_params.octaves, 1, 8);
    changed |= ImGui::SliderFloat("lacunarity", &_params.lacunarity, 1.5f, 4.0f, "%.1f");
    changed |= ImGui::SliderFloat("gain", &_params.gain, 0.1f, 0.9f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void RidgedMultifractal::generate(Image<float> &image, Stream &stream) {
    const auto octaves = static_cast<uint>(std::clamp(_params.octaves, 1, 8));
    const auto scale = std::clamp(_params.scale, 1.0f, 64.0f);
    const auto lacunarity = std::clamp(_params.lacunarity, 1.5f, 4.0f);
    const auto gain = std::clamp(_params.gain, 0.1f, 0.9f);
    const auto seed = static_cast<uint>(std::clamp(_params.seed, 0, 255));
    // norm = 1 / sum_{o<octaves} gain^o (host-side, deterministic).
    double s = 0.0;
    double a = 1.0;
    for (uint o = 0u; o < octaves; ++o) {
        s += a;
        a *= gain;
    }
    const auto norm = (s > 0.0) ? static_cast<float>(1.0 / s) : 1.0f;
    stream << _shader(image, octaves, scale, lacunarity, gain, norm, seed).dispatch(image.size());
}

bool RidgedMultifractal::validate(const luisa::vector<float4> &pixels, uint2 resolution) {
    // Generic invariants (finite, [0, 1], spatially varying).
    if (!ProceduralTexture::validate(pixels, resolution)) {
        return false;
    }
    // Ridge structure: high contrast between ridges and valleys, with the mean
    // in the mid range. A texture that collapsed to all-dark or all-bright
    // (e.g. wrong normalization) fails on the mean bound.
    auto min_v = 1e30f;
    auto max_v = -1e30f;
    double sum = 0.0;
    for (auto &p : pixels) {
        min_v = std::min(min_v, p.x);
        max_v = std::max(max_v, p.x);
        sum += p.x;
    }
    const auto mean = static_cast<float>(sum / static_cast<double>(pixels.size()));
    if (max_v - min_v < 0.3f) {
        LUISA_WARNING("[ridged_multifractal] ridge contrast too low (min {} max {})", min_v, max_v);
        return false;
    }
    if (mean < 0.05f || mean > 0.95f) {
        LUISA_WARNING("[ridged_multifractal] output collapsed (mean {})", mean);
        return false;
    }
    return true;
}

luisa::unique_ptr<ProceduralTexture> create_ridged_multifractal(Device &device) {
    return luisa::make_unique<RidgedMultifractal>(device);
}

}// namespace procedural_tex
