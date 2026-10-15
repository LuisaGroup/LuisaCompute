// Procedural Texture Generator -- Billow / creased fBm implementation.

#include "noise_turbulence.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

namespace {
// Host-side normalization of a fractal sum: 1 / sum_{o<octaves} gain^o
// (the divisor is octaves when gain ~= 1).
[[nodiscard]] float fractal_norm(float gain, int octaves) noexcept {
    auto n = std::max(1, octaves);
    if (std::fabs(gain - 1.0f) < 1e-4f) {
        return 1.0f / static_cast<float>(n);
    }
    auto num = 1.0f - std::pow(gain, static_cast<float>(n));
    auto den = 1.0f - gain;
    return 1.0f / (num / den);
}
}// namespace

Turbulence::Turbulence(Device &device) {
    auto perlin = make_perlin_2d();
    Kernel2D kernel = [&](ImageVar<float> image, Float freq, UInt seed,
                          UInt octaves, Float lacunarity, Float gain,
                          Float norm, Float invert) noexcept {
        auto coord = dispatch_id().xy();
        // p is in base lattice units (scale cells across the image), so the
        // per-octave frequency multiplier starts at 1: using freq here as well
        // would square the base frequency and flatten the field.
        Float2 p = make_float2(cast<float>(coord.x), cast<float>(coord.y)) * freq;
        Float f = def(1.0f);
        Float a = def(1.0f);
        Float sum = def(0.0f);
        // Runtime octave loop: it must be $for, because a plain C++ loop over a
        // kernel argument unrolls at AST-build time and freezes the count.
        $for(o, octaves) {
            UInt so = seed + cast<uint>(o) * 1013904223u;
            // Billow: the absolute value folds the negative lobes up, turning
            // Perlin ridges into creases.
            sum = sum + abs(perlin(p * f, so)) * a;
            f = f * lacunarity;
            a = a * gain;
        };
        // Unsigned accumulation, then optional inversion (1 - v) as a lerp.
        Float v = clamp(sum * norm, 0.0f, 1.0f);
        v = lerp(v, 1.0f - v, invert);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool Turbulence::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderInt("octaves", &_params.octaves, 1, 8);
    changed |= ImGui::SliderFloat("lacunarity", &_params.lacunarity, 1.5f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("gain", &_params.gain, 0.1f, 0.9f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    changed |= ImGui::Checkbox("invert", &_params.invert);
    ImGui::PopID();
    return changed;
}
#endif

void Turbulence::generate(Image<float> &image, Stream &stream) {
    auto res = static_cast<float>(std::max(1u, image.size().x));
    auto scale = std::max(1e-3f, _params.scale);
    auto freq = scale / res;
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    auto octaves = static_cast<uint>(std::clamp(_params.octaves, 1, 8));
    auto lacunarity = std::clamp(_params.lacunarity, 1.0f, 8.0f);
    auto gain = std::clamp(_params.gain, 0.01f, 0.99f);
    auto norm = fractal_norm(gain, static_cast<int>(octaves));
    auto invert = _params.invert ? 1.0f : 0.0f;
    stream << _shader(image, freq, seed, octaves, lacunarity, gain, norm, invert).dispatch(image.size());
}

luisa::unique_ptr<ProceduralTexture> create_turbulence(Device &device) {
    return luisa::make_unique<Turbulence>(device);
}

}// namespace procedural_tex
