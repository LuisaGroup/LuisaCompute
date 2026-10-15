// Procedural Texture Generator -- ping-pong fractal implementation.

#include "noise_pingpong.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

PingPong::PingPong(Device &device) {
    auto perlin = make_perlin_2d();
    Kernel2D kernel = [&](ImageVar<float> image, Float freq, UInt seed,
                          UInt octaves, Float lacunarity, Float gain,
                          Float strength, Float norm) noexcept {
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
            // Signed noise -> [0, 1], then pushed outwards by `strength` so the
            // value ping-pongs across the fold boundaries.
            Float n01 = perlin(p * f, so) * 0.5f + 0.5f;
            Float t = (n01 + 1.0f) * strength;
            // Triangle wave of period 2 in [0, 1] -- the "ping-pong" fold.
            // select(f, t, p) is p ? t : f. NOTE: at the default strength = 2 the
            // fold evaluates to exactly 1 - |n|, the turbulence billow, so the two
            // textures coincide there; any other strength gives the multi-band
            // sediment structure.
            Float tri = fract(t * 0.5f);
            tri = select(2.0f - tri * 2.0f, tri * 2.0f, tri < 0.5f);
            sum = sum + tri * a;
            f = f * lacunarity;
            a = a * gain;
        };
        // Unsigned accumulation with the host-computed normalization.
        Float v = clamp(sum * norm, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool PingPong::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderInt("octaves", &_params.octaves, 1, 8);
    changed |= ImGui::SliderFloat("lacunarity", &_params.lacunarity, 1.5f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("gain", &_params.gain, 0.1f, 0.9f, "%.2f");
    changed |= ImGui::SliderFloat("strength", &_params.strength, 0.5f, 4.0f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void PingPong::generate(Image<float> &image, Stream &stream) {
    auto res = static_cast<float>(std::max(1u, image.size().x));
    auto scale = std::max(1e-3f, _params.scale);
    auto freq = scale / res;
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    auto octaves = static_cast<uint>(std::clamp(_params.octaves, 1, 8));
    auto lacunarity = std::clamp(_params.lacunarity, 1.0f, 8.0f);
    auto gain = std::clamp(_params.gain, 0.01f, 0.99f);
    auto strength = std::clamp(_params.strength, 0.01f, 8.0f);
    // Triangle waves are in [0, 1], so the divisor is sum gain^o (octaves if gain ~= 1).
    auto norm = (std::fabs(gain - 1.0f) < 1e-4f)
                    ? 1.0f / static_cast<float>(octaves)
                    : (1.0f - gain) / (1.0f - std::pow(gain, static_cast<float>(octaves)));
    stream << _shader(image, freq, seed, octaves, lacunarity, gain, strength, norm).dispatch(image.size());
}

luisa::unique_ptr<ProceduralTexture> create_pingpong(Device &device) {
    return luisa::make_unique<PingPong>(device);
}

}// namespace procedural_tex
