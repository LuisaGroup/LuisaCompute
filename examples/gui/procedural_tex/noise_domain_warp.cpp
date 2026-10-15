// Procedural Texture Generator -- domain-warped fBm implementation.

#include "noise_domain_warp.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

DomainWarpFbm::DomainWarpFbm(Device &device) {
    auto perlin = make_perlin_2d();
    // Perlin fBm helper with runtime octave count: a Callable may contain a $for
    // loop over its Var arguments, so the same compiled body is shared by all
    // five warp samples. Returns the signed sum in ~[-1, 1] (normalized).
    auto fbm = Callable{[=](Float2 p, UInt octaves, Float lacunarity,
                            Float gain, UInt seed, Float norm) noexcept {
        Float f = def(1.0f);
        Float a = def(1.0f);
        Float sum = def(0.0f);
        $for(o, octaves) {
            UInt so = seed + cast<uint>(o) * 1013904223u;
            sum = sum + perlin(p * f, so) * a;
            f = f * lacunarity;
            a = a * gain;
        };
        return sum * norm;
    }};
    Kernel2D kernel = [&](ImageVar<float> image, Float freq, Float warp,
                          UInt seed, UInt octaves, Float lacunarity,
                          Float gain, Float norm) noexcept {
        auto coord = dispatch_id().xy();
        Float2 p = make_float2(cast<float>(coord.x), cast<float>(coord.y)) * freq;// lattice units
        // First warp stage.
        Float2 q = make_float2(fbm(p, octaves, lacunarity, gain, seed + 11u, norm),
                               fbm(p + make_float2(5.2f, 1.3f), octaves, lacunarity, gain, seed + 23u, norm));
        // Second warp stage, driven by the first.
        Float2 r = make_float2(fbm(p + warp * q + make_float2(1.7f, 9.2f), octaves, lacunarity, gain, seed + 37u, norm),
                               fbm(p + warp * q + make_float2(8.3f, 2.8f), octaves, lacunarity, gain, seed + 51u, norm));
        // Final fBm sampled at the doubly-warped position.
        Float c = fbm(p + warp * r, octaves, lacunarity, gain, seed + 67u, norm);
        // Signed sum -> grayscale.
        Float v = clamp(c * 0.5f + 0.5f, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool DomainWarpFbm::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 32.0f, "%.1f");
    changed |= ImGui::SliderFloat("warp", &_params.warp, 0.0f, 8.0f, "%.1f");
    changed |= ImGui::SliderInt("octaves", &_params.octaves, 1, 8);
    changed |= ImGui::SliderFloat("lacunarity", &_params.lacunarity, 1.5f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("gain", &_params.gain, 0.1f, 0.9f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void DomainWarpFbm::generate(Image<float> &image, Stream &stream) {
    auto res = static_cast<float>(std::max(1u, image.size().x));
    auto scale = std::max(1e-3f, _params.scale);
    auto freq = scale / res;
    auto warp = std::clamp(_params.warp, 0.0f, 16.0f);
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    auto octaves = static_cast<uint>(std::clamp(_params.octaves, 1, 8));
    auto lacunarity = std::clamp(_params.lacunarity, 1.0f, 8.0f);
    auto gain = std::clamp(_params.gain, 0.01f, 0.99f);
    // 1 / sum_{o<octaves} gain^o, guarded at gain ~= 1.
    auto amp_sum = (std::fabs(gain - 1.0f) < 1e-4f)
                       ? static_cast<float>(octaves)
                       : (1.0f - std::pow(gain, static_cast<float>(octaves))) / (1.0f - gain);
    auto norm = 1.0f / amp_sum;
    stream << _shader(image, freq, warp, seed, octaves, lacunarity, gain, norm).dispatch(image.size());
}

luisa::unique_ptr<ProceduralTexture> create_domain_warp(Device &device) {
    return luisa::make_unique<DomainWarpFbm>(device);
}

}// namespace procedural_tex
