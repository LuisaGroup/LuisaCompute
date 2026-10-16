// Procedural Texture Generator -- Perlin fBm / fractal Worley blend implementation.

#include "noise_perlin_worley.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

PerlinWorley::PerlinWorley(Device &device) {
    auto perlin = make_perlin_2d();
    auto worley = make_worley_2d();
    auto remap = make_remap();
    // Signed Perlin fBm, normalized by the host-computed amplitude sum.
    auto perlin_fbm = Callable{[=](Float2 p, UInt octaves, Float lacunarity,
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
    // Fractal Worley F1 (unsigned), normalized the same way.
    auto worley_fbm = Callable{[=](Float2 p, UInt octaves, Float lacunarity,
                                   Float gain, Float jitter, UInt seed,
                                   Float norm) noexcept {
        Float f = def(1.0f);
        Float a = def(1.0f);
        Float sum = def(0.0f);
        $for(o, octaves) {
            UInt so = seed + cast<uint>(o) * 1013904223u;
            sum = sum + worley(p * f, so, jitter).x * a;
            f = f * lacunarity;
            a = a * gain;
        };
        return sum * norm;
    }};
    Kernel2D kernel = [&](ImageVar<float> image, Float freq, UInt seed,
                          UInt octaves, Float lacunarity, Float gain,
                          Float jitter, Float worley_weight,
                          Float perlin_norm, Float worley_norm) noexcept {
        auto coord = dispatch_id().xy();
        Float2 p = make_float2(cast<float>(coord.x), cast<float>(coord.y)) * freq;// lattice / cell units
        Float pfbm = perlin_fbm(p, octaves, lacunarity, gain, seed, perlin_norm);
        Float pn = pfbm * 0.5f + 0.5f;// signed -> [0, 1]
        Float w = worley_fbm(p, octaves, lacunarity, gain, jitter, seed + 101u, worley_norm);
        Float wi = 1.0f - clamp(w, 0.0f, 1.0f);// 1 at cell centers, 0 at borders
        // Carve the Perlin density with the inverted Worley field.
        Float base = remap(pn, wi * worley_weight, 1.0f, 0.0f, 1.0f);
        Float v = clamp(base, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool PerlinWorley::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 32.0f, "%.1f");
    changed |= ImGui::SliderInt("octaves", &_params.octaves, 1, 8);
    changed |= ImGui::SliderFloat("lacunarity", &_params.lacunarity, 1.5f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("gain", &_params.gain, 0.1f, 0.9f, "%.2f");
    changed |= ImGui::SliderFloat("jitter", &_params.jitter, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("worley_weight", &_params.worley_weight, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void PerlinWorley::generate(Image<float> &image, Stream &stream) {
    auto res = static_cast<float>(std::max(1u, image.size().x));
    auto scale = std::max(1e-3f, _params.scale);
    auto freq = scale / res;
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    auto octaves = static_cast<uint>(std::clamp(_params.octaves, 1, 8));
    auto lacunarity = std::clamp(_params.lacunarity, 1.0f, 8.0f);
    auto gain = std::clamp(_params.gain, 0.01f, 0.99f);
    auto jitter = std::clamp(_params.jitter, 0.0f, 1.0f);
    auto worley_weight = std::clamp(_params.worley_weight, 0.0f, 1.0f);
    // 1 / sum_{o<octaves} gain^o, guarded at gain ~= 1.
    auto amp_sum = (std::fabs(gain - 1.0f) < 1e-4f)
                       ? static_cast<float>(octaves)
                       : (1.0f - std::pow(gain, static_cast<float>(octaves))) / (1.0f - gain);
    auto perlin_norm = 1.0f / amp_sum;
    // The shared Worley callable scales its distances by 1.2.
    auto worley_norm = 1.0f / (amp_sum * 1.2f);
    stream << _shader(image, freq, seed, octaves, lacunarity, gain, jitter,
                      worley_weight, perlin_norm, worley_norm)
                  .dispatch(image.size());
}

luisa::unique_ptr<ProceduralTexture> create_perlin_worley(Device &device) {
    return luisa::make_unique<PerlinWorley>(device);
}

}// namespace procedural_tex
