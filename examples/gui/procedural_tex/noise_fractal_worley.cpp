// Procedural Texture Generator -- fractal Worley implementation.

#include "noise_fractal_worley.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

FractalWorley::FractalWorley(Device &device) {
    auto worley = make_worley_2d();
    Kernel2D kernel = [&](ImageVar<float> image, Float freq, UInt seed,
                          UInt octaves, Float lacunarity, Float gain,
                          Float jitter, Float norm) noexcept {
        auto coord = dispatch_id().xy();
        Float2 p = make_float2(cast<float>(coord.x), cast<float>(coord.y)) * freq;// cell units
        // p is already in base lattice/cell units; the octave multiplier
        // therefore starts at 1.0 (using freq here would square the base frequency).
        Float f = def(1.0f);
        Float a = def(1.0f);
        Float sum = def(0.0f);
        // Runtime octave loop ($for): the count is a kernel argument. Each call
        // does a 3x3 cell scan inside the shared callable -- kept at 3x3 on
        // purpose, this is already the expensive fractal of the catalog.
        $for(o, octaves) {
            UInt so = seed + cast<uint>(o) * 1013904223u;
            Float2 w = worley(p * f, so, jitter);
            sum = sum + w.x * a;// F1 only
            f = f * lacunarity;
            a = a * gain;
        };
        // Unsigned accumulation (F1 is >= 0), normalized + range-compensated
        // host-side.
        Float v = clamp(sum * norm, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool FractalWorley::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderInt("octaves", &_params.octaves, 1, 8);
    changed |= ImGui::SliderFloat("lacunarity", &_params.lacunarity, 1.5f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("gain", &_params.gain, 0.1f, 0.9f, "%.2f");
    changed |= ImGui::SliderFloat("jitter", &_params.jitter, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void FractalWorley::generate(Image<float> &image, Stream &stream) {
    auto res = static_cast<float>(std::max(1u, image.size().x));
    auto scale = std::max(1e-3f, _params.scale);
    auto freq = scale / res;
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    auto octaves = static_cast<uint>(std::clamp(_params.octaves, 1, 8));
    auto lacunarity = std::clamp(_params.lacunarity, 1.0f, 8.0f);
    auto gain = std::clamp(_params.gain, 0.01f, 0.99f);
    auto jitter = std::clamp(_params.jitter, 0.0f, 1.0f);
    // 1 / sum_{o<octaves} gain^o, guarded at gain ~= 1; divided by the 1.2 the
    // shared callable multiplies its distances with.
    auto amp_sum = (std::fabs(gain - 1.0f) < 1e-4f)
                       ? static_cast<float>(octaves)
                       : (1.0f - std::pow(gain, static_cast<float>(octaves))) / (1.0f - gain);
    auto norm = 1.0f / (amp_sum * 1.2f);
    stream << _shader(image, freq, seed, octaves, lacunarity, gain, jitter, norm).dispatch(image.size());
}

luisa::unique_ptr<ProceduralTexture> create_fractal_worley(Device &device) {
    return luisa::make_unique<FractalWorley>(device);
}

}// namespace procedural_tex
