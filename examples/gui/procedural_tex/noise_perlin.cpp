// Procedural Texture Generator -- improved Perlin gradient noise implementation.

#include "noise_perlin.h"

#include <algorithm>

#include "noise_common.h"

namespace procedural_tex {

PerlinNoise::PerlinNoise(Device &device) {
    auto perlin = make_perlin_2d();
    Kernel2D kernel = [&](ImageVar<float> image, Float freq, UInt seed) noexcept {
        auto coord = dispatch_id().xy();
        Float2 p = make_float2(cast<float>(coord.x), cast<float>(coord.y)) * freq;// lattice units
        Float n = perlin(p, seed);                         // ~[-1, 1]
        Float v = clamp(n * 0.5f + 0.5f, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool PerlinNoise::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void PerlinNoise::generate(Image<float> &image, Stream &stream) {
    auto res = static_cast<float>(std::max(1u, image.size().x));
    auto scale = std::max(1e-3f, _params.scale);
    auto freq = scale / res;
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    stream << _shader(image, freq, seed).dispatch(image.size());
}

luisa::unique_ptr<ProceduralTexture> create_perlin_noise(Device &device) {
    return luisa::make_unique<PerlinNoise>(device);
}

}// namespace procedural_tex
