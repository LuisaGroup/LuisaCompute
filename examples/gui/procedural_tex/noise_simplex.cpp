// Procedural Texture Generator -- 2D simplex noise implementation.

#include "noise_simplex.h"

#include <algorithm>

#include "noise_common.h"

namespace procedural_tex {

SimplexNoise::SimplexNoise(Device &device) {
    auto simplex = make_simplex_2d();
    Kernel2D kernel = [&](ImageVar<float> image, Float freq, UInt seed) noexcept {
        auto coord = dispatch_id().xy();
        Float2 p = make_float2(cast<float>(coord.x), cast<float>(coord.y)) * freq;// simplex-lattice units
        Float n = simplex(p, seed);                        // ~[-1, 1]
        Float v = clamp(n * 0.5f + 0.5f, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool SimplexNoise::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 1.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void SimplexNoise::generate(Image<float> &image, Stream &stream) {
    auto res = static_cast<float>(std::max(1u, image.size().x));
    auto scale = std::max(1e-3f, _params.scale);
    auto freq = scale / res;
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    stream << _shader(image, freq, seed).dispatch(image.size());
}

luisa::unique_ptr<ProceduralTexture> create_simplex_noise(Device &device) {
    return luisa::make_unique<SimplexNoise>(device);
}

}// namespace procedural_tex
