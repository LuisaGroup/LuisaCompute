// Procedural Texture Generator -- Worley (cellular) noise, F1 metric.
//
// Per-pixel Euclidean distance to the nearest feature point of the cellular
// lattice via the shared make_worley_2d() callable. The (F1) component is
// written as grayscale, clamped to [0, 1]. Compilation happens once in the
// ctor; parameter changes only re-dispatch.

#include "noise_worley_f1.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

WorleyF1::WorleyF1(Device &device) {
    auto worley = make_worley_2d();
    Kernel2D kernel = [&](ImageVar<float> image, Float scale, Float jitter, UInt seed) noexcept {
        auto coord = dispatch_id().xy();
        Float2 uv = make_float2(cast<float>(coord.x) + 0.5f,
                                 cast<float>(coord.y) + 0.5f) /
                    make_float2(cast<float>(image.size().x), cast<float>(image.size().y));
        // worley returns (f1, f2) scaled to roughly [0, ~1.4] for jitter <= 1.
        Float2 f = worley(uv * scale, seed, jitter);
        Float v = clamp(f.x, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool WorleyF1::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 2.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderFloat("jitter", &_params.jitter, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void WorleyF1::generate(Image<float> &image, Stream &stream) {
    auto scale = std::max(_params.scale, 1e-3f);
    auto jitter = std::clamp(_params.jitter, 0.0f, 1.0f);
    auto seed = static_cast<uint>(std::clamp(_params.seed, 0, 255));
    stream << _shader(image, scale, jitter, seed).dispatch(image.size());
}

bool WorleyF1::validate(const luisa::vector<float4> &pixels, uint2 resolution) {
    // Generic invariants (finite, [0, 1], spatially varying).
    if (!ProceduralTexture::validate(pixels, resolution)) {
        return false;
    }
    // F1 covers most of the grayscale range: min near 0 (at feature points) and
    // max well above the cell centers.
    auto min_v = 1e30f;
    auto max_v = -1e30f;
    for (auto &p : pixels) {
        min_v = std::min(min_v, p.x);
        max_v = std::max(max_v, p.x);
    }
    if (max_v < 0.4f || min_v > 0.1f) {
        LUISA_WARNING("[worley_f1] unexpected F1 range (min {} max {})", min_v, max_v);
        return false;
    }
    return true;
}

luisa::unique_ptr<ProceduralTexture> create_worley_f1(Device &device) {
    return luisa::make_unique<WorleyF1>(device);
}

}// namespace procedural_tex
