// Procedural Texture Generator -- Worley noise, F2 - F1 metric implementation.
//
// Kernel captures the shared make_worley_2d() callable (which itself captures
// make_hash01()); compilation happens once in the ctor, parameter changes only
// re-dispatch.

#include "noise_worley_f2_minus_f1.h"

#include <algorithm>

#include "noise_common.h"

namespace procedural_tex {

WorleyF2MinusF1::WorleyF2MinusF1(Device &device) {
    auto worley = make_worley_2d();
    Kernel2D kernel = [&](ImageVar<float> image, Float scale, Float jitter, Float contrast, UInt seed) noexcept {
        auto coord = dispatch_id().xy();
        Float2 uv = make_float2(cast<float>(coord.x) + 0.5f,
                                 cast<float>(coord.y) + 0.5f) /
                    make_float2(cast<float>(image.size().x), cast<float>(image.size().y));
        // worley returns (f1, f2) scaled to roughly [0, ~1.4] for jitter <= 1.
        Float2 f = worley(uv * scale, seed, jitter);
        // Small f2 - f1 exactly at Voronoi borders -> bright cell outlines.
        Float v = clamp((f.y - f.x) * contrast, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool WorleyF2MinusF1::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("scale", &_params.scale, 2.0f, 64.0f, "%.1f");
    changed |= ImGui::SliderFloat("jitter", &_params.jitter, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("contrast", &_params.contrast, 1.0f, 16.0f, "%.1f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void WorleyF2MinusF1::generate(Image<float> &image, Stream &stream) {
    auto scale = std::max(_params.scale, 1e-3f);
    auto jitter = std::clamp(_params.jitter, 0.0f, 1.0f);
    auto contrast = std::max(_params.contrast, 0.0f);
    auto seed = static_cast<uint>(std::clamp(_params.seed, 0, 255));
    stream << _shader(image, scale, jitter, contrast, seed).dispatch(image.size());
}

bool WorleyF2MinusF1::validate(const luisa::vector<float4> &pixels, uint2 resolution) {
    // Generic invariants (finite, [0, 1], spatially varying).
    if (!ProceduralTexture::validate(pixels, resolution)) {
        return false;
    }
    // The metric must show both dark cell interiors (near 0) and lit borders
    // (near 1).
    auto min_v = 1e30f;
    auto max_v = -1e30f;
    for (auto &p : pixels) {
        min_v = std::min(min_v, p.x);
        max_v = std::max(max_v, p.x);
    }
    if (min_v > 0.05f || max_v < 0.8f) {
        LUISA_WARNING("[worley_f2_minus_f1] expected dark interiors and bright borders (min {} max {})", min_v, max_v);
        return false;
    }
    return true;
}

luisa::unique_ptr<ProceduralTexture> create_worley_f2_minus_f1(Device &device) {
    return luisa::make_unique<WorleyF2MinusF1>(device);
}

}// namespace procedural_tex
