// Procedural Texture Generator -- white noise implementation.
//
// Per-pixel (per-block) hash white noise. The kernel captures the shared
// make_hash01() callable created in the constructor; compilation happens once,
// parameter changes only re-dispatch.

#include "noise_white.h"

#include <algorithm>

#include "noise_common.h"

namespace procedural_tex {

WhiteNoise::WhiteNoise(Device &device) {
    auto hash = make_hash01();
    Kernel2D kernel = [&](ImageVar<float> image, UInt scale, UInt seed) noexcept {
        auto coord = dispatch_id().xy();
        // Coarse block coordinate: scale cells across the image.
        UInt bx = coord.x / scale;
        UInt by = coord.y / scale;
        // Decorrelate x/y channels of the hash with different seed mixing.
        Float v = hash(bx + 53u * seed, by + 97u * seed);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool WhiteNoise::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    changed |= ImGui::SliderInt("scale", &_params.scale, 1, 128);
    ImGui::PopID();
    return changed;
}
#endif

void WhiteNoise::generate(Image<float> &image, Stream &stream) {
    // scale >= 1 so the integer division in the kernel is well-defined.
    auto scale = static_cast<uint>(std::max(1, _params.scale));
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    stream << _shader(image, scale, seed).dispatch(image.size());
}

bool WhiteNoise::validate(const luisa::vector<float4> &pixels, uint2 resolution) {
    // Generic invariants (finite, [0, 1], spatially varying).
    if (!ProceduralTexture::validate(pixels, resolution)) {
        return false;
    }
    // White noise is high-contrast: the grayscale range must span at least ~0.5.
    auto min_v = 1e30f;
    auto max_v = -1e30f;
    for (auto &p : pixels) {
        min_v = std::min(min_v, p.x);
        max_v = std::max(max_v, p.x);
    }
    if (max_v - min_v < 0.5f) {
        LUISA_WARNING("[white_noise] contrast too low for white noise (min {} max {})", min_v, max_v);
        return false;
    }
    return true;
}

luisa::unique_ptr<ProceduralTexture> create_white_noise(Device &device) {
    return luisa::make_unique<WhiteNoise>(device);
}

}// namespace procedural_tex
