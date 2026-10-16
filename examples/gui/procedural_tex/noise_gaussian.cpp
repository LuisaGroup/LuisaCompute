// Procedural Texture Generator -- Gaussian noise implementation.
//
// Box-Muller normal deviates from two decorrelated uniform hashes; mapped to a
// grayscale color 0.5 + sigma * z, clamped to [0, 1].

#include "noise_gaussian.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

GaussianNoise::GaussianNoise(Device &device) {
    auto hash = make_hash01();
    Kernel2D kernel = [&](ImageVar<float> image, UInt seed, Float sigma) noexcept {
        auto coord = dispatch_id().xy();
        UInt px = coord.x;
        UInt py = coord.y;
        // Two decorrelated uniforms: mix distinct large primes into the lattice
        // coordinates so the two hash inputs land in different parts of the
        // hash's mixing space.
        Float u1 = hash(px + 1321u * seed, py + 1327u * seed);
        Float u2 = hash(px + 8191u + 2311u * seed, py + 131071u + 2333u * seed);
        // Box-Muller: z = sqrt(-2 ln u1) * cos(2 pi u2) ~ N(0, 1).
        Float r = sqrt(-2.0f * log(max(u1, 1e-6f)));
        Float theta = 2.0f * pi * u2;
        Float z = r * cos(theta);
        Float v = clamp(0.5f + sigma * z, 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _shader = device.compile<2>(kernel);
}

#if ENABLE_DISPLAY
bool GaussianNoise::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    changed |= ImGui::SliderFloat("sigma", &_params.sigma, 0.02f, 0.5f, "%.3f");
    ImGui::PopID();
    return changed;
}
#endif

void GaussianNoise::generate(Image<float> &image, Stream &stream) {
    auto seed = static_cast<uint>(std::max(0, _params.seed));
    auto sigma = _params.sigma;
    stream << _shader(image, seed, sigma).dispatch(image.size());
}

bool GaussianNoise::validate(const luisa::vector<float4> &pixels, uint2 resolution) {
    // Generic invariants (finite, [0, 1], spatially varying).
    if (!ProceduralTexture::validate(pixels, resolution)) {
        return false;
    }
    // Gaussian centered at 0.5: the mean must lie in [0.4, 0.6].
    double sum = 0.0;
    for (auto &p : pixels) {
        sum += p.x;
    }
    auto mean = static_cast<float>(sum / pixels.size());
    if (mean < 0.4f || mean > 0.6f) {
        LUISA_WARNING("[gaussian_noise] mean {:.3f} outside [0.4, 0.6]", mean);
        return false;
    }
    return true;
}

luisa::unique_ptr<ProceduralTexture> create_gaussian_noise(Device &device) {
    return luisa::make_unique<GaussianNoise>(device);
}

}// namespace procedural_tex
