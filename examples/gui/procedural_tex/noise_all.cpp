// Procedural Texture Generator -- algorithm registry and default validation.
// Every factory below lives in its own noise_*.cpp translation unit.

#include <cmath>

#include "procedural_texture.h"

namespace procedural_tex {

bool ProceduralTexture::validate(const luisa::vector<float4> &pixels, uint2 resolution) {
    if (pixels.size() != resolution.x * resolution.y) {
        LUISA_WARNING("pixel count mismatch: {} vs {}x{}", pixels.size(), resolution.x, resolution.y);
        return false;
    }
    // Generic invariants: finite, in [0, 1], and spatially varying.
    auto min_v = 1e30f;
    auto max_v = -1e30f;
    auto sum = 0.0;
    for (auto &p : pixels) {
        for (auto c = 0u; c < 3u; ++c) {
            auto v = p[c];
            if (!std::isfinite(v) || v < 0.0f || v > 1.0f) {
                LUISA_WARNING("pixel out of range: channel {} = {}", c, v);
                return false;
            }
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            sum += v;
        }
    }
    auto mean = static_cast<float>(sum / (3.0 * pixels.size()));
    if (max_v - min_v < 1e-3f) {
        LUISA_WARNING("texture is constant (min {}, max {})", min_v, max_v);
        return false;
    }
    LUISA_VERBOSE("validate: min {:.3f} max {:.3f} mean {:.3f}", min_v, max_v, mean);
    return true;
}

// Factories (one per noise_*.cpp).
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_white_noise(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_gaussian_noise(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_value_noise(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_perlin_noise(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_simplex_noise(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_worley_f1(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_worley_f2_minus_f1(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_plasma(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_fbm(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_turbulence(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_ridged_multifractal(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_pingpong(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_hetero_terrain(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_fractal_worley(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_domain_warp(Device &device);
[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_perlin_worley(Device &device);

luisa::vector<luisa::unique_ptr<ProceduralTexture>> create_all_procedural_textures(Device &device) {
    luisa::vector<luisa::unique_ptr<ProceduralTexture>> all;
    all.emplace_back(create_white_noise(device));
    all.emplace_back(create_gaussian_noise(device));
    all.emplace_back(create_value_noise(device));
    all.emplace_back(create_perlin_noise(device));
    all.emplace_back(create_simplex_noise(device));
    all.emplace_back(create_worley_f1(device));
    all.emplace_back(create_worley_f2_minus_f1(device));
    all.emplace_back(create_plasma(device));
    all.emplace_back(create_fbm(device));
    all.emplace_back(create_turbulence(device));
    all.emplace_back(create_ridged_multifractal(device));
    all.emplace_back(create_pingpong(device));
    all.emplace_back(create_hetero_terrain(device));
    all.emplace_back(create_fractal_worley(device));
    all.emplace_back(create_domain_warp(device));
    all.emplace_back(create_perlin_worley(device));
    return all;
}

}// namespace procedural_tex
