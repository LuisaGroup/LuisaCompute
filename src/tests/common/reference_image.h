#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace luisa::test {

static constexpr double DEFAULT_PSNR_THRESHOLD = 30.0;

inline double compute_psnr(const uint8_t *img_a, const uint8_t *img_b,
                           int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) { return 0.0; }
    double mse = 0.0;
    auto total = static_cast<size_t>(width) * height * channels;
    for (size_t i = 0; i < total; ++i) {
        double diff = static_cast<double>(img_a[i]) - static_cast<double>(img_b[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(total);
    if (mse < 1e-10) { return 100.0; }
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

struct ReferenceCompareResult {
    bool passed{false};
    double psnr{0.0};
    std::string message;
};

inline ReferenceCompareResult compare_with_reference_file(
    const uint8_t *rendered, int width, int height, int channels,
    const std::filesystem::path &ref_path,
    double threshold = DEFAULT_PSNR_THRESHOLD) {

    if (!std::filesystem::exists(ref_path)) {
        return {false, 0.0, "reference file does not exist: " + ref_path.string()};
    }
    int ref_w = 0, ref_h = 0, ref_c = 0;
    auto *ref_data = stbi_load(ref_path.string().c_str(), &ref_w, &ref_h, &ref_c, channels);
    if (!ref_data) {
        return {false, 0.0, "failed to load reference: " + ref_path.string()};
    }
    ReferenceCompareResult result;
    if (ref_w != width || ref_h != height) {
        result = {false, 0.0,
                  "resolution mismatch: rendered " + std::to_string(width) + "x" +
                      std::to_string(height) + " vs reference " +
                      std::to_string(ref_w) + "x" + std::to_string(ref_h)};
    } else {
        result.psnr = compute_psnr(rendered, ref_data, width, height, channels);
        result.passed = result.psnr >= threshold;
        result.message = "PSNR=" + std::to_string(result.psnr) +
                         "dB (threshold=" + std::to_string(threshold) + "dB) vs " +
                         ref_path.string();
    }
    stbi_image_free(ref_data);
    return result;
}

inline std::optional<std::filesystem::path> parse_compare_arg(int argc, const char *const *argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (!argv[i] || !argv[i + 1]) break;
        std::string_view a{argv[i]};
        if (a == "--compare" || a == "-c") {
            return std::filesystem::path{argv[i + 1]};
        }
    }
    return std::nullopt;
}

struct ImageTestOptions {
    bool offline{false};
    std::string output_dir{"."};
    std::optional<std::filesystem::path> compare_path;

    static ImageTestOptions parse(int argc, const char *const *argv) {
        ImageTestOptions opts;
        for (int i = 1; i < argc; ++i) {
            if (!argv[i]) break;
            std::string arg{argv[i]};
            if (arg == "--offline") {
                opts.offline = true;
            } else if (arg == "--output-dir" && i + 1 < argc) {
                opts.output_dir = argv[++i];
            } else if ((arg == "--compare" || arg == "-c") && i + 1 < argc) {
                opts.compare_path = std::filesystem::path{argv[++i]};
                opts.offline = true;
            }
        }
        return opts;
    }
};

}// namespace luisa::test
