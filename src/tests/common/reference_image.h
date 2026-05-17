#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace luisa::test {

static constexpr double DEFAULT_PSNR_THRESHOLD = 30.0;

// PSNR = 10 * log10(MAX^2 / MSE), where MAX = 255 for 8-bit images
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

inline std::filesystem::path find_reference_dir(const std::filesystem::path &exe_dir) {
    auto candidate = exe_dir;
    for (int i = 0; i < 10; ++i) {
        auto ref = candidate / "docs" / "gallery";
        if (std::filesystem::is_directory(ref)) { return ref; }
        if (!candidate.has_parent_path() || candidate == candidate.parent_path()) break;
        candidate = candidate.parent_path();
    }
    return exe_dir / "docs" / "gallery";
}

inline ReferenceCompareResult compare_with_reference(
    const uint8_t *rendered, int width, int height, int channels,
    const std::string &test_name,
    const std::filesystem::path &reference_dir,
    bool update_reference = false,
    double threshold = DEFAULT_PSNR_THRESHOLD) {

    auto ref_path = reference_dir / (test_name + ".png");

    if (update_reference || !std::filesystem::exists(ref_path)) {
        std::filesystem::create_directories(reference_dir);
        stbi_write_png(ref_path.string().c_str(), width, height, channels,
                       rendered, width * channels);
        return {true, 100.0,
                update_reference ? "reference image updated: " + ref_path.string() : "reference image created: " + ref_path.string()};
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
                         "dB (threshold=" + std::to_string(threshold) + "dB)";
    }

    stbi_image_free(ref_data);
    return result;
}

inline ReferenceCompareResult save_and_compare(
    const uint8_t *rendered, int width, int height, int channels,
    const std::string &test_name,
    const std::filesystem::path &output_dir,
    const std::filesystem::path &reference_dir,
    bool update_reference = false,
    double threshold = DEFAULT_PSNR_THRESHOLD) {

    std::filesystem::create_directories(output_dir);
    auto out_path = output_dir / (test_name + ".png");
    stbi_write_png(out_path.string().c_str(), width, height, channels,
                   rendered, width * channels);

    return compare_with_reference(rendered, width, height, channels,
                                  test_name, reference_dir,
                                  update_reference, threshold);
}

struct ImageTestOptions {
    bool offline{false};
    bool update_reference{false};
    std::string output_dir{"."};

    static ImageTestOptions parse(int argc, const char *const *argv) {
        ImageTestOptions opts;
        for (int i = 1; i < argc; ++i) {
            if (!argv[i]) break;
            std::string arg{argv[i]};
            if (arg == "--offline") {
                opts.offline = true;
            } else if (arg == "--update-reference") {
                opts.update_reference = true;
                opts.offline = true;
            } else if (arg == "--output-dir" && i + 1 < argc) {
                opts.output_dir = argv[++i];
            }
        }
        return opts;
    }
};

}// namespace luisa::test
