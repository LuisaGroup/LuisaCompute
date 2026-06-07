#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "../../src/ext/stb/stb/stb_image.h"
#include "../../src/ext/stb/stb/stb_image_write.h"

namespace luisa::ref {

static constexpr double DEFAULT_PSNR_THRESHOLD = 30.0;

struct ExampleOptions {
    bool offline{false};
    uint32_t spp{0u};
    std::optional<std::filesystem::path> compare_path;
    std::optional<std::filesystem::path> out_ref_path;
    bool out_ref_write{false};

    static ExampleOptions parse(int argc, char *argv[]) {
        ExampleOptions opts;
        for (int i = 2; i < argc; i++) {
            if (!argv[i]) break;
            std::string_view a{argv[i]};
            if (a == "--offline") {
                opts.offline = true;
            } else if ((a == "--compare" || a == "-c") && i + 1 < argc && argv[i + 1]) {
                opts.compare_path = std::filesystem::path{argv[++i]};
                opts.offline = true;
            } else if (a == "--spp" && i + 1 < argc && argv[i + 1]) {
                opts.spp = static_cast<uint32_t>(std::atoi(argv[++i]));
            } else if (a == "--out_ref" && i + 1 < argc && argv[i + 1]) {
                std::string_view mode{argv[++i]};
                if (mode == "write" && i + 1 < argc && argv[i + 1]) {
                    opts.out_ref_path = std::filesystem::path{argv[++i]};
                    opts.out_ref_write = true;
                    opts.offline = true;
                } else if (mode == "read" && i + 1 < argc && argv[i + 1]) {
                    opts.out_ref_path = std::filesystem::path{argv[++i]};
                    opts.out_ref_write = false;
                    opts.offline = true;
                }
            }
        }
        return opts;
    }
};

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

struct CompareResult {
    bool passed{false};
    double psnr{0.0};
    std::string message;
};

inline CompareResult compare_with_reference_file(
    const uint8_t *rendered, int width, int height, int channels,
    const std::filesystem::path &reference_path,
    double threshold = DEFAULT_PSNR_THRESHOLD) {

    if (!std::filesystem::exists(reference_path)) {
        return {false, 0.0, "reference not found: " + reference_path.string()};
    }
    int ref_w = 0, ref_h = 0, ref_c = 0;
    auto *ref_data = stbi_load(reference_path.string().c_str(), &ref_w, &ref_h, &ref_c, channels);
    if (!ref_data) {
        return {false, 0.0, "failed to load reference: " + reference_path.string()};
    }
    CompareResult result;
    if (ref_w != width || ref_h != height) {
        result = {false, 0.0,
                  "resolution mismatch: rendered " + std::to_string(width) + "x" +
                      std::to_string(height) + " vs reference " +
                      std::to_string(ref_w) + "x" + std::to_string(ref_h) +
                      " (" + reference_path.string() + ")"};
    } else {
        result.psnr = compute_psnr(rendered, ref_data, width, height, channels);
        result.passed = result.psnr >= threshold;
        result.message = "PSNR=" + std::to_string(result.psnr) +
                         "dB (threshold=" + std::to_string(threshold) +
                         "dB) ref=" + reference_path.string();
    }
    stbi_image_free(ref_data);
    return result;
}

inline std::optional<std::filesystem::path> parse_compare_arg(int argc, const char *const *argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (!argv[i] || !argv[i + 1]) { break; }
        std::string_view a{argv[i]};
        if (a == "--compare" || a == "-c") {
            return std::filesystem::path{argv[i + 1]};
        }
    }
    return std::nullopt;
}

}// namespace luisa::ref
