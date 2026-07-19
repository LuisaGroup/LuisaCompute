#pragma once

#include <algorithm>
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
static constexpr double DEFAULT_CORRELATION_THRESHOLD = 0.5;
static constexpr double MIN_CONTRAST_RATIO = 0.25;
static constexpr double MAX_CONTRAST_RATIO = 4.0;

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
    if (img_a == nullptr || img_b == nullptr || width <= 0 || height <= 0 || channels <= 0) { return 0.0; }
    double mse = 0.0;
    auto pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    auto compared_channels = channels == 2 ? 1 : std::min(channels, 3);
    for (size_t pixel = 0u; pixel < pixel_count; ++pixel) {
        auto offset = pixel * static_cast<size_t>(channels);
        for (int channel = 0; channel < compared_channels; ++channel) {
            double diff = static_cast<double>(img_a[offset + static_cast<size_t>(channel)]) -
                          static_cast<double>(img_b[offset + static_cast<size_t>(channel)]);
            mse += diff * diff;
        }
    }
    mse /= static_cast<double>(pixel_count * static_cast<size_t>(compared_channels));
    if (mse < 1e-10) { return 100.0; }
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

struct StructuralMetrics {
    double correlation{0.0};
    double contrast_ratio{0.0};
};

inline StructuralMetrics compute_structural_metrics(
    const uint8_t *img_a, const uint8_t *img_b,
    int width, int height, int channels) noexcept {
    if (img_a == nullptr || img_b == nullptr || width <= 0 || height <= 0 || channels <= 0) { return {}; }
    auto luminance = [channels](const uint8_t *pixel) noexcept {
        if (channels >= 3) {
            return 0.2126 * static_cast<double>(pixel[0]) +
                   0.7152 * static_cast<double>(pixel[1]) +
                   0.0722 * static_cast<double>(pixel[2]);
        }
        return static_cast<double>(pixel[0]);
    };
    auto pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    double mean_a = 0.0;
    double mean_b = 0.0;
    double covariance = 0.0;
    double variance_a = 0.0;
    double variance_b = 0.0;
    for (size_t i = 0u; i < pixel_count; ++i) {
        auto offset = i * static_cast<size_t>(channels);
        auto a = luminance(img_a + offset);
        auto b = luminance(img_b + offset);
        auto count = static_cast<double>(i + 1u);
        auto delta_a = a - mean_a;
        auto delta_b = b - mean_b;
        mean_a += delta_a / count;
        mean_b += delta_b / count;
        variance_a += delta_a * (a - mean_a);
        variance_b += delta_b * (b - mean_b);
        covariance += delta_a * (b - mean_b);
    }
    static constexpr double epsilon = 1e-12;
    if (variance_a <= epsilon && variance_b <= epsilon) {
        return {1.0, 1.0};
    }
    if (variance_a <= epsilon || variance_b <= epsilon) { return {}; }
    return {
        covariance / std::sqrt(variance_a * variance_b),
        std::sqrt(variance_a / variance_b)};
}

struct CompareResult {
    bool passed{false};
    double psnr{0.0};
    std::string message;
    double correlation{0.0};
    double contrast_ratio{0.0};
};

inline CompareResult compare_with_reference_file(
    const uint8_t *rendered, int width, int height, int channels,
    const std::filesystem::path &reference_path,
    double threshold = DEFAULT_PSNR_THRESHOLD) {

    if (rendered == nullptr || width <= 0 || height <= 0 || channels <= 0 || !std::isfinite(threshold)) {
        return {false, 0.0, "invalid rendered image or comparison threshold"};
    }
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
        auto structural = compute_structural_metrics(rendered, ref_data, width, height, channels);
        result.correlation = structural.correlation;
        result.contrast_ratio = structural.contrast_ratio;
        auto metrics_are_finite = std::isfinite(result.psnr) &&
                                  std::isfinite(result.correlation) &&
                                  std::isfinite(result.contrast_ratio);
        result.passed = metrics_are_finite &&
                        result.psnr >= threshold &&
                        result.correlation >= DEFAULT_CORRELATION_THRESHOLD &&
                        result.contrast_ratio >= MIN_CONTRAST_RATIO &&
                        result.contrast_ratio <= MAX_CONTRAST_RATIO;
        result.message = "RGB PSNR=" + std::to_string(result.psnr) +
                         "dB (threshold=" + std::to_string(threshold) +
                         "dB), luminance correlation=" + std::to_string(result.correlation) +
                         " (threshold=" + std::to_string(DEFAULT_CORRELATION_THRESHOLD) +
                         "), contrast ratio=" + std::to_string(result.contrast_ratio) +
                         " (range=" + std::to_string(MIN_CONTRAST_RATIO) +
                         "-" + std::to_string(MAX_CONTRAST_RATIO) +
                         ") ref=" + reference_path.string();
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
