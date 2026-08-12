#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "../../src/ext/stb/stb/stb_image.h"
#include "../../src/ext/stb/stb/stb_image_write.h"

namespace luisa::ref {

static constexpr double DEFAULT_PSNR_THRESHOLD = 30.0;
static constexpr double DEFAULT_CORRELATION_THRESHOLD = 0.5;
static constexpr double MIN_CONTRAST_RATIO = 0.25;
static constexpr double MAX_CONTRAST_RATIO = 4.0;

[[nodiscard]] inline std::optional<uint32_t> parse_uint32_option_value(
    std::string_view value) noexcept {
    if (value.empty()) { return std::nullopt; }
    uint32_t parsed_value = 0u;
    for (auto c : value) {
        if (c < '0' || c > '9') { return std::nullopt; }
        auto digit = static_cast<uint32_t>(c - '0');
        if (parsed_value >
            (std::numeric_limits<uint32_t>::max() - digit) / 10u) {
            return std::nullopt;
        }
        parsed_value = parsed_value * 10u + digit;
    }
    return parsed_value;
}

struct ExampleOptions {
    bool offline{false};
    uint32_t spp{0u};
    uint32_t iterations{1u};
    std::optional<uint32_t> max_spp_per_dispatch;
    std::optional<std::filesystem::path> compare_path;
    std::optional<std::filesystem::path> out_ref_path;
    bool out_ref_write{false};
    std::string error_message;

    [[nodiscard]] bool valid() const noexcept { return error_message.empty(); }

    [[nodiscard]] static ExampleOptions parse(int argc, char *argv[]) {
        ExampleOptions opts;
        auto fail = [&opts](std::string message) {
            opts.error_message = std::move(message);
            return opts;
        };
        auto missing_value = [](int index, int count, char *arguments[]) noexcept {
            return index + 1 >= count || arguments[index + 1] == nullptr ||
                   std::string_view{arguments[index + 1]}.empty();
        };
        auto option_token = [](std::string_view value) noexcept {
            return value.size() > 1u && value.front() == '-';
        };
        for (int i = 2; i < argc; i++) {
            if (!argv[i]) break;
            std::string_view a{argv[i]};
            if (a == "--offline") {
                opts.offline = true;
            } else if (a == "--compare" || a == "-c") {
                if (missing_value(i, argc, argv) ||
                    option_token(std::string_view{argv[i + 1]})) {
                    return fail("Missing value for " + std::string{a} + ".");
                }
                opts.compare_path = std::filesystem::path{argv[++i]};
                opts.offline = true;
            } else if (a == "--spp") {
                if (missing_value(i, argc, argv)) {
                    return fail("Missing value for --spp.");
                }
                std::string_view value{argv[++i]};
                auto parsed_value = parse_uint32_option_value(value);
                if (!parsed_value) {
                    return fail("Invalid unsigned integer for --spp: '" +
                                std::string{value} + "'.");
                }
                opts.spp = *parsed_value;
            } else if (a == "--iterations") {
                if (missing_value(i, argc, argv)) {
                    return fail("Missing value for --iterations.");
                }
                std::string_view value{argv[++i]};
                auto parsed_value = parse_uint32_option_value(value);
                if (!parsed_value || *parsed_value == 0u) {
                    return fail("Invalid positive integer for --iterations: '" +
                                std::string{value} + "'.");
                }
                opts.iterations = *parsed_value;
            } else if (a == "--max-spp-per-dispatch") {
                if (missing_value(i, argc, argv)) {
                    return fail("Missing value for --max-spp-per-dispatch.");
                }
                std::string_view value{argv[++i]};
                auto parsed_value = parse_uint32_option_value(value);
                if (!parsed_value || *parsed_value == 0u) {
                    return fail("Invalid positive integer for --max-spp-per-dispatch: '" +
                                std::string{value} + "'.");
                }
                opts.max_spp_per_dispatch = *parsed_value;
            } else if (a == "--out_ref") {
                if (missing_value(i, argc, argv)) {
                    return fail("Missing mode for --out_ref; expected 'write <path>' or 'read <path>'.");
                }
                std::string_view mode{argv[++i]};
                if (mode != "write" && mode != "read") {
                    return fail("Invalid mode for --out_ref: '" +
                                std::string{mode} +
                                "'; expected 'write' or 'read'.");
                }
                if (missing_value(i, argc, argv) ||
                    option_token(std::string_view{argv[i + 1]})) {
                    return fail("Missing path for --out_ref " +
                                std::string{mode} + ".");
                }
                opts.out_ref_path = std::filesystem::path{argv[++i]};
                opts.out_ref_write = mode == "write";
                opts.offline = true;
            }
            // Unknown arguments intentionally remain in argv so extension
            // parsers (for example coroutine schedulers) can consume them.
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

// Sparse particle simulations are not pixel-deterministic across GPU
// schedulers: unordered floating-point atomics perturb individual particles,
// while the physical distribution remains stable. Moments capture its center
// and spread, while a normalized 16x16 density grid rejects spatially
// rearranged mass with the same low-order moments. Coordinates and covariance
// are normalized by image dimensions so thresholds are resolution-independent.
struct ForegroundMoments {
    static constexpr auto density_grid_size = 16u;
    static constexpr auto density_bin_count =
        density_grid_size * density_grid_size;

    bool valid{false};
    size_t count{0u};
    double coverage{0.0};
    double centroid_x{0.0};
    double centroid_y{0.0};
    double covariance_xx{0.0};
    double covariance_xy{0.0};
    double covariance_yy{0.0};
    std::array<double, density_bin_count> density{};
};

inline ForegroundMoments compute_foreground_moments(
    const uint8_t *image, int width, int height, int channels,
    const std::array<uint8_t, 3u> &background,
    uint8_t background_tolerance = 0u) noexcept {
    ForegroundMoments result;
    if (image == nullptr || width <= 0 || height <= 0 || channels < 3) {
        return result;
    }
    auto w = static_cast<size_t>(width);
    auto h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / h) { return result; }
    auto pixel_count = w * h;
    auto channel_count = static_cast<size_t>(channels);
    if (pixel_count >
        std::numeric_limits<size_t>::max() / channel_count) {
        return result;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;
    double moment_xx = 0.0;
    double moment_xy = 0.0;
    double moment_yy = 0.0;
    for (auto y = 0; y < height; ++y) {
        for (auto x = 0; x < width; ++x) {
            auto offset = (static_cast<size_t>(y) * w +
                           static_cast<size_t>(x)) *
                          channel_count;
            auto foreground = false;
            for (auto channel = 0u; channel < 3u; ++channel) {
                auto difference = std::abs(
                    static_cast<int>(image[offset + channel]) -
                    static_cast<int>(background[channel]));
                foreground |= difference >
                              static_cast<int>(background_tolerance);
            }
            if (!foreground) { continue; }

            // Pixel centers avoid assigning an exact coordinate of zero to a
            // foreground sample on the image boundary.
            auto px = (static_cast<double>(x) + 0.5) /
                      static_cast<double>(width);
            auto py = (static_cast<double>(y) + 0.5) /
                      static_cast<double>(height);
            result.count++;
            auto count = static_cast<double>(result.count);
            auto delta_x = px - mean_x;
            auto delta_y = py - mean_y;
            mean_x += delta_x / count;
            mean_y += delta_y / count;
            moment_xx += delta_x * (px - mean_x);
            moment_xy += delta_x * (py - mean_y);
            moment_yy += delta_y * (py - mean_y);
            auto bin_x = std::min(
                static_cast<size_t>(ForegroundMoments::density_grid_size - 1u),
                static_cast<size_t>(
                    static_cast<uint64_t>(x) *
                    ForegroundMoments::density_grid_size /
                    static_cast<uint64_t>(width)));
            auto bin_y = std::min(
                static_cast<size_t>(ForegroundMoments::density_grid_size - 1u),
                static_cast<size_t>(
                    static_cast<uint64_t>(y) *
                    ForegroundMoments::density_grid_size /
                    static_cast<uint64_t>(height)));
            result.density[
                bin_y * ForegroundMoments::density_grid_size + bin_x] +=
                1.0;
        }
    }
    if (result.count < 2u) { return result; }

    auto count = static_cast<double>(result.count);
    result.coverage = count / static_cast<double>(pixel_count);
    result.centroid_x = mean_x;
    result.centroid_y = mean_y;
    result.covariance_xx = moment_xx / count;
    result.covariance_xy = moment_xy / count;
    result.covariance_yy = moment_yy / count;
    result.valid = std::isfinite(result.coverage) &&
                   std::isfinite(result.centroid_x) &&
                   std::isfinite(result.centroid_y) &&
                   std::isfinite(result.covariance_xx) &&
                   std::isfinite(result.covariance_xy) &&
                   std::isfinite(result.covariance_yy);
    for (auto &density : result.density) {
        density /= count;
        result.valid &= std::isfinite(density);
    }
    return result;
}

struct ForegroundMomentThresholds {
    double max_relative_count_error;
    double max_centroid_distance;
    double max_relative_covariance_error;
    double max_density_total_variation;
    double min_density_cosine_similarity;
};

struct ForegroundMomentCompareResult {
    bool passed{false};
    ForegroundMoments rendered;
    ForegroundMoments reference;
    double relative_count_error{0.0};
    double centroid_distance{0.0};
    double relative_covariance_error{0.0};
    double density_total_variation{0.0};
    double density_cosine_similarity{0.0};
    std::string message;
};

inline ForegroundMomentCompareResult compare_foreground_moments(
    const uint8_t *rendered, const uint8_t *reference,
    int width, int height, int channels,
    const std::array<uint8_t, 3u> &background,
    ForegroundMomentThresholds thresholds,
    uint8_t background_tolerance = 0u) {
    ForegroundMomentCompareResult result;
    auto valid_threshold = [](double value) noexcept {
        return std::isfinite(value) && value >= 0.0;
    };
    if (!valid_threshold(thresholds.max_relative_count_error) ||
        !valid_threshold(thresholds.max_centroid_distance) ||
        !valid_threshold(thresholds.max_relative_covariance_error) ||
        !valid_threshold(thresholds.max_density_total_variation) ||
        !valid_threshold(thresholds.min_density_cosine_similarity) ||
        thresholds.max_density_total_variation > 1.0 ||
        thresholds.min_density_cosine_similarity > 1.0) {
        result.message = "invalid foreground-moment thresholds";
        return result;
    }

    result.rendered = compute_foreground_moments(
        rendered, width, height, channels, background,
        background_tolerance);
    result.reference = compute_foreground_moments(
        reference, width, height, channels, background,
        background_tolerance);
    if (!result.rendered.valid) {
        result.message = "rendered image has fewer than two valid foreground pixels";
        return result;
    }
    if (!result.reference.valid) {
        result.message = "reference image has fewer than two valid foreground pixels";
        return result;
    }

    auto rendered_count = static_cast<double>(result.rendered.count);
    auto reference_count = static_cast<double>(result.reference.count);
    result.relative_count_error =
        std::abs(rendered_count - reference_count) / reference_count;
    result.centroid_distance = std::hypot(
        result.rendered.centroid_x - result.reference.centroid_x,
        result.rendered.centroid_y - result.reference.centroid_y);

    auto covariance_norm = [](const ForegroundMoments &moments) noexcept {
        return std::sqrt(
            moments.covariance_xx * moments.covariance_xx +
            2.0 * moments.covariance_xy * moments.covariance_xy +
            moments.covariance_yy * moments.covariance_yy);
    };
    auto reference_covariance_norm = covariance_norm(result.reference);
    static constexpr auto minimum_covariance_norm = 1e-12;
    if (!std::isfinite(reference_covariance_norm) ||
        reference_covariance_norm <= minimum_covariance_norm) {
        result.message = "reference foreground covariance is degenerate";
        return result;
    }
    auto covariance_xx_delta = result.rendered.covariance_xx -
                               result.reference.covariance_xx;
    auto covariance_xy_delta = result.rendered.covariance_xy -
                               result.reference.covariance_xy;
    auto covariance_yy_delta = result.rendered.covariance_yy -
                               result.reference.covariance_yy;
    result.relative_covariance_error =
        std::sqrt(covariance_xx_delta * covariance_xx_delta +
                  2.0 * covariance_xy_delta * covariance_xy_delta +
                  covariance_yy_delta * covariance_yy_delta) /
        reference_covariance_norm;

    double density_l1_distance = 0.0;
    double density_dot = 0.0;
    double rendered_density_norm_squared = 0.0;
    double reference_density_norm_squared = 0.0;
    for (auto i = 0u; i < ForegroundMoments::density_bin_count; ++i) {
        auto rendered_density = result.rendered.density[i];
        auto reference_density = result.reference.density[i];
        density_l1_distance +=
            std::abs(rendered_density - reference_density);
        density_dot += rendered_density * reference_density;
        rendered_density_norm_squared +=
            rendered_density * rendered_density;
        reference_density_norm_squared +=
            reference_density * reference_density;
    }
    // Half of normalized L1 is total-variation distance: zero means the
    // coarse foreground mass is identical and one means it is disjoint.
    result.density_total_variation = 0.5 * density_l1_distance;
    auto density_norm_product = std::sqrt(
        rendered_density_norm_squared * reference_density_norm_squared);
    if (!std::isfinite(density_norm_product) ||
        density_norm_product <= minimum_covariance_norm) {
        result.message = "foreground density histogram is degenerate";
        return result;
    }
    result.density_cosine_similarity = std::clamp(
        density_dot / density_norm_product, 0.0, 1.0);

    auto errors_are_finite =
        std::isfinite(result.relative_count_error) &&
        std::isfinite(result.centroid_distance) &&
        std::isfinite(result.relative_covariance_error) &&
        std::isfinite(result.density_total_variation) &&
        std::isfinite(result.density_cosine_similarity);
    result.passed = errors_are_finite &&
                    result.relative_count_error <=
                        thresholds.max_relative_count_error &&
                    result.centroid_distance <=
                        thresholds.max_centroid_distance &&
                    result.relative_covariance_error <=
                        thresholds.max_relative_covariance_error &&
                    result.density_total_variation <=
                        thresholds.max_density_total_variation &&
                    result.density_cosine_similarity >=
                        thresholds.min_density_cosine_similarity;
    result.message =
        "foreground count=" + std::to_string(result.rendered.count) +
        " vs " + std::to_string(result.reference.count) +
        ", relative count error=" +
        std::to_string(result.relative_count_error) +
        " (threshold=" +
        std::to_string(thresholds.max_relative_count_error) +
        "), normalized centroid distance=" +
        std::to_string(result.centroid_distance) +
        " (threshold=" +
        std::to_string(thresholds.max_centroid_distance) +
        "), relative covariance error=" +
        std::to_string(result.relative_covariance_error) +
        " (threshold=" +
        std::to_string(thresholds.max_relative_covariance_error) +
        "), 16x16 density total variation=" +
        std::to_string(result.density_total_variation) +
        " (threshold=" +
        std::to_string(thresholds.max_density_total_variation) +
        "), density cosine similarity=" +
        std::to_string(result.density_cosine_similarity) +
        " (threshold=" +
        std::to_string(thresholds.min_density_cosine_similarity) + ")";
    return result;
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

inline ForegroundMomentCompareResult
compare_foreground_moments_with_reference_file(
    const uint8_t *rendered, int width, int height, int channels,
    const std::filesystem::path &reference_path,
    const std::array<uint8_t, 3u> &background,
    ForegroundMomentThresholds thresholds,
    uint8_t background_tolerance = 0u) {
    ForegroundMomentCompareResult result;
    if (rendered == nullptr || width <= 0 || height <= 0 ||
        channels < 3) {
        result.message = "invalid rendered image for foreground-moment comparison";
        return result;
    }
    if (!std::filesystem::exists(reference_path)) {
        result.message = "reference not found: " + reference_path.string();
        return result;
    }
    int ref_w = 0, ref_h = 0, ref_c = 0;
    auto *ref_data = stbi_load(
        reference_path.string().c_str(), &ref_w, &ref_h, &ref_c,
        channels);
    if (ref_data == nullptr) {
        result.message = "failed to load reference: " +
                         reference_path.string();
        return result;
    }
    if (ref_w != width || ref_h != height) {
        result.message =
            "resolution mismatch: rendered " + std::to_string(width) +
            "x" + std::to_string(height) + " vs reference " +
            std::to_string(ref_w) + "x" + std::to_string(ref_h) +
            " (" + reference_path.string() + ")";
    } else {
        result = compare_foreground_moments(
            rendered, ref_data, width, height, channels, background,
            thresholds, background_tolerance);
        result.message += " ref=" + reference_path.string();
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
