// Test for sparse-foreground image comparison.
// This test covers:
// - exact and one-pixel-jittered distributions
// - count, centroid, and covariance regressions
// - same-moment distributions with different coarse topology
// - blank, malformed, and invalid-threshold rejection

#include "ut/ut.hpp"

#include "reference_compare.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto width = 64;
constexpr auto height = 64;
constexpr auto channels = 4;
constexpr std::array<uint8_t, 3u> background{26u, 51u, 77u};
constexpr std::array<uint8_t, 4u> foreground{102u, 153u, 153u, 255u};

using Image = std::vector<uint8_t>;

[[nodiscard]] Image make_background_image() {
    Image image(static_cast<size_t>(width * height * channels));
    for (auto y = 0; y < height; ++y) {
        for (auto x = 0; x < width; ++x) {
            auto offset = static_cast<size_t>(
                (y * width + x) * channels);
            image[offset + 0u] = background[0u];
            image[offset + 1u] = background[1u];
            image[offset + 2u] = background[2u];
            image[offset + 3u] = 255u;
        }
    }
    return image;
}

void set_foreground(Image &image, int x, int y) {
    auto offset = static_cast<size_t>((y * width + x) * channels);
    for (auto channel = 0u; channel < foreground.size(); ++channel) {
        image[offset + channel] = foreground[channel];
    }
}

[[nodiscard]] Image make_checkerboard(
    int x_begin, int x_end, int y_begin, int y_end,
    int phase = 0) {
    auto image = make_background_image();
    for (auto y = y_begin; y < y_end; ++y) {
        for (auto x = x_begin; x < x_end; ++x) {
            if (((x + y + phase) & 1) == 0) {
                set_foreground(image, x, y);
            }
        }
    }
    return image;
}

[[nodiscard]] Image make_filled_rectangle(
    int x_begin, int x_end, int y_begin, int y_end) {
    auto image = make_background_image();
    for (auto y = y_begin; y < y_end; ++y) {
        for (auto x = x_begin; x < x_end; ++x) {
            set_foreground(image, x, y);
        }
    }
    return image;
}

void set_two_by_two_block(Image &image, int x, int y) {
    for (auto dy = 0; dy < 2; ++dy) {
        for (auto dx = 0; dx < 2; ++dx) {
            set_foreground(image, x + dx, y + dy);
        }
    }
}

[[nodiscard]] Image make_cross_topology() {
    auto image = make_background_image();
    set_two_by_two_block(image, 14, 31);
    set_two_by_two_block(image, 48, 31);
    set_two_by_two_block(image, 31, 14);
    set_two_by_two_block(image, 31, 48);
    return image;
}

[[nodiscard]] Image make_diagonal_topology() {
    auto image = make_background_image();
    set_two_by_two_block(image, 19, 19);
    set_two_by_two_block(image, 43, 19);
    set_two_by_two_block(image, 19, 43);
    set_two_by_two_block(image, 43, 43);
    return image;
}

constexpr ref::ForegroundMomentThresholds strict_thresholds{
    .max_relative_count_error = 0.0,
    .max_centroid_distance = 0.0,
    .max_relative_covariance_error = 0.0,
    .max_density_total_variation = 0.0,
    .min_density_cosine_similarity = 1.0 - 1e-12,
};

constexpr ref::ForegroundMomentThresholds distribution_thresholds{
    .max_relative_count_error = 0.01,
    .max_centroid_distance = 0.02,
    .max_relative_covariance_error = 0.05,
    .max_density_total_variation = 0.06,
    .min_density_cosine_similarity = 0.99,
};

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "foreground_moments_accept_identical_images"_test = [] {
        auto reference = make_checkerboard(16, 48, 16, 48);
        auto result = ref::compare_foreground_moments(
            reference.data(), reference.data(), width, height, channels,
            background, strict_thresholds);
        expect(result.passed);
        expect(result.rendered.valid);
        expect(eq(result.rendered.count, size_t{512u}));
        expect(result.relative_count_error == 0.0);
        expect(result.centroid_distance == 0.0);
        expect(result.relative_covariance_error == 0.0);
    };

    "foreground_moments_accept_one_pixel_jitter_with_stable_distribution"_test = [] {
        auto reference = make_filled_rectangle(16, 48, 16, 48);
        auto jittered = make_filled_rectangle(17, 49, 16, 48);
        auto result = ref::compare_foreground_moments(
            jittered.data(), reference.data(), width, height, channels,
            background, distribution_thresholds);
        expect(result.passed)
            << "a one-pixel scheduler perturbation should pass the coarse distribution oracle";
        expect(result.density_total_variation > 0.0);
        expect(result.density_total_variation <=
               distribution_thresholds.max_density_total_variation);
        expect(result.density_cosine_similarity >=
               distribution_thresholds.min_density_cosine_similarity);
        expect(ref::compute_psnr(
                   jittered.data(), reference.data(), width, height,
                   channels) <
               ref::DEFAULT_PSNR_THRESHOLD)
            << "the fixture should prove why full-resolution PSNR is unsuitable";
    };

    "foreground_density_rejects_same_moment_different_topology"_test = [] {
        auto reference = make_cross_topology();
        auto scrambled = make_diagonal_topology();
        auto result = ref::compare_foreground_moments(
            scrambled.data(), reference.data(), width, height, channels,
            background, distribution_thresholds);
        expect(!result.passed);
        expect(result.relative_count_error == 0.0);
        expect(result.centroid_distance < 1e-12);
        expect(result.relative_covariance_error <
               distribution_thresholds.max_relative_covariance_error)
            << "the adversarial fixture must pass all moment checks";
        expect(result.density_total_variation >
               distribution_thresholds.max_density_total_variation);
        expect(result.density_cosine_similarity <
               distribution_thresholds.min_density_cosine_similarity);
    };

    "foreground_moments_reject_count_regression"_test = [] {
        auto reference = make_checkerboard(16, 48, 16, 48);
        auto sparse = make_checkerboard(16, 32, 16, 48);
        auto result = ref::compare_foreground_moments(
            sparse.data(), reference.data(), width, height, channels,
            background, distribution_thresholds);
        expect(!result.passed);
        expect(result.relative_count_error >
               distribution_thresholds.max_relative_count_error);
    };

    "foreground_moments_reject_centroid_regression"_test = [] {
        auto reference = make_checkerboard(16, 48, 16, 48);
        auto translated = make_checkerboard(16, 48, 24, 56);
        auto result = ref::compare_foreground_moments(
            translated.data(), reference.data(), width, height, channels,
            background, distribution_thresholds);
        expect(!result.passed);
        expect(result.centroid_distance >
               distribution_thresholds.max_centroid_distance);
        expect(result.relative_count_error == 0.0);
    };

    "foreground_moments_reject_covariance_regression"_test = [] {
        auto reference = make_checkerboard(16, 48, 16, 48);
        auto stretched = make_checkerboard(24, 40, 0, 64);
        auto result = ref::compare_foreground_moments(
            stretched.data(), reference.data(), width, height, channels,
            background, distribution_thresholds);
        expect(!result.passed);
        expect(result.relative_count_error == 0.0);
        expect(result.centroid_distance < 1e-12);
        expect(result.relative_covariance_error >
               distribution_thresholds.max_relative_covariance_error);
    };

    "foreground_moments_reject_blank_and_malformed_images"_test = [] {
        auto reference = make_checkerboard(16, 48, 16, 48);
        auto blank = make_background_image();
        auto blank_result = ref::compare_foreground_moments(
            blank.data(), reference.data(), width, height, channels,
            background, distribution_thresholds);
        expect(!blank_result.passed);
        expect(!blank_result.rendered.valid);

        auto malformed = ref::compute_foreground_moments(
            nullptr, width, height, channels, background);
        expect(!malformed.valid);
        expect(eq(malformed.count, size_t{0u}));
    };

    "foreground_moments_reject_invalid_thresholds"_test = [] {
        auto reference = make_checkerboard(16, 48, 16, 48);
        auto invalid = distribution_thresholds;
        invalid.max_centroid_distance =
            std::numeric_limits<double>::quiet_NaN();
        auto result = ref::compare_foreground_moments(
            reference.data(), reference.data(), width, height, channels,
            background, invalid);
        expect(!result.passed);
        expect(result.message == "invalid foreground-moment thresholds");
    };
}
