#pragma once
// OCR engine -- checked image decode + EXIF orientation (Phase 1). Host-only.
//
// JPEG/PNG decode through the repository's stb_image dependency, with byte /
// pixel / dimension limits, degenerate-image rejection, RGB/BGR conversion and
// explicit row stride. stb does NOT apply EXIF orientation, so a checked JPEG
// EXIF parser extracts the orientation tag and the pixels are re-oriented
// here. All memory transfers use luisa::span.
#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/quantization.h"
#include "ocr/status.h"

namespace luisa::ocr {

enum class PixelOrder : uint32_t {
    rgb = 0,
    bgr = 1,
};

[[nodiscard]] luisa::string_view pixel_order_name(PixelOrder order) noexcept;

// EXIF orientation values 1..8 (TIFF tag 0x0112). 1 = normal.
enum class ExifOrientation : uint32_t {
    normal = 1,
    flip_horizontal = 2,
    rotate_180 = 3,
    flip_vertical = 4,
    transpose = 5,
    rotate_90_cw = 6,
    transverse = 7,
    rotate_270_cw = 8,
};

struct ImageLimits {
    uint64_t max_file_bytes{64ull * 1024ull * 1024ull};
    uint64_t max_pixels{64ull * 1024ull * 1024ull};// width * height
    uint32_t max_width{16384};
    uint32_t max_height{16384};
};

struct ImageDecodeOptions {
    PixelOrder order{PixelOrder::rgb};
    bool apply_exif_orientation{true};
    ImageLimits limits;
};

// A decoded, oriented image. Pixels are tightly packed rows of 3 channels in
// the requested order; row_stride is in BYTES (== width*3 here, kept explicit
// for the preprocessing contract).
struct DecodedImage {
    uint32_t width{0};
    uint32_t height{0};
    uint32_t channels{3};
    uint64_t row_stride{0};
    PixelOrder order{PixelOrder::rgb};
    ExifOrientation applied_orientation{ExifOrientation::normal};
    luisa::vector<uint8_t> pixels;

    [[nodiscard]] luisa::span<const uint8_t> row(uint32_t y) const noexcept {
        return {pixels.data() + static_cast<size_t>(y) * row_stride, static_cast<size_t>(width) * channels};
    }
};

// Decode from memory / file. Rejects truncated/oversized/degenerate images and
// unsupported formats with a Status (never LUISA_ERROR on ordinary input).
[[nodiscard]] Result<DecodedImage> decode_image(luisa::span<const std::byte> bytes,
                                                const ImageDecodeOptions &options = {});
[[nodiscard]] Result<DecodedImage> decode_image_file(const luisa::filesystem::path &path,
                                                     const ImageDecodeOptions &options = {});

// Parse the EXIF orientation from a JPEG byte stream (checked; returns normal
// when no EXIF/orientation is present). Exposed for tests.
[[nodiscard]] Result<ExifOrientation> parse_jpeg_exif_orientation(luisa::span<const std::byte> bytes);

// Apply an EXIF orientation to tightly-packed 3-channel pixels (checked).
[[nodiscard]] Status orient_image(luisa::span<const uint8_t> src, uint32_t width, uint32_t height,
                                  ExifOrientation orientation, luisa::vector<uint8_t> &out,
                                  uint32_t &out_width, uint32_t &out_height);

// Convert RGB<->BGR in place (checked size).
[[nodiscard]] Status convert_rgb_bgr(luisa::span<uint8_t> pixels, uint64_t pixel_count);

}// namespace luisa::ocr