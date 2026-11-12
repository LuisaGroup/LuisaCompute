// Test for examples/ocr/ocr/image.h/.cpp -- checked decode + EXIF orientation.
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - PNG/JPEG decode from memory and file with limits;
// - degenerate/oversized/truncated rejection;
// - RGB/BGR conversion, row stride;
// - EXIF orientation parse + apply (stb alone does not orient);
// - frozen tiny fixtures (never regenerated here).
#include "ut/ut.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>

#include <luisa/core/stl/filesystem.h>

#include "ocr/image.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] luisa::filesystem::path data_dir() {
    return luisa::filesystem::path{LUISA_OCR_TEST_DATA_DIR};
}

[[nodiscard]] luisa::vector<std::byte> read_file_bytes(const luisa::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary);
    luisa::vector<char> chars((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    luisa::vector<std::byte> bytes(chars.size());
    if (!chars.empty()) { std::memcpy(bytes.data(), chars.data(), chars.size()); }
    return bytes;
}

void test_png_decode() {
    auto r = decode_image_file(data_dir() / "mixed_8x8.png");
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->width == 8u);
    expect(r.value->height == 8u);
    expect(r.value->channels == 3u);
    expect(r.value->row_stride == 24u);
    expect(r.value->pixels.size() == 8u * 8u * 3u);
    expect(r.value->order == PixelOrder::rgb);
    // Row accessor returns the right span.
    auto row0 = r.value->row(0);
    expect(row0.size() == 24u);
}

void test_blank_and_wide() {
    auto blank = decode_image_file(data_dir() / "blank_4x4.png");
    expect(blank.ok());
    if (blank.ok()) {
        expect(blank.value->width == 4u && blank.value->height == 4u);
        // All white.
        for (auto b : blank.value->pixels) { expect(b == 255u); }
    }
    auto wide = decode_image_file(data_dir() / "wide_16x2.png");
    expect(wide.ok());
    if (wide.ok()) {
        expect(wide.value->width == 16u && wide.value->height == 2u);
    }
}

void test_bgr_conversion() {
    ImageDecodeOptions options;
    options.order = PixelOrder::bgr;
    auto rgb = decode_image_file(data_dir() / "mixed_8x8.png");
    auto bgr = decode_image_file(data_dir() / "mixed_8x8.png", options);
    expect(rgb.ok() && bgr.ok());
    if (rgb.ok() && bgr.ok()) {
        // First pixel R<->B swapped.
        expect(bgr.value->pixels[0] == rgb.value->pixels[2]);
        expect(bgr.value->pixels[2] == rgb.value->pixels[0]);
        expect(bgr.value->pixels[1] == rgb.value->pixels[1]);
    }
}

void test_exif_orientation() {
    auto bytes = read_file_bytes(data_dir() / "exif_rotate90.jpg");
    auto orientation = parse_jpeg_exif_orientation(bytes);
    expect(orientation.ok());
    expect(*orientation.value == ExifOrientation::rotate_90_cw);
    auto normal = parse_jpeg_exif_orientation(read_file_bytes(data_dir() / "exif_normal.jpg"));
    expect(normal.ok());
    expect(*normal.value == ExifOrientation::normal);
    // Decode applies orientation (1x1 stays 1x1 but the tag is recorded).
    auto decoded = decode_image_file(data_dir() / "exif_rotate90.jpg");
    expect(decoded.ok());
    if (decoded.ok()) {
        expect(decoded.value->applied_orientation == ExifOrientation::rotate_90_cw);
    }
    // Disabling orientation keeps the tag normal.
    ImageDecodeOptions options;
    options.apply_exif_orientation = false;
    auto raw = decode_image_file(data_dir() / "exif_rotate90.jpg", options);
    expect(raw.ok());
    if (raw.ok()) {
        expect(raw.value->applied_orientation == ExifOrientation::normal);
    }
}

void test_orient_transform() {
    // 2x1 image [A, B] rotated 90 CW becomes 1x2 [A; B] (top->right).
    const uint8_t src[] = {1, 2, 3, 4, 5, 6};// pixel A=(1,2,3), B=(4,5,6)
    luisa::vector<uint8_t> out;
    uint32_t ow = 0, oh = 0;
    expect(orient_image(src, 2, 1, ExifOrientation::rotate_90_cw, out, ow, oh).ok());
    expect(ow == 1u && oh == 2u);
    // rotate90: dst(x=0,y=0) = src(x=0, y=height-1-0=0) = A; dst(0,1)=src(1,0)=B
    expect(out[0] == 1 && out[3] == 4);
    // 180 flips both.
    expect(orient_image(src, 2, 1, ExifOrientation::rotate_180, out, ow, oh).ok());
    expect(ow == 2u && oh == 1u);
    expect(out[0] == 4 && out[3] == 1);
}

// Independent expected-index oracle for every EXIF orientation on a NON-SQUARE
// image. Regression for orientation 7 (transverse) reading out of bounds.
void test_orient_all_nonsquare() {
    const uint32_t W = 3, H = 2;// wide, non-square
    luisa::vector<uint8_t> src(static_cast<size_t>(W) * H * 3u);
    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            const auto id = static_cast<uint8_t>(10 * (y + 1) + x);
            const auto base = (static_cast<size_t>(y) * W + x) * 3u;
            src[base] = id;
            src[base + 1] = static_cast<uint8_t>(id + 1);
            src[base + 2] = static_cast<uint8_t>(id + 2);
        }
    }
    for (uint32_t o = 1; o <= 8; o++) {
        luisa::vector<uint8_t> out;
        uint32_t ow = 0, oh = 0;
        auto s = orient_image(src, W, H, static_cast<ExifOrientation>(o), out, ow, oh);
        expect(s.ok()) << "orient " << o << " ok";
        if (!s.ok()) { continue; }
        const bool swaps = o >= 5;
        expect(ow == (swaps ? H : W)) << "orient " << o << " width";
        expect(oh == (swaps ? W : H)) << "orient " << o << " height";
        bool all_ok = true;
        for (uint32_t y = 0; y < oh && all_ok; y++) {
            for (uint32_t x = 0; x < ow && all_ok; x++) {
                uint32_t sx = x, sy = y;
                switch (o) {
                    case 1: sx = x; sy = y; break;
                    case 2: sx = W - 1 - x; sy = y; break;
                    case 3: sx = W - 1 - x; sy = H - 1 - y; break;
                    case 4: sx = x; sy = H - 1 - y; break;
                    case 5: sx = y; sy = x; break;
                    case 6: sx = y; sy = H - 1 - x; break;
                    case 7: sx = W - 1 - y; sy = H - 1 - x; break;
                    case 8: sx = W - 1 - y; sy = x; break;
                }
                const auto sb = (static_cast<size_t>(sy) * W + sx) * 3u;
                const auto db = (static_cast<size_t>(y) * ow + x) * 3u;
                if (out[db] != src[sb] || out[db + 1] != src[sb + 1] || out[db + 2] != src[sb + 2]) {
                    all_ok = false;
                }
            }
        }
        expect(all_ok) << "orient " << o << " pixel mapping (non-square oracle)";
    }
    // Also a tall non-square image (2 wide x 3 tall) for orientation 7.
    const uint32_t W2 = 2, H2 = 3;
    luisa::vector<uint8_t> src2(static_cast<size_t>(W2) * H2 * 3u);
    for (uint32_t y = 0; y < H2; y++) {
        for (uint32_t x = 0; x < W2; x++) {
            src2[(static_cast<size_t>(y) * W2 + x) * 3u] = static_cast<uint8_t>(10 * (y + 1) + x);
        }
    }
    luisa::vector<uint8_t> out2;
    uint32_t ow2 = 0, oh2 = 0;
    expect(orient_image(src2, W2, H2, ExifOrientation::transverse, out2, ow2, oh2).ok());
    expect(ow2 == H2 && oh2 == W2);
    bool ok2 = true;
    for (uint32_t y = 0; y < oh2 && ok2; y++) {
        for (uint32_t x = 0; x < ow2 && ok2; x++) {
            const uint32_t sx = W2 - 1 - y, sy = H2 - 1 - x;
            if (out2[(static_cast<size_t>(y) * ow2 + x) * 3u] != src2[(static_cast<size_t>(sy) * W2 + sx) * 3u]) {
                ok2 = false;
            }
        }
    }
    expect(ok2) << "orient 7 tall non-square oracle";
}

void test_limits_and_degenerate() {
    auto bytes = read_file_bytes(data_dir() / "mixed_8x8.png");
    // Pixel limit.
    ImageDecodeOptions options;
    options.limits.max_pixels = 4;// 8x8=64 > 4
    expect(!decode_image(bytes, options).ok());
    // Dimension limit.
    ImageDecodeOptions options2;
    options2.limits.max_width = 4;
    expect(!decode_image(bytes, options2).ok());
    // File-size limit.
    ImageDecodeOptions options3;
    options3.limits.max_file_bytes = 10;
    expect(!decode_image(bytes, options3).ok());
    // Empty buffer.
    expect(!decode_image({}).ok());
    // Garbage bytes.
    const std::byte garbage[] = {std::byte{1}, std::byte{2}, std::byte{3}};
    expect(!decode_image(garbage).ok());
    // Missing file.
    expect(!decode_image_file(data_dir() / "does_not_exist.png").ok());
}

void test_convert_rgb_bgr() {
    uint8_t pixels[] = {1, 2, 3, 4, 5, 6};
    expect(convert_rgb_bgr(pixels, 2).ok());
    expect(pixels[0] == 3 && pixels[2] == 1);
    expect(pixels[3] == 6 && pixels[5] == 4);
    // Size mismatch rejected.
    expect(!convert_rgb_bgr(pixels, 3).ok());
}

}// namespace

static auto test_ocr_image_registration = [] {
    "png_decode"_test = [] { test_png_decode(); };
    "blank_and_wide"_test = [] { test_blank_and_wide(); };
    "bgr_conversion"_test = [] { test_bgr_conversion(); };
    "exif_orientation"_test = [] { test_exif_orientation(); };
    "orient_transform"_test = [] { test_orient_transform(); };
    "orient_all_nonsquare"_test = [] { test_orient_all_nonsquare(); };
    "limits_and_degenerate"_test = [] { test_limits_and_degenerate(); };
    "convert_rgb_bgr"_test = [] { test_convert_rgb_bgr(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}