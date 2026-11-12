// Test for examples/ocr/ocr/preprocess.h/.cpp and feature_bundle.h/.cpp.
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - checked resize (bilinear/area), normalize, patch metadata;
// - source-coordinate transform provenance;
// - packed ragged batch offsets + budgets;
// - degenerate/overflow/invalid-normalization rejection;
// - .ocrfeat feature bundle writer->reader round trip + corruption.
#include "ut/ut.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>

#include <luisa/core/stl/filesystem.h>

#include "ocr/feature_bundle.h"
#include "ocr/preprocess.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct TempDir {
    luisa::filesystem::path path;
    TempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcocr_pp_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

[[nodiscard]] DecodedImage make_image(uint32_t w, uint32_t h, uint8_t fill) {
    DecodedImage image;
    image.width = w;
    image.height = h;
    image.channels = 3;
    image.row_stride = static_cast<uint64_t>(w) * 3u;
    image.pixels.assign(static_cast<size_t>(w) * h * 3u, fill);
    return image;
}

void test_identity_preprocess() {
    auto image = make_image(4, 4, 128);
    PreprocessSpec spec;
    auto r = preprocess_image(image, spec);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->width == 4u && r.value->height == 4u);
    expect(r.value->pixels.size() == 4u * 4u * 3u);
    // No normalization: pixel/255.
    expect(std::fabs(r.value->pixels[0] - 128.0f / 255.0f) < 1e-6f);
    // Identity transform.
    expect(r.value->transform.scale_x == 1.0f);
    expect(r.value->transform.scale_y == 1.0f);
}

void test_resize_and_provenance() {
    auto image = make_image(8, 8, 200);
    PreprocessSpec spec;
    spec.target_width = 4;
    spec.target_height = 4;
    spec.resize = ResizeConvention::bilinear;
    auto r = preprocess_image(image, spec);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->width == 4u && r.value->height == 4u);
    // Provenance maps preprocessed -> original: scale 8/4 = 2.
    expect(r.value->transform.scale_x == 2.0f);
    expect(r.value->transform.scale_y == 2.0f);
    auto src = r.value->transform.to_source(1.0f, 1.0f);
    expect(src.x == 2.0f && src.y == 2.0f);
    // Uniform source stays uniform after resize.
    expect(std::fabs(r.value->pixels[0] - 200.0f / 255.0f) < 1e-4f);
}

void test_normalize_and_patches() {
    auto image = make_image(8, 8, 255);
    PreprocessSpec spec;
    spec.mean = {0.5f, 0.5f, 0.5f};
    spec.std_dev = {0.5f, 0.5f, 0.5f};
    spec.patch_size = 4;
    auto r = preprocess_image(image, spec);
    expect(r.ok());
    if (!r.ok()) { return; }
    // (1 - 0.5)/0.5 = 1.
    expect(std::fabs(r.value->pixels[0] - 1.0f) < 1e-6f);
    expect(r.value->patches_w == 2u && r.value->patches_h == 2u);
    expect(r.value->patch_count == 4u);
    // CHW layout: channel stride = width*height.
    expect(r.value->pixels.size() == 3u * 8u * 8u);
}

void test_invalid_preprocess() {
    auto image = make_image(4, 4, 0);
    // Zero std rejected.
    PreprocessSpec spec;
    spec.std_dev = {0.0f, 1.0f, 1.0f};
    expect(!preprocess_image(image, spec).ok());
    // Wrong channel count.
    PreprocessSpec spec2;
    spec2.mean = {0.5f, 0.5f};
    expect(!preprocess_image(image, spec2).ok());
    // Patch size not dividing dimensions.
    PreprocessSpec spec3;
    spec3.patch_size = 3;
    expect(!preprocess_image(image, spec3).ok());
    // Pixel limit.
    PreprocessSpec spec4;
    spec4.target_width = 1000;
    spec4.target_height = 1000;
    spec4.max_pixels = 100;
    expect(!preprocess_image(image, spec4).ok());
    // Degenerate source.
    DecodedImage empty;
    expect(!preprocess_image(empty, PreprocessSpec{}).ok());
}

void test_ragged_batch() {
    luisa::vector<DecodedImage> images;
    images.emplace_back(make_image(4, 4, 10));
    images.emplace_back(make_image(8, 8, 20));
    PreprocessSpec spec;
    spec.patch_size = 4;
    auto r = pack_ragged_batch(images, spec, 100);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->images.size() == 2u);
    expect(r.value->patch_offsets.size() == 3u);
    expect(r.value->patch_offsets[0] == 0u);
    expect(r.value->patch_offsets[1] == 1u);// 4x4 / 4x4 = 1 patch
    expect(r.value->patch_offsets[2] == 5u);// + 8x8/4x4 = 4 patches
    expect(r.value->total_patches == 5u);
    // Budget exceeded.
    expect(!pack_ragged_batch(images, spec, 2).ok());
}

void test_feature_bundle_roundtrip() {
    TempDir tmp;
    const auto path = tmp.path / "test.ocrfeat";
    {
        auto writer = FeatureBundleWriter::create(path);
        expect(writer.ok());
        auto &w = *writer.value;
        const float pixels[] = {0.1f, 0.2f, 0.3f, 0.4f};
        FeatureBundleWriter::TensorSpec spec;
        spec.name = "pixel_values";
        spec.dtype = FeatureDtype::f32;
        spec.shape = {1, 4};
        spec.payload.resize(sizeof(pixels));
        std::memcpy(spec.payload.data(), pixels, sizeof(pixels));
        expect(w->add_tensor(spec).ok());
        FeatureBundleMeta meta;
        meta.producer = "test";
        meta.processor = "official-test";
        meta.processor_revision = "rev0";
        meta.source_image_hash = "abc";
        meta.decode_settings = "{\"deterministic\":true}";
        expect(w->finish(meta).ok());
    }
    auto reader = FeatureBundleReader::open(path);
    expect(reader.ok());
    if (!reader.ok()) { return; }
    expect((*reader.value)->meta().processor == "official-test");
    auto *t = (*reader.value)->find("pixel_values");
    expect(t != nullptr);
    float out[4];
    expect((*reader.value)->read_tensor_f32(*t, out).ok());
    expect(out[0] == 0.1f && out[3] == 0.4f);
}

void test_feature_bundle_corruption() {
    TempDir tmp;
    const auto path = tmp.path / "bad.ocrfeat";
    {
        auto writer = FeatureBundleWriter::create(path);
        auto &w = *writer.value;
        const float v[] = {1.0f};
        FeatureBundleWriter::TensorSpec spec;
        spec.name = "t";
        spec.dtype = FeatureDtype::f32;
        spec.shape = {1};
        spec.payload.resize(4);
        std::memcpy(spec.payload.data(), v, 4);
        expect(w->add_tensor(spec).ok());
        FeatureBundleMeta meta;
        meta.producer = "t";
        meta.processor = "p";
        expect(w->finish(meta).ok());
    }
    // Truncate.
    {
        auto trunc = path;
        std::filesystem::resize_file(trunc, 10);
        expect(!FeatureBundleReader::open(trunc).ok());
    }
}

// spec.order must be honored: an RGB source requested as BGR swaps channels
// BEFORE normalization, so output channel c uses mean[c]/std_dev[c] on the
// reordered channel. Regression for spec.order being ignored.
void test_channel_order() {
    DecodedImage image;
    image.width = 2;
    image.height = 1;
    image.channels = 3;
    image.row_stride = 6;
    image.order = PixelOrder::rgb;
    image.pixels = {10, 20, 30, 40, 50, 60};// pixel0=(R10,G20,B30), pixel1=(40,50,60)

    // Request BGR: output ch0=B, ch1=G, ch2=R.
    PreprocessSpec bgr;
    bgr.order = PixelOrder::bgr;
    auto r = preprocess_image(image, bgr);
    expect(r.ok());
    if (r.ok()) {
        const uint64_t np = 2;// pixels per channel plane
        expect(std::fabs(r.value->pixels[0] - 30.0f / 255.0f) < 1e-6f) << "BGR ch0 = B";
        expect(std::fabs(r.value->pixels[np] - 20.0f / 255.0f) < 1e-6f) << "BGR ch1 = G";
        expect(std::fabs(r.value->pixels[2 * np] - 10.0f / 255.0f) < 1e-6f) << "BGR ch2 = R";
    }
    // Request RGB (same as source): no swap.
    PreprocessSpec rgb;
    rgb.order = PixelOrder::rgb;
    auto r2 = preprocess_image(image, rgb);
    expect(r2.ok());
    if (r2.ok()) {
        expect(std::fabs(r2.value->pixels[0] - 10.0f / 255.0f) < 1e-6f) << "RGB ch0 = R";
    }
    // BGR + per-channel mean/std: mean/std index follows the REQUESTED order.
    PreprocessSpec norm;
    norm.order = PixelOrder::bgr;
    norm.mean = {0.0f, 0.0f, 0.0f};
    norm.std_dev = {1.0f, 2.0f, 3.0f};// ch0(B)/1, ch1(G)/2, ch2(R)/3
    auto r3 = preprocess_image(image, norm);
    expect(r3.ok());
    if (r3.ok()) {
        const uint64_t np = 2;
        expect(std::fabs(r3.value->pixels[0] - (30.0f / 255.0f) / 1.0f) < 1e-6f) << "BGR ch0 std=1";
        expect(std::fabs(r3.value->pixels[np] - (20.0f / 255.0f) / 2.0f) < 1e-6f) << "BGR ch1 std=2";
        expect(std::fabs(r3.value->pixels[2 * np] - (10.0f / 255.0f) / 3.0f) < 1e-6f) << "BGR ch2 std=3";
    }
    // BGR source requested as RGB also swaps.
    DecodedImage bgr_src = image;
    bgr_src.order = PixelOrder::bgr;// pixels now interpreted as B,G,R
    auto r4 = preprocess_image(bgr_src, rgb);
    expect(r4.ok());
    if (r4.ok()) {
        // bgr_src pixel0 = B10,G20,R30 -> RGB output ch0=R=30.
        expect(std::fabs(r4.value->pixels[0] - 30.0f / 255.0f) < 1e-6f) << "BGR->RGB ch0 = R";
    }
}

}// namespace

static auto test_ocr_preprocess_registration = [] {
    "identity_preprocess"_test = [] { test_identity_preprocess(); };
    "resize_and_provenance"_test = [] { test_resize_and_provenance(); };
    "normalize_and_patches"_test = [] { test_normalize_and_patches(); };
    "invalid_preprocess"_test = [] { test_invalid_preprocess(); };
    "ragged_batch"_test = [] { test_ragged_batch(); };
    "channel_order"_test = [] { test_channel_order(); };
    "feature_bundle_roundtrip"_test = [] { test_feature_bundle_roundtrip(); };
    "feature_bundle_corruption"_test = [] { test_feature_bundle_corruption(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}