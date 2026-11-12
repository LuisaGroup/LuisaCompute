// Test for ocr/ops/ops_image (Phase 2).
// Covers: fused color-convert + bilinear/area resize + normalization vs the
// Phase 1 preprocess_image convention (differential), identity resize,
// upscale/downscale ragged sizes, crop with out-of-range zero fill,
// letterbox pad, patchify packing. Oracle: Phase 1 preprocess_image for the
// fused path + independent host references for crop/letterbox/patchify.
// Tolerance: 2e-3 + 2e-3*|ref| (fp32 device resampling vs fp32/double host).
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/image.h"
#include "ocr/ops/ops_image.h"
#include "ocr/preprocess.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 2e-3f;
constexpr float kRtol = 2e-3f;

[[nodiscard]] DecodedImage make_image(uint32_t w, uint32_t h, PixelOrder order, ocr::test::Lcg &lcg) {
    DecodedImage img;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.row_stride = static_cast<uint64_t>(w) * 3u;
    img.order = order;
    img.pixels.resize(static_cast<size_t>(w) * h * 3u);
    for (auto &p : img.pixels) { p = static_cast<uint8_t>(lcg.next() % 256u); }
    return img;
}

void test_fused_resize_normalize(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    auto dummy = device.create_buffer<float>(1);
    auto shader = device.compile(make_image_resize_normalize_kernel());
    expect(shader.compile_ok());
    struct C { uint32_t sw, sh, dw, dh; bool bgr; ResizeConvention conv; bool norm; };
    const C cases[] = {
        {8, 6, 8, 6, false, ResizeConvention::bilinear, true},  // identity
        {8, 6, 5, 4, false, ResizeConvention::bilinear, true},  // downscale ragged
        {5, 4, 11, 9, true, ResizeConvention::bilinear, true},  // upscale + BGR
        {9, 7, 4, 3, false, ResizeConvention::area, true},      // area down
        {4, 3, 9, 7, true, ResizeConvention::area, false},      // area up, no norm
        {1, 1, 5, 3, false, ResizeConvention::bilinear, true},  // degenerate source
        {7, 5, 1, 1, false, ResizeConvention::area, true},      // degenerate target
    };
    for (auto c : cases) {
        const auto src_order = c.bgr ? PixelOrder::bgr : PixelOrder::rgb;
        auto img = make_image(c.sw, c.sh, src_order, lcg);
        PreprocessSpec spec;
        spec.resize = c.conv;
        spec.target_width = c.dw;
        spec.target_height = c.dh;
        spec.order = PixelOrder::rgb;// output always RGB
        if (c.norm) {
            spec.mean = {0.485f, 0.456f, 0.406f};
            spec.std_dev = {0.229f, 0.224f, 0.225f};
        }
        auto ref_img = preprocess_image(img, spec);
        expect(ref_img.ok());
        if (!ref_img.ok()) { continue; }
        auto words = pack_image_words(img.pixels);
        auto bsrc = ocr::test::upload(device, stream, luisa::span<const uint32_t>{words});
        auto bmean_storage = c.norm ? ocr::test::upload(device, stream, luisa::span<const float>{spec.mean}) : Buffer<float>{};
        auto bstd_storage = c.norm ? ocr::test::upload(device, stream, luisa::span<const float>{spec.std_dev}) : Buffer<float>{};
        auto bmean = c.norm ? bmean_storage.view() : dummy.view();
        auto bstd = c.norm ? bstd_storage.view() : dummy.view();
        auto bout = device.create_buffer<float>(static_cast<size_t>(c.dw) * c.dh * 3u);
        auto plan = plan_image_resize_dispatch(c.dw, c.dh);
        expect(plan.ok());
        OCR_DISPATCH_1D(stream, shader(bsrc, bout, bmean, bstd, c.sw, c.sh, c.dw, c.dh, c.bgr ? 1u : 0u,
                         c.conv == ResizeConvention::bilinear ? 0u : 1u, c.norm ? 1u : 0u), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bout, got, static_cast<size_t>(c.dw) * c.dh * 3u);
        stream << synchronize();
        ocr::test::expect_close(got, ref_img.value->pixels, kAtol, kRtol, "fused resize normalize vs Phase 1");
    }
}

void test_crop_letterbox_patchify(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    // Crop with a region partially outside the source (zero fill).
    constexpr uint32_t sw = 7u, sh = 5u, x0 = 5u, y0 = 3u, cw = 4u, ch = 4u;
    auto img = make_image(sw, sh, PixelOrder::rgb, lcg);
    {
        auto shader = device.compile(make_image_crop_kernel());
        auto bsrc = ocr::test::upload(device, stream, luisa::span<const uint8_t>{img.pixels});
        auto bdst = device.create_buffer<uint8_t>(static_cast<size_t>(cw) * ch * 3u);
        auto plan = plan_image_crop_dispatch(cw, ch);
        OCR_DISPATCH_1D(stream, shader(bsrc, bdst, sw, sh, x0, y0, cw, ch), *plan.value);
        luisa::vector<uint8_t> got;
        ocr::test::download(stream, bdst, got, static_cast<size_t>(cw) * ch * 3u);
        stream << synchronize();
        bool ok = true;
        for (uint32_t y = 0; y < ch && ok; y++) {
            for (uint32_t x = 0; x < cw && ok; x++) {
                for (uint32_t c = 0; c < 3 && ok; c++) {
                    const uint32_t sx = x0 + x, sy = y0 + y;
                    const uint8_t want = (sx < sw && sy < sh) ? img.pixels[(static_cast<size_t>(sy) * sw + sx) * 3u + c] : 0u;
                    ok = got[(static_cast<size_t>(y) * cw + x) * 3u + c] == want;
                }
            }
        }
        expect(ok) << "crop with out-of-range zero fill";
    }
    // Letterbox: place a 3x2 CHW image into 7x5 at (2,1), pad -1.
    {
        constexpr uint32_t iw = 3u, ih = 2u, dw = 7u, dh = 5u, px = 2u, py = 1u;
        luisa::vector<float> src(static_cast<size_t>(3) * iw * ih);
        lcg.fill(src, -1.f, 1.f);
        auto shader = device.compile(make_image_letterbox_kernel());
        auto bsrc = ocr::test::upload(device, stream, luisa::span<const float>{src});
        auto bdst = device.create_buffer<float>(static_cast<size_t>(3) * dw * dh);
        auto plan = plan_image_letterbox_dispatch(dw, dh);
        OCR_DISPATCH_1D(stream, shader(bsrc, bdst, iw, ih, dw, dh, px, py, -1.f), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bdst, got, static_cast<size_t>(3) * dw * dh);
        stream << synchronize();
        bool ok = true;
        for (uint32_t c = 0; c < 3 && ok; c++) {
            for (uint32_t y = 0; y < dh && ok; y++) {
                for (uint32_t x = 0; x < dw && ok; x++) {
                    float want = -1.f;
                    if (x >= px && y >= py && x - px < iw && y - py < ih) {
                        want = src[static_cast<size_t>(c) * iw * ih + (y - py) * iw + (x - px)];
                    }
                    ok = got[(static_cast<size_t>(c) * dh + y) * dw + x] == want;
                }
            }
        }
        expect(ok) << "letterbox pad exact";
    }
    // Patchify: CHW [3,4,6] patch 2 -> 6 patches of 12 elems; verify vs
    // an independent host loop.
    {
        constexpr uint32_t C = 3u, H = 4u, W = 6u, patch = 2u;
        luisa::vector<float> src(static_cast<size_t>(C) * H * W);
        lcg.fill(src, -1.f, 1.f);
        const uint32_t pw = W / patch, ph = H / patch, pe = C * patch * patch;
        luisa::vector<float> ref(static_cast<size_t>(pw) * ph * pe);
        for (uint32_t p = 0; p < pw * ph; p++) {
            const uint32_t px = p % pw, py = p / pw;
            for (uint32_t c = 0; c < C; c++) {
                for (uint32_t iy = 0; iy < patch; iy++) {
                    for (uint32_t ix = 0; ix < patch; ix++) {
                        ref[static_cast<size_t>(p) * pe + (c * patch + iy) * patch + ix] =
                            src[(static_cast<size_t>(c) * H + py * patch + iy) * W + px * patch + ix];
                    }
                }
            }
        }
        auto shader = device.compile(make_image_patchify_kernel());
        auto bsrc = ocr::test::upload(device, stream, luisa::span<const float>{src});
        auto bout = device.create_buffer<float>(ref.size());
        auto plan = plan_image_patchify_dispatch(static_cast<size_t>(pw) * ph, pe);
        OCR_DISPATCH_1D(stream, shader(bsrc, bout, C, H, W, patch), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bout, got, ref.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, 0.f, 0.f, "patchify exact");
    }
}

// M1: flattened image dispatch regression. A 4096x1366 canvas * 3 channels
// = 16,785,408 elements > 65535*256 forces a flattened plan; the letterbox
// border fill must cover EVERY element (poisoned canvas proves it).
void test_flattened_letterbox_coverage(Device &device) {
    auto stream = device.create_stream();
    constexpr uint32_t dw = 4096u, dh = 1366u;
    const uint64_t count = static_cast<uint64_t>(dw) * dh * 3u;
    auto plan = plan_image_letterbox_dispatch(dw, dh);
    expect(plan.ok());
    expect(plan.value->is_2d()) << "canvas must force flattening";
    // Empty source region (src 1x1 placed at 0,0); everything else = pad 7.
    constexpr uint32_t iw = 1u, ih = 1u;
    const float src_pixel = 42.f;
    luisa::vector<float> src = {src_pixel, src_pixel, src_pixel};
    auto bsrc = ocr::test::upload(device, stream, luisa::span<const float>{src});
    auto bdst = device.create_buffer<float>(static_cast<size_t>(count));
    luisa::vector<float> poison(static_cast<size_t>(count), -1.f);
    stream << bdst.copy_from(luisa::span<const float>{poison});
    auto shader = device.compile(make_image_letterbox_kernel());
    OCR_DISPATCH_1D(stream, shader(bsrc, bdst, iw, ih, dw, dh, 0u, 0u, 7.f), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bdst, got, static_cast<size_t>(count));
    stream << synchronize();
    uint64_t bad = 0;
    const uint64_t plane = static_cast<uint64_t>(dw) * dh;
    for (uint64_t i = 0; i < count; i++) {
        const uint64_t pix = i % plane;
        const float want = pix == 0 ? src_pixel : 7.f;
        if (got[static_cast<size_t>(i)] != want) { bad++; }
    }
    expect(bad == 0u) << "flattened letterbox must cover every element";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "image_fused_resize_normalize"_test = [&] { test_fused_resize_normalize(device); };
    "image_crop_letterbox_patchify"_test = [&] { test_crop_letterbox_patchify(device); };
    "flattened_letterbox_full_coverage"_test = [&] { test_flattened_letterbox_coverage(device); };
}