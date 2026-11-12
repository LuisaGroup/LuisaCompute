// Test for ocr/ops/ops_vision (Phase 2).
// Covers: pixel-unshuffle (r=2, ragged channels), spatial merge/token packing
// (m=2), absolute/learned position addition, DeepStack injection with
// caller-selected ranges (data, not hardcoded). Patch projection is covered
// by ops_gemm over patchified tokens (test_ocr_ops_image + gemm).
// Oracle: independent host references. Exact-copy ops use 0 tolerance.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_vision.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_pixel_unshuffle(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t n = 2u, c = 3u, h = 6u, w = 8u, r = 2u;
    luisa::vector<float> x(static_cast<size_t>(n) * c * h * w);
    lcg.fill(x, -1.f, 1.f);
    luisa::vector<float> ref;
    expect(ref_pixel_unshuffle(x, n, c, h, w, r, ref).ok());
    auto shader = device.compile(make_pixel_unshuffle_kernel());
    expect(shader.compile_ok());
    auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bo = device.create_buffer<float>(ref.size());
    auto plan = plan_pixel_unshuffle_dispatch(n, c, h, w);
    expect(plan.ok());
    OCR_DISPATCH_1D(stream, shader(bx, bo, n, c, h, w, r), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, ref.size());
    stream << synchronize();
    ocr::test::expect_close(got, ref, 0.f, 0.f, "pixel unshuffle exact");
}

void test_spatial_merge(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    // 4x6 token grid, m=2 -> 2x3 blocks = 6 output tokens of H*4.
    constexpr uint32_t tokens_w = 6u, tokens_h = 4u, H = 5u, m = 2u;
    constexpr uint32_t T = tokens_w * tokens_h;
    luisa::vector<float> x(static_cast<size_t>(T) * H);
    lcg.fill(x, -1.f, 1.f);
    const uint32_t out_tokens = T / (m * m), out_hidden = H * m * m;
    // Independent host reference: block (by,bx) concatenates its m*m tokens.
    luisa::vector<float> ref(static_cast<size_t>(out_tokens) * out_hidden);
    const uint32_t blocks_w = tokens_w / m;
    for (uint32_t ot = 0; ot < out_tokens; ot++) {
        const uint32_t bx = ot % blocks_w, by = ot / blocks_w;
        for (uint32_t sub = 0; sub < m * m; sub++) {
            const uint32_t sy = sub / m, sx = sub % m;
            const uint32_t src_token = (by * m + sy) * tokens_w + (bx * m + sx);
            for (uint32_t d = 0; d < H; d++) {
                ref[static_cast<size_t>(ot) * out_hidden + sub * H + d] = x[static_cast<size_t>(src_token) * H + d];
            }
        }
    }
    auto shader = device.compile(make_spatial_merge_kernel());
    expect(shader.compile_ok());
    auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bo = device.create_buffer<float>(ref.size());
    auto plan = plan_spatial_merge_dispatch(out_tokens, out_hidden);
    OCR_DISPATCH_1D(stream, shader(bx, bo, T, H, tokens_w, m), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, ref.size());
    stream << synchronize();
    ocr::test::expect_close(got, ref, 0.f, 0.f, "spatial merge exact");
}

void test_position_deepstack(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t tokens = 13u, hidden = 8u, pos_rows = 17u;
    luisa::vector<float> x(static_cast<size_t>(tokens) * hidden), pos(static_cast<size_t>(pos_rows) * hidden);
    lcg.fill(x, -1.f, 1.f);
    lcg.fill(pos, -0.5f, 0.5f);
    luisa::vector<int32_t> ids(tokens);
    for (uint32_t t = 0; t < tokens; t++) { ids[t] = static_cast<int32_t>((t * 3u) % pos_rows); }
    {
        luisa::vector<float> ref;
        expect(ref_add_position(x, pos, ids, tokens, hidden, ref).ok());
        auto shader = device.compile(make_add_position_kernel());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bp = ocr::test::upload(device, stream, luisa::span<const float>{pos});
        auto bi = ocr::test::upload(device, stream, luisa::span<const int32_t>{ids});
        auto bo = device.create_buffer<float>(x.size());
        auto plan = plan_add_position_dispatch(tokens, hidden);
        OCR_DISPATCH_1D(stream, shader(bx, bp, bi, bo, tokens, hidden), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, x.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, 1e-5f, 1e-5f, "add position");
    }
    {
        // DeepStack injection at a caller-selected range (e.g. layer index 2
        // injects into tokens [4, 4+5)); the range is DATA.
        constexpr uint32_t start = 4u, count = 5u;
        luisa::vector<float> proj(static_cast<size_t>(count) * hidden);
        lcg.fill(proj, -0.25f, 0.25f);
        luisa::vector<float> ref;
        expect(ref_deepstack_inject(x, proj, tokens, hidden, start, ref).ok());
        auto shader = device.compile(make_deepstack_inject_kernel());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bp = ocr::test::upload(device, stream, luisa::span<const float>{proj});
        auto bo = device.create_buffer<float>(x.size());
        auto plan = plan_deepstack_inject_dispatch(tokens, hidden);
        OCR_DISPATCH_1D(stream, shader(bx, bp, bo, tokens, hidden, start, count), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, x.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, 1e-5f, 1e-5f, "deepstack inject");
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "vision_pixel_unshuffle"_test = [&] { test_pixel_unshuffle(device); };
    "vision_spatial_merge"_test = [&] { test_spatial_merge(device); };
    "vision_position_deepstack"_test = [&] { test_position_deepstack(device); };
}