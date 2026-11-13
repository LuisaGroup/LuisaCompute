// Test for ocr/ops/ops_conv (Phase 2).
// Covers: Conv2D 1x1/3x3/5x5/7x7 + general (2,3), depthwise, groups,
// asymmetric stride (2,1), dilation, padding; Conv1D + causal depthwise;
// BatchNorm; avg/adaptive/global pool; concat/split/shuffle. Ragged tails.
// Oracle: independent double host reference. Tol 2e-4 + 2e-4*|ref|.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_conv.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 2e-4f;
constexpr float kRtol = 2e-4f;

struct Conv2dCase {
    uint32_t n, in_c, h, w, out_c, kh, kw, sh, sw, ph, pw, dh, dw, groups;
};

void test_conv2d(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    auto dummy = device.create_buffer<float>(1);
    auto shader = device.compile(make_conv2d_kernel());
    expect(shader.compile_ok());
    const Conv2dCase cases[] = {
        {1, 3, 8, 9, 4, 1, 1, 1, 1, 0, 0, 1, 1, 1},   // 1x1
        {2, 3, 9, 7, 5, 3, 3, 1, 1, 1, 1, 1, 1, 1},   // 3x3 pad
        {1, 2, 11, 13, 3, 5, 5, 2, 2, 2, 2, 1, 1, 1}, // 5x5 stride 2
        {1, 1, 15, 17, 2, 7, 7, 1, 1, 3, 3, 1, 1, 1}, // 7x7 pad 3
        {1, 4, 10, 9, 4, 3, 3, 2, 1, 1, 0, 1, 1, 4},  // depthwise asym stride (2,1)
        {1, 4, 12, 11, 8, 3, 3, 1, 1, 2, 2, 2, 2, 2}, // dilated groups=2
        {2, 6, 7, 5, 9, 2, 3, 1, 1, 0, 1, 1, 1, 3},   // general (2,3) groups=3
    };
    for (auto c : cases) {
        auto oh_r = conv2d_out_dim(c.h, c.kh, c.sh, c.ph, c.dh);
        auto ow_r = conv2d_out_dim(c.w, c.kw, c.sw, c.pw, c.dw);
        expect(oh_r.ok() && ow_r.ok());
        const uint32_t oh = *oh_r.value, ow = *ow_r.value;
        luisa::vector<float> x(static_cast<size_t>(c.n) * c.in_c * c.h * c.w);
        luisa::vector<float> w(static_cast<size_t>(c.out_c) * (c.in_c / c.groups) * c.kh * c.kw);
        luisa::vector<float> bias(c.out_c);
        lcg.fill(x, -1.f, 1.f);
        lcg.fill(w, -1.f, 1.f);
        lcg.fill(bias, -0.2f, 0.2f);
        luisa::vector<float> ref;
        uint32_t rh = 0, rw = 0;
        expect(ref_conv2d(x, w, bias, c.n, c.in_c, c.h, c.w, c.out_c, c.kh, c.kw,
                          c.sh, c.sw, c.ph, c.pw, c.dh, c.dw, c.groups, ref, rh, rw).ok());
        expect(rh == oh && rw == ow);
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bw = ocr::test::upload(device, stream, luisa::span<const float>{w});
        auto bb = ocr::test::upload(device, stream, luisa::span<const float>{bias});
        auto bo = device.create_buffer<float>(ref.size());
        auto plan = plan_conv2d_dispatch(c.n, c.out_c, oh, ow);
        expect(plan.ok());
        OCR_DISPATCH_1D(stream, shader(bx, bw, bb, bo, c.n, c.in_c, c.h, c.w, c.out_c, c.kh, c.kw,
                         c.sh, c.sw, c.ph, c.pw, c.dh, c.dw, c.groups, oh, ow, 1u), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, ref.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "conv2d");
    }
    // Degenerate geometry rejected host-side.
    expect(!conv2d_out_dim(2, 3, 1, 0, 1).ok());
    expect(!conv2d_out_dim(4, 3, 0, 0, 1).ok());
}

void test_conv1d(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    auto shader = device.compile(make_conv1d_kernel());
    expect(shader.compile_ok());
    struct C { uint32_t n, c, len, out_c, k, stride, pad, dil, groups; bool causal; };
    const C cases[] = {
        {1, 4, 17, 4, 3, 1, 1, 1, 1, false},
        {2, 6, 21, 6, 4, 1, 0, 1, 6, true}, // causal depthwise (k=4 -> pad_left 3)
        {1, 2, 19, 4, 3, 2, 0, 2, 2, false},// strided dilated grouped
    };
    for (auto c : cases) {
        auto ol_r = conv1d_out_len(c.len, c.k, c.stride, c.pad, c.dil, c.causal);
        expect(ol_r.ok());
        const uint32_t ol = *ol_r.value;
        luisa::vector<float> x(static_cast<size_t>(c.n) * c.c * c.len);
        luisa::vector<float> w(static_cast<size_t>(c.out_c) * (c.c / c.groups) * c.k);
        lcg.fill(x, -1.f, 1.f);
        lcg.fill(w, -1.f, 1.f);
        luisa::vector<float> ref;
        uint32_t rl = 0;
        expect(ref_conv1d(x, w, {}, c.n, c.c, c.len, c.out_c, c.k, c.stride, c.pad, c.dil,
                          c.groups, c.causal, ref, rl).ok());
        expect(rl == ol);
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bw = ocr::test::upload(device, stream, luisa::span<const float>{w});
        auto dummy = device.create_buffer<float>(1);
        auto bo = device.create_buffer<float>(ref.size());
        auto plan = plan_conv1d_dispatch(c.n, c.out_c, ol);
        OCR_DISPATCH_1D(stream, shader(bx, bw, dummy, bo, c.n, c.c, c.len, c.out_c, c.k, c.stride, c.pad,
                         c.dil, c.groups, c.causal ? 1u : 0u, ol, 0u), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, ref.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "conv1d");
    }
}

void test_norm_pool_shape(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t n = 2u, c = 6u, h = 9u, w = 7u;
    luisa::vector<float> x(static_cast<size_t>(n) * c * h * w);
    lcg.fill(x, -2.f, 2.f);
    // BatchNorm.
    {
        luisa::vector<float> mean(c), var(c), gamma(c), beta(c);
        lcg.fill(mean, -0.5f, 0.5f);
        lcg.fill(var, 0.5f, 2.f);
        lcg.fill(gamma, 0.5f, 1.5f);
        lcg.fill(beta, -0.2f, 0.2f);
        luisa::vector<float> ref;
        expect(ref_batch_norm(x, mean, var, gamma, beta, n, c, h * w, 1e-5f, ref).ok());
        auto shader = device.compile(make_batch_norm_kernel());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bm = ocr::test::upload(device, stream, luisa::span<const float>{mean});
        auto bv = ocr::test::upload(device, stream, luisa::span<const float>{var});
        auto bg = ocr::test::upload(device, stream, luisa::span<const float>{gamma});
        auto bb = ocr::test::upload(device, stream, luisa::span<const float>{beta});
        auto bo = device.create_buffer<float>(x.size());
        auto plan = plan_elementwise_dispatch(x.size());
        OCR_DISPATCH_1D(stream, shader(bx, bm, bv, bg, bb, bo, n, c, h * w, 1e-5f), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, x.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "batch_norm");
    }
    // Avg pool 3x2 stride 2 pad (1,0).
    {
        auto shader = device.compile(make_avg_pool2d_kernel());
        luisa::vector<float> ref;
        uint32_t oh = 0, ow = 0;
        expect(ref_avg_pool2d(x, n, c, h, w, 3, 2, 2, 2, 1, 0, ref, oh, ow).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bo = device.create_buffer<float>(ref.size());
        auto plan = plan_conv2d_dispatch(n, c, oh, ow);
        OCR_DISPATCH_1D(stream, shader(bx, bo, n, c, h, w, 3u, 2u, 2u, 2u, 1u, 0u, oh, ow), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, ref.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "avg_pool2d");
    }
    // Adaptive pool to 2x3 (non-divisible bins).
    {
        auto shader = device.compile(make_adaptive_avg_pool2d_kernel());
        luisa::vector<float> ref;
        expect(ref_adaptive_avg_pool2d(x, n, c, h, w, 2, 3, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bo = device.create_buffer<float>(ref.size());
        auto plan = plan_conv2d_dispatch(n, c, 2, 3);
        OCR_DISPATCH_1D(stream, shader(bx, bo, n, c, h, w, 2u, 3u), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, ref.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "adaptive_avg_pool2d");
    }
    // Global pool.
    {
        auto shader = device.compile(make_global_avg_pool2d_kernel());
        luisa::vector<float> ref;
        uint32_t oh = 0, ow = 0;
        expect(ref_adaptive_avg_pool2d(x, n, c, h, w, 1, 1, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bo = device.create_buffer<float>(static_cast<size_t>(n) * c);
        auto plan = plan_elementwise_dispatch(static_cast<size_t>(n) * c);
        OCR_DISPATCH_1D(stream, shader(bx, bo, n, c, h * w), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, static_cast<size_t>(n) * c);
        stream << synchronize();
        ocr::test::expect_close(got, ref, 1e-3f, 1e-4f, "global_avg_pool2d");
    }
    // Concat / split / shuffle round trip.
    {
        constexpr uint32_t ca = 4u, cb = 2u;
        luisa::vector<float> a(static_cast<size_t>(n) * ca * h * w), b(static_cast<size_t>(n) * cb * h * w);
        lcg.fill(a, -1.f, 1.f);
        lcg.fill(b, -1.f, 1.f);
        luisa::vector<float> cat_ref;
        expect(ref_concat_channels(a, b, n, ca, cb, h * w, cat_ref).ok());
        auto ba = ocr::test::upload(device, stream, luisa::span<const float>{a});
        auto bb = ocr::test::upload(device, stream, luisa::span<const float>{b});
        auto bcat = device.create_buffer<float>(cat_ref.size());
        auto cat_shader = device.compile(make_concat_channels_kernel());
        auto plan = plan_elementwise_dispatch(cat_ref.size());
        OCR_DISPATCH_1D(stream, cat_shader(ba, bb, bcat, n, ca, cb, h * w), *plan.value);
        luisa::vector<float> cat;
        ocr::test::download(stream, bcat, cat, cat_ref.size());
        stream << synchronize();
        ocr::test::expect_close(cat, cat_ref, 0.f, 0.f, "concat exact");
        // Split channels [2, 2+3) of the concatenated tensor.
        auto split_shader = device.compile(make_split_channels_kernel());
        auto bsplit = device.create_buffer<float>(static_cast<size_t>(n) * 3 * h * w);
        OCR_DISPATCH_1D(stream, split_shader(bcat, bsplit, n, ca + cb, 2u, 3u, h * w), *plan.value);
        luisa::vector<float> split;
        ocr::test::download(stream, bsplit, split, static_cast<size_t>(n) * 3 * h * w);
        stream << synchronize();
        bool ok = true;
        for (uint32_t ni = 0; ni < n && ok; ni++) {
            for (uint32_t ci = 0; ci < 3 && ok; ci++) {
                for (uint32_t p = 0; p < h * w && ok; p++) {
                    ok = split[(static_cast<size_t>(ni) * 3 + ci) * h * w + p] ==
                         cat_ref[(static_cast<size_t>(ni) * (ca + cb) + 2 + ci) * h * w + p];
                }
            }
        }
        expect(ok) << "split channels exact";
        // Channel shuffle groups=2.
        luisa::vector<float> shuf_ref;
        expect(ref_channel_shuffle(cat_ref, n, ca + cb, h * w, 2u, shuf_ref).ok());
        auto shuf_shader = device.compile(make_channel_shuffle_kernel());
        auto bshuf = device.create_buffer<float>(cat_ref.size());
        OCR_DISPATCH_1D(stream, shuf_shader(bcat, bshuf, n, ca + cb, h * w, 2u), *plan.value);
        luisa::vector<float> shuf;
        ocr::test::download(stream, bshuf, shuf, cat_ref.size());
        stream << synchronize();
        ocr::test::expect_close(shuf, shuf_ref, 0.f, 0.f, "channel shuffle exact");
    }
}

// S5: invalid groups/divisibility -> host invalid_argument; device writes
// defined zeros (never silently drops channels or divides by zero).
void test_conv_invalid_groups(Device &device) {
    expect(!validate_conv2d_inputs(1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1, 3).ok());// 4%3
    expect(!validate_conv2d_inputs(1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1, 0).ok());// groups 0
    expect(!validate_conv2d_inputs(1, 4, 8, 8, 5, 3, 3, 1, 1, 1, 1, 2).ok());// 5%2
    expect(validate_conv2d_inputs(1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1, 2).ok());
    expect(!validate_conv1d_inputs(1, 4, 8, 4, 3, 1, 1, 3).ok());
    expect(!validate_conv1d_inputs(1, 4, 8, 4, 3, 1, 1, 0).ok());
    auto stream = device.create_stream();
    // Device: groups=3 with in_c=4 -> defined zero output, no crash.
    constexpr uint32_t n = 1u, in_c = 4u, h = 5u, w = 5u, out_c = 6u, kh = 3u, kw = 3u;
    constexpr uint32_t oh = 3u, ow = 3u;// pad 1 stride 1 geometry
    luisa::vector<float> x(static_cast<size_t>(n) * in_c * h * w, 1.f);
    luisa::vector<float> wgt(static_cast<size_t>(out_c) * 2 * kh * kw, 1.f);
    auto shader = device.compile(make_conv2d_kernel());
    auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bw = ocr::test::upload(device, stream, luisa::span<const float>{wgt});
    auto dummy = device.create_buffer<float>(1);
    auto bo = device.create_buffer<float>(static_cast<size_t>(n) * out_c * oh * ow);
    auto plan = plan_conv2d_dispatch(n, out_c, oh, ow);
    OCR_DISPATCH_1D(stream, shader(bx, bw, dummy, bo, n, in_c, h, w, out_c, kh, kw,
                     1u, 1u, 1u, 1u, 1u, 1u, 3u, oh, ow, 0u), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, static_cast<size_t>(n) * out_c * oh * ow);
    stream << synchronize();
    bool all_zero = true;
    for (auto v : got) { all_zero = all_zero && v == 0.f; }
    expect(all_zero) << "conv2d invalid groups must write defined zeros";
}

// Layout-agnostic adaptive-pool TRUTH test: hand-computed PyTorch
// adaptive_avg_pool2d values (start=floor, end=ceil) on a channel-distinct
// input. A wrong-axis / wrong-layout pool cannot match these exact numbers.
// Independent of ref_adaptive_avg_pool2d (the expected values are literal).
void test_adaptive_pool_truth(Device &device) {
    auto stream = device.create_stream();
    auto shader = device.compile(make_adaptive_avg_pool2d_kernel());
    expect(shader.compile_ok());
    // n=1, c=2, h=3, w=3 -> out 2x2. Channel 0 values 1..9, channel 1 values
    // 11..19 (row-major). PyTorch bin edges for 3->2: start floor(i*3/2),
    // end ceil((i+1)*3/2) => bins [0,2) and [1,3) on each axis.
    const uint32_t n = 1, c = 2, h = 3, w = 3, oh = 2, ow = 2;
    luisa::vector<float> x(static_cast<size_t>(n) * c * h * w);
    for (uint32_t ci = 0; ci < c; ci++) {
        for (uint32_t p = 0; p < h * w; p++) {
            x[static_cast<size_t>(ci) * h * w + p] = static_cast<float>(ci * 10 + p + 1);
        }
    }
    // Hand-computed expected (channel-major output [c, oh, ow]):
    //  ch0: bin(0,0)=mean{1,2,4,5}=3; (0,1)=mean{3,6}=4;
    //       (1,0)=mean{4,5,7,8}=6; (1,1)=mean{6,9}=7.
    //  ch1: same +10 => 13, 14, 16, 17.
    const luisa::vector<float> expected = {3.f, 4.f, 6.f, 7.f, 13.f, 14.f, 16.f, 17.f};
    auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bo = device.create_buffer<float>(expected.size());
    auto plan = plan_conv2d_dispatch(n, c, oh, ow);
    OCR_DISPATCH_1D(stream, shader(bx, bo, n, c, h, w, oh, ow), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, expected.size());
    stream << synchronize();
    ocr::test::expect_close(got, expected, 1e-5f, 1e-5f, "adaptive pool PyTorch truth (floor/ceil bins)");
    // The CPU reference must agree with the same hand-computed truth.
    luisa::vector<float> ref;
    expect(ref_adaptive_avg_pool2d(x, n, c, h, w, oh, ow, ref).ok());
    ocr::test::expect_close(ref, expected, 1e-5f, 1e-5f, "adaptive pool reference truth");
    // A non-divisible 5->3 case exercises ceil bins. h=5,w=1,c=1 -> out 3x1;
    // values 1..5. PyTorch bins: [0,2)->mean{1,2}=1.5; [1,4)->mean{2,3,4}=3;
    // [3,5)->mean{4,5}=4.5. (The old floor/floor end gave [0,1),[1,3),[3,4)
    // = 1,2.5,4 -- so this case distinguishes the ceil convention.)
    {
        const uint32_t h2 = 5, w2 = 1, oh2 = 3, ow2 = 1;
        luisa::vector<float> x2 = {1.f, 2.f, 3.f, 4.f, 5.f};
        const luisa::vector<float> exp2 = {1.5f, 3.f, 4.5f};
        auto bx2 = ocr::test::upload(device, stream, luisa::span<const float>{x2});
        auto bo2 = device.create_buffer<float>(exp2.size());
        auto plan2 = plan_conv2d_dispatch(1, 1, oh2, ow2);
        OCR_DISPATCH_1D(stream, shader(bx2, bo2, 1u, 1u, h2, w2, oh2, ow2), *plan2.value);
        luisa::vector<float> got2;
        ocr::test::download(stream, bo2, got2, exp2.size());
        stream << synchronize();
        ocr::test::expect_close(got2, exp2, 1e-5f, 1e-5f, "adaptive pool 5->3 ceil-bin truth");
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "conv2d_all_geometries"_test = [&] { test_conv2d(device); };
    "conv1d_causal_depthwise"_test = [&] { test_conv1d(device); };
    "batchnorm_pool_shape_ops"_test = [&] { test_norm_pool_shape(device); };
    "conv_invalid_groups"_test = [&] { test_conv_invalid_groups(device); };
    "adaptive_pool_truth"_test = [&] { test_adaptive_pool_truth(device); };
}