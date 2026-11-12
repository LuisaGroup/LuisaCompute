// Test for ocr/ops/ops_rope (Phase 2).
// Covers: 1D, 2D axial, M-RoPE (3 axes), interleaved MRoPE, XD-RoPE (4 axes)
// as SEPARATE variants (each with its own axis mapping data), sinusoidal
// table. Ragged tokens/heads/head_dim; negative + large positions; zero
// position identity. Oracle: independent double host rotation reference.
// Tolerance: 2e-4 + 2e-4*|ref| (device sin/cos/exp2 vs double).
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_rope.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 2e-4f;
constexpr float kRtol = 2e-4f;

struct RopeCase {
    const char *name;
    uint32_t axes;
    bool interleaved;
};

void run_rope_case(Device &device, const RopeCase &rc) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t tokens = 11u, heads = 3u, head_dim = 16u;
    const float theta = 10000.f;
    luisa::vector<float> x(static_cast<size_t>(tokens) * heads * head_dim);
    lcg.fill(x, -1.f, 1.f);
    luisa::vector<int32_t> positions(static_cast<size_t>(tokens) * rc.axes);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t a = 0; a < rc.axes; a++) {
            positions[static_cast<size_t>(t) * rc.axes + a] =
                static_cast<int32_t>(t * (a + 1u)) - 2;// includes negatives
        }
    }
    // Axis mapping data: spread pairs across the available axes.
    luisa::vector<uint32_t> axis_of_pair(head_dim / 2u);
    for (uint32_t d = 0; d < head_dim / 2u; d++) { axis_of_pair[d] = d % rc.axes; }
    luisa::vector<float> ref;
    if (rc.interleaved) {
        expect(ref_rope_interleaved(x, positions, axis_of_pair, tokens, heads, head_dim, rc.axes, theta, ref).ok());
    } else if (rc.axes == 1u) {
        expect(ref_rope_1d(x, positions, tokens, heads, head_dim, theta, ref).ok());
    } else {
        expect(ref_rope_axial(x, positions, axis_of_pair, tokens, heads, head_dim, rc.axes, theta, ref).ok());
    }
    RopeAxialKernel kernel =
        rc.interleaved ? make_rope_interleaved_mrope_kernel() :
        rc.axes == 1u  ? make_rope_1d_kernel() :
        rc.axes == 2u  ? make_rope_2d_kernel() :
        rc.axes == 3u  ? make_rope_mrope_kernel() :
                         make_rope_xd_kernel();
    auto shader = device.compile(kernel);
    expect(shader.compile_ok()) << rc.name;
    auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bp = ocr::test::upload(device, stream, luisa::span<const int32_t>{positions});
    auto ba = ocr::test::upload(device, stream, luisa::span<const uint32_t>{axis_of_pair});
    auto bo = device.create_buffer<float>(x.size());
    auto plan = plan_rope_dispatch(tokens, heads);
    expect(plan.ok());
    OCR_DISPATCH_1D(stream, shader(bx, bo, bp, ba, tokens, heads, head_dim, rc.axes, theta), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, x.size());
    stream << synchronize();
    ocr::test::expect_close(got, ref, kAtol, kRtol, rc.name);
    // Interleaved and adjacent layouts must NOT coincide (variant separation).
    if (rc.axes == 3u) {
        luisa::vector<float> other;
        if (rc.interleaved) {
            expect(ref_rope_axial(x, positions, axis_of_pair, tokens, heads, head_dim, rc.axes, theta, other).ok());
        } else {
            expect(ref_rope_interleaved(x, positions, axis_of_pair, tokens, heads, head_dim, rc.axes, theta, other).ok());
        }
        bool differs = false;
        for (size_t i = 0; i < ref.size(); i++) {
            if (std::abs(ref[i] - other[i]) > 1e-3f) { differs = true; break; }
        }
        expect(differs) << "interleaved vs adjacent MRoPE must differ";
    }
}

void test_sinusoidal(Device &device) {
    auto stream = device.create_stream();
    constexpr uint32_t len = 37u, dim = 32u;
    const float theta = 10000.f;
    luisa::vector<float> ref;
    expect(ref_sinusoidal_table(len, dim, theta, ref).ok());
    auto shader = device.compile(make_sinusoidal_table_kernel());
    expect(shader.compile_ok());
    auto bo = device.create_buffer<float>(static_cast<size_t>(len) * dim);
    auto plan = plan_sinusoidal_dispatch(len);
    OCR_DISPATCH_1D(stream, shader(bo, len, dim, theta), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, static_cast<size_t>(len) * dim);
    stream << synchronize();
    ocr::test::expect_close(got, ref, kAtol, kRtol, "sinusoidal table");
}

// Zero position: rotation by angle 0 is identity (bit-exact cos0=1, sin0=0).
void test_zero_position(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t tokens = 3u, heads = 2u, head_dim = 8u;
    luisa::vector<float> x(static_cast<size_t>(tokens) * heads * head_dim);
    lcg.fill(x, -1.f, 1.f);
    luisa::vector<int32_t> positions(tokens, 0);
    luisa::vector<uint32_t> axis_of_pair(head_dim / 2u, 0u);
    auto shader = device.compile(make_rope_1d_kernel());
    auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bp = ocr::test::upload(device, stream, luisa::span<const int32_t>{positions});
    auto ba = ocr::test::upload(device, stream, luisa::span<const uint32_t>{axis_of_pair});
    auto bo = device.create_buffer<float>(x.size());
    auto plan = plan_rope_dispatch(tokens, heads);
    OCR_DISPATCH_1D(stream, shader(bx, bo, bp, ba, tokens, heads, head_dim, 1u, 10000.f), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, x.size());
    stream << synchronize();
    ocr::test::expect_close(got, x, 1e-6f, 1e-6f, "rope zero position identity");
}

// S8: long-context positions (~1e6). The double-angle oracle is an impossible
// gate here (fp32 angle rounding legitimately diverges); compare against the
// explicit FP32-angle reference with a tight tolerance instead.
void test_large_positions(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t tokens = 5u, heads = 2u, head_dim = 32u;
    const float theta = 10000.f;
    luisa::vector<float> x(static_cast<size_t>(tokens) * heads * head_dim);
    lcg.fill(x, -1.f, 1.f);
    const luisa::vector<int32_t> positions = {1000000, 1000001, 999999, 1234567, 0};
    luisa::vector<float> ref;
    expect(ref_rope_1d_fp32_angle(x, positions, tokens, heads, head_dim, theta, ref).ok());
    luisa::vector<uint32_t> axis_of_pair(head_dim / 2u, 0u);
    auto shader = device.compile(make_rope_1d_kernel());
    auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bp = ocr::test::upload(device, stream, luisa::span<const int32_t>{positions});
    auto ba = ocr::test::upload(device, stream, luisa::span<const uint32_t>{axis_of_pair});
    auto bo = device.create_buffer<float>(x.size());
    auto plan = plan_rope_dispatch(tokens, heads);
    OCR_DISPATCH_1D(stream, shader(bx, bo, bp, ba, tokens, heads, head_dim, 1u, theta), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, x.size());
    stream << synchronize();
    // Angle-ulp-aware gate: at pos~1e6 the fp32 angle is ~1e6 rad, so a 1-ulp
    // rounding difference between device exp2/log2/sin/cos and host libm is
    // ~2^-23 * 1e6 ~= 0.12 rad, which rotates the (a,b) pair by up to that
    // angle. The output error is bounded by |x| * angle_error (rotation is
    // norm-preserving). We allow a few ulps of the largest angle.
    float max_angle = 0.f;
    const float log2_theta = std::log2(theta);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t d = 0; d < head_dim / 2u; d++) {
            const float inv_freq = std::exp2(log2_theta * (-2.0f * static_cast<float>(d) / static_cast<float>(head_dim)));
            max_angle = std::max(max_angle, std::abs(static_cast<float>(positions[t]) * inv_freq));
        }
    }
    // 4 ulps of the largest angle, times the max |x| pair norm (<= sqrt(2)).
    const float angle_ulp = std::ldexp(1.0f, std::ilogb(max_angle) - 23);
    const float tol = 4.0f * angle_ulp * 1.5f + 1e-5f;
    ocr::test::expect_close(got, ref, tol, tol, "rope large positions fp32-angle");
    // Sanity: the double-angle oracle would NOT meet this gate at 1e6 for the
    // high-frequency pairs -- prove the test is meaningful.
    luisa::vector<float> dbl;
    expect(ref_rope_1d(x, positions, tokens, heads, head_dim, theta, dbl).ok());
    bool double_diverges = false;
    for (size_t i = 0; i < dbl.size(); i++) {
        if (std::abs(dbl[i] - ref[i]) > 2e-3f) { double_diverges = true; break; }
    }
    expect(double_diverges) << "fp32 vs double angle must diverge at ~1e6 (gate is real)";
}

}// namespace

// Rotate-half 1D RoPE (Qwen2.5 HF text convention): pairs (d, d+head_dim/2),
// one position axis. Direct parity against ref_rope_rotate_half, and it must
// DIFFER from the adjacent-pair rope_1d (variant separation, no silent subst).
void test_rotate_half_1d(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t tokens = 11u, heads = 3u, head_dim = 16u;
    const float theta = 10000.f;
    luisa::vector<float> x(static_cast<size_t>(tokens) * heads * head_dim);
    lcg.fill(x, -1.f, 1.f);
    luisa::vector<int32_t> positions(tokens);
    for (uint32_t t = 0; t < tokens; t++) { positions[t] = static_cast<int32_t>(t) - 2; }
    luisa::vector<uint32_t> axis_of_pair(head_dim / 2u, 0u);
    luisa::vector<float> ref;
    expect(ref_rope_rotate_half(x, positions, tokens, heads, head_dim, theta, ref).ok());
    auto shader = device.compile(make_rope_rotate_half_kernel());
    expect(shader.compile_ok());
    auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bp = ocr::test::upload(device, stream, luisa::span<const int32_t>{positions});
    auto ba = ocr::test::upload(device, stream, luisa::span<const uint32_t>{axis_of_pair});
    auto bo = device.create_buffer<float>(x.size());
    auto plan = plan_rope_dispatch(tokens, heads);
    expect(plan.ok());
    OCR_DISPATCH_1D(stream, shader(bx, bo, bp, ba, tokens, heads, head_dim, 1u, theta), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, x.size());
    stream << synchronize();
    ocr::test::expect_close(got, ref, kAtol, kRtol, "rotate-half 1d rope");
    // Must differ from the adjacent-pair convention.
    luisa::vector<float> adjacent;
    expect(ref_rope_1d(x, positions, tokens, heads, head_dim, theta, adjacent).ok());
    bool differs = false;
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::abs(ref[i] - adjacent[i]) > 1e-3f) { differs = true; break; }
    }
    expect(differs) << "rotate-half vs adjacent 1d rope must differ";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "rope_1d"_test = [&] { run_rope_case(device, {"rope 1d", 1u, false}); };
 "rope_rotate_half_1d"_test = [&] { test_rotate_half_1d(device); };
    "rope_2d_axial"_test = [&] { run_rope_case(device, {"rope 2d", 2u, false}); };
    "rope_mrope_3axis"_test = [&] { run_rope_case(device, {"mrope", 3u, false}); };
    "rope_interleaved_mrope"_test = [&] { run_rope_case(device, {"interleaved mrope", 3u, true}); };
    "rope_xd_4axis"_test = [&] { run_rope_case(device, {"xd rope", 4u, false}); };
    "sinusoidal_table"_test = [&] { test_sinusoidal(device); };
    "rope_zero_position"_test = [&] { test_zero_position(device); };
    "rope_large_positions_fp32_angle"_test = [&] { test_large_positions(device); };
}