// Test for ocr/ops/ops_gemm (Phase 2).
// Covers: ragged FP32 GEMM all epilogues; decode GEMV/small-M; packed
// FP16/BF16 B; grouped INT4 sym/asym (groups 64/128, odd nibble counts);
// grouped INT8; dequantized-B oracle arm; gated SwiGLU/GeGLU; W4A16
// fail-closed; shader-cache compile-once; NaN propagation; aliasing rules.
// Oracle: independent double-accumulating host reference (cpu_reference).
// Tolerance: |diff| <= 2e-3 + 2e-3*|ref| (FP32 accumulation vs double).
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/runtime/dispatch_adapter.h"
#include "ocr/runtime/shader_cache.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 2e-3f;
constexpr float kRtol = 2e-3f;

struct GemmCase { uint32_t m, n, k; };
constexpr GemmCase kCases[] = {
    {1, 1, 1}, {3, 1, 17}, {1, 17, 3}, {16, 16, 16}, {17, 17, 17},
    {37, 3, 16}, {3, 37, 129}, {5, 7, 65}, {7, 5, 127}, {9, 11, 131},
};

constexpr GemmEpilogue kEpilogues[] = {
    GemmEpilogue::none, GemmEpilogue::bias, GemmEpilogue::bias_gelu_erf,
    GemmEpilogue::bias_gelu_tanh, GemmEpilogue::bias_relu, GemmEpilogue::bias_relu6,
    GemmEpilogue::bias_silu, GemmEpilogue::bias_hard_swish, GemmEpilogue::residual,
    GemmEpilogue::bias_residual,
};

[[nodiscard]] RefGemmEpilogue to_ref(GemmEpilogue ep) noexcept {
    return static_cast<RefGemmEpilogue>(static_cast<uint32_t>(ep));
}

[[nodiscard]] bool needs_residual(GemmEpilogue ep) noexcept {
    return ep == GemmEpilogue::residual || ep == GemmEpilogue::bias_residual;
}

void test_gemm_fp32(Device &device) {
    auto stream = device.create_stream();
    ShaderKeyRegistry registry;
    TypedShaderCache<GemmKernel> cache(device, registry);
    ocr::test::Lcg lcg;
    auto dummy = device.create_buffer<float>(1);
    for (auto ep : kEpilogues) {
        auto shader_r = cache.get_or_compile(gemm_variant_key(ep, GemmWeightFormat::fp32, false),
                                             [ep] { return make_gemm_kernel(ep); });
        expect(shader_r.ok()) << "compile fp32 gemm";
        if (!shader_r.ok()) { continue; }
        const auto &shader = **shader_r.value;
        for (auto c : kCases) {
            luisa::vector<float> a(static_cast<size_t>(c.m) * c.k), b(static_cast<size_t>(c.k) * c.n);
            lcg.fill(a, -1.f, 1.f);
            lcg.fill(b, -1.f, 1.f);
            // S6: bias_residual aux = [bias N | residual M*N], distinct values.
            luisa::vector<float> aux;
            if (ep == GemmEpilogue::bias_residual) {
                aux.resize(static_cast<size_t>(c.n) + static_cast<size_t>(c.m) * c.n);
                for (uint32_t j = 0; j < c.n; j++) { aux[j] = 0.25f + 0.01f * static_cast<float>(j); }
                luisa::vector<float> res(static_cast<size_t>(c.m) * c.n);
                lcg.fill(res, -0.5f, 0.5f);
                for (size_t i = 0; i < res.size(); i++) { aux[c.n + i] = res[i]; }
            } else {
                aux.resize(needs_residual(ep) ? static_cast<size_t>(c.m) * c.n : c.n);
                lcg.fill(aux, -0.5f, 0.5f);
            }
            luisa::vector<float> ref;
            auto rs = ref_gemm(a, b, aux, c.m, c.n, c.k, to_ref(ep), ref);
            expect(rs.ok());
            auto ba = ocr::test::upload(device, stream, luisa::span<const float>{a});
            auto bb = ocr::test::upload(device, stream, luisa::span<const float>{b});
            auto baux = ocr::test::upload(device, stream, luisa::span<const float>{aux});
            auto bc = device.create_buffer<float>(static_cast<size_t>(c.m) * c.n);
            auto plan = plan_gemm_dispatch(c.m, c.n, c.k);
            expect(plan.ok());
            OCR_DISPATCH_2D(stream, shader(ba, bb, bc, baux, c.m, c.n, c.k, plan.value->padded_cols), *plan.value);
            luisa::vector<float> got;
            ocr::test::download(stream, bc, got, static_cast<size_t>(c.m) * c.n);
            stream << synchronize();
            ocr::test::expect_close(got, ref, kAtol, kRtol, "gemm fp32 epilogue case");
        }
    }
    // compile-once: each epilogue key compiled exactly once despite 10 cases.
    for (auto ep : kEpilogues) {
        expect(registry.compile_count(gemm_variant_key(ep, GemmWeightFormat::fp32, false)) == 1u);
    }
}

void test_gemv(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    for (auto ep : {GemmEpilogue::none, GemmEpilogue::bias, GemmEpilogue::bias_silu, GemmEpilogue::residual}) {
        auto shader = device.compile(make_gemv_kernel(ep));
        expect(shader.compile_ok());
        for (uint32_t m : {1u, 2u, 4u}) {
            constexpr uint32_t n = 37u, k = 129u;
            luisa::vector<float> a(static_cast<size_t>(m) * k), b(static_cast<size_t>(k) * n);
            lcg.fill(a, -1.f, 1.f);
            lcg.fill(b, -1.f, 1.f);
            luisa::vector<float> aux(needs_residual(ep) ? static_cast<size_t>(m) * n : n);
            lcg.fill(aux, -0.5f, 0.5f);
            luisa::vector<float> ref;
            expect(ref_gemm(a, b, aux, m, n, k, to_ref(ep), ref).ok());
            auto ba = ocr::test::upload(device, stream, luisa::span<const float>{a});
            auto bb = ocr::test::upload(device, stream, luisa::span<const float>{b});
            auto baux = ocr::test::upload(device, stream, luisa::span<const float>{aux});
            auto bc = device.create_buffer<float>(static_cast<size_t>(m) * n);
            auto plan = plan_gemv_dispatch(m, n, k);
            expect(plan.ok());
            OCR_DISPATCH_1D(stream, shader(ba, bb, bc, baux, m, n, k), *plan.value);
            luisa::vector<float> got;
            ocr::test::download(stream, bc, got, static_cast<size_t>(m) * n);
            stream << synchronize();
            ocr::test::expect_close(got, ref, kAtol, kRtol, "gemv fp32");
        }
    }
}

// Packed-B GEMM against the host-rounded/dequantized oracle arm.
void test_gemm_packed(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    auto dummy_f = device.create_buffer<float>(1);
    auto dummy_u = device.create_buffer<uint32_t>(1);
    struct FmtCase { GemmWeightFormat fmt; uint32_t group; };
    const FmtCase formats[] = {
        {GemmWeightFormat::fp16, 0}, {GemmWeightFormat::bf16, 0},
        {GemmWeightFormat::int4_sym, 64}, {GemmWeightFormat::int4_asym, 64},
        {GemmWeightFormat::int4_sym, 128}, {GemmWeightFormat::int4_asym, 128},
        {GemmWeightFormat::int8_sym, 64}, {GemmWeightFormat::int8_sym, 128},
    };
    for (auto fc : formats) {
        auto ep = GemmEpilogue::bias_gelu_erf;
        auto shader = device.compile(make_gemm_packed_kernel(ep, fc.fmt));
        expect(shader.compile_ok()) << "compile packed gemm";
        // Ragged K*N incl. odd totals: (3,5,17)=255 odd, (5,7,65)=2275 odd,
        // (9,11,131)=12969 odd, (7,5,127)=4445 odd.
        for (auto c : {GemmCase{3, 5, 17}, GemmCase{5, 7, 65}, GemmCase{9, 11, 131}, GemmCase{7, 5, 127}, GemmCase{2, 3, 128}}) {
            const size_t kn = static_cast<size_t>(c.k) * c.n;
            luisa::vector<float> a(static_cast<size_t>(c.m) * c.k), b(kn), bias(c.n);
            lcg.fill(a, -1.f, 1.f);
            lcg.fill(b, -1.f, 1.f);
            lcg.fill(bias, -0.5f, 0.5f);
            // Build the stored B + the exact host-decoded oracle B.
            luisa::vector<float> b_ref;
            luisa::vector<uint32_t> words;
            luisa::vector<float> scales;
            luisa::vector<uint32_t> zps;
            if (fc.fmt == GemmWeightFormat::fp16 || fc.fmt == GemmWeightFormat::bf16) {
                words = ocr::test::pack_half_words(b, fc.fmt == GemmWeightFormat::bf16, b_ref);
            } else if (fc.fmt == GemmWeightFormat::int4_sym || fc.fmt == GemmWeightFormat::int4_asym) {
                const bool sym = fc.fmt == GemmWeightFormat::int4_sym;
                luisa::vector<uint8_t> packed, zp8;
                expect(quantize_int4(b, fc.group, sym, packed, scales, zp8).ok());
                luisa::vector<uint8_t> codes(kn);
                expect(unpack_int4_codes(packed, kn, codes).ok());
                words = ocr::test::pack_int4_words(codes);
                zps = ocr::test::widen_bytes(zp8);
                expect(ref_dequant_int4(packed, kn, scales, zp8, fc.group, sym, b_ref).ok());
            } else {
                luisa::vector<int8_t> codes;
                expect(quantize_int8(b, fc.group, codes, scales).ok());
                words = ocr::test::pack_int8_words(codes);
                expect(ref_dequant_int8(codes, kn, scales, fc.group, b_ref).ok());
            }
            luisa::vector<float> ref;
            expect(ref_gemm(a, b_ref, bias, c.m, c.n, c.k, to_ref(ep), ref).ok());
            auto ba = ocr::test::upload(device, stream, luisa::span<const float>{a});
            auto bb = ocr::test::upload(device, stream, luisa::span<const uint32_t>{words});
            auto bbias = ocr::test::upload(device, stream, luisa::span<const float>{bias});
            auto bscales_storage = scales.empty() ? Buffer<float>{} : ocr::test::upload(device, stream, luisa::span<const float>{scales});
            auto bzps_storage = zps.empty() ? Buffer<uint32_t>{} : ocr::test::upload(device, stream, luisa::span<const uint32_t>{zps});
            auto bscales = scales.empty() ? dummy_f.view() : bscales_storage.view();
            auto bzps = zps.empty() ? dummy_u.view() : bzps_storage.view();
            auto bc = device.create_buffer<float>(static_cast<size_t>(c.m) * c.n);
            auto plan = plan_gemm_dispatch(c.m, c.n, c.k);
            expect(plan.ok());
            OCR_DISPATCH_2D(stream, shader(ba, bb, bc, bbias, bscales, bzps, c.m, c.n, c.k, fc.group, plan.value->padded_cols), *plan.value);
            luisa::vector<float> got;
            ocr::test::download(stream, bc, got, static_cast<size_t>(c.m) * c.n);
            stream << synchronize();
            // Quantized arms carry storage rounding; the oracle decodes the
            // SAME stored codes, so the gate stays tight.
            ocr::test::expect_close(got, ref, kAtol, kRtol, "gemm packed");
        }
    }
}

// Packed GEMV decode path (INT4 asym, group 64) vs dequantized oracle.
void test_gemv_packed(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t m = 1u, n = 19u, k = 130u;// K*N = 2470, odd nibble tail
    const uint32_t group = 64u;
    luisa::vector<float> a(static_cast<size_t>(m) * k), b(static_cast<size_t>(k) * n), bias(n);
    lcg.fill(a, -1.f, 1.f);
    lcg.fill(b, -1.f, 1.f);
    lcg.fill(bias, -0.5f, 0.5f);
    luisa::vector<uint8_t> packed, zp8;
    luisa::vector<float> scales;
    expect(quantize_int4(b, group, false, packed, scales, zp8).ok());
    luisa::vector<uint8_t> codes(static_cast<size_t>(k) * n);
    expect(unpack_int4_codes(packed, codes.size(), codes).ok());
    auto words = ocr::test::pack_int4_words(codes);
    auto zps = ocr::test::widen_bytes(zp8);
    luisa::vector<float> b_ref;
    expect(ref_dequant_int4(packed, codes.size(), scales, zp8, group, false, b_ref).ok());
    luisa::vector<float> ref;
    expect(ref_gemm(a, b_ref, bias, m, n, k, RefGemmEpilogue::bias, ref).ok());
    auto shader = device.compile(make_gemv_packed_kernel(GemmEpilogue::bias, GemmWeightFormat::int4_asym));
    expect(shader.compile_ok());
    auto ba = ocr::test::upload(device, stream, luisa::span<const float>{a});
    auto bb = ocr::test::upload(device, stream, luisa::span<const uint32_t>{words});
    auto bbias = ocr::test::upload(device, stream, luisa::span<const float>{bias});
    auto bscales = ocr::test::upload(device, stream, luisa::span<const float>{scales});
    auto bzps = ocr::test::upload(device, stream, luisa::span<const uint32_t>{zps});
    auto bc = device.create_buffer<float>(static_cast<size_t>(m) * n);
    auto plan = plan_gemv_dispatch(m, n, k);
    OCR_DISPATCH_1D(stream, shader(ba, bb, bc, bbias, bscales, bzps, m, n, k, group), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bc, got, static_cast<size_t>(m) * n);
    stream << synchronize();
    ocr::test::expect_close(got, ref, kAtol, kRtol, "gemv int4 asym");
}

void test_gated_act(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t count = 1027u;// ragged tail
    luisa::vector<float> gate(count), up(count);
    lcg.fill(gate, -3.f, 3.f);
    lcg.fill(up, -2.f, 2.f);
    for (uint32_t act : {0u, 1u, 2u, 3u, 4u}) {
        auto shader = device.compile(make_gated_act_kernel(act));
        expect(shader.compile_ok());
        luisa::vector<float> ref;
        expect(ref_gated_act(gate, up, act, ref).ok());
        auto bg = ocr::test::upload(device, stream, luisa::span<const float>{gate});
        auto bu = ocr::test::upload(device, stream, luisa::span<const float>{up});
        auto bc = device.create_buffer<float>(count);
        auto plan = plan_gated_act_dispatch(count);
        OCR_DISPATCH_1D(stream, shader(bg, bu, bc, count), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bc, got, count);
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "gated act");
    }
}

void test_w4a16_fails_closed() {
    auto r = make_gemm_w4a16_kernel(GemmEpilogue::bias);
    expect(!r.ok());
    expect(r.status.code == StatusCode::unsupported);
}

void test_gemm_nan(Device &device) {
    auto stream = device.create_stream();
    constexpr uint32_t m = 2u, n = 3u, k = 5u;
    luisa::vector<float> a(static_cast<size_t>(m) * k, 1.f), b(static_cast<size_t>(k) * n, 1.f);
    a[2] = std::numeric_limits<float>::quiet_NaN();
    auto dummy = device.create_buffer<float>(1);
    auto shader = device.compile(make_gemm_kernel(GemmEpilogue::none));
    auto ba = ocr::test::upload(device, stream, luisa::span<const float>{a});
    auto bb = ocr::test::upload(device, stream, luisa::span<const float>{b});
    auto bc = device.create_buffer<float>(static_cast<size_t>(m) * n);
    auto plan = plan_gemm_dispatch(m, n, k);
    OCR_DISPATCH_2D(stream, shader(ba, bb, bc, dummy, m, n, k, plan.value->padded_cols), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bc, got, static_cast<size_t>(m) * n);
    stream << synchronize();
    // Row 0 of A has the NaN at k=2 -> entire output row 0 NaN; row 1 finite.
    bool ok = true;
    for (uint32_t j = 0; j < n; j++) {
        ok = ok && std::isnan(got[j]);
        ok = ok && std::abs(got[n + j] - static_cast<float>(k)) < 1e-4f;
    }
    expect(ok) << "NaN propagates through the affected GEMM row only";
}

// Legal in-place aliasing: residual epilogue with C == aux (residual buffer).
void test_gemm_alias_residual(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t m = 5u, n = 7u, k = 9u;
    luisa::vector<float> a(static_cast<size_t>(m) * k), b(static_cast<size_t>(k) * n);
    luisa::vector<float> r(static_cast<size_t>(m) * n);
    lcg.fill(a, -1.f, 1.f);
    lcg.fill(b, -1.f, 1.f);
    lcg.fill(r, -1.f, 1.f);
    luisa::vector<float> ref;
    expect(ref_gemm(a, b, r, m, n, k, RefGemmEpilogue::residual, ref).ok());
    auto shader = device.compile(make_gemm_kernel(GemmEpilogue::residual));
    auto ba = ocr::test::upload(device, stream, luisa::span<const float>{a});
    auto bb = ocr::test::upload(device, stream, luisa::span<const float>{b});
    auto bc = ocr::test::upload(device, stream, luisa::span<const float>{r});
    auto plan = plan_gemm_dispatch(m, n, k);
    // C and aux bind the SAME buffer: each thread reads aux[i] before writing
    // C[i] at the same index -- elementwise in-place is legal.
    OCR_DISPATCH_2D(stream, shader(ba, bb, bc, bc, m, n, k, plan.value->padded_cols), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bc, got, static_cast<size_t>(m) * n);
    stream << synchronize();
    ocr::test::expect_close(got, ref, kAtol, kRtol, "gemm in-place residual");
}

// S6: bias_residual with DISTINCT bias and residual values on device.
void test_bias_residual_layout(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t m = 7u, n = 11u, k = 13u;
    luisa::vector<float> a(static_cast<size_t>(m) * k), b(static_cast<size_t>(k) * n);
    lcg.fill(a, -1.f, 1.f);
    lcg.fill(b, -1.f, 1.f);
    luisa::vector<float> aux(static_cast<size_t>(n) + static_cast<size_t>(m) * n);
    for (uint32_t j = 0; j < n; j++) { aux[j] = 2.0f + 0.1f * static_cast<float>(j); }// bias: clearly distinct
    for (size_t i = 0; i < static_cast<size_t>(m) * n; i++) { aux[n + i] = -0.75f + 0.001f * static_cast<float>(i); }
    luisa::vector<float> ref;
    expect(ref_gemm(a, b, aux, m, n, k, RefGemmEpilogue::bias_residual, ref).ok());
    auto shader = device.compile(make_gemm_kernel(GemmEpilogue::bias_residual));
    auto ba = ocr::test::upload(device, stream, luisa::span<const float>{a});
    auto bb = ocr::test::upload(device, stream, luisa::span<const float>{b});
    auto baux = ocr::test::upload(device, stream, luisa::span<const float>{aux});
    auto bc = device.create_buffer<float>(static_cast<size_t>(m) * n);
    auto plan = plan_gemm_dispatch(m, n, k);
    expect(plan.ok());
    OCR_DISPATCH_2D(stream, shader(ba, bb, bc, baux, m, n, k, plan.value->padded_cols), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bc, got, static_cast<size_t>(m) * n);
    stream << synchronize();
    ocr::test::expect_close(got, ref, kAtol, kRtol, "bias_residual [bias|residual] layout");
    // Host aux-size helper agrees.
    auto aux_n = gemm_aux_elements(GemmEpilogue::bias_residual, m, n);
    expect(aux_n.ok());
    expect(*aux_n.value == static_cast<uint64_t>(n) + static_cast<uint64_t>(m) * n);
}

// S1: plan helpers fail closed on uint32 index-envelope overflow (host-only,
// no allocation).
void test_gemm_envelope_overflow() {
    // M*K overflow: m=70000, k=70000 -> 4.9e9 > UINT32_MAX.
    expect(!plan_gemm_dispatch(70000, 4, 70000).ok());
    expect(!plan_gemv_dispatch(70000, 4, 70000).ok());
    // K*N overflow.
    expect(!plan_gemm_dispatch(4, 70000, 70000).ok());
    // M*N overflow.
    expect(!plan_gemm_dispatch(70000, 70000, 4).ok());
    // uint64 multiplication overflow.
    expect(!plan_gemm_dispatch(UINT64_MAX / 2, 4, 4).ok());
    // Exactly at the envelope is allowed (no launch here).
    expect(plan_gemm_dispatch(1, 1, UINT32_MAX).ok());
    expect(!plan_gemm_dispatch(1, 2, UINT32_MAX).ok());// K*N = 2*UINT32_MAX
    // aux size helper overflow.
    expect(!gemm_aux_elements(GemmEpilogue::bias_residual, UINT64_MAX / 2, 4).ok());
    // S1 aux envelope: bias_residual aux = N + M*N must fit uint32 even when
    // M*N alone fits. m=4112, n=1044480 (verified by scan): M*N=4294901760
    // (< 2^32), the 2D grid fits, but aux = 4294901760 + 1044480 = 4295946240
    // > UINT32_MAX -> the epilogue-aware plan rejects it.
    expect(plan_gemm_dispatch(4112, 1044480ull, 1).ok());// M*N + grid fit
    expect(!plan_gemm_dispatch_epilogue(4112, 1044480ull, 1, GemmEpilogue::bias_residual).ok());
    // non-bias_residual epilogues are unaffected by the aux bound.
    expect(plan_gemm_dispatch_epilogue(4112, 1044480ull, 1, GemmEpilogue::bias).ok());
    // small case passes the epilogue-aware plan.
    expect(plan_gemm_dispatch_epilogue(7, 11, 13, GemmEpilogue::bias_residual).ok());
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "gemm_fp32_all_epilogues"_test = [&] { test_gemm_fp32(device); };
    "gemv_small_m"_test = [&] { test_gemv(device); };
    "gemm_packed_formats"_test = [&] { test_gemm_packed(device); };
    "gemv_packed_int4"_test = [&] { test_gemv_packed(device); };
    "gated_activations"_test = [&] { test_gated_act(device); };
    "w4a16_unsupported"_test = [] { test_w4a16_fails_closed(); };
    "gemm_nan_row"_test = [&] { test_gemm_nan(device); };
    "gemm_alias_residual"_test = [&] { test_gemm_alias_residual(device); };
    "gemm_bias_residual_layout"_test = [&] { test_bias_residual_layout(device); };
    "gemm_envelope_overflow"_test = [] { test_gemm_envelope_overflow(); };
}