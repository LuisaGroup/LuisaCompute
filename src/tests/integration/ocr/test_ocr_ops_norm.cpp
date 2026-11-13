// Test for ocr/ops/ops_norm + ops_decode (Phase 2).
// Covers: LayerNorm/RMSNorm (weighted + parameterless), QK-norm shape
// (rows = tokens*heads), L2, row-sum reduction, residual add (incl. in-place
// aliasing), embedding lookup, multimodal scatter, argmax ties, softmax,
// zero/NaN degenerate rows. Oracle: independent double host reference.
// Tolerance: 1e-4 + 1e-4*|ref| (norms); 1e-5 for exact copies.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_decode.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-4f;

void test_layer_norm(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    auto dummy = device.create_buffer<float>(1);
    auto shader = device.compile(make_layer_norm_kernel());
    expect(shader.compile_ok());
    for (uint32_t rows : {1u, 7u, 257u}) {
        for (uint32_t h : {1u, 33u, 128u}) {
            for (bool weighted : {false, true}) {
                luisa::vector<float> x(static_cast<size_t>(rows) * h);
                lcg.fill(x, -2.f, 2.f);
                luisa::vector<float> gamma(weighted ? h : 0), beta(weighted ? h : 0);
                if (weighted) { lcg.fill(gamma, 0.5f, 1.5f); lcg.fill(beta, -0.2f, 0.2f); }
                luisa::vector<float> ref;
                expect(ref_layer_norm(x, gamma, beta, rows, h, 1e-5f, ref).ok());
                auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
                auto bg_storage = weighted ? ocr::test::upload(device, stream, luisa::span<const float>{gamma}) : Buffer<float>{};
                auto bb_storage = weighted ? ocr::test::upload(device, stream, luisa::span<const float>{beta}) : Buffer<float>{};
                auto bg = weighted ? bg_storage.view() : dummy.view();
                auto bb = weighted ? bb_storage.view() : dummy.view();
                auto by = device.create_buffer<float>(x.size());
                auto plan = plan_norm_dispatch(rows);
                OCR_DISPATCH_1D(stream, shader(bx, by, bg, bb, rows, h, 1e-5f, weighted ? 1u : 0u), *plan.value);
                luisa::vector<float> got;
                ocr::test::download(stream, by, got, x.size());
                stream << synchronize();
                ocr::test::expect_close(got, ref, kAtol, kRtol, "layer_norm");
            }
        }
    }
    // Degenerate: zero row -> zero output (eps guards division); NaN row -> NaN.
    {
        constexpr uint32_t rows = 2u, h = 8u;
        luisa::vector<float> x(static_cast<size_t>(rows) * h, 0.f);
        for (uint32_t i = 0; i < h; i++) { x[h + i] = std::numeric_limits<float>::quiet_NaN(); }
        luisa::vector<float> ref;
        expect(ref_layer_norm(x, {}, {}, rows, h, 1e-5f, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto by = device.create_buffer<float>(x.size());
        auto plan = plan_norm_dispatch(rows);
        OCR_DISPATCH_1D(stream, shader(bx, by, dummy, dummy, rows, h, 1e-5f, 0u), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, by, got, x.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "layer_norm degenerate");
    }
}

void test_rms_norm_qk(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    auto dummy = device.create_buffer<float>(1);
    auto shader = device.compile(make_rms_norm_kernel());
    expect(shader.compile_ok());
    // QK-Norm shape: tokens*heads rows of head_dim.
    constexpr uint32_t tokens = 13u, heads = 6u, head_dim = 24u;
    constexpr uint32_t rows = tokens * heads;
    for (bool weighted : {false, true}) {
        luisa::vector<float> x(static_cast<size_t>(rows) * head_dim);
        lcg.fill(x, -2.f, 2.f);
        luisa::vector<float> gamma(weighted ? head_dim : 0);
        if (weighted) { lcg.fill(gamma, 0.5f, 1.5f); }
        luisa::vector<float> ref;
        expect(ref_rms_norm(x, gamma, rows, head_dim, 1e-6f, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bg_storage = weighted ? ocr::test::upload(device, stream, luisa::span<const float>{gamma}) : Buffer<float>{};
        auto bg = weighted ? bg_storage.view() : dummy.view();
        auto by = device.create_buffer<float>(x.size());
        auto plan = plan_norm_dispatch(rows);
        OCR_DISPATCH_1D(stream, shader(bx, by, bg, rows, head_dim, 1e-6f, weighted ? 1u : 0u), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, by, got, x.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "rms_norm qk");
    }
    // All-zero input: eps floor keeps output finite (zero).
    {
        luisa::vector<float> x(head_dim, 0.f);
        luisa::vector<float> ref;
        expect(ref_rms_norm(x, {}, 1u, head_dim, 1e-6f, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto by = device.create_buffer<float>(x.size());
        auto plan = plan_norm_dispatch(1);
        OCR_DISPATCH_1D(stream, shader(bx, by, dummy, 1u, head_dim, 1e-6f, 0u), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, by, got, x.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "rms_norm zero");
    }
}

void test_l2_rowsum_residual(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t rows = 19u, h = 65u;
    luisa::vector<float> x(static_cast<size_t>(rows) * h);
    lcg.fill(x, -2.f, 2.f);
    // Zero row exercises the L2 floor.
    for (uint32_t i = 0; i < h; i++) { x[i] = 0.f; }
    {
        auto shader = device.compile(make_l2_norm_kernel());
        luisa::vector<float> ref;
        expect(ref_l2_norm(x, rows, h, 1e-12f, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto by = device.create_buffer<float>(x.size());
        auto plan = plan_norm_dispatch(rows);
        OCR_DISPATCH_1D(stream, shader(bx, by, rows, h, 1e-12f), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, by, got, x.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, kAtol, kRtol, "l2_norm");
    }
    {
        auto shader = device.compile(make_row_sum_kernel());
        luisa::vector<float> ref;
        expect(ref_row_sum(x, rows, h, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bo = device.create_buffer<float>(rows);
        auto plan = plan_norm_dispatch(rows);
        OCR_DISPATCH_1D(stream, shader(bx, bo, rows, h), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, rows);
        stream << synchronize();
        ocr::test::expect_close(got, ref, 1e-3f, 1e-4f, "row_sum");
    }
    {
        auto shader = device.compile(make_residual_add_kernel());
        luisa::vector<float> y(x.size());
        lcg.fill(y, -1.f, 1.f);
        luisa::vector<float> ref;
        expect(ref_residual_add(x, y, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto by = ocr::test::upload(device, stream, luisa::span<const float>{y});
        auto plan = plan_residual_add_dispatch(x.size());
        // In-place: out aliases a (legal elementwise).
        OCR_DISPATCH_1D(stream, shader(bx, by, bx, static_cast<uint32_t>(x.size())), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bx, got, x.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, 1e-5f, 1e-5f, "residual in-place");
    }
}

void test_residual_rms_norm_fusion(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    constexpr uint32_t rows = 7u, h = 33u;
    luisa::vector<float> projection(static_cast<size_t>(rows) * h), residual(projection.size()), gamma(h);
    lcg.fill(projection, -2.f, 2.f);
    lcg.fill(residual, -1.f, 1.f);
    lcg.fill(gamma, 0.5f, 1.5f);
    luisa::vector<float> residual_ref, norm_ref;
    expect(ref_residual_add(projection, residual, residual_ref).ok());
    expect(ref_rms_norm(residual_ref, gamma, rows, h, 1e-6f, norm_ref).ok());
    auto bp = ocr::test::upload(device, stream, luisa::span<const float>{projection});
    auto br = ocr::test::upload(device, stream, luisa::span<const float>{residual});
    auto bg = ocr::test::upload(device, stream, luisa::span<const float>{gamma});
    auto bn = device.create_buffer<float>(projection.size());
    auto shader = device.compile(make_residual_rms_norm_kernel());
    expect(shader.compile_ok());
    auto plan = plan_norm_dispatch(rows);
    // residual_out aliases residual, matching the decoder's in-place stream.
    OCR_DISPATCH_1D(stream, shader(bp, br, bg, br, bn, rows, h, 1e-6f), *plan.value);
    luisa::vector<float> got_residual, got_norm;
    ocr::test::download(stream, br, got_residual, projection.size());
    ocr::test::download(stream, bn, got_norm, projection.size());
    stream << synchronize();
    ocr::test::expect_close(got_residual, residual_ref, 1e-5f, 1e-5f, "fused residual");
    ocr::test::expect_close(got_norm, norm_ref, kAtol, kRtol, "fused residual rms norm");
}

void test_decode_ops(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    // Embedding lookup with ragged token count.
    constexpr uint32_t vocab = 37u, hidden = 16u, num_ids = 9u;
    luisa::vector<float> table(static_cast<size_t>(vocab) * hidden);
    lcg.fill(table, -1.f, 1.f);
    luisa::vector<int32_t> ids(num_ids);
    for (uint32_t i = 0; i < num_ids; i++) { ids[i] = static_cast<int32_t>(lcg.next() % vocab); }
    {
        auto shader = device.compile(make_embedding_lookup_kernel());
        luisa::vector<float> ref;
        expect(ref_embedding_lookup(table, ids, vocab, hidden, ref).ok());
        auto bt = ocr::test::upload(device, stream, luisa::span<const float>{table});
        auto bi = ocr::test::upload(device, stream, luisa::span<const int32_t>{ids});
        auto bo = device.create_buffer<float>(static_cast<size_t>(num_ids) * hidden);
        auto plan = plan_embedding_dispatch(num_ids, hidden);
        OCR_DISPATCH_1D(stream, shader(bt, bi, bo, num_ids, hidden), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, static_cast<size_t>(num_ids) * hidden);
        stream << synchronize();
        ocr::test::expect_close(got, ref, 0.f, 0.f, "embedding lookup exact");
    }
    // Multimodal scatter.
    constexpr uint32_t seq = 21u, start = 5u, count = 7u;
    luisa::vector<float> emb(static_cast<size_t>(seq) * hidden), feat(static_cast<size_t>(count) * hidden);
    lcg.fill(emb, -1.f, 1.f);
    lcg.fill(feat, 10.f, 11.f);
    {
        auto shader = device.compile(make_mm_scatter_kernel());
        luisa::vector<float> ref;
        expect(ref_mm_scatter(emb, feat, seq, hidden, start, ref).ok());
        auto be = ocr::test::upload(device, stream, luisa::span<const float>{emb});
        auto bf = ocr::test::upload(device, stream, luisa::span<const float>{feat});
        auto bo = device.create_buffer<float>(emb.size());
        auto plan = plan_mm_scatter_dispatch(seq, hidden);
        OCR_DISPATCH_1D(stream, shader(be, bf, bo, seq, hidden, start, count), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, emb.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, 0.f, 0.f, "mm scatter exact");
    }
    // Argmax with deliberate ties (lowest index wins) + ragged rows.
    constexpr uint32_t rows = 5u, cols = 33u;
    luisa::vector<float> logits(static_cast<size_t>(rows) * cols);
    lcg.fill(logits, -1.f, 1.f);
    logits[cols + 3] = 5.f;
    logits[cols + 9] = 5.f;// tie in row 1 -> index 3
    {
        auto shader = device.compile(make_argmax_kernel());
        luisa::vector<int32_t> ref;
        expect(ref_argmax(logits, rows, cols, ref).ok());
        auto bl = ocr::test::upload(device, stream, luisa::span<const float>{logits});
        auto bo = device.create_buffer<int32_t>(rows);
        auto plan = plan_argmax_dispatch(rows);
        OCR_DISPATCH_1D(stream, shader(bl, bo, rows, cols), *plan.value);
        luisa::vector<int32_t> got;
        ocr::test::download(stream, bo, got, rows);
        stream << synchronize();
        bool ok = got == ref;
        expect(ok) << "argmax deterministic ties";
        expect(got[1] == 3);
    }
    // Temperature softmax rows sum to 1.
    {
        auto shader = device.compile(make_softmax_kernel());
        luisa::vector<float> ref;
        expect(ref_softmax(logits, rows, cols, 0.7f, ref).ok());
        auto bl = ocr::test::upload(device, stream, luisa::span<const float>{logits});
        auto bo = device.create_buffer<float>(logits.size());
        auto plan = plan_softmax_dispatch(rows);
        OCR_DISPATCH_1D(stream, shader(bl, bo, rows, cols, 0.7f), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, logits.size());
        stream << synchronize();
        ocr::test::expect_close(got, ref, 1e-5f, 1e-4f, "softmax");
    }
}

// M1: flattened 1D dispatch regression. 65536*256 + 3 elements force a
// >65535-group plan; EVERY element must be written (a total_x-only dispatch
// would leave all y-rows untouched). ~64 MiB per buffer, bounded.
void test_flattened_residual_coverage(Device &device) {
    auto stream = device.create_stream();
    const uint64_t count = 65536ull * 256ull + 3ull;// 16,777,219
    auto plan = plan_residual_add_dispatch(count);
    expect(plan.ok());
    expect(plan.value->is_2d()) << "count must force flattening";
    expect(plan.value->groups_x <= kMaxGroupsPerDim);
    expect(plan.value->groups_y <= kMaxGroupsPerDim);
    // a[i] = 1, b[i] = 2 -> out[i] = 3 everywhere.
    luisa::vector<float> host(static_cast<size_t>(count), 1.f);
    auto ba = ocr::test::upload(device, stream, luisa::span<const float>{host});
    for (auto &v : host) { v = 2.f; }
    auto bb = ocr::test::upload(device, stream, luisa::span<const float>{host});
    auto bo = device.create_buffer<float>(static_cast<size_t>(count));
    // Poison the output so an uncovered element cannot pass.
    for (auto &v : host) { v = -1.f; }
    stream << bo.copy_from(luisa::span<const float>{host});
    auto shader = device.compile(make_residual_add_kernel());
    OCR_DISPATCH_1D(stream, shader(ba, bb, bo, static_cast<uint32_t>(count)), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, static_cast<size_t>(count));
    stream << synchronize();
    uint64_t bad = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (got[static_cast<size_t>(i)] != 3.f) { bad++; }
    }
    expect(bad == 0u) << "every element of the flattened launch must be covered";
}

// S5: temperature <= 0 / non-finite -> host invalid_argument; device writes
// defined zeros (never NaN) when validation is skipped.
void test_softmax_invalid_temperature(Device &device) {
    expect(!validate_softmax_inputs(2, 4, 0.f).ok());
    expect(!validate_softmax_inputs(2, 4, -1.f).ok());
    expect(!validate_softmax_inputs(2, 4, std::numeric_limits<float>::infinity()).ok());
    expect(!validate_softmax_inputs(2, 4, std::numeric_limits<float>::quiet_NaN()).ok());
    expect(validate_softmax_inputs(2, 4, 1.f).ok());
    expect(!validate_softmax_inputs(0, 4, 1.f).ok());
    auto stream = device.create_stream();
    constexpr uint32_t rows = 2u, cols = 4u;
    luisa::vector<float> logits(static_cast<size_t>(rows) * cols, 1.f);
    auto shader = device.compile(make_softmax_kernel());
    auto bl = ocr::test::upload(device, stream, luisa::span<const float>{logits});
    auto bo = device.create_buffer<float>(logits.size());
    auto plan = plan_softmax_dispatch(rows);
    OCR_DISPATCH_1D(stream, shader(bl, bo, rows, cols, 0.f), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, logits.size());
    stream << synchronize();
    bool all_zero_finite = true;
    for (auto v : got) { all_zero_finite = all_zero_finite && v == 0.f; }
    expect(all_zero_finite) << "softmax temperature 0 must write defined zeros";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "layer_norm_variants"_test = [&] { test_layer_norm(device); };
    "rms_norm_qk_norm"_test = [&] { test_rms_norm_qk(device); };
    "l2_rowsum_residual"_test = [&] { test_l2_rowsum_residual(device); };
    "residual_rms_norm_fusion"_test = [&] { test_residual_rms_norm_fusion(device); };
    "decode_embedding_scatter_argmax_softmax"_test = [&] { test_decode_ops(device); };
    "softmax_invalid_temperature"_test = [&] { test_softmax_invalid_temperature(device); };
    "flattened_residual_full_coverage"_test = [&] { test_flattened_residual_coverage(device); };
}