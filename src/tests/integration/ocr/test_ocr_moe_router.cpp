// Test for ocr/ops/ops_moe router (Phase 10, device).
// This test covers:
// - FP32 router softmax vs the independent double-accumulation oracle
// - deterministic top-k selection with lowest-index tie-break + normalized weights
// - ties / all-zero / negative / extreme logits, top-k normalization
// - router projection (GEMM) + softmax + top-k composed vs ref_moe_route
// Synthetic only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/ops_moe.h"
#include "ocr/ops/ops_gemm.h"

#include "ocr/runtime/dispatch_adapter.h"
#include "ocr/runtime/shader_cache.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>


using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 2e-5f;
constexpr float kRtol = 2e-5f;

// Run the device softmax over logits[tokens, experts] and download probs.
luisa::vector<float> device_softmax(Device &device, luisa::span<const float> logits,
                                    uint32_t tokens, uint32_t experts) {
    auto stream = device.create_stream();
    auto logits_buf = ocr::test::upload(device, stream, luisa::span<const float>{logits});
    auto probs_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * experts);
    ShaderKeyRegistry reg;
    TypedShaderCache<MoeRouterSoftmaxKernel> cache{device, reg};
    auto sh = cache.get_or_compile("softmax", [] { return make_moe_router_softmax_kernel(); });
    LUISA_ASSERT(sh.ok(), "compile softmax");
    auto plan = plan_moe_router_dispatch(tokens);
    LUISA_ASSERT(plan.ok(), "plan softmax");
    stream << logits_buf.view().copy_from(luisa::span<const float>{logits});
    if (auto cmd = dispatch_1d((**sh.value)(logits_buf.view(), probs_buf.view(), tokens, experts), *plan.value)) {
        stream << std::move(cmd);
    }
    luisa::vector<float> out;
    ocr::test::download(stream, probs_buf, out, static_cast<size_t>(tokens) * experts);
    stream << synchronize();
    return out;
}

// Run the device top-k over probs and download ids + weights.
void device_topk(Device &device, luisa::span<const float> probs, uint32_t tokens, uint32_t experts,
                 uint32_t top_k, luisa::vector<uint32_t> &ids, luisa::vector<float> &weights) {
    auto stream = device.create_stream();
    auto probs_buf = ocr::test::upload(device, stream, luisa::span<const float>{probs});
    auto ids_buf = device.create_buffer<uint32_t>(static_cast<size_t>(tokens) * top_k);
    auto w_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * top_k);
    ShaderKeyRegistry reg;
    TypedShaderCache<MoeTopKKernel> cache{device, reg};
    auto sh = cache.get_or_compile("topk", [] { return make_moe_topk_kernel(); });
    LUISA_ASSERT(sh.ok(), "compile topk");
    auto plan = plan_moe_topk_dispatch(tokens);
    LUISA_ASSERT(plan.ok(), "plan topk");
    stream << probs_buf.view().copy_from(luisa::span<const float>{probs});
    if (auto cmd = dispatch_1d((**sh.value)(probs_buf.view(), ids_buf.view(), w_buf.view(), tokens, experts, top_k), *plan.value)) {
        stream << std::move(cmd);
    }
    ocr::test::download(stream, ids_buf, ids, static_cast<size_t>(tokens) * top_k);
    ocr::test::download(stream, w_buf, weights, static_cast<size_t>(tokens) * top_k);
    stream << synchronize();
}

void test_softmax(Device &device) {
    // Random logits, several token rows, 64 experts (the exact DeepSeek count).
    const uint32_t tokens = 5, experts = 64;
    ocr::test::Lcg lcg;
    lcg.state = 0x1234u;
    luisa::vector<float> logits(static_cast<size_t>(tokens) * experts);
    lcg.fill(logits, -6.0f, 6.0f);
    auto got = device_softmax(device, logits, tokens, experts);
    luisa::vector<float> ref;
    LUISA_ASSERT(ref_moe_softmax(logits, tokens, experts, ref).ok(), "ref softmax");
    ocr::test::expect_close(got, ref, kAtol, kRtol, "moe router softmax");
    // Each row sums to 1.
    for (uint32_t t = 0; t < tokens; t++) {
        double sum = 0.0;
        for (uint32_t e = 0; e < experts; e++) { sum += got[static_cast<size_t>(t) * experts + e]; }
        expect(std::abs(sum - 1.0) < 1e-4) << "softmax row sums to 1";
    }
}

void test_topk_ties_and_norm(Device &device) {
    // All-equal probabilities => lowest-index tie-break: ids 0..top_k-1, equal weights.
    const uint32_t tokens = 3, experts = 64, top_k = 6;
    luisa::vector<float> probs(static_cast<size_t>(tokens) * experts, 1.0f / experts);
    luisa::vector<uint32_t> ids;
    luisa::vector<float> weights;
    device_topk(device, probs, tokens, experts, top_k, ids, weights);
    luisa::vector<uint32_t> ref_ids;
    luisa::vector<float> ref_w;
    LUISA_ASSERT(ref_moe_topk(probs, tokens, experts, top_k, ref_ids, ref_w).ok(), "ref topk");
    expect(ids == ref_ids) << "topk ids match oracle (lowest-index ties)";
    ocr::test::expect_close(weights, ref_w, kAtol, kRtol, "topk normalized weights");
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t j = 0; j < top_k; j++) {
            expect(eq(ids[static_cast<size_t>(t) * top_k + j], j)) << "tie selects lowest index";
        }
        double sum = 0.0;
        for (uint32_t j = 0; j < top_k; j++) { sum += weights[static_cast<size_t>(t) * top_k + j]; }
        expect(std::abs(sum - 1.0) < 1e-4) << "top-k weights normalized to 1";
    }
}

void test_topk_distinct(Device &device) {
    // Distinct probabilities: top-k picks the largest, normalized.
    const uint32_t tokens = 4, experts = 64, top_k = 6;
    ocr::test::Lcg lcg;
    lcg.state = 0x99u;
    luisa::vector<float> probs(static_cast<size_t>(tokens) * experts);
    for (uint32_t t = 0; t < tokens; t++) {
        // Make a clear ranking with a few near-ties.
        for (uint32_t e = 0; e < experts; e++) {
            probs[static_cast<size_t>(t) * experts + e] = lcg.uniform(0.0f, 1.0f);
        }
    }
    luisa::vector<uint32_t> ids;
    luisa::vector<float> weights;
    device_topk(device, probs, tokens, experts, top_k, ids, weights);
    luisa::vector<uint32_t> ref_ids;
    luisa::vector<float> ref_w;
    LUISA_ASSERT(ref_moe_topk(probs, tokens, experts, top_k, ref_ids, ref_w).ok(), "ref topk");
    expect(ids == ref_ids) << "distinct topk ids match oracle";
    ocr::test::expect_close(weights, ref_w, kAtol, kRtol, "distinct topk weights");
}

void test_extreme_logits(Device &device) {
    // Extreme / negative / zero logits through softmax then top-k. The softmax
    // must stay finite (max-subtracted) and top-k must remain deterministic.
    const uint32_t experts = 64, top_k = 6;
    const uint32_t tokens = 4;
    luisa::vector<float> logits(static_cast<size_t>(tokens) * experts);
    // row 0: all zeros -> uniform probs.
    // row 1: large negative spread.
    for (uint32_t e = 0; e < experts; e++) { logits[static_cast<size_t>(1) * experts + e] = -1000.0f + static_cast<float>(e); }
    // row 2: large positive spread (max-subtraction must prevent overflow).
    for (uint32_t e = 0; e < experts; e++) { logits[static_cast<size_t>(2) * experts + e] = 1000.0f - static_cast<float>(e); }
    // row 3: a single dominant expert.
    for (uint32_t e = 0; e < experts; e++) { logits[static_cast<size_t>(3) * experts + e] = -5.0f; }
    logits[static_cast<size_t>(3) * experts + 40] = 50.0f;
    auto probs = device_softmax(device, logits, tokens, experts);
    luisa::vector<float> ref_probs;
    LUISA_ASSERT(ref_moe_softmax(logits, tokens, experts, ref_probs).ok(), "ref softmax extreme");
    ocr::test::expect_close(probs, ref_probs, kAtol, kRtol, "extreme softmax");
    for (auto p : probs) { expect(std::isfinite(p)) << "softmax finite for extreme logits"; }
    luisa::vector<uint32_t> ids;
    luisa::vector<float> weights;
    device_topk(device, probs, tokens, experts, top_k, ids, weights);
    luisa::vector<uint32_t> ref_ids;
    luisa::vector<float> ref_w;
    LUISA_ASSERT(ref_moe_topk(probs, tokens, experts, top_k, ref_ids, ref_w).ok(), "ref topk extreme");
    expect(ids == ref_ids) << "extreme topk ids match oracle";
    // row 3 dominant expert 40 must be selected first.
    expect(eq(ids[static_cast<size_t>(3) * top_k], 40u)) << "dominant expert selected";
}

void test_full_route(Device &device) {
    // Composed router projection (GEMM) + softmax + top-k + permutation vs the
    // independent ref_moe_route. Uses the device GEMM for the projection.
    const uint32_t tokens = 6, hidden = 16, experts = 64, top_k = 6;
    ocr::test::Lcg lcg;
    lcg.state = 0x4242u;
    luisa::vector<float> x(static_cast<size_t>(tokens) * hidden);
    luisa::vector<float> router_w(static_cast<size_t>(hidden) * experts);
    lcg.fill(x, -0.5f, 0.5f);
    lcg.fill(router_w, -0.4f, 0.4f);
    // Device projection.
    auto stream = device.create_stream();
    auto x_buf = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto w_buf = ocr::test::upload(device, stream, luisa::span<const float>{router_w});
    auto logits_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * experts);
    auto dummy = device.create_buffer<float>(1);
    ShaderKeyRegistry reg;
    TypedShaderCache<GemmKernel> gemm{device, reg};
    auto sh = gemm.get_or_compile("gemm", [] { return make_gemm_kernel(GemmEpilogue::none); });
    LUISA_ASSERT(sh.ok(), "compile gemm");
    auto plan = plan_gemm_dispatch(tokens, experts, hidden);
    LUISA_ASSERT(plan.ok(), "plan gemm");
    stream << x_buf.view().copy_from(luisa::span<const float>{x}) << w_buf.view().copy_from(luisa::span<const float>{router_w});
    if (auto cmd = dispatch_2d((**sh.value)(x_buf.view(), w_buf.view(), logits_buf.view(), dummy.view(),
                                            tokens, experts, hidden, plan.value->padded_cols), *plan.value)) {
        stream << std::move(cmd);
    }
    luisa::vector<float> logits;
    ocr::test::download(stream, logits_buf, logits, static_cast<size_t>(tokens) * experts);
    stream << synchronize();
    // Oracle route from the SAME x/router_w (independent GEMM).
    MoeRouterConfig rc{experts, top_k};
    MoeRouting ref;
    LUISA_ASSERT(ref_moe_route(x, router_w, tokens, hidden, rc, ref).ok(), "ref route");
    ocr::test::expect_close(logits, ref.logits, 1e-4f, 1e-4f, "router projection logits");
    // Device softmax + topk on the device-projected logits.
    auto probs = device_softmax(device, logits, tokens, experts);
    luisa::vector<uint32_t> ids;
    luisa::vector<float> weights;
    device_topk(device, probs, tokens, experts, top_k, ids, weights);
    // The oracle's logits differ from device logits by fp32 rounding; recompute
    // the oracle selection from the DEVICE logits so ties are compared exactly.
    luisa::vector<float> probs_from_dev;
    LUISA_ASSERT(ref_moe_softmax(logits, tokens, experts, probs_from_dev).ok(), "ref softmax dev");
    luisa::vector<uint32_t> ref_ids;
    luisa::vector<float> ref_w;
    LUISA_ASSERT(ref_moe_topk(probs_from_dev, tokens, experts, top_k, ref_ids, ref_w).ok(), "ref topk dev");
    expect(ids == ref_ids) << "composed route ids match oracle";
    ocr::test::expect_close(weights, ref_w, kAtol, kRtol, "composed route weights");
}


// M9: a FINITE extreme row whose values are all below the old -3.0e38 sentinel
// used to make every exp() underflow, so the sum was 0 and the division produced
// NaN. With a true element-seeded max the row stays a valid distribution.
void test_extreme_finite_no_nan(Device &device) {
    const uint32_t experts = 64, top_k = 6;
    // Row 0: all values below the old sentinel (-3.0e38), still finite FP32.
    // Row 1: mixed very-negative magnitudes. Row 2: near the FP32 finite limit.
    const uint32_t tokens = 3;
    luisa::vector<float> logits(static_cast<size_t>(tokens) * experts);
    for (uint32_t e = 0; e < experts; e++) {
        logits[static_cast<size_t>(0) * experts + e] = -3.2e38f;
        logits[static_cast<size_t>(1) * experts + e] = -3.3e38f + static_cast<float>(e) * 1.0e36f;
        logits[static_cast<size_t>(2) * experts + e] = -3.4028234e38f;// ~ -FLT_MAX
    }
    for (auto v : logits) { expect(std::isfinite(v)) << "test logits are finite"; }
    auto probs = device_softmax(device, logits, tokens, experts);
    for (size_t i = 0; i < probs.size(); i++) {
        expect(std::isfinite(probs[i])) << "M9: extreme finite row must not produce NaN";
    }
    // Each row is a valid distribution (uniform for the constant rows).
    for (uint32_t t = 0; t < tokens; t++) {
        double sum = 0.0;
        for (uint32_t e = 0; e < experts; e++) { sum += probs[static_cast<size_t>(t) * experts + e]; }
        expect(std::abs(sum - 1.0) < 1e-4) << "extreme finite row sums to 1";
    }
    luisa::vector<float> ref;
    LUISA_ASSERT(ref_moe_softmax(logits, tokens, experts, ref).ok(), "ref softmax extreme finite");
    ocr::test::expect_close(probs, ref, kAtol, kRtol, "M9 extreme finite softmax vs oracle");
    // top-k stays deterministic on the degenerate uniform rows.
    luisa::vector<uint32_t> ids;
    luisa::vector<float> weights;
    device_topk(device, probs, tokens, experts, top_k, ids, weights);
    for (uint32_t j = 0; j < top_k; j++) {
        expect(eq(ids[static_cast<size_t>(0) * top_k + j], j)) << "uniform extreme row: lowest-index ties";
    }
    // A non-finite row is still defensively zeroed (device AND oracle agree).
    luisa::vector<float> nan_row(static_cast<size_t>(experts), 1.0f);
    nan_row[3] = std::numeric_limits<float>::infinity();
    auto np = device_softmax(device, nan_row, 1, experts);
    for (auto v : np) { expect(v == 0.0f) << "non-finite row writes zeros"; }
    luisa::vector<float> nref;
    LUISA_ASSERT(ref_moe_softmax(nan_row, 1, experts, nref).ok(), "ref non-finite");
    ocr::test::expect_close(np, nref, 0.0f, 0.0f, "non-finite row zeros match oracle");
}

// S11: the composed route must compare EXACT device-vs-host ids derived from the
// SAME x / router weight, using the host's own double-accumulating projection --
// not ids recomputed from the device probabilities.
void test_composed_route_exact_ids(Device &device) {
    const uint32_t tokens = 6, hidden = 16, experts = 64, top_k = 6;
    ocr::test::Lcg lcg;
    lcg.state = 0x5151u;
    luisa::vector<float> x(static_cast<size_t>(tokens) * hidden);
    luisa::vector<float> router_w(static_cast<size_t>(hidden) * experts);
    lcg.fill(x, -0.5f, 0.5f);
    lcg.fill(router_w, -0.4f, 0.4f);
    // Device: projection -> softmax -> top-k.
    auto stream = device.create_stream();
    auto x_buf = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto w_buf = ocr::test::upload(device, stream, luisa::span<const float>{router_w});
    auto logits_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * experts);
    auto dummy = device.create_buffer<float>(1);
    ShaderKeyRegistry reg;
    TypedShaderCache<GemmKernel> gemm{device, reg};
    auto sh = gemm.get_or_compile("gemm", [] { return make_gemm_kernel(GemmEpilogue::none); });
    LUISA_ASSERT(sh.ok(), "compile gemm");
    auto plan = plan_gemm_dispatch(tokens, experts, hidden);
    LUISA_ASSERT(plan.ok(), "plan gemm");
    stream << x_buf.view().copy_from(luisa::span<const float>{x})
           << w_buf.view().copy_from(luisa::span<const float>{router_w});
    if (auto cmd = dispatch_2d((**sh.value)(x_buf.view(), w_buf.view(), logits_buf.view(), dummy.view(),
                                            tokens, experts, hidden, plan.value->padded_cols), *plan.value)) {
        stream << std::move(cmd);
    }
    luisa::vector<float> dev_logits;
    ocr::test::download(stream, logits_buf, dev_logits, static_cast<size_t>(tokens) * experts);
    stream << synchronize();
    auto probs = device_softmax(device, dev_logits, tokens, experts);
    luisa::vector<uint32_t> dev_ids;
    luisa::vector<float> dev_w;
    device_topk(device, probs, tokens, experts, top_k, dev_ids, dev_w);
    // Host: the INDEPENDENT full route from the same x / router_w.
    MoeRouterConfig rc{experts, top_k};
    MoeRouting host;
    LUISA_ASSERT(ref_moe_route(x, router_w, tokens, hidden, rc, host).ok(), "host route");
    ocr::test::expect_close(dev_logits, host.logits, 1e-4f, 1e-4f, "S11 router logits vs independent host projection");
    expect(dev_ids == host.ids) << "S11 EXACT device-vs-host expert ids from the same x/W";
    ocr::test::expect_close(dev_w, host.weights, kAtol, kRtol, "S11 selected weights vs host route");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "moe_router_softmax"_test = [&] { test_softmax(device); };
    "moe_router_topk_ties_and_norm"_test = [&] { test_topk_ties_and_norm(device); };
    "moe_router_topk_distinct"_test = [&] { test_topk_distinct(device); };
    "moe_router_extreme_logits"_test = [&] { test_extreme_logits(device); };
    "moe_router_full_route"_test = [&] { test_full_route(device); };
    "moe_router_extreme_finite_no_nan"_test = [&] { test_extreme_finite_no_nan(device); };
    "moe_router_composed_exact_ids"_test = [&] { test_composed_route_exact_ids(device); };
}
