// Test for ocr/ops/ops_moe expert math (Phase 10, device).
// This test covers:
// - the full reference MoE layer: 64 routed experts / top-6 / two shared experts
// - gate/up projections, SwiGLU, down projection, weighted restore + shared
// - every shared/routed contribution vs the independent double-accum oracle
// - empty expert assignments (inactive experts skipped)
// Synthetic only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/ops_moe.h"
#include "ocr/ops/ops_gemm.h"

#include "ocr/runtime/dispatch_adapter.h"
#include "ocr/runtime/shader_cache.h"

#include <cstdint>

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

constexpr float kAtol = 2e-4f;
constexpr float kRtol = 2e-4f;

// Resize + uniform fill (Lcg::fill does not resize).
void fill_vec(ocr::test::Lcg &lcg, luisa::vector<float> &v, size_t n, float lo, float hi) {
    v.resize(n);
    for (auto &x : v) { x = lcg.uniform(lo, hi); }
}


// Device MoE layer driver: mirrors the executor's active-expert loop but over
// caller-supplied weights, so the expert math is tested in isolation. Records
// router->permutation->gather, reads counts, runs per-active-expert FFN + shared
// + scatter. Returns the unweighted expert_out, shared0/1, and final out.
struct MoeLayerDevice {
    Device *device;
    ShaderKeyRegistry reg;
    TypedShaderCache<MoeRouterSoftmaxKernel> softmax;
    TypedShaderCache<MoeTopKKernel> topk;
    TypedShaderCache<MoeHistogramKernel> hist;
    TypedShaderCache<MoeOffsetsKernel> offs;
    TypedShaderCache<MoePermutationKernel> perm;
    TypedShaderCache<MoeGatherKernel> gather;
    TypedShaderCache<MoeScatterKernel> scatter;
    TypedShaderCache<GemmKernel> gemm;
    TypedShaderCache<GemvKernel> gemv;
    TypedShaderCache<GatedActKernel> gated;
    const TypedShaderCache<MoeRouterSoftmaxKernel>::ShaderT *sh_softmax{nullptr};
    const TypedShaderCache<MoeTopKKernel>::ShaderT *sh_topk{nullptr};
    const TypedShaderCache<MoeHistogramKernel>::ShaderT *sh_hist{nullptr};
    const TypedShaderCache<MoeOffsetsKernel>::ShaderT *sh_offs{nullptr};
    const TypedShaderCache<MoePermutationKernel>::ShaderT *sh_perm{nullptr};
    const TypedShaderCache<MoeGatherKernel>::ShaderT *sh_gather{nullptr};
    const TypedShaderCache<MoeScatterKernel>::ShaderT *sh_scatter{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *sh_gemm{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *sh_gemv{nullptr};
    const TypedShaderCache<GatedActKernel>::ShaderT *sh_gated{nullptr};
    void init(Device &d) {
        device = &d;
        softmax = TypedShaderCache<MoeRouterSoftmaxKernel>{d, reg};
        topk = TypedShaderCache<MoeTopKKernel>{d, reg};
        hist = TypedShaderCache<MoeHistogramKernel>{d, reg};
        offs = TypedShaderCache<MoeOffsetsKernel>{d, reg};
        perm = TypedShaderCache<MoePermutationKernel>{d, reg};
        gather = TypedShaderCache<MoeGatherKernel>{d, reg};
        scatter = TypedShaderCache<MoeScatterKernel>{d, reg};
        gemm = TypedShaderCache<GemmKernel>{d, reg};
        gemv = TypedShaderCache<GemvKernel>{d, reg};
        gated = TypedShaderCache<GatedActKernel>{d, reg};
        auto a = softmax.get_or_compile("sm", [] { return make_moe_router_softmax_kernel(); });
        auto b = topk.get_or_compile("tk", [] { return make_moe_topk_kernel(); });
        auto c = hist.get_or_compile("hi", [] { return make_moe_histogram_kernel(); });
        auto e = offs.get_or_compile("of", [] { return make_moe_offsets_kernel(); });
        auto f = perm.get_or_compile("pe", [] { return make_moe_permutation_kernel(); });
        auto g = gather.get_or_compile("ga", [] { return make_moe_gather_kernel(); });
        auto h = scatter.get_or_compile("sc", [] { return make_moe_scatter_kernel(); });
        auto i = gemm.get_or_compile("gm", [] { return make_gemm_kernel(GemmEpilogue::none); });
        auto j = gemv.get_or_compile("gv", [] { return make_gemv_kernel(GemmEpilogue::none); });
        auto k = gated.get_or_compile("gt", [] { return make_gated_act_kernel(0u); });
        LUISA_ASSERT(a.ok() && b.ok() && c.ok() && e.ok() && f.ok() && g.ok() && h.ok() && i.ok() && j.ok() && k.ok(), "compile moe layer");
        sh_softmax = *a.value; sh_topk = *b.value; sh_hist = *c.value; sh_offs = *e.value;
        sh_perm = *f.value; sh_gather = *g.value; sh_scatter = *h.value; sh_gemm = *i.value;
        sh_gemv = *j.value; sh_gated = *k.value;
    }
    Status linear(CommandList &list, BufferView<float> A, BufferView<float> B, BufferView<float> C,
                  uint32_t m, uint32_t n, uint32_t k) {
        if (m <= kGemvSmallM) {
            auto plan = plan_gemv_dispatch(m, n, k);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*sh_gemv)(A, B, C, C, m, n, k), *plan.value)) { list << std::move(cmd); }
            return Status::success();
        }
        auto plan = plan_gemm_dispatch(m, n, k);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_2d((*sh_gemm)(A, B, C, C, m, n, k, plan.value->padded_cols), *plan.value)) { list << std::move(cmd); }
        return Status::success();
    }
};

// Upload a weight vector and return its buffer (kept alive by the caller).
struct DevBuf {
    compute::Buffer<float> buf;
    [[nodiscard]] compute::BufferView<float> view() const { return buf.view(); }
};

void test_full_moe_layer(MoeLayerDevice &md) {
    const uint32_t tokens = 5, hidden = 8, inter = 6, shared_inter = 6;
    const uint32_t experts = 64, top_k = 6;
    const uint32_t slots = tokens * top_k;
    ocr::test::Lcg lcg;
    lcg.state = 0xBEEFu;
    luisa::vector<float> x(static_cast<size_t>(tokens) * hidden);
    luisa::vector<float> router_w(static_cast<size_t>(hidden) * experts);
    lcg.fill(x, -0.5f, 0.5f);
    lcg.fill(router_w, -0.4f, 0.4f);
    // Routed expert weights.
    luisa::vector<MoeExpertWeights> routed(experts);
    for (auto &e : routed) {
        fill_vec(lcg, e.gate_w, static_cast<size_t>(hidden) * inter, -0.3f, 0.3f);
        fill_vec(lcg, e.up_w, static_cast<size_t>(hidden) * inter, -0.3f, 0.3f);
        fill_vec(lcg, e.down_w, static_cast<size_t>(inter) * hidden, -0.3f, 0.3f);
    }
    MoeExpertWeights shared0, shared1;
    fill_vec(lcg, shared0.gate_w, static_cast<size_t>(hidden) * shared_inter, -0.3f, 0.3f);
    fill_vec(lcg, shared0.up_w, static_cast<size_t>(hidden) * shared_inter, -0.3f, 0.3f);
    fill_vec(lcg, shared0.down_w, static_cast<size_t>(shared_inter) * hidden, -0.3f, 0.3f);
    fill_vec(lcg, shared1.gate_w, static_cast<size_t>(hidden) * shared_inter, -0.3f, 0.3f);
    fill_vec(lcg, shared1.up_w, static_cast<size_t>(hidden) * shared_inter, -0.3f, 0.3f);
    fill_vec(lcg, shared1.down_w, static_cast<size_t>(shared_inter) * hidden, -0.3f, 0.3f);
    // Independent oracle.
    MoeRouterConfig rc{experts, top_k};
    MoeRouting routing;
    LUISA_ASSERT(ref_moe_route(x, router_w, tokens, hidden, rc, routing).ok(), "ref route");
    luisa::vector<float> ref_expert_out, ref_weighted, ref_s0, ref_s1, ref_out;
    LUISA_ASSERT(ref_moe_layer(x, routing, routed, shared0, shared1, tokens, hidden, inter, shared_inter,
                               ref_expert_out, ref_weighted, ref_s0, ref_s1, ref_out).ok(), "ref moe layer");

    auto stream = md.device->create_stream();
    auto &device = *md.device;
    // Upload x, router_w, all expert weights.
    auto x_buf = ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto rw_buf = ocr::test::upload(device, stream, luisa::span<const float>{router_w});
    luisa::vector<DevBuf> gate_bufs(experts), up_bufs(experts), down_bufs(experts);
    for (uint32_t e = 0; e < experts; e++) {
        gate_bufs[e].buf = ocr::test::upload(device, stream, luisa::span<const float>{routed[e].gate_w});
        up_bufs[e].buf = ocr::test::upload(device, stream, luisa::span<const float>{routed[e].up_w});
        down_bufs[e].buf = ocr::test::upload(device, stream, luisa::span<const float>{routed[e].down_w});
    }
    auto s0g = ocr::test::upload(device, stream, luisa::span<const float>{shared0.gate_w});
    auto s0u = ocr::test::upload(device, stream, luisa::span<const float>{shared0.up_w});
    auto s0d = ocr::test::upload(device, stream, luisa::span<const float>{shared0.down_w});
    auto s1g = ocr::test::upload(device, stream, luisa::span<const float>{shared1.gate_w});
    auto s1u = ocr::test::upload(device, stream, luisa::span<const float>{shared1.up_w});
    auto s1d = ocr::test::upload(device, stream, luisa::span<const float>{shared1.down_w});
    // Scratch.
    auto logits_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * experts);
    auto probs_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * experts);
    auto ids_buf = device.create_buffer<uint32_t>(slots);
    auto weights_buf = device.create_buffer<float>(slots);
    auto counts_buf = device.create_buffer<uint32_t>(experts);
    auto offs_buf = device.create_buffer<uint32_t>(experts);
    auto ps_buf = device.create_buffer<uint32_t>(slots);
    auto sp_buf = device.create_buffer<uint32_t>(slots);
    auto pt_buf = device.create_buffer<uint32_t>(slots);
    auto gathered_buf = device.create_buffer<float>(static_cast<size_t>(slots) * hidden);
    auto gate_buf = device.create_buffer<float>(static_cast<size_t>(slots) * inter);
    auto up_buf = device.create_buffer<float>(static_cast<size_t>(slots) * inter);
    auto act_buf = device.create_buffer<float>(static_cast<size_t>(slots) * inter);
    auto expert_out_buf = device.create_buffer<float>(static_cast<size_t>(slots) * hidden);
    auto shared0_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * hidden);
    auto shared1_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * hidden);
    auto out_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * hidden);
    // Upload x/router (the upload helper already copied; re-copy to be safe).
    stream << x_buf.view().copy_from(luisa::span<const float>{x})
           << rw_buf.view().copy_from(luisa::span<const float>{router_w});
    auto list = CommandList::create();
    // Router projection.
    LUISA_ASSERT(md.linear(list, x_buf.view(), rw_buf.view(), logits_buf.view(), tokens, experts, hidden).ok(), "router gemm");
    const auto disp1 = [&](auto &&invoke, const Result<LaunchPlan1D> &plan) {
        LUISA_ASSERT(plan.ok(), "plan");
        if (auto c = dispatch_1d(std::forward<decltype(invoke)>(invoke), *plan.value)) { list << std::move(c); }
    };
    disp1((*md.sh_softmax)(logits_buf.view(), probs_buf.view(), tokens, experts), plan_moe_router_dispatch(tokens));
    disp1((*md.sh_topk)(probs_buf.view(), ids_buf.view(), weights_buf.view(), tokens, experts, top_k), plan_moe_topk_dispatch(tokens));
    disp1((*md.sh_hist)(ids_buf.view(), counts_buf.view(), slots, experts), plan_moe_histogram_dispatch(experts));
    disp1((*md.sh_offs)(counts_buf.view(), offs_buf.view(), experts), plan_moe_offsets_dispatch(experts));
    disp1((*md.sh_perm)(ids_buf.view(), offs_buf.view(), ps_buf.view(), sp_buf.view(), pt_buf.view(), slots, experts, top_k), plan_moe_permutation_dispatch(experts));
    disp1((*md.sh_gather)(x_buf.view(), pt_buf.view(), gathered_buf.view(), slots, hidden), plan_moe_gather_dispatch(slots, hidden));
    stream << list.commit() << synchronize();
    // Read counts/offsets on the host.
    luisa::vector<uint32_t> counts(experts), offsets(experts);
    stream << counts_buf.view().copy_to(luisa::span<uint32_t>{counts})
           << offs_buf.view().copy_to(luisa::span<uint32_t>{offsets}) << synchronize();
    expect(counts == routing.counts) << "device counts match oracle";
    // Per-active-expert FFN.
    list.clear();
    for (uint32_t e = 0; e < experts; e++) {
        const uint32_t count = counts[e];
        if (count == 0u) { continue; }
        const uint32_t off = offsets[e];
        auto g_rows = gathered_buf.view().subview(static_cast<size_t>(off) * hidden, static_cast<size_t>(count) * hidden);
        auto gate_e = gate_buf.view().subview(0, static_cast<size_t>(count) * inter);
        auto up_e = up_buf.view().subview(0, static_cast<size_t>(count) * inter);
        auto act_e = act_buf.view().subview(0, static_cast<size_t>(count) * inter);
        auto out_e = expert_out_buf.view().subview(static_cast<size_t>(off) * hidden, static_cast<size_t>(count) * hidden);
        LUISA_ASSERT(md.linear(list, g_rows, gate_bufs[e].view(), gate_e, count, inter, hidden).ok(), "expert gate");
        LUISA_ASSERT(md.linear(list, g_rows, up_bufs[e].view(), up_e, count, inter, hidden).ok(), "expert up");
        disp1((*md.sh_gated)(gate_e, up_e, act_e, count * inter), plan_gated_act_dispatch(static_cast<uint64_t>(count) * inter));
        LUISA_ASSERT(md.linear(list, act_e, down_bufs[e].view(), out_e, count, hidden, inter).ok(), "expert down");
    }
    // Shared experts (reuse gate/up/act scratch sized for slots*inter >= tokens*shared_inter).
    const auto shared_ffn = [&](BufferView<float> gw, BufferView<float> uw, BufferView<float> dw, BufferView<float> out) {
        auto gs = gate_buf.view().subview(0, static_cast<size_t>(tokens) * shared_inter);
        auto us = up_buf.view().subview(0, static_cast<size_t>(tokens) * shared_inter);
        auto as = act_buf.view().subview(0, static_cast<size_t>(tokens) * shared_inter);
        LUISA_ASSERT(md.linear(list, x_buf.view(), gw, gs, tokens, shared_inter, hidden).ok(), "shared gate");
        LUISA_ASSERT(md.linear(list, x_buf.view(), uw, us, tokens, shared_inter, hidden).ok(), "shared up");
        disp1((*md.sh_gated)(gs, us, as, tokens * shared_inter), plan_gated_act_dispatch(static_cast<uint64_t>(tokens) * shared_inter));
        LUISA_ASSERT(md.linear(list, as, dw, out, tokens, hidden, shared_inter).ok(), "shared down");
    };
    shared_ffn(s0g.view(), s0u.view(), s0d.view(), shared0_buf.view());
    shared_ffn(s1g.view(), s1u.view(), s1d.view(), shared1_buf.view());
    disp1((*md.sh_scatter)(expert_out_buf.view(), sp_buf.view(), weights_buf.view(),
                           shared0_buf.view(), shared1_buf.view(), out_buf.view(), tokens, hidden, top_k),
          plan_moe_scatter_dispatch(tokens, hidden));
    stream << list.commit() << synchronize();
    // Download + compare every contribution.
    luisa::vector<float> expert_out, s0, s1, out;
    ocr::test::download(stream, expert_out_buf, expert_out, static_cast<size_t>(slots) * hidden);
    ocr::test::download(stream, shared0_buf, s0, static_cast<size_t>(tokens) * hidden);
    ocr::test::download(stream, shared1_buf, s1, static_cast<size_t>(tokens) * hidden);
    ocr::test::download(stream, out_buf, out, static_cast<size_t>(tokens) * hidden);
    stream << synchronize();
    ocr::test::expect_close(expert_out, ref_expert_out, kAtol, kRtol, "moe routed expert_out (all 64/top6)");
    ocr::test::expect_close(s0, ref_s0, kAtol, kRtol, "moe shared expert 0");
    ocr::test::expect_close(s1, ref_s1, kAtol, kRtol, "moe shared expert 1");
    ocr::test::expect_close(out, ref_out, kAtol, kRtol, "moe final output (weighted restore + 2 shared)");
}

void test_empty_assignments(MoeLayerDevice &md) {
    // Force ALL tokens to a single expert (router weight column 0 dominant) so
    // 63 experts are inactive. The active-expert loop must skip them and the
    // result must still match the oracle.
    const uint32_t tokens = 3, hidden = 8, inter = 6, shared_inter = 6;
    const uint32_t experts = 64, top_k = 6;
    const uint32_t slots = tokens * top_k;
    luisa::vector<float> x(static_cast<size_t>(tokens) * hidden, 0.25f);
    // Router: expert 0 gets a huge positive weight on every hidden dim; others 0.
    luisa::vector<float> router_w(static_cast<size_t>(hidden) * experts, 0.0f);
    for (uint32_t h = 0; h < hidden; h++) { router_w[static_cast<size_t>(h) * experts + 0] = 10.0f; }
    ocr::test::Lcg lcg;
    lcg.state = 0x51u;
    luisa::vector<MoeExpertWeights> routed(experts);
    for (auto &e : routed) {
        fill_vec(lcg, e.gate_w, static_cast<size_t>(hidden) * inter, -0.3f, 0.3f);
        fill_vec(lcg, e.up_w, static_cast<size_t>(hidden) * inter, -0.3f, 0.3f);
        fill_vec(lcg, e.down_w, static_cast<size_t>(inter) * hidden, -0.3f, 0.3f);
    }
    MoeExpertWeights shared0, shared1;
    fill_vec(lcg, shared0.gate_w, static_cast<size_t>(hidden) * shared_inter, -0.3f, 0.3f);
    fill_vec(lcg, shared0.up_w, static_cast<size_t>(hidden) * shared_inter, -0.3f, 0.3f);
    fill_vec(lcg, shared0.down_w, static_cast<size_t>(shared_inter) * hidden, -0.3f, 0.3f);
    fill_vec(lcg, shared1.gate_w, static_cast<size_t>(hidden) * shared_inter, -0.3f, 0.3f);
    fill_vec(lcg, shared1.up_w, static_cast<size_t>(hidden) * shared_inter, -0.3f, 0.3f);
    fill_vec(lcg, shared1.down_w, static_cast<size_t>(shared_inter) * hidden, -0.3f, 0.3f);
    MoeRouterConfig rc{experts, top_k};
    MoeRouting routing;
    LUISA_ASSERT(ref_moe_route(x, router_w, tokens, hidden, rc, routing).ok(), "ref route empty");
    // All top-k slots select expert 0 (ties broken to lowest index after expert 0
    // dominates; the remaining 5 are the lowest-index zero-prob experts 1..5).
    luisa::vector<float> ref_expert_out, ref_weighted, ref_s0, ref_s1, ref_out;
    LUISA_ASSERT(ref_moe_layer(x, routing, routed, shared0, shared1, tokens, hidden, inter, shared_inter,
                               ref_expert_out, ref_weighted, ref_s0, ref_s1, ref_out).ok(), "ref moe empty");
    // Count active experts in the oracle routing.
    uint32_t active = 0;
    for (auto c : routing.counts) { if (c > 0u) { active++; } }
    expect(active <= top_k) << "only top-k experts can be active per token batch";
    // Device run (same driver as the full test, abbreviated): reuse the oracle
    // routing counts to drive the device expert loop, verifying inactive experts
    // are skipped and the gathered/scattered result matches.
    auto stream = md.device->create_stream();
    auto &device = *md.device;
    auto x_buf = ocr::test::upload(device, stream, luisa::span<const float>{x});
    luisa::vector<DevBuf> gate_bufs(experts), up_bufs(experts), down_bufs(experts);
    for (uint32_t e = 0; e < experts; e++) {
        gate_bufs[e].buf = ocr::test::upload(device, stream, luisa::span<const float>{routed[e].gate_w});
        up_bufs[e].buf = ocr::test::upload(device, stream, luisa::span<const float>{routed[e].up_w});
        down_bufs[e].buf = ocr::test::upload(device, stream, luisa::span<const float>{routed[e].down_w});
    }
    auto s0g = ocr::test::upload(device, stream, luisa::span<const float>{shared0.gate_w});
    auto s0u = ocr::test::upload(device, stream, luisa::span<const float>{shared0.up_w});
    auto s0d = ocr::test::upload(device, stream, luisa::span<const float>{shared0.down_w});
    auto s1g = ocr::test::upload(device, stream, luisa::span<const float>{shared1.gate_w});
    auto s1u = ocr::test::upload(device, stream, luisa::span<const float>{shared1.up_w});
    auto s1d = ocr::test::upload(device, stream, luisa::span<const float>{shared1.down_w});
    // S10: run the REAL device router -> top-k -> histogram -> offsets ->
    // permutation chain (no uploaded oracle routing), then drive the expert loop
    // from the DEVICE counts/offsets after validating the readback.
    auto rw_buf = ocr::test::upload(device, stream, luisa::span<const float>{router_w});
    auto logits_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * experts);
    auto probs_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * experts);
    auto ids_buf = device.create_buffer<uint32_t>(slots);
    auto w_buf = device.create_buffer<float>(slots);
    auto counts_buf = device.create_buffer<uint32_t>(experts);
    auto offs_buf = device.create_buffer<uint32_t>(experts);
    auto ps_buf = device.create_buffer<uint32_t>(slots);
    auto sp_buf = device.create_buffer<uint32_t>(slots);
    auto pt_buf = device.create_buffer<uint32_t>(slots);
    auto gathered_buf = device.create_buffer<float>(static_cast<size_t>(slots) * hidden);
    auto gate_buf = device.create_buffer<float>(static_cast<size_t>(slots) * inter);
    auto up_buf = device.create_buffer<float>(static_cast<size_t>(slots) * inter);
    auto act_buf = device.create_buffer<float>(static_cast<size_t>(slots) * inter);
    auto expert_out_buf = device.create_buffer<float>(static_cast<size_t>(slots) * hidden);
    auto shared0_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * hidden);
    auto shared1_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * hidden);
    auto out_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * hidden);
    stream << x_buf.view().copy_from(luisa::span<const float>{x})
           << rw_buf.view().copy_from(luisa::span<const float>{router_w});
    auto list = CommandList::create();
    const auto disp1 = [&](auto &&invoke, const Result<LaunchPlan1D> &plan) {
        LUISA_ASSERT(plan.ok(), "plan");
        if (auto c = dispatch_1d(std::forward<decltype(invoke)>(invoke), *plan.value)) { list << std::move(c); }
    };
    // Device routing chain.
    LUISA_ASSERT(md.linear(list, x_buf.view(), rw_buf.view(), logits_buf.view(), tokens, experts, hidden).ok(), "router gemm");
    disp1((*md.sh_softmax)(logits_buf.view(), probs_buf.view(), tokens, experts), plan_moe_router_dispatch(tokens));
    disp1((*md.sh_topk)(probs_buf.view(), ids_buf.view(), w_buf.view(), tokens, experts, top_k), plan_moe_topk_dispatch(tokens));
    disp1((*md.sh_hist)(ids_buf.view(), counts_buf.view(), slots, experts), plan_moe_histogram_dispatch(experts));
    disp1((*md.sh_offs)(counts_buf.view(), offs_buf.view(), experts), plan_moe_offsets_dispatch(experts));
    disp1((*md.sh_perm)(ids_buf.view(), offs_buf.view(), ps_buf.view(), sp_buf.view(), pt_buf.view(), slots, experts, top_k), plan_moe_permutation_dispatch(experts));
    disp1((*md.sh_gather)(x_buf.view(), pt_buf.view(), gathered_buf.view(), slots, hidden), plan_moe_gather_dispatch(slots, hidden));
    stream << list.commit() << synchronize();
    // Read back + VALIDATE the device routing before it drives any subview (S3).
    luisa::vector<uint32_t> dev_counts(experts), dev_offsets(experts), dev_ids(slots), dev_pt(slots), dev_sp(slots);
    luisa::vector<float> dev_w(slots);
    stream << counts_buf.view().copy_to(luisa::span<uint32_t>{dev_counts})
           << offs_buf.view().copy_to(luisa::span<uint32_t>{dev_offsets})
           << ids_buf.view().copy_to(luisa::span<uint32_t>{dev_ids})
           << pt_buf.view().copy_to(luisa::span<uint32_t>{dev_pt})
           << sp_buf.view().copy_to(luisa::span<uint32_t>{dev_sp})
           << w_buf.view().copy_to(luisa::span<float>{dev_w}) << synchronize();
    LUISA_ASSERT(validate_moe_routing_tables(dev_counts, dev_offsets, slots).ok(), "S3 device routing readback valid");
    // The DEVICE routing must equal the independent host routing exactly.
    expect(dev_counts == routing.counts) << "S10 device counts match oracle";
    expect(dev_offsets == routing.offsets) << "S10 device offsets match oracle";
    expect(dev_ids == routing.ids) << "S10 device selected experts match oracle";
    expect(dev_pt == routing.perm_token) << "S10 device perm_token match oracle";
    expect(dev_sp == routing.slot_pos) << "S10 device slot_pos match oracle";
    list.clear();
    for (uint32_t e = 0; e < experts; e++) {
        const uint32_t count = dev_counts[e];
        if (count == 0u) { continue; }
        const uint32_t off = dev_offsets[e];
        auto g_rows = gathered_buf.view().subview(static_cast<size_t>(off) * hidden, static_cast<size_t>(count) * hidden);
        auto gate_e = gate_buf.view().subview(0, static_cast<size_t>(count) * inter);
        auto up_e = up_buf.view().subview(0, static_cast<size_t>(count) * inter);
        auto act_e = act_buf.view().subview(0, static_cast<size_t>(count) * inter);
        auto out_e = expert_out_buf.view().subview(static_cast<size_t>(off) * hidden, static_cast<size_t>(count) * hidden);
        LUISA_ASSERT(md.linear(list, g_rows, gate_bufs[e].view(), gate_e, count, inter, hidden).ok(), "gate");
        LUISA_ASSERT(md.linear(list, g_rows, up_bufs[e].view(), up_e, count, inter, hidden).ok(), "up");
        disp1((*md.sh_gated)(gate_e, up_e, act_e, count * inter), plan_gated_act_dispatch(static_cast<uint64_t>(count) * inter));
        LUISA_ASSERT(md.linear(list, act_e, down_bufs[e].view(), out_e, count, hidden, inter).ok(), "down");
    }
    const auto shared_ffn = [&](BufferView<float> gw, BufferView<float> uw, BufferView<float> dw, BufferView<float> out) {
        auto gs = gate_buf.view().subview(0, static_cast<size_t>(tokens) * shared_inter);
        auto us = up_buf.view().subview(0, static_cast<size_t>(tokens) * shared_inter);
        auto as = act_buf.view().subview(0, static_cast<size_t>(tokens) * shared_inter);
        LUISA_ASSERT(md.linear(list, x_buf.view(), gw, gs, tokens, shared_inter, hidden).ok(), "sg");
        LUISA_ASSERT(md.linear(list, x_buf.view(), uw, us, tokens, shared_inter, hidden).ok(), "su");
        disp1((*md.sh_gated)(gs, us, as, tokens * shared_inter), plan_gated_act_dispatch(static_cast<uint64_t>(tokens) * shared_inter));
        LUISA_ASSERT(md.linear(list, as, dw, out, tokens, hidden, shared_inter).ok(), "sd");
    };
    shared_ffn(s0g.view(), s0u.view(), s0d.view(), shared0_buf.view());
    shared_ffn(s1g.view(), s1u.view(), s1d.view(), shared1_buf.view());
    disp1((*md.sh_scatter)(expert_out_buf.view(), sp_buf.view(), w_buf.view(),
                           shared0_buf.view(), shared1_buf.view(), out_buf.view(), tokens, hidden, top_k),
          plan_moe_scatter_dispatch(tokens, hidden));
    stream << list.commit() << synchronize();
    luisa::vector<float> out;
    ocr::test::download(stream, out_buf, out, static_cast<size_t>(tokens) * hidden);
    stream << synchronize();
    ocr::test::expect_close(out, ref_out, kAtol, kRtol, "moe empty-assignment output");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    MoeLayerDevice md;
    md.init(device);
    "moe_full_layer_64_top6_two_shared"_test = [&] { test_full_moe_layer(md); };
    "moe_empty_assignments"_test = [&] { test_empty_assignments(md); };
}
