// Test for ocr/ops/ops_moe permutation (Phase 10, device).
// This test covers:
// - histogram/counts, exclusive-prefix offsets, token-to-expert permutation
// - duplicate experts within a token's top-k, inactive experts, gather/scatter
// - weighted restore to token order + two shared experts vs the independent oracle
// Synthetic only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include "ocr/ops/ops_moe.h"

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

constexpr float kAtol = 2e-5f;
constexpr float kRtol = 2e-5f;

struct PermDevice {
    Device *device;
    ShaderKeyRegistry reg;
    TypedShaderCache<MoeHistogramKernel> hist;
    TypedShaderCache<MoeOffsetsKernel> offs;
    TypedShaderCache<MoePermutationKernel> perm;
    TypedShaderCache<MoeGatherKernel> gather;
    TypedShaderCache<MoeScatterKernel> scatter;
    const TypedShaderCache<MoeHistogramKernel>::ShaderT *sh_hist{nullptr};
    const TypedShaderCache<MoeOffsetsKernel>::ShaderT *sh_offs{nullptr};
    const TypedShaderCache<MoePermutationKernel>::ShaderT *sh_perm{nullptr};
    const TypedShaderCache<MoeGatherKernel>::ShaderT *sh_gather{nullptr};
    const TypedShaderCache<MoeScatterKernel>::ShaderT *sh_scatter{nullptr};
    void init(Device &d) {
        device = &d;
        hist = TypedShaderCache<MoeHistogramKernel>{d, reg};
        offs = TypedShaderCache<MoeOffsetsKernel>{d, reg};
        perm = TypedShaderCache<MoePermutationKernel>{d, reg};
        gather = TypedShaderCache<MoeGatherKernel>{d, reg};
        scatter = TypedShaderCache<MoeScatterKernel>{d, reg};
        auto h = hist.get_or_compile("hist", [] { return make_moe_histogram_kernel(); });
        auto o = offs.get_or_compile("offs", [] { return make_moe_offsets_kernel(); });
        auto p = perm.get_or_compile("perm", [] { return make_moe_permutation_kernel(); });
        auto g = gather.get_or_compile("gather", [] { return make_moe_gather_kernel(); });
        auto s = scatter.get_or_compile("scatter", [] { return make_moe_scatter_kernel(); });
        LUISA_ASSERT(h.ok() && o.ok() && p.ok() && g.ok() && s.ok(), "compile perm kernels");
        sh_hist = *h.value; sh_offs = *o.value; sh_perm = *p.value; sh_gather = *g.value; sh_scatter = *s.value;
    }
};

// Run histogram -> offsets -> permutation on the device for given ids.
void device_permute(PermDevice &pd, luisa::span<const uint32_t> ids, uint32_t experts, uint32_t top_k,
                    luisa::vector<uint32_t> &counts, luisa::vector<uint32_t> &offsets,
                    luisa::vector<uint32_t> &perm_slot, luisa::vector<uint32_t> &slot_pos,
                    luisa::vector<uint32_t> &perm_token) {
    auto stream = pd.device->create_stream();
    const uint32_t slots = static_cast<uint32_t>(ids.size());
    auto ids_buf = ocr::test::upload(*pd.device, stream, luisa::span<const uint32_t>{ids});
    auto counts_buf = pd.device->create_buffer<uint32_t>(experts);
    auto offs_buf = pd.device->create_buffer<uint32_t>(experts);
    auto ps_buf = pd.device->create_buffer<uint32_t>(slots);
    auto sp_buf = pd.device->create_buffer<uint32_t>(slots);
    auto pt_buf = pd.device->create_buffer<uint32_t>(slots);
    stream << ids_buf.view().copy_from(luisa::span<const uint32_t>{ids});
    auto ph = plan_moe_histogram_dispatch(experts);
    auto po = plan_moe_offsets_dispatch(experts);
    auto pp = plan_moe_permutation_dispatch(experts);
    LUISA_ASSERT(ph.ok() && po.ok() && pp.ok(), "plan perm");
    if (auto c = dispatch_1d((*pd.sh_hist)(ids_buf.view(), counts_buf.view(), slots, experts), *ph.value)) { stream << std::move(c); }
    if (auto c = dispatch_1d((*pd.sh_offs)(counts_buf.view(), offs_buf.view(), experts), *po.value)) { stream << std::move(c); }
    if (auto c = dispatch_1d((*pd.sh_perm)(ids_buf.view(), offs_buf.view(), ps_buf.view(), sp_buf.view(), pt_buf.view(), slots, experts, top_k), *pp.value)) { stream << std::move(c); }
    ocr::test::download(stream, counts_buf, counts, experts);
    ocr::test::download(stream, offs_buf, offsets, experts);
    ocr::test::download(stream, ps_buf, perm_slot, slots);
    ocr::test::download(stream, sp_buf, slot_pos, slots);
    ocr::test::download(stream, pt_buf, perm_token, slots);
    stream << synchronize();
}

void test_permutation_basic(PermDevice &pd) {
    // ids with duplicates across tokens and inactive experts.
    const uint32_t experts = 64, top_k = 6, tokens = 5;
    const uint32_t slots = tokens * top_k;
    ocr::test::Lcg lcg;
    lcg.state = 0x77u;
    luisa::vector<uint32_t> ids(slots);
    for (auto &id : ids) { id = lcg.next() % experts; }
    // Force some inactive experts by only using ids 0..9.
    for (auto &id : ids) { id = id % 10u; }
    luisa::vector<uint32_t> counts, offsets, perm_slot, slot_pos, perm_token;
    device_permute(pd, ids, experts, top_k, counts, offsets, perm_slot, slot_pos, perm_token);
    // Independent oracle permutation.
    luisa::vector<uint32_t> ref_counts(experts, 0u), ref_offsets(experts), ref_ps(slots), ref_sp(slots), ref_pt(slots);
    for (auto id : ids) { ref_counts[id]++; }
    uint32_t acc = 0;
    for (uint32_t e = 0; e < experts; e++) { ref_offsets[e] = acc; acc += ref_counts[e]; }
    luisa::vector<uint32_t> run(experts);
    for (uint32_t e = 0; e < experts; e++) { run[e] = ref_offsets[e]; }
    for (uint32_t s = 0; s < slots; s++) {
        const uint32_t e = ids[s];
        const uint32_t dst = run[e]++;
        ref_ps[dst] = s; ref_sp[s] = dst; ref_pt[dst] = s / top_k;
    }
    expect(counts == ref_counts) << "histogram counts match";
    expect(offsets == ref_offsets) << "exclusive-prefix offsets match";
    expect(perm_slot == ref_ps) << "perm_slot matches";
    expect(slot_pos == ref_sp) << "slot_pos inverse matches";
    expect(perm_token == ref_pt) << "perm_token matches";
    // Inactive experts have zero counts.
    uint32_t active = 0;
    for (uint32_t e = 0; e < experts; e++) { if (counts[e] > 0u) { active++; } }
    expect(active <= 10u) << "only experts 0..9 active";
    // Permutation is a bijection: every slot appears exactly once in perm_slot.
    luisa::vector<uint32_t> seen(slots, 0u);
    for (auto s : perm_slot) { seen[s]++; }
    for (auto c : seen) { expect(eq(c, 1u)) << "permutation is a bijection"; }
}

void test_gather_scatter(PermDevice &pd) {
    // Gather token rows into expert-sorted order, then weighted scatter restore
    // + two shared experts, vs the independent oracle.
    const uint32_t experts = 64, top_k = 6, tokens = 4, hidden = 8;
    const uint32_t slots = tokens * top_k;
    ocr::test::Lcg lcg;
    lcg.state = 0x3131u;
    luisa::vector<uint32_t> ids(slots);
    for (auto &id : ids) { id = lcg.next() % experts; }
    luisa::vector<float> x(static_cast<size_t>(tokens) * hidden);
    lcg.fill(x, -1.0f, 1.0f);
    luisa::vector<uint32_t> counts, offsets, perm_slot, slot_pos, perm_token;
    device_permute(pd, ids, experts, top_k, counts, offsets, perm_slot, slot_pos, perm_token);
    // Device gather.
    auto stream = pd.device->create_stream();
    auto x_buf = ocr::test::upload(*pd.device, stream, luisa::span<const float>{x});
    auto pt_buf = ocr::test::upload(*pd.device, stream, luisa::span<const uint32_t>{perm_token});
    auto gathered_buf = pd.device->create_buffer<float>(static_cast<size_t>(slots) * hidden);
    stream << x_buf.view().copy_from(luisa::span<const float>{x}) << pt_buf.view().copy_from(luisa::span<const uint32_t>{perm_token});
    auto pg = plan_moe_gather_dispatch(slots, hidden);
    LUISA_ASSERT(pg.ok(), "plan gather");
    if (auto c = dispatch_1d((*pd.sh_gather)(x_buf.view(), pt_buf.view(), gathered_buf.view(), slots, hidden), *pg.value)) { stream << std::move(c); }
    luisa::vector<float> gathered;
    ocr::test::download(stream, gathered_buf, gathered, static_cast<size_t>(slots) * hidden);
    stream << synchronize();
    luisa::vector<float> ref_gathered;
    LUISA_ASSERT(ref_moe_gather(x, perm_token, slots, hidden, ref_gathered).ok(), "ref gather");
    ocr::test::expect_close(gathered, ref_gathered, kAtol, kRtol, "moe gather");
    // Weighted scatter restore + two shared experts. Use the gathered rows as
    // the unweighted expert outputs (identity expert) so the test isolates the
    // scatter math.
    luisa::vector<float> weights(slots);
    lcg.fill(weights, 0.0f, 1.0f);
    // Normalize weights per token (sum 1).
    for (uint32_t t = 0; t < tokens; t++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < top_k; j++) { sum += weights[static_cast<size_t>(t) * top_k + j]; }
        for (uint32_t j = 0; j < top_k; j++) { weights[static_cast<size_t>(t) * top_k + j] /= sum; }
    }
    luisa::vector<float> shared0(static_cast<size_t>(tokens) * hidden), shared1(static_cast<size_t>(tokens) * hidden);
    lcg.fill(shared0, -0.5f, 0.5f);
    lcg.fill(shared1, -0.5f, 0.5f);
    auto sp_buf = ocr::test::upload(*pd.device, stream, luisa::span<const uint32_t>{slot_pos});
    auto w_buf = ocr::test::upload(*pd.device, stream, luisa::span<const float>{weights});
    auto s0_buf = ocr::test::upload(*pd.device, stream, luisa::span<const float>{shared0});
    auto s1_buf = ocr::test::upload(*pd.device, stream, luisa::span<const float>{shared1});
    auto out_buf = pd.device->create_buffer<float>(static_cast<size_t>(tokens) * hidden);
    stream << sp_buf.view().copy_from(luisa::span<const uint32_t>{slot_pos})
           << w_buf.view().copy_from(luisa::span<const float>{weights})
           << s0_buf.view().copy_from(luisa::span<const float>{shared0})
           << s1_buf.view().copy_from(luisa::span<const float>{shared1});
    auto ps = plan_moe_scatter_dispatch(tokens, hidden);
    LUISA_ASSERT(ps.ok(), "plan scatter");
    if (auto c = dispatch_1d((*pd.sh_scatter)(gathered_buf.view(), sp_buf.view(), w_buf.view(),
                                              s0_buf.view(), s1_buf.view(), out_buf.view(), tokens, hidden, top_k), *ps.value)) { stream << std::move(c); }
    luisa::vector<float> out;
    ocr::test::download(stream, out_buf, out, static_cast<size_t>(tokens) * hidden);
    stream << synchronize();
    luisa::vector<float> ref_out;
    LUISA_ASSERT(ref_moe_scatter(gathered, slot_pos, weights, shared0, shared1, tokens, hidden, top_k, ref_out).ok(), "ref scatter");
    ocr::test::expect_close(out, ref_out, kAtol, kRtol, "moe weighted scatter + shared");
}

void test_duplicate_experts(PermDevice &pd) {
    // A token selects the SAME expert multiple times in its top-k (possible when
    // top-k > distinct available). The permutation must place each slot stably.
    const uint32_t experts = 8, top_k = 6, tokens = 2;
    const uint32_t slots = tokens * top_k;
    luisa::vector<uint32_t> ids = {3, 3, 3, 5, 5, 1,  7, 7, 0, 0, 0, 2};
    luisa::vector<uint32_t> counts, offsets, perm_slot, slot_pos, perm_token;
    device_permute(pd, ids, experts, top_k, counts, offsets, perm_slot, slot_pos, perm_token);
    // Oracle.
    luisa::vector<uint32_t> ref_counts(experts, 0u);
    for (auto id : ids) { ref_counts[id]++; }
    expect(counts == ref_counts) << "duplicate-expert counts";
    // Within expert 3, slots 0,1,2 must appear in ascending order.
    luisa::vector<uint32_t> e3_slots;
    for (uint32_t d = offsets[3]; d < offsets[3] + counts[3]; d++) { e3_slots.push_back(perm_slot[d]); }
    expect(e3_slots == luisa::vector<uint32_t>({0u, 1u, 2u})) << "stable within-expert order";
    // perm_token for those slots is token 0.
    for (uint32_t d = offsets[3]; d < offsets[3] + counts[3]; d++) { expect(eq(perm_token[d], 0u)); }
}


// M5/M6/S3: the host oracles and the shared routing validator must REJECT
// malformed extents/indices instead of reading out of bounds.
void test_oracle_boundary_validation() {
    const uint32_t hidden = 4, top_k = 2, tokens = 2;
    const uint32_t slots = tokens * top_k;
    luisa::vector<float> x(static_cast<size_t>(tokens) * hidden, 0.5f);
    luisa::vector<float> out;
    // M5: perm_token index outside the source rows.
    {
        luisa::vector<uint32_t> bad_perm = {0u, 1u, 2u, 0u};// token 2 does not exist
        expect(!ref_moe_gather(x, bad_perm, slots, hidden, out).ok()) << "M5 gather OOB perm_token refused";
    }
    // M5: x size not a whole number of rows.
    {
        luisa::vector<float> ragged(x.size() + 1u, 0.0f);
        luisa::vector<uint32_t> perm = {0u, 1u, 0u, 1u};
        expect(!ref_moe_gather(ragged, perm, slots, hidden, out).ok()) << "M5 ragged x refused";
    }
    // M5: zero hidden.
    {
        luisa::vector<uint32_t> perm = {0u, 1u, 0u, 1u};
        expect(!ref_moe_gather(x, perm, slots, 0u, out).ok()) << "M5 zero hidden refused";
    }
    // M5: valid gather still works.
    {
        luisa::vector<uint32_t> perm = {0u, 1u, 0u, 1u};
        expect(ref_moe_gather(x, perm, slots, hidden, out).ok()) << "valid gather accepted";
        expect(eq(out.size(), static_cast<size_t>(slots) * hidden));
    }
    // M6: expert_out too small, and slot_pos outside the slot range.
    luisa::vector<float> expert_out(static_cast<size_t>(slots) * hidden, 0.25f);
    luisa::vector<float> weights(slots, 0.5f);
    luisa::vector<float> shared(static_cast<size_t>(tokens) * hidden, 0.1f);
    {
        luisa::vector<uint32_t> slot_pos = {0u, 1u, 2u, 3u};
        luisa::vector<float> small_expert(expert_out.size() - 1u, 0.0f);
        expect(!ref_moe_scatter(small_expert, slot_pos, weights, shared, shared, tokens, hidden, top_k, out).ok())
            << "M6 undersized expert_out refused";
        luisa::vector<uint32_t> bad_pos = {0u, 1u, 2u, slots};// == slots is OOB
        expect(!ref_moe_scatter(expert_out, bad_pos, weights, shared, shared, tokens, hidden, top_k, out).ok())
            << "M6 OOB slot_pos refused";
        expect(ref_moe_scatter(expert_out, slot_pos, weights, shared, shared, tokens, hidden, top_k, out).ok())
            << "valid scatter accepted";
    }
    // S3: the shared routing-table validator (used by BOTH the device readback
    // path and the oracle) rejects non-prefix offsets, wrong sums, and overflow.
    {
        luisa::vector<uint32_t> counts = {2u, 2u};
        luisa::vector<uint32_t> offsets = {0u, 2u};
        expect(validate_moe_routing_tables(counts, offsets, slots).ok()) << "valid routing tables accepted";
        luisa::vector<uint32_t> bad_offsets = {0u, 3u};
        expect(!validate_moe_routing_tables(counts, bad_offsets, slots).ok()) << "S3 non-prefix offsets refused";
        luisa::vector<uint32_t> bad_counts = {2u, 3u};// sum 5 != slots 4
        luisa::vector<uint32_t> off2 = {0u, 2u};
        expect(!validate_moe_routing_tables(bad_counts, off2, slots).ok()) << "S3 wrong count sum refused";
        luisa::vector<uint32_t> overflow_counts = {3u, 3u};
        luisa::vector<uint32_t> off3 = {0u, 3u};
        expect(!validate_moe_routing_tables(overflow_counts, off3, slots).ok()) << "S3 offset+count overflow refused";
        luisa::vector<uint32_t> short_offsets = {0u};
        expect(!validate_moe_routing_tables(counts, short_offsets, slots).ok()) << "S3 size mismatch refused";
    }
    // M7: ref_moe_layer rejects tokens==0 BEFORE dividing ids.size() by tokens,
    // and rejects an inconsistent routing table.
    {
        MoeRouterConfig rc{4u, 2u};
        luisa::vector<float> rx(static_cast<size_t>(tokens) * hidden, 0.3f);
        luisa::vector<float> rw(static_cast<size_t>(hidden) * 4u, 0.2f);
        MoeRouting routing;
        LUISA_ASSERT(ref_moe_route(rx, rw, tokens, hidden, rc, routing).ok(), "route for layer validation");
        luisa::vector<MoeExpertWeights> routed(4u);
        for (auto &e : routed) {
            e.gate_w.assign(static_cast<size_t>(hidden) * hidden, 0.1f);
            e.up_w.assign(static_cast<size_t>(hidden) * hidden, 0.1f);
            e.down_w.assign(static_cast<size_t>(hidden) * hidden, 0.1f);
        }
        MoeExpertWeights s0 = routed[0], s1 = routed[1];
        luisa::vector<float> eo, wt, o0, o1, mo;
        expect(!ref_moe_layer(rx, routing, routed, s0, s1, 0u, hidden, hidden, hidden, eo, wt, o0, o1, mo).ok())
            << "M7 tokens==0 refused before division";
        auto bad = routing;
        bad.offsets[0] = 1u;// break the exclusive prefix
        expect(!ref_moe_layer(rx, bad, routed, s0, s1, tokens, hidden, hidden, hidden, eo, wt, o0, o1, mo).ok())
            << "M7 inconsistent routing refused";
        expect(ref_moe_layer(rx, routing, routed, s0, s1, tokens, hidden, hidden, hidden, eo, wt, o0, o1, mo).ok())
            << "valid MoE layer accepted";
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    PermDevice pd;
    pd.init(device);
    "moe_permutation_basic"_test = [&] { test_permutation_basic(pd); };
    "moe_gather_scatter"_test = [&] { test_gather_scatter(pd); };
    "moe_duplicate_experts"_test = [&] { test_duplicate_experts(pd); };
    "moe_oracle_boundary_validation"_test = [] { test_oracle_boundary_validation(); };
}
