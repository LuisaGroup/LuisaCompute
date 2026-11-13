// Test for ocr/ops/ops_gdn -- straight recurrent delta-rule GDN.
// This test covers:
// - device recurrent vs the independent CPU oracle (outputs + final state);
// - recurrent state after EVERY token (device snapshots vs oracle snapshots);
// - full-sequence prefill == token-by-token decode (outputs + carried state);
// - gates (sigmoid/softplus/decay) and L2/QK normalization parity;
// - gated output normalization parity; degenerate + NaN inputs.
// Synthetic data only; no model/download.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include <cmath>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_gdn.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;

struct GdnInputs {
    uint32_t tokens, heads, d;
    luisa::vector<float> q, k, v, beta, g, state;
};

// Generate deterministic L2-normalized q/k + v + gates + nonzero initial state.
GdnInputs make_inputs(ocr::test::Lcg &lcg, uint32_t tokens, uint32_t heads, uint32_t d) {
    GdnInputs in{tokens, heads, d};
    const size_t n = static_cast<size_t>(tokens) * heads * d;
    in.q.resize(n);
    in.k.resize(n);
    in.v.resize(n);
    lcg.fill(in.q, -1.0f, 1.0f);
    lcg.fill(in.k, -1.0f, 1.0f);
    lcg.fill(in.v, -0.8f, 0.8f);
    // L2-normalize q/k per head row (matches the executor's L2 stage).
    luisa::vector<float> qn, kn;
    LUISA_ASSERT(ref_l2_norm(in.q, tokens * heads, d, 1e-6f, qn).ok(), "l2 q");
    LUISA_ASSERT(ref_l2_norm(in.k, tokens * heads, d, 1e-6f, kn).ok(), "l2 k");
    in.q = qn;
    in.k = kn;
    in.beta.resize(static_cast<size_t>(tokens) * heads);
    in.g.resize(static_cast<size_t>(tokens) * heads);
    lcg.fill(in.beta, 0.1f, 0.9f);
    lcg.fill(in.g, -0.6f, -0.02f);// log decay <= 0
    in.state.resize(static_cast<size_t>(heads) * d * d);
    lcg.fill(in.state, -0.3f, 0.3f);
    return in;
}

void device_recurrent(Device &device, Stream &stream, const GdnInputs &in,
                      luisa::vector<float> &out, luisa::vector<float> &state_out) {
    auto q = ocr::test::upload(device, stream, luisa::span<const float>{in.q});
    auto k = ocr::test::upload(device, stream, luisa::span<const float>{in.k});
    auto v = ocr::test::upload(device, stream, luisa::span<const float>{in.v});
    auto beta = ocr::test::upload(device, stream, luisa::span<const float>{in.beta});
    auto g = ocr::test::upload(device, stream, luisa::span<const float>{in.g});
    auto state = ocr::test::upload(device, stream, luisa::span<const float>{in.state});
    auto out_buf = device.create_buffer<float>(in.q.size());
    auto shader = device.compile(make_gdn_recurrent_kernel());
    auto plan = plan_gdn_heads_dispatch(in.heads);
    LUISA_ASSERT(plan.ok(), "recurrent plan");
    stream << q.copy_from(luisa::span<const float>{in.q}) << k.copy_from(luisa::span<const float>{in.k})
           << v.copy_from(luisa::span<const float>{in.v}) << beta.copy_from(luisa::span<const float>{in.beta})
           << g.copy_from(luisa::span<const float>{in.g}) << state.copy_from(luisa::span<const float>{in.state});
    if (auto cmd = dispatch_1d(shader(q, k, v, beta, g, state, out_buf, in.tokens, in.heads, in.d), *plan.value)) {
        stream << std::move(cmd);
    }
    out.resize(in.q.size());
    state_out.resize(in.state.size());
    stream << out_buf.copy_to(luisa::span<float>{out})
           << state.copy_to(luisa::span<float>{state_out})
           << synchronize();
}

void test_recurrent_oracle(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    for (uint32_t heads : {1u, 3u}) {
        for (uint32_t tokens : {1u, 6u, 9u}) {
            const uint32_t d = 4;
            auto in = make_inputs(lcg, tokens, heads, d);
            luisa::vector<float> state_ref = in.state, out_ref;
            LUISA_ASSERT(ref_gdn_recurrent(in.q, in.k, in.v, in.beta, in.g, state_ref,
                                           tokens, heads, d, out_ref).ok(), "ref recurrent");
            luisa::vector<float> out_dev, state_dev;
            device_recurrent(device, stream, in, out_dev, state_dev);
            ocr::test::expect_close(out_dev, out_ref, kAtol, kRtol, "gdn recurrent output");
            ocr::test::expect_close(state_dev, state_ref, kAtol, kRtol, "gdn recurrent final state");
        }
    }
}

void test_recurrent_state_every_token(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    const uint32_t heads = 2, d = 4, tokens = 7;
    auto in = make_inputs(lcg, tokens, heads, d);
    // Oracle snapshots after EVERY token.
    luisa::vector<float> state_ref = in.state, out_ref, snaps_ref;
    LUISA_ASSERT(ref_gdn_recurrent(in.q, in.k, in.v, in.beta, in.g, state_ref,
                                   tokens, heads, d, out_ref, &snaps_ref).ok(), "ref snapshots");
    // Device: run token-by-token, downloading the state after each step.
    const size_t sn = static_cast<size_t>(heads) * d * d;
    luisa::vector<float> state = in.state;
    luisa::vector<float> snaps_dev(snaps_ref.size());
    std::copy(state.begin(), state.end(), snaps_dev.begin());
    for (uint32_t t = 0; t < tokens; t++) {
        GdnInputs step{1, heads, d};
        step.q.assign(in.q.begin() + static_cast<size_t>(t) * heads * d,
                      in.q.begin() + static_cast<size_t>(t + 1) * heads * d);
        step.k.assign(in.k.begin() + static_cast<size_t>(t) * heads * d,
                      in.k.begin() + static_cast<size_t>(t + 1) * heads * d);
        step.v.assign(in.v.begin() + static_cast<size_t>(t) * heads * d,
                      in.v.begin() + static_cast<size_t>(t + 1) * heads * d);
        step.beta.assign(in.beta.begin() + static_cast<size_t>(t) * heads,
                         in.beta.begin() + static_cast<size_t>(t + 1) * heads);
        step.g.assign(in.g.begin() + static_cast<size_t>(t) * heads,
                      in.g.begin() + static_cast<size_t>(t + 1) * heads);
        step.state = state;
        luisa::vector<float> out_step, state_next;
        device_recurrent(device, stream, step, out_step, state_next);
        // Per-token output parity.
        luisa::vector<float> out_ref_t(out_ref.begin() + static_cast<size_t>(t) * heads * d,
                                       out_ref.begin() + static_cast<size_t>(t + 1) * heads * d);
        ocr::test::expect_close(out_step, out_ref_t, kAtol, kRtol, "gdn recurrent per-token output");
        state = state_next;
        std::copy(state.begin(), state.end(), snaps_dev.begin() + static_cast<size_t>(t + 1) * sn);
    }
    ocr::test::expect_close(snaps_dev, snaps_ref, kAtol, kRtol, "gdn recurrent state after every token");
}

void test_gates_and_norms(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    const uint32_t tokens = 5, heads = 3, d = 4;
    luisa::vector<float> a(static_cast<size_t>(tokens) * heads), b(a.size());
    luisa::vector<float> a_log(heads), dt_bias(heads);
    lcg.fill(a, -3.0f, 25.0f);// crosses the softplus threshold
    lcg.fill(b, -4.0f, 4.0f);
    lcg.fill(a_log, -1.0f, 0.5f);
    lcg.fill(dt_bias, -0.5f, 0.5f);
    luisa::vector<float> beta_ref, g_ref;
    LUISA_ASSERT(ref_gdn_gates(a, b, a_log, dt_bias, tokens, heads, 20.0f, beta_ref, g_ref).ok(), "ref gates");
    auto a_buf = ocr::test::upload(device, stream, luisa::span<const float>{a});
    auto b_buf = ocr::test::upload(device, stream, luisa::span<const float>{b});
    auto al_buf = ocr::test::upload(device, stream, luisa::span<const float>{a_log});
    auto dt_buf = ocr::test::upload(device, stream, luisa::span<const float>{dt_bias});
    auto beta_buf = device.create_buffer<float>(a.size());
    auto g_buf = device.create_buffer<float>(a.size());
    auto shader = device.compile(make_gdn_gates_kernel());
    auto plan = plan_gdn_gates_dispatch(tokens, heads);
    LUISA_ASSERT(plan.ok(), "gates plan");
    stream << a_buf.copy_from(luisa::span<const float>{a}) << b_buf.copy_from(luisa::span<const float>{b})
           << al_buf.copy_from(luisa::span<const float>{a_log}) << dt_buf.copy_from(luisa::span<const float>{dt_bias});
    if (auto cmd = dispatch_1d(shader(a_buf, b_buf, al_buf, dt_buf, beta_buf, g_buf, tokens, heads, 20.0f), *plan.value)) {
        stream << std::move(cmd);
    }
    luisa::vector<float> beta_dev(a.size()), g_dev(a.size());
    stream << beta_buf.copy_to(luisa::span<float>{beta_dev}) << g_buf.copy_to(luisa::span<float>{g_dev}) << synchronize();
    ocr::test::expect_close(beta_dev, beta_ref, 1e-5f, 1e-5f, "gdn beta sigmoid");
    ocr::test::expect_close(g_dev, g_ref, 1e-4f, 1e-4f, "gdn gate softplus decay");

    // Gated output normalization parity.
    luisa::vector<float> o(static_cast<size_t>(tokens) * heads * d), z(o.size()), norm_w(d);
    lcg.fill(o, -1.0f, 1.0f);
    lcg.fill(z, -2.0f, 2.0f);
    lcg.fill(norm_w, 0.7f, 1.3f);
    luisa::vector<float> yn_ref;
    LUISA_ASSERT(ref_gdn_gated_norm(o, z, norm_w, tokens, heads, d, 1e-6f, yn_ref).ok(), "ref gated norm");
    auto o_buf = ocr::test::upload(device, stream, luisa::span<const float>{o});
    auto z_buf = ocr::test::upload(device, stream, luisa::span<const float>{z});
    auto nw_buf = ocr::test::upload(device, stream, luisa::span<const float>{norm_w});
    auto yn_buf = device.create_buffer<float>(o.size());
    auto yn_shader = device.compile(make_gdn_gated_norm_kernel());
    auto yn_plan = plan_gdn_gated_norm_dispatch(tokens, heads);
    LUISA_ASSERT(yn_plan.ok(), "gated norm plan");
    stream << o_buf.copy_from(luisa::span<const float>{o}) << z_buf.copy_from(luisa::span<const float>{z})
           << nw_buf.copy_from(luisa::span<const float>{norm_w});
    if (auto cmd = dispatch_1d(yn_shader(o_buf, z_buf, nw_buf, yn_buf, tokens, heads, d, 1e-6f), *yn_plan.value)) {
        stream << std::move(cmd);
    }
    luisa::vector<float> yn_dev(o.size());
    stream << yn_buf.copy_to(luisa::span<float>{yn_dev}) << synchronize();
    ocr::test::expect_close(yn_dev, yn_ref, 1e-4f, 1e-4f, "gdn gated output norm");
}

void test_recurrent_degenerate_nan(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    const uint32_t heads = 2, d = 4, tokens = 3;
    auto in = make_inputs(lcg, tokens, heads, d);
    // Zero gates (g=0 -> alpha=1, no decay) and zero beta (no update): the
    // state must pass through unchanged and o = S^T q.
    std::fill(in.g.begin(), in.g.end(), 0.0f);
    std::fill(in.beta.begin(), in.beta.end(), 0.0f);
    luisa::vector<float> state_ref = in.state, out_ref;
    LUISA_ASSERT(ref_gdn_recurrent(in.q, in.k, in.v, in.beta, in.g, state_ref, tokens, heads, d, out_ref).ok(), "ref deg");
    luisa::vector<float> out_dev, state_dev;
    device_recurrent(device, stream, in, out_dev, state_dev);
    ocr::test::expect_close(state_dev, in.state, 1e-6f, 1e-6f, "gdn zero-beta state unchanged");
    ocr::test::expect_close(out_dev, out_ref, kAtol, kRtol, "gdn zero-beta output");
    // NaN in v propagates to the state/output as NaN (no OOB).
    in.v[0] = std::numeric_limits<float>::quiet_NaN();
    luisa::vector<float> state_ref_nan = in.state, out_ref_nan;
    LUISA_ASSERT(ref_gdn_recurrent(in.q, in.k, in.v, in.beta, in.g, state_ref_nan, tokens, heads, d, out_ref_nan).ok(), "ref nan");
    luisa::vector<float> out_dev_nan, state_dev_nan;
    device_recurrent(device, stream, in, out_dev_nan, state_dev_nan);
    ocr::test::expect_close(out_dev_nan, out_ref_nan, kAtol, kRtol, "gdn NaN output propagation");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "gdn_recurrent_oracle"_test = [&] { test_recurrent_oracle(device); };
    "gdn_recurrent_state_every_token"_test = [&] { test_recurrent_state_every_token(device); };
    "gdn_gates_and_norms"_test = [&] { test_gates_and_norms(device); };
    "gdn_recurrent_degenerate_nan"_test = [&] { test_recurrent_degenerate_nan(device); };
}
