// Test for ocr/ops/ops_gdn -- chunked WY/UT-transform GDN prefill.
// This test covers:
// - device chunk kernel vs the independent full-L-matrix CPU oracle;
// - chunked == straight recurrent (outputs + carried state) at EVERY chunk
//   boundary, for chunk sizes 1 / nondivisible / large (whole sequence);
// - inter-chunk state transitions with a NONZERO carried initial state;
// - degenerate (single token) + NaN inputs.
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

GdnInputs make_inputs(ocr::test::Lcg &lcg, uint32_t tokens, uint32_t heads, uint32_t d) {
    GdnInputs in{tokens, heads, d};
    const size_t n = static_cast<size_t>(tokens) * heads * d;
    in.q.resize(n);
    in.k.resize(n);
    in.v.resize(n);
    lcg.fill(in.q, -1.0f, 1.0f);
    lcg.fill(in.k, -1.0f, 1.0f);
    lcg.fill(in.v, -0.8f, 0.8f);
    luisa::vector<float> qn, kn;
    LUISA_ASSERT(ref_l2_norm(in.q, tokens * heads, d, 1e-6f, qn).ok(), "l2 q");
    LUISA_ASSERT(ref_l2_norm(in.k, tokens * heads, d, 1e-6f, kn).ok(), "l2 k");
    in.q = qn;
    in.k = kn;
    in.beta.resize(static_cast<size_t>(tokens) * heads);
    in.g.resize(static_cast<size_t>(tokens) * heads);
    lcg.fill(in.beta, 0.1f, 0.9f);
    lcg.fill(in.g, -0.6f, -0.02f);
    in.state.resize(static_cast<size_t>(heads) * d * d);
    lcg.fill(in.state, -0.3f, 0.3f);
    return in;
}

// Run the device CHUNK kernel over one chunk; returns out + updated state.
void device_chunk(Device &device, Stream &stream, const GdnInputs &in,
                  luisa::vector<float> &out, luisa::vector<float> &state_out) {
    auto q = ocr::test::upload(device, stream, luisa::span<const float>{in.q});
    auto k = ocr::test::upload(device, stream, luisa::span<const float>{in.k});
    auto v = ocr::test::upload(device, stream, luisa::span<const float>{in.v});
    auto beta = ocr::test::upload(device, stream, luisa::span<const float>{in.beta});
    auto g = ocr::test::upload(device, stream, luisa::span<const float>{in.g});
    auto state = ocr::test::upload(device, stream, luisa::span<const float>{in.state});
    auto out_buf = device.create_buffer<float>(in.q.size());
    auto w_buf = device.create_buffer<float>(in.q.size());
    auto u_buf = device.create_buffer<float>(in.q.size());
    auto c_buf = device.create_buffer<float>(static_cast<size_t>(in.tokens) * in.heads);
    auto shader = device.compile(make_gdn_chunk_kernel());
    auto plan = plan_gdn_heads_dispatch(in.heads);
    LUISA_ASSERT(plan.ok(), "chunk plan");
    stream << q.copy_from(luisa::span<const float>{in.q}) << k.copy_from(luisa::span<const float>{in.k})
           << v.copy_from(luisa::span<const float>{in.v}) << beta.copy_from(luisa::span<const float>{in.beta})
           << g.copy_from(luisa::span<const float>{in.g}) << state.copy_from(luisa::span<const float>{in.state});
    if (auto cmd = dispatch_1d(shader(q, k, v, beta, g, state, out_buf, w_buf, u_buf, c_buf,
                                      in.tokens, in.heads, in.d), *plan.value)) {
        stream << std::move(cmd);
    }
    out.resize(in.q.size());
    state_out.resize(in.state.size());
    stream << out_buf.copy_to(luisa::span<float>{out})
           << state.copy_to(luisa::span<float>{state_out})
           << synchronize();
}

void test_chunk_oracle(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    for (uint32_t heads : {1u, 2u}) {
        for (uint32_t tokens : {1u, 4u, 7u}) {
            const uint32_t d = 4;
            auto in = make_inputs(lcg, tokens, heads, d);
            luisa::vector<float> state_ref = in.state, out_ref;
            LUISA_ASSERT(ref_gdn_chunk(in.q, in.k, in.v, in.beta, in.g, state_ref, tokens, heads, d, out_ref).ok(), "ref chunk");
            luisa::vector<float> out_dev, state_dev;
            device_chunk(device, stream, in, out_dev, state_dev);
            ocr::test::expect_close(out_dev, out_ref, kAtol, kRtol, "gdn chunk output vs oracle");
            ocr::test::expect_close(state_dev, state_ref, kAtol, kRtol, "gdn chunk state vs oracle");
        }
    }
}

// Chunked device path (carried state) vs the straight-recurrent oracle over the
// WHOLE sequence, comparing outputs AND the state at EVERY chunk boundary.
void test_chunk_vs_recurrent_boundaries(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    const uint32_t heads = 2, d = 4, total = 13;
    auto in = make_inputs(lcg, total, heads, d);
    // Straight-recurrent oracle with per-token state snapshots.
    luisa::vector<float> state_ref = in.state, out_ref, snaps_ref;
    LUISA_ASSERT(ref_gdn_recurrent(in.q, in.k, in.v, in.beta, in.g, state_ref,
                                   total, heads, d, out_ref, &snaps_ref).ok(), "ref recurrent");
    const size_t sn = static_cast<size_t>(heads) * d * d;
    for (uint32_t chunk : {1u, 4u, 5u, 13u, 32u}) {
        luisa::vector<float> state = in.state;
        luisa::vector<float> out_chunked;
        uint32_t pos0 = 0;
        bool boundary_ok = true;
        while (pos0 < total) {
            const uint32_t rows = std::min(chunk, total - pos0);
            GdnInputs c{rows, heads, d};
            c.q.assign(in.q.begin() + static_cast<size_t>(pos0) * heads * d,
                       in.q.begin() + static_cast<size_t>(pos0 + rows) * heads * d);
            c.k.assign(in.k.begin() + static_cast<size_t>(pos0) * heads * d,
                       in.k.begin() + static_cast<size_t>(pos0 + rows) * heads * d);
            c.v.assign(in.v.begin() + static_cast<size_t>(pos0) * heads * d,
                       in.v.begin() + static_cast<size_t>(pos0 + rows) * heads * d);
            c.beta.assign(in.beta.begin() + static_cast<size_t>(pos0) * heads,
                          in.beta.begin() + static_cast<size_t>(pos0 + rows) * heads);
            c.g.assign(in.g.begin() + static_cast<size_t>(pos0) * heads,
                       in.g.begin() + static_cast<size_t>(pos0 + rows) * heads);
            c.state = state;
            luisa::vector<float> out_c, state_next;
            device_chunk(device, stream, c, out_c, state_next);
            out_chunked.insert(out_chunked.end(), out_c.begin(), out_c.end());
            state = state_next;
            pos0 += rows;
            // Compare the carried state to the recurrent snapshot at this boundary.
            luisa::vector<float> snap_ref(snaps_ref.begin() + static_cast<size_t>(pos0) * sn,
                                          snaps_ref.begin() + static_cast<size_t>(pos0 + 1) * sn);
            for (size_t i = 0; i < sn; i++) {
                const float diff = std::abs(state[i] - snap_ref[i]);
                if (!(diff <= kAtol + kRtol * std::abs(snap_ref[i]))) { boundary_ok = false; }
            }
        }
        ocr::test::expect_close(out_chunked, out_ref, kAtol, kRtol, "gdn chunked == recurrent output");
        expect(boundary_ok) << "gdn chunked state matches recurrent at every chunk boundary";
    }
}

void test_chunk_degenerate_nan(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    const uint32_t heads = 2, d = 4;
    // Single token chunk == recurrent single step.
    auto in = make_inputs(lcg, 1, heads, d);
    luisa::vector<float> state_ref = in.state, out_ref;
    LUISA_ASSERT(ref_gdn_recurrent(in.q, in.k, in.v, in.beta, in.g, state_ref, 1, heads, d, out_ref).ok(), "ref single");
    luisa::vector<float> out_dev, state_dev;
    device_chunk(device, stream, in, out_dev, state_dev);
    ocr::test::expect_close(out_dev, out_ref, kAtol, kRtol, "gdn single-token chunk output");
    ocr::test::expect_close(state_dev, state_ref, kAtol, kRtol, "gdn single-token chunk state");
    // NaN in v propagates (no OOB / crash).
    auto in_nan = make_inputs(lcg, 3, heads, d);
    in_nan.v[0] = std::numeric_limits<float>::quiet_NaN();
    luisa::vector<float> state_ref_nan = in_nan.state, out_ref_nan;
    LUISA_ASSERT(ref_gdn_chunk(in_nan.q, in_nan.k, in_nan.v, in_nan.beta, in_nan.g, state_ref_nan, 3, heads, d, out_ref_nan).ok(), "ref nan");
    luisa::vector<float> out_dev_nan, state_dev_nan;
    device_chunk(device, stream, in_nan, out_dev_nan, state_dev_nan);
    ocr::test::expect_close(out_dev_nan, out_ref_nan, kAtol, kRtol, "gdn chunk NaN propagation");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "gdn_chunk_oracle"_test = [&] { test_chunk_oracle(device); };
    "gdn_chunk_vs_recurrent_boundaries"_test = [&] { test_chunk_vs_recurrent_boundaries(device); };
    "gdn_chunk_degenerate_nan"_test = [&] { test_chunk_degenerate_nan(device); };
}
