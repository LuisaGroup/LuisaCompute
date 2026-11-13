// Test for ocr/ops/ops_gdn -- causal depthwise Conv1D + SiLU + ring state.
// This test covers:
// - device conv+silu+ring vs the independent double-accumulation CPU oracle;
// - Conv ring state AFTER prefill (exact last K-1 pre-activation taps);
// - chunked conv continuity (ring carried across chunks == one-shot);
// - kernel sizes 2/3/4, ragged token counts, degenerate + NaN inputs.
// Synthetic data only; no model/download.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include <cmath>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/ops/ops_gdn.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 2e-3f;
constexpr float kRtol = 2e-3f;

// Run the device conv+silu+ring kernel over one chunk; returns out + updated ring.
void device_conv(Device &device, Stream &stream, luisa::span<const float> src,
                 luisa::span<const float> weight, luisa::span<const float> bias,
                 luisa::span<const float> ring_in, uint32_t tokens, uint32_t channels,
                 uint32_t K, uint32_t pos0, luisa::vector<float> &out, luisa::vector<float> &ring_out) {
    auto src_buf = ocr::test::upload(device, stream, src);
    auto w_buf = ocr::test::upload(device, stream, weight);
    auto b_buf = ocr::test::upload(device, stream, bias);
    auto ring_buf = ocr::test::upload(device, stream, ring_in);
    auto out_buf = device.create_buffer<float>(static_cast<size_t>(tokens) * channels);
    auto shader = device.compile(make_gdn_conv_silu_kernel());
    auto plan = plan_gdn_conv_dispatch(channels);
    LUISA_ASSERT(plan.ok(), "conv plan");
    stream << src_buf.copy_from(src) << w_buf.copy_from(weight) << b_buf.copy_from(bias)
           << ring_buf.copy_from(ring_in);
    if (auto cmd = dispatch_1d(shader(src_buf, w_buf, b_buf, ring_buf, out_buf, tokens, channels, K, pos0), *plan.value)) {
        stream << std::move(cmd);
    }
    out.resize(static_cast<size_t>(tokens) * channels);
    ring_out.resize(ring_in.size());
    stream << out_buf.copy_to(luisa::span<float>{out})
           << ring_buf.copy_to(luisa::span<float>{ring_out})
           << synchronize();
}

void test_conv_oracle(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    for (uint32_t K : {2u, 3u, 4u}) {
        for (uint32_t tokens : {1u, 5u, 7u}) {
            const uint32_t channels = 6;
            luisa::vector<float> src(static_cast<size_t>(tokens) * channels);
            luisa::vector<float> weight(static_cast<size_t>(channels) * K);
            luisa::vector<float> bias(channels);
            luisa::vector<float> ring(static_cast<size_t>(channels) * (K - 1u), 0.0f);
            lcg.fill(src, -1.0f, 1.0f);
            lcg.fill(weight, -0.5f, 0.5f);
            lcg.fill(bias, -0.2f, 0.2f);
            // Oracle arm.
            luisa::vector<float> ring_ref = ring;
            luisa::vector<float> out_ref;
            auto s = ref_gdn_conv_silu(src, weight, bias, ring_ref, tokens, channels, K, 0u, out_ref);
            LUISA_ASSERT(s.ok(), "ref conv");
            // Device arm.
            luisa::vector<float> out_dev, ring_dev;
            device_conv(device, stream, src, weight, bias, ring, tokens, channels, K, 0u, out_dev, ring_dev);
            ocr::test::expect_close(out_dev, out_ref, kAtol, kRtol, "gdn conv+silu output");
            // Ring after prefill == last K-1 pre-activation taps per channel.
            ocr::test::expect_close(ring_dev, ring_ref, 0.0f, 0.0f, "gdn conv ring state after prefill");
            // Explicit ring check: slot s holds src at absolute position p with p%(K-1)==s.
            bool ring_ok = true;
            for (uint32_t c = 0; c < channels; c++) {
                for (uint32_t s = 0; s < K - 1u; s++) {
                    // The most recent position p < tokens with p%(K-1)==s.
                    int latest = -1;
                    for (uint32_t p = 0; p < tokens; p++) {
                        if (p % (K - 1u) == s) { latest = static_cast<int>(p); }
                    }
                    const float want = latest >= 0 ? src[static_cast<size_t>(latest) * channels + c] : 0.0f;
                    if (ring_dev[static_cast<size_t>(c) * (K - 1u) + s] != want) { ring_ok = false; }
                }
            }
            expect(ring_ok) << "ring slots hold the latest pre-activation taps";
        }
    }
}

void test_conv_chunk_continuity(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    const uint32_t K = 4, channels = 5, total = 11;
    luisa::vector<float> src(static_cast<size_t>(total) * channels);
    luisa::vector<float> weight(static_cast<size_t>(channels) * K);
    luisa::vector<float> bias(channels);
    lcg.fill(src, -1.0f, 1.0f);
    lcg.fill(weight, -0.5f, 0.5f);
    lcg.fill(bias, -0.2f, 0.2f);
    // One-shot oracle.
    luisa::vector<float> ring_ref(static_cast<size_t>(channels) * (K - 1u), 0.0f);
    luisa::vector<float> out_ref;
    LUISA_ASSERT(ref_gdn_conv_silu(src, weight, bias, ring_ref, total, channels, K, 0u, out_ref).ok(), "ref");
    // Device chunked: chunks of 3 (nondivisible), carrying the ring.
    luisa::vector<float> ring(static_cast<size_t>(channels) * (K - 1u), 0.0f);
    luisa::vector<float> out_chunked;
    uint32_t pos0 = 0;
    while (pos0 < total) {
        const uint32_t rows = std::min(3u, total - pos0);
        luisa::vector<float> chunk_src(src.begin() + static_cast<size_t>(pos0) * channels,
                                       src.begin() + static_cast<size_t>(pos0 + rows) * channels);
        luisa::vector<float> out_c, ring_next;
        device_conv(device, stream, chunk_src, weight, bias, ring, rows, channels, K, pos0, out_c, ring_next);
        out_chunked.insert(out_chunked.end(), out_c.begin(), out_c.end());
        ring = ring_next;
        pos0 += rows;
    }
    ocr::test::expect_close(out_chunked, out_ref, kAtol, kRtol, "gdn conv chunked == one-shot output");
    ocr::test::expect_close(ring, ring_ref, 0.0f, 0.0f, "gdn conv chunked == one-shot ring");
}

void test_conv_degenerate_nan(Device &device) {
    auto stream = device.create_stream();
    const uint32_t K = 3, channels = 4, tokens = 4;
    luisa::vector<float> src(static_cast<size_t>(tokens) * channels, 0.0f);
    luisa::vector<float> weight(static_cast<size_t>(channels) * K, 0.0f);
    luisa::vector<float> bias(channels, 0.0f);
    luisa::vector<float> ring(static_cast<size_t>(channels) * (K - 1u), 0.0f);
    // All-zero input -> silu(bias)=silu(0)=0; ring stays zero.
    luisa::vector<float> out_ref, ring_ref = ring;
    LUISA_ASSERT(ref_gdn_conv_silu(src, weight, bias, ring_ref, tokens, channels, K, 0u, out_ref).ok(), "ref zero");
    luisa::vector<float> out_dev, ring_dev;
    device_conv(device, stream, src, weight, bias, ring, tokens, channels, K, 0u, out_dev, ring_dev);
    ocr::test::expect_close(out_dev, out_ref, 1e-6f, 1e-6f, "gdn conv zero input");
    // NaN input propagates as NaN (no OOB / crash); oracle and device agree.
    src[0] = std::numeric_limits<float>::quiet_NaN();
    luisa::vector<float> out_ref_nan, ring_ref_nan = ring;
    LUISA_ASSERT(ref_gdn_conv_silu(src, weight, bias, ring_ref_nan, tokens, channels, K, 0u, out_ref_nan).ok(), "ref nan");
    luisa::vector<float> out_dev_nan, ring_dev_nan;
    device_conv(device, stream, src, weight, bias, ring, tokens, channels, K, 0u, out_dev_nan, ring_dev_nan);
    ocr::test::expect_close(out_dev_nan, out_ref_nan, kAtol, kRtol, "gdn conv NaN propagation");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "gdn_conv_oracle"_test = [&] { test_conv_oracle(device); };
    "gdn_conv_chunk_continuity"_test = [&] { test_conv_chunk_continuity(device); };
    "gdn_conv_degenerate_nan"_test = [&] { test_conv_degenerate_nan(device); };
}
