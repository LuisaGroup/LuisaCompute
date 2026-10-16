#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_llm_test_utils.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>
#include <limits>

#ifdef LUISA_TEST_TILE_XIR_TIRX
#include "tile_tirx_test_utils.h"
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

void check(span<const float> actual, span<const double> expected) {
    expect(eq(actual.size(), expected.size()));
    for (size_t i = 0u; i < expected.size(); i++) {
        expect(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= 5e-5 + 5e-5 * std::abs(expected[i])) << "element " << i << " actual " << actual[i] << " expected " << expected[i];
    }
}

void run(Device &device, const test::tile_llm::Case &fixture, bool compare_tirx = true, bool uniform_attention_pattern = false) {
    LUISA_INFO("Checking {} with {} output elements", fixture.kernel.function().name(), fixture.expected.size());
    expect(fixture.kernel.valid());
    auto shader = tile::compile(device, fixture.kernel);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto a = device.create_buffer<float>(fixture.inputs[0].size());
    auto b = device.create_buffer<float>(fixture.inputs[1].size());
    auto c = device.create_buffer<float>(fixture.inputs[2].size());
    constexpr size_t pad = 17u;
    constexpr auto guard = -719.5f;
    vector<float> output(fixture.expected.size() + 2u * pad, guard);
    auto d = device.create_buffer<float>(output.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto dispatch = [&](span<const float> query, span<const float> key, span<const float> value, span<const double> expected) {
        std::fill(output.begin(), output.end(), guard);
        std::fill(output.begin() + pad, output.end() - pad, std::numeric_limits<float>::quiet_NaN());
        stream << a.copy_from(query) << b.copy_from(key) << c.copy_from(value)
               << d.copy_from(span{output}) << shader(a, b, c, d.view(pad, fixture.expected.size())).dispatch()
               << d.copy_to(span{output}) << synchronize();
        check(span{output}.subspan(pad, fixture.expected.size()), expected);
        expect(std::all_of(output.begin(), output.begin() + pad, [](float x) { return x == guard; }));
        expect(std::all_of(output.end() - pad, output.end(), [](float x) { return x == guard; }));
    };
    dispatch(fixture.inputs[0], fixture.inputs[1], fixture.inputs[2], fixture.expected);
    if (uniform_attention_pattern) {
        // Reuse the compiled kernel and allocations with an independent,
        // closed-form causal oracle: zero Q/K gives a uniform visible prefix.
        // Signed, nonconstant V exposes masking, head mapping and tail errors.
        LUISA_ASSERT(fixture.shapes[0].size() == 4u && fixture.shapes[1].size() == 4u && fixture.shapes[2].size() == 4u,
                     "Uniform attention pattern requires an attention fixture");
        vector<float> query(fixture.inputs[0].size(), 0.0f), key(fixture.inputs[1].size(), 0.0f), value(fixture.inputs[2].size());
        for (size_t i = 0u; i < value.size(); i++) { value[i] = static_cast<float>(static_cast<int64_t>((i * 17u + 3u) % 29u) - 14) * 0.125f; }
        auto batches = fixture.shapes[0][0], heads = fixture.shapes[0][1], queries = fixture.shapes[0][2];
        auto kv_heads = fixture.shapes[2][1], keys = fixture.shapes[2][2], channels = fixture.shapes[2][3];
        vector<double> expected(fixture.expected.size());
        for (int64_t batch = 0; batch < batches; batch++) {
            for (int64_t head = 0; head < heads; head++) {
                auto kv_head = head / (heads / kv_heads);
                for (int64_t q = 0; q < queries; q++) {
                    auto visible = keys - queries + q + 1;
                    for (int64_t channel = 0; channel < channels; channel++) {
                        auto sum = 0.0;
                        for (int64_t k = 0; k < visible; k++) { sum += value[((batch * kv_heads + kv_head) * keys + k) * channels + channel]; }
                        expected[((batch * heads + head) * queries + q) * channels + channel] = sum / static_cast<double>(visible);
                    }
                }
            }
        }
        LUISA_INFO("Checking uniform causal attention pattern on the same shader");
        dispatch(query, key, value, expected);
    }
#ifdef LUISA_TEST_TILE_XIR_TIRX
    if (!compare_tirx) { return; }
    test::tile_tirx::Runtime runtime{"cpu", true};
    auto executable = runtime.build(fixture.kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto upload = [&](size_t i) {
        auto &s = fixture.shapes[i];
        return s.size() == 2u ? runtime.upload<float>({s[0], s[1]}, fixture.inputs[i]) : runtime.upload<float>({s[0], s[1], s[2], s[3]}, fixture.inputs[i]);
    };
    auto ta = upload(0), tb = upload(1), tc = upload(2);
    auto &s = fixture.shapes[3];
    auto td = s.size() == 2u ? runtime.allocate<float>({s[0], s[1]}) : runtime.allocate<float>({s[0], s[1], s[2], s[3]});
    (*executable.entry)(ta, tb, tc, td);
    check(runtime.download<float>(td, fixture.expected.size()), fixture.expected);
#else
    static_cast<void>(compare_tirx);
#endif
}

}// namespace

int main(int argc, char *argv[]) {
    // CTest runs fatal fixture checks in fresh processes, before any device or
    // worker pool exists. The wrapper requires the expected diagnostic too.
    if (argc == 2 && string_view{argv[1]} == "--reject-attention-heads") {
        static_cast<void>(test::tile_llm::attention(1, 0, 1, 1, 1, 1, 1));
        return 0;// Unexpected acceptance makes the wrapper fail.
    }
    if (argc == 2 && string_view{argv[1]} == "--reject-attention-block") {
        static_cast<void>(test::tile_llm::attention(1, 1, 1, 1, 1, 1, 1, 0, 3));
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
    using test::tile_llm::RowOp;
    "tile_xir_llm_large_private_workspace"_test = [&] {
        // More than one block forces execution on the persistent worker pool,
        // whose native stack is smaller than the caller's stack on macOS.
        run(device, test::tile_llm::rows(RowOp::RMS_NORM, 64, 16384), false);
        run(device, test::tile_llm::rows(RowOp::MASKED_SOFTMAX, 64, 4096), false);
        run(device, test::tile_llm::rows(RowOp::MASKED_SOFTMAX, 64, 16384), false);
    };
    "tile_xir_llm_normalization_activation_masked_softmax"_test = [&] {
        for (auto op : {RowOp::RMS_NORM, RowOp::LAYER_NORM, RowOp::SWIGLU, RowOp::GELU_RESIDUAL, RowOp::MASKED_SOFTMAX}) {
            for (auto width : {7, 32, 65}) { run(device, test::tile_llm::rows(op, 17, width)); }
        }
    };
    "tile_xir_llm_rope"_test = [&] {
        for (auto width : {6, 32, 66}) { run(device, test::tile_llm::rows(RowOp::ROPE, 17, width)); }
    };
    "tile_xir_llm_online_prefill_decode_gqa"_test = [&] {
        run(device, test::tile_llm::attention(1, 2, 2, 4, 5, 4, 3));
        run(device, test::tile_llm::attention(2, 4, 2, 7, 11, 8, 7));
        run(device, test::tile_llm::attention(2, 4, 2, 1, 17, 8, 7));
    };
    "tile_xir_llm_attention_block_shapes"_test = [&] {
        // Keep unit-test SSA expansion bounded. Larger physical tiles belong
        // to the benchmark's explicit compile-time/timeout evidence.
        for (auto block : {std::array<int64_t, 2>{1, 1}, {2, 4}, {3, 5}}) {
            run(device, test::tile_llm::attention(1, 2, 1, 7, 11, 8, 7, block[0], block[1]));
        }
    };
    "tile_xir_llm_attention_qk_reduction_probe"_test = [&] {
        // Independent FP64 oracle, including reduction and output tails.
        run(device, test::tile_llm::attention(1, 2, 1, 4, 5, 7, 3, 2, 3, true));
        run(device, test::tile_llm::attention(1, 2, 1, 1, 17, 33, 7, 1, 4, true));
    };
    "tile_xir_llm_attention_pv_reduction_probe"_test = [&] {
        // Prefill and ragged decode use the independent FP64 oracle and output
        // guards, not bitwise comparison with a differently ordered MMA path.
        for (auto qk_reduction : {false, true}) {
            LUISA_INFO("Checking attention PV reduction probe with QK={}", qk_reduction ? "reduce" : "mma");
            run(device, test::tile_llm::attention(1, 2, 1, 4, 5, 7, 3, 2, 3, qk_reduction, true));
            run(device, test::tile_llm::attention(1, 2, 1, 1, 17, 33, 7, 1, 4, qk_reduction, true));
        }
    };
    "tile_xir_llm_attention_decomposition_matrix"_test = [&] {
        // Twelve bounded compile configurations, each with two input patterns.
        // Q == K exercises a true triangular prefill, including fully masked
        // key blocks for early queries. The other cases cover MQA/GQA, batches and
        // both query/key tails without large benchmark-sized compile graphs.
        for (auto qk_reduction : {false, true}) {
            for (auto pv_reduction : {false, true}) {
                LUISA_INFO("Checking attention decomposition QK={} PV={}", qk_reduction ? "reduce" : "mma", pv_reduction ? "reduce" : "mma");
                run(device, test::tile_llm::attention(1, 2, 2, 5, 5, 5, 3, 2, 3, qk_reduction, pv_reduction), false, true);
                run(device, test::tile_llm::attention(2, 4, 1, 3, 7, 7, 5, 2, 3, qk_reduction, pv_reduction), false, true);
                run(device, test::tile_llm::attention(2, 4, 2, 1, 9, 5, 3, 1, 4, qk_reduction, pv_reduction), false, true);
            }
        }
    };
}
