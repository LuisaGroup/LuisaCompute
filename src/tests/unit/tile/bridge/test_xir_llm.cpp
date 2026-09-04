#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_llm_test_utils.h"
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

void run(Device &device, const test::tile_llm::Case &fixture) {
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
    std::fill(output.begin() + pad, output.end() - pad, std::numeric_limits<float>::quiet_NaN());
    auto d = device.create_buffer<float>(output.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.copy_from(fixture.inputs[0].data()) << b.copy_from(fixture.inputs[1].data()) << c.copy_from(fixture.inputs[2].data())
           << d.copy_from(output.data()) << shader(a, b, c, d.view(pad, fixture.expected.size())).dispatch()
           << d.copy_to(output.data()) << synchronize();
    check(span{output}.subspan(pad, fixture.expected.size()), fixture.expected);
    expect(std::all_of(output.begin(), output.begin() + pad, [](float x) { return x == guard; }));
    expect(std::all_of(output.end() - pad, output.end(), [](float x) { return x == guard; }));
#ifdef LUISA_TEST_TILE_XIR_TIRX
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
#endif
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
    using test::tile_llm::RowOp;
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
}
