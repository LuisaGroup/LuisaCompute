// Test for mixed-precision dot2add DSL execution.
// This test covers:
// - half2 buffer inputs with a float accumulator and result
// - numerically exact cases shared by fused and unfused lowerings
// - explicit skipping on backends without a dot2add implementation

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    if (device.backend_name() != "dx" && device.backend_name() != "vk") {
        LUISA_INFO("Skipping dot2add runtime test on unsupported backend '{}'.",
                   device.backend_name());
        return 0;
    }

    constexpr auto count = 16u;
    luisa::vector<half2> a(count);
    luisa::vector<half2> b(count);
    luisa::vector<float> acc(count);
    luisa::vector<float> expected(count);
    for (auto i = 0u; i < count; ++i) {
        auto v = static_cast<float>(i + 1u);
        a[i] = half2{half{0.25f * v}, half{-0.5f}};
        b[i] = half2{half{0.5f}, half{0.125f * v}};
        acc[i] = 0.5f;
        expected[i] = 0.125f * v - 0.0625f * v + 0.5f;
    }

    auto buffer_a = device.create_buffer<half2>(count);
    auto buffer_b = device.create_buffer<half2>(count);
    auto buffer_acc = device.create_buffer<float>(count);
    auto buffer_out = device.create_buffer<float>(count);
    auto stream = device.create_stream();
    Kernel1D kernel = [](BufferHalf2 a, BufferHalf2 b,
                         BufferFloat acc, BufferFloat out) noexcept {
        auto i = dispatch_x();
        out.write(i, dot2add(a.read(i), b.read(i), acc.read(i)));
    };
    auto shader = device.compile(kernel);
    luisa::vector<float> result(count);
    stream << buffer_a.copy_from(luisa::span{a})
           << buffer_b.copy_from(luisa::span{b})
           << buffer_acc.copy_from(luisa::span{acc})
           << shader(buffer_a, buffer_b, buffer_acc, buffer_out).dispatch(count)
           << buffer_out.copy_to(luisa::span{result})
           << synchronize();

    auto correct = true;
    for (auto i = 0u; i < count; ++i) {
        if (std::abs(result[i] - expected[i]) > 1e-6f) {
            LUISA_WARNING("dot2add mismatch at {}: got {}, expected {}.",
                          i, result[i], expected[i]);
            correct = false;
        }
    }
    expect(correct) << "dot2add results must match the exact host oracle";
}
