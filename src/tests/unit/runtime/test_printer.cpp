// Deterministic shader-printer test.
//
// The stream callback is the observable printer output. This test checks the
// exact scalar messages and the formatter representation of vectors, matrices,
// and user structures without relying on a human to inspect stdout.
//
// Matrices are covered on purpose: the printer payload follows the Luisa type
// layout (matrix columns stored back to back, each padded to its vector size),
// which is what the host formatter decodes. A backend must never let its
// compiler's aggregate packing decide those bytes, so the 3x3 (padded column),
// 4x4, matrix-inside-structure, and matrix-inside-array cases are all pinned
// here.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/core/stl/algorithm.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct MyStruct {
    float2 a;
    uint2 b;
};

LUISA_STRUCT(MyStruct, a, b) {};

struct MatrixHolder {
    float3x3 m;
    float2 tail;
};

LUISA_STRUCT(MatrixHolder, m, tail) {};

struct MatrixArrayHolder {
    float2x2 m[2];
    float2 tail;
};

LUISA_STRUCT(MatrixArrayHolder, m, tail) {};

namespace {

struct CapturedMessages {
    std::mutex mutex;
    luisa::vector<luisa::string> messages;
};

}// namespace

void test_printer(Device &device) {
    auto captured = std::make_shared<CapturedMessages>();
    auto stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        std::scoped_lock lock{captured->mutex};
        captured->messages.emplace_back(message);
    });

    Kernel1D kernel = []() noexcept {
        auto id = dispatch_x();
        device_log("printer-id={}", id);
        $if (id == 0u) {
            Var<MyStruct> value;
            value.a = make_float2(0.25f, 0.5f);
            value.b = make_uint2(7u, 9u);
            device_log("printer-composite vector={} matrix={} struct={}",
                       make_int3(-1, 2, 3),
                       make_float2x2(1.0f, 2.0f, 3.0f, 4.0f),
                       value);
            Var<MatrixHolder> holder;
            holder.m = make_float3x3(1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f);
            holder.tail = make_float2(11.0f, 12.0f);
            device_log("printer-layout matrix3={} matrix4={} struct={}",
                       make_float3x3(1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f),
                       make_float4x4(1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f,
                                     9.0f, 10.0f, 11.0f, 12.0f,
                                     13.0f, 14.0f, 15.0f, 16.0f),
                       holder);
            Var<MatrixArrayHolder> array;
            array.m[0] = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
            array.m[1] = make_float2x2(5.0f, 6.0f, 7.0f, 8.0f);
            array.tail = make_float2(9.0f, 10.0f);
            device_log("printer-array struct={}", array);
        };
    };
    auto shader = device.compile(kernel);
    stream << shader().dispatch(4u)
           << synchronize();

    luisa::vector<luisa::string> messages;
    {
        std::scoped_lock lock{captured->mutex};
        messages = captured->messages;
    }
    luisa::sort(messages.begin(), messages.end());

    expect(static_cast<bool>(messages.size() == 7u))
        << "four scalar messages and three composite messages must reach the callback";
    for (auto id = 0u; id < 4u; id++) {
        auto expected = luisa::format("printer-id={}", id);
        expect(static_cast<bool>(std::find(messages.begin(), messages.end(), expected) !=
                                 messages.end()))
            << "the printer must preserve every dispatched scalar value";
    }

    auto find_message = [&messages](luisa::string_view prefix) noexcept {
        return std::find_if(
            messages.begin(), messages.end(), [prefix](auto &&message) noexcept {
                return luisa::string_view{message}.starts_with(prefix);
            });
    };
    auto expect_message = [&find_message, &messages](luisa::string_view prefix,
                                                     luisa::string_view expected,
                                                     luisa::string_view description) noexcept {
        auto message = find_message(prefix);
        expect(static_cast<bool>(message != messages.end()))
            << description;
        if (message != messages.end()) {
            expect(static_cast<bool>(*message == expected)) << description;
        }
    };

    expect_message(
        "printer-composite ",
        "printer-composite vector=(-1, 2, 3) matrix=<(1, 2), (3, 4)> struct={(0.25, 0.5), (7, 9)}",
        "vector, matrix, and user-structure formatting must match the canonical printer syntax");
    expect_message(
        "printer-layout ",
        "printer-layout matrix3=<(1, 2, 3), (4, 5, 6), (7, 8, 9)> "
        "matrix4=<(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)> "
        "struct={<(1, 2, 3), (4, 5, 6), (7, 8, 9)>, (11, 12)}",
        "matrix padding and nested matrix members must match the canonical printer syntax");
    expect_message(
        "printer-array ",
        "printer-array struct={[<(1, 2), (3, 4)>, <(5, 6), (7, 8)>], (9, 10)}",
        "matrix members nested in an array must match the canonical printer syntax");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    auto &device = dc->device;
    test_printer(device);
}
