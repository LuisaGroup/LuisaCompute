// Deterministic shader-printer test.
//
// The stream callback is the observable printer output. This test checks the
// exact scalar messages and the formatter representation of vectors, matrices,
// and user structures without relying on a human to inspect stdout.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string_view>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct MyStruct {
    float2 a;
    uint2 b;
};

LUISA_STRUCT(MyStruct, a, b) {};

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
    std::sort(messages.begin(), messages.end());

    expect(static_cast<bool>(messages.size() == 5u))
        << "four scalar messages and one composite message must reach the callback";
    for (auto id = 0u; id < 4u; id++) {
        auto expected = luisa::format("printer-id={}", id);
        expect(static_cast<bool>(std::find(messages.begin(), messages.end(), expected) !=
                                 messages.end()))
            << "the printer must preserve every dispatched scalar value";
    }

    auto composite = std::find_if(
        messages.begin(), messages.end(), [](auto &&message) noexcept {
            return std::string_view{message}.starts_with("printer-composite ");
        });
    expect(static_cast<bool>(composite != messages.end()))
        << "the composite printer message must reach the callback";
    if (composite != messages.end()) {
        expect(static_cast<bool>(
            *composite ==
            "printer-composite vector=(-1, 2, 3) matrix=<(1, 2), (3, 4)> struct={(0.25, 0.5), (7, 9)}"))
            << "vector, matrix, and user-structure formatting must match the canonical printer syntax";
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    auto &device = dc->device;
    test_printer(device);
}
