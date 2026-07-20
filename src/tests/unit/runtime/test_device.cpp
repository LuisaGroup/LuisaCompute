// Test runtime device creation and shared-handle wrapping.
//
// The wrapped Device must remain usable after the original Device object is
// released. A deterministic buffer transformation verifies that the retained
// backend handle can still create resources, compile, dispatch, and read back.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <array>
#include <utility>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

void test_wrapped_device(Device device) {
    expect(static_cast<bool>(device)) << "context must create a valid device";
    auto *implementation = device.impl();
    auto backend_name = luisa::string{device.backend_name()};
    expect(implementation != nullptr) << "valid device must expose an implementation";
    expect(!backend_name.empty()) << "valid device must report its backend name";

    Device wrapped{device.impl_shared()};
    expect(wrapped.impl() == implementation)
        << "wrapping a shared backend handle must preserve the implementation";
    expect(wrapped.backend_name() == backend_name)
        << "wrapped device must preserve the backend identity";

    // Drop the original owner before exercising the wrapper. This is the
    // lifetime behavior the old test_wrapped_device stub was meant to cover.
    device = Device{};
    expect(static_cast<bool>(wrapped))
        << "wrapped device must retain the shared backend implementation";

    constexpr auto element_count = 17u;
    std::array<uint, element_count> input{};
    std::array<uint, element_count> output{};
    for (auto i = 0u; i < element_count; i++) {
        input[i] = 0x12345678u ^ (i * 0x9e3779b9u);
    }

    auto buffer = wrapped.create_buffer<uint>(element_count);
    Kernel1D transform = [](BufferUInt values) noexcept {
        auto index = dispatch_x();
        auto value = values.read(index);
        values.write(index, value * 1664525u + 1013904223u);
    };
    auto shader = wrapped.compile(transform);
    auto stream = wrapped.create_stream();
    stream << buffer.copy_from(luisa::span{input})
           << shader(buffer).dispatch(element_count)
           << buffer.copy_to(luisa::span{output})
           << synchronize();

    size_t mismatch_count = 0u;
    for (auto i = 0u; i < element_count; i++) {
        auto expected = input[i] * 1664525u + 1013904223u;
        if (output[i] != expected) {
            if (mismatch_count == 0u) {
                LUISA_WARNING("Wrapped-device mismatch at {}: expected {}, got {}.",
                              i, expected, output[i]);
            }
            mismatch_count++;
        }
    }
    expect(mismatch_count == 0u)
        << luisa::format("wrapped device produced {} incorrect values", mismatch_count);
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    test_wrapped_device(std::move(dc->device));
}
