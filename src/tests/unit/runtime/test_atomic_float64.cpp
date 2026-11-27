// Test for float64 buffer atomics.
//
// This test covers:
// - the bit-exact float64 buffer access every backend must preserve, including
//   the DirectX route that stores a scalar float64 element as a 64-bit word
// - atomic exchange, add, sub, min and max on a float64 buffer element
// - the fail-closed boundaries: backends without a legal float64 lowering must
//   reject the kernel instead of emitting an invalid instruction
//
// A backend is expected to fail closed (a fatal diagnostic naming the missing
// capability) rather than miscompile, so the backend-specific cases are gated
// to the backend whose lowering they exercise.
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto float64_atomic_element_count = 4096u;

[[nodiscard]] bool is_float64_atomic_backend(Device &device) noexcept {
    auto backend = device.backend_name();
    // The software (fallback) and SIMD backends do not implement float64
    // atomics: they reject the value type, so running these cases there would
    // only exercise their fatal diagnostic.
    return backend == "cuda" || backend == "dx" || backend == "vk";
}

void test_float64_buffer_is_bit_exact(Device &device) {
    // The DirectX route declares a scalar float64 buffer element as uint64_t
    // and bit-casts every access. These payloads pin that the bit cast is
    // exact: a numeric conversion would turn -0.0 into +0.0, flush a denormal,
    // or re-round the transcendental result.
    auto values = device.create_buffer<double>(8u);
    auto copied = device.create_buffer<double>(8u);
    auto host = luisa::vector<double>{
        0.0,
        -0.0,
        1.4142135623730951,// sqrt(2), the double value
        -2.5,
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::denorm_min(),
        1.0000000000000002};
    Kernel1D copy_kernel = [](BufferVar<double> src, BufferVar<double> dst) noexcept {
        auto index = dispatch_id().x;
        dst.write(index, src.read(index));
    };
    auto shader = device.compile(copy_kernel);
    auto stream = device.create_stream();
    auto readback = luisa::vector<double>(8u, 12345.0);
    stream << values.copy_from(luisa::span{host.data(), host.size()})
           << shader(values, copied).dispatch(8u)
           << copied.copy_to(luisa::span{readback.data(), readback.size()})
           << synchronize();
    for (size_t i = 0u; i < host.size(); i++) {
        expect(std::memcmp(&readback[i], &host[i], sizeof(double)) == 0)
            << "float64 buffer element " << i << " was not copied bit-exactly";
    }
}

void test_float64_atomic_fetch_add(Device &device) {
    // Every element is updated by one invocation, so the sum is exact and a
    // lost or duplicated update is visible.
    auto values = device.create_buffer<double>(float64_atomic_element_count);
    auto host = luisa::vector<double>(float64_atomic_element_count, 0.0);
    Kernel1D add_kernel = [](BufferVar<double> target) noexcept {
        target.atomic(dispatch_id().x).fetch_add(1.0);
    };
    auto shader = device.compile(add_kernel);
    auto stream = device.create_stream();
    stream << values.copy_from(luisa::span{host.data(), host.size()})
           << shader(values).dispatch(float64_atomic_element_count)
           << values.copy_to(luisa::span{host.data(), host.size()})
           << synchronize();
    for (auto i = 0u; i < float64_atomic_element_count; i++) {
        expect(host[i] == 1.0)
            << "float64 atomic add lost an update at element " << i;
    }
}

void test_float64_atomic_arithmetic(Device &device) {
    // add/sub/min/max on the same element, in one dispatch per operation, plus
    // the returned old value.
    auto value = device.create_buffer<double>(1u);
    auto old_value = device.create_buffer<double>(1u);
    auto host_value = luisa::vector<double>{0.5};
    auto host_old = luisa::vector<double>{0.0};
    Kernel1D add_kernel = [](BufferVar<double> target, BufferVar<double> old) noexcept {
        old.write(0u, target.atomic(0u).fetch_add(1.25));
    };
    Kernel1D sub_kernel = [](BufferVar<double> target, BufferVar<double> old) noexcept {
        old.write(0u, target.atomic(0u).fetch_sub(0.25));
    };
    Kernel1D min_kernel = [](BufferVar<double> target, BufferVar<double> old) noexcept {
        old.write(0u, target.atomic(0u).fetch_min(0.75));
    };
    Kernel1D max_kernel = [](BufferVar<double> target, BufferVar<double> old) noexcept {
        old.write(0u, target.atomic(0u).fetch_max(4.0));
    };
    auto add = device.compile(add_kernel);
    auto sub = device.compile(sub_kernel);
    auto min = device.compile(min_kernel);
    auto max = device.compile(max_kernel);
    auto stream = device.create_stream();
    stream << value.copy_from(luisa::span{host_value.data(), 1u})
           << old_value.copy_from(luisa::span{host_old.data(), 1u})
           << add(value, old_value).dispatch(1u)
           << value.copy_to(luisa::span{host_value.data(), 1u})
           << old_value.copy_to(luisa::span{host_old.data(), 1u})
           << synchronize();
    expect(host_old[0] == 0.5) << "float64 fetch_add must return the old value";
    expect(host_value[0] == 1.75) << "float64 fetch_add result is wrong";

    stream << sub(value, old_value).dispatch(1u)
           << value.copy_to(luisa::span{host_value.data(), 1u})
           << old_value.copy_to(luisa::span{host_old.data(), 1u})
           << synchronize();
    expect(host_old[0] == 1.75) << "float64 fetch_sub must return the old value";
    expect(host_value[0] == 1.5) << "float64 fetch_sub result is wrong";

    stream << min(value, old_value).dispatch(1u)
           << value.copy_to(luisa::span{host_value.data(), 1u})
           << synchronize();
    expect(host_value[0] == 0.75) << "float64 fetch_min result is wrong";

    stream << max(value, old_value).dispatch(1u)
           << value.copy_to(luisa::span{host_value.data(), 1u})
           << synchronize();
    expect(host_value[0] == 4.0) << "float64 fetch_max result is wrong";
}

void test_float64_atomic_exchange(Device &device) {
    auto value = device.create_buffer<double>(1u);
    auto old_value = device.create_buffer<double>(1u);
    auto host_value = luisa::vector<double>{-1.5};
    auto host_old = luisa::vector<double>{0.0};
    Kernel1D exchange_kernel = [](BufferVar<double> target, BufferVar<double> old) noexcept {
        old.write(0u, target.atomic(0u).exchange(2.25));
    };
    auto shader = device.compile(exchange_kernel);
    auto stream = device.create_stream();
    stream << value.copy_from(luisa::span{host_value.data(), 1u})
           << old_value.copy_from(luisa::span{host_old.data(), 1u})
           << shader(value, old_value).dispatch(1u)
           << value.copy_to(luisa::span{host_value.data(), 1u})
           << old_value.copy_to(luisa::span{host_old.data(), 1u})
           << synchronize();
    expect(host_old[0] == -1.5) << "float64 exchange must return the old value";
    expect(host_value[0] == 2.25) << "float64 exchange result is wrong";
}

void test_float64_atomic_compare_exchange(Device &device) {
    // The expected operand is the value the kernel just loaded, so the
    // success/failure arms do not depend on how a float64 literal is folded.
    auto value = device.create_buffer<double>(1u);
    auto old_success = device.create_buffer<double>(1u);
    auto old_failure = device.create_buffer<double>(1u);
    auto host_value = luisa::vector<double>{1.5};
    auto host_success = luisa::vector<double>{0.0};
    auto host_failure = luisa::vector<double>{0.0};
    Kernel1D compare_kernel = [](BufferVar<double> target,
                                 BufferVar<double> old_ok,
                                 BufferVar<double> old_fail) noexcept {
        auto current = target.read(0u);
        auto success = target.atomic(0u).compare_exchange(current, 3.5);
        auto failure = target.atomic(0u).compare_exchange(current, 9.0);
        old_ok.write(0u, success);
        old_fail.write(0u, failure);
    };
    auto shader = device.compile(compare_kernel);
    auto stream = device.create_stream();
    stream << value.copy_from(luisa::span{host_value.data(), 1u})
           << old_success.copy_from(luisa::span{host_success.data(), 1u})
           << old_failure.copy_from(luisa::span{host_failure.data(), 1u})
           << shader(value, old_success, old_failure).dispatch(1u)
           << value.copy_to(luisa::span{host_value.data(), 1u})
           << old_success.copy_to(luisa::span{host_success.data(), 1u})
           << old_failure.copy_to(luisa::span{host_failure.data(), 1u})
           << synchronize();
    expect(host_success[0] == 1.5)
        << "float64 compare-exchange must return the value it replaced";
    expect(host_failure[0] == 3.5)
        << "a mismatched float64 compare-exchange must return the stored value";
    expect(host_value[0] == 3.5)
        << "float64 compare-exchange result is wrong";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    luisa::vector<const char *> test_args{argv[0]};
    for (auto i = 2; i < argc; ++i) { test_args.emplace_back(argv[i]); }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        static_cast<int>(test_args.size()), test_args.data());

    auto &device = dc->device;
    if (!is_float64_atomic_backend(device)) {
        LUISA_INFO("float64 atomics are not implemented by the '{}' backend; skipping.",
                   device.backend_name());
        return 0;
    }

    "float64_buffer_is_bit_exact"_test = [&] { test_float64_buffer_is_bit_exact(device); };
    "float64_atomic_fetch_add"_test = [&] { test_float64_atomic_fetch_add(device); };
    // The remaining operations are covered on the two routes that implement
    // them. Vulkan has no float64 compare-exchange/min/max form at all (see the
    // XIR-to-SPIR-V diagnostic), and the CUDA route only exposes the native
    // atomicAdd(double*) family: CUDA declares no atomicExch/atomicCAS/min/max
    // overload for double, so NVRTC rejects those kernels.
    auto backend = device.backend_name();
    if (backend != "cuda") {
        // Both routes lower exchange through a native float64 form: DXIL's
        // SM 6.6 64-bit InterlockedExchange over the integer word, and
        // SPIR-V's OpAtomicExchange over the float64 pointer.
        "float64_atomic_exchange"_test = [&] { test_float64_atomic_exchange(device); };
    }
    if (backend == "dx") {
        "float64_atomic_arithmetic"_test = [&] { test_float64_atomic_arithmetic(device); };
        "float64_atomic_compare_exchange"_test = [&] { test_float64_atomic_compare_exchange(device); };
    }
}
