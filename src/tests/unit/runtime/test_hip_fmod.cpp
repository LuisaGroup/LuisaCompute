// Native floating-remainder ABI and range-reduction regression. Host fmod is
// a primitive arithmetic check, not a renderer/shader reference evaluator.
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>

#include <array>
#include <cmath>
#include <limits>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

template<typename T>
void test_remainder(Device &device) {
    using V = Vector<T, 4u>;
    constexpr auto count = 64u;
    const auto largest = static_cast<double>(std::numeric_limits<T>::max());
    const auto smallest = static_cast<double>(std::numeric_limits<T>::min());
    const std::array<std::array<double, 2u>, 16u> cases{{
        {5.75, 2.0}, {-5.75, 2.0}, {5.75, -2.0}, {-5.75, -2.0},
        {largest, 6.283185307179586}, {-largest, 6.283185307179586},
        {largest, 0.1}, {largest, smallest},
        {0.0, 2.0}, {-0.0, 2.0}, {2.0, 2.0}, {-2.0, 2.0},
        {0.3, 0.1}, {-0.3, 0.1}, {smallest, largest}, {-smallest, largest}}};
    std::array<V, count> lhs{}, rhs{};
    auto state = 0x63f7259bu;
    const auto random_unit = [&]() {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return static_cast<double>(state) / 4294967296.0;
    };
    for (auto i = 0u; i < count; i++) {
        for (auto lane = 0u; lane < 4u; lane++) {
            auto index = i * 4u + lane;
            if (index < cases.size()) {
                lhs[i][lane] = static_cast<T>(cases[index][0]);
                rhs[i][lane] = static_cast<T>(cases[index][1]);
            } else {
                // Generate normal operands over the type's exponent range.
                // Large quotients must retain their residue even with fast math.
                const auto exponent = [&]() {
                    return static_cast<int>(random_unit() *
                        (std::numeric_limits<T>::max_exponent -
                         std::numeric_limits<T>::min_exponent - 2)) +
                        std::numeric_limits<T>::min_exponent;
                };
                lhs[i][lane] = static_cast<T>(std::ldexp(0.5 + 0.5 * random_unit(), exponent()));
                rhs[i][lane] = static_cast<T>(std::ldexp(0.5 + 0.5 * random_unit(), exponent()));
                if (index & 1u) { lhs[i][lane] = -lhs[i][lane]; }
                if (index & 2u) { rhs[i][lane] = -rhs[i][lane]; }
            }
        }
    }

    auto stream = device.create_stream();
    auto x = device.create_buffer<V>(count);
    auto y = device.create_buffer<V>(count);
    auto scalar = device.create_buffer<T>(count);
    auto vector = device.create_buffer<V>(count);
    auto broadcast = device.create_buffer<V>(count);
    Kernel1D kernel = [](Var<Buffer<V>> x, Var<Buffer<V>> y, Var<Buffer<T>> scalar,
                         Var<Buffer<V>> vector, Var<Buffer<V>> broadcast) noexcept {
        auto i = dispatch_x();
        auto a = x.read(i);
        auto b = y.read(i);
        scalar.write(i, fmod(a.x, b.x));
        vector.write(i, fmod(a, b));
        broadcast.write(i, fmod(a, b.x));
    };
    stream << x.copy_from(luisa::span{lhs}) << y.copy_from(luisa::span{rhs});
    for (auto fast : {false, true}) {
        auto shader = device.compile(kernel, ShaderOption{
            .enable_cache = false, .enable_fast_math = fast});
        std::array<T, count> scalar_result{};
        std::array<V, count> vector_result{}, broadcast_result{};
        stream << shader(x, y, scalar, vector, broadcast).dispatch(count)
               << scalar.copy_to(luisa::span{scalar_result})
               << vector.copy_to(luisa::span{vector_result})
               << broadcast.copy_to(luisa::span{broadcast_result}) << synchronize();
        const auto check = [&](T actual, T a, T b) {
            auto expected = static_cast<T>(std::fmod(static_cast<double>(a), static_cast<double>(b)));
            auto result = static_cast<double>(actual);
            auto reference = static_cast<double>(expected);
            auto tolerance = 4.0 * static_cast<double>(std::numeric_limits<T>::epsilon()) * std::abs(reference) +
                             4.0 * static_cast<double>(std::numeric_limits<T>::denorm_min());
            expect(std::isfinite(result) && std::abs(result - reference) <= tolerance)
                << "fmod bytes=" << sizeof(T) << " fast=" << fast << ": "
                << static_cast<double>(a) << " % " << static_cast<double>(b)
                << " expected " << reference << " got " << result;
            if (!fast) {
                expect(std::signbit(result) == std::signbit(reference));
            }
        };
        for (auto i = 0u; i < count; i++) {
            check(scalar_result[i], lhs[i].x, rhs[i].x);
            for (auto lane = 0u; lane < 4u; lane++) {
                check(vector_result[i][lane], lhs[i][lane], rhs[i][lane]);
                check(broadcast_result[i][lane], lhs[i][lane], rhs[i].x);
            }
        }
    }

    // Fast math permits finite-input assumptions; exceptional inputs belong
    // to the precise contract. Check classes/signs, not NaN payload bits.
    auto infinity = std::numeric_limits<T>::infinity();
    auto nan = std::numeric_limits<T>::quiet_NaN();
    auto zero = static_cast<T>(0.0);
    auto one = static_cast<T>(1.0);
    lhs[0] = V{infinity, -infinity, nan, one};
    rhs[0] = V{one, one, one, zero};
    lhs[1] = V{one, -one, zero, -zero};
    rhs[1] = V{infinity, -infinity, infinity, -infinity};
    lhs[2] = V{infinity, nan, one, zero};
    rhs[2] = V{infinity, nan, nan, zero};
    auto precise = device.compile(kernel, ShaderOption{
        .enable_cache = false, .enable_fast_math = false});
    std::array<V, count> exceptional_result{};
    stream << x.copy_from(luisa::span{lhs}) << y.copy_from(luisa::span{rhs})
           << precise(x, y, scalar, vector, broadcast).dispatch(count)
           << vector.copy_to(luisa::span{exceptional_result}) << synchronize();
    for (auto i = 0u; i < 3u; i++) {
        for (auto lane = 0u; lane < 4u; lane++) {
            auto expected = std::fmod(static_cast<double>(lhs[i][lane]),
                                      static_cast<double>(rhs[i][lane]));
            auto actual = static_cast<double>(exceptional_result[i][lane]);
            if (std::isnan(expected)) {
                expect(std::isnan(actual));
            } else {
                expect(actual == expected);
                expect(std::signbit(actual) == std::signbit(expected));
            }
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    // Optional type selection helps isolate a backend's native scalar ABI;
    // the default always exercises all three types, including known failures.
    const auto type = argc > 2 ? std::string_view{argv[2]} : "all";
    if (type == "all" || type == "half") { test_remainder<half>(dc->device); }
    if (type == "all" || type == "float") { test_remainder<float>(dc->device); }
    if (type == "all" || type == "double") { test_remainder<double>(dc->device); }
    if (type != "all" && type != "half" && type != "float" && type != "double") { return 2; }
}
