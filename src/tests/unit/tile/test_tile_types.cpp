#include "ut/ut.hpp"
#include <bit>
#include <cmath>
#include <limits>
#include <type_traits>
#include <luisa/tile/runtime.h>
#include <luisa/tile/value.h>
#include <luisa/tile/memory.h>
#ifdef LUISA_TILE_TEST_XIR
#include <luisa/tile/bridge/xir/lower.h>
#endif
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <compute/tile/kernels.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
namespace t = luisa::compute::tile;

template<typename T>
void test_storage(Device &device, Stream &stream, const t::CompileOptions &options) {
    using Bits = std::conditional_t<sizeof(T) == 1u, uint8_t, uint16_t>;
    constexpr auto patterns = size_t{1u} << (sizeof(T) * 8u);
    constexpr auto count = patterns + 3u;
    auto definition = t::tile_kernel("low_precision_storage", [](t::TensorView<const T, 1> A,
                                                                 t::TensorView<T, 1> B) {
        auto n = t::axis("n", 32);
        for (auto &nest : t::parallel(t::shape(ceil_div(count, size_t{32u})))) {
            auto origin = t::coord(nest.index() * 32);
            B(origin, t::shape(n)).store(A(origin, t::shape(n)).load());
        }
    });
    auto kernel = definition.capture(t::tensor_shape(count), t::tensor_shape(count));
    auto shader = t::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    vector<T> input(count), output(count + 2u, std::bit_cast<T>(Bits{0xa5u}));
    for (size_t i = 0u; i < count; i++) { input[i] = std::bit_cast<T>(static_cast<Bits>(i)); }
    auto a = device.create_buffer<T>(count), b = device.create_buffer<T>(output.size());
    stream << a.copy_from(span{input}) << b.copy_from(span{output})
           << shader(a, b.view(1u, count)).dispatch() << b.copy_to(span{output}) << synchronize();
    auto correct = std::bit_cast<Bits>(output.front()) == Bits{0xa5u} && std::bit_cast<Bits>(output.back()) == Bits{0xa5u};
    for (size_t i = 0u; i < count; i++) { correct &= std::bit_cast<Bits>(output[i + 1u]) == std::bit_cast<Bits>(input[i]); }
    expect(correct) << "all encodings, including NaN payloads and subnormals, must copy bit-exactly";
}

template<typename T>
void test_fp8_runtime_boundary(Device &device, const t::CompileOptions &options) {
    auto definition = t::tile_kernel("fp8_storage_boundary", [](t::TensorView<const T, 1> A, t::TensorView<T, 1> B) {
        for (auto &nest : t::parallel(t::shape(17))) {
            B(t::coord(nest.index()), t::shape(1)).store(A(t::coord(nest.index()), t::shape(1)).load());
        }
    });
    auto kernel = definition.capture(t::tensor_shape(17), t::tensor_shape(17));
    expect(kernel.valid());
    auto shader = t::compile(device, kernel, options);
    expect(!static_cast<bool>(shader));
    expect(shader.metadata().error.find("FP8") != string::npos) << shader.metadata().error;
    // Storage buffers can be allocated and uploaded independently of whether
    // this target has a qualified FP8 instruction/conversion realization.
    auto buffer = device.create_buffer<T>(17u);
    expect(eq(buffer.stride(), size_t{1u}));
}

template<typename T>
[[nodiscard]] auto conversion_kernel(int64_t count, bool manual_memory = false) {
    auto definition = t::tile_kernel("low_precision_conversion", [=](t::TensorView<const float, 1> A,
                                                                     t::TensorView<T, 1> B,
                                                                     t::TensorView<float, 1> C) {
        auto n = t::axis("n", 16);
        for (auto &nest : t::parallel(t::shape(ceil_div(count, int64_t{16})))) {
            auto origin = t::coord(nest.index() * 16);
            auto low = t::cast<T>(A(origin, t::shape(n)).load());
            if (manual_memory) {
                auto temporary = t::memory<T>(t::shape(n));
                temporary.store(low);
                B(origin, t::shape(n)).store(temporary.load());
                C(origin, t::shape(n)).store(t::cast<float>(temporary.load()));
            } else {
                B(origin, t::shape(n)).store(low);
                C(origin, t::shape(n)).store(t::cast<float>(low));
            }
        }
    });
    return definition.capture(t::tensor_shape(count), t::tensor_shape(count), t::tensor_shape(count));
}

template<typename T>
void test_conversion(Device &device, Stream &stream, const t::CompileOptions &options, bool manual_memory = false) {
    constexpr auto count = size_t{67u};
    LUISA_INFO("Low precision conversion: element={}, manual_memory={}", to_underlying(t::scalar_type_v<T>), manual_memory);
    auto kernel = conversion_kernel<T>(count, manual_memory);
    expect(kernel.valid());
    auto shader = t::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    vector<float> input(count);
    for (size_t i = 0u; i < count; i++) { input[i] = static_cast<float>(static_cast<int>(i) - 32) * 0.03135f; }
    // Tie-even in both directions, normal values, signed zero, infinities.
    const float edge[]{0.0f, -0.0f, 1.00390625f, 1.01171875f, 1.00048828125f,
                       1.00146484375f, -1.00390625f, 65504.0f, 65536.0f,
                       std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    std::copy(std::begin(edge), std::end(edge), input.begin());
    vector<T> output(count + 2u, T{-7.0f});
    vector<float> restored(count + 2u, -9.0f);
    auto a = device.create_buffer<float>(count);
    auto b = device.create_buffer<T>(output.size());
    auto c = device.create_buffer<float>(restored.size());
    expect(eq(b.stride(), sizeof(T)));
    expect(eq(shader.metadata().arguments[1u].minimum_size_bytes, count * sizeof(T)));
    stream << a.copy_from(span{input}) << b.copy_from(span{output}) << c.copy_from(span{restored})
           << shader(a, b.view(1u, count), c.view(1u, count)).dispatch()
           << b.copy_to(span{output}) << c.copy_to(span{restored}) << synchronize();
    expect(eq(static_cast<float>(output.front()), -7.0f));
    expect(eq(static_cast<float>(output.back()), -7.0f));
    expect(eq(restored.front(), -9.0f));
    expect(eq(restored.back(), -9.0f));
    for (size_t i = 0u; i < count; i++) {
        auto expected = T{input[i]};
        expect(eq(std::bit_cast<uint16_t>(output[i + 1u]), std::bit_cast<uint16_t>(expected))) << i << input[i];
        expect(eq(std::bit_cast<uint32_t>(restored[i + 1u]), std::bit_cast<uint32_t>(static_cast<float>(expected)))) << i;
    }
}

template<typename T, typename Acc = float>
void test_gemm(Device &device, Stream &stream, const t::CompileOptions &options) {
    for (auto dims : {std::array<int64_t, 3>{1, 1, 1}, {7, 9, 11}}) {
        auto [m, n, k] = dims;
        auto kernel = example::tile::gemm<T, Acc>(m, n, k, {2, 2, 4});
        expect(kernel.valid());
        auto shader = t::compile(device, kernel, options);
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { continue; }
        vector<T> a(m * k), b(k * n), c(m * n + 2u, static_cast<T>(-7));
        for (size_t i = 0u; i < a.size(); i++) { a[i] = static_cast<T>(static_cast<int>(i % 5u) - (std::is_unsigned_v<T> ? 0 : 2)); }
        for (size_t i = 0u; i < b.size(); i++) { b[i] = static_cast<T>(static_cast<int>(i % 4u) - (std::is_unsigned_v<T> ? 0 : 2)); }
        auto A = device.create_buffer<T>(a.size()), B = device.create_buffer<T>(b.size()), C = device.create_buffer<T>(c.size());
        stream << A.copy_from(span{a}) << B.copy_from(span{b}) << C.copy_from(span{c})
               << shader(A, B, C.view(1u, m * n)).dispatch() << C.copy_to(span{c}) << synchronize();
        expect(eq(static_cast<float>(c.front()), static_cast<float>(static_cast<T>(-7))));
        expect(eq(static_cast<float>(c.back()), static_cast<float>(static_cast<T>(-7))));
        for (int64_t row = 0; row < m; row++) {
            for (int64_t col = 0; col < n; col++) {
                auto expected = 0.0;
                for (int64_t j = 0; j < k; j++) { expected += static_cast<float>(a[row * k + j]) * static_cast<double>(static_cast<float>(b[j * n + col])); }
                expect(eq(static_cast<float>(c[1u + row * n + col]), static_cast<float>(static_cast<T>(expected))));
            }
        }
    }
}

int main(int argc, char *argv[]) {
    "low_precision_host_layout_and_formats"_test = [] {
        static_assert(t::scalar_type_v<half> == t::ScalarType::FLOAT16);
        static_assert(t::scalar_type_v<t::bf16> == t::ScalarType::BFLOAT16);
        static_assert(t::scalar_type_v<t::float8_e4m3fn> == t::ScalarType::FLOAT8_E4M3FN);
        static_assert(t::ScalarType::FLOAT8_E4M3 == t::ScalarType::FLOAT8_E4M3FN);
        static_assert(t::floating_scalar_cpp_type<half> && t::floating_scalar_cpp_type<t::bf16>);
        expect(eq(Type::of<t::bf16>()->size(), sizeof(t::bf16)));
        expect(eq(Type::of<t::float8_e4m3fn>()->size(), sizeof(t::float8_e4m3fn)));
        expect(eq(t::scalar_type_size(t::ScalarType::FLOAT8_E5M2), size_t{1u}));
        // Exhaustive BF16 decode/encode, including signed zero/subnormals and
        // all NaN payloads. Numeric conversion quiets NaNs; raw bits do not.
        auto correct = true;
        for (uint32_t bits = 0u; bits < 65536u; bits++) {
            auto value = t::bf16::from_bits(static_cast<uint16_t>(bits));
            auto decoded = static_cast<float>(value);
            correct &= std::bit_cast<uint32_t>(decoded) == (bits << 16u);
            auto expected = (bits & 0x7fffu) > 0x7f80u ? bits | 0x40u : bits;
            correct &= t::bf16{decoded}.bits() == expected;
        }
        expect(correct);
        expect(eq(t::bf16{1.00390625f}.bits(), uint16_t{0x3f80u}));
        expect(eq(t::bf16{1.01171875f}.bits(), uint16_t{0x3f82u}));
        expect(eq(static_cast<float>(t::float8_e4m3fn::from_bits(0x7eu)), 448.0f));
        expect(eq(static_cast<float>(t::float8_e5m2::from_bits(0x7bu)), 57344.0f));
        expect(eq(static_cast<float>(t::float8_e4m3fn::from_bits(0x01u)), 0x1p-9f));
        expect(eq(static_cast<float>(t::float8_e5m2::from_bits(0x01u)), 0x1p-16f));
        expect(std::isnan(static_cast<float>(t::float8_e4m3fn::from_bits(0x7fu))));
        expect(std::isinf(static_cast<float>(t::float8_e5m2::from_bits(0x7cu))));
        expect(std::signbit(static_cast<float>(t::float8_e4m3fn::from_bits(0x80u))));
        expect(std::isinf(static_cast<float>(t::maximum.identity<half>())));
        expect(std::isinf(static_cast<float>(t::minimum.identity<t::bf16>())));
        expect(eq(static_cast<float>(t::maximum.identity<t::float8_e4m3fn>()), -448.0f));
    };
    "low_precision_typed_capture_and_fp8_fail_closed"_test = [] {
        auto a = conversion_kernel<half>(17), b = conversion_kernel<t::bf16>(17);
        expect(a.valid());
        expect(b.valid());
        auto definition = t::tile_kernel("fp8_decode", [](t::TensorView<const t::float8_e4m3fn, 1> A,
                                                          t::TensorView<float, 1> B) {
            for (auto &nest : t::parallel(t::shape(16))) {
                B(t::coord(nest.index()), t::shape(1)).store(t::cast<float>(A(t::coord(nest.index()), t::shape(1)).load()));
            }
        });
        auto fp8 = definition.capture(t::tensor_shape(16), t::tensor_shape(16));
        expect(fp8.valid());
#ifdef LUISA_TILE_TEST_XIR
        auto lowered = t::bridge::xir::lower(fp8.function());
        expect(!static_cast<bool>(lowered));
        expect(lowered.error.find("FP8") != string::npos);
#endif
    };
    if (argc == 1) { return 0; }
    if (argc != 2) { return 2; }
    auto backend = string_view{argv[1]};
    if (backend != "metal" && backend != "simd") { return 2; }
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    t::CompileOptions options;
    options.lowering = backend == "metal" ? t::Lowering::TIRX : t::Lowering::NATIVE;
    "low_precision_all_storage_encodings"_test = [&] {
        test_storage<half>(device, stream, options);
        test_storage<t::bf16>(device, stream, options);
    };
    "fp8_runtime_is_explicitly_unavailable"_test = [&] {
        test_fp8_runtime_boundary<t::float8_e4m3fn>(device, options);
        test_fp8_runtime_boundary<t::float8_e5m2>(device, options);
    };
    "low_precision_runtime_conversion"_test = [&] {
        test_conversion<half>(device, stream, options);
        test_conversion<t::bf16>(device, stream, options);
        // Manual addressable Memory is not yet admitted by the XIR planner.
        // Do not hide this existing route boundary with an implicit fallback.
        if (backend == "metal") {
            test_conversion<half>(device, stream, options, true);
            test_conversion<t::bf16>(device, stream, options, true);
        }
    };
    "low_precision_mixed_mma"_test = [&] {
        test_gemm<half>(device, stream, options);
        test_gemm<t::bf16>(device, stream, options);
        test_gemm<int8_t, int32_t>(device, stream, options);
        test_gemm<uint8_t, int32_t>(device, stream, options);
    };
    "bf16_mma_accumulator_is_explicitly_unavailable"_test = [&] {
        auto kernel = example::tile::gemm<t::bf16, t::bf16>(1, 1, 1, {1, 1, 1});
        expect(kernel.valid());
        auto shader = t::compile(device, kernel, options);
        expect(!static_cast<bool>(shader));
        expect(shader.metadata().error.find("BF16") != string::npos) << shader.metadata().error;
    };
}
