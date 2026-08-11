#include "test_llvm_native_math_fast.h"

// Cross-width numerical, special-value, IR-shape, and assembly audit for the
// precise/fast fixed-vector provider pair.

#include "llvm_jit.h"
#include "llvm_native_math.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

using namespace luisa::compute;

namespace {

enum struct Operation : uint8_t {
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    exp,
    exp2,
    exp10,
    log,
    log2,
    log10,
};

constexpr std::array operations{
    Operation::sin, Operation::cos, Operation::tan,
    Operation::asin, Operation::acos, Operation::atan,
    Operation::atan2,
    Operation::exp, Operation::exp2, Operation::exp10,
    Operation::log, Operation::log2, Operation::log10};

constexpr std::array widths{2u, 3u, 4u, 8u, 16u};

struct MathModule {
    std::unique_ptr<::llvm::LLVMContext> context;
    std::unique_ptr<::llvm::Module> module;
};

[[nodiscard]] bool check(
    bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "fast native-math audit failed: "
                  << message << '\n';
    }
    return condition;
}

[[nodiscard]] constexpr std::string_view operation_name(
    Operation operation) noexcept {
    switch (operation) {
        case Operation::sin: return "sin";
        case Operation::cos: return "cos";
        case Operation::tan: return "tan";
        case Operation::asin: return "asin";
        case Operation::acos: return "acos";
        case Operation::atan: return "atan";
        case Operation::atan2: return "atan2";
        case Operation::exp: return "exp";
        case Operation::exp2: return "exp2";
        case Operation::exp10: return "exp10";
        case Operation::log: return "log";
        case Operation::log2: return "log2";
        case Operation::log10: return "log10";
    }
    return {};
}

[[nodiscard]] constexpr std::string_view provider_name(
    Operation operation) noexcept {
    return operation_name(operation);
}

[[nodiscard]] std::string entry_name(
    Operation operation, uint32_t width,
    cpu::LLVMNativeMathMode mode) {
    return "audit_" + std::string{operation_name(operation)} +
           "_w" + std::to_string(width) + "_" +
           (mode == cpu::LLVMNativeMathMode::fast ? "fast" : "precise");
}

[[nodiscard]] ::llvm::Value *emit_operation(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *input, ::llvm::Value *secondary,
    Operation operation,
    cpu::LLVMNativeMathMode mode) {
    switch (operation) {
        case Operation::sin:
            return cpu::LLVMNativeMath::emit_sin_f32(
                module, builder, input, mode);
        case Operation::cos:
            return cpu::LLVMNativeMath::emit_cos_f32(
                module, builder, input, mode);
        case Operation::tan:
            return cpu::LLVMNativeMath::emit_tan_f32(
                module, builder, input, mode);
        case Operation::asin:
            return cpu::LLVMNativeMath::emit_asin_f32(
                module, builder, input, mode);
        case Operation::acos:
            return cpu::LLVMNativeMath::emit_acos_f32(
                module, builder, input, mode);
        case Operation::atan:
            return cpu::LLVMNativeMath::emit_atan_f32(
                module, builder, input, mode);
        case Operation::atan2:
            return cpu::LLVMNativeMath::emit_atan2_f32(
                module, builder, input, secondary, mode);
        case Operation::exp:
            return cpu::LLVMNativeMath::emit_exp_f32(
                module, builder, input, mode);
        case Operation::exp2:
            return cpu::LLVMNativeMath::emit_exp2_f32(
                module, builder, input, mode);
        case Operation::exp10:
            return cpu::LLVMNativeMath::emit_exp10_f32(
                module, builder, input, mode);
        case Operation::log:
            return cpu::LLVMNativeMath::emit_log_f32(
                module, builder, input, mode);
        case Operation::log2:
            return cpu::LLVMNativeMath::emit_log2_f32(
                module, builder, input, mode);
        case Operation::log10:
            return cpu::LLVMNativeMath::emit_log10_f32(
                module, builder, input, mode);
    }
    return nullptr;
}

void add_entry(
    ::llvm::Module &module, uint32_t width, Operation operation,
    cpu::LLVMNativeMathMode mode) {
    auto &context = module.getContext();
    auto *vector_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getFloatTy(context), width);
    auto *pointer = ::llvm::PointerType::get(context, 0u);
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context),
        {pointer, pointer, pointer}, false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::ExternalLinkage,
        entry_name(operation, width, mode), module);
    auto *entry = ::llvm::BasicBlock::Create(
        context, "entry", function);
    ::llvm::IRBuilder<> builder{entry};
    auto *input = builder.CreateAlignedLoad(
        vector_type, function->getArg(0u), ::llvm::Align{4u});
    auto *secondary = builder.CreateAlignedLoad(
        vector_type, function->getArg(1u), ::llvm::Align{4u});
    auto *result = emit_operation(
        module, builder, input, secondary, operation, mode);
    builder.CreateAlignedStore(
        result, function->getArg(2u), ::llvm::Align{4u});
    builder.CreateRetVoid();
}

[[nodiscard]] MathModule make_math_module() {
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "native-math-tier-audit", *context);
    for (auto mode : {cpu::LLVMNativeMathMode::precise,
                      cpu::LLVMNativeMathMode::fast}) {
        for (auto width : widths) {
            for (auto operation : operations) {
                add_entry(*module, width, operation, mode);
            }
        }
    }
    return {std::move(context), std::move(module)};
}

[[nodiscard]] std::string module_text(const ::llvm::Module &module) {
    std::string text;
    ::llvm::raw_string_ostream stream{text};
    module.print(stream, nullptr);
    stream.flush();
    return text;
}

[[nodiscard]] uint64_t ulp_distance(float lhs, float rhs) {
    auto ordered = [](float value) noexcept {
        auto bits = std::bit_cast<uint32_t>(value);
        return (bits & 0x80000000u) != 0u ?
                   ~bits :
                   bits | 0x80000000u;
    };
    auto a = static_cast<uint64_t>(ordered(lhs));
    auto b = static_cast<uint64_t>(ordered(rhs));
    return a > b ? a - b : b - a;
}

[[nodiscard]] float reference(
    Operation operation, float input, float secondary) {
    switch (operation) {
        case Operation::sin: return std::sin(input);
        case Operation::cos: return std::cos(input);
        case Operation::tan: return std::tan(input);
        case Operation::asin: return std::asin(input);
        case Operation::acos: return std::acos(input);
        case Operation::atan: return std::atan(input);
        case Operation::atan2: return std::atan2(input, secondary);
        case Operation::exp: return std::exp(input);
        case Operation::exp2: return std::exp2(input);
        case Operation::exp10: return std::pow(10.0f, input);
        case Operation::log: return std::log(input);
        case Operation::log2: return std::log2(input);
        case Operation::log10: return std::log10(input);
    }
    return std::numeric_limits<float>::quiet_NaN();
}

[[nodiscard]] bool is_positive_subnormal(float value) noexcept {
    auto bits = std::bit_cast<uint32_t>(value);
    return bits != 0u && bits < 0x00800000u;
}

[[nodiscard]] float tier_reference(
    Operation operation, float input, float secondary,
    cpu::LLVMNativeMathMode mode) {
    if (mode == cpu::LLVMNativeMathMode::fast &&
        (operation == Operation::log ||
         operation == Operation::log2 ||
         operation == Operation::log10) &&
        is_positive_subnormal(input)) {
        return -std::numeric_limits<float>::infinity();
    }
    auto expected = reference(operation, input, secondary);
    if (mode == cpu::LLVMNativeMathMode::fast &&
        (operation == Operation::exp ||
         operation == Operation::exp2 ||
         operation == Operation::exp10) &&
        is_positive_subnormal(expected)) {
        return 0.0f;
    }
    return expected;
}

struct ErrorBound {
    float absolute;
    float relative;
    uint32_t ulp;
};

[[nodiscard]] constexpr ErrorBound fast_bound(
    Operation operation) noexcept {
    switch (operation) {
        case Operation::sin:
        case Operation::cos: return {2.0e-5f, 2.0e-5f, 0u};
        case Operation::tan: return {2.0e-4f, 4.0e-4f, 0u};
        case Operation::asin:
        case Operation::acos: return {2.0e-4f, 0.0f, 0u};
        case Operation::atan: return {1.0e-5f, 1.0e-5f, 0u};
        case Operation::atan2: return {3.0e-6f, 1.0e-6f, 0u};
        case Operation::exp:
        case Operation::exp2:
        case Operation::exp10: return {2.0e-7f, 1.0e-4f, 0u};
        case Operation::log: return {3.0e-6f, 1.0e-6f, 0u};
        case Operation::log2: return {5.0e-6f, 2.0e-6f, 0u};
        case Operation::log10: return {2.0e-6f, 1.0e-6f, 0u};
    }
    return {};
}

[[nodiscard]] constexpr ErrorBound precise_bound(
    Operation operation) noexcept {
    switch (operation) {
        case Operation::log: return {0.0f, 0.0f, 5u};
        case Operation::atan2: return {0.0f, 0.0f, 2u};
        case Operation::exp2:
        case Operation::exp10: return {0.0f, 0.0f, 1u};
        case Operation::log2:
        case Operation::log10: return {0.0f, 0.0f, 3u};
        default: return {0.0f, 0.0f, 4u};
    }
}

[[nodiscard]] bool within_bound(
    float actual, float expected, ErrorBound bound) {
    if (std::isnan(expected)) { return std::isnan(actual); }
    if (std::isinf(expected)) {
        return actual == expected &&
               std::signbit(actual) == std::signbit(expected);
    }
    if (expected == 0.0f) {
        return actual == 0.0f &&
               std::signbit(actual) == std::signbit(expected);
    }
    if (!std::isfinite(actual)) { return false; }
    if (bound.ulp != 0u) {
        return ulp_distance(actual, expected) <= bound.ulp;
    }
    auto error = std::abs(actual - expected);
    return error <= bound.absolute +
                        bound.relative * std::abs(expected);
}

[[nodiscard]] float domain_sample(
    Operation operation, uint32_t bits) noexcept {
    if (operation == Operation::atan2) {
        auto exponent = ((bits >> 16u) % 254u) + 1u;
        auto value_bits = (bits & 0x807fffffu) |
                          (exponent << 23u);
        return std::bit_cast<float>(value_bits);
    }
    auto unit = static_cast<float>(std::bit_cast<int32_t>(bits)) /
                2147483648.0f;
    switch (operation) {
        case Operation::asin:
        case Operation::acos: return unit;
        case Operation::sin:
        case Operation::cos:
        case Operation::tan: return unit * 128.0f;
        case Operation::atan: return unit * 16.0f;
        case Operation::atan2: break;
        case Operation::exp: return unit * 100.0f;
        case Operation::exp2: return unit * 150.0f;
        case Operation::exp10: return unit * 45.0f;
        case Operation::log:
        case Operation::log2:
        case Operation::log10: {
            auto positive = (bits & 0x7fffffffu) | 0x00800000u;
            return std::bit_cast<float>(positive);
        }
    }
    return 0.0f;
}

[[nodiscard]] std::pair<float, float> focused_atan2_pair(
    uint32_t index) noexcept {
    auto exponent = (index % 254u) + 1u;
    auto mantissa = (index * 0x9e3779b9u) & 0x007fffffu;
    auto magnitude = std::bit_cast<float>(
        (exponent << 23u) | mantissa);
    auto above = std::nextafter(
        magnitude, std::numeric_limits<float>::infinity());
    auto ratio = static_cast<float>(index % 8193u) / 8192.0f;
    auto scaled = magnitude * ratio;
    switch ((index / 254u) % 16u) {
        case 0u: return {scaled, magnitude};
        case 1u: return {magnitude, scaled};
        case 2u: return {scaled, -magnitude};
        case 3u: return {-scaled, magnitude};
        case 4u: return {-scaled, -magnitude};
        case 5u: return {magnitude, -scaled};
        case 6u: return {magnitude, magnitude};
        case 7u: return {above, magnitude};
        case 8u: return {0.0f, magnitude};
        case 9u: return {-0.0f, -magnitude};
        case 10u:
            return {magnitude, 0.0f};
        case 11u:
            return {-magnitude, -0.0f};
        case 12u:
            return {std::numeric_limits<float>::infinity(), magnitude};
        case 13u:
            return {magnitude, std::numeric_limits<float>::infinity()};
        case 14u:
            return {std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()};
        default:
            return {-std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity()};
    }
}

[[nodiscard]] float focused_sample(
    Operation operation, uint32_t index) noexcept {
    switch (operation) {
        case Operation::sin:
        case Operation::cos:
        case Operation::tan: {
            auto k = static_cast<int32_t>(index % 163u) - 81;
            auto center = static_cast<float>(
                static_cast<double>(k) *
                1.5707963267948966192);
            switch ((index / 163u) % 3u) {
                case 0u: return center;
                case 1u:
                    return std::nextafter(
                        center, std::numeric_limits<float>::infinity());
                default:
                    return std::nextafter(
                        center, -std::numeric_limits<float>::infinity());
            }
        }
        case Operation::asin:
        case Operation::acos: {
            const std::array boundary{
                -1.0f,
                std::nextafter(-1.0f, 0.0f),
                std::nextafter(-1.0f, -2.0f),
                -0.5f,
                std::nextafter(-0.5f, 0.0f),
                -0.0f,
                0.0f,
                std::nextafter(0.5f, 0.0f),
                0.5f,
                std::nextafter(1.0f, 0.0f),
                1.0f,
                std::nextafter(1.0f, 2.0f),
            };
            return boundary[index % boundary.size()];
        }
        case Operation::atan: {
            constexpr auto low = 0.41421356237309504880f;
            constexpr auto high = 2.4142135623730950488f;
            const std::array boundary{
                -std::numeric_limits<float>::infinity(),
                std::nextafter(-high, -std::numeric_limits<float>::infinity()),
                -high,
                std::nextafter(-high, 0.0f),
                std::nextafter(-low, -std::numeric_limits<float>::infinity()),
                -low,
                std::nextafter(-low, 0.0f),
                -0.0f,
                0.0f,
                std::nextafter(low, 0.0f),
                low,
                std::nextafter(low, std::numeric_limits<float>::infinity()),
                std::nextafter(high, 0.0f),
                high,
                std::nextafter(high, std::numeric_limits<float>::infinity()),
                std::numeric_limits<float>::infinity(),
            };
            return boundary[index % boundary.size()];
        }
        case Operation::atan2:
            return focused_atan2_pair(index).first;
        case Operation::exp: {
            const std::array boundary{
                -104.0f,
                -87.336544750553108986f,
                std::nextafter(
                    -87.336544750553108986f,
                    -std::numeric_limits<float>::infinity()),
                -0.0f,
                0.0f,
                std::nextafter(
                    88.722839052068353053f, 0.0f),
                88.722839052068353053f,
                std::nextafter(
                    88.722839052068353053f,
                    std::numeric_limits<float>::infinity()),
            };
            return boundary[index % boundary.size()];
        }
        case Operation::exp2: {
            const std::array boundary{
                -std::numeric_limits<float>::infinity(),
                std::nextafter(-150.0f, -std::numeric_limits<float>::infinity()),
                -150.0f,
                std::nextafter(-150.0f, std::numeric_limits<float>::infinity()),
                std::nextafter(-126.0f, -std::numeric_limits<float>::infinity()),
                -126.0f,
                std::nextafter(-126.0f, std::numeric_limits<float>::infinity()),
                -0.0f,
                0.0f,
                std::nextafter(128.0f, 0.0f),
                128.0f,
                std::nextafter(128.0f, std::numeric_limits<float>::infinity()),
                std::numeric_limits<float>::infinity(),
            };
            if (index < boundary.size()) { return boundary[index]; }
            auto sample = index - static_cast<uint32_t>(boundary.size());
            auto integer = static_cast<int32_t>(sample % 279u) - 150;
            auto center = static_cast<float>(integer);
            switch ((sample / 279u) % 6u) {
                case 0u: return center;
                case 1u: return std::nextafter(center, -std::numeric_limits<float>::infinity());
                case 2u: return std::nextafter(center, std::numeric_limits<float>::infinity());
                case 3u: return center + 0.5f;
                case 4u: return std::nextafter(center + 0.5f, center);
                default: return std::nextafter(center + 0.5f, center + 1.0f);
            }
        }
        case Operation::exp10: {
            constexpr auto lower_normal = -37.929779453661631102f;
            constexpr auto upper_finite = 38.531839419103623894f;
            const std::array boundary{
                -std::numeric_limits<float>::infinity(),
                -50.0f,
                std::nextafter(lower_normal, -std::numeric_limits<float>::infinity()),
                lower_normal,
                std::nextafter(lower_normal, std::numeric_limits<float>::infinity()),
                -0.0f,
                0.0f,
                std::nextafter(upper_finite, 0.0f),
                upper_finite,
                std::nextafter(upper_finite, std::numeric_limits<float>::infinity()),
                std::numeric_limits<float>::infinity(),
            };
            if (index < boundary.size()) { return boundary[index]; }
            auto sample = index - static_cast<uint32_t>(boundary.size());
            auto exponent = static_cast<int32_t>(sample % 295u) - 166;
            auto center = static_cast<float>(exponent) *
                          0.30102999566398119521f;
            switch ((sample / 295u) % 6u) {
                case 0u: return center;
                case 1u: return std::nextafter(center, -std::numeric_limits<float>::infinity());
                case 2u: return std::nextafter(center, std::numeric_limits<float>::infinity());
                case 3u: return center + 0.15051499783199059760f;
                case 4u: return std::nextafter(center + 0.15051499783199059760f, center);
                default: return std::nextafter(center + 0.15051499783199059760f, std::numeric_limits<float>::infinity());
            }
        }
        case Operation::log: {
            auto exponent = (index % 254u) + 1u;
            auto mantissa = (index * 0x9e3779b9u) & 0x007fffffu;
            return std::bit_cast<float>((exponent << 23u) | mantissa);
        }
        case Operation::log2:
        case Operation::log10: {
            auto exponent = (index % 254u) + 1u;
            auto power_bits = exponent << 23u;
            switch ((index / 254u) % 9u) {
                case 0u: return std::bit_cast<float>(power_bits);
                case 1u: return std::bit_cast<float>(power_bits - 1u);
                case 2u: return std::bit_cast<float>(power_bits + 1u);
                case 3u: {
                    auto value = std::ldexp(
                        1.4142135623730950488f,
                        static_cast<int32_t>(exponent) - 127);
                    return value;
                }
                case 4u: {
                    auto value = std::ldexp(
                        1.4142135623730950488f,
                        static_cast<int32_t>(exponent) - 127);
                    return std::nextafter(value, 0.0f);
                }
                case 5u: {
                    auto value = std::ldexp(
                        1.4142135623730950488f,
                        static_cast<int32_t>(exponent) - 127);
                    return std::nextafter(
                        value, std::numeric_limits<float>::infinity());
                }
                case 6u: {
                    return std::ldexp(
                        1.5f,
                        static_cast<int32_t>(exponent) - 127);
                }
                case 7u: {
                    auto value = std::ldexp(
                        1.5f,
                        static_cast<int32_t>(exponent) - 127);
                    return std::nextafter(value, 0.0f);
                }
                default: {
                    auto value = std::ldexp(
                        1.5f,
                        static_cast<int32_t>(exponent) - 127);
                    return std::nextafter(
                        value, std::numeric_limits<float>::infinity());
                }
            }
        }
    }
    return 0.0f;
}

[[nodiscard]] float focused_secondary(
    Operation operation, uint32_t index) noexcept {
    return operation == Operation::atan2 ?
               focused_atan2_pair(index).second :
               1.0f;
}

[[nodiscard]] bool run_numerical_audit(simd::LLVMJIT &jit) {
    constexpr std::array corpus{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::max(),
        -39001.0f,
        -129.0f,
        -128.0f,
        -127.999f,
        -88.8f,
        -87.4f,
        -10.0f,
        -3.1415926535897932385f,
        -1.5707963267948966192f,
        -1.0f,
        -0.5f,
        -0.0f,
        0.0f,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        0.5f,
        1.0f,
        std::bit_cast<float>(0x3f9216dbu),
        1.5707963267948966192f,
        3.1415926535897932385f,
        10.0f,
        87.4f,
        88.8f,
        127.999f,
        128.0f,
        129.0f,
        39001.0f,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    using Entry = void(const float *, const float *, float *);
    alignas(64) std::array<float, 16u> input{};
    alignas(64) std::array<float, 16u> secondary{};
    alignas(64) std::array<float, 16u> output{};
    for (auto mode : {cpu::LLVMNativeMathMode::precise,
                      cpu::LLVMNativeMathMode::fast}) {
        for (auto width : widths) {
            for (auto operation : operations) {
                auto name = entry_name(operation, width, mode);
                auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
                if (!check(entry != nullptr, name + " lookup")) {
                    return false;
                }
                auto bound = mode == cpu::LLVMNativeMathMode::fast ?
                                 fast_bound(operation) :
                                 precise_bound(operation);
                auto independent_exp_log =
                    operation == Operation::atan2 ||
                    operation == Operation::exp2 ||
                    operation == Operation::exp10 ||
                    operation == Operation::log2 ||
                    operation == Operation::log10;
                auto random_sample_count = independent_exp_log ?
                                               65536u :
                                               8192u;
                auto focused_sample_count = independent_exp_log ?
                                                16384u :
                                                4096u;
                auto check_batch = [&](std::string_view source) {
                    entry(
                        input.data(), secondary.data(), output.data());
                    for (auto lane = size_t{0u}; lane < width; lane++) {
                        auto expected = tier_reference(
                            operation, input[lane], secondary[lane], mode);
                        if (!within_bound(output[lane], expected, bound)) {
                            std::cerr
                                << "native " << operation_name(operation)
                                << ' ' << (mode == cpu::LLVMNativeMathMode::fast ? "fast" : "precise")
                                << " W" << width << ' ' << source
                                << " lane=" << lane
                                << " bits=0x" << std::hex
                                << std::bit_cast<uint32_t>(input[lane])
                                << std::dec << " input=" << input[lane]
                                << " secondary_bits=0x" << std::hex
                                << std::bit_cast<uint32_t>(secondary[lane])
                                << std::dec
                                << " secondary=" << secondary[lane]
                                << " actual=" << output[lane]
                                << " expected=" << expected
                                << " ulp=" << ulp_distance(output[lane], expected)
                                << '\n';
                            return false;
                        }
                    }
                    return true;
                };
                for (auto base = size_t{0u}; base < corpus.size();
                     base += width) {
                    for (auto lane = size_t{0u}; lane < width; lane++) {
                        input[lane] = corpus[(base + lane) % corpus.size()];
                        secondary[lane] = corpus[(base + lane + 7u) % corpus.size()];
                    }
                    if (!check_batch("boundary")) { return false; }
                }
                if (operation == Operation::atan2) {
                    constexpr std::pair counterexample_bits{
                        0x88e8041cu, 0x089650cbu};
                    for (auto lane = size_t{0u}; lane < width; lane++) {
                        input[lane] = std::bit_cast<float>(
                            counterexample_bits.first);
                        secondary[lane] = std::bit_cast<float>(
                            counterexample_bits.second);
                    }
                    if (!check_batch("fixed-counterexample")) {
                        return false;
                    }
                }
                auto state = uint32_t{0x9e3779b9u} ^
                             (width * 0x85ebca6bu) ^
                             static_cast<uint32_t>(operation);
                for (auto base = size_t{0u}; base < random_sample_count;
                     base += width) {
                    for (auto lane = size_t{0u}; lane < width; lane++) {
                        state = state * 1664525u + 1013904223u;
                        input[lane] = std::bit_cast<float>(state);
                        state = state * 1664525u + 1013904223u;
                        secondary[lane] = std::bit_cast<float>(state);
                    }
                    if (!check_batch("raw-bits")) { return false; }
                }
                for (auto base = size_t{0u}; base < random_sample_count;
                     base += width) {
                    for (auto lane = size_t{0u}; lane < width; lane++) {
                        state = state * 1664525u + 1013904223u;
                        input[lane] = domain_sample(operation, state);
                        state = state * 1664525u + 1013904223u;
                        secondary[lane] = domain_sample(
                            operation, state);
                    }
                    if (!check_batch("domain")) { return false; }
                }
                for (auto base = size_t{0u}; base < focused_sample_count;
                     base += width) {
                    for (auto lane = size_t{0u}; lane < width; lane++) {
                        input[lane] = focused_sample(
                            operation,
                            static_cast<uint32_t>(base + lane));
                        secondary[lane] = focused_secondary(
                            operation,
                            static_cast<uint32_t>(base + lane));
                    }
                    if (!check_batch("focused")) { return false; }
                }
            }
        }
    }
    return true;
}

[[nodiscard]] bool run_fast_special_value_contract(simd::LLVMJIT &jit) {
    constexpr auto width = 16u;
    using Entry = void(const float *, const float *, float *);
    alignas(64) std::array<float, width> input{
        -0.0f,
        0.0f,
        std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        std::nextafter(1.0f, 2.0f),
        std::nextafter(-1.0f, -2.0f),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
    };
    alignas(64) std::array<float, width> secondary{
        0.0f,
        -0.0f,
        1.0f,
        1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        0.0f,
        -0.0f,
        std::numeric_limits<float>::quiet_NaN(),
        -1.0f,
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
    };
    alignas(64) std::array<float, width> output{};
    auto expect_bits = [&](Operation operation, size_t lane,
                           uint32_t expected) {
        auto actual = std::bit_cast<uint32_t>(output[lane]);
        if (actual == expected) { return true; }
        std::cerr << "native " << operation_name(operation)
                  << " fast special lane=" << lane
                  << " input_bits=0x" << std::hex
                  << std::bit_cast<uint32_t>(input[lane])
                  << " actual_bits=0x" << actual
                  << " expected_bits=0x" << expected
                  << std::dec << '\n';
        return false;
    };
    constexpr auto qnan = 0x7fc00000u;
    constexpr auto positive_zero = 0x00000000u;
    constexpr auto negative_zero = 0x80000000u;
    constexpr auto one = 0x3f800000u;
    constexpr auto positive_infinity = 0x7f800000u;
    constexpr auto negative_infinity = 0xff800000u;
    auto half_pi = std::bit_cast<uint32_t>(
        1.5707963267948966192f);
    auto negative_half_pi = half_pi | 0x80000000u;
    auto pi = std::bit_cast<uint32_t>(
        3.1415926535897932385f);
    auto quarter_pi = std::bit_cast<uint32_t>(
        0.78539816339744830962f);
    auto three_quarter_pi = std::bit_cast<uint32_t>(
        2.3561944901923449288f);

    for (auto operation : operations) {
        auto name = entry_name(
            operation, width, cpu::LLVMNativeMathMode::fast);
        auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
        if (!check(entry != nullptr, name + " special lookup")) {
            return false;
        }
        entry(input.data(), secondary.data(), output.data());
        switch (operation) {
            case Operation::sin:
            case Operation::tan:
                if (!expect_bits(operation, 0u, negative_zero) ||
                    !expect_bits(operation, 1u, positive_zero) ||
                    !expect_bits(operation, 2u, 0x00000001u) ||
                    !expect_bits(operation, 3u, 0x80000001u) ||
                    !expect_bits(operation, 4u, qnan) ||
                    !expect_bits(operation, 5u, qnan) ||
                    !expect_bits(operation, 6u, qnan)) {
                    return false;
                }
                break;
            case Operation::cos:
                for (auto lane : {0u, 1u, 2u, 3u}) {
                    if (!expect_bits(operation, lane, one)) { return false; }
                }
                for (auto lane : {4u, 5u, 6u}) {
                    if (!expect_bits(operation, lane, qnan)) { return false; }
                }
                break;
            case Operation::asin:
                if (!expect_bits(operation, 0u, negative_zero) ||
                    !expect_bits(operation, 1u, positive_zero) ||
                    !expect_bits(operation, 2u, 0x00000001u) ||
                    !expect_bits(operation, 3u, 0x80000001u) ||
                    !expect_bits(operation, 4u, qnan) ||
                    !expect_bits(operation, 5u, qnan) ||
                    !expect_bits(operation, 6u, qnan) ||
                    !expect_bits(operation, 7u, half_pi) ||
                    !expect_bits(operation, 8u, negative_half_pi) ||
                    !expect_bits(operation, 9u, qnan) ||
                    !expect_bits(operation, 10u, qnan)) {
                    return false;
                }
                break;
            case Operation::acos:
                if (!expect_bits(operation, 0u, half_pi) ||
                    !expect_bits(operation, 1u, half_pi) ||
                    !expect_bits(operation, 4u, qnan) ||
                    !expect_bits(operation, 5u, qnan) ||
                    !expect_bits(operation, 6u, qnan) ||
                    !expect_bits(operation, 7u, positive_zero) ||
                    !expect_bits(operation, 8u, pi) ||
                    !expect_bits(operation, 9u, qnan) ||
                    !expect_bits(operation, 10u, qnan)) {
                    return false;
                }
                break;
            case Operation::atan:
                if (!expect_bits(operation, 0u, negative_zero) ||
                    !expect_bits(operation, 1u, positive_zero) ||
                    !expect_bits(operation, 2u, 0x00000001u) ||
                    !expect_bits(operation, 3u, 0x80000001u) ||
                    !expect_bits(operation, 4u, half_pi) ||
                    !expect_bits(operation, 5u, negative_half_pi) ||
                    !expect_bits(operation, 6u, qnan)) {
                    return false;
                }
                break;
            case Operation::atan2:
                if (!expect_bits(operation, 0u, negative_zero) ||
                    !expect_bits(operation, 1u, pi) ||
                    !expect_bits(operation, 2u, 0x00000001u) ||
                    !expect_bits(operation, 3u, 0x80000001u) ||
                    !expect_bits(operation, 4u, quarter_pi) ||
                    !expect_bits(
                        operation, 5u,
                        three_quarter_pi | 0x80000000u) ||
                    !expect_bits(operation, 6u, qnan) ||
                    !expect_bits(operation, 7u, positive_zero) ||
                    !expect_bits(
                        operation, 8u, pi | 0x80000000u) ||
                    !expect_bits(operation, 9u, half_pi) ||
                    !expect_bits(operation, 10u, negative_half_pi) ||
                    !expect_bits(operation, 11u, qnan)) {
                    return false;
                }
                break;
            case Operation::exp:
            case Operation::exp2:
            case Operation::exp10:
                for (auto lane : {0u, 1u, 2u, 3u}) {
                    if (!expect_bits(operation, lane, one)) { return false; }
                }
                if (!expect_bits(operation, 4u, positive_infinity) ||
                    !expect_bits(operation, 5u, positive_zero) ||
                    !expect_bits(operation, 6u, qnan)) {
                    return false;
                }
                break;
            case Operation::log:
            case Operation::log2:
            case Operation::log10:
                if (!expect_bits(operation, 0u, negative_infinity) ||
                    !expect_bits(operation, 1u, negative_infinity) ||
                    !expect_bits(operation, 2u, negative_infinity) ||
                    !expect_bits(operation, 3u, qnan) ||
                    !expect_bits(operation, 4u, positive_infinity) ||
                    !expect_bits(operation, 5u, qnan) ||
                    !expect_bits(operation, 6u, qnan) ||
                    !expect_bits(operation, 7u, positive_zero) ||
                    !expect_bits(operation, 8u, qnan)) {
                    return false;
                }
                break;
        }
    }
    return true;
}

}// namespace

bool test_llvm_native_math_fast() {
    auto shape = make_math_module();
    if (!check(
            !::llvm::verifyModule(*shape.module, &::llvm::errs()),
            "module verification")) {
        return false;
    }
    auto ir = module_text(*shape.module);
    if (!check(ir.find("extractelement") == std::string::npos,
               "no lane extraction") ||
        !check(ir.find("insertelement") == std::string::npos,
               "no lane insertion") ||
        !check(ir.find("llvm.x86.") == std::string::npos,
               "no x86 intrinsic") ||
        !check(ir.find("llvm.aarch64.") == std::string::npos,
               "no AArch64 intrinsic")) {
        return false;
    }
    for (auto width : widths) {
        for (auto operation : operations) {
            auto provider = provider_name(operation);
            auto precise_suffix = provider == "sin" ||
                                          provider == "cos" ||
                                          provider == "exp" ||
                                          provider == "exp2" ||
                                          provider == "exp10" ?
                                      "u10" :
                                      "u35";
            auto prefix = "__luisa_cpu_native_" +
                          std::string{provider} + "_f32_v" +
                          std::to_string(width) + "_";
            if (!check(ir.find(prefix + "fast") != std::string::npos,
                       prefix + "fast symbol") ||
                !check(ir.find(prefix + precise_suffix) !=
                           std::string::npos,
                       prefix + precise_suffix + " symbol")) {
                return false;
            }
        }
    }

    auto assembly_module = make_math_module();
    simd::LLVMJIT assembly_target;
    if (!check(assembly_target.succeeded(), "assembly JIT creation")) {
        return false;
    }
    auto assembly = assembly_target.emit_assembly(
        std::move(assembly_module.module),
        std::move(assembly_module.context));
    std::transform(
        assembly.begin(), assembly.end(), assembly.begin(),
        [](unsigned char c) noexcept {
            return static_cast<char>(std::tolower(c));
        });
    for (auto symbol : {"sinf", "cosf", "tanf", "asinf", "acosf",
                        "atanf", "atan2f", "expf", "exp2f", "exp10f",
                        "logf", "log2f", "log10f"}) {
        if (!check(assembly.find(symbol) == std::string::npos,
                   std::string{"no scalar symbol "} + symbol)) {
            return false;
        }
    }

    auto executable = make_math_module();
    simd::LLVMJIT jit;
    if (!check(jit.succeeded(), "execution JIT creation") ||
        !check(jit.add_module(
                   std::move(executable.module),
                   std::move(executable.context)),
               "execution JIT module")) {
        return false;
    }
    return run_numerical_audit(jit) &&
           run_fast_special_value_contract(jit);
}
