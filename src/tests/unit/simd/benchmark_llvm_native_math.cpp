#include "llvm_jit.h"
#include "llvm_native_math.h"

// Interleaved throughput and final-assembly audit for the native-math tiers.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

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
    pow,
    pow2_canonicalization,
    pow10_canonicalization,
};

constexpr std::array operations{
    Operation::sin, Operation::cos, Operation::tan,
    Operation::asin, Operation::acos, Operation::atan,
    Operation::atan2,
    Operation::exp, Operation::exp2, Operation::exp10,
    Operation::log, Operation::log2, Operation::log10,
    Operation::pow};

// These pairs reuse the benchmark's interleaved precise/fast slots as
// baseline/canonical slots. Both implementations use the fast tier: the
// baseline emits generic pow(radix, x), while the candidate emits the
// contract-equivalent dedicated radix operation selected by the XIR pass.
constexpr std::array canonicalization_operations{
    Operation::pow2_canonicalization,
    Operation::pow10_canonicalization};

constexpr std::array widths{2u, 3u, 4u, 8u, 16u};
constexpr auto packet_count = uint64_t{4096u};

struct MathModule {
    std::unique_ptr<::llvm::LLVMContext> context;
    std::unique_ptr<::llvm::Module> module;
};

struct Measurement {
    double precise_ns;
    double fast_ns;
    size_t precise_instructions;
    size_t fast_instructions;
};

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
        case Operation::pow: return "pow";
        case Operation::pow2_canonicalization:
            return "pow2_canonicalization";
        case Operation::pow10_canonicalization:
            return "pow10_canonicalization";
    }
    return {};
}

[[nodiscard]] std::string entry_name(
    Operation operation, uint32_t width,
    cpu::LLVMNativeMathMode mode) {
    return "benchmark_native_" +
           std::string{operation_name(operation)} + "_w" +
           std::to_string(width) + "_" +
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
        case Operation::pow:
            return cpu::LLVMNativeMath::emit_pow_f32(
                module, builder, input, secondary, mode);
        case Operation::pow2_canonicalization:
        case Operation::pow10_canonicalization: {
            auto radix = operation == Operation::pow2_canonicalization ?
                             2.0 :
                             10.0;
            auto *scalar = ::llvm::ConstantFP::get(
                builder.getFloatTy(), radix);
            auto *vector_type =
                ::llvm::cast<::llvm::FixedVectorType>(input->getType());
            auto *base = ::llvm::ConstantVector::getSplat(
                vector_type->getElementCount(), scalar);
            if (mode == cpu::LLVMNativeMathMode::precise) {
                return cpu::LLVMNativeMath::emit_pow_f32(
                    module, builder, base, input,
                    cpu::LLVMNativeMathMode::fast);
            }
            return operation == Operation::pow2_canonicalization ?
                       cpu::LLVMNativeMath::emit_exp2_f32(
                           module, builder, input,
                           cpu::LLVMNativeMathMode::fast) :
                       cpu::LLVMNativeMath::emit_exp10_f32(
                           module, builder, input,
                           cpu::LLVMNativeMathMode::fast);
        }
    }
    return nullptr;
}

void add_entry(
    ::llvm::Module &module, uint32_t width, Operation operation,
    cpu::LLVMNativeMathMode mode) {
    auto &context = module.getContext();
    auto *float_type = ::llvm::Type::getFloatTy(context);
    auto *vector_type = ::llvm::FixedVectorType::get(
        float_type, width);
    auto *pointer = ::llvm::PointerType::get(context, 0u);
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context),
        {pointer, pointer, pointer,
         ::llvm::Type::getInt64Ty(context)},
        false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::ExternalLinkage,
        entry_name(operation, width, mode), module);
    auto *entry = ::llvm::BasicBlock::Create(
        context, "entry", function);
    auto *loop = ::llvm::BasicBlock::Create(
        context, "packet.loop", function);
    auto *exit = ::llvm::BasicBlock::Create(
        context, "exit", function);
    ::llvm::IRBuilder<> builder{entry};
    builder.CreateCondBr(
        builder.CreateICmpEQ(function->getArg(3u), builder.getInt64(0u)),
        exit, loop);

    builder.SetInsertPoint(loop);
    auto *index = builder.CreatePHI(builder.getInt64Ty(), 2u, "packet");
    index->addIncoming(builder.getInt64(0u), entry);
    auto *element = builder.CreateMul(index, builder.getInt64(width));
    auto *input_pointer = builder.CreateGEP(
        float_type, function->getArg(0u), element);
    auto *secondary_pointer = builder.CreateGEP(
        float_type, function->getArg(1u), element);
    auto *output_pointer = builder.CreateGEP(
        float_type, function->getArg(2u), element);
    auto *input = builder.CreateAlignedLoad(
        vector_type, input_pointer, ::llvm::Align{4u});
    auto *secondary = builder.CreateAlignedLoad(
        vector_type, secondary_pointer, ::llvm::Align{4u});
    auto *result = emit_operation(
        module, builder, input, secondary, operation, mode);
    builder.CreateAlignedStore(
        result, output_pointer, ::llvm::Align{4u});
    auto *next = builder.CreateAdd(index, builder.getInt64(1u));
    index->addIncoming(next, loop);
    builder.CreateCondBr(
        builder.CreateICmpULT(next, function->getArg(3u)), loop, exit);

    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
}

[[nodiscard]] MathModule make_math_module() {
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "native-math-benchmark", *context);
    for (auto mode : {cpu::LLVMNativeMathMode::precise,
                      cpu::LLVMNativeMathMode::fast}) {
        for (auto width : widths) {
            for (auto operation : operations) {
                add_entry(*module, width, operation, mode);
            }
            for (auto operation : canonicalization_operations) {
                add_entry(*module, width, operation, mode);
            }
        }
    }
    return {std::move(context), std::move(module)};
}

[[nodiscard]] float input_value(
    Operation operation, size_t index) noexcept {
    auto unit = static_cast<float>(
                    static_cast<int32_t>(
                        (index * 747796405u + 2891336453u) &
                        0x00ffffffu)) /
                    8388608.0f -
                1.0f;
    switch (operation) {
        case Operation::sin:
        case Operation::cos:
        case Operation::tan: return unit * 1.25f;
        case Operation::asin:
        case Operation::acos: return unit * 0.95f;
        case Operation::atan:
        case Operation::atan2: return unit * 4.0f;
        case Operation::exp: return unit * 5.0f;
        case Operation::exp2: return unit * 7.0f;
        case Operation::exp10: return unit * 2.0f;
        case Operation::log:
        case Operation::log2:
        case Operation::log10:
        case Operation::pow: return std::exp2(unit * 3.0f);
        case Operation::pow2_canonicalization: return unit * 7.0f;
        case Operation::pow10_canonicalization: return unit * 2.0f;
    }
    return 0.0f;
}

[[nodiscard]] float secondary_input_value(
    Operation operation, size_t index) noexcept {
    if (operation != Operation::atan2 &&
        operation != Operation::pow) {
        return 1.0f;
    }
    auto bits = static_cast<uint32_t>(
        index * 277803737u + 1013904223u);
    auto unit = static_cast<float>(
                    static_cast<int32_t>(bits & 0x00ffffffu)) /
                    8388608.0f -
                1.0f;
    return operation == Operation::pow ?
               unit * 4.0f :
               unit * 4.0f + 0.125f;
}

volatile float benchmark_sink = 0.0f;

using Entry = void(
    const float *, const float *, float *, uint64_t);

[[nodiscard]] double measure(
    Entry *entry, const std::vector<float> &input,
    const std::vector<float> &secondary,
    std::vector<float> &output, uint64_t repetitions,
    uint32_t width) {
    auto begin = std::chrono::steady_clock::now();
    for (auto i = uint64_t{0u}; i < repetitions; i++) {
        entry(
            input.data(), secondary.data(), output.data(), packet_count);
    }
    auto end = std::chrono::steady_clock::now();
    benchmark_sink = benchmark_sink +
                     output[(repetitions * 17u) % output.size()];
    auto elapsed = std::chrono::duration<double, std::nano>{
        end - begin}
                       .count();
    return elapsed /
           static_cast<double>(repetitions * packet_count * width);
}

[[nodiscard]] double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

[[nodiscard]] size_t instruction_count(
    std::string_view assembly, std::string_view function) {
    auto label = std::string{"\n"} + std::string{function} + ":";
    auto begin = assembly.find(label);
    if (begin == std::string_view::npos) { return 0u; }
    begin += label.size();
    auto end = assembly.find("\n.Lfunc_end", begin);
    if (end == std::string_view::npos) {
        end = assembly.find("\n\t.size", begin);
    }
    if (end == std::string_view::npos) { end = assembly.size(); }
    auto count = size_t{0u};
    while (begin < end) {
        auto line_end = assembly.find('\n', begin);
        if (line_end == std::string_view::npos || line_end > end) {
            line_end = end;
        }
        auto line = assembly.substr(begin, line_end - begin);
        auto first = line.find_first_not_of(" \t");
        if (first != std::string_view::npos &&
            line[first] != '.' && line[first] != '#' &&
            line.back() != ':') {
            ++count;
        }
        begin = line_end + 1u;
    }
    return count;
}

[[nodiscard]] bool has_scalar_libm_symbol(std::string assembly) {
    std::transform(
        assembly.begin(), assembly.end(), assembly.begin(),
        [](unsigned char c) noexcept {
            return static_cast<char>(std::tolower(c));
        });
    constexpr std::array symbols{
        "sinf", "cosf", "tanf", "asinf", "acosf", "atanf",
        "atan2f", "expf", "exp2f", "exp10f", "logf", "log2f",
        "log10f", "powf"};
    return std::ranges::any_of(symbols, [&](auto symbol) noexcept {
        return assembly.find(symbol) != std::string::npos;
    });
}

[[nodiscard]] Measurement benchmark_pair(
    simd::LLVMJIT &jit, std::string_view assembly,
    Operation operation, uint32_t width) {
    auto precise_name = entry_name(
        operation, width, cpu::LLVMNativeMathMode::precise);
    auto fast_name = entry_name(
        operation, width, cpu::LLVMNativeMathMode::fast);
    auto precise = reinterpret_cast<Entry *>(jit.lookup(precise_name));
    auto fast = reinterpret_cast<Entry *>(jit.lookup(fast_name));
    if (precise == nullptr || fast == nullptr) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(), 0u, 0u};
    }
    std::vector<float> input(packet_count * width);
    std::vector<float> secondary(packet_count * width);
    std::vector<float> output(packet_count * width);
    for (auto i = size_t{0u}; i < input.size(); i++) {
        input[i] = input_value(operation, i);
        secondary[i] = secondary_input_value(operation, i);
    }
    precise(
        input.data(), secondary.data(), output.data(), packet_count);
    fast(input.data(), secondary.data(), output.data(), packet_count);

    auto repetitions = uint64_t{1u};
    while (measure(
               precise, input, secondary, output, repetitions, width) *
               static_cast<double>(
                   repetitions * packet_count * width) <
           2.0e7) {
        repetitions *= 2u;
    }
    std::vector<double> precise_samples;
    std::vector<double> fast_samples;
    precise_samples.reserve(9u);
    fast_samples.reserve(9u);
    for (auto trial = uint32_t{0u}; trial < 9u; trial++) {
        if ((trial & 1u) == 0u) {
            precise_samples.emplace_back(measure(
                precise, input, secondary, output, repetitions, width));
            fast_samples.emplace_back(measure(
                fast, input, secondary, output, repetitions, width));
        } else {
            fast_samples.emplace_back(measure(
                fast, input, secondary, output, repetitions, width));
            precise_samples.emplace_back(measure(
                precise, input, secondary, output, repetitions, width));
        }
    }
    return {
        median(std::move(precise_samples)),
        median(std::move(fast_samples)),
        instruction_count(assembly, precise_name),
        instruction_count(assembly, fast_name),
    };
}

void print_row(
    std::string_view surface, uint32_t width, Operation operation,
    const Measurement &measurement, bool scalar_libm) {
    std::cout << surface << ',' << width << ','
              << operation_name(operation) << ',' << std::fixed
              << std::setprecision(3) << measurement.precise_ns << ','
              << measurement.fast_ns << ','
              << measurement.precise_ns / measurement.fast_ns << ','
              << measurement.precise_instructions << ','
              << measurement.fast_instructions << ','
              << (scalar_libm ? "yes" : "no") << '\n';
}

}// namespace

int main(int argc, char *argv[]) {
    auto canonicalization_only =
        argc == 2 &&
        std::string_view{argv[1]} == "--canonicalization-only";
    auto assembly_module = make_math_module();
    simd::LLVMJIT assembly_target;
    if (!assembly_target.succeeded()) {
        std::cerr << assembly_target.error() << '\n';
        return 1;
    }
    auto assembly = assembly_target.emit_assembly(
        std::move(assembly_module.module),
        std::move(assembly_module.context));
    if (assembly.empty()) {
        std::cerr << assembly_target.error() << '\n';
        return 1;
    }
    auto scalar_libm = has_scalar_libm_symbol(assembly);

    auto executable = make_math_module();
    simd::LLVMJIT jit;
    if (!jit.succeeded() ||
        !jit.add_module(
            std::move(executable.module),
            std::move(executable.context))) {
        std::cerr << jit.error() << '\n';
        return 1;
    }

    auto stable = !scalar_libm;
    if (!canonicalization_only) {
        std::cout
            << "surface,width,operation,precise_ns_per_element,"
               "fast_ns_per_element,speedup,precise_entry_instructions,"
               "fast_entry_instructions,scalar_libm\n";
        for (auto width : widths) {
            auto aggregate_precise = 0.0;
            auto aggregate_fast = 0.0;
            for (auto operation : operations) {
                auto measurement = benchmark_pair(
                    jit, assembly, operation, width);
                aggregate_precise += measurement.precise_ns;
                aggregate_fast += measurement.fast_ns;
                if (width <= 4u) {
                    print_row(
                        "fallback", width, operation,
                        measurement, scalar_libm);
                }
                if (width >= 4u) {
                    print_row(
                        "simd", width, operation,
                        measurement, scalar_libm);
                }
            }
            auto aggregate_speedup =
                aggregate_precise / aggregate_fast;
            std::cerr << "W" << width << " aggregate speedup: "
                      << std::fixed << std::setprecision(3)
                      << aggregate_speedup << "x\n";
            stable = stable && aggregate_speedup >= 1.05;
        }
    }
    for (auto width : widths) {
        for (auto operation : canonicalization_operations) {
            auto measurement = benchmark_pair(
                jit, assembly, operation, width);
            auto speedup =
                measurement.precise_ns / measurement.fast_ns;
            std::cerr << "W" << width << ' '
                      << operation_name(operation)
                      << " fast-pow baseline -> dedicated radix: "
                      << std::fixed << std::setprecision(3)
                      << speedup << "x ("
                      << measurement.precise_ns << " -> "
                      << measurement.fast_ns << " ns/element, "
                      << measurement.precise_instructions << " -> "
                      << measurement.fast_instructions
                      << " entry instructions)\n";
            stable = stable && speedup >= 1.05;
        }
    }
    if (!stable) {
        std::cerr
            << "fast native math or radix canonicalization did not clear "
               "the 1.05x throughput gate, or emitted a scalar libm "
               "symbol.\n";
        return 1;
    }
    return benchmark_sink == std::numeric_limits<float>::infinity() ? 1 : 0;
}
