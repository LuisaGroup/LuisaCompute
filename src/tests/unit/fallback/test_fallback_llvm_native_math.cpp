#include "fallback_codegen.h"
#include "fallback_llvm_options.h"
#include "llvm_jit.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

using namespace luisa::compute;

namespace luisa::compute::fallback::api {

// fallback_codegen only needs these functions while translating ray-query
// types. This math-only regression never creates one, so keep the standalone
// codegen test independent of Embree and the complete fallback device library.
extern "C" size_t luisa_fallback_ray_query_object_size() noexcept {
    return 256u;
}

extern "C" size_t luisa_fallback_ray_query_object_alignment() noexcept {
    return 16u;
}

}// namespace luisa::compute::fallback::api

namespace {

[[nodiscard]] bool check(bool condition, const char *expression,
                         const char *file, int line) noexcept {
    if (!condition) {
        std::cerr << file << ':' << line << ": check failed: "
                  << expression << '\n';
    }
    return condition;
}

#define CHECK(EXPR)                                                       \
    do {                                                                  \
        if (!check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)) { \
            return false;                                                 \
        }                                                                 \
    } while (false)

struct FallbackMathModule {
    std::unique_ptr<::llvm::LLVMContext> context;
    std::unique_ptr<::llvm::Module> module;
};

enum struct Operation : uint8_t {
    acos,
    asin,
    atan,
    sin,
    cos,
    tan,
    exp,
    exp2,
    exp10,
    log,
    log2,
    log10,
};

constexpr std::array operations{
    Operation::acos, Operation::asin, Operation::atan,
    Operation::sin, Operation::cos, Operation::tan,
    Operation::exp, Operation::exp2, Operation::exp10,
    Operation::log, Operation::log2, Operation::log10};

[[nodiscard]] constexpr std::string_view operation_name(
    Operation operation) noexcept {
    switch (operation) {
        case Operation::acos: return "acos";
        case Operation::asin: return "asin";
        case Operation::atan: return "atan";
        case Operation::sin: return "sin";
        case Operation::cos: return "cos";
        case Operation::tan: return "tan";
        case Operation::exp: return "exp";
        case Operation::exp2: return "exp2";
        case Operation::exp10: return "exp10";
        case Operation::log: return "log";
        case Operation::log2: return "log2";
        case Operation::log10: return "log10";
    }
    return {};
}

[[nodiscard]] constexpr xir::ArithmeticOp xir_operation(
    Operation operation) noexcept {
    switch (operation) {
        case Operation::acos: return xir::ArithmeticOp::ACOS;
        case Operation::asin: return xir::ArithmeticOp::ASIN;
        case Operation::atan: return xir::ArithmeticOp::ATAN;
        case Operation::sin: return xir::ArithmeticOp::SIN;
        case Operation::cos: return xir::ArithmeticOp::COS;
        case Operation::tan: return xir::ArithmeticOp::TAN;
        case Operation::exp: return xir::ArithmeticOp::EXP;
        case Operation::exp2: return xir::ArithmeticOp::EXP2;
        case Operation::exp10: return xir::ArithmeticOp::EXP10;
        case Operation::log: return xir::ArithmeticOp::LOG;
        case Operation::log2: return xir::ArithmeticOp::LOG2;
        case Operation::log10: return xir::ArithmeticOp::LOG10;
    }
    return xir::ArithmeticOp::SIN;
}

[[nodiscard]] std::string function_name(
    Operation operation, uint32_t width) {
    return "fallback_vector_" + std::string{operation_name(operation)} +
           "_v" + std::to_string(width);
}

void add_math_callable(
    xir::Module &module, uint32_t width, Operation operation) {
    auto *type = Type::vector(Type::of<float>(), width);
    auto *function = module.create_callable(type);
    function->set_name(function_name(operation, width));
    auto *input = function->create_value_argument(type);
    auto *body = function->create_body_block();
    xir::XIRBuilder builder;
    builder.set_insertion_point(body);
    auto *result = builder.call(
        type, xir_operation(operation), {input});
    builder.return_(result);
}

[[nodiscard]] FallbackMathModule make_math_module(bool fast_math) {
    xir::Module xir_module;
    for (auto operation : operations) {
        add_math_callable(xir_module, 2u, operation);
        add_math_callable(xir_module, 3u, operation);
        add_math_callable(xir_module, 4u, operation);
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "fallback-native-math", *context);
    static_cast<void>(fallback::luisa_fallback_backend_codegen(
        *context, module.get(), &xir_module, fast_math));
    return {std::move(context), std::move(module)};
}

void add_entry_wrapper(
    ::llvm::Module &module, uint32_t width, Operation operation) {
    auto name = function_name(operation, width);
    auto *native = module.getFunction(name);
    if (native == nullptr) { return; }
    auto *pointer = ::llvm::PointerType::get(module.getContext(), 0u);
    auto *wrapper_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(module.getContext()),
        {pointer, pointer}, false);
    auto *wrapper = ::llvm::Function::Create(
        wrapper_type, ::llvm::GlobalValue::ExternalLinkage,
        name + "_entry", module);
    auto *entry = ::llvm::BasicBlock::Create(
        module.getContext(), "entry", wrapper);
    ::llvm::IRBuilder<> builder{entry};
    auto *vector_type = ::llvm::FixedVectorType::get(
        builder.getFloatTy(), width);
    auto *input = builder.CreateAlignedLoad(
        vector_type, wrapper->getArg(0u), ::llvm::Align{4u});
    ::llvm::SmallVector<::llvm::Value *, 8u> arguments;
    arguments.emplace_back(input);
    for (auto i = 1u; i < native->arg_size(); i++) {
        arguments.emplace_back(::llvm::Constant::getNullValue(
            native->getFunctionType()->getParamType(i)));
    }
    auto *result = builder.CreateCall(native, arguments);
    result->setCallingConv(::llvm::CallingConv::Fast);
    builder.CreateAlignedStore(
        result, wrapper->getArg(1u), ::llvm::Align{4u});
    builder.CreateRetVoid();
}

void add_entry_wrappers(::llvm::Module &module) {
    for (auto operation : operations) {
        add_entry_wrapper(module, 2u, operation);
        add_entry_wrapper(module, 3u, operation);
        add_entry_wrapper(module, 4u, operation);
    }
}

[[nodiscard]] std::string module_text(const ::llvm::Module &module) {
    std::string text;
    ::llvm::raw_string_ostream stream{text};
    module.print(stream, nullptr);
    stream.flush();
    return text;
}

[[nodiscard]] bool approximately_equal(float actual, float expected) {
    if (std::isnan(expected)) { return std::isnan(actual); }
    if (std::isinf(expected)) { return actual == expected; }
    if (expected == 0.0f) {
        return actual == 0.0f &&
               std::signbit(actual) == std::signbit(expected);
    }
    auto error = std::abs(actual - expected);
    return error <= 2.0e-6f ||
           error <= 2.0e-6f * std::abs(expected);
}

template<Operation Op>
[[nodiscard]] float reference(float x) {
    if constexpr (Op == Operation::acos) { return std::acos(x); }
    if constexpr (Op == Operation::asin) { return std::asin(x); }
    if constexpr (Op == Operation::atan) { return std::atan(x); }
    if constexpr (Op == Operation::sin) { return std::sin(x); }
    if constexpr (Op == Operation::cos) { return std::cos(x); }
    if constexpr (Op == Operation::tan) { return std::tan(x); }
    if constexpr (Op == Operation::exp) { return std::exp(x); }
    if constexpr (Op == Operation::exp2) { return std::exp2(x); }
    if constexpr (Op == Operation::exp10) {
        return std::pow(10.0f, x);
    }
    if constexpr (Op == Operation::log) { return std::log(x); }
    if constexpr (Op == Operation::log2) { return std::log2(x); }
    if constexpr (Op == Operation::log10) { return std::log10(x); }
}

template<size_t Width, Operation Op>
[[nodiscard]] bool check_entry(
    simd::LLVMJIT &jit, bool fast_math) {
    using Entry = void(const float *, float *);
    auto name = function_name(Op, Width) + "_entry";
    auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(entry != nullptr);
    alignas(16) std::array<float, Width> input{};
    alignas(16) std::array<float, Width> output{};
    constexpr std::array general_values{
        -0.0f, 0.5f, 39001.25f, -1.0e10f};
    constexpr std::array unit_values{
        -1.0f, -0.5f, 0.5f, 1.0f};
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        if constexpr (Op == Operation::asin ||
                      Op == Operation::acos) {
            input[lane] = unit_values[lane];
        } else {
            input[lane] = general_values[lane];
        }
    }
    entry(input.data(), output.data());
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        auto expected = reference<Op>(input[lane]);
        auto matches = fast_math ?
                           (std::isnan(expected) ? std::isnan(output[lane]) :
                            std::isinf(expected) ? output[lane] == expected :
                                                   std::abs(output[lane] - expected) <=
                                                       2.0e-4f * (1.0f + std::abs(expected))) :
                           approximately_equal(output[lane], expected);
        CHECK(matches);
    }
    return true;
}

[[nodiscard]] bool run() {
    ::llvm::TargetOptions precise_options;
    fallback::apply_fallback_math_target_options(
        precise_options, false);
    CHECK(precise_options.AllowFPOpFusion ==
          ::llvm::FPOpFusion::Standard);
    ::llvm::TargetOptions fast_options;
    fallback::apply_fallback_math_target_options(
        fast_options, true);
    CHECK(fast_options.AllowFPOpFusion ==
          ::llvm::FPOpFusion::Fast);

    for (auto fast_math : {false, true}) {
        auto shape = make_math_module(fast_math);
        CHECK(!::llvm::verifyModule(*shape.module, &::llvm::errs()));
        auto ir = module_text(*shape.module);
        for (auto operation : operations) {
            auto provider = operation_name(operation);
            auto suffix = fast_math ? "fast" :
                          provider == "log" || provider == "log2" ||
                                  provider == "log10" ||
                                  provider == "tan" ||
                                  provider == "asin" ||
                                  provider == "acos" ||
                                  provider == "atan" ?
                                      "u35" :
                                      "u10";
            for (auto width : {2u, 3u, 4u}) {
                CHECK(ir.find("__luisa_cpu_native_" +
                              std::string{provider} + "_f32_v" +
                              std::to_string(width) + "_" + suffix) !=
                      std::string::npos);
                CHECK(ir.find("llvm." +
                              std::string{operation_name(operation)} + ".v" +
                              std::to_string(width) + "f32") ==
                      std::string::npos);
            }
        }
        CHECK(ir.find("extractelement") == std::string::npos);
        CHECK(ir.find("insertelement") == std::string::npos);
        CHECK(ir.find("llvm.x86.") == std::string::npos);
        CHECK(ir.find("llvm.aarch64.") == std::string::npos);
    }

    for (auto fast_math : {false, true}) {
        auto assembly_module = make_math_module(fast_math);
        add_entry_wrappers(*assembly_module.module);
        simd::LLVMJIT assembly_target;
        CHECK(assembly_target.succeeded());
        auto assembly = assembly_target.emit_assembly(
            std::move(assembly_module.module),
            std::move(assembly_module.context));
        std::transform(
            assembly.begin(), assembly.end(), assembly.begin(),
            [](unsigned char c) noexcept {
                return static_cast<char>(std::tolower(c));
            });
        CHECK(assembly.find("sinf") == std::string::npos);
        CHECK(assembly.find("cosf") == std::string::npos);
        CHECK(assembly.find("tanf") == std::string::npos);
        CHECK(assembly.find("asinf") == std::string::npos);
        CHECK(assembly.find("acosf") == std::string::npos);
        CHECK(assembly.find("atanf") == std::string::npos);
        CHECK(assembly.find("expf") == std::string::npos);
        CHECK(assembly.find("exp2f") == std::string::npos);
        CHECK(assembly.find("exp10f") == std::string::npos);
        CHECK(assembly.find("logf") == std::string::npos);
        CHECK(assembly.find("log2f") == std::string::npos);
        CHECK(assembly.find("log10f") == std::string::npos);
    }

    for (auto fast_math : {false, true}) {
        auto executable = make_math_module(fast_math);
        add_entry_wrappers(*executable.module);
        simd::LLVMJIT jit;
        CHECK(jit.succeeded());
        CHECK(jit.add_module(
            std::move(executable.module),
            std::move(executable.context)));
        CHECK((check_entry<2u, Operation::acos>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::acos>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::acos>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::asin>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::asin>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::asin>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::atan>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::atan>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::atan>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::sin>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::sin>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::sin>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::cos>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::cos>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::cos>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::tan>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::tan>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::tan>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::exp>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::exp2>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::exp10>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::log>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::log2>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::log10>(jit, fast_math)));
    }
    return true;
}

}// namespace

int main() {
    return run() ? 0 : 1;
}
