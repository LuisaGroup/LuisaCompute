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
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/fast_math_simplify.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

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
    luisa::unique_ptr<::llvm::LLVMContext> context;
    luisa::unique_ptr<::llvm::Module> module;
    size_t canonicalized_radix_pow_count{0u};
};

enum struct Operation : uint8_t {
    acos,
    asin,
    atan,
    atan2,
    sin,
    cos,
    tan,
    exp,
    exp2,
    exp10,
    log,
    log2,
    log10,
    pow,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
};

constexpr std::array operations{
    Operation::acos, Operation::asin, Operation::atan, Operation::atan2,
    Operation::sin, Operation::cos, Operation::tan,
    Operation::exp, Operation::exp2, Operation::exp10,
    Operation::log, Operation::log2, Operation::log10,
    Operation::pow,
    Operation::sinh, Operation::cosh, Operation::tanh,
    Operation::asinh, Operation::acosh, Operation::atanh};

[[nodiscard]] constexpr bool is_binary(
    Operation operation) noexcept {
    return operation == Operation::atan2 ||
           operation == Operation::pow;
}

[[nodiscard]] constexpr luisa::string_view operation_name(
    Operation operation) noexcept {
    switch (operation) {
        case Operation::acos: return "acos";
        case Operation::asin: return "asin";
        case Operation::atan: return "atan";
        case Operation::atan2: return "atan2";
        case Operation::sin: return "sin";
        case Operation::cos: return "cos";
        case Operation::tan: return "tan";
        case Operation::exp: return "exp";
        case Operation::exp2: return "exp2";
        case Operation::exp10: return "exp10";
        case Operation::log: return "log";
        case Operation::log2: return "log2";
        case Operation::log10: return "log10";
        case Operation::pow: return "pow";
        case Operation::sinh: return "sinh";
        case Operation::cosh: return "cosh";
        case Operation::tanh: return "tanh";
        case Operation::asinh: return "asinh";
        case Operation::acosh: return "acosh";
        case Operation::atanh: return "atanh";
    }
    return {};
}

[[nodiscard]] constexpr xir::ArithmeticOp xir_operation(
    Operation operation) noexcept {
    switch (operation) {
        case Operation::acos: return xir::ArithmeticOp::ACOS;
        case Operation::asin: return xir::ArithmeticOp::ASIN;
        case Operation::atan: return xir::ArithmeticOp::ATAN;
        case Operation::atan2: return xir::ArithmeticOp::ATAN2;
        case Operation::sin: return xir::ArithmeticOp::SIN;
        case Operation::cos: return xir::ArithmeticOp::COS;
        case Operation::tan: return xir::ArithmeticOp::TAN;
        case Operation::exp: return xir::ArithmeticOp::EXP;
        case Operation::exp2: return xir::ArithmeticOp::EXP2;
        case Operation::exp10: return xir::ArithmeticOp::EXP10;
        case Operation::log: return xir::ArithmeticOp::LOG;
        case Operation::log2: return xir::ArithmeticOp::LOG2;
        case Operation::log10: return xir::ArithmeticOp::LOG10;
        case Operation::pow: return xir::ArithmeticOp::POW;
        case Operation::sinh: return xir::ArithmeticOp::SINH;
        case Operation::cosh: return xir::ArithmeticOp::COSH;
        case Operation::tanh: return xir::ArithmeticOp::TANH;
        case Operation::asinh: return xir::ArithmeticOp::ASINH;
        case Operation::acosh: return xir::ArithmeticOp::ACOSH;
        case Operation::atanh: return xir::ArithmeticOp::ATANH;
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
    auto *secondary = is_binary(operation) ?
                          function->create_value_argument(type) :
                          nullptr;
    auto *body = function->create_body_block();
    xir::XIRBuilder builder;
    builder.set_insertion_point(body);
    auto *result = is_binary(operation) ?
                       builder.call(
                           type, xir_operation(operation),
                           {input, secondary}) :
                       builder.call(
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
    auto context = luisa::make_unique<::llvm::LLVMContext>();
    auto module = luisa::make_unique<::llvm::Module>(
        "fallback-native-math", *context);
    static_cast<void>(fallback::luisa_fallback_backend_codegen(
        *context, module.get(), &xir_module, fast_math));
    for (auto &function : *module) {
        fallback::apply_fallback_math_function_options(function, fast_math);
    }
    return {std::move(context), std::move(module), 0u};
}

[[nodiscard]] FallbackMathModule make_radix_pow_module(
    bool fast_math) {
    xir::Module xir_module;
    for (auto width : {2u, 3u, 4u}) {
        auto *type = Type::vector(Type::of<float>(), width);
        auto *function = xir_module.create_callable(type);
        function->set_name(
            "fallback_radix_pow_v" + std::to_string(width));
        auto *exponent = function->create_value_argument(type);
        auto *body = function->create_body_block();
        std::array<float, 4u> base2_values{
            2.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 4u> base10_values{
            10.0f, 10.0f, 10.0f, 10.0f};
        auto *base2 = xir_module.create_constant(
            type, base2_values.data());
        auto *base10 = xir_module.create_constant(
            type, base10_values.data());
        xir::XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *pow2 = builder.call(
            type, xir::ArithmeticOp::POW, {base2, exponent});
        auto *pow10 = builder.call(
            type, xir::ArithmeticOp::POW, {base10, exponent});
        builder.return_(builder.call(
            type, xir::ArithmeticOp::BINARY_ADD, {pow2, pow10}));
    }
    auto info = xir::fast_math_simplify_pass_run_on_module(
        &xir_module, {.enable_fast_math = fast_math});
    auto context = luisa::make_unique<::llvm::LLVMContext>();
    auto module = luisa::make_unique<::llvm::Module>(
        "fallback-radix-pow", *context);
    static_cast<void>(fallback::luisa_fallback_backend_codegen(
        *context, module.get(), &xir_module, fast_math));
    for (auto &function : *module) {
        fallback::apply_fallback_math_function_options(function, fast_math);
    }
    return {std::move(context), std::move(module),
            info.radix_pow_count};
}

void add_entry_wrapper(
    ::llvm::Module &module, uint32_t width, Operation operation) {
    auto name = function_name(operation, width);
    auto *native = module.getFunction(name);
    if (native == nullptr) { return; }
    auto *pointer = ::llvm::PointerType::get(module.getContext(), 0u);
    auto *wrapper_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(module.getContext()),
        {pointer, pointer, pointer}, false);
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
    auto *secondary = builder.CreateAlignedLoad(
        vector_type, wrapper->getArg(1u), ::llvm::Align{4u});
    ::llvm::SmallVector<::llvm::Value *, 8u> arguments;
    arguments.emplace_back(input);
    if (is_binary(operation)) {
        arguments.emplace_back(secondary);
    }
    for (auto i = arguments.size(); i < native->arg_size(); i++) {
        arguments.emplace_back(::llvm::Constant::getNullValue(
            native->getFunctionType()->getParamType(i)));
    }
    auto *result = builder.CreateCall(native, arguments);
    result->setCallingConv(::llvm::CallingConv::Fast);
    builder.CreateAlignedStore(
        result, wrapper->getArg(2u), ::llvm::Align{4u});
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
[[nodiscard]] float reference(float x, float secondary) {
    if constexpr (Op == Operation::acos) { return std::acos(x); }
    if constexpr (Op == Operation::asin) { return std::asin(x); }
    if constexpr (Op == Operation::atan) { return std::atan(x); }
    if constexpr (Op == Operation::atan2) {
        return std::atan2(x, secondary);
    }
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
    if constexpr (Op == Operation::pow) { return std::pow(x, secondary); }
    if constexpr (Op == Operation::sinh) { return std::sinh(x); }
    if constexpr (Op == Operation::cosh) { return std::cosh(x); }
    if constexpr (Op == Operation::tanh) { return std::tanh(x); }
    if constexpr (Op == Operation::asinh) { return std::asinh(x); }
    if constexpr (Op == Operation::acosh) { return std::acosh(x); }
    if constexpr (Op == Operation::atanh) { return std::atanh(x); }
}

template<size_t Width, Operation Op>
[[nodiscard]] bool check_entry(
    simd::LLVMJIT &jit, bool fast_math) {
    using Entry = void(const float *, const float *, float *);
    auto name = function_name(Op, Width) + "_entry";
    auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(entry != nullptr);
    alignas(16) std::array<float, Width> input{};
    alignas(16) std::array<float, Width> secondary{};
    alignas(16) std::array<float, Width> output{};
    constexpr std::array general_values{
        -0.0f, 0.5f, 39001.25f, -1.0e10f};
    constexpr std::array unit_values{
        -1.0f, -0.5f, 0.5f, 1.0f};
    constexpr std::array atan2_x_values{
        -0.0f, 0.5f, -0.5f, 1.0f};
    constexpr std::array pow_base_values{
        -2.0f, -0.0f, 0.5f, 4.0f};
    constexpr std::array pow_exponent_values{
        3.0f, -3.0f, -2.0f, 0.5f};
    constexpr std::array hyperbolic_values{
        -0.0f, 0.5f, 10.0f, -10.0f};
    constexpr std::array acosh_values{
        1.0f, 1.5f, 10.0f, 4096.0f};
    constexpr std::array atanh_values{
        -0.0f, 0.25f, -0.5f, 0x1.fffffep-1f};
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        if constexpr (Op == Operation::pow) {
            input[lane] = pow_base_values[lane];
            secondary[lane] = pow_exponent_values[lane];
        } else if constexpr (Op == Operation::asin ||
                             Op == Operation::acos) {
            input[lane] = unit_values[lane];
        } else if constexpr (Op == Operation::acosh) {
            input[lane] = acosh_values[lane];
        } else if constexpr (Op == Operation::atanh) {
            input[lane] = atanh_values[lane];
        } else if constexpr (Op == Operation::sinh ||
                             Op == Operation::cosh ||
                             Op == Operation::tanh ||
                             Op == Operation::asinh) {
            input[lane] = hyperbolic_values[lane];
        } else {
            input[lane] = general_values[lane];
        }
        if constexpr (Op != Operation::pow) {
            secondary[lane] = atan2_x_values[lane];
        }
    }
    entry(input.data(), secondary.data(), output.data());
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        auto expected = reference<Op>(input[lane], secondary[lane]);
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

template<typename Options>
[[nodiscard]] bool check_legacy_math_target_options() {
    Options precise_options;
    fallback::apply_fallback_math_target_options(precise_options, false);
    Options fast_options;
    fallback::apply_fallback_math_target_options(fast_options, true);
    // Newer LLVM snapshots encode this policy only in IR. Keep checking the
    // legacy fields where present, without treating a missing field as a pass
    // for the IR and numerical regressions below.
    if constexpr (requires { precise_options.AllowFPOpFusion; }) {
        using Fusion = decltype(precise_options.AllowFPOpFusion);
        CHECK(precise_options.AllowFPOpFusion == Fusion::Standard);
        CHECK(fast_options.AllowFPOpFusion == Fusion::Fast);
    }
    if constexpr (requires { precise_options.NoTrappingFPMath; }) {
        CHECK(precise_options.NoTrappingFPMath);
        CHECK(fast_options.NoTrappingFPMath);
    }
    return true;
}

[[nodiscard]] bool check_math_function_options() {
    ::llvm::LLVMContext context;
    ::llvm::Module module{"fallback-fp-policy", context};
    auto *float_type = ::llvm::Type::getFloatTy(context);
    auto *type = ::llvm::FunctionType::get(
        float_type, {float_type, float_type, float_type}, false);
    constexpr std::array inherited_fast_attributes{
        "unsafe-fp-math", "no-infs-fp-math", "no-nans-fp-math",
        "no-signed-zeros-fp-math", "approx-func-fp-math"};
    for (auto fast_math : {false, true}) {
        for (auto kind : {0u, 1u, 2u}) {
            auto *function = ::llvm::Function::Create(
                type, ::llvm::GlobalValue::ExternalLinkage, "fp_policy", module);
            auto *entry = ::llvm::BasicBlock::Create(context, "entry", function);
            ::llvm::IRBuilder<> builder{entry};
            auto *product = ::llvm::cast<::llvm::Instruction>(
                builder.CreateFMul(function->getArg(0u), function->getArg(1u)));
            auto *sum = ::llvm::cast<::llvm::Instruction>(
                builder.CreateFAdd(product, function->getArg(2u)));
            builder.CreateRet(sum);
            auto native = kind == 2u;
            if (kind != 0u) {
                // Models attributes retained when parsing compiler-produced
                // builtin bitcode; clearing instruction FMF alone is not enough.
                for (auto attribute : inherited_fast_attributes) {
                    function->addFnAttr(attribute, "true");
                }
            }
            ::llvm::FastMathFlags original_flags;
            if (native) {
                function->addFnAttr("luisa.cpu.native_math");
                original_flags.setAllowContract(true);
            } else {
                original_flags = ::llvm::FastMathFlags::getFast();
            }
            product->setFastMathFlags(original_flags);
            sum->setFastMathFlags(original_flags);
            fallback::apply_fallback_math_function_options(*function, fast_math);
            CHECK(function->getFnAttribute("no-trapping-math").getValueAsString() == "true");
            for (auto *instruction : {product, sum}) {
                auto flags = instruction->getFastMathFlags();
                if (native) {
                    CHECK(!(flags != original_flags));
                } else if (fast_math) {
                    CHECK(flags.isFast());
                    CHECK(flags.allowContract());
                } else {
                    CHECK(!flags.any());
                }
            }
            if (!native && !fast_math) {
                for (auto attribute : inherited_fast_attributes) {
                    CHECK(!function->hasFnAttribute(attribute));
                }
            }
            if (native) {
                for (auto attribute : inherited_fast_attributes) {
                    CHECK(function->getFnAttribute(attribute).getValueAsString() == "true");
                }
            }
        }
    }
    CHECK(!::llvm::verifyModule(module, &::llvm::errs()));
    return true;
}

[[nodiscard]] bool check_common_fusion_options() {
    using Fusion = luisa::compute::detail::LLVMFPOpFusion;
    ::llvm::LLVMContext context;
    ::llvm::Module module{"llvm-fusion-policy", context};
    auto *float_type = ::llvm::Type::getFloatTy(context);
    auto *type = ::llvm::FunctionType::get(
        float_type, {float_type, float_type, float_type}, false);
    for (auto fusion : {Fusion::Strict, Fusion::Standard, Fusion::Fast}) {
        auto *function = ::llvm::Function::Create(
            type, ::llvm::GlobalValue::ExternalLinkage, "fusion", module);
        auto *entry = ::llvm::BasicBlock::Create(context, "entry", function);
        ::llvm::IRBuilder<> builder{entry};
        auto *a = function->getArg(0u);
        auto *b = function->getArg(1u);
        auto *c = function->getArg(2u);
        auto *plain = ::llvm::cast<::llvm::Instruction>(builder.CreateFAdd(a, b));
        ::llvm::FastMathFlags plain_flags;
        plain_flags.setNoNaNs(true);
        plain_flags.setNoSignedZeros(true);
        plain->setFastMathFlags(plain_flags);
        auto *implicit = ::llvm::cast<::llvm::Instruction>(
            builder.CreateIntrinsic(::llvm::Intrinsic::fmuladd, {float_type}, {a, b, c}));
        auto *contracted = ::llvm::cast<::llvm::Instruction>(
            builder.CreateIntrinsic(::llvm::Intrinsic::fmuladd, {float_type}, {a, b, c}));
        contracted->setHasAllowContract(true);
        auto *explicit_fma = builder.CreateIntrinsic(::llvm::Intrinsic::fma, {float_type}, {a, b, c});
        auto *sum = builder.CreateFAdd(implicit, contracted);
        sum = builder.CreateFAdd(sum, explicit_fma);
        builder.CreateRet(builder.CreateFAdd(sum, plain));
        luisa::compute::detail::apply_llvm_fp_function_options(*function, fusion);
        CHECK(function->getFnAttribute("no-trapping-math").getValueAsString() == "true");
        auto expected_plain_flags = plain_flags;
        if (fusion == Fusion::Fast) { expected_plain_flags.setAllowContract(true); }
        CHECK(!(plain->getFastMathFlags() != expected_plain_flags));
        CHECK(!plain->getFastMathFlags().noInfs());
        uint32_t implicit_count = 0u;
        uint32_t explicit_count = 0u;
        uint32_t contract_products = 0u;
        uint32_t precise_products = 0u;
        for (auto &instruction : *entry) {
            if (auto *intrinsic = ::llvm::dyn_cast<::llvm::IntrinsicInst>(&instruction)) {
                implicit_count += intrinsic->getIntrinsicID() == ::llvm::Intrinsic::fmuladd;
                explicit_count += intrinsic->getIntrinsicID() == ::llvm::Intrinsic::fma;
            }
            if (instruction.getOpcode() == ::llvm::Instruction::FMul) {
                auto flags = instruction.getFastMathFlags();
                if (flags.allowContract()) {
                    ::llvm::FastMathFlags only_contract;
                    only_contract.setAllowContract(true);
                    CHECK(!(flags != only_contract));
                    contract_products++;
                } else {
                    CHECK(!flags.any());
                    precise_products++;
                }
            }
        }
        CHECK(explicit_count == 1u);
        if (fusion == Fusion::Strict) {
            CHECK(implicit_count == 0u);
            CHECK(contract_products == 1u && precise_products == 1u);
        } else {
            CHECK(implicit_count == 2u);
            CHECK(contract_products == 0u && precise_products == 0u);
            CHECK(implicit->getFastMathFlags().allowContract() == (fusion == Fusion::Fast));
            CHECK(contracted->getFastMathFlags().allowContract());
        }
    }
    CHECK(!::llvm::verifyModule(module, &::llvm::errs()));
    return true;
}

[[nodiscard]] bool check_precise_arithmetic_contract() {
    auto context = luisa::make_unique<::llvm::LLVMContext>();
    auto module = luisa::make_unique<::llvm::Module>("fallback-fp-contract", *context);
    auto *float_type = ::llvm::Type::getFloatTy(*context);
    auto *pointer_type = ::llvm::PointerType::get(*context, 0u);
    auto *type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(*context),
        {float_type, float_type, float_type, pointer_type}, false);
    auto *function = ::llvm::Function::Create(
        type, ::llvm::GlobalValue::ExternalLinkage, "fp_contract_entry", *module);
    auto *entry = ::llvm::BasicBlock::Create(*context, "entry", function);
    ::llvm::IRBuilder<> builder{entry};
    auto *a = function->getArg(0u);
    auto *b = function->getArg(1u);
    auto *c = function->getArg(2u);
    auto *product = builder.CreateFMul(a, b);
    std::array<::llvm::Value *, 6u> results{
        builder.CreateFAdd(product, c),
        builder.CreateIntrinsic(::llvm::Intrinsic::fma, {float_type}, {a, b, c}),
        product,
        builder.CreateFDiv(::llvm::ConstantFP::get(float_type, 1.0), a),
        builder.CreateUIToFP(builder.CreateFCmpUNO(a, a), float_type),
        builder.CreateIntrinsic(::llvm::Intrinsic::fmuladd, {float_type}, {a, b, c})};
    for (auto i = size_t{0u}; i < results.size(); i++) {
        auto *address = builder.CreateGEP(
            float_type, function->getArg(3u), builder.getInt64(i));
        builder.CreateAlignedStore(results[i], address, ::llvm::Align{4u});
    }
    builder.CreateRetVoid();
    fallback::apply_fallback_math_function_options(*function, false);
    luisa::compute::detail::apply_llvm_fp_function_options(
        *function, luisa::compute::detail::LLVMFPOpFusion::Strict);
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
    simd::LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(module), std::move(context)));
    using Entry = void(float, float, float, float *);
    auto function_pointer = reinterpret_cast<Entry *>(jit.lookup("fp_contract_entry"));
    CHECK(function_pointer != nullptr);
    std::array<float, 6u> output{};
    // Inputs cross the JIT boundary at runtime. The rounded product is 1,
    // whereas an explicit fused operation retains the negative 2^-46 residue.
    function_pointer(0x1.000002p0f, 0x1.fffffcp-1f, -1.0f, output.data());
    CHECK(output[0u] == 0.0f && !std::signbit(output[0u]));
    CHECK(output[1u] == -0x1p-46f);
    CHECK(output[5u] == 0.0f && !std::signbit(output[5u]));
    function_pointer(-0.0f, 1.0f, 0.0f, output.data());
    CHECK(output[2u] == 0.0f && std::signbit(output[2u]));
    CHECK(std::isinf(output[3u]) && std::signbit(output[3u]));
    auto infinity = std::numeric_limits<float>::infinity();
    function_pointer(infinity, 1.0f, -infinity, output.data());
    CHECK(std::isnan(output[0u]) && std::isnan(output[1u]));
    CHECK(output[2u] == infinity);
    CHECK(output[3u] == 0.0f && !std::signbit(output[3u]));
    function_pointer(-infinity, 1.0f, 0.0f, output.data());
    CHECK(output[2u] == -infinity);
    CHECK(output[3u] == 0.0f && std::signbit(output[3u]));
    function_pointer(std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.0f, output.data());
    CHECK(std::isnan(output[0u]) && std::isnan(output[1u]));
    CHECK(std::isnan(output[2u]) && std::isnan(output[3u]));
    CHECK(output[4u] == 1.0f);
    return true;
}

[[nodiscard]] bool run() {
    CHECK(check_legacy_math_target_options<::llvm::TargetOptions>());
    CHECK(check_math_function_options());
    CHECK(check_common_fusion_options());
    CHECK(check_precise_arithmetic_contract());

    for (auto fast_math : {false, true}) {
        auto shape = make_math_module(fast_math);
        CHECK(!::llvm::verifyModule(*shape.module, &::llvm::errs()));
        auto ir = module_text(*shape.module);
        for (auto operation : operations) {
            auto provider = operation_name(operation);
            auto suffix = fast_math ? "fast" :
                          provider == "sinh" || provider == "cosh" ||
                                  provider == "tanh" || provider == "asinh" ||
                                  provider == "acosh" || provider == "atanh" ?
                                      "precise" :
                          provider == "log" || provider == "log2" ||
                                  provider == "log10" ||
                                  provider == "tan" ||
                                  provider == "asin" ||
                                  provider == "acos" ||
                                  provider == "atan" ||
                                  provider == "atan2" ?
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
        auto shape = make_radix_pow_module(fast_math);
        CHECK(!::llvm::verifyModule(*shape.module, &::llvm::errs()));
        CHECK(shape.canonicalized_radix_pow_count ==
              (fast_math ? 6u : 0u));
        auto ir = module_text(*shape.module);
        for (auto width : {2u, 3u, 4u}) {
            auto width_suffix =
                "_f32_v" + std::to_string(width) + "_fast";
            if (fast_math) {
                CHECK(ir.find("__luisa_cpu_native_exp2" +
                              width_suffix) != std::string::npos);
                CHECK(ir.find("__luisa_cpu_native_exp10" +
                              width_suffix) != std::string::npos);
                CHECK(ir.find("__luisa_cpu_native_pow" +
                              width_suffix) == std::string::npos);
            } else {
                CHECK(ir.find("__luisa_cpu_native_pow_f32_v" +
                              std::to_string(width) + "_u10") !=
                      std::string::npos);
            }
        }
    }

    for (auto fast_math : {false, true}) {
        auto assembly_module = make_math_module(fast_math);
        add_entry_wrappers(*assembly_module.module);
        simd::LLVMJIT assembly_target;
        CHECK(assembly_target.succeeded());
        auto assembly = assembly_target.emit_assembly(
            std::move(assembly_module.module),
            std::move(assembly_module.context));
        luisa::transform(
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
        CHECK(assembly.find("atan2f") == std::string::npos);
        CHECK(assembly.find("expf") == std::string::npos);
        CHECK(assembly.find("exp2f") == std::string::npos);
        CHECK(assembly.find("exp10f") == std::string::npos);
        CHECK(assembly.find("logf") == std::string::npos);
        CHECK(assembly.find("log2f") == std::string::npos);
        CHECK(assembly.find("log10f") == std::string::npos);
        CHECK(assembly.find("powf") == std::string::npos);
        CHECK(assembly.find("sinhf") == std::string::npos);
        CHECK(assembly.find("coshf") == std::string::npos);
        CHECK(assembly.find("tanhf") == std::string::npos);
        CHECK(assembly.find("asinhf") == std::string::npos);
        CHECK(assembly.find("acoshf") == std::string::npos);
        CHECK(assembly.find("atanhf") == std::string::npos);
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
        CHECK((check_entry<2u, Operation::atan2>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::atan2>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::atan2>(jit, fast_math)));
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
        CHECK((check_entry<2u, Operation::pow>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::pow>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::pow>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::sinh>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::sinh>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::sinh>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::cosh>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::cosh>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::cosh>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::tanh>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::tanh>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::tanh>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::asinh>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::asinh>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::asinh>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::acosh>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::acosh>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::acosh>(jit, fast_math)));
        CHECK((check_entry<2u, Operation::atanh>(jit, fast_math)));
        CHECK((check_entry<3u, Operation::atanh>(jit, fast_math)));
        CHECK((check_entry<4u, Operation::atanh>(jit, fast_math)));
    }
    return true;
}

}// namespace

int main() {
    return run() ? 0 : 1;
}
