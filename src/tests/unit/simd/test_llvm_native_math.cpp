#include "llvm_jit.h"
#include "llvm_native_math.h"
#include "llvm_schedule_codegen.h"
#include "test_llvm_native_math_fast.h"
#include "xir_to_schedule.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

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

struct MathModule {
    std::unique_ptr<::llvm::LLVMContext> context;
    std::unique_ptr<::llvm::Module> module;
    std::string entry_name;
};

enum struct ExtendedTrigOperation : uint8_t {
    tan,
    asin,
    acos,
    atan,
};

[[nodiscard]] constexpr std::string_view extended_trig_name(
    ExtendedTrigOperation operation) noexcept {
    switch (operation) {
        case ExtendedTrigOperation::tan: return "tan";
        case ExtendedTrigOperation::asin: return "asin";
        case ExtendedTrigOperation::acos: return "acos";
        case ExtendedTrigOperation::atan: return "atan";
    }
    return {};
}

[[nodiscard]] MathModule make_trig_module(
    uint32_t width, bool cosine) {
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-native-math", *context);
    auto *vector_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getFloatTy(*context), width);
    auto *pointer = ::llvm::PointerType::get(*context, 0u);
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(*context), {pointer, pointer}, false);
    auto entry_name = std::string{cosine ? "native_cos_w" :
                                           "native_sin_w"} +
                      std::to_string(width);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::ExternalLinkage,
        entry_name, *module);
    auto *entry = ::llvm::BasicBlock::Create(
        *context, "entry", function);
    ::llvm::IRBuilder<> builder{entry};
    auto *input = builder.CreateAlignedLoad(
        vector_type, function->getArg(0u), ::llvm::Align{4u});
    auto *result = cosine ?
                       cpu::LLVMNativeMath::emit_cos_f32(
                           *module, builder, input,
                           cpu::LLVMNativeMathMode::precise) :
                       cpu::LLVMNativeMath::emit_sin_f32(
                           *module, builder, input,
                           cpu::LLVMNativeMathMode::precise);
    builder.CreateAlignedStore(
        result, function->getArg(1u), ::llvm::Align{4u});
    builder.CreateRetVoid();
    return {std::move(context), std::move(module), std::move(entry_name)};
}

[[nodiscard]] MathModule make_exp_log_module(
    uint32_t width, bool logarithm) {
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-native-exp-log", *context);
    auto *vector_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getFloatTy(*context), width);
    auto *pointer = ::llvm::PointerType::get(*context, 0u);
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(*context), {pointer, pointer}, false);
    auto entry_name = std::string{logarithm ? "native_log_w" :
                                              "native_exp_w"} +
                      std::to_string(width);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::ExternalLinkage,
        entry_name, *module);
    auto *entry = ::llvm::BasicBlock::Create(
        *context, "entry", function);
    ::llvm::IRBuilder<> builder{entry};
    auto *input = builder.CreateAlignedLoad(
        vector_type, function->getArg(0u), ::llvm::Align{4u});
    auto *result = logarithm ?
                       cpu::LLVMNativeMath::emit_log_f32(
                           *module, builder, input,
                           cpu::LLVMNativeMathMode::precise) :
                       cpu::LLVMNativeMath::emit_exp_f32(
                           *module, builder, input,
                           cpu::LLVMNativeMathMode::precise);
    builder.CreateAlignedStore(
        result, function->getArg(1u), ::llvm::Align{4u});
    builder.CreateRetVoid();
    return {std::move(context), std::move(module), std::move(entry_name)};
}

[[nodiscard]] MathModule make_extended_trig_module(
    uint32_t width, ExtendedTrigOperation operation) {
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-native-extended-trig", *context);
    auto *vector_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getFloatTy(*context), width);
    auto *pointer = ::llvm::PointerType::get(*context, 0u);
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(*context), {pointer, pointer}, false);
    auto entry_name = "native_" +
                      std::string{extended_trig_name(operation)} + "_w" +
                      std::to_string(width);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::ExternalLinkage,
        entry_name, *module);
    auto *entry = ::llvm::BasicBlock::Create(
        *context, "entry", function);
    ::llvm::IRBuilder<> builder{entry};
    auto *input = builder.CreateAlignedLoad(
        vector_type, function->getArg(0u), ::llvm::Align{4u});
    ::llvm::Value *result = nullptr;
    switch (operation) {
        case ExtendedTrigOperation::tan:
            result = cpu::LLVMNativeMath::emit_tan_f32(
                *module, builder, input,
                cpu::LLVMNativeMathMode::precise);
            break;
        case ExtendedTrigOperation::asin:
            result = cpu::LLVMNativeMath::emit_asin_f32(
                *module, builder, input,
                cpu::LLVMNativeMathMode::precise);
            break;
        case ExtendedTrigOperation::acos:
            result = cpu::LLVMNativeMath::emit_acos_f32(
                *module, builder, input,
                cpu::LLVMNativeMathMode::precise);
            break;
        case ExtendedTrigOperation::atan:
            result = cpu::LLVMNativeMath::emit_atan_f32(
                *module, builder, input,
                cpu::LLVMNativeMathMode::precise);
            break;
    }
    builder.CreateAlignedStore(
        result, function->getArg(1u), ::llvm::Align{4u});
    builder.CreateRetVoid();
    return {std::move(context), std::move(module), std::move(entry_name)};
}

[[nodiscard]] std::optional<MathModule>
make_schedule_math_module(uint32_t width, bool fast_math = false) {
    xir::Module xir_module;
    auto *kernel = xir_module.create_kernel();
    kernel->set_name("schedule_native_math");
    auto *base = kernel->create_value_argument(Type::of<float>());
    auto *entry = kernel->create_body_block();
    auto *lane = xir_module.create_warp_lane_id();
    auto *one = xir_module.create_constant_one(Type::of<float>());
    auto scale_value = 0.125f;
    auto *scale = xir_module.create_constant(
        Type::of<float>(), &scale_value);
    auto scale10_value = 0.03125f;
    auto *scale10 = xir_module.create_constant(
        Type::of<float>(), &scale10_value);
    auto inverse_scale_value = 0.1f;
    auto *inverse_scale = xir_module.create_constant(
        Type::of<float>(), &inverse_scale_value);
    auto inverse_offset_value = -0.75f;
    auto *inverse_offset = xir_module.create_constant(
        Type::of<float>(), &inverse_offset_value);
    auto exp2_weight_value = 2.0f;
    auto *exp2_weight = xir_module.create_constant(
        Type::of<float>(), &exp2_weight_value);
    auto exp10_weight_value = 3.0f;
    auto *exp10_weight = xir_module.create_constant(
        Type::of<float>(), &exp10_weight_value);
    auto log2_weight_value = 5.0f;
    auto *log2_weight = xir_module.create_constant(
        Type::of<float>(), &log2_weight_value);
    auto log10_weight_value = 7.0f;
    auto *log10_weight = xir_module.create_constant(
        Type::of<float>(), &log10_weight_value);
    auto atan2_x_scale_value = 0.2f;
    auto *atan2_x_scale = xir_module.create_constant(
        Type::of<float>(), &atan2_x_scale_value);
    auto atan2_x_offset_value = -1.25f;
    auto *atan2_x_offset = xir_module.create_constant(
        Type::of<float>(), &atan2_x_offset_value);
    auto atan2_weight_value = 11.0f;
    auto *atan2_weight = xir_module.create_constant(
        Type::of<float>(), &atan2_weight_value);
    auto pow_weight_value = 13.0f;
    auto *pow_weight = xir_module.create_constant(
        Type::of<float>(), &pow_weight_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *lane_f32 = builder.static_cast_(Type::of<float>(), lane);
    auto *trig_input = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {base, lane_f32});
    auto *sin_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SIN, {trig_input});
    auto *cos_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::COS, {trig_input});
    auto *tan_input = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {lane_f32, scale10});
    auto *tan_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::TAN, {tan_input});
    auto *inverse_input_unbiased = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {lane_f32, inverse_scale});
    auto *inverse_input = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {inverse_input_unbiased, inverse_offset});
    auto *asin_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ASIN, {inverse_input});
    auto *acos_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ACOS, {inverse_input});
    auto *atan_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ATAN, {inverse_input});
    auto *atan2_x_unbiased = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {lane_f32, atan2_x_scale});
    auto *atan2_x = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {atan2_x_unbiased, atan2_x_offset});
    auto *atan2_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ATAN2,
        {inverse_input, atan2_x});
    auto *weighted_atan2 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {atan2_result, atan2_weight});
    auto *exp_input = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {lane_f32, scale});
    auto *exp_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::EXP, {exp_input});
    auto *exp2_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::EXP2, {exp_input});
    auto *exp10_input = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {lane_f32, scale10});
    auto *exp10_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::EXP10, {exp10_input});
    auto *log_input = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {lane_f32, one});
    auto *log_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::LOG, {log_input});
    auto *log2_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::LOG2, {log_input});
    auto *log10_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::LOG10, {log_input});
    auto *sinh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SINH, {inverse_input});
    auto *cosh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::COSH, {inverse_input});
    auto *tanh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::TANH, {inverse_input});
    auto *asinh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ASINH, {inverse_input});
    auto *acosh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ACOSH, {log_input});
    auto *atanh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ATANH, {inverse_input});
    auto *pow_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::POW,
        {log_input, tan_input});
    auto *weighted_pow = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {pow_result, pow_weight});
    auto *weighted_exp2 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {exp2_result, exp2_weight});
    auto *weighted_exp10 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {exp10_result, exp10_weight});
    auto *weighted_log2 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {log2_result, log2_weight});
    auto *weighted_log10 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {log10_result, log10_weight});
    auto *trig_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {sin_result, cos_result});
    auto *exp_log_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {exp_result, log_result});
    auto *derived_exp_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {weighted_exp2, weighted_exp10});
    auto *derived_log_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {weighted_log2, weighted_log10});
    auto *base_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {trig_sum, exp_log_sum});
    auto *derived_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {derived_exp_sum, derived_log_sum});
    auto *base_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {base_sum, derived_sum});
    auto *tan_inverse_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {tan_result, asin_result});
    auto *inverse_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {acos_result, atan_result});
    auto *extended_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {tan_inverse_sum, inverse_sum});
    auto *extended_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {extended_sum, weighted_atan2});
    auto *extended_pow_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {extended_result, weighted_pow});
    auto *hyperbolic_pair0 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {sinh_result, cosh_result});
    auto *hyperbolic_pair1 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {tanh_result, asinh_result});
    auto *hyperbolic_pair2 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {acosh_result, atanh_result});
    auto *hyperbolic_sum0 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {hyperbolic_pair0, hyperbolic_pair1});
    auto *hyperbolic_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {hyperbolic_sum0, hyperbolic_pair2});
    auto *non_hyperbolic_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {base_result, extended_pow_result});
    auto *result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {non_hyperbolic_result, hyperbolic_sum});
    result->set_name("native_math_result");
    builder.return_void();

    auto lowered = simd::schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        for (auto &&diagnostic : lowered.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return std::nullopt;
    }
    std::optional<simd::schedule::ValueId> result_id;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "native_math_result") {
            result_id = value.id;
        }
    }
    if (!result_id) { return std::nullopt; }
    for (auto &block : lowered.function->blocks()) {
        if (std::holds_alternative<simd::schedule::ReturnTerminator>(
                block.terminator)) {
            block.terminator =
                simd::schedule::ReturnTerminator{result_id};
        }
    }
    if (!simd::schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }

    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "schedule-native-math", *context);
    auto entry_name = std::string{"schedule_native_math_w"} +
                      std::to_string(width) +
                      (fast_math ? "_fast" : "_precise");
    auto codegen = simd::lower_schedule_to_llvm(
        *module, *lowered.function, width, entry_name, fast_math);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return std::nullopt;
    }
    if (codegen.argument_buffer_size != 16u ||
        ::llvm::verifyModule(*module, &::llvm::errs())) {
        return std::nullopt;
    }
    return MathModule{
        std::move(context), std::move(module), std::move(entry_name)};
}

[[nodiscard]] std::optional<MathModule>
make_uniform_schedule_math_module(
    uint32_t width, bool fast_math = false) {
    xir::Module xir_module;
    auto *kernel = xir_module.create_kernel();
    kernel->set_name("schedule_uniform_math");
    auto *input = kernel->create_value_argument(Type::of<float>());
    auto *entry = kernel->create_body_block();
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *sin_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SIN, {input});
    auto *cos_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::COS, {input});
    auto *tan_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::TAN, {input});
    auto *asin_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ASIN, {input});
    auto *acos_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ACOS, {input});
    auto *atan_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ATAN, {input});
    auto *one = xir_module.create_constant_one(Type::of<float>());
    auto *atan2_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ATAN2, {input, one});
    auto *pow_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::POW, {input, input});
    auto *exp_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::EXP, {input});
    auto *log_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::LOG, {input});
    auto *sinh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SINH, {input});
    auto *cosh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::COSH, {input});
    auto *tanh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::TANH, {input});
    auto *asinh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ASINH, {input});
    auto *acosh_input = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {input, one});
    auto *acosh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ACOSH, {acosh_input});
    auto *atanh_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::ATANH, {input});
    auto *trig_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {sin_result, cos_result});
    auto *exp_log_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {exp_result, log_result});
    auto *base_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {trig_sum, exp_log_sum});
    auto *tan_inverse_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {tan_result, asin_result});
    auto *inverse_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {acos_result, atan_result});
    auto *extended_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {tan_inverse_sum, inverse_sum});
    auto *extended_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {extended_sum, atan2_result});
    auto *extended_pow_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {extended_result, pow_result});
    auto *hyperbolic_pair0 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {sinh_result, cosh_result});
    auto *hyperbolic_pair1 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {tanh_result, asinh_result});
    auto *hyperbolic_pair2 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {acosh_result, atanh_result});
    auto *hyperbolic_sum0 = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {hyperbolic_pair0, hyperbolic_pair1});
    auto *hyperbolic_sum = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {hyperbolic_sum0, hyperbolic_pair2});
    auto *non_hyperbolic_result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {base_result, extended_pow_result});
    auto *result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {non_hyperbolic_result, hyperbolic_sum});
    result->set_name("uniform_math_result");
    builder.return_void();

    auto lowered = simd::schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) { return std::nullopt; }
    std::optional<simd::schedule::ValueId> result_id;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "uniform_math_result") {
            if (value.value_class !=
                simd::schedule::ValueClass::warp_uniform) {
                return std::nullopt;
            }
            result_id = value.id;
        }
    }
    if (!result_id) { return std::nullopt; }
    for (auto &block : lowered.function->blocks()) {
        if (std::holds_alternative<simd::schedule::ReturnTerminator>(
                block.terminator)) {
            block.terminator =
                simd::schedule::ReturnTerminator{result_id};
        }
    }
    if (!simd::schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }

    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "schedule-uniform-math", *context);
    auto entry_name = std::string{"schedule_uniform_math_w"} +
                      std::to_string(width) +
                      (fast_math ? "_fast" : "_precise");
    auto codegen = simd::lower_schedule_to_llvm(
        *module, *lowered.function, width, entry_name, fast_math);
    if (!codegen.succeeded() || codegen.argument_buffer_size != 16u ||
        ::llvm::verifyModule(*module, &::llvm::errs())) {
        return std::nullopt;
    }
    return MathModule{
        std::move(context), std::move(module), std::move(entry_name)};
}

[[nodiscard]] std::string module_text(const ::llvm::Module &module) {
    std::string text;
    ::llvm::raw_string_ostream stream{text};
    module.print(stream, nullptr);
    stream.flush();
    return text;
}

[[nodiscard]] size_t count_occurrences(
    std::string_view text, std::string_view needle) {
    auto count = size_t{0u};
    for (auto position = text.find(needle);
         position != std::string_view::npos;
         position = text.find(needle, position + needle.size())) {
        ++count;
    }
    return count;
}

[[nodiscard]] bool approximately_equal(float actual, float expected) {
    if (std::isnan(expected)) { return std::isnan(actual); }
    if (std::isinf(expected)) {
        return actual == expected &&
               std::signbit(actual) == std::signbit(expected);
    }
    if (expected == 0.0f) {
        return actual == 0.0f &&
               std::signbit(actual) == std::signbit(expected);
    }
    auto error = std::abs(actual - expected);
    return error <= 2.0e-6f ||
           error <= 2.0e-6f * std::abs(expected);
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

[[nodiscard]] bool within_ulp(
    float actual, float expected, uint64_t max_ulp) {
    if (std::isnan(expected)) { return std::isnan(actual); }
    if (std::isinf(expected)) { return actual == expected; }
    if (expected == 0.0f) {
        return actual == 0.0f &&
               std::signbit(actual) == std::signbit(expected);
    }
    return std::isfinite(actual) &&
           ulp_distance(actual, expected) <= max_ulp;
}

[[nodiscard]] float extended_trig_reference(
    ExtendedTrigOperation operation, float x) {
    switch (operation) {
        case ExtendedTrigOperation::tan: return std::tan(x);
        case ExtendedTrigOperation::asin: return std::asin(x);
        case ExtendedTrigOperation::acos: return std::acos(x);
        case ExtendedTrigOperation::atan: return std::atan(x);
    }
    return std::numeric_limits<float>::quiet_NaN();
}

template<size_t Width>
[[nodiscard]] bool test_extended_trig_width(
    ExtendedTrigOperation operation) {
    auto shape_module = make_extended_trig_module(Width, operation);
    CHECK(!::llvm::verifyModule(*shape_module.module, &::llvm::errs()));
    auto name = extended_trig_name(operation);
    auto ir = module_text(*shape_module.module);
    CHECK(ir.find("llvm." + std::string{name}) == std::string::npos);
    CHECK(ir.find("extractelement") == std::string::npos);
    CHECK(ir.find("insertelement") == std::string::npos);
    CHECK(ir.find("<" + std::to_string(Width) + " x float>") !=
          std::string::npos);
    if (operation == ExtendedTrigOperation::tan) {
        CHECK(ir.find("llvm.sin") == std::string::npos);
        CHECK(ir.find("llvm.cos") == std::string::npos);
    }

    if constexpr (Width == 8u) {
        auto assembly_module = make_extended_trig_module(
            Width, operation);
        simd::LLVMJIT assembly_target;
        CHECK(assembly_target.succeeded());
        auto assembly = assembly_target.emit_assembly(
            std::move(assembly_module.module),
            std::move(assembly_module.context));
        if (assembly.empty()) {
            std::cerr << assembly_target.error() << '\n';
            return false;
        }
        std::transform(
            assembly.begin(), assembly.end(), assembly.begin(),
            [](unsigned char c) noexcept {
                return static_cast<char>(std::tolower(c));
            });
        CHECK(assembly.find(std::string{name} + "f") ==
              std::string::npos);
    }

    auto executable_module = make_extended_trig_module(Width, operation);
    auto entry_name = executable_module.entry_name;
    simd::LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(
        std::move(executable_module.module),
        std::move(executable_module.context)));
    using Entry = void(const float *, float *);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(entry_name));
    CHECK(entry != nullptr);

    constexpr std::array general_corpus{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::max(),
        -1.0e10f,
        -39001.0f,
        -124.75f,
        -30.0f,
        -2.0f,
        -1.5707963267948966192f,
        -1.0f,
        -0.5f,
        -0.2f,
        -1.0e-20f,
        -0.0f,
        0.0f,
        1.0e-20f,
        0.2f,
        0.5f,
        1.0f,
        1.5707963267948966192f,
        2.0f,
        30.0f,
        126.0f,
        39001.0f,
        1.0e10f,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    const std::array unit_corpus{
        -std::numeric_limits<float>::infinity(),
        -2.0f,
        std::nextafter(-1.0f, -2.0f),
        -1.0f,
        std::nextafter(-1.0f, 0.0f),
        -0.75f,
        -0.5f,
        -0.2f,
        -1.0e-20f,
        -0.0f,
        0.0f,
        1.0e-20f,
        0.2f,
        0.5f,
        0.75f,
        std::nextafter(1.0f, 0.0f),
        1.0f,
        std::nextafter(1.0f, 2.0f),
        2.0f,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    alignas(64) std::array<float, Width> input{};
    alignas(64) std::array<float, Width> output{};
    auto check_batch = [&](const char *source) {
        entry(input.data(), output.data());
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            auto expected = extended_trig_reference(
                operation, input[lane]);
            if (!within_ulp(output[lane], expected, 4u)) {
                std::cerr << "native " << name << ' ' << source
                          << " W" << Width << " lane " << lane
                          << " bits=" << std::hex
                          << std::bit_cast<uint32_t>(input[lane])
                          << std::dec
                          << " input=" << input[lane]
                          << " actual=" << output[lane]
                          << " expected=" << expected
                          << " ulp=" << ulp_distance(output[lane], expected) << '\n';
                return false;
            }
        }
        return true;
    };
    auto test_corpus = [&](auto &&corpus) {
        for (auto base = size_t{0u}; base < corpus.size();
             base += Width) {
            for (auto lane = size_t{0u}; lane < Width; lane++) {
                input[lane] = corpus[(base + lane) % corpus.size()];
            }
            if (!check_batch("corpus")) { return false; }
        }
        return true;
    };
    auto unit_domain = operation == ExtendedTrigOperation::asin ||
                       operation == ExtendedTrigOperation::acos;
    CHECK(unit_domain ? test_corpus(unit_corpus) :
                        test_corpus(general_corpus));

    auto state = uint32_t{0xa4093822u};
    for (auto base = size_t{0u}; base < 16384u; base += Width) {
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            state = state * 1664525u + 1013904223u;
            input[lane] = std::bit_cast<float>(state);
        }
        CHECK(check_batch("raw-bits"));
    }
    if (unit_domain) {
        for (auto base = size_t{0u}; base < 16384u; base += Width) {
            for (auto lane = size_t{0u}; lane < Width; lane++) {
                state = state * 1664525u + 1013904223u;
                input[lane] = static_cast<float>(
                                  std::bit_cast<int32_t>(state)) /
                              2147483648.0f;
            }
            CHECK(check_batch("unit-domain"));
        }
    }
    return true;
}

[[nodiscard]] bool test_extended_trig_operation(
    ExtendedTrigOperation operation) {
    return test_extended_trig_width<2u>(operation) &&
           test_extended_trig_width<3u>(operation) &&
           test_extended_trig_width<4u>(operation) &&
           test_extended_trig_width<8u>(operation) &&
           test_extended_trig_width<16u>(operation);
}

[[nodiscard]] bool test_jit_object_capture() {
    auto executable_module = make_trig_module(4u, false);
    auto entry_name = executable_module.entry_name;
    simd::LLVMJIT jit{true};
    CHECK(jit.succeeded());
    CHECK(jit.object().empty());
    CHECK(jit.add_module(
        std::move(executable_module.module),
        std::move(executable_module.context)));
    // ORC materializes lazily: the compiler-object transform runs at lookup.
    CHECK(jit.object().empty());
    simd::LLVMJIT moved_jit{std::move(jit)};
    CHECK(moved_jit.lookup(entry_name) != nullptr);
    CHECK(!moved_jit.object().empty());
    return true;
}

template<size_t Width>
[[nodiscard]] bool test_width(bool cosine) {
    auto shape_module = make_trig_module(Width, cosine);
    CHECK(!::llvm::verifyModule(*shape_module.module, &::llvm::errs()));
    auto ir = module_text(*shape_module.module);
    CHECK(ir.find(cosine ? "llvm.cos" : "llvm.sin") ==
          std::string::npos);
    CHECK(ir.find("extractelement") == std::string::npos);
    CHECK(ir.find("insertelement") == std::string::npos);
    CHECK(ir.find("llvm.masked.gather.v" + std::to_string(Width)) !=
          std::string::npos);
    CHECK(ir.find("<" + std::to_string(Width) + " x float>") !=
          std::string::npos);

    auto assembly_module = make_trig_module(Width, cosine);
    simd::LLVMJIT assembly_target;
    CHECK(assembly_target.succeeded());
    auto assembly = assembly_target.emit_assembly(
        std::move(assembly_module.module),
        std::move(assembly_module.context));
    if (assembly.empty()) {
        std::cerr << assembly_target.error() << '\n';
        return false;
    }
    std::transform(
        assembly.begin(), assembly.end(), assembly.begin(),
        [](unsigned char c) noexcept {
            return static_cast<char>(std::tolower(c));
        });
    CHECK(assembly.find(cosine ? "cosf" : "sinf") ==
          std::string::npos);

    auto executable_module = make_trig_module(Width, cosine);
    auto entry_name = executable_module.entry_name;
    simd::LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(
        std::move(executable_module.module),
        std::move(executable_module.context)));
    using Entry = void(const float *, float *);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(entry_name));
    if (entry == nullptr) { std::cerr << jit.error() << '\n'; }
    CHECK(entry != nullptr);

    constexpr std::array corpus{
        -0.0f,
        0.0f,
        -1.0e-20f,
        1.0e-6f,
        -0.2f,
        0.5f,
        -1.0f,
        1.5707963267948966192f,
        -2.0f,
        30.0f,
        -124.75f,
        126.0f,
        -38999.0f,
        39001.0f,
        -1.0e10f,
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    alignas(64) std::array<float, Width> input{};
    alignas(64) std::array<float, Width> output{};
    for (auto base = size_t{0u}; base < corpus.size(); base += Width) {
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            input[lane] = corpus[(base + lane) % corpus.size()];
            output[lane] = 1234.0f;
        }
        entry(input.data(), output.data());
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            auto expected = cosine ?
                                std::cos(input[lane]) :
                                std::sin(input[lane]);
            if (!within_ulp(output[lane], expected, 4u)) {
                std::cerr << "native " << (cosine ? "cos" : "sin")
                          << " W" << Width << " lane " << lane
                          << " input=" << input[lane]
                          << " actual=" << output[lane]
                          << " expected=" << expected
                          << " ulp=" << ulp_distance(output[lane], expected) << '\n';
                return false;
            }
        }
    }

    // A deterministic bit-pattern sweep covers every exponent class and
    // continually exercises the large-argument reduction path. Keeping this
    // here turns numerical failures found by the formal audit into a stable
    // regression instead of relying only on hand-picked values.
    auto state = uint32_t{0x243f6a88u};
    for (auto base = size_t{0u}; base < 16384u; base += Width) {
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            state = state * 1664525u + 1013904223u;
            input[lane] = std::bit_cast<float>(state);
        }
        entry(input.data(), output.data());
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            auto expected = cosine ?
                                std::cos(input[lane]) :
                                std::sin(input[lane]);
            if (!within_ulp(output[lane], expected, 4u)) {
                std::cerr << "native " << (cosine ? "cos" : "sin")
                          << " random W" << Width
                          << " lane " << lane
                          << " bits=" << std::hex
                          << std::bit_cast<uint32_t>(input[lane])
                          << std::dec
                          << " input=" << input[lane]
                          << " actual=" << output[lane]
                          << " expected=" << expected
                          << " ulp=" << ulp_distance(output[lane], expected) << '\n';
                return false;
            }
        }
    }
    return true;
}

template<size_t Width>
[[nodiscard]] bool test_exp_log_width(bool logarithm) {
    auto shape_module = make_exp_log_module(Width, logarithm);
    CHECK(!::llvm::verifyModule(*shape_module.module, &::llvm::errs()));
    auto ir = module_text(*shape_module.module);
    CHECK(ir.find(logarithm ? "llvm.log" : "llvm.exp") ==
          std::string::npos);
    CHECK(ir.find("extractelement") == std::string::npos);
    CHECK(ir.find("insertelement") == std::string::npos);
    CHECK(ir.find("<" + std::to_string(Width) + " x float>") !=
          std::string::npos);

    auto assembly_module = make_exp_log_module(Width, logarithm);
    simd::LLVMJIT assembly_target;
    CHECK(assembly_target.succeeded());
    auto assembly = assembly_target.emit_assembly(
        std::move(assembly_module.module),
        std::move(assembly_module.context));
    if (assembly.empty()) {
        std::cerr << assembly_target.error() << '\n';
        return false;
    }
    std::transform(
        assembly.begin(), assembly.end(), assembly.begin(),
        [](unsigned char c) noexcept {
            return static_cast<char>(std::tolower(c));
        });
    CHECK(assembly.find(logarithm ? "logf" : "expf") ==
          std::string::npos);

    auto executable_module = make_exp_log_module(Width, logarithm);
    auto entry_name = executable_module.entry_name;
    simd::LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(
        std::move(executable_module.module),
        std::move(executable_module.context)));
    using Entry = void(const float *, float *);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(entry_name));
    CHECK(entry != nullptr);

    constexpr std::array corpus{
        -std::numeric_limits<float>::infinity(),
        -104.0f,
        -88.0f,
        -1.0f,
        -0.0f,
        0.0f,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        0.5f,
        1.0f,
        2.0f,
        10.0f,
        80.0f,
        88.0f,
        100.0f,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    alignas(64) std::array<float, Width> input{};
    alignas(64) std::array<float, Width> output{};
    auto check_batch = [&]() {
        entry(input.data(), output.data());
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            auto expected = logarithm ?
                                std::log(input[lane]) :
                                std::exp(input[lane]);
            auto max_ulp = logarithm ? 5u : 4u;
            if (!within_ulp(output[lane], expected, max_ulp)) {
                std::cerr << "native " << (logarithm ? "log" : "exp")
                          << " W" << Width << " lane " << lane
                          << " bits=" << std::hex
                          << std::bit_cast<uint32_t>(input[lane])
                          << std::dec
                          << " input=" << input[lane]
                          << " actual=" << output[lane]
                          << " expected=" << expected
                          << " ulp=" << ulp_distance(output[lane], expected) << '\n';
                return false;
            }
        }
        return true;
    };
    for (auto base = size_t{0u}; base < corpus.size(); base += Width) {
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            input[lane] = corpus[(base + lane) % corpus.size()];
        }
        CHECK(check_batch());
    }
    auto state = uint32_t{0x13198a2eu};
    for (auto base = size_t{0u}; base < 16384u; base += Width) {
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            state = state * 1664525u + 1013904223u;
            input[lane] = std::bit_cast<float>(state);
        }
        CHECK(check_batch());
    }
    return true;
}

template<size_t Width>
[[nodiscard]] bool test_schedule_width(bool fast_math = false) {
    auto shape_module = make_schedule_math_module(Width, fast_math);
    CHECK(shape_module.has_value());
    auto ir = module_text(*shape_module->module);
    CHECK(ir.find("llvm.sin.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.cos.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.tan.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.asin.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.acos.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.atan.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.atan2.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.exp.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.log.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.exp2.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.exp10.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.log2.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.log10.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    CHECK(ir.find("llvm.pow.v" + std::to_string(Width) + "f32") ==
          std::string::npos);
    for (auto operation : {"sinh", "cosh", "tanh",
                           "asinh", "acosh", "atanh"}) {
        CHECK(ir.find("llvm." + std::string{operation} + ".v" +
                      std::to_string(Width) + "f32") ==
              std::string::npos);
    }
    CHECK(ir.find("luisa.cpu.native_math") != std::string::npos);
    CHECK(ir.find("__luisa_cpu_native_sin_f32_v" +
                  std::to_string(Width) +
                  (fast_math ? "_fast" : "_u10")) !=
          std::string::npos);
    CHECK(ir.find("__luisa_cpu_native_exp2_f32_v" +
                  std::to_string(Width) +
                  (fast_math ? "_fast" : "_u10")) !=
          std::string::npos);
    CHECK(ir.find("__luisa_cpu_native_exp10_f32_v" +
                  std::to_string(Width) +
                  (fast_math ? "_fast" : "_u10")) !=
          std::string::npos);
    CHECK(ir.find("__luisa_cpu_native_log2_f32_v" +
                  std::to_string(Width) +
                  (fast_math ? "_fast" : "_u35")) !=
          std::string::npos);
    CHECK(ir.find("__luisa_cpu_native_log10_f32_v" +
                  std::to_string(Width) +
                  (fast_math ? "_fast" : "_u35")) !=
          std::string::npos);
    CHECK(ir.find("__luisa_cpu_native_atan2_f32_v" +
                  std::to_string(Width) +
                  (fast_math ? "_fast" : "_u35")) !=
          std::string::npos);
    CHECK(ir.find("__luisa_cpu_native_pow_f32_v" +
                  std::to_string(Width) +
                  (fast_math ? "_fast" : "_u10")) !=
          std::string::npos);
    for (auto operation : {"sinh", "cosh", "tanh",
                           "asinh", "acosh", "atanh"}) {
        CHECK(ir.find("__luisa_cpu_native_" +
                      std::string{operation} + "_f32_v" +
                      std::to_string(Width) +
                      (fast_math ? "_fast" : "_precise")) !=
              std::string::npos);
    }
    CHECK(ir.find("llvm.x86.") == std::string::npos);
    CHECK(ir.find("llvm.aarch64.") == std::string::npos);

    if constexpr (Width == 8u) {
        auto assembly_module = make_schedule_math_module(
            Width, fast_math);
        CHECK(assembly_module.has_value());
        simd::LLVMJIT assembly_target;
        CHECK(assembly_target.succeeded());
        auto assembly = assembly_target.emit_assembly(
            std::move(assembly_module->module),
            std::move(assembly_module->context));
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
        CHECK(assembly.find("atan2f") == std::string::npos);
        CHECK(assembly.find("expf") == std::string::npos);
        CHECK(assembly.find("logf") == std::string::npos);
        CHECK(assembly.find("exp2f") == std::string::npos);
        CHECK(assembly.find("exp10f") == std::string::npos);
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

    auto executable_module = make_schedule_math_module(
        Width, fast_math);
    CHECK(executable_module.has_value());
    auto entry_name = executable_module->entry_name;
    simd::LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(
        std::move(executable_module->module),
        std::move(executable_module->context)));
    using Entry = void(
        const void *, float *, const simd::SIMDPacketLaunchConfig *,
        uint32_t);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(entry_name));
    CHECK(entry != nullptr);

    alignas(16) float base = 39001.25f;
    simd::SIMDPacketLaunchConfig config{};
    config.dispatch_size[0u] = Width;
    config.dispatch_size[1u] = 1u;
    config.dispatch_size[2u] = 1u;
    config.block_size[0u] = Width;
    config.block_size[1u] = 1u;
    config.block_size[2u] = 1u;
    for (auto active_lanes :
         {static_cast<uint32_t>(Width),
          static_cast<uint32_t>(Width - 1u)}) {
        std::array<float, Width> output{};
        output.fill(1234.0f);
        entry(&base, output.data(), &config, active_lanes);
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            if (lane >= active_lanes) {
                CHECK(output[lane] == 1234.0f);
            } else {
                auto lane_f32 = static_cast<float>(lane);
                auto trig_input = base + lane_f32;
                auto inverse_input = lane_f32 * 0.1f - 0.75f;
                auto expected = std::sin(trig_input) +
                                std::cos(trig_input) +
                                std::tan(lane_f32 * 0.03125f) +
                                std::asin(inverse_input) +
                                std::acos(inverse_input) +
                                std::atan(inverse_input) +
                                std::exp(lane_f32 * 0.125f) +
                                std::log(lane_f32 + 1.0f) +
                                2.0f * std::exp2(lane_f32 * 0.125f) +
                                3.0f * std::pow(10.0f, lane_f32 * 0.03125f) +
                                5.0f * std::log2(lane_f32 + 1.0f) +
                                7.0f * std::log10(lane_f32 + 1.0f) +
                                11.0f * std::atan2(
                                            inverse_input,
                                            lane_f32 * 0.2f - 1.25f) +
                                13.0f * std::pow(
                                            lane_f32 + 1.0f,
                                            lane_f32 * 0.03125f) +
                                std::sinh(inverse_input) +
                                std::cosh(inverse_input) +
                                std::tanh(inverse_input) +
                                std::asinh(inverse_input) +
                                std::acosh(lane_f32 + 1.0f) +
                                std::atanh(inverse_input);
                auto equal = fast_math ?
                                 std::abs(output[lane] - expected) <=
                                     2.0e-3f * (1.0f + std::abs(expected)) :
                                 std::abs(output[lane] - expected) <=
                                     1.0e-5f * (1.0f + std::abs(expected));
                if (!equal) {
                    std::cerr << "schedule native math W" << Width
                              << (fast_math ? " fast" : " precise")
                              << " lane=" << lane
                              << " actual=" << output[lane]
                              << " expected=" << expected
                              << " error="
                              << std::abs(output[lane] - expected) << '\n';
                }
                CHECK(equal);
            }
        }
    }
    return true;
}

[[nodiscard]] bool test_uniform_schedule_math(bool fast_math = false) {
    constexpr auto width = 8u;
    auto shape_module = make_uniform_schedule_math_module(
        width, fast_math);
    CHECK(shape_module.has_value());
    auto ir = module_text(*shape_module->module);
    CHECK(ir.find("__luisa_cpu_native_") == std::string::npos);
    CHECK(count_occurrences(ir, "call float @llvm.sin.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.cos.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.tan.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.asin.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.acos.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.atan.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.atan2.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.pow.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.exp.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.log.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.sinh.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.cosh.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @llvm.tanh.f32") == 1u);
    CHECK(count_occurrences(ir, "call float @asinhf") == 1u);
    CHECK(count_occurrences(ir, "call float @acoshf") == 1u);
    CHECK(count_occurrences(ir, "call float @atanhf") == 1u);
    CHECK(ir.find("llvm.sin.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.cos.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.tan.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.asin.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.acos.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.atan.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.atan2.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.exp.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.log.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.pow.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.sinh.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.cosh.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.tanh.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.asinh.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.acosh.v8f32") == std::string::npos);
    CHECK(ir.find("llvm.atanh.v8f32") == std::string::npos);

    auto executable = make_uniform_schedule_math_module(
        width, fast_math);
    CHECK(executable.has_value());
    auto entry_name = executable->entry_name;
    simd::LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(
        std::move(executable->module),
        std::move(executable->context)));
    using Entry = void(
        const void *, float *, const simd::SIMDPacketLaunchConfig *,
        uint32_t);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(entry_name));
    CHECK(entry != nullptr);
    alignas(16) float input = 0.5f;
    std::array<float, width> output{};
    output.fill(1234.0f);
    simd::SIMDPacketLaunchConfig config{};
    config.dispatch_size[0u] = 7u;
    config.dispatch_size[1u] = 1u;
    config.dispatch_size[2u] = 1u;
    config.block_size[0u] = width;
    config.block_size[1u] = 1u;
    config.block_size[2u] = 1u;
    entry(&input, output.data(), &config, 7u);
    auto expected = std::sin(input) + std::cos(input) +
                    std::tan(input) + std::asin(input) +
                    std::acos(input) + std::atan(input) +
                    std::atan2(input, 1.0f) +
                    std::pow(input, input) +
                    std::exp(input) + std::log(input) +
                    std::sinh(input) + std::cosh(input) +
                    std::tanh(input) + std::asinh(input) +
                    std::acosh(input + 1.0f) + std::atanh(input);
    for (auto lane = size_t{0u}; lane < width; lane++) {
        CHECK(lane < 7u ?
                  approximately_equal(output[lane], expected) :
                  output[lane] == 1234.0f);
    }
    return true;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc == 2 && std::string_view{argv[1]} == "--fast-only") {
        return test_schedule_width<1u>(true) &&
                       test_schedule_width<2u>(true) &&
                       test_schedule_width<4u>(true) &&
                       test_schedule_width<8u>(true) &&
                       test_schedule_width<16u>(true) &&
                       test_uniform_schedule_math(true) &&
                       test_llvm_native_math_fast() ?
                   0 :
                   1;
    }
    return test_jit_object_capture() &&
                   test_width<2u>(false) &&
                   test_width<3u>(false) &&
                   test_width<4u>(false) &&
                   test_width<8u>(false) &&
                   test_width<16u>(false) &&
                   test_width<2u>(true) &&
                   test_width<3u>(true) &&
                   test_width<4u>(true) &&
                   test_width<8u>(true) &&
                   test_width<16u>(true) &&
                   test_extended_trig_operation(
                       ExtendedTrigOperation::tan) &&
                   test_extended_trig_operation(
                       ExtendedTrigOperation::asin) &&
                   test_extended_trig_operation(
                       ExtendedTrigOperation::acos) &&
                   test_extended_trig_operation(
                       ExtendedTrigOperation::atan) &&
                   test_exp_log_width<2u>(false) &&
                   test_exp_log_width<3u>(false) &&
                   test_exp_log_width<4u>(false) &&
                   test_exp_log_width<8u>(false) &&
                   test_exp_log_width<16u>(false) &&
                   test_exp_log_width<2u>(true) &&
                   test_exp_log_width<3u>(true) &&
                   test_exp_log_width<4u>(true) &&
                   test_exp_log_width<8u>(true) &&
                   test_exp_log_width<16u>(true) &&
                   test_schedule_width<1u>() &&
                   test_schedule_width<2u>() &&
                   test_schedule_width<4u>() &&
                   test_schedule_width<8u>() &&
                   test_schedule_width<16u>() &&
                   test_uniform_schedule_math() &&
                   test_schedule_width<1u>(true) &&
                   test_schedule_width<2u>(true) &&
                   test_schedule_width<4u>(true) &&
                   test_schedule_width<8u>(true) &&
                   test_schedule_width<16u>(true) &&
                   test_uniform_schedule_math(true) &&
                   test_llvm_native_math_fast() ?
               0 :
               1;
}
