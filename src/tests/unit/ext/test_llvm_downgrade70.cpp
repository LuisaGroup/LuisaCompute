#include "llvm_downgrade.h"

#include <array>
#include <memory>
#include <string>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>

namespace {

using Mode = llvm::DenormalMode;

struct TestCase {
    const char *name;
    Mode default_mode;
    Mode f32_mode;
    const char *default_legacy;
    const char *f32_legacy;
};

// Every field differs from the other three, so swapping output/input or
// default/f32 cannot accidentally pass. The second function reverses them.
constexpr std::array cases{
    TestCase{"forward", {Mode::IEEE, Mode::PositiveZero}, {Mode::PreserveSign, Mode::Dynamic}, "ieee,positive-zero", "preserve-sign,dynamic"},
    TestCase{"reverse", {Mode::Dynamic, Mode::PreserveSign}, {Mode::PositiveZero, Mode::IEEE}, "dynamic,preserve-sign", "positive-zero,ieee"}};

bool check_mode(llvm::StringRef stage, llvm::StringRef function,
                llvm::StringRef field, Mode actual, Mode expected) {
    const bool valid = actual.Output == expected.Output && actual.Input == expected.Input;
    llvm::outs() << stage << ' ' << function << ' ' << field
                 << " output=" << llvm::denormalModeKindName(actual.Output)
                 << " input=" << llvm::denormalModeKindName(actual.Input)
                 << (valid ? " PASS\n" : " FAIL\n");
    if (!valid) {
        llvm::errs() << "Expected " << expected.str() << ", got " << actual.str() << '\n';
    }
    return valid;
}

bool check_function(llvm::StringRef stage, const llvm::Module &module, const TestCase &test) {
    const auto *function = module.getFunction(test.name);
    if (function == nullptr || function->isDeclaration() || !function->getReturnType()->isFloatTy()) {
        llvm::errs() << stage << ": missing defined f32 function " << test.name << '\n';
        return false;
    }
#if LLVM_VERSION_MAJOR >= 23
    if (!function->hasFnAttribute(llvm::Attribute::DenormalFPEnv)) {
        llvm::errs() << stage << ": missing modern DenormalFPEnv after SDK reader upgrade\n";
        return false;
    }
    const auto environment = function->getDenormalFPEnv();
    const auto default_mode = environment.DefaultMode;
    const auto f32_mode = environment.F32Mode;
#else
    // Verify the raw legacy strings as well as their interpreted four fields.
    if (function->getFnAttribute("denormal-fp-math").getValueAsString() != test.default_legacy ||
        function->getFnAttribute("denormal-fp-math-f32").getValueAsString() != test.f32_legacy) {
        llvm::errs() << stage << ": exact legacy attributes differ for " << test.name << '\n';
        return false;
    }
    const auto default_mode = function->getDenormalModeRaw();
    const auto f32_mode = function->getDenormalModeF32Raw();
#endif
    bool valid = check_mode(stage, test.name, "default", default_mode, test.default_mode);
    valid &= check_mode(stage, test.name, "f32", f32_mode, test.f32_mode);
    // Check effective semantics too: default applies to f64; f32 uses its override.
    valid &= check_mode(stage, test.name, "effective-f64",
                        function->getDenormalMode(llvm::APFloat::IEEEdouble()), test.default_mode);
    valid &= check_mode(stage, test.name, "effective-f32",
                        function->getDenormalMode(llvm::APFloat::IEEEsingle()), test.f32_mode);
    return valid;
}

enum class SplatKind { half,
                       f32,
                       f64,
                       fp80,
                       fp128,
                       ppc128,
                       integer };

struct SplatCase {
    const char *name;
    SplatKind kind;
    unsigned width;
    const char *bits;
};

constexpr std::array splat_cases{
    SplatCase{"half_zero", SplatKind::half, 16u, "0000"},
    SplatCase{"half_negzero", SplatKind::half, 16u, "8000"},
    SplatCase{"half_one", SplatKind::half, 16u, "3c00"},
    SplatCase{"half_qnan", SplatKind::half, 16u, "7e55"},
    SplatCase{"half_snan", SplatKind::half, 16u, "7c55"},
    SplatCase{"half_subnormal", SplatKind::half, 16u, "0001"},
    SplatCase{"float_zero", SplatKind::f32, 32u, "00000000"},
    SplatCase{"float_negzero", SplatKind::f32, 32u, "80000000"},
    SplatCase{"float_one", SplatKind::f32, 32u, "3f800000"},
    SplatCase{"float_qnan", SplatKind::f32, 32u, "7fc12345"},
    SplatCase{"float_snan", SplatKind::f32, 32u, "7f812345"},
    SplatCase{"float_subnormal", SplatKind::f32, 32u, "00000001"},
    SplatCase{"double_zero", SplatKind::f64, 64u, "0000000000000000"},
    SplatCase{"double_negzero", SplatKind::f64, 64u, "8000000000000000"},
    SplatCase{"double_one", SplatKind::f64, 64u, "3ff0000000000000"},
    SplatCase{"double_qnan", SplatKind::f64, 64u, "7ff8123456789abc"},
    SplatCase{"double_snan", SplatKind::f64, 64u, "7ff0123456789abc"},
    SplatCase{"double_subnormal", SplatKind::f64, 64u, "0000000000000001"},
    SplatCase{"fp80_one", SplatKind::fp80, 80u, "3fff8000000000000001"},
    SplatCase{"fp128_one", SplatKind::fp128, 128u, "3fff000000000000123456789abcdef0"},
    SplatCase{"ppc128_one", SplatKind::ppc128, 128u, "00000000000000003ff0000000000000"},
    SplatCase{"i1_true", SplatKind::integer, 1u, "1"},
    SplatCase{"i8_allones", SplatKind::integer, 8u, "ff"},
    SplatCase{"i16_high", SplatKind::integer, 16u, "8001"},
    SplatCase{"i32_pattern", SplatKind::integer, 32u, "deadbeef"},
    SplatCase{"i64_pattern", SplatKind::integer, 64u, "fedcba9876543210"},
    SplatCase{"i128_pattern", SplatKind::integer, 128u, "fedcba98765432100123456789abcdef"}};

llvm::Type *splat_element_type(llvm::LLVMContext &context, const SplatCase &test) {
    switch (test.kind) {
        case SplatKind::half: return llvm::Type::getHalfTy(context);
        case SplatKind::f32: return llvm::Type::getFloatTy(context);
        case SplatKind::f64: return llvm::Type::getDoubleTy(context);
        case SplatKind::fp80: return llvm::Type::getX86_FP80Ty(context);
        case SplatKind::fp128: return llvm::Type::getFP128Ty(context);
        case SplatKind::ppc128: return llvm::Type::getPPC_FP128Ty(context);
        case SplatKind::integer: return llvm::Type::getIntNTy(context, test.width);
    }
    return nullptr;
}

llvm::APInt splat_bits(const SplatCase &test, bool function_local = false) {
    llvm::APInt bits{test.width, test.bits, 16u};
    // Function-only constants differ from their globals, forcing function-local
    // enumeration as well as module enumeration. Change a low significand bit;
    // preserve FP sign/exponent and x87's explicit integer bit.
    if (function_local) { bits.flipBit(test.width == 1u ? 0u : 1u); }
    return bits;
}

llvm::Constant *make_splat(llvm::LLVMContext &context, const SplatCase &test,
                           bool function_local = false) {
    auto *element = splat_element_type(context, test);
    auto *vector = llvm::FixedVectorType::get(element, 4u);
    auto bits = splat_bits(test, function_local);
    // Use the public typed factory itself: LLVM 22 returns an aggregate/data
    // vector, while LLVM 23 can return a zero-operand vector ConstantFP/Int.
    if (test.kind == SplatKind::integer) { return llvm::ConstantInt::get(vector, bits); }
    return llvm::ConstantFP::get(vector, llvm::APFloat{element->getFltSemantics(), bits});
}

void add_splat_cases(llvm::Module &module) {
    for (const auto &test : splat_cases) {
        auto *constant = make_splat(module.getContext(), test);
        new llvm::GlobalVariable(module, constant->getType(), true,
                                 llvm::GlobalValue::ExternalLinkage, constant,
                                 std::string{"splat_global_"} + test.name);
        auto *nested = llvm::ConstantStruct::getAnon({constant, constant});
        new llvm::GlobalVariable(module, nested->getType(), true,
                                 llvm::GlobalValue::ExternalLinkage, nested,
                                 std::string{"splat_nested_"} + test.name);
        auto *local = make_splat(module.getContext(), test, true);
        auto *type = llvm::FunctionType::get(local->getType(), false);
        auto *function = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                                std::string{"splat_return_"} + test.name, module);
        auto *block = llvm::BasicBlock::Create(module.getContext(), "entry", function);
        llvm::IRBuilder<>{block}.CreateRet(local);
    }
}

bool check_splat(llvm::StringRef stage, llvm::StringRef role, const SplatCase &test,
                 const llvm::Constant *value, bool function_local = false) {
    if (value == nullptr || !value->getType()->isVectorTy()) {
        llvm::errs() << stage << ' ' << role << ' ' << test.name << ": missing vector constant\n";
        return false;
    }
    const auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(value->getType());
    if (vector == nullptr || vector->getNumElements() != 4u ||
        vector->getElementType() != splat_element_type(value->getContext(), test)) {
        llvm::errs() << stage << ' ' << role << ' ' << test.name << ": vector type changed\n";
        return false;
    }
    const auto expected = splat_bits(test, function_local);
    for (unsigned lane = 0u; lane < 4u; lane++) {
        const llvm::Constant *element = value;
        if (!llvm::isa<llvm::ConstantFP, llvm::ConstantInt>(value)) {
            element = value->getAggregateElement(lane);
        }
        const auto *fp = llvm::dyn_cast_or_null<llvm::ConstantFP>(element);
        const auto *integer = llvm::dyn_cast_or_null<llvm::ConstantInt>(element);
        if ((test.kind == SplatKind::integer && integer == nullptr) ||
            (test.kind != SplatKind::integer && fp == nullptr)) {
            llvm::errs() << stage << ' ' << role << ' ' << test.name << ": lane is not literal\n";
            return false;
        }
        const auto actual = fp != nullptr ? fp->getValueAPF().bitcastToAPInt() : integer->getValue();
        if (actual != expected) {
            llvm::errs() << stage << ' ' << role << ' ' << test.name << " lane " << lane
                         << ": expected " << expected << ", got " << actual << '\n';
            return false;
        }
    }
    return true;
}

bool check_splat_cases(llvm::StringRef stage, const llvm::Module &module) {
    bool valid = true;
    for (const auto &test : splat_cases) {
        const auto *global = module.getGlobalVariable(std::string{"splat_global_"} + test.name);
        valid &= check_splat(stage, "global", test,
                             global != nullptr && global->hasInitializer() ? global->getInitializer() : nullptr);
        const auto *nested = module.getGlobalVariable(std::string{"splat_nested_"} + test.name);
        const auto *aggregate = nested != nullptr && nested->hasInitializer() ? nested->getInitializer() : nullptr;
        for (unsigned field = 0u; field < 2u; field++) {
            valid &= check_splat(stage, "nested", test,
                                 aggregate != nullptr ? aggregate->getAggregateElement(field) : nullptr);
        }
        const auto *function = module.getFunction(std::string{"splat_return_"} + test.name);
        const auto *ret = function != nullptr && !function->isDeclaration() ?
                              llvm::dyn_cast<llvm::ReturnInst>(function->front().getTerminator()) :
                              nullptr;
        valid &= check_splat(stage, "return", test,
                             ret != nullptr ? llvm::dyn_cast<llvm::Constant>(ret->getReturnValue()) : nullptr, true);
    }
    llvm::outs() << stage << " vector splats: " << splat_cases.size()
                 << " patterns x (global + 2 nested fields + function-local return) x 4 exact-bit lanes "
                 << (valid ? "PASS\n" : "FAIL\n");
    return valid;
}

}// namespace

int main() {
    llvm::LLVMContext input_context;
    auto module = std::make_unique<llvm::Module>("llvm-downgrade70-roundtrip", input_context);
    auto *f32 = llvm::Type::getFloatTy(input_context);
    auto *type = llvm::FunctionType::get(f32, {f32}, false);
    for (const auto &test : cases) {
        auto *function = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, test.name, *module);
#if LLVM_VERSION_MAJOR >= 23
        llvm::AttrBuilder attributes{input_context};
        attributes.addDenormalFPEnvAttr({test.default_mode, test.f32_mode});
        function->addFnAttrs(attributes);
#else
        function->addFnAttr("denormal-fp-math", test.default_legacy);
        function->addFnAttr("denormal-fp-math-f32", test.f32_legacy);
#endif
        auto *block = llvm::BasicBlock::Create(input_context, "entry", function);
        llvm::IRBuilder<> builder{block};
        auto *argument = function->getArg(0);
        argument->setName("value");
        builder.CreateRet(builder.CreateFAdd(argument, llvm::ConstantFP::get(f32, 1.0), "sum"));
    }
    add_splat_cases(*module);
    if (llvm::verifyModule(*module, &llvm::errs())) { return 1; }
    if (!check_splat_cases("input", *module)) { return 1; }
    for (const auto &test : cases) {
        if (!check_function("input", *module, test)) { return 1; }
    }

    // This is the real project wrapper/library, not a copied writer implementation.
    auto bitcode = luisa::compute::llvm_downgrade_to_7(std::move(module));
    if (bitcode.empty()) {
        llvm::errs() << "Project LLVM 7 writer returned no bytes\n";
        return 1;
    }
    llvm::LLVMContext output_context;
    llvm::MemoryBufferRef buffer{
        llvm::StringRef{reinterpret_cast<const char *>(bitcode.data()), bitcode.size()},
        "project-writer70.bc"};
    auto parsed = llvm::parseBitcodeFile(buffer, output_context);
    if (!parsed) {
        llvm::errs() << "Same-SDK reader failed: " << llvm::toString(parsed.takeError()) << '\n';
        return 1;
    }
    const auto &roundtrip = **parsed;
    if (llvm::verifyModule(roundtrip, &llvm::errs())) { return 1; }
    bool valid = check_splat_cases("readback", roundtrip);
    for (const auto &test : cases) { valid &= check_function("readback", roundtrip, test); }
    if (!valid) { return 1; }
    llvm::outs() << "PASS LLVM " << LLVM_VERSION_STRING << " -> project writer70 -> same-SDK reader: "
                 << "2 denormal functions and 27 exact-bit vector patterns preserved; "
                 << bitcode.size() << " bitcode bytes\n";
    return 0;
}
