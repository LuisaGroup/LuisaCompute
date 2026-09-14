#include "hip_floating_remainder.h"
#include "ut/ut.hpp"

#include <array>
#include <string>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>

using namespace luisa::compute::hip;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "HIP remainder exposes native scalar range reduction before IPO"_test = [] {
        llvm::LLVMContext context;
        const std::array<llvm::Type *, 3u> scalar_types{
            llvm::Type::getHalfTy(context), llvm::Type::getFloatTy(context),
            llvm::Type::getDoubleTy(context)};
        const std::array suffixes{"f16", "f32", "f64"};
        for (auto type_index = 0u; type_index < scalar_types.size(); type_index++) {
            for (auto lanes = 1u; lanes <= 4u; lanes++) {
                llvm::Module module{"remainder", context};
                auto scalar_type = scalar_types[type_index];
                auto native = llvm::Function::Create(
                    llvm::FunctionType::get(scalar_type, {scalar_type, scalar_type}, false),
                    llvm::Function::ExternalLinkage,
                    std::string{"__ocml_fmod_"} + suffixes[type_index], module);
                llvm::Type *type = lanes == 1u ? scalar_type :
                    llvm::FixedVectorType::get(scalar_type, lanes);
                auto function = llvm::Function::Create(
                    llvm::FunctionType::get(type, {type, type}, false),
                    llvm::Function::ExternalLinkage, "dynamic_remainder", module);
                auto block = llvm::BasicBlock::Create(context, "entry", function);
                llvm::IRBuilder<> builder{block};
                auto result = emit_hip_floating_remainder(
                    builder, module, function->getArg(0), function->getArg(1));
                builder.CreateRet(result);

                expect(!llvm::verifyModule(module, &llvm::errs()));
                auto calls = 0u;
                auto remainders = 0u;
                auto casts = 0u;
                for (auto &inst : *block) {
                    remainders += inst.getOpcode() == llvm::Instruction::FRem;
                    casts += llvm::isa<llvm::FPExtInst, llvm::FPTruncInst>(inst);
                    if (auto call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                        expect(call->getCalledFunction() == native);
                        expect(call->getType() == scalar_type);
                        expect(call->getArgOperand(0)->getType() == scalar_type);
                        expect(call->getArgOperand(1)->getType() == scalar_type);
                        expect(!call->hasFnAttr(llvm::Attribute::AlwaysInline));
                        expect(!call->hasFnAttr(llvm::Attribute::NoInline));
                        for (auto operand = 0u; operand < 2u; operand++) {
                            if (lanes == 1u) {
                                expect(call->getArgOperand(operand) == function->getArg(operand));
                            } else {
                                auto extract = llvm::dyn_cast<llvm::ExtractElementInst>(call->getArgOperand(operand));
                                expect(extract != nullptr);
                                if (extract != nullptr) {
                                    expect(extract->getVectorOperand() == function->getArg(operand));
                                    expect(llvm::cast<llvm::ConstantInt>(extract->getIndexOperand())->getZExtValue() == calls);
                                }
                            }
                        }
                        calls++;
                    }
                }
                expect(calls == lanes) << suffixes[type_index] << " x " << lanes;
                expect(remainders == 0u) << "late generic remainder bypasses native IPO";
                expect(casts == 0u) << "native f16/f32/f64 ABI must be exact";
            }
        }
    };
}
