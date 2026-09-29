#include "llvm_downgrade.h"

#include <cstring>
#include <string>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

namespace luisa::compute {

namespace {

void validate_llvm_7_module(const llvm::Module &module) noexcept {
    llvm::SmallPtrSet<llvm::Type *, 32u> visited_types;
    llvm::SmallPtrSet<const llvm::Value *, 32u> visited_values;
    llvm::SmallVector<llvm::Type *, 32u> types;
    llvm::SmallVector<const llvm::Value *, 32u> values;
    auto collect_attribute_types = [&](llvm::AttributeList attributes) noexcept {
        for (auto set : attributes) {
            for (auto attribute : set) {
                if (attribute.isTypeAttribute()) { types.emplace_back(attribute.getValueAsType()); }
            }
        }
    };
    for (auto &global : module.global_values()) { values.emplace_back(&global); }
    for (auto &function : module) {
        collect_attribute_types(function.getAttributes());
        for (auto &block : function) {
            for (auto &inst : block) {
                // The extracted writer removes freeze, which is not valid for
                // possibly poison/undef operands. Keep this boundary explicit
                // until a semantics-preserving legalization is available.
                if (llvm::isa<llvm::FreezeInst>(inst)) {
                    llvm::report_fatal_error("LLVM 7 downgrade cannot preserve freeze semantics; legalize freeze before serialization.", false);
                }
                if (auto call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
                    collect_attribute_types(call->getAttributes());
                }
                values.emplace_back(&inst);
            }
        }
    }
    while (!values.empty()) {
        auto value = values.pop_back_val();
        if (!visited_values.insert(value).second) { continue; }
        types.emplace_back(value->getType());
        if (auto global = llvm::dyn_cast<llvm::GlobalValue>(value)) {
            types.emplace_back(global->getValueType());
        }
        if (auto alloca = llvm::dyn_cast<llvm::AllocaInst>(value)) {
            types.emplace_back(alloca->getAllocatedType());
        }
        if (auto gep = llvm::dyn_cast<llvm::GEPOperator>(value)) {
            types.emplace_back(gep->getSourceElementType());
        }
        if (auto assembly = llvm::dyn_cast<llvm::InlineAsm>(value)) {
            types.emplace_back(assembly->getFunctionType());
        }
        if (auto user = llvm::dyn_cast<llvm::User>(value)) {
            for (auto &operand : user->operands()) { values.emplace_back(operand.get()); }
        }
    }
    while (!types.empty()) {
        auto type = types.pop_back_val();
        if (!visited_types.insert(type).second) { continue; }
        switch (type->getTypeID()) {
            case llvm::Type::ScalableVectorTyID:
            case llvm::Type::BFloatTyID:
            case llvm::Type::X86_AMXTyID:
            case llvm::Type::TargetExtTyID: {
                std::string description;
                llvm::raw_string_ostream stream{description};
                type->print(stream);
                llvm::report_fatal_error(llvm::Twine{"LLVM 7 downgrade cannot encode type: "} + description, false);
            }
            default: break;
        }
        for (auto subtype : type->subtypes()) { types.emplace_back(subtype); }
    }
}

}// namespace

std::vector<std::byte>
llvm_downgrade_to_7(std::unique_ptr<llvm::Module> module) noexcept {
    if (module == nullptr) {
        llvm::report_fatal_error("LLVM 7 downgrade requires a module.", false);
    }
    validate_llvm_7_module(*module);
    // LLVM 7 permits call fast-math flags only on scalar/vector FP returns.
    // Modern LLVM also permits homogeneous aggregates (e.g. OptiX barycentrics).
    // Dropping those optimization permissions preserves the call's semantics.
    for (auto &function : *module) {
        for (auto &block : function) {
            for (auto &inst : block) {
                if (auto call = llvm::dyn_cast<llvm::CallBase>(&inst);
                    call != nullptr && call->getType()->isAggregateType() &&
                    llvm::isa<llvm::FPMathOperator>(call)) {
                    call->copyFastMathFlags(llvm::FastMathFlags{});
                }
            }
        }
    }
    // Pointer preparation inserts no-op casts used by the legacy writer to
    // reconstruct typed pointers. Do not run optimization passes after it.
    llvm::BitcodeWriter70::prepareModule(*module);
    llvm::SmallVector<char, 0u> storage;
    llvm::raw_svector_ostream stream{storage};
    llvm::WriteBitcode70ToFile(*module, stream);
    std::vector<std::byte> bitcode(storage.size());
    std::memcpy(bitcode.data(), storage.data(), storage.size());
    return bitcode;
}

}// namespace luisa::compute
