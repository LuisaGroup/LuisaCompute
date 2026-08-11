#include "llvm_jit.h"

#include <utility>

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

namespace luisa::compute::simd {

void LLVMJIT::_fail(std::string message) noexcept {
    if (_error.empty()) { _error = std::move(message); }
}

LLVMJIT::LLVMJIT() noexcept {
    if (::llvm::InitializeNativeTarget() ||
        ::llvm::InitializeNativeTargetAsmPrinter()) {
        _fail("failed to initialize the LLVM native target");
        return;
    }
    auto host = ::llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!host) {
        _fail("failed to detect the LLVM host target: " +
              ::llvm::toString(host.takeError()));
        return;
    }
    host->setCodeGenOptLevel(::llvm::CodeGenOptLevel::Aggressive);
    _target_triple = host->getTargetTriple().str();
    auto target_machine = host->createTargetMachine();
    if (!target_machine) {
        _fail("failed to create the LLVM host target machine: " +
              ::llvm::toString(target_machine.takeError()));
        return;
    }
    _target_machine = std::move(*target_machine);
    ::llvm::orc::LLJITBuilder builder;
    builder.setJITTargetMachineBuilder(std::move(*host));
    auto jit = builder.create();
    if (!jit) {
        _fail("failed to create LLVM ORC JIT: " +
              ::llvm::toString(jit.takeError()));
        _target_machine.reset();
        return;
    }
    _jit = std::move(*jit);
}

LLVMJIT::~LLVMJIT() noexcept = default;
LLVMJIT::LLVMJIT(LLVMJIT &&) noexcept = default;
LLVMJIT &LLVMJIT::operator=(LLVMJIT &&) noexcept = default;

bool LLVMJIT::add_module(
    std::unique_ptr<::llvm::Module> module,
    std::unique_ptr<::llvm::LLVMContext> context) noexcept {
    if (!succeeded()) { return false; }
    if (module == nullptr || context == nullptr ||
        &module->getContext() != context.get()) {
        _fail("LLVM JIT module and context ownership do not match");
        return false;
    }
    if (!_prepare_module(*module)) { return false; }

    if (auto error = _jit->addIRModule(::llvm::orc::ThreadSafeModule(
            std::move(module), std::move(context)))) {
        _fail("failed to add LLVM IR module to ORC JIT: " +
              ::llvm::toString(std::move(error)));
        return false;
    }
    return true;
}

bool LLVMJIT::_prepare_module(::llvm::Module &module) noexcept {
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    if (::llvm::verifyModule(module, &::llvm::errs())) {
        _fail("refusing to process an invalid LLVM module");
        return false;
    }

    ::llvm::LoopAnalysisManager loop_analyses;
    ::llvm::FunctionAnalysisManager function_analyses;
    ::llvm::CGSCCAnalysisManager cgscc_analyses;
    ::llvm::ModuleAnalysisManager module_analyses;
    ::llvm::PassBuilder pass_builder{_target_machine.get()};
    pass_builder.registerModuleAnalyses(module_analyses);
    pass_builder.registerCGSCCAnalyses(cgscc_analyses);
    pass_builder.registerFunctionAnalyses(function_analyses);
    pass_builder.registerLoopAnalyses(loop_analyses);
    pass_builder.crossRegisterProxies(
        loop_analyses, function_analyses,
        cgscc_analyses, module_analyses);
    auto pipeline = pass_builder.buildPerModuleDefaultPipeline(
        ::llvm::OptimizationLevel::O2);
    pipeline.run(module, module_analyses);
    return true;
}

std::string LLVMJIT::emit_assembly(
    std::unique_ptr<::llvm::Module> module,
    std::unique_ptr<::llvm::LLVMContext> context) noexcept {
    if (!succeeded()) { return {}; }
    if (module == nullptr || context == nullptr ||
        &module->getContext() != context.get()) {
        _fail("LLVM assembly module and context ownership do not match");
        return {};
    }
    if (!_prepare_module(*module)) { return {}; }
    ::llvm::SmallVector<char, 0u> storage;
    ::llvm::raw_svector_ostream output{storage};
    ::llvm::legacy::PassManager codegen;
    if (_target_machine->addPassesToEmitFile(
            codegen, output, nullptr,
            ::llvm::CodeGenFileType::AssemblyFile)) {
        _fail("LLVM host target cannot emit assembly");
        return {};
    }
    codegen.run(*module);
    return std::string{storage.begin(), storage.end()};
}

void *LLVMJIT::lookup(std::string_view name) noexcept {
    if (!succeeded()) { return nullptr; }
    auto symbol = _jit->lookup(::llvm::StringRef{name.data(), name.size()});
    if (!symbol) {
        _fail("failed to look up LLVM JIT symbol '" +
              std::string{name} + "': " +
              ::llvm::toString(symbol.takeError()));
        return nullptr;
    }
    return symbol->toPtr<void *>();
}

}// namespace luisa::compute::simd
