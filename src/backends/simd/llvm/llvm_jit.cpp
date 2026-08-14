#include "llvm_jit.h"

#include <utility>

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace luisa::compute::simd {

namespace {

// Module keeps a non-owning reference to its LLVMContext. Function parameters
// are otherwise destroyed in the opposite order of this public (module,
// context) API, so every early return must explicitly release the module
// first. Moved-from pointers make the successful ORC path a no-op here.
struct ModuleContextLifetime {
    std::unique_ptr<::llvm::Module> &module;
    std::unique_ptr<::llvm::LLVMContext> &context;

    ~ModuleContextLifetime() noexcept {
        module.reset();
        context.reset();
    }
};

}// namespace

void LLVMJIT::_fail(std::string message) noexcept {
    if (_error.empty()) { _error = std::move(message); }
}

LLVMJIT::LLVMJIT(bool capture_object) noexcept {
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
    // LLJIT selects PIC for an unspecified relocation model. Make that
    // choice explicit before creating the audit TargetMachine as well, so
    // captured assembly and ORC's final relocatable object have identical
    // instruction layout.
    host->setRelocationModel(::llvm::Reloc::PIC_);
    host->setCodeModel(::llvm::CodeModel::Small);
    _target_triple = host->getTargetTriple().str();
    auto target_machine = host->createTargetMachine();
    if (!target_machine) {
        _fail("failed to create the LLVM host target machine: " +
              ::llvm::toString(target_machine.takeError()));
        return;
    }
    _target_machine = std::move(*target_machine);
    _target_machine->Options.MCOptions.AsmVerbose = true;
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
    if (capture_object) {
        _object = std::make_shared<std::string>();
        _jit->getObjTransformLayer().setTransform(
            [capture = _object](
                std::unique_ptr<::llvm::MemoryBuffer> object)
                -> ::llvm::Expected<
                    std::unique_ptr<::llvm::MemoryBuffer>> {
                auto buffer = object->getBuffer();
                capture->assign(buffer.data(), buffer.size());
                return std::move(object);
            });
    }
}

LLVMJIT::~LLVMJIT() noexcept = default;
LLVMJIT::LLVMJIT(LLVMJIT &&) noexcept = default;
LLVMJIT &LLVMJIT::operator=(LLVMJIT &&) noexcept = default;

bool LLVMJIT::add_module(
    std::unique_ptr<::llvm::Module> module,
    std::unique_ptr<::llvm::LLVMContext> context) noexcept {
    ModuleContextLifetime lifetime{module, context};
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

std::string LLVMJIT::_emit_assembly(::llvm::Module &module) noexcept {
    if (!_prepare_module(module)) { return {}; }
    ::llvm::SmallVector<char, 0u> storage;
    ::llvm::raw_svector_ostream output{storage};
    ::llvm::legacy::PassManager codegen;
    if (_target_machine->addPassesToEmitFile(
            codegen, output, nullptr,
            ::llvm::CodeGenFileType::AssemblyFile)) {
        _fail("LLVM host target cannot emit assembly");
        return {};
    }
    codegen.run(module);
    return std::string{storage.begin(), storage.end()};
}

std::string LLVMJIT::emit_assembly(
    std::unique_ptr<::llvm::Module> module,
    std::unique_ptr<::llvm::LLVMContext> context) noexcept {
    ModuleContextLifetime lifetime{module, context};
    if (!succeeded()) { return {}; }
    if (module == nullptr || context == nullptr ||
        &module->getContext() != context.get()) {
        _fail("LLVM assembly module and context ownership do not match");
        return {};
    }
    return _emit_assembly(*module);
}

std::string LLVMJIT::emit_assembly_copy(
    const ::llvm::Module &module) noexcept {
    if (!succeeded()) { return {}; }
    auto copy = ::llvm::CloneModule(module);
    return _emit_assembly(*copy);
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

bool LLVMJIT::supports_native_paired_leaf_gather(
    uint32_t width) const noexcept {
    // The measured rewrite is specific to one 512-bit vector of eight packed
    // 64-bit lanes. W4 was neutral and W16 requires two such instructions.
    // The generic legality/scalarization query does not establish relative
    // profitability against two 32-bit gathers. Keep that distinction in the
    // measured exact-W8 policy instead of treating legality as a cost proof.
    if (!succeeded() || width != 8u) { return false; }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-paired-gather-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt64Ty(context), width);
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_paired_gather_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedGather(type, ::llvm::Align{1u}) &&
           !target.forceScalarizeMaskedGather(type, ::llvm::Align{1u});
}

bool LLVMJIT::supports_native_predicated_loop(
    uint32_t width) const noexcept {
    if (!succeeded() || (width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-predicated-loop-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_predicated_loop_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    auto *gather_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), width);
    auto alignment = ::llvm::Align{alignof(uint32_t)};
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedGather(gather_type, alignment) &&
           !target.forceScalarizeMaskedGather(gather_type, alignment);
}

}// namespace luisa::compute::simd
