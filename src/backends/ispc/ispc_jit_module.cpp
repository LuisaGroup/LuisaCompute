//
// Created by Mike Smith on 2021/11/15.
//

#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>

#include <core/logging.h>
#include <runtime/context.h>
#include <backends/ispc/ispc_jit_module.h>

namespace luisa::compute::ispc {

ISPCJITModule::ISPCJITModule(
    luisa::unique_ptr<llvm::LLVMContext> ctx,
    llvm::ExecutionEngine *engine) noexcept
    : _context{std::move(ctx)}, _engine{engine} {
    _f_ptr = reinterpret_cast<function_type *>(
        _engine->getFunctionAddress("kernel_main"));
}

ISPCJITModule::~ISPCJITModule() noexcept = default;

[[nodiscard]] auto get_target_machine() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        LLVMLinkInMCJIT();
    });
    std::string err;
    auto target_triple = llvm::sys::getDefaultTargetTriple();
    LUISA_INFO("Target: {}.", target_triple);
    auto target = llvm::TargetRegistry::lookupTarget(target_triple, err);
    if (target == nullptr) {
        LUISA_ERROR_WITH_LOCATION("Failed to get target machine: {}.", err);
    }
    llvm::TargetOptions options;
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    options.AllowFPOpFusion = ::llvm::FPOpFusion::Fast;
    options.UnsafeFPMath = true;
    options.NoInfsFPMath = true;
    options.NoNaNsFPMath = true;
    options.NoTrappingFPMath = true;
    options.EnableIPRA = true;
    options.StackSymbolOrdering = true;
    auto mcpu = llvm::sys::getHostCPUName();
    auto machine = target->createTargetMachine(
        target_triple, mcpu,
#if defined(LUISA_PLATFORM_APPLE) && defined(__aarch64__)
        "+neon",
#else
        "+avx2",
#endif
        options, {}, {},
        llvm::CodeGenOpt::Aggressive, true);
    if (machine == nullptr) {
        LUISA_ERROR_WITH_LOCATION("Failed to create target machine.");
    }
    return machine;
}

luisa::shared_ptr<ISPCModule> ISPCJITModule::load(
    const Context &ctx, const std::filesystem::path &ir_path) noexcept {
    // load
    auto context = luisa::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic error;
    LUISA_INFO("Loading LLVM IR: '{}'.", ir_path.string());
    auto module = llvm::parseIRFile(ir_path.string(), error, *context);
    if (module == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load module: {}.",
            error.getMessage().data());
    }

    // optimize: machine
    auto machine = get_target_machine();
    llvm::PassManagerBuilder pass_manager_builder;
    pass_manager_builder.OptLevel = llvm::CodeGenOpt::Aggressive;
    pass_manager_builder.Inliner = llvm::createFunctionInliningPass(
        pass_manager_builder.OptLevel, 0, false);
    pass_manager_builder.LoopsInterleaved = true;
    pass_manager_builder.LoopVectorize = true;
    pass_manager_builder.SLPVectorize = true;
    pass_manager_builder.MergeFunctions = true;
    pass_manager_builder.EnablePGOCSInstrGen = false;
    pass_manager_builder.EnablePGOCSInstrUse = false;
    pass_manager_builder.EnablePGOInstrGen = false;
    pass_manager_builder.CallGraphProfile = false;
    pass_manager_builder.PerformThinLTO = true;
    machine->adjustPassManager(pass_manager_builder);
    module->setDataLayout(machine->createDataLayout());

    // optimize: function passes
    {
        llvm::legacy::FunctionPassManager pass_manager{module.get()};
        pass_manager_builder.populateFunctionPassManager(pass_manager);
        pass_manager.add(llvm::createTargetTransformInfoWrapperPass(
            machine->getTargetIRAnalysis()));
        pass_manager_builder.populateFunctionPassManager(pass_manager);
        pass_manager.doInitialization();
        for (auto &&f : module->functions()) {
            pass_manager.run(f);
        }
        pass_manager.doFinalization();
    }

    // optimize: module passes
    {
        llvm::legacy::PassManager pass_manager;
        pass_manager.add(
            llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
        pass_manager_builder.populateModulePassManager(pass_manager);
        pass_manager.run(*module);
    }

    // jit
    std::string err;
    auto engine = llvm::EngineBuilder{std::move(module)}
                      .setErrorStr(&err)
                      .setOptLevel(llvm::CodeGenOpt::Aggressive)
                      .setEngineKind(llvm::EngineKind::JIT)
                      .create(machine);
    engine->DisableLazyCompilation(true);
    engine->DisableSymbolSearching(false);// to support print in ispc
    engine->addGlobalMapping("fflush", reinterpret_cast<uint64_t>(&fflush));
    engine->addGlobalMapping("fputs", reinterpret_cast<uint64_t>(&fputs));
    engine->addGlobalMapping("snprintf", reinterpret_cast<uint64_t>(&snprintf));
    engine->addGlobalMapping("vsnprintf", reinterpret_cast<uint64_t>(&vsnprintf));
    if (engine == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create execution engine: {}.", err);
    }
    return luisa::make_shared<ISPCJITModule>(
        ISPCJITModule{std::move(context), engine});
}

}// namespace luisa::compute::ispc
