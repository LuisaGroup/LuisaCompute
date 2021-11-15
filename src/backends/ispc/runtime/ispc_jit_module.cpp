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

#include <backends/ispc/runtime/ispc_jit_module.h>

namespace lc::ispc {

JITModule::JITModule(luisa::unique_ptr<llvm::LLVMContext> ctx, std::unique_ptr<llvm::ExecutionEngine> engine) noexcept
    : _context{std::move(ctx)}, _engine{std::move(engine)} {
    _run = reinterpret_cast<function_type *>(_engine->getFunctionAddress("run"));
}

JITModule::~JITModule() noexcept = default;

void JITModule::operator()(luisa::uint3 thread_count, luisa::uint3 thread_start, const void *args) const noexcept {
    _run(thread_count.x, thread_count.y, thread_count.z,
         thread_start.x, thread_start.y, thread_start.z,
         reinterpret_cast<uint64_t>(args));
}

[[nodiscard]] auto get_target_machine() {
    static std::once_flag flag;
    static llvm::TargetMachine *machine{nullptr};
    std::call_once(flag, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        LLVMLinkInMCJIT();
        std::string err;
        auto target_triple = llvm::sys::getDefaultTargetTriple();
        LUISA_INFO("Target: {}.", target_triple);
        auto target = llvm::TargetRegistry::lookupTarget(target_triple, err);
        if (target == nullptr) {
            LUISA_ERROR_WITH_LOCATION("Failed to get target machine: {}.", err);
        }
        machine = target->createTargetMachine(
            target_triple, "generic", {"+avx2"},
            {}, {}, {}, llvm::CodeGenOpt::Aggressive, true);
        if (machine == nullptr) {
            LUISA_ERROR_WITH_LOCATION("Failed to create target machine.");
        }
    });
    return machine;
}

JITModule JITModule::load(const std::filesystem::path &ir_path) noexcept {
    auto context = luisa::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic error;
    LUISA_INFO("Loading LLVM IR: {}.", ir_path.string());
    auto module = llvm::parseIRFile(ir_path.string(), error, *context);
    if (module == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load module: {}.",
            error.getMessage().data());
    }
    std::string err;
    llvm::EngineBuilder engine_builder{std::move(module)};
    engine_builder.setErrorStr(&err);
    engine_builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
    std::unique_ptr<llvm::ExecutionEngine> exec_engine{
        engine_builder.create(get_target_machine())};
    if (exec_engine == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create execution engine: {}.", err);
    }
    return {std::move(context), std::move(exec_engine)};
}

}// namespace lc::ispc
