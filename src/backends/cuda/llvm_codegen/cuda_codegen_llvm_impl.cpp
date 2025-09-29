//
// Created by mike on 9/19/25.
//

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>

#include <luisa/core/clock.h>

#include "cuda_codegen_llvm_device_bitcode.h"
#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

CUDACodegenLLVMImpl::CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept
    : _config{std::move(config)} {
    Clock clk;
    _initialize();
    LUISA_VERBOSE_WITH_LOCATION("CUDA LLVM codegen initialized in {} ms.", clk.toc());
}

CUDACodegenLLVMImpl::FunctionContext::FunctionContext(llvm::Function *f) noexcept
    : llvm_func{f},
      llvm_alloca_block{llvm::BasicBlock::Create(llvm_func->getContext(), "alloca", llvm_func)},
      llvm_entry_block{llvm::BasicBlock::Create(llvm_func->getContext(), "entry", llvm_func)} {
    IB b{llvm_alloca_block};
    b.CreateBr(llvm_entry_block);
}

const llvm::Target *CUDACodegenLLVMImpl::_get_nvptx_target() noexcept {
    // initialize NVPTX target
    static std::once_flag once_flag;
    std::call_once(once_flag, [] {
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();
    });
    // lookup target
    static auto target = [] {
        std::string error;
        if (auto target = llvm::TargetRegistry::lookupTarget(nvptx_target_triple, error)) {
            return target;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to lookup target '{}': {}", nvptx_target_triple, error);
    }();
    return target;
}

inline void CUDACodegenLLVMImpl::_initialize() noexcept {

    // create target machine
    _target_machine = [this] {
        llvm::TargetOptions options;
        options.NoTrappingFPMath = true;
        if (_config.enable_fast_math) {
            options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
            options.UnsafeFPMath = true;
            options.NoInfsFPMath = true;
            options.NoNaNsFPMath = true;
            options.NoSignedZerosFPMath = true;
            options.ApproxFuncFPMath = true;
        } else {
            options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
            options.UnsafeFPMath = false;
            options.NoInfsFPMath = false;
            options.NoNaNsFPMath = false;
            options.NoSignedZerosFPMath = false;
            options.ApproxFuncFPMath = false;
        }
        if (_config.enable_debug_info) {
            options.TrapUnreachable = true;
            options.NoTrapAfterNoreturn = false;
        } else {
            options.TrapUnreachable = false;
            options.NoTrapAfterNoreturn = true;
        }
        auto opt_level = llvm::CodeGenOptLevel::Default;
        switch (_config.opt_level) {
            case CUDACodegenLLVMConfig::OptLevel::LEVEL_NONE: opt_level = llvm::CodeGenOptLevel::None; break;
            case CUDACodegenLLVMConfig::OptLevel::LEVEL_LESS: opt_level = llvm::CodeGenOptLevel::Less; break;
            case CUDACodegenLLVMConfig::OptLevel::LEVEL_DEFAULT: opt_level = llvm::CodeGenOptLevel::Default; break;
            case CUDACodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE: opt_level = llvm::CodeGenOptLevel::Aggressive; break;
        }
        auto cpu_name = luisa::format("sm_{}", _config.cuda_arch);
        return _get_nvptx_target()->createTargetMachine(
            nvptx_target_triple, llvm::StringRef{cpu_name}, {},
            options, llvm::Reloc::Static, llvm::CodeModel::Small, opt_level);
    }();

    _data_layout = std::make_unique<llvm::DataLayout>(_target_machine->createDataLayout());

    // parse libdevice bitcode
    _llvm_module = [&] {
        llvm::SMDiagnostic error;
        llvm::StringRef bc{reinterpret_cast<const char *>(luisa_compute_cuda_libdevice_10),
                           luisa_compute_cuda_libdevice_10_size};
        if (auto m = llvm::parseIR({bc, "libdevice.10.bc"}, error, _llvm_context)) { return m; }
        LUISA_ERROR_WITH_LOCATION("Failed to parse libdevice bitcode: {}", error.getMessage());
    }();

    // set the target triple
    _llvm_module->setTargetTriple(nvptx_target_triple);
    _llvm_module->setDataLayout(*_data_layout);

    // internalize all device functions
    for (auto &&f : *_llvm_module) {
        if (f.getName().starts_with("__nv_")) {
            f.setLinkage(llvm::Function::PrivateLinkage);
        }
    }

    auto parse_llvm_constant_string = [](llvm::Value *c) noexcept -> llvm::StringRef {
        if (auto gv = llvm::dyn_cast<llvm::GlobalVariable>(c)) {
            if (auto init = gv->getInitializer()) {
                if (auto ca = llvm::dyn_cast<llvm::ConstantDataArray>(init)) {
                    if (ca->isCString()) {
                        return ca->getAsCString();
                    }
                }
            }
        }
        return {};
    };

    // handle __nvvm_reflect
    if (auto f = _llvm_module->getFunction("__nvvm_reflect")) {
        auto const_one = llvm::ConstantInt::get(llvm::Type::getInt32Ty(_llvm_context), 1);
        auto const_zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(_llvm_context), 0);
        auto const_arch = llvm::ConstantInt::get(llvm::Type::getInt32Ty(_llvm_context), _config.cuda_arch * 10);
        llvm::SmallVector<llvm::Instruction *> reflected;
        for (auto user : f->users()) {
            if (auto call = llvm::dyn_cast<llvm::CallInst>(user)) {
                // try to parse the argument string
                if (auto s = parse_llvm_constant_string(call->getArgOperand(0)); s == "__CUDA_FTZ") {
                    call->replaceAllUsesWith(_config.enable_fast_math ? const_one : const_zero);
                    reflected.emplace_back(call);
                } else if (s == "__CUDA_PREC_SQRT" || s == "__CUDA_PREC_DIV") {
                    call->replaceAllUsesWith(_config.enable_fast_math ? const_zero : const_one);
                    reflected.emplace_back(call);
                } else if (s == "__CUDA_ARCH") {
                    call->replaceAllUsesWith(const_arch);
                    reflected.emplace_back(call);
                }
            }
        }
        for (auto i : reflected) { i->eraseFromParent(); }
        if (f->user_empty()) { f->eraseFromParent(); }
    }
}

void CUDACodegenLLVMImpl::_dump_module(const std::filesystem::path &path) const noexcept {
    std::error_code ec;
    llvm::raw_fd_ostream out{path.string(), ec};
    if (ec) {
        LUISA_WARNING_WITH_LOCATION("Failed to open file for dumping LLVM module: {}.", ec.message());
    } else {
        _llvm_module->print(out, nullptr);
    }
}

void CUDACodegenLLVMImpl::_run_optimization_passes() noexcept {

    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PipelineTuningOptions PTO;
    PTO.LoopInterleaving = true;
#if LLVM_VERSION_MAJOR >= 21
    PTO.LoopInterchange = true;
#endif
    PTO.LoopVectorization = true;
    PTO.SLPVectorization = true;
    PTO.LoopUnrolling = true;
    PTO.MergeFunctions = true;
    llvm::PassBuilder PB{_target_machine, PTO};
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
#if LLVM_VERSION_MAJOR >= 19
    _target_machine->registerPassBuilderCallbacks(PB);
#else
    _target_machine->registerPassBuilderCallbacks(PB, true);
#endif

    auto opt_level = llvm::OptimizationLevel::O2;
    switch (_config.opt_level) {
        case CUDACodegenLLVMConfig::OptLevel::LEVEL_NONE: opt_level = llvm::OptimizationLevel::O0; break;
        case CUDACodegenLLVMConfig::OptLevel::LEVEL_LESS: opt_level = llvm::OptimizationLevel::O1; break;
        case CUDACodegenLLVMConfig::OptLevel::LEVEL_DEFAULT: opt_level = llvm::OptimizationLevel::O2; break;
        case CUDACodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE: opt_level = llvm::OptimizationLevel::O3; break;
    }
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt_level);
    MPM.run(*_llvm_module, MAM);
}

luisa::string CUDACodegenLLVMImpl::_generate_ptx() const noexcept {
    llvm::SmallVector<char, 256> ptx;
    llvm::raw_svector_ostream os{ptx};
    llvm::legacy::PassManager passManager;
    if (_target_machine->addPassesToEmitFile(passManager, os, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
        LUISA_ERROR_WITH_LOCATION("TargetMachine can't emit PTX.");
    }
    passManager.run(*_llvm_module);
    return {ptx.begin(), ptx.end()};
}

luisa::string CUDACodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    _llvm_module->setSourceFileName(xir_module.name().value_or("cuda_kernel.cu"));
    _run_optimization_passes();
    return _generate_ptx();
}

}// namespace luisa::compute::cuda
