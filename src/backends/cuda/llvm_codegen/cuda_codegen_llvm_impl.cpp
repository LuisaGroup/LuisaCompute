//
// Created by mike on 9/19/25.
//

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/DebugInfo.h>
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
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Transforms/Utils/CodeExtractor.h>

#include <luisa/core/clock.h>

#include "cuda_codegen_llvm_device_bitcode.h"
#include "cuda_codegen_llvm_impl.h"

#undef None

namespace luisa::compute::cuda {

CUDACodegenLLVMImpl::CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept
    : _config{std::move(config)} {
    LUISA_ASSERT(_config.block_size[0] > 0u && _config.block_size[1] > 0u && _config.block_size[2] > 0u,
                 "Block size must be constant and greater than zero for now.");
    Clock clk;
    _initialize();
    LUISA_VERBOSE_WITH_LOCATION("CUDA LLVM codegen initialized in {} ms.", clk.toc());
}

CUDACodegenLLVMImpl::FunctionContext::FunctionContext(llvm::Function *f) noexcept
    : llvm_func{f},
      llvm_alloca_block{llvm::BasicBlock::Create(f->getContext(), "alloca", f)},
      llvm_entry_block{llvm::BasicBlock::Create(f->getContext(), "entry", f)} {
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
#if LLVM_VERSION_MAJOR >= 22
        if (auto target = llvm::TargetRegistry::lookupTarget(llvm::Triple(nvptx_target_triple), error)) {
#else
        if (auto target = llvm::TargetRegistry::lookupTarget(nvptx_target_triple, error)) {
#endif
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
#if LLVM_VERSION_MAJOR < 22
            options.UnsafeFPMath = true;
#endif
#if LLVM_VERSION_MAJOR < 22
            options.NoInfsFPMath = true;
            options.NoNaNsFPMath = true;
#endif
            options.NoSignedZerosFPMath = true;
#if LLVM_VERSION_MAJOR < 22
            options.ApproxFuncFPMath = true;
#endif
        } else {
            options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
#if LLVM_VERSION_MAJOR < 22
            options.UnsafeFPMath = false;
#endif
#if LLVM_VERSION_MAJOR < 22
            options.NoInfsFPMath = false;
            options.NoNaNsFPMath = false;
#endif
            options.NoSignedZerosFPMath = false;
#if LLVM_VERSION_MAJOR < 22
            options.ApproxFuncFPMath = false;
#endif
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
        auto cpu_name = fmt::format("sm_{}", _config.cuda_arch);
        return _get_nvptx_target()->createTargetMachine(
            llvm::Triple{nvptx_target_triple}, llvm::StringRef{cpu_name}, {},
            options, llvm::Reloc::Static, llvm::CodeModel::Small, opt_level);
    }();

    _data_layout = std::make_unique<llvm::DataLayout>(_target_machine->createDataLayout());

    // parse libdevice bitcode
    _llvm_module = [&] {
        llvm::SMDiagnostic error;
        llvm::StringRef bc{reinterpret_cast<const char *>(luisa_compute_cuda_libdevice_10),
                           luisa_compute_cuda_libdevice_10_size};
        if (auto m = llvm::parseIR({bc, "libdevice.10.bc"}, error, _llvm_context)) {
            llvm::StripDebugInfo(*m);
            return m;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to parse libdevice bitcode: {}", error.getMessage());
    }();

    // set the target triple
    _llvm_module->setTargetTriple(llvm::Triple{nvptx_target_triple});
    _llvm_module->setDataLayout(*_data_layout);

    // internalize all device functions
    for (auto &&f : *_llvm_module) {
        if (f.getName().starts_with("__nv_")) {
            f.setLinkage(llvm::Function::PrivateLinkage);
            f.removeFnAttr(llvm::Attribute::StackProtect);
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

void CUDACodegenLLVMImpl::_run_optimization_passes(LLVMModulePassManagerCallback callback) noexcept {

    // add fast-math flags to FPMathOperators
    if (_config.enable_fast_math) {
        for (auto &f : *_llvm_module) {
            for (auto &bb : f) {
                for (auto &inst : bb) {
                    if (llvm::isa<llvm::FPMathOperator>(inst)) {
                        if (inst.getOpcode() == llvm::Instruction::FAdd) {
                            // for some mysterious reason, `fadd` with `no inf` causes bad precision in some cases
                            auto flags = llvm::FastMathFlags::getFast();
                            flags.setNoInfs(false);
                            inst.setFastMathFlags(flags);
                        } else {
                            inst.setFast(true);
                        }
                    }
                }
            }
        }
    }

    auto do_optimize = [&] {
        // run optimization passes
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
        if (callback) { callback(MPM); }
        MPM.run(*_llvm_module, MAM);
    };

    // primary optimization pass
    do_optimize();

    // run a second pass if any device function is not inlined
    {
        auto any_not_inlined = false;
        for (auto &f : *_llvm_module) {
            if (!f.isDeclaration() && f.getCallingConv() == llvm::CallingConv::PTX_Device) {
                f.addFnAttr(llvm::Attribute::AlwaysInline);
                any_not_inlined = true;
            }
        }
        if (any_not_inlined) {
            LUISA_VERBOSE("Running secondary optimization passes to inline device functions...");
            do_optimize();
        }
    }
}

namespace detail {

// A stub PassManager to filter out the buggy "NVPTX Replace Image Handles" pass
class NVPTXPassManagerStub : public llvm::legacy::PassManager {
public:
    void add(llvm::Pass *pass) override {
        constexpr llvm::StringRef replace_image_handles_pass_name = "NVPTX Replace Image Handles";
        if (pass->getPassName() == replace_image_handles_pass_name) {
            LUISA_WARNING_WITH_LOCATION("Skipping buggy pass: {}", replace_image_handles_pass_name);
        } else {
            PassManager::add(pass);
        }
    }
};

// module pass that extracts ray query loops into separate functions and normalize the pipelines
class RayQueryLoopExtraction : public llvm::PassInfoMixin<RayQueryLoopExtraction> {

private:
    llvm::DenseSet<llvm::Function *> RayQueryFunctions;

    [[nodiscard]] bool extractRayQueryLoops(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) noexcept {
        auto Init = M.getFunction(CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_initialize);
        auto RemoveUnusedInitCalls = [&]() noexcept {
            if (Init == nullptr) { return false; }
            if (!Init->user_empty()) {
                LUISA_WARNING_WITH_LOCATION("Unused ray query 'initialize' intrinsic remaining.");
                while (!Init->user_empty()) {
                    if (auto Call = llvm::dyn_cast<llvm::CallInst>(*Init->user_begin())) {
                        Call->eraseFromParent();
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid user of ray query 'initialize' intrinsic.");
                    }
                }
            }
            Init->eraseFromParent();
            return true;
        };
        auto ReplaceIntrinsic = [&](llvm::CallInst *Inst, llvm::StringRef NewIntrinsic) noexcept {
            auto Func = M.getFunction(NewIntrinsic);
            if (Func == nullptr) {
                Func = llvm::Function::Create(Inst->getFunctionType(), llvm::Function::ExternalLinkage,
                                              NewIntrinsic, &M);
            }
            Inst->setCalledFunction(Func);
        };
        auto Proceed = M.getFunction(CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_proceed);
        if (Proceed == nullptr) { return RemoveUnusedInitCalls(); }
        while (!Proceed->user_empty()) {
            auto Call = llvm::dyn_cast<llvm::CallInst>(*Proceed->user_begin());
            LUISA_ASSERT(Call != nullptr, "Invalid user of ray query 'proceed' intrinsic.");
            auto &FAM = MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();
            auto F = Call->getFunction();
            auto &DT = FAM.getResult<llvm::DominatorTreeAnalysis>(*F);
            auto &LI = FAM.getResult<llvm::LoopAnalysis>(*F);
            auto L = LI.getLoopFor(Call->getParent());
            LUISA_ASSERT(L != nullptr && Init != nullptr, "Ray query 'proceed' call not inside a loop or missing 'initialize' intrinsic.");
            // find the `initialize` call that dominates this `proceed` call
            auto InitCall = [&]() noexcept -> llvm::CallInst * {
                auto IDom = [&DT](auto B) noexcept {
                    auto IDomNode = DT.getNode(B)->getIDom();
                    return IDomNode ? IDomNode->getBlock() : nullptr;
                };
                for (auto BB = IDom(Call->getParent()); BB != nullptr; BB = IDom(BB)) {
                    for (auto &I : llvm::reverse(*BB)) {
                        if (auto C = llvm::dyn_cast<llvm::CallInst>(&I);
                            C != nullptr && C->getCalledFunction() == Init) {
                            return C;
                        }
                    }
                }
                return nullptr;
            }();
            LUISA_ASSERT(InitCall != nullptr, "No dominating ray query 'initialize' call found for 'proceed' call.");
            // find the outermost loop that is dominated by this `initialize` call
            auto DominatesLoop = [&DT, InitCall](llvm::Loop *loop) noexcept {
                return DT.dominates(InitCall->getParent(), loop->getHeader());
            };
            while (auto ParentL = L->getParentLoop()) {
                if (DominatesLoop(ParentL)) {
                    L = ParentL;
                } else {
                    break;
                }
            }
            LUISA_ASSERT(L != nullptr && DominatesLoop(L), "Failed to find the outermost ray query loop for 'proceed' call.");
            auto AC = FAM.getCachedResult<llvm::AssumptionAnalysis>(*F);
            llvm::CodeExtractor Extractor{L->getBlocks(), &DT, false, nullptr, nullptr, AC};
            llvm::CodeExtractorAnalysisCache CEAC{*F};
            auto NewF = Extractor.extractCodeRegion(CEAC);
            LUISA_ASSERT(NewF != nullptr, "Failed to extract ray query loop into a separate function.");
            NewF->addFnAttr(llvm::Attribute::NoInline);
            NewF->setName("ray.query.loop.extracted");
            RayQueryFunctions.insert(NewF);
            ReplaceIntrinsic(InitCall, CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_spawn);
            ReplaceIntrinsic(Call, CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_dispatch);
            FAM.invalidate(*F, llvm::PreservedAnalyses::none());
        }
        // remove these functions
        Proceed->eraseFromParent();
        RemoveUnusedInitCalls();
        return true;
    }

public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) noexcept {
        if (extractRayQueryLoops(M, MAM)) {
            return llvm::PreservedAnalyses::none();
        }
        return llvm::PreservedAnalyses::all();
    }
};

}// namespace detail

luisa::string CUDACodegenLLVMImpl::_generate_ptx() const noexcept {
    llvm::SmallVector<char, 256> ptx;
    llvm::raw_svector_ostream os{ptx};
    detail::NVPTXPassManagerStub pass_manager;
    if (_target_machine->addPassesToEmitFile(pass_manager, os, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
        LUISA_ERROR_WITH_LOCATION("TargetMachine can't emit PTX.");
    }
    pass_manager.run(*_llvm_module);
    return {ptx.begin(), ptx.end()};
}

luisa::string CUDACodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    _analyze_ray_tracing_usage(xir_module);
    _llvm_module->setSourceFileName(std::string_view{_config.source_file});
    _llvm_module->setModuleIdentifier(xir_module.name().value_or(""));
    for (auto func : xir_module.function_list()) {
        if (auto def = func->definition()) {
            [[maybe_unused]] auto llvm_f = _translate_function(def);
        }
    }
    auto verify = [&] {
        if (llvm::verifyModule(*_llvm_module, &llvm::errs())) {
            std::error_code ec;
            if (llvm::raw_fd_ostream os{"debug.ll", ec}; ec) {
                LUISA_WARNING_WITH_LOCATION("Failed to create debug.ll: {}", ec.message());
            } else {
                _llvm_module->print(os, nullptr, true, true);
            }
            LUISA_ERROR_WITH_LOCATION("LLVM module verification failed. IR dumped to debug.ll");
        }
    };
    verify();
    if (_rt_analysis.uses_ray_query) {
        // we need to inline all device functions so that ray query extraction can work
        for (auto &&f : *_llvm_module) {
            if (!f.isDeclaration() && f.getCallingConv() == llvm::CallingConv::PTX_Device) {
                f.removeFnAttr(llvm::Attribute::NoInline);
                f.addFnAttr(llvm::Attribute::AlwaysInline);
            }
        }
        _run_optimization_passes([](auto &MPM) noexcept {
            MPM.addPass(detail::RayQueryLoopExtraction{});
        });
        _materialize_ray_query_loops();
        verify();
    }
    _run_optimization_passes();
    static auto dump_llvm_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    if (dump_llvm_ir) {
        _llvm_module->print(llvm::errs(), nullptr, false, true);
    }
    return _generate_ptx();
}

}// namespace luisa::compute::cuda
