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

#include <luisa/core/clock.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

#include "cuda_codegen_llvm_device_bitcode.h"
#include "cuda_codegen_llvm_impl.h"

#ifdef LUISA_COMPUTE_ENABLE_CUDA_OPTIX_IR
#include <llvm/IR/Metadata.h>
#include "../../../ext/llvm_downgrade.h"
#include "cuda_codegen_llvm_optix_ir.h"
#include "cuda_codegen_llvm_optix_ir_legalize.h"
#endif

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

void CUDACodegenLLVMImpl::_dump_module(const luisa::filesystem::path &path) const noexcept {
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

luisa::string CUDACodegenLLVMImpl::_generate_optix_ir() noexcept {
#ifdef LUISA_COMPUTE_ENABLE_CUDA_OPTIX_IR
    LUISA_ASSERT(_rt_analysis.uses_ray_tracing, "OptiX IR requires a ray-tracing kernel.");
    _legalize_optix_ir_atomics();
    luisa_compute_cuda_llvm_legalize_optix_ir(*_llvm_module);
    // Interpret kernel annotations before setting NVIDIA's consumed marker.
    // PTX_Kernel already carries the entry ABI and must not be rewritten.
    llvm::DenseSet<llvm::Function *> annotated_kernels;
    if (auto annotations = _llvm_module->getNamedMetadata("nvvm.annotations")) {
        for (auto node : annotations->operands()) {
            LUISA_ASSERT(node != nullptr && node->getNumOperands() >= 3u,
                         "Malformed NVVM annotation in OptiX IR input.");
            auto annotation = llvm::dyn_cast_or_null<llvm::MDString>(node->getOperand(1));
            // The supplied pure-writer protocol only establishes the kernel
            // transplant. Do not mark other unhandled semantics as consumed.
            LUISA_ASSERT(annotation != nullptr && annotation->getString() == "kernel",
                         "Unsupported NVVM annotation in the experimental OptiX IR writer.");
            auto constant = llvm::dyn_cast_or_null<llvm::ConstantAsMetadata>(node->getOperand(2));
            auto enabled = constant == nullptr ? nullptr : llvm::dyn_cast<llvm::ConstantInt>(constant->getValue());
            LUISA_ASSERT(enabled != nullptr, "Invalid NVVM kernel annotation value.");
            if (enabled->isZero()) { continue; }
            auto value = llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(node->getOperand(0));
            auto function = value == nullptr ? nullptr : llvm::dyn_cast<llvm::Function>(value->getValue());
            LUISA_ASSERT(function != nullptr, "Invalid NVVM kernel annotation target.");
            annotated_kernels.insert(function);
        }
    }
    constexpr std::array<std::string_view, 9u> entry_prefixes{
        "__raygen__", "__miss__", "__closesthit__", "__anyhit__", "__intersection__",
        "__direct_callable__", "__continuation_callable__", "__exception__", "__callable__"};
    for (auto &function : *_llvm_module) {
        if (function.isDeclaration()) { continue; }
        auto ptx_kernel = function.getCallingConv() == llvm::CallingConv::PTX_Kernel;
        auto optix_entry = false;
        for (auto prefix : entry_prefixes) {
            optix_entry |= function.getName().starts_with(prefix);
        }
        auto annotated_kernel = annotated_kernels.contains(&function);
        if (ptx_kernel || optix_entry || annotated_kernel) {
            function.removeFnAttr(llvm::Attribute::NoInline);
            function.removeFnAttr(llvm::Attribute::OptimizeNone);
            function.addFnAttr(llvm::Attribute::AlwaysInline);
        }
        if (!ptx_kernel && (optix_entry || annotated_kernel)) {
            function.addFnAttr("nvvm.kernel");
        }
        function.addFnAttr("nvvm.annotations_transplanted");
    }
    LUISA_ASSERT(!llvm::verifyModule(*_llvm_module, &llvm::errs()),
                 "Invalid LLVM module after OptiX IR annotation transplant.");
    // The downgrade consumes the module and writes immediately after typed
    // pointer reconstruction. Never run LLVM optimization on that result.
    auto bitcode = llvm_downgrade_to_7(std::move(_llvm_module));
    return luisa_compute_cuda_llvm_encode_optix_ir({bitcode.data(), bitcode.size()}, _config.cuda_arch);
#else
    LUISA_ERROR_WITH_LOCATION("CUDA OptiX IR output was requested without LUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_OPTIX_IR.");
#endif
}

luisa::string CUDACodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    _analyze_ray_tracing_usage(xir_module);
    _llvm_module->setSourceFileName(luisa::string_view{_config.source_file});
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
    if (_rt_analysis.uses_ray_query) {
        _materialize_ray_query_pipelines();
        // Inline callbacks and helpers into their OptiX entry programs.
        for (auto &&f : *_llvm_module) {
            if (!f.isDeclaration() && f.getCallingConv() == llvm::CallingConv::PTX_Device) {
                f.removeFnAttr(llvm::Attribute::NoInline);
                f.addFnAttr(llvm::Attribute::AlwaysInline);
            }
        }
    }
    verify();
    _run_optimization_passes();
    verify();
    static auto dump_llvm_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    if (dump_llvm_ir) {
        _llvm_module->print(llvm::errs(), nullptr, false, true);
    }
    switch (_config.output_format) {
        case CUDACodegenLLVMConfig::OutputFormat::PTX: return _generate_ptx();
        case CUDACodegenLLVMConfig::OutputFormat::OPTIX_IR: return _generate_optix_ir();
    }
    LUISA_ERROR_WITH_LOCATION("Invalid CUDA LLVM output format.");
}

}// namespace luisa::compute::cuda
