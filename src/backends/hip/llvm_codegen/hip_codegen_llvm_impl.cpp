//
// Created by mike on 3/18/26.
//

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/core/clock.h>
#include "hip_codegen_llvm_impl.h"
#include "hip_rt_device_wrapper.hip"
#include "hip_codegen_llvm_device_bitcode.h"
#include "hip_rt_wrapper_bitcode_embedded.h"

// Per-arch HIPRT library bitcode (extracted from the HIPRT bundle at build time).
// Each provides luisa_compute_hip_hiprt_library_gfxNNNN[] and _size.
#include "hiprt_library_gfx1030_embedded.h"
#include "hiprt_library_gfx1100_embedded.h"
#include "hiprt_library_gfx1200_embedded.h"
#include "hiprt_library_gfx1201_embedded.h"

#undef None

namespace luisa::compute::hip {

HIPCodegenLLVMImpl::FunctionContext::FunctionContext(llvm::Function *f) noexcept
    : llvm_func{f},
      llvm_alloca_block{llvm::BasicBlock::Create(f->getContext(), "alloca", f)},
      llvm_entry_block{llvm::BasicBlock::Create(f->getContext(), "entry", f)} {
    IB b{llvm_alloca_block};
    b.CreateBr(llvm_entry_block);
}

HIPCodegenLLVMImpl::HIPCodegenLLVMImpl(HIPCodegenLLVMConfig config) noexcept
    : _config{std::move(config)} {
    LUISA_ASSERT(_config.block_size[0] > 0u && _config.block_size[1] > 0u && _config.block_size[2] > 0u,
                 "Block size must be constant and greater than zero for now.");
    Clock clk;
    _initialize();
    LUISA_VERBOSE_WITH_LOCATION("HIP LLVM codegen initialized in {} ms.", clk.toc());
}

void HIPCodegenLLVMImpl::_initialize() noexcept {
    static std::once_flag once_flag;
    std::call_once(once_flag, [] {
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmPrinter();
        LLVMInitializeAMDGPUAsmParser();
    });

    static auto target = [] {
        std::string error;
        if (auto t = llvm::TargetRegistry::lookupTarget(llvm::Triple{amdgpu_target_triple}, error)) {
            return t;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to lookup target '{}': {}", amdgpu_target_triple, error);
    }();

    llvm::TargetOptions options;
    options.NoTrappingFPMath = true;
    if (_config.enable_fast_math) {
        options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    }

    auto opt_level = llvm::CodeGenOptLevel::Default;
    switch (_config.opt_level) {
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_NONE: opt_level = llvm::CodeGenOptLevel::None; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_LESS: opt_level = llvm::CodeGenOptLevel::Less; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_DEFAULT: opt_level = llvm::CodeGenOptLevel::Default; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE: opt_level = llvm::CodeGenOptLevel::Aggressive; break;
    }

    auto cpu_name = fmt::format("gfx{}", _config.amdgpu_arch);
    _target_machine = target->createTargetMachine(
        llvm::Triple{amdgpu_target_triple}, llvm::StringRef{cpu_name}, {},
        options, llvm::Reloc::Static, llvm::CodeModel::Small, opt_level);

    _data_layout = std::make_unique<llvm::DataLayout>(_target_machine->createDataLayout());

    // parse OCML bitcode as the starting module (like CUDA's libdevice)
    _llvm_module = [&] {
        llvm::SMDiagnostic error;
        llvm::StringRef bc{reinterpret_cast<const char *>(luisa_compute_hip_ocml),
                           luisa_compute_hip_ocml_size};
        if (auto m = llvm::parseIR({bc, "ocml.bc"}, error, _llvm_context)) {
            llvm::StripDebugInfo(*m);
            return m;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to parse OCML bitcode: {}", error.getMessage());
    }();

    // set target triple and data layout
    _llvm_module->setTargetTriple(llvm::Triple{amdgpu_target_triple});
    _llvm_module->setDataLayout(*_data_layout);

    // internalize all OCML functions
    for (auto &&f : *_llvm_module) {
        if (f.getName().starts_with("__ocml_")) {
            f.setLinkage(llvm::Function::PrivateLinkage);
            f.removeFnAttr(llvm::Attribute::StackProtect);
        }
    }

    // provide OCLC configuration globals that OCML depends on
    auto set_oclc_option = [&](llvm::StringRef name, llvm::Value *value) {
        if (auto gv = _llvm_module->getGlobalVariable(name)) {
            llvm::SmallVector<llvm::LoadInst *, 8> loads;
            for (auto user : gv->users()) {
                if (auto load = llvm::dyn_cast<llvm::LoadInst>(user)) {
                    loads.emplace_back(load);
                }
            }
            for (auto load : loads) {
                load->replaceAllUsesWith(value);
                load->eraseFromParent();
            }
            if (gv->use_empty()) {
                gv->eraseFromParent();
            }
        }
    };
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    set_oclc_option("__oclc_finite_only_opt", llvm::ConstantInt::get(llvm_i8_type, _config.enable_fast_math ? 1 : 0));
    set_oclc_option("__oclc_unsafe_math_opt", llvm::ConstantInt::get(llvm_i8_type, _config.enable_fast_math ? 1 : 0));
    auto isa_version = (_config.amdgpu_arch / 100) * 1000 + (_config.amdgpu_arch % 100) * 10;
    set_oclc_option("__oclc_ISA_version", llvm::ConstantInt::get(llvm_i32_type, isa_version));

    _llvm_buffer_type = _get_llvm_buffer_type();
    _llvm_texture_type = _get_llvm_texture_type();
    _llvm_bindless_array_type = _get_llvm_bindless_array_type();
    _llvm_bindless_array_slot_type = _get_llvm_bindless_array_slot_type();
    _llvm_accel_type = _get_llvm_accel_type();
    _llvm_accel_instance_type = _get_llvm_accel_instance_type();
    _llvm_ray_type = _get_llvm_ray_type();
    _llvm_surface_hit_type = _get_llvm_surface_hit_type();
    _llvm_procedural_hit_type = _get_llvm_procedural_hit_type();
    _llvm_committed_hit_type = _get_llvm_committed_hit_type();
    _llvm_ray_query_type = _get_llvm_ray_query_type();
}

void HIPCodegenLLVMImpl::_postprocess_rt_kernel() noexcept {

    if (!_rt_analysis.uses_ray_tracing) { return; }

    // Step 1: Link the RT wrapper bitcode (hiprt traversal wrappers)
    llvm::StringRef wrapper_bc{
        reinterpret_cast<const char *>(luisa_compute_hip_hip_rt_wrapper),
        luisa_compute_hip_hip_rt_wrapper_size};
    auto wrapper_buf = llvm::MemoryBuffer::getMemBuffer(wrapper_bc, "hip_rt_wrapper", false);
    auto wrapper_module = llvm::parseBitcodeFile(*wrapper_buf, _llvm_context);
    if (!wrapper_module) {
        LUISA_ERROR_WITH_LOCATION("Failed to parse HIPRT wrapper bitcode.");
    }

    if (auto *wrapper_flags = (*wrapper_module)->getNamedMetadata("llvm.module.flags")) {
        wrapper_flags->eraseFromParent();
    }

    if (llvm::Linker::linkModules(*_llvm_module, std::move(*wrapper_module))) {
        LUISA_ERROR_WITH_LOCATION("Failed to link kernel module with HIPRT wrapper bitcode.");
    }

    // Step 2: Replace the extern __shared__ declaration of luisa_hiprt_shared_stack_cache
    // with a sized definition based on the actual kernel block size.
    {
        auto block_size = _config.block_size[0] * _config.block_size[1] * _config.block_size[2];
        LUISA_ASSERT(block_size > 0u, "Block size must be greater than zero.");
        auto shared_array_size = LUISA_HIPRT_SHARED_STACK_SIZE * block_size;
        if (auto old_gv = _llvm_module->getGlobalVariable("luisa_hiprt_shared_stack_cache")) {
            auto i32_ty = llvm::Type::getInt32Ty(_llvm_context);
            auto array_ty = llvm::ArrayType::get(i32_ty, shared_array_size);
            auto new_gv = new llvm::GlobalVariable(
                *_llvm_module, array_ty, false,
                llvm::GlobalValue::InternalLinkage,
                llvm::UndefValue::get(array_ty),
                "luisa_hiprt_shared_stack_cache_tmp",
                nullptr,
                llvm::GlobalValue::NotThreadLocal,
                3u);// addrspace(3) = shared/LDS
            new_gv->setAlignment(llvm::Align(4));
            LUISA_INFO("Replacing shared stack cache: {} uses, old type = [0 x i32], new type = [{} x i32]",
                       old_gv->getNumUses(), shared_array_size);
            old_gv->replaceAllUsesWith(new_gv);
            old_gv->eraseFromParent();
            new_gv->setName("luisa_hiprt_shared_stack_cache");
        } else {
            LUISA_ERROR_WITH_LOCATION("Could not find luisa_hiprt_shared_stack_cache in linked module!");
        }
    }

    // Step 3: Link arch-specific HIPRT library bitcode for full LTO inlining.
    // This replaces HIPRT's orortcLinkComplete which only links without optimizing.
    {
        const unsigned char *hiprt_lib_data = nullptr;
        unsigned long long hiprt_lib_size = 0;
        switch (_config.amdgpu_arch) {
            case 1030:
                hiprt_lib_data = luisa_compute_hip_hiprt_library_gfx1030;
                hiprt_lib_size = luisa_compute_hip_hiprt_library_gfx1030_size;
                break;
            case 1100:
                hiprt_lib_data = luisa_compute_hip_hiprt_library_gfx1100;
                hiprt_lib_size = luisa_compute_hip_hiprt_library_gfx1100_size;
                break;
            case 1200:
                hiprt_lib_data = luisa_compute_hip_hiprt_library_gfx1200;
                hiprt_lib_size = luisa_compute_hip_hiprt_library_gfx1200_size;
                break;
            case 1201:
                hiprt_lib_data = luisa_compute_hip_hiprt_library_gfx1201;
                hiprt_lib_size = luisa_compute_hip_hiprt_library_gfx1201_size;
                break;
            default:
                LUISA_ERROR_WITH_LOCATION("Unsupported AMDGPU arch {} for HIPRT LTO.", _config.amdgpu_arch);
        }
        LUISA_ASSERT(hiprt_lib_data != nullptr && hiprt_lib_size > 0,
                     "HIPRT library bitcode is empty for arch gfx{}.", _config.amdgpu_arch);

        llvm::StringRef hiprt_bc{reinterpret_cast<const char *>(hiprt_lib_data),
                                 static_cast<size_t>(hiprt_lib_size)};
        auto hiprt_buf = llvm::MemoryBuffer::getMemBuffer(hiprt_bc, "hiprt_library", false);
        auto hiprt_module = llvm::parseBitcodeFile(*hiprt_buf, _llvm_context);
        if (!hiprt_module) {
            LUISA_ERROR_WITH_LOCATION("Failed to parse HIPRT library bitcode for gfx{}.", _config.amdgpu_arch);
        }

        if (auto *flags = (*hiprt_module)->getNamedMetadata("llvm.module.flags")) {
            flags->eraseFromParent();
        }

        LUISA_INFO("Linking HIPRT library bitcode ({} bytes, {} functions) for gfx{}.",
                   hiprt_lib_size,
                   std::distance((*hiprt_module)->begin(), (*hiprt_module)->end()),
                   _config.amdgpu_arch);

        if (llvm::Linker::linkModules(*_llvm_module, std::move(*hiprt_module),
                                      llvm::Linker::Flags::OverrideFromSrc)) {
            LUISA_ERROR_WITH_LOCATION("Failed to link HIPRT library bitcode.");
        }

        // HIPRT bitcode uses "comdat any" which prevents LLVM from inlining
        // even with alwaysinline — strip to enable full inlining into kernel.
        for (auto &func : *_llvm_module) {
            func.setComdat(nullptr);
        }
        for (auto &gv : _llvm_module->globals()) {
            gv.setComdat(nullptr);
        }
        _llvm_module->getComdatSymbolTable().clear();
        LUISA_INFO("Stripped comdat groups from HIPRT library functions.");
    }

    // Step 4: Provide trivial intersectFunc/filterFunc definitions.
    // HIPRT library calls these for custom geometry/filter callbacks.
    // We use numGeomTypes=0, numRayTypes=1, funcNameSets=nullptr, so both always return false.
    {
        for (auto &func : *_llvm_module) {
            if (!func.isDeclaration()) { continue; }
            auto name = func.getName();
            if (!name.contains("intersectFunc") && !name.contains("filterFunc")) { continue; }
            if (func.getReturnType() != llvm::Type::getInt1Ty(_llvm_context)) { continue; }
            auto *entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", &func);
            IB builder{entry_bb};
            builder.CreateRet(builder.getFalse());
            LUISA_INFO("Provided trivial definition for HIPRT function: {}", name.str());
        }
    }
}

void HIPCodegenLLVMImpl::_dump_module(const std::filesystem::path &path) const noexcept {
    std::error_code ec;
    llvm::raw_fd_ostream out{path.string(), ec};
    if (ec) {
        LUISA_WARNING_WITH_LOCATION("Failed to open file for dumping LLVM module: {}.", ec.message());
    } else {
        _llvm_module->print(out, nullptr);
    }
}

void HIPCodegenLLVMImpl::_run_optimization_passes() noexcept {

    if (_config.enable_fast_math) {
        for (auto &f : *_llvm_module) {
            for (auto &bb : f) {
                for (auto &inst : bb) {
                    if (llvm::isa<llvm::FPMathOperator>(inst)) {
                        inst.setFast(true);
                    }
                }
            }
        }
    }

    for (auto &&func : *_llvm_module) {
        if (!func.isDeclaration() && func.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL) {
            func.setLinkage(llvm::Function::PrivateLinkage);
            func.addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }

    // Resolve aliases to actual functions so they get PrivateLinkage + AlwaysInline.
    // HIPRT bitcode has C++ ctor/dtor delegation aliases (C1→C2, D1→D2) which the
    // function iterator above doesn't visit — leaving 18 functions un-inlined.
    {
        llvm::SmallVector<llvm::GlobalAlias *, 32> aliases_to_resolve;
        for (auto &alias : _llvm_module->aliases()) {
            aliases_to_resolve.push_back(&alias);
        }
        for (auto *alias : aliases_to_resolve) {
            auto *aliasee = alias->getAliasee();
            auto *fn = llvm::dyn_cast<llvm::Function>(aliasee);
            if (!fn) { continue; }
            auto *new_fn = llvm::Function::Create(
                fn->getFunctionType(), llvm::Function::PrivateLinkage,
                alias->getName() + ".resolved", _llvm_module.get());
            new_fn->copyAttributesFrom(fn);
            new_fn->setLinkage(llvm::Function::PrivateLinkage);
            new_fn->addFnAttr(llvm::Attribute::AlwaysInline);
            auto *entry = llvm::BasicBlock::Create(_llvm_context, "entry", new_fn);
            IB builder{entry};
            llvm::SmallVector<llvm::Value *, 8> args;
            for (auto &arg : new_fn->args()) {
                args.push_back(&arg);
            }
            auto *call = builder.CreateCall(fn, args);
            call->setCallingConv(fn->getCallingConv());
            call->setTailCall(true);
            if (fn->getReturnType()->isVoidTy()) {
                builder.CreateRetVoid();
            } else {
                builder.CreateRet(call);
            }
            alias->replaceAllUsesWith(new_fn);
            alias->eraseFromParent();
        }
    }

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
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_NONE: opt_level = llvm::OptimizationLevel::O0; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_LESS: opt_level = llvm::OptimizationLevel::O1; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_DEFAULT: opt_level = llvm::OptimizationLevel::O2; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE: opt_level = llvm::OptimizationLevel::O3; break;
    }
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt_level);
    MPM.run(*_llvm_module, MAM);

    // make hiprt/hiprtc happy
    for (auto &func : *_llvm_module) {
        auto attrs = func.getAttributes().removeAttributeAtIndex(
            func.getContext(), llvm::AttributeList::FunctionIndex,
            llvm::Attribute::NoCreateUndefOrPoison);
        func.setAttributes(attrs);
        for (auto &bb : func) {
            for (auto &inst : bb) {
                if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&inst)) {
                    auto cb_attrs = cb->getAttributes().removeAttributeAtIndex(
                        cb->getContext(), llvm::AttributeList::FunctionIndex,
                        llvm::Attribute::NoCreateUndefOrPoison);
                    cb->setAttributes(cb_attrs);
                }
            }
        }
    }
}

luisa::string HIPCodegenLLVMImpl::_generate_code() const noexcept {
    std::string code;
    llvm::raw_string_ostream os{code};
    llvm::WriteBitcodeToFile(*_llvm_module, os);
    os.flush();
    return luisa::string{code};
}

luisa::string HIPCodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    Clock clk;
    _initialize();

    _rt_analysis.uses_ray_tracing = _config.requires_ray_tracing;

    for (auto f : xir_module.function_list()) {
        if (auto def = f->definition()) {
            static_cast<void>(_translate_function(def));
        }
    }

    _postprocess_rt_kernel();

    if (llvm::verifyModule(*_llvm_module, &llvm::errs())) {
        _dump_module("debug_bad_module.ll");
        LUISA_ERROR_WITH_LOCATION("Module verification failed.");
    }

    static auto dump_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    static std::atomic<uint32_t> dump_counter{0u};
    uint32_t dump_idx = 0u;
    if (dump_ir) {
        dump_idx = dump_counter.fetch_add(1u);
        auto filename = fmt::format("hip_kernel_before_opt_{}.ll", dump_idx);
        _dump_module(filename);
        LUISA_INFO("Dumped LLVM IR to: {}", filename);
    }

    _run_optimization_passes();

    if (dump_ir) {
        auto after_opt_filename = fmt::format("hip_kernel_after_opt_{}.ll", dump_idx);
        _dump_module(after_opt_filename);
        LUISA_INFO("Dumped post-optimization LLVM IR to: {}", after_opt_filename);
    }

    auto target_cpu = _target_machine->getTargetCPU();
    auto target_features = _target_machine->getTargetFeatureString();
    for (auto &func : *_llvm_module) {
        // Collect amdgpu-no-* string attributes from kernel_main before stripping.
        // These are added by AMDGPUAttributor during optimization and are critical
        // for correct kernarg segment layout: without them, the AMDGPU backend
        // assumes 256 bytes of implicit arguments, which can cause memory faults.
        llvm::SmallVector<llvm::StringRef, 24> amdgpu_no_attrs;
        if (func.getName() == "kernel_main") {
            for (auto &attr : func.getAttributes().getFnAttrs()) {
                if (attr.isStringAttribute() &&
                    attr.getKindAsString().starts_with("amdgpu-no-")) {
                    amdgpu_no_attrs.push_back(attr.getKindAsString());
                }
            }
        }

        func.setAttributes(llvm::AttributeList{});

        if (func.getName() == "kernel_main") {
            func.addFnAttr(llvm::Attribute::NoInline);
            // Restore amdgpu-no-* attributes that the optimizer determined are safe.
            for (auto &attr_name : amdgpu_no_attrs) {
                func.addFnAttr(attr_name);
            }
        }
        // Re-add target CPU and features so that downstream consumers
        // (e.g., HIPRT bitcode compiler) know the GPU architecture.
        if (!func.isDeclaration()) {
            func.addFnAttr("target-cpu", target_cpu);
            if (!target_features.empty()) {
                func.addFnAttr("target-features", target_features);
            }
        }
    }

    static auto print_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_PRINT_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    if (print_ir) {
        _llvm_module->print(llvm::outs(), nullptr);
    }

    LUISA_INFO_WITH_LOCATION("HIP LLVM codegen completed in {} ms.", clk.toc());
    return _generate_code();
}

}// namespace luisa::compute::hip
