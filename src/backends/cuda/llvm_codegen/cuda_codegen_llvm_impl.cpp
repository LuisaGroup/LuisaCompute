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
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/Utils/Cloning.h>

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

void CUDACodegenLLVMImpl::_run_inline_pass_only() noexcept {
    // Run only the inliner to inline alwaysinline functions
    // This must run before other optimizations to prevent call target corruption
    LUISA_INFO("Running inline pass for ray query handlers...");

    // First, try the standard AlwaysInliner pass
    llvm::PassBuilder PB;
    llvm::ModuleAnalysisManager MAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::LoopAnalysisManager LAM;
    llvm::CGSCCAnalysisManager CGAM;

    // Register all analyses
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);

    // Cross-register analysis managers
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::ModulePassManager MPM;
    // Add only the always-inline pass
    MPM.addPass(llvm::AlwaysInlinerPass());

    MPM.run(*_llvm_module, MAM);

    // Helper to detect handler functions by intrinsic scanning (not name matching)
    auto is_handler = [](llvm::Function *f) -> bool {
        if (f == nullptr || f->isDeclaration()) return false;
        for (auto &block : *f) {
            for (auto &inst : block) {
                if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                    if (auto *callee = call->getCalledFunction()) {
                        if (callee->getName() == llvm::StringRef(CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_dispatch)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    };

    // Manually inline any remaining handler calls
    // The AlwaysInliner might not inline functions with pointer arguments
    LUISA_INFO("Checking for remaining handler calls to inline manually...");
    int call_count = 0;
    for (auto &func : *_llvm_module) {
        for (auto &block : func) {
            for (auto &inst : block) {
                if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                    if (auto *callee = call->getCalledFunction()) {
                        if (is_handler(callee)) {
                            call_count++;
                        }
                    }
                }
            }
        }
    }
    LUISA_INFO("Found {} handler calls remaining", call_count);

    bool inlined = true;
    int iterations = 0;
    while (inlined && iterations < 10) {
        inlined = false;
        iterations++;
        for (auto &func : *_llvm_module) {
            // Only inline handlers into OptiX entry points, not into kernel or other functions
            auto func_name = func.getName();
            bool is_optix_entry_point = (func_name == "__anyhit__ray_query" ||
                                         func_name == "__intersection__ray_query");
            if (!is_optix_entry_point) continue;

            for (auto &block : func) {
                for (auto it = block.begin(); it != block.end();) {
                    auto *call = llvm::dyn_cast<llvm::CallInst>(&*it++);
                    if (!call) continue;

                    auto *callee = call->getCalledFunction();
                    if (!callee) continue;

                    // Check if this is a handler function using intrinsic scanning
                    if (is_handler(callee)) {
                        LUISA_INFO("Manually inlining handler call to: {}", callee->getName().str());
                        llvm::InlineFunctionInfo info;
                        auto result = llvm::InlineFunction(*call, info);
                        if (result.isSuccess()) {
                            inlined = true;
                            LUISA_INFO("Successfully inlined handler");
                        } else {
                            LUISA_WARNING("Failed to inline handler: {}", result.getFailureReason());
                        }
                    }
                }
            }
        }
    }

    LUISA_INFO("Inline pass completed");

    // Force remove all handler functions
    // They cause "Illegal call target" errors in OptiX even if not called
    LUISA_INFO("Force removing all handler functions...");
    llvm::SmallVector<llvm::Function *, 8> handlers_to_remove;
    LUISA_INFO("Iterating over {} functions in module", std::distance(_llvm_module->begin(), _llvm_module->end()));
    for (auto &func : *_llvm_module) {
        LUISA_INFO("Function: {}", func.getName().str());
        // Use intrinsic scanning instead of name matching to detect handlers
        if (is_handler(&func)) {
            LUISA_INFO("  -> matches handler pattern (via intrinsic scanning)!");
            handlers_to_remove.push_back(&func);
        }
    }
    LUISA_INFO("Found {} handler functions to remove", handlers_to_remove.size());
    for (auto *func : handlers_to_remove) {
        LUISA_INFO("Removing handler function: {}", func->getName().str());
        // Replace all uses with null first (shouldn't have any if inlined)
        if (!func->user_empty()) {
            LUISA_WARNING("Handler {} has {} uses, replacing with null",
                          func->getName().str(), func->getNumUses());
            func->replaceAllUsesWith(llvm::Constant::getNullValue(func->getType()));
        }
        func->eraseFromParent();
    }
}

void CUDACodegenLLVMImpl::_remove_unused_pseudo_intrinsics() noexcept {
    // Lower and remove ray query pseudo-intrinsics that may be called from kernel
    LUISA_INFO("Lowering and removing pseudo-intrinsics...");

    // First, lower any remaining intrinsics in the module
    for (auto &func : *_llvm_module) {
        for (auto &block : func) {
            for (auto it = block.begin(); it != block.end();) {
                auto *call = llvm::dyn_cast<llvm::CallInst>(&*it++);
                if (!call) continue;

                auto *callee = call->getCalledFunction();
                if (!callee) continue;

                auto name = callee->getName();
                if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_committed_hit)) {
                    // luisa.ray.query.committed.hit() - construct committed hit from OptiX hit object
                    // Follows the logic in lc_ray_query_decode_hit() from cuda_device_resource.h
                    // { i32 inst_id, i32 prim_id, <2 x float> bary, i32 hit_kind, float t }
                    // where hit_kind is transformed from OptiX hit kind to LCHitType enum:
                    // - If OptiX hit_kind > 127 (triangle), use LCHitType::BUILTIN (1)
                    // - Otherwise, use LCHitType::PROCEDURAL (2)
                    // - If no hit, use LCHitType::MISS (0)
                    IB b{_llvm_context};
                    b.SetInsertPoint(call);

                    // Get hit data from OptiX hit object
                    auto is_hit = _call_optix_hit_object_is_hit(b);
                    auto invalid_id = b.getInt32(~0u);
                    auto inst_id = b.CreateSelect(is_hit, _call_optix_hit_object_instance_index(b), invalid_id);
                    auto prim_id = _call_optix_hit_object_primitive_index(b);
                    auto bary = _call_optix_hit_object_triangle_barycentrics(b);
                    auto optix_hit_kind = _call_optix_hit_object_hit_kind(b);
                    auto t = _call_optix_hit_object_ray_t_max(b);

                    // Transform OptiX hit kind to LCHitType
                    // LCHitType: MISS = 0, BUILTIN = 1, PROCEDURAL = 2
                    // OptiX hit_kind > 127 means triangle (BUILTIN), else PROCEDURAL
                    auto hit_kind_gt_127 = b.CreateICmpUGT(optix_hit_kind, b.getInt32(127u));
                    auto hit_kind = b.CreateSelect(hit_kind_gt_127,
                                                   b.getInt32(CUDACodegenLLVMImpl::llvm_hit_type_builtin),
                                                   b.getInt32(CUDACodegenLLVMImpl::llvm_hit_type_procedural));

                    // If no hit, set hit_kind to MISS
                    auto miss_kind = b.getInt32(CUDACodegenLLVMImpl::llvm_hit_type_miss);
                    hit_kind = b.CreateSelect(is_hit, hit_kind, miss_kind);

                    // Construct the committed hit struct
                    llvm::Value *result = llvm::PoisonValue::get(call->getType());
                    result = b.CreateInsertValue(result, inst_id, 0u);
                    result = b.CreateInsertValue(result, prim_id, 1u);
                    result = b.CreateInsertValue(result, bary, 2u);
                    result = b.CreateInsertValue(result, hit_kind, 3u);
                    result = b.CreateInsertValue(result, t, 4u);

                    call->replaceAllUsesWith(result);
                    call->eraseFromParent();
                }
            }
        }
    }

    // Now remove unused declarations
    llvm::SmallVector<llvm::Function *, 16> to_remove;
    for (auto &func : *_llvm_module) {
        auto name = func.getName();
        if (name.starts_with("luisa.ray.query.")) {
            if (func.user_empty()) {
                to_remove.push_back(&func);
                LUISA_INFO("Removing unused pseudo-intrinsic: {}", name.str());
            } else {
                LUISA_WARNING("Pseudo-intrinsic still has users: {} ({} users)",
                              name.str(), func.getNumUses());
            }
        }
    }

    for (auto *func : to_remove) {
        func->eraseFromParent();
    }

    LUISA_INFO("Removed {} unused pseudo-intrinsics", to_remove.size());
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
        // Disable MergeFunctions for ray query shaders to prevent OptiX entry points from being merged
        PTO.MergeFunctions = !_rt_analysis.uses_ray_query;
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
            // Note: We no longer track functions here. Instead, we scan for them later
            // by looking for functions containing luisa.ray.query.dispatch calls.
            // This handles dead code elimination safely.
            ReplaceIntrinsic(InitCall, CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_spawn);
            ReplaceIntrinsic(Call, CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_dispatch);

            // Remove the call to the extracted handler from the kernel
            // The handler should only be called by OptiX entry points, not the kernel
            // Find and remove calls to the newly extracted function from the kernel
            for (auto &BB : *F) {
                for (auto It = BB.begin(); It != BB.end();) {
                    auto *CI = llvm::dyn_cast<llvm::CallInst>(&*It++);
                    if (CI && CI->getCalledFunction() == NewF) {
                        // This is a call to the extracted handler from the kernel
                        // Remove it - the handler will be called by OptiX entry points instead
                        LUISA_INFO("Removing handler call from kernel: {}", NewF->getName().str());
                        CI->eraseFromParent();
                    }
                }
            }

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

        // Debug: Dump module after extraction
        {
            std::error_code ec;
            llvm::raw_fd_ostream dump_file("debug_after_extraction.ll", ec);
            if (!ec) {
                _llvm_module->print(dump_file, nullptr);
                dump_file.close();
                LUISA_INFO("Dumped module to debug_after_extraction.ll after extraction");
            }
        }

        _materialize_ray_query_loops();
        verify();

        // Run inline pass immediately to inline handlers before other optimizations
        // This prevents mergefunc and other passes from corrupting call targets
        _run_inline_pass_only();

        // Debug: Dump module after inline pass
        {
            std::error_code ec;
            llvm::raw_fd_ostream dump_file("debug_after_inline.ll", ec);
            if (!ec) {
                _llvm_module->print(dump_file, nullptr);
                dump_file.close();
                LUISA_INFO("Dumped module to debug_after_inline.ll after inline pass");
            }
        }

        // Clean up unused pseudo-intrinsics
        _remove_unused_pseudo_intrinsics();
    }
    _run_optimization_passes();

    // Debug: Dump module after optimization
    {
        std::error_code ec;
        llvm::raw_fd_ostream dump_file("debug_after_opt.ll", ec);
        if (!ec) {
            _llvm_module->print(dump_file, nullptr);
            dump_file.close();
            LUISA_INFO("Dumped module to debug_after_opt.ll after optimization");
        }
    }

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

namespace {
// Robustly detect ray query handler functions by checking for dispatch intrinsic or ray query blocks
// This avoids issues with dead code elimination making stored pointers dangling
[[nodiscard]] bool _is_ray_query_handler(llvm::Function *func) noexcept {
    if (func == nullptr || func->isDeclaration()) {
        return false;
    }
    // Check if function contains luisa.ray.query.dispatch call
    // OR if it has a block named "ray.query.dispatch" (for already-extracted handlers)
    for (auto &block : *func) {
        // Check for ray.query.dispatch block (indicates already-extracted handler)
        if (block.getName().contains("ray.query.dispatch")) {
            return true;
        }
        for (auto &inst : block) {
            if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                auto *callee = call->getCalledFunction();
                if (callee != nullptr) {
                    auto name = callee->getName();
                    if (name == llvm::StringRef(CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_dispatch)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Check if handler contains surface candidate hit (triangle) processing
// This is used to determine if the handler should be called from __intersection__
[[nodiscard]] bool _handler_has_surface_candidate_hit(llvm::Function *func) noexcept {
    if (func == nullptr || func->isDeclaration()) {
        return false;
    }
    for (auto &block : *func) {
        for (auto &inst : block) {
            if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                auto *callee = call->getCalledFunction();
                if (callee != nullptr) {
                    auto name = callee->getName();
                    if (name == llvm::StringRef(CUDACodegenLLVMImpl::llvm_ray_query_intrinsic_name_surface_candidate_hit)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
}// namespace

void CUDACodegenLLVMImpl::_lower_ray_query_spawn_calls() noexcept {
    // Find and lower all luisa.ray.query.spawn calls to OptiX trace calls
    auto spawn_func = _llvm_module->getFunction(llvm::StringRef(llvm_ray_query_intrinsic_name_spawn));
    if (!spawn_func) return;

    IB b{_llvm_context};

    // Collect all spawn calls first
    llvm::SmallVector<llvm::CallInst *, 16> spawn_calls;
    for (auto *user : spawn_func->users()) {
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(user)) {
            spawn_calls.push_back(call);
        }
    }

    // Process all spawn calls
    for (auto *call : spawn_calls) {

        b.SetInsertPoint(call);

        // Extract arguments from the spawn call
        // luisa.ray.query.spawn(accel, ray, time, mask, flags)
        auto accel = call->getArgOperand(0);
        auto ray = call->getArgOperand(1);
        auto time = call->getArgOperand(2);
        auto mask = call->getArgOperand(3);
        auto flags = call->getArgOperand(4);

        // For now, allocate a small context buffer on the stack
        // TODO: Implement proper context struct generation for captured variables
        // This allocates 16 bytes which should be enough for simple captured variables
        auto ctx_alloca = b.CreateAlloca(b.getInt8Ty(), b.getInt32(16), "ctx");
        b.CreateStore(llvm::Constant::getNullValue(llvm::ArrayType::get(b.getInt8Ty(), 16)), ctx_alloca);
        auto ctx_ptr = b.CreateBitCast(ctx_alloca, b.getPtrTy());

        // Encode query ID (0 for now) and context pointer in payload
        // r0 = (query_id << 24) | (ctx_ptr_high & 0xffffff)
        // r1 = ctx_ptr_low
        auto query_id = b.getInt32(0u);
        auto ctx_int = b.CreatePtrToInt(ctx_ptr, b.getInt64Ty());
        auto ctx_high = b.CreateLShr(ctx_int, 32);
        auto ctx_high_masked = b.CreateAnd(ctx_high, b.getInt64(0xffffff));
        auto query_id_shifted = b.CreateShl(b.CreateZExt(query_id, b.getInt64Ty()), 24);
        auto r0 = b.CreateOr(query_id_shifted, ctx_high_masked);
        auto r1 = b.CreateTrunc(ctx_int, b.getInt32Ty());

        // Convert r0 to i32
        auto r0_i32 = b.CreateTrunc(r0, b.getInt32Ty());

        // Use payload type for ray query (matching AST codegen)
        _call_optix_trace(b, CUDACodegenLLVMImpl::llvm_payload_type_ray_query, 0u, 0u, accel, ray, time, mask, {r0_i32, r1});

        // Remove the spawn call
        call->eraseFromParent();
    }

    // Remove the spawn function declaration
    spawn_func->eraseFromParent();
}

void CUDACodegenLLVMImpl::_materialize_ray_query_loops() noexcept {
    // Lower spawn calls first (before handler processing)
    _lower_ray_query_spawn_calls();

    // Scan module for ray query handler functions
    // We scan dynamically instead of using stored pointers to handle dead code elimination
    llvm::SmallVector<llvm::Function *, 8> ray_query_handlers;
    for (auto &func : *_llvm_module) {
        if (_is_ray_query_handler(&func)) {
            ray_query_handlers.push_back(&func);
            LUISA_INFO("Found ray query handler: {}", func.getName().str());
        }
    }

    if (ray_query_handlers.empty()) {
        LUISA_INFO("No ray query handlers found");
        return;
    }

    // Check which handlers handle surface candidates (triangles) BEFORE lowering
    // This must be done before _lower_ray_query_handler because intrinsics are replaced
    llvm::SmallVector<bool, 8> handler_has_surface_hit;
    for (auto *func : ray_query_handlers) {
        bool has_surface = _handler_has_surface_candidate_hit(func);
        handler_has_surface_hit.push_back(has_surface);
        if (has_surface) {
            LUISA_INFO("Handler {} handles surface candidates (triangles)", func->getName().str());
        } else {
            LUISA_INFO("Handler {} is procedural only", func->getName().str());
        }
    }

    // Filter handlers to only include those that return void
    // Handlers with non-void returns are not valid ray query handlers for OptiX
    llvm::SmallVector<llvm::Function *, 8> void_handlers;
    llvm::SmallVector<bool, 8> void_handler_has_surface_hit;
    for (size_t i = 0; i < ray_query_handlers.size(); ++i) {
        if (ray_query_handlers[i]->getReturnType()->isVoidTy()) {
            void_handlers.push_back(ray_query_handlers[i]);
            void_handler_has_surface_hit.push_back(handler_has_surface_hit[i]);
        } else {
            LUISA_INFO("Skipping handler {}: does not return void, returns {}",
                       ray_query_handlers[i]->getName().str(),
                       ray_query_handlers[i]->getReturnType()->isStructTy()  ? "struct" :
                       ray_query_handlers[i]->getReturnType()->isIntegerTy() ? "integer" :
                                                                               "other");
        }
    }
    ray_query_handlers = std::move(void_handlers);
    handler_has_surface_hit = std::move(void_handler_has_surface_hit);

    if (ray_query_handlers.empty()) {
        LUISA_INFO("No void-returning ray query handlers found");
        return;
    }

    // Assign unique query IDs to each function
    size_t query_id = 0;
    for (auto *func : ray_query_handlers) {
        // Rename to include query ID for clarity
        func->setName("ray.query.handler." + std::to_string(query_id));
        // Keep NoInline - we don't want the handler inlined back into the kernel
        // It will be manually inlined into the OptiX entry points only
        func->addFnAttr(llvm::Attribute::NoInline);
        query_id++;
    }

    // Lower intrinsics in each handler FIRST (transforms return type)
    for (auto *func : ray_query_handlers) {
        _lower_ray_query_handler(func);
    }

    // Generate OptiX entry points that dispatch to these handlers
    LUISA_INFO("Generating entry points for {} handlers", ray_query_handlers.size());
    for (size_t i = 0; i < ray_query_handlers.size(); ++i) {
        LUISA_INFO("Handler {}: name={}, ptr={}", i, ray_query_handlers[i]->getName().str(), (void *)ray_query_handlers[i]);
    }
    _generate_ray_query_entry_points(ray_query_handlers, handler_has_surface_hit);
}

void CUDACodegenLLVMImpl::_generate_ray_query_entry_points(llvm::ArrayRef<llvm::Function *> handlers,
                                                           llvm::ArrayRef<bool> handler_has_surface_hit) noexcept {
    if (handlers.empty()) return;

    // Generate __intersection__ray_query for procedural hits
    _generate_intersection_program(handlers, handler_has_surface_hit);

    // Generate __anyhit__ray_query for triangle hits
    _generate_anyhit_program(handlers);
}

void CUDACodegenLLVMImpl::_generate_intersection_program(llvm::ArrayRef<llvm::Function *> handlers,
                                                         llvm::ArrayRef<bool> handler_has_surface_hit) noexcept {
    IB b{_llvm_context};
    auto void_type = llvm::Type::getVoidTy(_llvm_context);
    auto func_type = llvm::FunctionType::get(void_type, {}, false);
    auto func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage,
                                       "__intersection__ray_query", _llvm_module.get());
    func->setCallingConv(llvm::CallingConv::PTX_Kernel);
    // Prevent optimization passes from merging this OptiX entry point with others
    func->addFnAttr(llvm::Attribute::NoDuplicate);
    func->addFnAttr(llvm::Attribute::NoInline);

    auto entry_block = llvm::BasicBlock::Create(_llvm_context, "entry", func);
    b.SetInsertPoint(entry_block);

    // Set payload types before accessing payloads
    _call_optix_set_payload_types(b, b.getInt32(CUDACodegenLLVMImpl::llvm_payload_type_ray_query));

    // Decode query ID and context pointer from payload registers
    // r0 = (query_id << 24) | (ctx_ptr_high & 0xffffff)
    // r1 = ctx_ptr_low
    auto r0 = _call_optix_get_payload(b, b.getInt32(0));
    auto r1 = _call_optix_get_payload(b, b.getInt32(1));
    auto query_id = b.CreateLShr(r0, 24);

    // Decode context pointer
    auto ctx_hi = b.CreateAnd(r0, b.getInt32(0xffffff));
    auto ctx_hi_64 = b.CreateZExt(ctx_hi, b.getInt64Ty());
    auto ctx_hi_shifted = b.CreateShl(ctx_hi_64, 32);
    auto ctx_lo_64 = b.CreateZExt(r1, b.getInt64Ty());
    auto ctx_int = b.CreateOr(ctx_hi_shifted, ctx_lo_64);
    auto ctx_ptr = b.CreateIntToPtr(ctx_int, b.getPtrTy(), "ctx.ptr");

    // Filter handlers to only include those that handle procedural hits
    // Surface candidate (triangle) handlers should NOT be called from intersection program
    // because they use optixGetTriangleBarycentrics which is invalid in __intersection__
    llvm::SmallVector<llvm::Function *, 8> procedural_handlers;
    for (size_t i = 0; i < handlers.size(); ++i) {
        // Use the pre-computed handler_has_surface_hit array (checked before intrinsics were lowered)
        if (!handler_has_surface_hit[i]) {
            procedural_handlers.push_back(handlers[i]);
            LUISA_INFO("Intersection program: Including procedural handler {}: {}",
                       procedural_handlers.size() - 1, handlers[i]->getName().str());
        } else {
            LUISA_INFO("Intersection program: Skipping surface handler {}: {} (contains surface candidate hit)",
                       i, handlers[i]->getName().str());
        }
    }

    // Create switch to dispatch to correct procedural handler
    if (!procedural_handlers.empty()) {
        // Create default block for switch (fallthrough case)
        auto default_block = llvm::BasicBlock::Create(_llvm_context, "default", func);
        auto switch_inst = b.CreateSwitch(query_id, default_block, procedural_handlers.size());

        for (size_t i = 0; i < procedural_handlers.size(); ++i) {
            auto handler_block = llvm::BasicBlock::Create(_llvm_context,
                                                          "handler_" + std::to_string(i), func);
            switch_inst->addCase(b.getInt32(i), handler_block);
            b.SetInsertPoint(handler_block);

            // Allocate result struct for this handler invocation
            // LCIntersectionResult: { float t_hit, i8 committed, i8 terminated }
            auto result_type = _get_llvm_intersection_result_type();
            auto result_alloca = b.CreateAlloca(result_type, nullptr, "result");
            b.CreateStore(llvm::Constant::getNullValue(result_type), result_alloca);

            // Call the handler - LLVM optimization will inline it automatically
            // Build argument list - pass null for buffer pointers, ctx_ptr for output params
            LUISA_INFO("Intersection program: Creating call to handler {}: name={}, ptr={}, num_args={}",
                       i, procedural_handlers[i]->getName().str(), (void *)procedural_handlers[i], procedural_handlers[i]->arg_size());

            llvm::SmallVector<llvm::Value *, 4> handler_args;
            for (auto &arg : procedural_handlers[i]->args()) {
                auto arg_type = arg.getType();
                if (arg_type->isPointerTy()) {
                    // Check if this is a context pointer (generic pointer) or buffer pointer (addrspace(1))
                    auto ptr_type = llvm::dyn_cast<llvm::PointerType>(arg_type);
                    if (ptr_type && ptr_type->getAddressSpace() == 0) {
                        // Generic pointer - assume this is the context pointer
                        handler_args.push_back(ctx_ptr);
                    } else {
                        // Buffer pointer or other - pass null for now
                        handler_args.push_back(llvm::ConstantPointerNull::get(ptr_type));
                    }
                } else {
                    // Non-pointer argument - pass 0
                    handler_args.push_back(llvm::Constant::getNullValue(arg_type));
                }
            }

            if (handler_args.empty()) {
                b.CreateCall(procedural_handlers[i]);
            } else {
                b.CreateCall(procedural_handlers[i], handler_args);
            }

            // Read result fields and conditionally report intersection
            // LCIntersectionResult: { float t_hit, i8 committed, i8 terminated }
            auto committed_ptr = b.CreateStructGEP(result_type, result_alloca, CUDACodegenLLVMImpl::llvm_intersection_result_committed_index, "committed.ptr");
            auto committed_val = b.CreateLoad(b.getInt8Ty(), committed_ptr, "committed");
            auto is_committed = b.CreateICmpNE(committed_val, b.getInt8(0), "is.committed");

            auto terminated_ptr = b.CreateStructGEP(result_type, result_alloca, CUDACodegenLLVMImpl::llvm_intersection_result_terminated_index, "terminated.ptr");
            auto terminated_val = b.CreateLoad(b.getInt8Ty(), terminated_ptr, "terminated");
            auto is_terminated = b.CreateICmpNE(terminated_val, b.getInt8(0), "is.terminated");

            // Create blocks for conditional report
            auto report_block = llvm::BasicBlock::Create(_llvm_context, "report", func);
            auto end_block = llvm::BasicBlock::Create(_llvm_context, "end", func);

            b.CreateCondBr(is_committed, report_block, end_block);

            // Report intersection block
            b.SetInsertPoint(report_block);

            // Read t_hit from result
            // LCIntersectionResult: { float t_hit, i8 committed, i8 terminated }
            auto t_hit_ptr = b.CreateStructGEP(result_type, result_alloca, CUDACodegenLLVMImpl::llvm_intersection_result_t_hit_index, "t_hit.ptr");
            auto t_hit_val = b.CreateLoad(b.getFloatTy(), t_hit_ptr, "t_hit");

            // Determine hit kind: PROCEDURAL_TERMINATED if terminated, else PROCEDURAL
            auto hit_kind_val = b.CreateSelect(is_terminated,
                                               b.getInt32(CUDACodegenLLVMImpl::llvm_hit_kind_procedural_terminated),
                                               b.getInt32(CUDACodegenLLVMImpl::llvm_hit_kind_procedural),
                                               "hit_kind");

            _call_optix_report_intersection(b, hit_kind_val, t_hit_val);
            b.CreateBr(end_block);

            // End block
            b.SetInsertPoint(end_block);
            b.CreateRetVoid();
        }

        // Set insert point to default block and return
        b.SetInsertPoint(default_block);
        b.CreateRetVoid();
    } else {
        // No handlers, just return
        b.CreateRetVoid();
    }
}

void CUDACodegenLLVMImpl::_generate_anyhit_program(llvm::ArrayRef<llvm::Function *> handlers) noexcept {
    IB b{_llvm_context};
    auto void_type = llvm::Type::getVoidTy(_llvm_context);
    auto func_type = llvm::FunctionType::get(void_type, {}, false);
    auto func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage,
                                       "__anyhit__ray_query", _llvm_module.get());
    func->setCallingConv(llvm::CallingConv::PTX_Kernel);
    // Prevent optimization passes from merging this OptiX entry point with others
    func->addFnAttr(llvm::Attribute::NoDuplicate);
    func->addFnAttr(llvm::Attribute::NoInline);

    auto entry_block = llvm::BasicBlock::Create(_llvm_context, "entry", func);
    b.SetInsertPoint(entry_block);

    // Set payload types before accessing payloads
    _call_optix_set_payload_types(b, b.getInt32(CUDACodegenLLVMImpl::llvm_payload_type_ray_query));

    // Decode query ID and context pointer from payload registers
    // r0 = (query_id << 24) | (ctx_ptr_high & 0xffffff)
    // r1 = ctx_ptr_low
    auto r0 = _call_optix_get_payload(b, b.getInt32(0));
    auto r1 = _call_optix_get_payload(b, b.getInt32(1));
    auto query_id = b.CreateLShr(r0, 24);

    // Decode context pointer
    auto ctx_hi = b.CreateAnd(r0, b.getInt32(0xffffff));
    auto ctx_hi_64 = b.CreateZExt(ctx_hi, b.getInt64Ty());
    auto ctx_hi_shifted = b.CreateShl(ctx_hi_64, 32);
    auto ctx_lo_64 = b.CreateZExt(r1, b.getInt64Ty());
    auto ctx_int = b.CreateOr(ctx_hi_shifted, ctx_lo_64);
    auto ctx_ptr = b.CreateIntToPtr(ctx_int, b.getPtrTy(), "ctx.ptr");

    // Create switch to dispatch to correct triangle handler
    if (!handlers.empty()) {
        // Create default block for switch (fallthrough case)
        auto default_block = llvm::BasicBlock::Create(_llvm_context, "default", func);
        auto switch_inst = b.CreateSwitch(query_id, default_block, handlers.size());

        for (size_t i = 0; i < handlers.size(); ++i) {
            auto handler_block = llvm::BasicBlock::Create(_llvm_context,
                                                          "handler_" + std::to_string(i), func);
            switch_inst->addCase(b.getInt32(i), handler_block);
            b.SetInsertPoint(handler_block);

            // Call the handler - LLVM optimization will inline it automatically
            // Build argument list - pass null for buffer pointers, ctx_ptr for output params
            LUISA_INFO("Anyhit program: Creating call to handler {}: name={}, ptr={}, num_args={}",
                       i, handlers[i]->getName().str(), (void *)handlers[i], handlers[i]->arg_size());

            llvm::SmallVector<llvm::Value *, 4> handler_args;
            for (auto &arg : handlers[i]->args()) {
                auto arg_type = arg.getType();
                if (arg_type->isPointerTy()) {
                    // Check if this is a context pointer (generic pointer) or buffer pointer (addrspace(1))
                    auto ptr_type = llvm::dyn_cast<llvm::PointerType>(arg_type);
                    if (ptr_type && ptr_type->getAddressSpace() == 0) {
                        // Generic pointer - assume this is the context pointer
                        handler_args.push_back(ctx_ptr);
                    } else {
                        // Buffer pointer or other - pass null for now
                        handler_args.push_back(llvm::ConstantPointerNull::get(ptr_type));
                    }
                } else {
                    // Non-pointer argument - pass 0
                    handler_args.push_back(llvm::Constant::getNullValue(arg_type));
                }
            }

            if (handler_args.empty()) {
                b.CreateCall(handlers[i]);
            } else {
                b.CreateCall(handlers[i], handler_args);
            }

            // TODO: Check committed flag and conditionally call _call_optix_ignore_intersection
            // For now, always accept

            b.CreateRetVoid();
        }

        // Set insert point to default block and return
        b.SetInsertPoint(default_block);
        b.CreateRetVoid();
    } else {
        // No handlers, just return
        b.CreateRetVoid();
    }
}

void CUDACodegenLLVMImpl::_lower_ray_query_handler(llvm::Function *handler) noexcept {
    IB b{_llvm_context};

    // Get LCIntersectionResult type: { i8 committed, i8 terminated }
    auto result_type = _get_llvm_intersection_result_type();
    LUISA_ASSERT(result_type != nullptr, "_get_llvm_intersection_result_type() returned null");
    auto struct_type = llvm::dyn_cast<llvm::StructType>(result_type);
    LUISA_ASSERT(struct_type != nullptr, "result_type is not a StructType");

    // Simpler approach: Add a pointer parameter to the existing handler
    // We'll pass a pointer to the result struct when calling the handler

    // Allocate result struct in the entry block
    auto entry_block = &handler->getEntryBlock();
    b.SetInsertPoint(&entry_block->front());
    auto result_alloca = b.CreateAlloca(result_type, nullptr, "result");
    b.CreateStore(llvm::Constant::getNullValue(result_type), result_alloca);

    // Debug: Print handler function before processing
    std::string func_str;
    llvm::raw_string_ostream rso(func_str);
    handler->print(rso);
    LUISA_INFO("Handler function before lowering:\n{}", rso.str());

    // Collect all instructions that need to be replaced
    llvm::SmallVector<llvm::CallInst *, 16> calls_to_replace;
    for (auto &block : *handler) {
        for (auto &inst : block) {
            if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                if (auto *callee = call->getCalledFunction()) {
                    calls_to_replace.push_back(call);
                }
            }
        }
    }

    for (auto *call : calls_to_replace) {
        auto *callee = call->getCalledFunction();
        if (!callee) continue;

        auto name = callee->getName();
        b.SetInsertPoint(call);

        if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_dispatch)) {
            // luisa.ray.query.dispatch() - this is a no-op in OptiX, just remove it
            call->eraseFromParent();
        } else if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_state)) {
            // luisa.ray.query.state() - get hit type from OptiX
            // For now, replace with constant based on handler type
            auto state_value = b.getInt8(llvm_ray_query_state_surface_candidate);
            call->replaceAllUsesWith(state_value);
            call->eraseFromParent();
        } else if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_world_space_ray)) {
            // luisa.ray.query.world.space.ray() - get world space ray from OptiX
            // Returns Ray { [3 x float] origin, float t_min, [3 x float] direction, float t_max }
            auto world_ray = _call_optix_get_world_space_ray(b);
            call->replaceAllUsesWith(world_ray);
            call->eraseFromParent();
        } else if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_surface_candidate_hit)) {
            // luisa.ray.query.surface.candidate.hit() - get triangle hit info from OptiX
            // Returns LCTriangleHit { i32 inst, i32 prim, <2 x float> bary, float t }
            auto inst_id = _call_optix_read_instance_index(b);
            auto prim_id = _call_optix_read_primitive_index(b);
            auto bary = _call_optix_get_triangle_barycentrics(b);
            auto t_hit = _call_optix_get_hit_distance(b);

            // Construct LCTriangleHit struct
            llvm::Value *result = llvm::PoisonValue::get(call->getType());
            result = b.CreateInsertValue(result, inst_id, 0u);
            result = b.CreateInsertValue(result, prim_id, 1u);
            result = b.CreateInsertValue(result, bary, 2u);
            result = b.CreateInsertValue(result, t_hit, 3u);

            call->replaceAllUsesWith(result);
            call->eraseFromParent();
        } else if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_procedural_candidate_hit)) {
            // luisa.ray.query.procedural.candidate.hit() - get procedural hit info from OptiX
            // Returns LCProceduralHit { i32 inst, i32 prim }
            LUISA_INFO("Procedural candidate hit: call type = {}", call->getType()->isStructTy() ? "struct" : "other");
            auto inst_id = _call_optix_read_instance_index(b);
            auto prim_id = _call_optix_read_primitive_index(b);

            // Construct LCProceduralHit struct
            auto result_type = call->getType();
            llvm::Value *result = llvm::PoisonValue::get(result_type);
            if (auto *struct_type = llvm::dyn_cast<llvm::StructType>(result_type)) {
                LUISA_INFO("Creating InsertValue for procedural hit, num elements = {}", struct_type->getNumElements());
            }
            result = b.CreateInsertValue(result, inst_id, static_cast<unsigned>(0));
            result = b.CreateInsertValue(result, prim_id, static_cast<unsigned>(1));

            call->replaceAllUsesWith(result);
            call->eraseFromParent();
        } else if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_commit_surface_hit) ||
                   name == llvm::StringRef(llvm_ray_query_intrinsic_name_commit_procedural_hit)) {
            // luisa.ray.query.commit.*() - set committed flag
            LUISA_INFO("Creating GEP for commit, struct_type={}, result_alloca={}",
                       (void *)struct_type, (void *)result_alloca);
            LUISA_ASSERT(struct_type != nullptr, "struct_type is null");
            LUISA_ASSERT(result_alloca != nullptr, "result_alloca is null");
            LUISA_ASSERT(llvm::isa<llvm::StructType>(struct_type), "struct_type is not a StructType");

            // Use CreateStructGEP to get pointer to committed field
            // LCIntersectionResult: { float t_hit, i8 committed, i8 terminated }
            auto committed_ptr = b.CreateStructGEP(
                struct_type, result_alloca, CUDACodegenLLVMImpl::llvm_intersection_result_committed_index, "committed.ptr");
            b.CreateStore(b.getInt8(1), committed_ptr);
            call->eraseFromParent();
        } else if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_terminate)) {
            // luisa.ray.query.terminate() - set terminated flag
            // Use CreateStructGEP to get pointer to terminated field
            // LCIntersectionResult: { float t_hit, i8 committed, i8 terminated }
            auto terminated_ptr = b.CreateStructGEP(
                struct_type, result_alloca, CUDACodegenLLVMImpl::llvm_intersection_result_terminated_index, "terminated.ptr");
            b.CreateStore(b.getInt8(1), terminated_ptr);
            call->eraseFromParent();
        } else if (name == llvm::StringRef(llvm_ray_query_intrinsic_name_committed_hit)) {
            // luisa.ray.query.committed.hit() - get committed hit data from OptiX
            // For now, replace with undef values
            auto undef_val = llvm::UndefValue::get(call->getType());
            call->replaceAllUsesWith(undef_val);
            call->eraseFromParent();
        } else if (name.starts_with("llvm.vector.reduce.fadd")) {
            // Vector horizontal sum reduction
            // llvm.vector.reduce.fadd.vNf32(start_value, vector) -> float
            auto *operand = call->getArgOperand(1);
            auto *vec_type = llvm::dyn_cast<llvm::FixedVectorType>(operand->getType());
            if (vec_type) {
                unsigned num_elems = vec_type->getNumElements();
                llvm::Value *result = nullptr;
                // Extract elements and sum them
                for (unsigned i = 0; i < num_elems; ++i) {
                    auto *elem = b.CreateExtractElement(operand, b.getInt32(i));
                    if (i == 0) {
                        result = elem;
                    } else {
                        result = b.CreateFAdd(result, elem);
                    }
                }
                if (result) {
                    call->replaceAllUsesWith(result);
                }
            }
            call->eraseFromParent();
        } else if (name.starts_with("llvm.nvvm.sqrt.approx")) {
            // Fast approximate square root
            // llvm.nvvm.sqrt.approx.ftz.f(float) -> float
            auto *operand = call->getArgOperand(0);
            auto *result = b.CreateCall(
                llvm::Intrinsic::getDeclaration(handler->getParent(), llvm::Intrinsic::sqrt,
                                                {operand->getType()}),
                {operand});
            call->replaceAllUsesWith(result);
            call->eraseFromParent();
        } else if (name.starts_with("llvm.nvvm.rsqrt.approx")) {
            // Fast approximate reciprocal square root
            // llvm.nvvm.rsqrt.approx.ftz.f(float) -> float
            auto *operand = call->getArgOperand(0);
            // rsqrt(x) = 1.0 / sqrt(x)
            auto *sqrt_val = b.CreateCall(
                llvm::Intrinsic::getDeclaration(handler->getParent(), llvm::Intrinsic::sqrt,
                                                {operand->getType()}),
                {operand});
            auto *one = llvm::ConstantFP::get(operand->getType(), 1.0);
            auto *result = b.CreateFDiv(one, sqrt_val);
            call->replaceAllUsesWith(result);
            call->eraseFromParent();
        } else {
            // Log unhandled calls
            LUISA_INFO("Unhandled call in handler: {}", name.str());
        }
    }

    // Fix infinite loops: Change backedge branches to return
    // In OptiX, handlers should process one candidate and return
    for (auto &block : *handler) {
        if (block.getName().contains("backedge")) {
            // Find the terminator instruction
            if (auto *term = block.getTerminator()) {
                if (auto *br = llvm::dyn_cast<llvm::BranchInst>(term)) {
                    if (br->isUnconditional()) {
                        // Replace unconditional branch to loop header with return
                        b.SetInsertPoint(term);
                        b.CreateRetVoid();
                        term->eraseFromParent();
                        LUISA_INFO("Fixed infinite loop in block: {}", block.getName().str());
                    }
                }
            }
        }
    }

    // Debug: Print handler function after processing
    std::string func_str_after;
    llvm::raw_string_ostream rso_after(func_str_after);
    handler->print(rso_after);
    LUISA_INFO("Handler function after lowering:\n{}", rso_after.str());
}

}// namespace luisa::compute::cuda
