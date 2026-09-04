#include "hip_inlining_policy.h"

#include "hip_callable_abi.h"

#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/ModuleInliner.h>

namespace luisa::compute::hip {

namespace {

[[nodiscard]] llvm::CallBase *unique_direct_call_use(
    llvm::Function &function) noexcept {
    if (!function.hasOneUse()) { return nullptr; }
    auto *call = llvm::dyn_cast<llvm::CallBase>(*function.user_begin());
    return call != nullptr &&
                   call->getCalledOperand()->stripPointerCasts() == &function ?
               call :
               nullptr;
}

[[nodiscard]] bool is_exact_argument_forwarder(
    const llvm::Function &wrapper,
    const llvm::CallBase &call) noexcept {
    if (!wrapper.getReturnType()->isVoidTy() ||
        !call.getType()->isVoidTy() ||
        call.arg_size() != wrapper.arg_size()) {
        return false;
    }
    for (auto index = 0u; index < call.arg_size(); ++index) {
        if (call.getArgOperand(index) != wrapper.getArg(index)) {
            return false;
        }
    }
    for (const auto &block : wrapper) {
        for (const auto &instruction : block) {
            if (&instruction == &call ||
                llvm::isa<llvm::BranchInst>(instruction) ||
                llvm::isa<llvm::ReturnInst>(instruction)) {
                continue;
            }
            return false;
        }
    }
    return true;
}

}// namespace

void configure_hip_cgscc_canonicalization_inlining(
    llvm::PipelineTuningOptions &options) noexcept {
    // LLVM accepts a modeled cost of zero because the final comparison is
    // cost < max(1, threshold). Thus threshold zero denotes the complete
    // non-positive-cost fixed point, rather than disabling inlining.
    options.InlinerThreshold = 0;
}

llvm::InlineParams hip_module_priority_inline_params(
    llvm::OptimizationLevel level) noexcept {
#if LLVM_VERSION_MAJOR >= 23
    auto params = llvm::getInlineParamsFromOptLevel(
        level.getSpeedupLevel());
#else
    auto params = llvm::getInlineParams(
        level.getSpeedupLevel(), level.getSizeLevel());
#endif
    // Deferral compensates for bottom-up SCC visitation. The module inliner
    // uses a global priority queue, so deferral has no semantic role and would
    // only make the selected edge set depend on a policy for the wrong order.
    params.EnableDeferral = false;
    return params;
}

void add_hip_module_priority_inliner(
    llvm::ModulePassManager &pipeline,
    llvm::OptimizationLevel level) noexcept {
    pipeline.addPass(llvm::ModuleInlinerPass{
        hip_module_priority_inline_params(level),
        llvm::InliningAdvisorMode::Default,
        llvm::ThinOrFullLTOPhase::None});
}

size_t mark_hip_single_use_forwarded_kernel_callables_for_inlining(
    llvm::Module &module) noexcept {
    llvm::SmallVector<llvm::Function *, 8> selected;
    for (auto &body : module) {
        if (body.isDeclaration() ||
            !body.getReturnType()->isVoidTy() ||
            !body.hasFnAttribute(llvm_generated_callable_attribute)) {
            continue;
        }
        auto *body_call = unique_direct_call_use(body);
        if (body_call == nullptr) { continue; }
        auto *wrapper = body_call->getFunction();
        if (wrapper == nullptr || wrapper == &body ||
            wrapper->isDeclaration() ||
            !wrapper->hasFnAttribute(llvm_generated_callable_attribute) ||
            !is_exact_argument_forwarder(*wrapper, *body_call)) {
            continue;
        }
        auto *wrapper_call = unique_direct_call_use(*wrapper);
        if (wrapper_call == nullptr) { continue; }
        auto *kernel = wrapper_call->getFunction();
        if (kernel == nullptr ||
            kernel->getCallingConv() !=
                llvm::CallingConv::AMDGPU_KERNEL) {
            continue;
        }
        selected.emplace_back(&body);
    }
    for (auto *body : selected) {
        body->removeFnAttr(llvm::Attribute::NoInline);
        body->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    return selected.size();
}

}// namespace luisa::compute::hip
