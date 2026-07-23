// lc-spirv-llvm — LLVM IR codegen for LuisaCompute
// All implementation is in llvm_codegen_utility.cpp, llvm_state_visitor.cpp, llvm_codegen_stack_data.cpp

#include "spirv_llvm.h"
#include "llvm_codegen_utility.h"
#include <luisa/core/logging.h>
#include <llvm/Support/TargetSelect.h>
#include <mutex>

namespace lc::llvm_codegen {

void InitializeLLVMSPIRVTarget() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        // Target registration mutates LLVM's global registries. Register in
        // the dependency order used by TargetSelect and do it once even when
        // shaders are compiled concurrently.
        LLVMInitializeSPIRVTargetInfo();
        LLVMInitializeSPIRVTarget();
        LLVMInitializeSPIRVTargetMC();
        LLVMInitializeSPIRVAsmPrinter();
    });
}

LLVMCodegenResult compile_spirv(
    luisa::compute::Function kernel,
    const luisa::compute::ShaderOption &option) {
    return LLVMCodegenUtility::CompileSPIRV(kernel, option);
}

} // namespace lc::llvm_codegen
