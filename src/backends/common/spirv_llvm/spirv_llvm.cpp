// lc-spirv-llvm — LLVM IR codegen for LuisaCompute
// All implementation is in llvm_codegen_utility.cpp, llvm_state_visitor.cpp, llvm_codegen_stack_data.cpp

#include "llvm_codegen_utility.h"
#include <luisa/core/logging.h>

// LLVM SPIRV target initialization symbols (global namespace, from LLVMSPIRVCodeGen etc.)
extern void LLVMInitializeSPIRVTarget();
extern void LLVMInitializeSPIRVTargetInfo();
extern void LLVMInitializeSPIRVTargetMC();
extern void LLVMInitializeSPIRVAsmPrinter();

namespace lc::llvm_codegen {

void InitializeLLVMSPIRVTarget() {
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
}

} // namespace lc::llvm_codegen
