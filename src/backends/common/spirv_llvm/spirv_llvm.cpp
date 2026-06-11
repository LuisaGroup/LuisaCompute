// lc-spirv-llvm — LLVM IR codegen for LuisaCompute
// All implementation is in llvm_codegen_utility.cpp, llvm_state_visitor.cpp, llvm_codegen_stack_data.cpp

#include "llvm_codegen_utility.h"
#include <luisa/core/logging.h>

// LLVM SPIRV target initialization
// These symbols come from the LLVM SPIRV target libraries (LLVMSPIRVCodeGen, etc.)
namespace {
struct SPIRVLLVMInit {
    SPIRVLLVMInit() {
        extern void LLVMInitializeSPIRVTarget();
        extern void LLVMInitializeSPIRVTargetInfo();
        extern void LLVMInitializeSPIRVTargetMC();
        extern void LLVMInitializeSPIRVAsmPrinter();
        LLVMInitializeSPIRVTarget();
        LLVMInitializeSPIRVTargetInfo();
        LLVMInitializeSPIRVTargetMC();
        LLVMInitializeSPIRVAsmPrinter();
    }
};
[[maybe_unused]] static SPIRVLLVMInit s_spirv_init;
} // namespace
