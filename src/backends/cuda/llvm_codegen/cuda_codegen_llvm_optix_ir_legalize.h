#pragma once

namespace llvm {
class Module;
}// namespace llvm

namespace luisa::compute::cuda {

// Run after LLVM optimization and before LLVM 7 typed-pointer serialization.
// The NVIDIA reader accepts a smaller intrinsic set than LLVM's PTX backend.
void luisa_compute_cuda_llvm_legalize_optix_ir(llvm::Module &module) noexcept;

}// namespace luisa::compute::cuda
