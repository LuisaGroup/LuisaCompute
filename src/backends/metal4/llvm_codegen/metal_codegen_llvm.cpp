#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal {

MetalCodegenLLVMResult::MetalCodegenLLVMResult() noexcept = default;
MetalCodegenLLVMResult::MetalCodegenLLVMResult(MetalCodegenLLVMResult &&) noexcept = default;
MetalCodegenLLVMResult &MetalCodegenLLVMResult::operator=(MetalCodegenLLVMResult &&) noexcept = default;
MetalCodegenLLVMResult::~MetalCodegenLLVMResult() noexcept = default;

luisa::string MetalCodegenLLVMResult::ir() const noexcept {
    LUISA_ASSERT(module != nullptr, "Cannot print an empty Metal LLVM codegen result.");
    std::string text;
    llvm::raw_string_ostream stream{text};
    module->print(stream, nullptr, false, true);
    stream.flush();
    return {text.data(), text.size()};
}

luisa::vector<std::byte> MetalCodegenLLVMResult::bitcode() const noexcept {
    LUISA_ASSERT(module != nullptr, "Cannot serialize an empty Metal LLVM codegen result.");
    llvm::SmallVector<char, 0> storage;
    llvm::raw_svector_ostream stream{storage};
    llvm::WriteBitcodeToFile(*module, stream);
    luisa::vector<std::byte> result;
    result.resize(storage.size());
    std::memcpy(result.data(), storage.data(), storage.size());
    return result;
}

MetalCodegenLLVMResult luisa_compute_metal_codegen_llvm(
    const xir::Module &xir_module,
    const MetalCodegenLLVMConfig &config) noexcept {
    detail::MetalCodegenLLVMImpl implementation{config};
    return implementation.generate(xir_module);
}

}// namespace luisa::compute::metal
