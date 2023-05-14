//
// Created by Mike Smith on 2023/4/15.
//

#include <core/logging.h>
#include <backends/metal/metal_builtin_embedded.h>
#include <backends/metal/metal_codegen_ast.h>

namespace luisa::compute::metal {

MetalCodegenAST::MetalCodegenAST(StringScratch &scratch) noexcept
    : _scratch{scratch} {}

size_t MetalCodegenAST::type_size_bytes(const Type *type) noexcept {
    if (!type->is_custom()) { return type->size(); }
    LUISA_ERROR_WITH_LOCATION("Cannot get size of custom type.");
}

void MetalCodegenAST::emit(Function kernel) noexcept {

    _scratch << luisa::string_view{luisa_metal_builtin_metal_device_lib,
                                   sizeof(luisa_metal_builtin_metal_device_lib)}
             << "\n"
             << "// block_size = ("
             << kernel.block_size().x << ", "
             << kernel.block_size().y << ", "
             << kernel.block_size().z << ")\n\n";

    _scratch << "[[kernel]] void kernel_main() {}\n";

}

}// namespace luisa::compute::metal
