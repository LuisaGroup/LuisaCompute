//
// Created by Mike Smith on 2023/4/15.
//

#include <core/logging.h>
#include <backends/metal/metal_codegen_ast.h>

namespace luisa::compute::metal {

size_t MetalCodegenAST::type_size_bytes(const Type *type) noexcept {
    if (!type->is_custom()) { return type->size(); }
    LUISA_ERROR_WITH_LOCATION("Cannot get size of custom type.");
}

}// namespace luisa::compute::metal
