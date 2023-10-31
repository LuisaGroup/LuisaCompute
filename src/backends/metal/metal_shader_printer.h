#pragma once

#include <luisa/ast/type.h>

namespace luisa::compute::metal {

class MetalShaderPrinter {

public:
    explicit MetalShaderPrinter(luisa::span<const std::pair<luisa::string, const Type *>> print_formats) noexcept;
    ~MetalShaderPrinter() noexcept;
};

}// namespace luisa::compute::metal
