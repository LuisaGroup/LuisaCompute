//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <ast/function.h>

namespace luisa::compute::metal {

class MetalCodegenAST {

public:
    [[nodiscard]] static size_t type_size_bytes(const Type *type) noexcept;
};

}// namespace luisa::compute::metal
