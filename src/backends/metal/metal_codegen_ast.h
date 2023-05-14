//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <ast/function.h>
#include <backends/common/string_scratch.h>

namespace luisa::compute::metal {

class MetalCodegenAST {

private:
    StringScratch &_scratch;

public:
    explicit MetalCodegenAST(StringScratch &scratch) noexcept;
    void emit(Function kernel) noexcept;
    [[nodiscard]] static size_t type_size_bytes(const Type *type) noexcept;
};

}// namespace luisa::compute::metal
