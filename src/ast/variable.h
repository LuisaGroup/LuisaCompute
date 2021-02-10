//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <core/type_info.h>
#include <core/union.h>

namespace luisa::compute {

class Expression;

class Variable {
    
    using Binding = Union<const Expression *>;

private:
    const TypeInfo *_type;
    Binding _binding;

public:
    [[nodiscard]] auto type() const noexcept { return _type; }
};

}// namespace luisa::compute::ast
