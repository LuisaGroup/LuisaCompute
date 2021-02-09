//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <variant>
#include <core/type_info.h>

namespace luisa::compute {

class Expression;

class Variable {

private:
    const TypeInfo *_type;

public:
    [[nodiscard]] auto type() const noexcept { return _type; }
};

}// namespace luisa::compute::ast
