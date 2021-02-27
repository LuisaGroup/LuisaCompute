//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute::dsl {

template<typename T>
class Var {

private:
    Variable _variable;

public:
    Var(Var &&) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    
};

}
