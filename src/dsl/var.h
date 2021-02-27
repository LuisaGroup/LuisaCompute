//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute::dsl {

template<typename T>
class VarBase {

private:
    Variable _variable;

public:
    VarBase(VarBase &&) noexcept = default;
    VarBase &operator=(VarBase &&) noexcept = default;
    [[nodiscard]] constexpr auto variable() const noexcept { return _variable; }
};

}
