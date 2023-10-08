#pragma once

#include <luisa/ast/variable.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/function.h>
#include <luisa/core/stl/unordered_map.h>

#include <luisa/rust/ir.hpp>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}// namespace detail

class LC_IR_API AST2IR {
    
public:
    [[nodiscard]] static luisa::shared_ptr<ir::CArc<ir::KernelModule>> build_kernel(Function function) noexcept;
    [[nodiscard]] static luisa::shared_ptr<ir::CArc<ir::CallableModule>> build_callable(Function function) noexcept;
    [[nodiscard]] static ir::CArc<ir::Type> build_type(const Type *type) noexcept;
};

}// namespace luisa::compute
