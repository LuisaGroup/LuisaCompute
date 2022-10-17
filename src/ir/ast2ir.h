//
// Created by Mike Smith on 2022/10/17.
//

#include <bindings.hpp>

#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <ast/function.h>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}

class AST2IR {

private:
    luisa::unordered_map<uint64_t, ir::NodeRef> _constant_stack;
    luisa::unordered_map<uint64_t, ir::NodeRef> _variable_stack;
    luisa::vector<ir::IrBuilder> _builders;

public:
    AST2IR() noexcept = default;
    [[nodiscard]] ir::Module convert(Function function) const noexcept;
};

}// namespace luisa::compute
