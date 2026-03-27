#pragma once

#include <luisa/ast/function_builder.h>
#include <luisa/dsl/expr.h>
#include <luisa/core/basic_types.h>

namespace luisa::compute {

template<typename T>
class WorkGraphOutput {
public:
    explicit WorkGraphOutput(uint output_index) noexcept : _output_index(output_index) {}

    void output(Expr<T> data, Expr<bool> should_write) const noexcept {
        auto f = detail::FunctionBuilder::current();
        f->call(CallOp::WORK_GRAPH_OUTPUT, {
            f->literal(Type::of<uint>(), _output_index),
            f->literal(Type::of<uint>(), 0u),
            data.expression(),
            should_write.expression()
        });
    }

private:
    uint _output_index;
};

}; // namespace luisa::compute