//
//
// Created by Mike Smith on 2021/2/23.
//

#include <ast/function.h>
#include <ast/function_builder.h>

namespace luisa::compute {

luisa::span<const Variable> Function::builtin_variables() const noexcept {
    return _builder->builtin_variables();
}

luisa::span<const Function::Constant> Function::constants() const noexcept {
    return _builder->constants();
}

luisa::span<const Variable> Function::arguments() const noexcept {
    return _builder->arguments();
}

Function::Tag Function::tag() const noexcept {
    return _builder->tag();
}

const ScopeStmt *Function::body() const noexcept {
    return _builder->body();
}

luisa::span<const luisa::shared_ptr<const detail::FunctionBuilder>> Function::custom_callables() const noexcept {
    return _builder->custom_callables();
}

CallOpSet Function::builtin_callables() const noexcept {
    return _builder->builtin_callables();
}

const Type *Function::return_type() const noexcept {
    return _builder->return_type();
}

Usage Function::variable_usage(uint32_t uid) const noexcept {
    return _builder->variable_usage(uid);
}

uint3 Function::block_size() const noexcept {
    return _builder->block_size();
}

uint64_t Function::hash() const noexcept {
    return _builder->hash();
}

bool Function::requires_atomic() const noexcept {
    return _builder->requires_atomic();
}

bool Function::requires_atomic_float() const noexcept {
    return _builder->requires_atomic_float();
}

bool Function::requires_raytracing() const noexcept {
    return _builder->requires_raytracing();
}

luisa::shared_ptr<const detail::FunctionBuilder> Function::shared_builder() const noexcept {
    return _builder->shared_from_this();
}

luisa::span<const Variable> Function::local_variables() const noexcept {
    return _builder->local_variables();
}

luisa::span<const Variable> Function::shared_variables() const noexcept {
    return _builder->shared_variables();
}

}// namespace luisa::compute
