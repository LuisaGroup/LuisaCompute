//
//
// Created by Mike Smith on 2021/2/23.
//

#include <ast/function.h>
#include <ast/function_builder.h>

namespace luisa::compute {

std::span<const Variable> Function::builtin_variables() const noexcept {
    return _builder->builtin_variables();
}

std::span<const Variable> Function::shared_variables() const noexcept {
    return _builder->shared_variables();
}

std::span<const Function::ConstantBinding> Function::constants() const noexcept {
    return _builder->constants();
}

std::span<const Function::BufferBinding> Function::captured_buffers() const noexcept {
    return _builder->captured_buffers();
}

std::span<const Function::TextureBinding> Function::captured_textures() const noexcept {
    return _builder->captured_textures();
}

std::span<const Variable> Function::arguments() const noexcept {
    return _builder->arguments();
}

Function::Tag Function::tag() const noexcept {
    return _builder->tag();
}

const ScopeStmt *Function::body() const noexcept {
    return _builder->body();
}

std::span<const Function> Function::custom_callables() const noexcept {
    return _builder->custom_callables();
}

std::span<const CallOp> Function::builtin_callables() const noexcept {
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

bool Function::raytracing() const noexcept {
    return _builder->raytracing();
}

std::span<const Function::TextureHeapBinding> Function::captured_texture_heaps() const noexcept {
    return _builder->captured_texture_heaps();
}

}// namespace luisa::compute
