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

const MetaStmt *Function::body() const noexcept {
    return _builder->body();
}

std::span<const luisa::shared_ptr<const detail::FunctionBuilder>> Function::custom_callables() const noexcept {
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

bool Function::raytracing() const noexcept {
    return _builder->raytracing();
}

std::span<const Function::HeapBinding> Function::captured_heaps() const noexcept {
    return _builder->captured_heaps();
}

std::span<const Function::AccelBinding> Function::captured_accels() const noexcept {
    return _builder->captured_accels();
}

luisa::shared_ptr<const detail::FunctionBuilder> Function::shared_builder() const noexcept {
    return _builder->shared_from_this();
}

}// namespace luisa::compute
