//
// Created by Mike Smith on 2021/2/23.
//

#include <ast/function.h>
#include <ast/function_builder.h>

namespace luisa::compute {

std::span<const Variable> Function::builtin_variables() const noexcept {
    return _builder.builtin_variables();
}

std::span<const Variable> Function::shared_variables() const noexcept {
    return _builder.shared_variables();
}

std::span<const Function::ConstantData> Function::constant_variables() const noexcept {
    return _builder.constant_variables();
}

std::span<const Function::BufferBinding> Function::captured_buffers() const noexcept {
    return _builder.captured_buffers();
}

std::span<const Function::TextureBinding> Function::captured_textures() const noexcept {
    return _builder.captured_textures();
}

std::span<const Function::UniformBinding> Function::captured_uniforms() const noexcept {
    return _builder.captured_uniforms();
}

std::span<const Variable> Function::arguments() const noexcept {
    return _builder.arguments();
}

Function::Tag Function::tag() const noexcept {
    return _builder.tag();
}

const ScopeStmt *Function::body() const noexcept {
    return _builder.body();
}

}
