#pragma once

#include <span>
#include <ast/variable.h>

namespace luisa::compute {

class FunctionBuilder;
class ScopeStmt;

class Function {

public:
    enum struct Tag {
        KERNEL,
        CALLABLE,
        // TODO: Ray-tracing functions...
    };

    struct BufferBinding {
        Variable variable;
        uint64_t handle;
        size_t offset_bytes;
    };

    struct TextureBinding {
        // TODO...
    };

    struct UniformBinding {
        Variable variable;
        const void *data;
    };

    struct ConstantData {
        Variable variable;
        const void *data;
    };

private:
    const FunctionBuilder &_builder;

public:
    Function(const FunctionBuilder &builder) noexcept : _builder{builder} {}
    [[nodiscard]] std::span<const Variable> builtin_variables() const noexcept;
    [[nodiscard]] std::span<const Variable> shared_variables() const noexcept;
    [[nodiscard]] std::span<const ConstantData> constant_variables() const noexcept;
    [[nodiscard]] std::span<const BufferBinding> captured_buffers() const noexcept;
    [[nodiscard]] std::span<const TextureBinding> captured_textures() const noexcept;
    [[nodiscard]] std::span<const UniformBinding> captured_uniforms() const noexcept;
    [[nodiscard]] std::span<const Variable> arguments() const noexcept;
    [[nodiscard]] std::span<const uint32_t> custom_callables() const noexcept;
    [[nodiscard]] std::span<const std::string_view> builtin_callables() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] uint32_t uid() const noexcept;
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    [[nodiscard]] static Function custom_callable(size_t uid) noexcept;
};

}// namespace luisa::compute
