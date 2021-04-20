#pragma once

#include <span>
#include <variant>

#include <core/basic_types.h>
#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/constant_data.h>

namespace luisa::compute {

class FunctionBuilder;
class ScopeStmt;

class Function {

public:
    enum struct Tag : uint {
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
        Variable variable;
        uint64_t handle;
    };
    
    struct ConstantBinding {
        const Type *type;
        uint64_t hash;
    };

private:
    const FunctionBuilder *_builder{nullptr};
    
private:
    friend class FunctionBuilder;
    explicit Function(const FunctionBuilder *builder) noexcept : _builder{builder} {}

public:
    Function() noexcept = default;
    [[nodiscard]] std::span<const Variable> builtin_variables() const noexcept;
    [[nodiscard]] std::span<const Variable> shared_variables() const noexcept;
    [[nodiscard]] std::span<const ConstantBinding> constants() const noexcept;
    [[nodiscard]] std::span<const BufferBinding> captured_buffers() const noexcept;
    [[nodiscard]] std::span<const TextureBinding> captured_images() const noexcept;
    [[nodiscard]] std::span<const Variable> arguments() const noexcept;
    [[nodiscard]] std::span<const uint32_t> custom_callables() const noexcept;
    [[nodiscard]] std::span<const CallOp> builtin_callables() const noexcept;
    [[nodiscard]] uint3 block_size() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] uint32_t uid() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
    [[nodiscard]] Variable::Usage variable_usage(uint32_t uid) const noexcept;
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    [[nodiscard]] static Function at(uint32_t uid) noexcept;
    [[nodiscard]] static Function callable(uint32_t uid) noexcept;
    [[nodiscard]] static Function kernel(uint32_t uid) noexcept;
};

}// namespace luisa::compute
