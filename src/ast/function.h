#pragma once

#include <span>
#include <variant>

#include <core/basic_types.h>
#include <ast/variable.h>
#include <ast/op.h>
#include <ast/constant_data.h>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}

class ScopeStmt;
class Expression;
class ScopeStmt;

class Function {

public:
    struct Hash {
        [[nodiscard]] auto operator()(Function f) const noexcept { return f.hash(); }
    };

public:
    enum struct Tag : uint {
        KERNEL,
        CALLABLE,
        // TODO: Ray-tracing functions, e.g. custom intersectors...
    };

    struct BufferBinding {
        Variable variable;
        uint64_t handle;
        size_t offset_bytes;
    };

    struct TextureBinding {
        Variable variable;
        uint64_t handle;
        TextureBinding(Variable v, uint64_t handle) noexcept
            : variable{v}, handle{handle} {}
    };

    struct HeapBinding {
        Variable variable;
        uint64_t handle;
        HeapBinding(Variable v, uint64_t handle) noexcept
            : variable{v}, handle{handle} {}
    };

    struct AccelBinding {
        Variable variable;
        uint64_t handle;
        AccelBinding(Variable v, uint64_t handle) noexcept
            : variable{v}, handle{handle} {}
    };

    struct ConstantBinding {
        const Type *type{nullptr};
        ConstantData data;
    };

private:
    const detail::FunctionBuilder *_builder{nullptr};

private:
    friend class detail::FunctionBuilder;

public:
    Function() noexcept = default;
    Function(const detail::FunctionBuilder *builder) noexcept : _builder{builder} {}
    [[nodiscard]] std::span<const Variable> builtin_variables() const noexcept;
    [[nodiscard]] std::span<const Variable> shared_variables() const noexcept;
    [[nodiscard]] std::span<const Variable> local_variables() const noexcept;
    [[nodiscard]] std::span<const ConstantBinding> constants() const noexcept;
    [[nodiscard]] std::span<const BufferBinding> captured_buffers() const noexcept;
    [[nodiscard]] std::span<const TextureBinding> captured_textures() const noexcept;
    [[nodiscard]] std::span<const HeapBinding> captured_heaps() const noexcept;
    [[nodiscard]] std::span<const AccelBinding> captured_accels() const noexcept;
    [[nodiscard]] std::span<const Variable> arguments() const noexcept;
    [[nodiscard]] std::span<const Function> custom_callables() const noexcept;
    [[nodiscard]] CallOpSet builtin_callables() const noexcept;
    [[nodiscard]] uint3 block_size() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
    [[nodiscard]] Usage variable_usage(uint32_t uid) const noexcept;
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] bool raytracing() const noexcept;
    [[nodiscard]] auto builder() const noexcept { return _builder; }
    [[nodiscard]] auto operator==(Function rhs) const noexcept { return _builder == rhs._builder; }
    [[nodiscard]] explicit operator bool() const noexcept { return _builder != nullptr; }
};

}// namespace luisa::compute
