#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/ast/op.h>
#include <luisa/ast/variable.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/constant_data.h>
#include <luisa/runtime/rhi/argument.h>

namespace luisa::compute {

class CurveBasisSet;

namespace detail {
class FunctionBuilder;
}

class ScopeStmt;
class Expression;

class ExternalFunction;

/**
 * @brief Function class
 * 
 */
class LC_AST_API Function {

public:
    /// Function types
    enum struct Tag : uint {
        KERNEL,
        CALLABLE,
        RASTER_STAGE
    };

    using Constant = ConstantData;

    /**
     * @brief %Buffer binding.
     *
     * Bind buffer handle and offset.
     */
    struct BufferBinding : public Argument::Buffer {
        BufferBinding() noexcept = default;
        explicit BufferBinding(uint64_t handle, size_t offset_bytes, size_t size_bytes) noexcept
            : Argument::Buffer{.handle = handle,
                               .offset = offset_bytes,
                               .size = size_bytes} {}
        [[nodiscard]] uint64_t hash() const noexcept;
    };

    /**
     * @brief Texture binding.
     *
     * Bind texture handle and level.
     */
    struct TextureBinding : public Argument::Texture {
        TextureBinding() noexcept = default;
        explicit TextureBinding(uint64_t handle, uint32_t level) noexcept
            : Argument::Texture{.handle = handle,
                                .level = level} {}
        [[nodiscard]] uint64_t hash() const noexcept;
    };

    /**
     * @brief Bindless array binding.
     *
     * Bind array handle.
     */
    struct BindlessArrayBinding : public Argument::BindlessArray {
        BindlessArrayBinding() noexcept = default;
        explicit BindlessArrayBinding(uint64_t handle) noexcept
            : Argument::BindlessArray{.handle = handle} {}
        [[nodiscard]] uint64_t hash() const noexcept;
    };

    /**
     * @brief Acceleration structure binding.
     *
     * Bind accel handle.
     */
    struct AccelBinding : public Argument::Accel {
        AccelBinding() noexcept = default;
        explicit AccelBinding(uint64_t handle) noexcept
            : Argument::Accel{.handle = handle} {}
        [[nodiscard]] uint64_t hash() const noexcept;
    };
    using Binding = luisa::variant<
        luisa::monostate,// not bound
        BufferBinding,
        TextureBinding,
        BindlessArrayBinding,
        AccelBinding>;

private:
    const detail::FunctionBuilder *_builder{nullptr};

private:
    friend class detail::FunctionBuilder;

public:
    Function() noexcept = default;
    /// Construct function object from FunctionBuilder
    explicit Function(const detail::FunctionBuilder *builder) noexcept : _builder{builder} {}
    /// Return builtin variables
    [[nodiscard]] luisa::span<const Variable> builtin_variables() const noexcept;
    /// Return local variables
    [[nodiscard]] luisa::span<const Variable> local_variables() const noexcept;
    /// Return shared variables
    [[nodiscard]] luisa::span<const Variable> shared_variables() const noexcept;
    /// Return constants
    [[nodiscard]] luisa::span<const Constant> constants() const noexcept;
    /// Return arguments
    [[nodiscard]] luisa::span<const Variable> arguments() const noexcept;
    /// Return bound arguments
    [[nodiscard]] luisa::span<const Binding> bound_arguments() const noexcept;
    /// Return unbound arguments
    [[nodiscard]] luisa::span<const Variable> unbound_arguments() const noexcept;
    /// Return custom callables
    [[nodiscard]] luisa::span<const luisa::shared_ptr<const detail::FunctionBuilder>> custom_callables() const noexcept;
    /// Return external callables
    [[nodiscard]] luisa::span<const luisa::shared_ptr<const ExternalFunction>> external_callables() const noexcept;
    /// Return builtin callables that are *directly* used by this function
    [[nodiscard]] CallOpSet direct_builtin_callables() const noexcept;
    /// Return builtin callables that are used by this function and the functions it calls
    [[nodiscard]] CallOpSet propagated_builtin_callables() const noexcept;
    /// Return block size
    [[nodiscard]] uint3 block_size() const noexcept;
    /// Return function tag
    [[nodiscard]] Tag tag() const noexcept;
    /// Return return type
    [[nodiscard]] const Type *return_type() const noexcept;
    /// Return variable usage of given uid
    [[nodiscard]] Usage variable_usage(uint32_t uid) const noexcept;
    /// Return pointer to body statement
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    /// Return hash
    [[nodiscard]] bool hash_computed() const noexcept;
    [[nodiscard]] uint64_t hash() const noexcept;
    /// Return if is ray tracing function
    [[nodiscard]] bool requires_raytracing() const noexcept;
    /// Return if requires motion blur
    [[nodiscard]] bool requires_motion_blur() const noexcept;
    /// Return whether the function requires atomic operations
    [[nodiscard]] bool requires_atomic() const noexcept;
    /// Return whether the function requires atomic float operations
    [[nodiscard]] bool requires_atomic_float() const noexcept;
    /// Return whether the function requires automatic differentiation
    [[nodiscard]] bool requires_autodiff() const noexcept;
    /// Return whether the function requires printing
    [[nodiscard]] bool requires_printing() const noexcept;
    /// Return required curve bases
    [[nodiscard]] CurveBasisSet required_curve_bases() const noexcept;
    /// Return function builder
    [[nodiscard]] auto builder() const noexcept { return _builder; }
    /// Return shared pointer to function builder
    [[nodiscard]] luisa::shared_ptr<const detail::FunctionBuilder> shared_builder() const noexcept;
    /// Equal operation reload
    [[nodiscard]] auto operator==(Function rhs) const noexcept { return _builder == rhs._builder; }
    /// Cast to bool, true if builder is not nullptr
    [[nodiscard]] explicit operator bool() const noexcept { return _builder != nullptr; }
};

}// namespace luisa::compute

