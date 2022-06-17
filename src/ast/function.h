#pragma once

#include <span>
#include <variant>
#include <numeric>

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

/**
 * @brief Function class
 * 
 */
class LC_AST_API Function {

public:
    struct Hash {
        [[nodiscard]] auto operator()(Function f) const noexcept { return f.hash(); }
    };

public:
    /// Function types
    enum struct Tag : uint {
        KERNEL,
        CALLABLE,
        // TODO: Ray-tracing functions, e.g. custom intersectors...
    };

    struct Constant {
        const Type *type{nullptr};
        ConstantData data;
        [[nodiscard]] auto hash() const noexcept {
            using namespace std::string_view_literals;
            return hash64(data.hash(), hash64(type->hash(), hash64("__hash_constant_binding")));
        }
    };

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
    /// Return custom callables
    [[nodiscard]] luisa::span<const luisa::shared_ptr<const detail::FunctionBuilder>> custom_callables() const noexcept;
    /// Return builtin callables
    [[nodiscard]] CallOpSet builtin_callables() const noexcept;
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
    [[nodiscard]] uint64_t hash() const noexcept;
    /// Return if is ray tracing function
    [[nodiscard]] bool raytracing() const noexcept;
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
