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

class MetaStmt;
class Expression;

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
    explicit Function(const detail::FunctionBuilder *builder) noexcept : _builder{builder} {}
    [[nodiscard]] luisa::span<const Variable> builtin_variables() const noexcept;
    [[nodiscard]] luisa::span<const Constant> constants() const noexcept;
    [[nodiscard]] luisa::span<const Variable> arguments() const noexcept;
    [[nodiscard]] luisa::span<const luisa::shared_ptr<const detail::FunctionBuilder>> custom_callables() const noexcept;
    [[nodiscard]] CallOpSet builtin_callables() const noexcept;
    [[nodiscard]] uint3 block_size() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
    [[nodiscard]] Usage variable_usage(uint32_t uid) const noexcept;
    [[nodiscard]] const MetaStmt *body() const noexcept;
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] bool raytracing() const noexcept;
    [[nodiscard]] auto builder() const noexcept { return _builder; }
    [[nodiscard]] luisa::shared_ptr<const detail::FunctionBuilder> shared_builder() const noexcept;
    [[nodiscard]] auto operator==(Function rhs) const noexcept { return _builder == rhs._builder; }
    [[nodiscard]] explicit operator bool() const noexcept { return _builder != nullptr; }
};

}// namespace luisa::compute
