//
// Created by Mike Smith on 2021/2/27.
//

#pragma once

#include <dsl/var.h>
#include <dsl/expr.h>
#include <dsl/buffer.h>
#include <dsl/func.h>
#include <dsl/constant.h>
#include <dsl/shared.h>

// for custom structs
#undef LUISA_STRUCT// to extend it...

#define LUISA_STRUCT_MAKE_MEMBER_EXPR(m)                                    \
private:                                                                    \
    using Type_##m = std::remove_cvref_t<decltype(std::declval<This>().m)>; \
                                                                            \
public:                                                                     \
    Expr<Type_##m> m{FunctionBuilder::current()->member(                    \
        Type::of<Type_##m>(),                                               \
        ExprBase<This>::_expression,                                        \
        _member_index(#m))};

#define LUISA_STRUCT(S, ...)                                                                                     \
    LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, __VA_ARGS__)                                                \
    namespace luisa::compute::dsl::detail {                                                                      \
    template<>                                                                                                   \
    struct Expr<S> : public ExprBase<S> {                                                                        \
    private:                                                                                                     \
        using This = S;                                                                                          \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept {                    \
            constexpr const std::string_view member_names[]{LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};       \
            return std::find(std::begin(member_names), std::end(member_names), name) - std::begin(member_names); \
        }                                                                                                        \
                                                                                                                 \
    public:                                                                                                      \
        using ExprBase<S>::ExprBase;                                                                             \
        Expr(Expr &&another) noexcept = default;                                                                 \
        Expr(const Expr &another) noexcept = default;                                                            \
        void operator=(Expr &&rhs) noexcept { ExprBase<S>::operator=(rhs); }                                     \
        void operator=(const Expr &rhs) noexcept { ExprBase<S>::operator=(rhs); }                                \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_EXPR, __VA_ARGS__)                                                    \
    };                                                                                                           \
    }

namespace luisa::compute::dsl::detail {

struct KernelBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Kernel{std::forward<F>(def)}; }
};

struct CallableBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Callable{std::forward<F>(def)}; }
};

}// namespace luisa::compute::dsl::detail

#define LUISA_KERNEL ::luisa::compute::dsl::detail::KernelBuilder{} % [&]
#define LUISA_CALLABLE ::luisa::compute::dsl::detail::CallableBuilder{} % [&]

namespace luisa::compute::dsl {

// TODO: disable operators...
template<typename T>
[[nodiscard]] inline auto shared(size_t n) noexcept {
    return detail::Expr<T[1]>{FunctionBuilder::current()->shared(
        fmt::format("array<{},{}>", Type::of<T>()->description(), n))};
}

template<typename T>
[[nodiscard]] inline auto constant(std::span<T> data) noexcept {
    return detail::Expr<T[1]>{FunctionBuilder::current()->constant(data)};
}

template<typename T>
[[nodiscard]] inline auto constant(std::initializer_list<T> data) noexcept {
    return detail::Expr<T[1]>{FunctionBuilder::current()->constant(data)};
}

}
