//
// Created by Mike Smith on 2021/3/5.
//

#pragma once

#include <dsl/var.h>

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
    LUISA_STRUCT_REFLECT(S, __VA_ARGS__)                                                                         \
    namespace luisa::compute::detail {                                                                           \
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
