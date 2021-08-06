//
// Created by Mike Smith on 2021/3/5.
//

#pragma once

#include <dsl/var.h>

namespace luisa::compute::detail {

template<typename T>
struct c_array_to_std_array {
    using type = T;
};

template<typename T, size_t N>
struct c_array_to_std_array<T[N]> {
    using type = std::array<T, N>;
};

template<typename T>
using c_array_to_std_array_t = typename c_array_to_std_array<T>::type;

};// namespace luisa::compute::detail

#define LUISA_STRUCT_MAKE_MEMBER_EXPR(m)                         \
private:                                                         \
    using Type_##m = detail::c_array_to_std_array_t<             \
        std::remove_cvref_t<                                     \
            decltype(std::declval<This>().m)>>;                  \
                                                                 \
public:                                                          \
    Expr<Type_##m> m{detail::FunctionBuilder::current()->member( \
        Type::of<Type_##m>(),                                    \
        this->_expression,                                       \
        _member_index(#m))};

#define LUISA_STRUCT(S, ...)                                                                                     \
    LUISA_STRUCT_REFLECT(S, __VA_ARGS__)                                                                         \
    namespace luisa::compute {                                                                                   \
    template<>                                                                                                   \
    struct Expr<S> {                                                                                             \
    private:                                                                                                     \
        const Expression *_expression;                                                                           \
        using This = S;                                                                                          \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept {                    \
            constexpr const std::string_view member_names[]{LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};       \
            return std::find(std::begin(member_names), std::end(member_names), name) - std::begin(member_names); \
        }                                                                                                        \
                                                                                                                 \
    public:                                                                                                      \
        explicit Expr(const Expression *e) noexcept : _expression{e} {}                                          \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }                             \
        Expr(Expr &&another) noexcept = default;                                                                 \
        Expr(const Expr &another) noexcept = default;                                                            \
        void operator=(const Expr &rhs) noexcept {                                                               \
            detail::FunctionBuilder::current()->assign(                                                          \
                AssignOp::ASSIGN,                                                                                \
                this->expression(),                                                                              \
                rhs.expression());                                                                               \
        }                                                                                                        \
        void operator=(Expr &&rhs) noexcept {                                                                    \
            detail::FunctionBuilder::current()->assign(                                                          \
                AssignOp::ASSIGN,                                                                                \
                this->expression(),                                                                              \
                rhs.expression());                                                                               \
        }                                                                                                        \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_EXPR, __VA_ARGS__)                                                    \
    };                                                                                                           \
    }
