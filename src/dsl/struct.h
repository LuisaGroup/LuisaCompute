//
// Created by Mike Smith on 2021/3/5.
//

#pragma once

#include <cstdint>
#include <cstddef>

#include <dsl/var.h>
#include <dsl/func.h>
#include <runtime/shader.h>

namespace luisa::compute::detail {

template<typename T>
struct dsl_struct_extension {
    static_assert(luisa::always_false_v<T>);
};

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

#define LUISA_STRUCT_MAKE_MEMBER_TYPE(m)                                    \
    using member_type_##m = luisa::compute::detail::c_array_to_std_array_t< \
        std::remove_cvref_t<                                                \
            decltype(std::declval<this_type>().m)>>;

#define LUISA_STRUCT_MAKE_MEMBER_INIT(m)                          \
    m(luisa::compute::detail::FunctionBuilder::current()->member( \
        luisa::compute::Type::of<member_type_##m>(),              \
        this->_expression,                                        \
        _member_index(#m)))

#define LUISA_STRUCT_MAKE_MEMBER_REF_DECL(m) \
    luisa::compute::Var<member_type_##m> m;

#define LUISA_STRUCT_MAKE_MEMBER_EXPR_DECL(m) \
    luisa::compute::Expr<member_type_##m> m;

#define LUISA_STRUCT(S, ...)                                                                  \
    LUISA_STRUCT_REFLECT(S, __VA_ARGS__)                                                      \
    template<>                                                                                \
    struct luisa::compute::detail::dsl_struct_extension<S>;                                   \
    template<>                                                                                \
    struct luisa::compute::Expr<S> {                                                          \
    private:                                                                                  \
        using this_type = S;                                                                  \
        const luisa::compute::Expression *_expression;                                        \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                                 \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept { \
            constexpr const std::string_view member_names[]{                                  \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                \
            return std::find(std::begin(member_names),                                        \
                             std::end(member_names),                                          \
                             name) -                                                          \
                   std::begin(member_names);                                                  \
        }                                                                                     \
                                                                                              \
    public:                                                                                   \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_EXPR_DECL, __VA_ARGS__)                            \
        explicit Expr(const luisa::compute::Expression *e) noexcept                           \
            : _expression{e},                                                                 \
              LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__) {}                   \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }          \
        Expr(Expr &&another) noexcept = default;                                              \
        Expr(const Expr &another) noexcept = default;                                         \
        Expr &operator=(Expr) noexcept = delete;                                              \
        template<size_t i>                                                                    \
        [[nodiscard]] auto get() const noexcept {                                             \
            using M = std::tuple_element_t<i, struct_member_tuple_t<S>>;                      \
            return luisa::compute::Expr<M>{                                                   \
                luisa::compute::detail::FunctionBuilder::current()->member(                   \
                    luisa::compute::Type::of<M>(), this->expression(), i)};                   \
        };                                                                                    \
    };                                                                                        \
    template<>                                                                                \
    struct luisa::compute::detail::Ref<S> {                                                   \
    private:                                                                                  \
        using this_type = S;                                                                  \
        const luisa::compute::Expression *_expression;                                        \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                                 \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept { \
            constexpr const std::string_view member_names[]{                                  \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                \
            return std::find(std::begin(member_names), std::end(member_names), name) -        \
                   std::begin(member_names);                                                  \
        }                                                                                     \
                                                                                              \
    public:                                                                                   \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_REF_DECL, __VA_ARGS__)                             \
        explicit Ref(const luisa::compute::Expression *e) noexcept                            \
            : _expression{e},                                                                 \
              LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__) {}                   \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }          \
        Ref(Ref &&another) noexcept = default;                                                \
        Ref(const Ref &another) noexcept = default;                                           \
        [[nodiscard]] operator luisa::compute::Expr<S>() const noexcept {                     \
            return luisa::compute::Expr<S>{this->expression()};                               \
        }                                                                                     \
        template<typename Rhs>                                                                \
        void operator=(Rhs &&rhs) &noexcept {                                                 \
            luisa::compute::dsl::assign(*this, std::forward<Rhs>(rhs));                       \
        }                                                                                     \
        void operator=(Ref rhs) &noexcept { (*this) = Expr{rhs}; }                            \
        template<size_t i>                                                                    \
        [[nodiscard]] auto get() const noexcept {                                             \
            using M = std::tuple_element_t<i, struct_member_tuple_t<S>>;                      \
            return Ref<M>{luisa::compute::detail::FunctionBuilder::current()->member(         \
                luisa::compute::Type::of<M>(), this->expression(), i)};                       \
        };                                                                                    \
        [[nodiscard]] auto operator->() noexcept {                                            \
            return reinterpret_cast<                                                          \
                luisa::compute::detail::dsl_struct_extension<S> *>(this);                     \
        }                                                                                     \
        [[nodiscard]] auto operator->() const noexcept {                                      \
            return reinterpret_cast<                                                          \
                const luisa::compute::detail::dsl_struct_extension<S> *>(this);               \
        }                                                                                     \
    };                                                                                        \
    template<>                                                                                \
    struct luisa::compute::detail::dsl_struct_extension<S> final : luisa::compute::detail::Ref<S>

#define LUISA_BINDING_GROUP_MAKE_MEMBER_VAR_DECL(m) \
    luisa::compute::Var<member_type_##m> m;

#define LUISA_BINDING_GROUP_MAKE_MEMBER_EXPR_DECL(m) \
    luisa::compute::Expr<member_type_##m> m;

#define LUISA_BINDING_GROUP_MAKE_MEMBER_VAR_INIT(m) \
    m(luisa::compute::detail::ArgumentCreation{})

#define LUISA_BINDING_GROUP_MAKE_MEMBER_EXPR_INIT(m) \
    m(s.m)

#define LUISA_BINDING_GROUP_MAKE_INVOKE(m) \
    invoke << s.m;

#define LUISA_BINDING_GROUP(S, ...)                                                     \
    template<>                                                                          \
    struct luisa::compute::Var<S> {                                                     \
        using this_type = S;                                                            \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                           \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_MEMBER_VAR_DECL, __VA_ARGS__)                \
        explicit Var(luisa::compute::detail::ArgumentCreation) noexcept                 \
            : LUISA_MAP_LIST(LUISA_BINDING_GROUP_MAKE_MEMBER_VAR_INIT, __VA_ARGS__) {}  \
        Var(Var &&) noexcept = default;                                                 \
        Var(const Var &) noexcept = delete;                                             \
        Var &operator=(Var &&) noexcept = delete;                                       \
        Var &operator=(const Var &) noexcept = delete;                                  \
    };                                                                                  \
    template<>                                                                          \
    struct luisa::compute::Expr<S> {                                                    \
        using this_type = S;                                                            \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                           \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_MEMBER_EXPR_DECL, __VA_ARGS__)               \
        Expr(const luisa::compute::Var<S> &s) noexcept                                  \
            : LUISA_MAP_LIST(LUISA_BINDING_GROUP_MAKE_MEMBER_EXPR_INIT, __VA_ARGS__) {} \
        Expr(Expr &&another) noexcept = default;                                        \
        Expr(const Expr &another) noexcept = default;                                   \
        Expr &operator=(Expr) noexcept = delete;                                        \
    };                                                                                  \
    luisa::compute::detail::CallableInvoke &operator<<(                                 \
        luisa::compute::detail::CallableInvoke &invoke,                                 \
        luisa::compute::Expr<S> s) noexcept {                                           \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_INVOKE, __VA_ARGS__)                         \
        return invoke;                                                                  \
    }                                                                                   \
    luisa::compute::detail::ShaderInvokeBase &operator<<(                               \
        luisa::compute::detail::ShaderInvokeBase &invoke, const S &s) noexcept {        \
        LUISA_MAP(LUISA_BINDING_GROUP_MAKE_INVOKE, __VA_ARGS__)                         \
        return invoke;                                                                  \
    }
