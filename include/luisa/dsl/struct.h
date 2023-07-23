//
// Created by Mike Smith on 2021/3/5.
//

#pragma once

#include <cstdint>
#include <cstddef>

#include <luisa/dsl/var.h>
#include <luisa/dsl/atomic.h>
#include <luisa/dsl/func.h>
#include <luisa/runtime/shader.h>

template<typename T>
struct luisa_compute_extension {};

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

#define LUISA_STRUCT_MAKE_MEMBER_TYPE(m)                    \
    using member_type_##m = detail::c_array_to_std_array_t< \
        std::remove_cvref_t<                                \
            decltype(std::declval<this_type>().m)>>;

#define LUISA_STRUCT_MAKE_MEMBER_INIT(m)          \
    m(detail::FunctionBuilder::current()->member( \
        Type::of<member_type_##m>(),              \
        this->_expression,                        \
        _member_index(#m)))

#define LUISA_STRUCT_MAKE_MEMBER_REF_DECL(m) \
    Var<member_type_##m> m;

#define LUISA_STRUCT_MAKE_MEMBER_EXPR_DECL(m) \
    Expr<member_type_##m> m;

#define LUISA_STRUCT_MAKE_MEMBER_ATOMIC_REF_DECL(m) \
    AtomicRef<member_type_##m> m{                   \
        this->member<member_type_##m>(_member_index(#m))};

#define LUISA_STRUCT(S, ...)                                                                  \
    LUISA_STRUCT_REFLECT(S, __VA_ARGS__)                                                      \
    template<>                                                                                \
    struct luisa_compute_extension<S>;                                                        \
    namespace luisa::compute {                                                                \
    namespace detail {                                                                        \
    template<>                                                                                \
    class AtomicRef<S> : private AtomicRefBase {                                              \
    private:                                                                                  \
        using this_type = S;                                                                  \
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
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_ATOMIC_REF_DECL, __VA_ARGS__)                      \
        explicit AtomicRef(const AtomicRefNode *node) noexcept                                \
            : AtomicRefBase{node} {}                                                          \
    };                                                                                        \
    }                                                                                         \
    template<>                                                                                \
    struct Expr<S> {                                                                          \
    private:                                                                                  \
        using this_type = S;                                                                  \
        const Expression *_expression;                                                        \
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
        explicit Expr(const Expression *e) noexcept                                           \
            : _expression{e},                                                                 \
              LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__) {}                   \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }          \
        Expr(Expr &&another) noexcept = default;                                              \
        Expr(const Expr &another) noexcept = default;                                         \
        Expr &operator=(Expr) noexcept = delete;                                              \
        template<size_t i>                                                                    \
        [[nodiscard]] auto get() const noexcept {                                             \
            using M = std::tuple_element_t<i, struct_member_tuple_t<S>>;                      \
            return Expr<M>{detail::FunctionBuilder::current()->member(                        \
                Type::of<M>(), this->expression(), i)};                                       \
        };                                                                                    \
    };                                                                                        \
    namespace detail {                                                                        \
    template<>                                                                                \
    struct Ref<S> {                                                                           \
    private:                                                                                  \
        using this_type = S;                                                                  \
        const Expression *_expression;                                                        \
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
        explicit Ref(const Expression *e) noexcept                                            \
            : _expression{e},                                                                 \
              LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__) {}                   \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }          \
        Ref(Ref &&another) noexcept = default;                                                \
        Ref(const Ref &another) noexcept = default;                                           \
        [[nodiscard]] operator Expr<S>() const noexcept {                                     \
            return Expr<S>{this->expression()};                                               \
        }                                                                                     \
        template<typename Rhs>                                                                \
        void operator=(Rhs &&rhs) & noexcept { dsl::assign(*this, std::forward<Rhs>(rhs)); }  \
        void operator=(Ref rhs) & noexcept { (*this) = Expr{rhs}; }                           \
        template<size_t i>                                                                    \
        [[nodiscard]] auto get() const noexcept {                                             \
            using M = std::tuple_element_t<i, struct_member_tuple_t<S>>;                      \
            return Ref<M>{detail::FunctionBuilder::current()->member(                         \
                Type::of<M>(), this->expression(), i)};                                       \
        };                                                                                    \
        [[nodiscard]] auto operator->() noexcept {                                            \
            return reinterpret_cast<luisa_compute_extension<S> *>(this);                      \
        }                                                                                     \
        [[nodiscard]] auto operator->() const noexcept {                                      \
            return reinterpret_cast<const luisa_compute_extension<S> *>(this);                \
        }                                                                                     \
    };                                                                                        \
    }                                                                                         \
    }                                                                                         \
    template<>                                                                                \
    struct luisa_compute_extension<S> final : luisa::compute::detail::Ref<S>

#define LUISA_CUSTOM_STRUCT_EXT(S)                                                           \
    template<>                                                                               \
    struct luisa_compute_extension<S>;                                                       \
    namespace luisa::compute {                                                               \
    template<>                                                                               \
    struct Expr<S> {                                                                         \
    private:                                                                                 \
        using this_type = S;                                                                 \
        const Expression *_expression;                                                       \
                                                                                             \
    public:                                                                                  \
        explicit Expr(const Expression *e) noexcept                                          \
            : _expression{e} {}                                                              \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }         \
        Expr(Expr &&another) noexcept = default;                                             \
        Expr(const Expr &another) noexcept = default;                                        \
        Expr &operator=(Expr) noexcept = delete;                                             \
    };                                                                                       \
    namespace detail {                                                                       \
    template<>                                                                               \
    struct Ref<S> {                                                                          \
    private:                                                                                 \
        using this_type = S;                                                                 \
        const Expression *_expression;                                                       \
                                                                                             \
    public:                                                                                  \
        explicit Ref(const Expression *e) noexcept                                           \
            : _expression{e} {}                                                              \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }         \
        Ref(Ref &&another) noexcept = default;                                               \
        Ref(const Ref &another) noexcept = default;                                          \
        [[nodiscard]] operator Expr<this_type>() const noexcept {                            \
            return Expr<this_type>{this->expression()};                                      \
        }                                                                                    \
        template<typename Rhs>                                                               \
        void operator=(Rhs &&rhs) & noexcept { dsl::assign(*this, std::forward<Rhs>(rhs)); } \
        void operator=(Ref rhs) & noexcept { (*this) = Expr{rhs}; }                          \
        [[nodiscard]] auto operator->() noexcept {                                           \
            return reinterpret_cast<luisa_compute_extension<this_type> *>(this);             \
        }                                                                                    \
        [[nodiscard]] auto operator->() const noexcept {                                     \
            return reinterpret_cast<const luisa_compute_extension<this_type> *>(this);       \
        }                                                                                    \
    };                                                                                       \
    }                                                                                        \
    }                                                                                        \
    template<>                                                                               \
    struct luisa_compute_extension<S> final : luisa::compute::detail::Ref<S>

