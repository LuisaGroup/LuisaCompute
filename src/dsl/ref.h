//
// Created by Mike Smith on 2021/8/26.
//

#pragma once
#ifndef LC_DISABLE_DSL
#include <dsl/expr.h>

namespace luisa::compute {

inline namespace dsl {
/// Assign rhs to lhs
template<typename Lhs, typename Rhs>
void assign(Lhs &&lhs, Rhs &&rhs) noexcept;// defined in dsl/stmt.h
}

namespace detail {

/// Enable subscript access
template<typename T>
struct RefEnableSubscriptAccess {

public:
    /// Access index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto operator[](I &&index) const &noexcept {
        auto self = def<T>(static_cast<const T *>(this)->expression());
        using Elem = std::remove_cvref_t<
            decltype(std::declval<expr_value_t<T>>()[0])>;
        return def<Elem>(FunctionBuilder::current()->access(
            Type::of<Elem>(), self.expression(),
            extract_expression(std::forward<I>(index))));
    }
    /// Access index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto &operator[](I &&index) &noexcept {
        auto i = def(std::forward<I>(index));
        using Elem = std::remove_cvref_t<
            decltype(std::declval<expr_value_t<T>>()[0])>;
        auto f = FunctionBuilder::current();
        auto expr = f->access(
            Type::of<Elem>(),
            static_cast<const T *>(this)->expression(),
            i.expression());
        return *f->create_temporary<Var<Elem>>(expr);
    }
};

/// Enbale get member by index
template<typename T>
struct RefEnableGetMemberByIndex {
    /// Get member by index
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < dimension_v<expr_value_t<T>>);
        auto self = const_cast<T *>(static_cast<const T *>(this));
        return Ref{self->operator[](static_cast<uint>(i))};
    }
};

/// Ref class common definition
#define LUISA_REF_COMMON(...)                                              \
private:                                                                   \
    const Expression *_expression;                                         \
                                                                           \
public:                                                                    \
    explicit Ref(const Expression *e) noexcept : _expression{e} {}         \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Ref(Ref &&) noexcept = default;                                        \
    Ref(const Ref &) noexcept = default;                                   \
    template<typename Rhs>                                                 \
    void operator=(Rhs &&rhs) &noexcept {                                  \
        dsl::assign(*this, std::forward<Rhs>(rhs));                        \
    }                                                                      \
    [[nodiscard]] operator Expr<__VA_ARGS__>() const noexcept {            \
        return Expr<__VA_ARGS__>{this->expression()};                      \
    }                                                                      \
    void operator=(Ref rhs) &noexcept { (*this) = Expr<__VA_ARGS__>{rhs}; }

/// Ref<T>
template<typename T>
struct Ref
    : detail::ExprEnableStaticCast<Ref<T>>,
      detail::ExprEnableBitwiseCast<Ref<T>> {
    static_assert(concepts::scalar<T>);
    LUISA_REF_COMMON(T)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i == 0u);
        return *this;
    }
};

// error reporters for Ref<DynamicStruct>
void ref_dynamic_struct_error_member_out_of_range(size_t member_count, size_t index) noexcept;
void ref_dynamic_struct_error_member_type_mismatched(const Type *requested, const Type *actual) noexcept;
void ref_dynamic_struct_error_member_not_found(luisa::string_view name) noexcept;

template<>
struct Ref<DynamicStruct> {
private:
    const Expression *_expression;

protected:
    const DynamicStruct *_type;

public:
    explicit Ref(const DynamicStruct *type, const Expression *e) noexcept : _expression{e}, _type(type) {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    Ref(Ref &&) noexcept = default;
    Ref(const Ref &) noexcept = default;
    [[nodiscard]] operator Expr<DynamicStruct>() const noexcept {
        return Expr<DynamicStruct>{this->expression()};
    }
    void operator=(Ref rhs) &noexcept {
        _type = rhs._type;
        detail::FunctionBuilder::current()->assign(
            _expression,
            detail::extract_expression(std::move(rhs)));
    }
    template<typename M>
    [[nodiscard]] auto get(size_t i) const noexcept {
        auto struct_type = _expression->type();
        auto ele_type = Type::of<M>();
        auto members = struct_type->members();
        if (i >= members.size()) [[unlikely]] {
            ref_dynamic_struct_error_member_out_of_range(members.size(), i);
        }
        if (*members[i] != *ele_type) [[unlikely]] {
            ref_dynamic_struct_error_member_type_mismatched(ele_type, members[i]);
        }
        return Ref<M>{detail::FunctionBuilder::current()->member(
            ele_type, this->expression(), i)};
    }
    template<typename M>
    [[nodiscard]] auto get(luisa::string_view name) const noexcept {
        auto idx = _type->member_index(name);
        if(idx == ~0ull) [[unlikely]]{
            ref_dynamic_struct_error_member_not_found(name);
        }
        return get<M>(idx);
    }
};

/// Ref<std::array<T, N>>
template<typename T, size_t N>
struct Ref<std::array<T, N>>
    : detail::RefEnableSubscriptAccess<Ref<std::array<T, N>>>,
      detail::RefEnableGetMemberByIndex<Ref<std::array<T, N>>> {
    LUISA_REF_COMMON(std::array<T, N>)
};

/// Ref<std::array<T, N>>
template<typename T, size_t N>
struct Ref<T[N]>
    : detail::RefEnableSubscriptAccess<Ref<T[N]>>,
      detail::RefEnableGetMemberByIndex<Ref<T[N]>> {
    LUISA_REF_COMMON(T[N])
};

/// Ref<Matrix<N>>
template<size_t N>
struct Ref<Matrix<N>>
    : detail::RefEnableSubscriptAccess<Ref<Matrix<N>>>,
      detail::RefEnableGetMemberByIndex<Ref<Matrix<N>>> {
    LUISA_REF_COMMON(Matrix<N>)
};

/// Ref<std::tuple<T...>>
template<typename... T>
struct Ref<std::tuple<T...>> {
    LUISA_REF_COMMON(std::tuple<T...>)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        using M = std::tuple_element_t<i, std::tuple<T...>>;
        return Ref<M>{detail::FunctionBuilder::current()->member(
            Type::of<M>(), this->expression(), i)};
    }
};

/// Ref<Vector<T, 2>>
template<typename T>
struct Ref<Vector<T, 2>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 2>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 2>>>,
      detail::RefEnableSubscriptAccess<Ref<Vector<T, 2>>>,
      detail::RefEnableGetMemberByIndex<Ref<Vector<T, 2>>> {
    LUISA_REF_COMMON(Vector<T, 2>)
    Var<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Var<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

/// Ref<Vector<T, 3>>
template<typename T>
struct Ref<Vector<T, 3>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 3>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 3>>>,
      detail::RefEnableSubscriptAccess<Ref<Vector<T, 3>>>,
      detail::RefEnableGetMemberByIndex<Ref<Vector<T, 3>>> {
    LUISA_REF_COMMON(Vector<T, 3>)
    Var<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Var<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Var<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

/// Ref<Vector<T, 4>>
template<typename T>
struct Ref<Vector<T, 4>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 4>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 4>>>,
      detail::RefEnableSubscriptAccess<Ref<Vector<T, 4>>>,
      detail::RefEnableGetMemberByIndex<Ref<Vector<T, 4>>> {
    LUISA_REF_COMMON(Vector<T, 4>)
    Var<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Var<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Var<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
    Var<T> w{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x3u)};
#include <dsl/swizzle_4.inl.h>
};

#undef LUISA_REF_COMMON

}// namespace detail
}// namespace luisa::compute
#endif