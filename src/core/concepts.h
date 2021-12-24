//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <span>
#include <atomic>
#include <concepts>
#include <type_traits>
#include <string_view>

#include <core/macro.h>
#include <core/basic_types.h>
#include <core/atomic.h>
#include <core/allocator.h>

namespace luisa::concepts {

struct Noncopyable {
    Noncopyable() noexcept = default;
    Noncopyable(const Noncopyable &) noexcept = delete;
    Noncopyable &operator=(const Noncopyable &) noexcept = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

template<typename T>
concept iterable = requires(T v) {
    v.begin();
    v.end();
};

template<typename T>
concept string_viewable = requires(T v) {
    std::string_view{v};
};

template<typename T>
concept span_convertible = requires(T v) {
    luisa::span{v};
};

template<typename T, typename... Args>
concept constructible = requires(Args... args) {
    T{args...};
};

template<typename T>
concept trivially_default_constructible = std::is_trivially_constructible_v<T>;

template<typename Src, typename Dest>
concept static_convertible = requires(Src s) {
    static_cast<Dest>(s);
};

template<typename Src, typename Dest>
concept bitwise_convertible = sizeof(Src) >= sizeof(Dest);

template<typename Src, typename Dest>
concept reinterpret_convertible = requires(Src s) {
    reinterpret_cast<Dest *>(&s);
};

template<typename F, typename... Args>
concept invocable = std::is_invocable_v<F, Args...>;

template<typename Ret, typename F, typename... Args>
concept invocable_with_return = std::is_invocable_r_v<Ret, F, Args...>;

template<typename T>
concept pointer = std::is_pointer_v<T>;

template<typename T>
concept non_pointer = std::negation_v<std::is_pointer<T>>;

template<typename T>
concept container = requires(T a) {
    a.begin();
    a.size();
};

template<typename T>
concept integral = is_integral_v<T>;

template<typename T>
concept scalar = is_scalar_v<T>;

template<typename T>
concept vector = is_vector_v<T>;

template<typename T>
concept vector2 = is_vector2_v<T>;

template<typename T>
concept vector3 = is_vector3_v<T>;

template<typename T>
concept vector4 = is_vector4_v<T>;

template<typename T>
concept bool_vector = is_bool_vector_v<T>;

template<typename T>
concept float_vector = is_float_vector_v<T>;

template<typename T>
concept int_vector = is_int_vector_v<T>;

template<typename T>
concept uint_vector = is_uint_vector_v<T>;

template<typename T>
concept matrix = is_matrix_v<T>;

template<typename T>
concept matrix2 = is_matrix2_v<T>;

template<typename T>
concept matrix3 = is_matrix3_v<T>;

template<typename T>
concept matrix4 = is_matrix4_v<T>;

template<typename T>
concept basic = is_basic_v<T>;

template<typename T>
concept atomic = is_atomic_v<T>;

// operator traits
#define LUISA_MAKE_UNARY_OP_CONCEPT(op, op_name) \
    template<typename Operand>                   \
    concept op_name = requires(Operand operand) { op operand; };
LUISA_MAKE_UNARY_OP_CONCEPT(+, operator_plus)
LUISA_MAKE_UNARY_OP_CONCEPT(-, operator_minus)
LUISA_MAKE_UNARY_OP_CONCEPT(!, operator_not)
LUISA_MAKE_UNARY_OP_CONCEPT(~, operator_bit_not)
#undef LUISA_MAKE_UNARY_OP_CONCEPT

#define LUISA_MAKE_BINARY_OP_CONCEPT(op, op_name) \
    template<typename Lhs, typename Rhs>          \
    concept op_name = requires(Lhs lhs, Rhs rhs) { lhs op rhs; };
LUISA_MAKE_BINARY_OP_CONCEPT(+, operator_add)
LUISA_MAKE_BINARY_OP_CONCEPT(-, operator_sub)
LUISA_MAKE_BINARY_OP_CONCEPT(*, operator_mul)
LUISA_MAKE_BINARY_OP_CONCEPT(/, operator_div)
LUISA_MAKE_BINARY_OP_CONCEPT(%, operator_mod)
LUISA_MAKE_BINARY_OP_CONCEPT(&, operator_bit_and)
LUISA_MAKE_BINARY_OP_CONCEPT(|, operator_bit_or)
LUISA_MAKE_BINARY_OP_CONCEPT(^, operator_bit_Xor)
LUISA_MAKE_BINARY_OP_CONCEPT(<<, operator_shift_left)
LUISA_MAKE_BINARY_OP_CONCEPT(>>, operator_shift_right)
LUISA_MAKE_BINARY_OP_CONCEPT(&&, operator_and)
LUISA_MAKE_BINARY_OP_CONCEPT(||, operator_or)
LUISA_MAKE_BINARY_OP_CONCEPT(==, operator_equal)
LUISA_MAKE_BINARY_OP_CONCEPT(!=, operator_not_equal)
LUISA_MAKE_BINARY_OP_CONCEPT(<, operator_less)
LUISA_MAKE_BINARY_OP_CONCEPT(<=, operator_less_equal)
LUISA_MAKE_BINARY_OP_CONCEPT(>, operator_greater)
LUISA_MAKE_BINARY_OP_CONCEPT(>=, operator_greater_equal)

LUISA_MAKE_BINARY_OP_CONCEPT(=, assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(+=, add_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(-=, sub_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(*=, mul_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(/=, div_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(%=, mod_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(&=, bit_and_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(|=, bit_or_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(^=, bit_xor_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(<<=, shift_left_assignable)
LUISA_MAKE_BINARY_OP_CONCEPT(>>=, shift_right_assignable)
#undef LUISA_MAKE_BINARY_OP_CONCEPT

template<typename Lhs, typename Rhs>
concept operator_access = requires(Lhs lhs, Rhs rhs) { lhs[rhs]; };

template<typename T>
concept function = std::is_function_v<T>;

namespace detail {

    template<typename... T>
    struct all_same_impl : std::false_type {};

    template<>
    struct all_same_impl<> : std::true_type {};

    template<typename T>
    struct all_same_impl<T> : std::true_type {};

    template<typename First, typename... Other>
    struct all_same_impl<First, Other...> : std::conjunction<std::is_same<First, Other>...> {};

}// namespace detail

template<typename... T>
using is_same = detail::all_same_impl<T...>;

template<typename... T>
constexpr auto is_same_v = is_same<T...>::value;

template<typename... T>
concept same = is_same_v<T...>;

template<typename A, typename B>
concept different = !same<A, B>;

template<typename... T>
concept vector_same_dimension = is_vector_same_dimension_v<T...>;

}// namespace luisa::concepts
