//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <span>
#include <atomic>
#include <concepts>
#include <type_traits>

#include <core/macro.h>
#include <core/basic_types.h>
#include <core/atomic.h>

namespace luisa::concepts {

struct Noncopyable {
    Noncopyable() = default;
    Noncopyable(const Noncopyable &) = delete;
    Noncopyable &operator=(const Noncopyable &) = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

template<typename T>
concept span_convertible = requires(T v) {
    std::span{std::forward<T>(v)};
};

template<typename T, typename... Args>
concept constructible = requires(Args... args) {
    T{args...};
};

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
concept matrix = is_matrix_v<T>;

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

}// namespace luisa::concepts
