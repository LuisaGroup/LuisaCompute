//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <type_traits>
#include <core/data_types.h>

namespace luisa::concepts {

struct Noncopyable {
    Noncopyable() = default;
    Noncopyable(const Noncopyable &) = delete;
    Noncopyable &operator=(const Noncopyable &) = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

template<typename T>
concept SpanConvertible = requires(T v) {
    std::span{std::forward<T>(v)};
};

template<typename T, typename... Args>
concept Constructible = requires(Args ...args) {
    T{args...};
};

template<typename F, typename... Args>
concept Invocable = std::is_invocable_v<F, Args...>;

template<typename Ret, typename F, typename... Args>
concept InvocableRet = std::is_invocable_r_v<Ret, F, Args...>;

template<typename T>
concept Container = requires(T a) {
    a.begin();
    a.size();
};

template<typename T>
concept Integral = is_integral_v<T>;

template<typename T>
concept Unsigned = std::conjunction_v<is_integral<T>, std::is_unsigned<T>>;

template<typename T>
concept Scalar = is_scalar_v<T>;

template<typename T>
concept Vector = is_vector_v<T>;

template<typename T>
concept Matrix = is_matrix_v<T>;

template<typename T>
concept Native = Scalar<T> || Vector<T> || Matrix<T>;

// operator traits
#define LUISA_MAKE_UNARY_OP_CONCEPT(op, op_name) \
    template<typename Operand>                   \
    concept op_name = requires(Operand operand) { op operand; };
#define LUISA_MAKE_UNARY_OP_CONCEPT_FROM_PAIR(op_and_name) LUISA_MAKE_UNARY_OP_CONCEPT op_and_name
LUISA_MAP(LUISA_MAKE_UNARY_OP_CONCEPT_FROM_PAIR,
          (+, Plus),
          (-, Minus),
          (!, Not ),
          (~, BitNot))
#undef LUISA_MAKE_UNARY_OP_CONCEPT
#undef LUISA_MAKE_UNARY_OP_CONCEPT_FROM_PAIR

#define LUISA_MAKE_BINARY_OP_CONCEPT(op, op_name) \
    template<typename Lhs, typename Rhs>          \
    concept op_name = requires(Lhs lhs, Rhs rhs) { lhs op rhs; };

#define LUISA_MAKE_BINARY_OP_CONCEPT_FROM_PAIR(op_and_name) LUISA_MAKE_BINARY_OP_CONCEPT op_and_name

LUISA_MAP(LUISA_MAKE_BINARY_OP_CONCEPT_FROM_PAIR,
          (+, Add),
          (-, Sub),
          (*, Mul),
          (/, Div),
          (%, Mod),
          (&, BitAnd),
          (|, BitOr),
          (^, BitXor),
          (<<, ShiftLeft),
          (>>, ShiftRight),
          (&&, And),
          (||, Or),
          (==, Equal),
          (!=, NotEqual),
          (<, Less),
          (<=, LessEqual),
          (>, Greater),
          (>=, GreaterEqual),
          (=, Assign),
          (+=, AddAssign),
          (-=, SubAssign),
          (*=, MulAssign),
          (/=, DivAssign),
          (%=, ModAssign),
          (&=, BitAndAssign),
          (|=, BitOrAssign),
          (^=, BitXorAssign),
          (<<=, ShiftLeftAssign),
          (>>=, ShiftRightAssign))

#undef LUISA_MAKE_BINARY_OP_CONCEPT
#undef LUISA_MAKE_BINARY_OP_CONCEPT_FROM_PAIR

template<typename Lhs, typename Rhs>
concept Access = requires(Lhs lhs, Rhs rhs) { lhs[rhs]; };

}// namespace luisa::concepts
