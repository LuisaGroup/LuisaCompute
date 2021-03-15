//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <span>
#include <type_traits>

#include <core/macro.h>
#include <core/basic_types.h>

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
concept Constructible = requires(Args... args) {
    T{args...};
};

template<typename Src, typename Dest>
concept StaticConvertible = requires(Src s) {
    static_cast<Dest>(s);
};

template<typename Src, typename Dest>
concept BitwiseConvertible = sizeof(Src) >= sizeof(Dest);

template<typename Src, typename Dest>
concept ReinterpretConvertible = requires(Src s) {
    reinterpret_cast<Dest *>(&s);
};

template<typename F, typename... Args>
concept Invocable = std::is_invocable_v<F, Args...>;

template<typename Ret, typename F, typename... Args>
concept InvocableRet = std::is_invocable_r_v<Ret, F, Args...>;

template<typename T>
concept Pointer = std::is_pointer_v<T>;

template<typename T>
concept NonPointer = std::negation_v<std::is_pointer<T>>;

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
concept Basic = is_basic_v<T>;

// operator traits
#define LUISA_MAKE_UNARY_OP_CONCEPT(op, op_name) \
    template<typename Operand>                   \
    concept op_name = requires(Operand operand) { op operand; };
LUISA_MAKE_UNARY_OP_CONCEPT(+, Plus)
LUISA_MAKE_UNARY_OP_CONCEPT(-, Minus)
LUISA_MAKE_UNARY_OP_CONCEPT(!, Not)
LUISA_MAKE_UNARY_OP_CONCEPT(~, BitNot)
#undef LUISA_MAKE_UNARY_OP_CONCEPT

#define LUISA_MAKE_BINARY_OP_CONCEPT(op, op_name) \
    template<typename Lhs, typename Rhs>          \
    concept op_name = requires(Lhs lhs, Rhs rhs) { lhs op rhs; };
LUISA_MAKE_BINARY_OP_CONCEPT(+, Add)
LUISA_MAKE_BINARY_OP_CONCEPT(-, Sub)
LUISA_MAKE_BINARY_OP_CONCEPT(*, Mul)
LUISA_MAKE_BINARY_OP_CONCEPT(/, Div)
LUISA_MAKE_BINARY_OP_CONCEPT(%, Mod)
LUISA_MAKE_BINARY_OP_CONCEPT(&, BitAnd)
LUISA_MAKE_BINARY_OP_CONCEPT(|, BitOr)
LUISA_MAKE_BINARY_OP_CONCEPT(^, BitXor)
LUISA_MAKE_BINARY_OP_CONCEPT(<<, ShiftLeft)
LUISA_MAKE_BINARY_OP_CONCEPT(>>, ShiftRight)
LUISA_MAKE_BINARY_OP_CONCEPT(&&, And)
LUISA_MAKE_BINARY_OP_CONCEPT(||, Or)
LUISA_MAKE_BINARY_OP_CONCEPT(==, Equal)
LUISA_MAKE_BINARY_OP_CONCEPT(!=, NotEqual)
LUISA_MAKE_BINARY_OP_CONCEPT(<, Less)
LUISA_MAKE_BINARY_OP_CONCEPT(<=, LessEqual)
LUISA_MAKE_BINARY_OP_CONCEPT(>, Greater)
LUISA_MAKE_BINARY_OP_CONCEPT(>=, GreaterEqual)
LUISA_MAKE_BINARY_OP_CONCEPT(=, Assign)
LUISA_MAKE_BINARY_OP_CONCEPT(+=, AddAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(-=, SubAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(*=, MulAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(/=, DivAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(%=, ModAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(&=, BitAndAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(|=, BitOrAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(^=, BitXorAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(<<=, ShiftLeftAssign)
LUISA_MAKE_BINARY_OP_CONCEPT(>>=, ShiftRightAssign)
#undef LUISA_MAKE_BINARY_OP_CONCEPT

template<typename Lhs, typename Rhs>
concept Access = requires(Lhs lhs, Rhs rhs) { lhs[rhs]; };

template<typename T>
concept Function = std::is_function_v<T>;

}// namespace luisa::concepts
