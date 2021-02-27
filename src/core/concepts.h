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
concept span_convertible = requires(T v) {
    std::span{std::forward<T>(v)};
};

template<typename T, typename... Args>
concept constructable = std::is_constructible_v<T, Args...>;

template<typename T>
concept container_type = requires(T a) {
    a.begin();
    a.size();
};

template<typename T>
concept scalar_type = is_scalar_v<T>;

template<typename T>
concept vector_type = is_vector_v<T>;

template<typename T>
concept matrix_type = is_matrix_v<T>;

template<typename T>
concept core_data_type = scalar_type<T> || vector_type<T> || matrix_type<T>;

namespace detail {

template<typename T>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

template<typename T>
concept tuple_type = is_tuple_v<T>;

}// namespace detail

// TODO: seems not working...
//namespace detail {
//
//template<size_t N>
//struct StructuralBinding {
//    static_assert(always_false<std::index_sequence<N>>);
//};
//
//#define LUISA_TEST_STRUTURAL_BINDING_VAR(id) v##id
//#define LUISA_MAKE_TEST_STRUCTURAL_BINDING(N)                                                         \
//    template<>                                                                                        \
//    struct StructuralBinding<N> {                                                                     \
//        template<typename T>                                                                          \
//        [[nodiscard]] auto operator()(T &&v) noexcept {                                               \
//            auto &&[LUISA_MAP_LIST(LUISA_TEST_STRUTURAL_BINDING_VAR, LUISA_RANGE(N))] = v;            \
//            return std::make_tuple(LUISA_MAP_LIST(LUISA_TEST_STRUTURAL_BINDING_VAR, LUISA_RANGE(N))); \
//        }                                                                                             \
//    };
//
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(1)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(2)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(3)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(4)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(5)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(6)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(7)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(8)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(9)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(10)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(11)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(12)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(13)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(14)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(15)
//LUISA_MAKE_TEST_STRUCTURAL_BINDING(16)
//
//#undef LUISA_TEST_STRUTURAL_BINDING_VAR
//#undef LUISA_MAKE_TEST_STRUCTURAL_BINDING
//
//template<typename T, size_t N>
//using is_structural_bindable = is_tuple<decltype(StructuralBinding<N>{}(std::declval<T>()))>;
//
//template<typename T, size_t N>
//constexpr auto is_structural_bindable_v = is_structural_bindable<T, N>::value;
//
//}// namespace detail
//
//template<typename T, size_t N>
//concept structural_bindable = detail::is_structural_bindable_v<T, N>;

// operator traits
#define LUISA_MAKE_UNARY_OP_CONCEPT(op, op_name) \
    template<typename Operand>                   \
    concept operator_##op_name = requires(Operand operand) { op operand; };
#define LUISA_MAKE_UNARY_OP_CONCEPT_FROM_PAIR(op_and_name) LUISA_MAKE_UNARY_OP_CONCEPT op_and_name
LUISA_MAP(LUISA_MAKE_UNARY_OP_CONCEPT_FROM_PAIR,
          (+, plus),
          (-, minus),
          (!, not ),
          (~, bit_not))
#undef LUISA_MAKE_UNARY_OP_CONCEPT
#undef LUISA_MAKE_UNARY_OP_CONCEPT_FROM_PAIR

#define LUISA_MAKE_BINARY_OP_CONCEPT(op, op_name) \
    template<typename Lhs, typename Rhs>          \
    concept operator_##op_name = requires(Lhs lhs, Rhs rhs) { lhs op rhs; };

#define LUISA_MAKE_BINARY_OP_CONCEPT_FROM_PAIR(op_and_name) LUISA_MAKE_BINARY_OP_CONCEPT op_and_name

LUISA_MAP(LUISA_MAKE_BINARY_OP_CONCEPT_FROM_PAIR,
          (+, add),
          (-, sub),
          (*, mul),
          (/, div),
          (%, mod),
          (&, bit_and),
          (|, bit_or),
          (^, bit_xor),
          (>>, shr),
          (<<, shl),
          (&&, and),
          (||, or),
          (==, equal),
          (!=, not_equal),
          (<, less),
          (<=, less_equal),
          (>, greater),
          (>=, greater_equal),
          (=, assign),
          (+=, add_assign),
          (-=, sub_assign),
          (*=, mul_assign),
          (/=, div_assign),
          (%=, mod_assign),
          (&=, bit_and_assign),
          (|=, bit_or_assign),
          (^=, bit_xor_assign),
          (>>=, shr_assign),
          (<<=, shl_assign))

#undef LUISA_MAKE_BINARY_OP_CONCEPT
#undef LUISA_MAKE_BINARY_OP_CONCEPT_FROM_PAIR

template<typename Lhs, typename Rhs>
concept operator_access = requires(Lhs lhs, Rhs rhs) { lhs[rhs]; };

}// namespace luisa::concepts
