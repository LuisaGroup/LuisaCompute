#pragma once
#include "./../attributes.hpp"

using int16 = short;
using uint16 = unsigned short;
using int32 = int;
using int64 = long long;
using uint32 = unsigned int;
using uint = uint32;
using uint64 = unsigned long long;

template<typename T, typename U>
inline constexpr bool is_same_v = __is_same_as(T, U);

template <class T, class Arg>
inline constexpr bool is_assignable_v = __is_assignable(T, Arg);

template <class T, template <class...> class Template>
inline constexpr bool is_specialization_v = false; // true if and only if T is a specialization of Template

template <template <class...> class Template, class... Args>
inline constexpr bool is_specialization_v<Template<Args...>, Template> = true;

template<class T>
inline constexpr bool is_void_v = __is_void(T);

template<class T>
using decay_t = __decay(T);

template<class T>
static constexpr bool is_signed_v = __is_signed(T);

template<class T>
using signed_t = __make_signed(T);

template<class T>
static constexpr bool is_unsigned_v = __is_unsigned(T);

template<class T>
using unsigned_t = __make_unsigned(T);

template<class T>
using remove_extent_t = __remove_extent(T);

template<typename T>
static constexpr bool is_reference_v = __is_reference(T);

template<class T>
using remove_reference_t = __remove_reference_t(T);

template<class T>
static constexpr bool is_const_v = __is_const(T);

template<class T>
using remove_const_t = __remove_const(T);

template<class T>
static constexpr bool is_volatile_v = __is_volatile(T);

template<class T>
using remove_volatile_t = __remove_volatile(T);

template<class T>
using remove_cv_t = __remove_cv(T);

template<class T>
using remove_cvref_t = __remove_cvref(T);

template<class T, T v>
trait integral_constant {
    static constexpr const T value = v;
    typedef T value_type;
    typedef integral_constant type;
    explicit constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
};
typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

/*
template<typename T>
[[ignore]] constexpr T&& forward(typename remove_cvref<T>::type& Arg) {
    return static_cast<T&&>(Arg);
}

template<typename T>
[[ignore]] constexpr T&& forward(typename remove_cvref<T>::type&& Arg) {
    return static_cast<T&&>(Arg);
}
*/

namespace luisa::shader {

struct [[builtin("half")]] half {
    [[ignore]] half() = default;
    [[ignore]] explicit half(float);
    [[ignore]] explicit half(uint32);
    [[ignore]] explicit half(int32);
private:
    short v;
};

template<typename T, uint64 N>
struct vec;

template<uint64 N>
struct matrix;

}// namespace luisa::shader