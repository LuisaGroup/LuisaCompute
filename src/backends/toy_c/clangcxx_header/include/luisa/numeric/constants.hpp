#pragma once
#include "../type_traits.hpp"

namespace luisa::shader {

/// pi
constexpr float pi = 3.14159265358979323846264338327950288f;

/// pi/2
constexpr float pi_over_two = 1.57079632679489661923132169163975144f;

/// pi/4
constexpr float pi_over_four = 0.785398163397448309615660845819875721f;

/// 1/pi
constexpr float inv_pi = 0.318309886183790671537767526745028724f;

/// 2/pi
constexpr float two_over_pi = 0.636619772367581343075535053490057448f;

/// sqrt(2)
constexpr float sqrt_two = 1.41421356237309504880168872420969808f;

/// 1/sqrt(2)
constexpr float inv_sqrt_two = 0.707106781186547524400844362104849039f;

/// 1-epsilon
constexpr float one_minus_epsilon = 0x1.fffffep-1f;

template <typename T>
inline constexpr T identity;

template <concepts::arithmetic_scalar T> inline constexpr T identity<T> = 1;
template <concepts::arithmetic_scalar T, uint32 N> inline constexpr vec<T, N> identity<vec<T, N>> = vec<T, N>(1);
template <uint32 N> inline constexpr matrix<N> identity<matrix<N>> = matrix<N>::identity();


}