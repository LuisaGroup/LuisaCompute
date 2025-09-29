#pragma once
#include "./../attributes.hpp"
#include "./../type_traits.hpp"
#include "./../types/vec.hpp"
#include "./../types/matrix.hpp"

#include <std/type_traits>
#include <std/array>

namespace luisa::shader {

template<concepts::arithmetic T, concepts::arithmetic U>
	requires(sizeof(T) == sizeof(U))
[[expr("bit_cast")]] extern T bit_cast(U v);

template<concepts::bool_family T>
[[callop("ALL")]] extern bool all(const T& x);

template<concepts::bool_family T>
[[callop("ANY")]] extern bool any(const T& x);

template<concepts::primitive T, concepts::bool_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("SELECT")]] extern T select(const T& false_v, const T& true_v, const B& bool_v);

template<concepts::primitive T, concepts::bool_family B>
	requires(vec_dim_v<T> == vec_dim_v<B> || vec_dim_v<B> == 1)
inline T ite(B bool_v, T true_v, T false_v) {
	return select(false_v, true_v, bool_v);
}

template<concepts::arithmetic T, concepts::arithmetic A, concepts::arithmetic B>
	requires((same_dim_v<T, B> || is_scalar_v<B>) && (same_dim_v<T, A> || is_scalar_v<A>))
[[callop("CLAMP")]] extern T clamp(const T& v, const A& min_v, const B& max_v);

template<concepts::float_family T, concepts::float_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("LERP")]] extern T lerp(const T& left_v, const T& right_v, const B& step);

template<concepts::float_family T, concepts::float_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("SMOOTHSTEP")]] extern T smoothstep(const T& left_v, const B& right_v, const B& step);

template<concepts::float_family T, concepts::float_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("STEP")]] extern T step(const T& left_v, const B& right_v);

template<concepts::float_family T>
[[callop("SATURATE")]] extern T saturate(const T& v);

template<concepts::signed_arithmetic T>
[[callop("ABS")]] extern T abs(const T& v);

template<concepts::arithmetic T, concepts::arithmetic U>
[[callop("MIN")]] extern T min(const T& a, U b)
	requires(vec_dim_v<T> == vec_dim_v<U>);

template<concepts::arithmetic T, concepts::arithmetic U>
[[callop("MAX")]] extern T max(const T& v, U b)
	requires(vec_dim_v<T> == vec_dim_v<U>);

template<class T>
	requires(std::is_enum_v<T>)
inline T min(const T& a, const T& b) {
	return a < b ? a : b;
}
template<class T>
	requires(std::is_enum_v<T>)
inline T max(const T& a, const T& b) {
	return a > b ? a : b;
}

template<concepts::uint_family T>
[[callop("CLZ")]] extern T clz(const T& v);

template<concepts::uint_family T>
[[callop("CTZ")]] extern T ctz(const T& v);

template<concepts::uint_family T>
[[callop("POPCOUNT")]] extern T popcount(const T& v);

template<concepts::uint_family T>
[[callop("REVERSE")]] extern T reverse(const T& v);

template<concepts::float_vec_family T>
[[callop("ISINF")]] extern vec<bool, vec_dim_v<T>> is_inf(const T& v);

template<concepts::float_vec_family T>
[[callop("ISNAN")]] extern vec<bool, vec_dim_v<T>> is_nan(const T& v);

[[callop("ISINF")]] extern bool is_inf(float v);

[[callop("ISNAN")]] extern bool is_nan(float v);

template<concepts::float_vec_family T>
inline vec<bool, vec_dim_v<T>> is_finite(T const& v) {
	return (is_nan(v) || is_inf(v)) == false;
}

inline bool is_finite(float v) {
	return !(is_nan(v) || is_inf(v));
}

template<concepts::float_family T>
[[callop("ACOS")]] extern T acos(const T& v);

template<concepts::float_family T>
[[callop("ACOSH")]] extern T acosh(const T& v);

template<concepts::float_family T>
[[callop("ASIN")]] extern T asin(const T& v);

template<concepts::float_family T>
[[callop("ASINH")]] extern T asinh(const T& v);

template<concepts::float_family T>
[[callop("ATAN")]] extern T atan(const T& v);

template<concepts::float_family T, concepts::float_family U>
[[callop("ATAN2")]] extern T atan2(const T& a, const U& b)
	requires(vec_dim_v<T> == vec_dim_v<U>);

template<concepts::float_family T>
[[callop("ATANH")]] extern T atanh(const T& v);

template<concepts::float_family T>
[[callop("COS")]] extern T cos(const T& v);

template<concepts::float_family T>
[[callop("COSH")]] extern T cosh(const T& v);

template<concepts::float_family T>
[[callop("SIN")]] extern T sin(const T& v);

template<concepts::float_family T>
[[callop("SINH")]] extern T sinh(const T& v);

template<concepts::float_family T>
[[callop("TAN")]] extern T tan(const T& v);

template<concepts::float_family T>
[[callop("TANH")]] extern T tanh(const T& v);

template<concepts::float_family T>
[[callop("EXP")]] extern T exp(const T& v);

template<concepts::float_family T>
[[callop("EXP2")]] extern T exp2(const T& v);

template<concepts::float_family T>
[[callop("EXP10")]] extern T exp10(const T& v);

template<concepts::float_family T>
[[callop("LOG")]] extern T log(const T& v);

template<concepts::float_family T>
[[callop("LOG2")]] extern T log2(const T& v);

template<concepts::float_family T>
[[callop("LOG10")]] extern T log10(const T& v);

template<concepts::float_family T, concepts::float_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("POW")]] extern T pow(const T& base, const B& rate);

template<concepts::float_family T>
[[callop("SQRT")]] extern T sqrt(const T& v);

template<concepts::float_family T>
constexpr T sqr(const T& v) {
	return v * v;
}

template<concepts::float_family T>
constexpr T pow4(T x) {
	return sqr(sqr(x));
}

template<concepts::float_family T>
constexpr T pow5(T x) {
	return pow4(x) * x;
}

template<concepts::float_family T>
constexpr T pow6(T x) {
	T x2 = sqr(x);
	return sqr(x2) * x2;
}

template<concepts::float_family T>
[[callop("RSQRT")]] extern T rsqrt(const T& v);

template<concepts::float_family T>
[[callop("CEIL")]] extern T ceil(const T& v);

template<concepts::float_family T>
[[callop("FLOOR")]] extern T floor(const T& v);

template<concepts::float_family T>
[[callop("FRACT")]] extern T fract(const T& v);

template<concepts::float_family T>
[[callop("TRUNC")]] extern T trunc(const T& v);

template<concepts::float_family T>
[[callop("ROUND")]] extern T round(const T& v);

template<concepts::float_family T>
[[callop("FMA")]] extern T fma(const T& a, const T& b, const T& c);

template<concepts::float_family T>
[[callop("COPYSIGN")]] extern T copysign(const T& a, T b);

template<concepts::float_vec_family T>
[[callop("CROSS")]] extern T cross(const T& a, const T& b);

[[callop("FACEFORWARD")]] extern float3 faceforward(const float3& a, const float3& b, const float3& c);
[[callop("FACEFORWARD")]] extern half3 faceforward(const half3& a, const half3& b, const half3& c);

[[callop("REFLECT")]] extern float3 reflect(const float3& i, const float3& n);
[[callop("REFLECT")]] extern half3 reflect(const half3& i, const half3& n);

template<concepts::float_vec_family T>
[[callop("DOT")]] extern scalar_type<T> dot(const T& a, const T& b);

template<concepts::float_vec_family T>
[[callop("LENGTH")]] extern scalar_type<T> length(const T& v);

template<concepts::float_vec_family T>
scalar_type<T> distance(const T& a, const T& b) { return length(a - b); }

template<concepts::float_vec_family T>
[[callop("LENGTH_SQUARED")]] extern scalar_type<T> length_squared(const T& v);

template<concepts::float_vec_family T>
[[callop("NORMALIZE")]] extern T normalize(const T& v);

template<concepts::arithmetic_vec T>
[[callop("REDUCE_SUM")]] extern scalar_type<T> reduce_sum(const T& v);

template<concepts::arithmetic_vec T>
[[callop("REDUCE_PRODUCT")]] extern scalar_type<T> reduce_product(const T& v);

template<concepts::arithmetic_vec T>
[[callop("REDUCE_MIN")]] extern scalar_type<T> reduce_min(const T& v);

template<concepts::arithmetic_vec T>
[[callop("REDUCE_MAX")]] extern scalar_type<T> reduce_max(const T& v);

template<concepts::matrix T>
[[callop("DETERMINANT")]] extern T determinant(const T& v);

template<concepts::matrix T>
[[callop("TRANSPOSE")]] extern T transpose(const T& v);

template<concepts::matrix T>
[[callop("INVERSE")]] extern T inverse(const T& v);

template<concepts::float_family T>
[[callop("DDX")]] extern T ddx();

template<concepts::float_family T>
[[callop("DDY")]] extern T ddy();

template<concepts::float_family T>
auto sign(T v) {
	using Type = copy_dim<int, T>::type;
	using MaskType = copy_dim<uint, T>::type;
	auto low = select(Type(1), Type(0), (bit_cast<MaskType>(v) & MaskType(0x7fffffffu)) == MaskType(0));
	auto high = select(Type(2), Type(0), (bit_cast<MaskType>(v) & MaskType(0x80000000u)) == MaskType(0));
	return low - high;
}

template<concepts::float_family T>
constexpr T rcp(T t) {
	return T(1.0f) / t;
}

template<concepts::float_family T>
constexpr T inverse_smoothstep(T y) { return T(0.5f) - sin(asin(T(1.0f) - T(2.0f) * y) / T(3.0f)); }

inline float lgamma(float x) {
	constexpr float const gam0 = 1.0f / 12.0f;
	constexpr float const gam1 = 1.0f / 30.0f;
	constexpr float const gam2 = 53.0f / 210.0f;
	constexpr float const gam3 = 195.0f / 371.0f;
	constexpr float const gam4 = 22999.0f / 22737.0f;
	constexpr float const gam5 = 29944523.0f / 19733142.0f;
	constexpr float const gam6 = 109535241009.0f / 48264275462.0f;

	return 0.5f * log(2 * pi) - x + (x - 0.5f) * log(x) + gam0 / (x + gam1 / (x + gam2 / (x + gam3 / (x + gam4 / (x + gam5 / (x + gam6 / x))))));
}

inline float tgamma(float x) {
	float result;
	result = exp(lgamma(x + 5)) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4));
	return result;
}

inline float beta(float m, float n) {
	return (tgamma(m) * tgamma(n) / tgamma(m + n));
}

template<concepts::float_family T>
inline T sinc(const T& v) {
	return ite(v == T(0), T(1), sin(v) / v);
}

inline float factorial(uint n) {
	return tgamma(float(n + 1));
}

// Bessel functions of the first kind, of order nu

// This is a good rational approximation for non-negative integer orders, where roughly |x|<nu. Not supported for |nu|>=7.
// ! Numerically unstable for large x.
inline float cyl_bessel_j_saa(int nu, float x) {
	if (nu >= 7)
		return 0.0f;
	const float x2 = sqr(x);
	const float x4 = sqr(x2);
	const float x6 = x2 * x4;
	const float x8 = sqr(x4);

	const float s = 6.0f * pow(x / 2.0f, (float)nu);
	const float a = 1.0f / (6.0f * factorial(nu));
	const float b = -x2 / (24.0f * factorial(nu + 1));
	const float c1 = 11.0f * x8 - 864.0f * (6 + nu) * (x6 - 12.0f * (5.0f + nu) * (3.0f * x4 + 32.0f * (4 + nu) * (-2.0f * x2 + 27.0f * (3 + nu))));
	const float c = c1 / float(factorial(6 + nu)) * x4 * 5.652695401144601e-10f;
	return s * (a + b + c);
}
// This is a good approximation for x>>|nu|
inline float cyl_bessel_j_laa(int nu, float x) {
	return sqrt((2.0f / pi) / x) * cos(x - nu * (pi / 2.0f) - pi / 4.0f);
}
// Interpolate between the two
inline float cyl_bessel_j(int nu, float x) {
	const float t = nu > 0 || (-nu) % 2 == 0 ? 1.0f : -1.0f;
	nu = abs(nu);

	float cutoff = float(nu) + 1.0f;
	float interp = cutoff / 10.0f;
	float laa = cyl_bessel_j_laa(nu, x);
	return (nu >= 7 || abs(x) >= cutoff + interp) ?
			   laa * t :
			   lerp(cyl_bessel_j_saa(nu, x), laa, max(.0f, min(1.0f, 0.5f * (x - cutoff - interp) / interp))) * t;
}

}// namespace luisa::shader