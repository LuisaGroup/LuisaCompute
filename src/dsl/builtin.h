//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute {

template<typename Dest, typename Src>
[[nodiscard]] inline auto cast(detail::Expr<Src> s) noexcept { return s.template cast<Dest>(); }

template<typename Dest, typename Src>
[[nodiscard]] inline auto as(detail::Expr<Src> s) noexcept { return s.template as<Dest>(); }

[[nodiscard]] inline auto thread_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->thread_id()};
}

[[nodiscard]] inline auto thread_x() noexcept {
    return thread_id().x;
}

[[nodiscard]] inline auto thread_y() noexcept {
    return thread_id().y;
}

[[nodiscard]] inline auto thread_z() noexcept {
    return thread_id().z;
}

[[nodiscard]] inline auto block_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->block_id()};
}

[[nodiscard]] inline auto block_x() noexcept {
    return block_id().x;
}

[[nodiscard]] inline auto block_y() noexcept {
    return block_id().y;
}

[[nodiscard]] inline auto block_z() noexcept {
    return block_id().z;
}

[[nodiscard]] inline auto dispatch_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->dispatch_id()};
}

[[nodiscard]] inline auto dispatch_x() noexcept {
    return dispatch_id().x;
}

[[nodiscard]] inline auto dispatch_y() noexcept {
    return dispatch_id().y;
}

[[nodiscard]] inline auto dispatch_z() noexcept {
    return dispatch_id().z;
}

[[nodiscard]] inline auto launch_size() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->launch_size()};
}

[[nodiscard]] inline auto block_size() noexcept {
    return FunctionBuilder::current()->block_size();
}

inline void set_block_size(uint x, uint y = 1u, uint z = 1u) noexcept {
    FunctionBuilder::current()->set_block_size(
        uint3{std::max(x, 1u), std::max(y, 1u), std::max(z, 1u)});
}

template<typename... T>
[[nodiscard]] inline auto multiple(T &&...v) noexcept {
    return std::make_tuple(detail::Expr{v}...);
}

// math functions

// atomic functions

// sync functions

// make_vector functions

namespace detail {

template<typename T>
[[nodiscard]] constexpr auto make_vector_tag() noexcept {
    if constexpr (std::is_same_v<T, bool2>) {
        return CallOp::MAKE_BOOL2;
    } else if constexpr (std::is_same_v<T, bool3>) {
        return CallOp::MAKE_BOOL3;
    } else if constexpr (std::is_same_v<T, bool4>) {
        return CallOp::MAKE_BOOL4;
    } else if constexpr (std::is_same_v<T, int2>) {
        return CallOp::MAKE_INT2;
    } else if constexpr (std::is_same_v<T, int3>) {
        return CallOp::MAKE_INT3;
    } else if constexpr (std::is_same_v<T, int4>) {
        return CallOp::MAKE_INT4;
    } else if constexpr (std::is_same_v<T, uint2>) {
        return CallOp::MAKE_UINT2;
    } else if constexpr (std::is_same_v<T, uint3>) {
        return CallOp::MAKE_UINT3;
    } else if constexpr (std::is_same_v<T, uint4>) {
        return CallOp::MAKE_UINT4;
    } else if constexpr (std::is_same_v<T, float2>) {
        return CallOp::MAKE_FLOAT2;
    } else if constexpr (std::is_same_v<T, float3>) {
        return CallOp::MAKE_FLOAT3;
    } else if constexpr (std::is_same_v<T, float4>) {
        return CallOp::MAKE_FLOAT4;
    } else {
        static_assert(always_false_v<T>);
    }
}
template<typename T>
struct is_vector {
    static constexpr bool value = false;
    using ScalarType = T;
};

template<typename T, size_t N>
struct is_vector<Vector<T, N>> {
    static constexpr bool value =
        std::is_same_v<T, float> || std::is_same_v<T, uint> || std::is_same_v<T, int32_t> || std::is_same_v<T, float>;
    using ScalarType = T;
};

template<typename A, typename B>
struct vector_check {
    static constexpr bool is_dim_same = false;
    static constexpr bool is_type_same = false;
};

template<typename A, typename B, size_t NA, size_t NB>
struct vector_check<Vector<A, NA>, Vector<B, NB>> {
    static constexpr bool is_dim_same = NA == NB;
    static constexpr bool is_type_same = std::is_same_v<A, B>;
};

template<typename T>
constexpr bool is_vector_t = is_vector<T>::value;
template<typename T>
constexpr bool is_matrix_t = std::is_same_v<T, float3x3> || std::is_same_v<T, float4x4>;
template<typename T>
constexpr bool is_scalar_t = !(is_vector_t<T> || is_matrix_t<T>);
template<typename T>
using scalar_type = typename is_vector<T>::ScalarType;
template<typename A, typename B>
constexpr bool is_vector_dim_same = vector_check<A, B>::is_dim_same;
template<typename A, typename B>
constexpr bool is_vector_eletype_same = vector_check<A, B>::is_type_same;
}// namespace detail

template<typename T>
[[nodiscard]] inline auto make_vector(detail::Expr<T> x, detail::Expr<T> y) noexcept {
    using V = Vector<T, 2>;
    return detail::Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression()})};
}

template<typename T>
[[nodiscard]] inline auto make_vector(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) noexcept {
    using V = Vector<T, 3>;
    return detail::Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression(), z.expression()})};
}

template<typename T>
[[nodiscard]] inline auto make_vector(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z, detail::Expr<T> w) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression(), z.expression(), w.expression()})};
}

template<typename T>
[[nodiscard]] inline auto all(detail::Expr<T> x) {
    static_assert(detail::is_vector_t<T>, "must be vector!");
    return detail::Expr<bool>{
        FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ALL, {x.expression()})};
}

template<typename T>
[[nodiscard]] inline auto any(detail::Expr<T> x) {
    static_assert(detail::is_vector_t<T>, "must be vector!");
    return detail::Expr<bool>{
        FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ANY, {x.expression()})};
}

template<typename T>
[[nodiscard]] inline auto none(detail::Expr<T> x) {
    static_assert(detail::is_vector_t<T>, "must be vector!");
    return detail::Expr<bool>{
        FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::NONE, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto select(detail::Expr<T> falseValue, detail::Expr<T> trueValue, detail::Expr<T> t) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SELECT, {falseValue.expression(), trueValue.expression(), t.expression()})};
}
template<typename T>
[[nodiscard]] inline auto clamp(detail::Expr<T> value, detail::Expr<T> minValue, detail::Expr<T> maxValue) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CLAMP, {value.expression(), minValue.expression(), maxValue.expression()})};
}

template<typename T>
[[nodiscard]] inline auto lerp(detail::Expr<T> left, detail::Expr<T> right, detail::Expr<T> t) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::LERP, {left.expression(), right.expression(), t.expression()})};
}

template<typename T>
[[nodiscard]] inline auto saturate(detail::Expr<T> value) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SATURATE, {value.expression()})};
}
template<typename T>
[[nodiscard]] inline auto sign(detail::Expr<T> value) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SIGN, {value.expression()})};
}
template<typename T>
[[nodiscard]] inline auto step(detail::Expr<T> falseValue, detail::Expr<T> trueValue, detail::Expr<T> t) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::STEP, {falseValue.expression(), trueValue.expression(), t.expression()})};
}
template<typename T>
[[nodiscard]] inline auto smoothstep(detail::Expr<T> left, detail::Expr<T> right, detail::Expr<T> t) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SMOOTHSTEP, {left.expression(), right.expression(), t.expression()})};
}

template<typename T>
[[nodiscard]] inline auto abs(detail::Expr<T> x) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ABS, {x.expression()})};
}

template<typename T>
[[nodiscard]] inline auto min(detail::Expr<T> x, detail::Expr<T> y) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<typename T>
[[nodiscard]] inline auto max(detail::Expr<T> x, detail::Expr<T> y) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<typename T>
[[nodiscard]] inline auto min(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MIN3, {x.expression(), y.expression(), z.expression()})};
}

template<typename T>
[[nodiscard]] inline auto max(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MAX3, {x.expression(), y.expression(), z.expression()})};
}
template<typename T>
[[nodiscard]] inline auto clz(detail::Expr<T> x) {
    static_assert(detail::is_scalar_t<T>, "must be scalar!");
    static_assert(std::is_same_v<detail::scalar_type<T>, uint> || std::is_same_v<detail::scalar_type<T>, int>, "must be integer!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CLZ, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto ctz(detail::Expr<T> x) {
    static_assert(detail::is_scalar_t<T>, "must be scalar!");
    static_assert(std::is_same_v<detail::scalar_type<T>, uint> || std::is_same_v<detail::scalar_type<T>, int>, "must be integer!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CTZ, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto popcount(detail::Expr<T> x) {
    static_assert(detail::is_scalar_t<T>, "must be scalar!");
    static_assert(std::is_same_v<detail::scalar_type<T>, uint> || std::is_same_v<detail::scalar_type<T>, int>, "must be integer!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::POPCOUNT, {x.expression()})};
}

template<typename T>
[[nodiscard]] inline auto isinf(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<bool>{
        FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ISINF, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto isnan(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<bool>{
        FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ISNAN, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto acos(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ACOS, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto acosh(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ACOSH, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto asin(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ASIN, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto asinh(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ASINH, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto atan(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATAN, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto atan2(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATAN2, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto atanh(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATANH, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto cos(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::COS, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto cosh(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::COSH, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto sin(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SIN, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto sinh(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SINH, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto tan(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TAN, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto tanh(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TANH, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto exp(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::EXP, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto exp2(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::EXP2, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto exp10(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::EXP10, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto log(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::LOG, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto log2(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::LOG2, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto log10(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::LOG10, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto sqrt(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SQRT, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto rsqrt(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::RSQRT, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto ceil(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CEIL, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto floor(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FLOOR, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto fract(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FRACT, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto trunc(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TRUNC, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto round(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ROUND, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto fmod(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FMOD, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto degrees(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::DEGREES, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto radians(detail::Expr<T> x) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::RADIANS, {x.expression()})};
}
template<typename T>
[[nodiscard]] inline auto fma(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FMA, {x.expression(), y.expression(), z.expression()})};
}
template<typename T>
[[nodiscard]] inline auto copysign(detail::Expr<T> x, detail::Expr<T> y) {
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::COPYSIGN, {x.expression(), y.expression()})};
}

template<typename T>
[[nodiscard]] inline auto cross(detail::Expr<T> x, detail::Expr<T> y) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CROSS, {x.expression(), y.expression()})};
}
template<typename T>
[[nodiscard]] inline auto dot(detail::Expr<T> x, detail::Expr<T> y) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::DOT, {x.expression(), y.expression()})};
}
template<typename T>
[[nodiscard]] inline auto distance(detail::Expr<T> x, detail::Expr<T> y) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::DISTANCE, {x.expression(), y.expression()})};
}
template<typename T>
[[nodiscard]] inline auto distance_squared(detail::Expr<T> x, detail::Expr<T> y) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::DISTANCE_SQUARED, {x.expression(), y.expression()})};
}
template<typename T>
[[nodiscard]] inline auto faceforward(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) {
    static_assert(std::is_same_v<detail::scalar_type<T>, float>, "must be float!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FACEFORWARD, {x.expression(), y.expression(), z.expression()})};
}

template<typename T>
[[nodiscard]] inline auto determinant(detail::Expr<T> mat) {
    static_assert(detail::is_matrix_t<T>, "must be matrix!");
    return detail::Expr<float>{
        FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DETERMINANT, {mat.expression()})};
}

template<typename T>
[[nodiscard]] inline auto transpose(detail::Expr<T> mat) {
    static_assert(detail::is_matrix_t<T>, "must be matrix!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TRNASPOSE, {mat.expression()})};
}
template<typename T>
[[nodiscard]] inline auto inverse(detail::Expr<T> mat) {
    static_assert(detail::is_matrix_t<T>, "must be matrix!");
    return detail::Expr<T>{
        FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::INVERSE, {mat.expression()})};
}

inline void group_memory_barrier() {
    FunctionBuilder::current()->call(
        CallOp::GROUP_MEMORY_BARRIER, {});
}
inline void all_memory_barrier() {
    FunctionBuilder::current()->call(
        CallOp::ALL_MEMORY_BARRIER, {});
}
inline void device_memory_barrier() {
    FunctionBuilder::current()->call(
        CallOp::DEVICE_MEMORY_BARRIER, {});
}
//atomic todo

/*
    return detail::Expr<T> {
        FunctionBuilder::current->call(

        )
    }
}

*/

#define LUISA_MAKE_VECTOR(type)                                                                                \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<type> s) noexcept {                                  \
        return make_vector(s, s);                                                                              \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<type> x, detail::Expr<type> y) noexcept {            \
        return make_vector(x, y);                                                                              \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<type, 3>> v) noexcept {                       \
        return make_vector(v.x, v.y);                                                                          \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<type, 4>> v) noexcept {                       \
        return make_vector(v.x, v.y);                                                                          \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<T, 2>> v) noexcept {                          \
        return make_vector(cast<type>(v.x), cast<type>(v.y));                                                  \
    }                                                                                                          \
                                                                                                               \
    [[nodiscard]] inline auto make_##type##3(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<type> z) noexcept {                           \
        return make_vector(x, y, z);                                                                           \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<type> s) noexcept {                                  \
        return make_vector(s, s, s);                                                                           \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<type, 2>> v, detail::Expr<type> z) noexcept { \
        return make_vector(v.x, v.y, z);                                                                       \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<type> x, detail::Expr<Vector<type, 2>> v) noexcept { \
        return make_vector(x, v.x, v.y);                                                                       \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<type, 4>> v) noexcept {                       \
        return make_vector(v.x, v.y, v.z);                                                                     \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<T, 3>> v) noexcept {                          \
        return make_vector(cast<type>(v.x), cast<type>(v.y), cast<type>(v.z));                                 \
    }                                                                                                          \
                                                                                                               \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<type> z, detail::Expr<type> w) noexcept {     \
        return make_vector(x, y, z, w);                                                                        \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 2>> v, detail::Expr<type> z, detail::Expr<type> w) noexcept {                \
        return make_vector(v.x, v.y, z, w);                                                                    \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<Vector<type, 2>> yz, detail::Expr<type> w) noexcept {               \
        return make_vector(x, yz.x, yz.y, w);                                                                  \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<Vector<type, 2>> zw) noexcept {               \
        return make_vector(x, y, zw.x, zw.y);                                                                  \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 2>> xy, detail::Expr<Vector<type, 2>> zw) noexcept {                         \
        return make_vector(xy.x, xy.y, zw.x, zw.y);                                                            \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 3>> xyz, detail::Expr<type> w) noexcept {                                    \
        return make_vector(xyz.x, xyz.y, xyz.z, w);                                                            \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<Vector<type, 3>> yzw) noexcept {                                    \
        return make_vector(x, yzw.x, yzw.y, yzw.z);                                                            \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##4(detail::Expr<Vector<T, 4>> v) noexcept {                          \
        return make_vector(cast<type>(v.x), cast<type>(v.y), cast<type>(v.z), cast<type>(v.w));                \
    }
LUISA_MAKE_VECTOR(bool)
LUISA_MAKE_VECTOR(int)
LUISA_MAKE_VECTOR(uint)
LUISA_MAKE_VECTOR(float)
#undef LUISA_MAKE_VECTOR

}// namespace luisa::compute
