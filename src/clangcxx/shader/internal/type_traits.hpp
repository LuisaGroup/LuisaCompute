#pragma once
#include "attributes.hpp"

namespace luisa::shader {

using int32 = int;
using int64 = long long;
using uint32 = unsigned int;
using uint64 = unsigned long long;

template<typename T>
struct remove_cvref {
    using type = T;
};
template<typename T>
struct remove_cvref<T &> {
    using type = T;
};
template<typename T>
struct remove_cvref<T const> {
    using type = T;
};
template<typename T>
struct remove_cvref<T volatile> {
    using type = T;
};
template<typename T>
struct remove_cvref<T &&> {
    using type = T;
};

template<typename T, uint64 N>
struct [[type_ex("vec", N)]] vec;

template<typename T>
struct [[ignore]] vec_dim {
    static constexpr uint64 value = 1;
};

template<typename T, uint64 N>
struct [[ignore]] vec_dim<vec<T, N>> {
    static constexpr uint64 value = N;
};
template<typename T>
[[ignore]] constexpr uint64 vec_dim_v = vec_dim<typename remove_cvref<T>::type>::value;

template<uint64 dim, typename T, typename... Ts>
[[ignore]] consteval uint64 sum_dim() {
    constexpr auto new_dim = dim + vec_dim_v<T>;
    if constexpr (sizeof...(Ts) == 0) {
        return new_dim;
    } else {
        return sum_dim<new_dim, Ts...>();
    }
}

template<typename T, uint64 N>
struct [[type_ex("vec", N)]] vec {
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == N)
    [[ignore]] explicit vec(Args &&...args) {}
};

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
using double2 = vec<double, 2>;
using double3 = vec<double, 3>;
using double4 = vec<double, 4>;
using int2 = vec<int32, 2>;
using int3 = vec<int32, 3>;
using int4 = vec<int32, 4>;
using uint2 = vec<uint32, 2>;
using uint3 = vec<uint32, 3>;
using uint4 = vec<uint32, 4>;

}