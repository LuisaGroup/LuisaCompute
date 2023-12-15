#pragma once
#include "type_traits.hpp"

namespace luisa::shader {

template<typename ET, typename T = ET>
union [[swizzle]] Swizzle
{
    [[bypass]] operator T&();
    [[bypass]] T& operator()();

    union U{
        ET EMIT_ERROR;
    } SHOULD_NEVER_SCAN_THIS;
};

template<typename ET, typename T = vec<ET, 2>>
union [[swizzle]] Swizzle2
{
    [[bypass]] operator T&();
    [[bypass]] T& operator()();
    
    union U{
        ET EMIT_ERROR;
    } SHOULD_NEVER_SCAN_THIS;
};

template<typename ET, typename T = vec<ET, 3>>
union [[swizzle]] Swizzle3
{
    [[bypass]] operator T&();
    [[bypass]] T& operator()();
    
    union U{
        ET EMIT_ERROR;
    } SHOULD_NEVER_SCAN_THIS;
};

template<typename ET, typename T = vec<ET, 4>>
union [[swizzle]] Swizzle4
{
    [[bypass]] operator T&();
    [[bypass]] T& operator()();

    union U{
        ET EMIT_ERROR;
    } SHOULD_NEVER_SCAN_THIS;
};

template<typename T, uint64 N>
struct [[builtin("vec")]] vec {
    [[ignore]] vec() noexcept = default;
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == N)
    [[ignore]] explicit vec(Args &&...args);
private:
    T v[N];
};

template<typename T>
struct alignas(8) [[builtin("vec")]] vec<T, 2> {
    [[ignore]] vec() noexcept = default;
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == 2)
    [[ignore]] explicit vec(Args &&...args);

    [[bypass]] union 
    {
        #include "swizzle/swizzle2.inl"
        T zz_V[2];
    };
};

template<typename T>
struct alignas(16) [[builtin("vec")]] vec<T, 3> {
    [[ignore]] vec() noexcept = default;
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == 3)
    [[ignore]] explicit vec(Args &&...args);

    [[bypass]] union 
    {
        #include "swizzle/swizzle3.inl"
        T zz_V[3];
    };
};

template<typename T>
struct alignas(16) [[builtin("vec")]] vec<T, 4> {
    [[ignore]] vec() noexcept = default;
    template<typename... Args>
        requires(sum_dim<0ull, Args...>() == 4)
    [[ignore]] explicit vec(Args &&...args);

    [[bypass]] union 
    {
        #include "swizzle/swizzle4.inl"
        T zz_V[4];
    };
};

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
// using double2 = vec<double, 2>;
// using double3 = vec<double, 3>;
// using double4 = vec<double, 4>;
using int2 = vec<int32, 2>;
using int3 = vec<int32, 3>;
using int4 = vec<int32, 4>;
using uint2 = vec<uint32, 2>;
using uint3 = vec<uint32, 3>;
using uint4 = vec<uint32, 4>;
using half2 = vec<half, 2>;
using half3 = vec<half, 3>;
using half4 = vec<half, 4>;
using bool2 = vec<bool, 2>;
using bool3 = vec<bool, 3>;
using bool4 = vec<bool, 4>;

}; // namespace luisa::shader