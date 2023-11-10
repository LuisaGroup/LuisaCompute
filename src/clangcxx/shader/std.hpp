#pragma once

#define type(name) clang::annotate("luisa-shader", "type", (name), 1)
#define type_ex(name, ...) clang::annotate("luisa-shader", "type", (name), __VA_ARGS__)
#define builtin(name) clang::annotate("luisa-shader", "builtin", (name), 1)
#define kernel_1d(x) clang::annotate("luisa-shader", "kernel_1d", (x))
#define kernel_2d(x, y) clang::annotate("luisa-shader", "kernel_2d", (x), (y))
#define kernel_3d(x, y, z) clang::annotate("luisa-shader", "kernel_3d", (x), (y), (z))

namespace luisa::shader
{
using int32 = int;
using int64 = long long;
using uint32 = unsigned int;
using uint64 = unsigned long long;

template <typename T, uint64 N>
struct [[type_ex("vec", N)]] vec
{
    template <typename U, uint64 X, typename... Args>
    explicit vec(vec<U, X> v, Args&&...args) 
    {
        static_assert((sizeof...(args) + X) == N, "!");
    }
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

[[builtin("dispatch_id")]] extern uint3 dispatch_id();
[[builtin("sin")]] extern float sin(float rad);
[[builtin("cos")]] extern float cos(float rad);

template<typename Type = void, uint32 CacheFlags = 0>
struct [[type("Buffer")]] Buffer
{
    [[builtin("BUFFER_READ")]] Type load(int3 loc);
    Type operator[](uint2 loc) { return load(int3(loc, 0)); };

    [[builtin("BUFFER_WRITE")]] void store(uint32 loc, Type value);
};

}