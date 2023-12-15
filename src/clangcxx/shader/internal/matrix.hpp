#pragma once
#include "vec.hpp"

namespace luisa::shader {

template<>
struct [[builtin("matrix")]] matrix<2> {
    [[ignore]] matrix() noexcept = default;
    [[ignore]] matrix(float2 col0, float2 col1);
    [[ignore]] matrix(float m00, float m01, float m10, float m11);
    [[ignore]] matrix(matrix<3> float3x3);
    [[ignore]] matrix(matrix<4> float4x4);
private:
    float2 v[2];
};
template<>
struct [[builtin("matrix")]] matrix<3> {
    [[ignore]] matrix() noexcept = default;
    [[ignore]] matrix(float3 col0, float3 col1, float3 col2);
    [[ignore]] matrix(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22);
    [[ignore]] matrix(matrix<2> float2x2);
    [[ignore]] matrix(matrix<4> float4x4);
private:
    float3 v[3];
};
template<>
struct alignas(16) [[builtin("matrix")]] matrix<4> {
    [[ignore]] matrix() noexcept = default;
    [[ignore]] matrix(float4 col0, float4 col1, float4 col2, float4 col3);
    [[ignore]] matrix(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33);
    [[ignore]] matrix(matrix<2> float2x2);
    [[ignore]] matrix(matrix<3> float3x3);
private:
    float4 v[4];
};
using float2x2 = matrix<2>;
using float3x3 = matrix<3>;
using float4x4 = matrix<4>;

} // namespace luisa::shader