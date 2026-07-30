#define LUISA_COMPUTE_FALLBACK_DEVICE_LIB
#include "../fallback_device_api.h"

extern "C" {// wrappers

#define LUISA_FALLBACK_WRAPPER __attribute__((visibility("hidden"))) __attribute__((used)) __attribute__((always_inline))
#define LUISA_FALLBACK_INTERNAL __attribute__((visibility("hidden"))) __attribute__((always_inline)) inline

using llvm_uint2 = uint __attribute__((ext_vector_type(2)));
using llvm_uint3 = uint __attribute__((ext_vector_type(3)));
using llvm_uint4 = uint __attribute__((ext_vector_type(4)));

using llvm_float2 = float __attribute__((ext_vector_type(2)));
using llvm_float3 = float __attribute__((ext_vector_type(3)));
using llvm_float4 = float __attribute__((ext_vector_type(4)));

static_assert(alignof(llvm_float2) == alignof(float2) && sizeof(llvm_float2) == sizeof(float2));
static_assert(alignof(llvm_float3) == alignof(float3) && sizeof(llvm_float3) == sizeof(float3));
static_assert(alignof(llvm_float4) == alignof(float4) && sizeof(llvm_float4) == sizeof(float4));

struct llvm_float2x2 {
    llvm_float2 cols[2];
};

struct llvm_float3x3 {
    llvm_float3 cols[3];
};

struct llvm_float4x4 {
    llvm_float4 cols[4];
};

static_assert(alignof(llvm_float2x2) == alignof(float2x2) && sizeof(llvm_float2x2) == sizeof(float2x2));
static_assert(alignof(llvm_float3x3) == alignof(float3x3) && sizeof(llvm_float3x3) == sizeof(float3x3));
static_assert(alignof(llvm_float4x4) == alignof(float4x4) && sizeof(llvm_float4x4) == sizeof(float4x4));

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix2d_mul_vector(const float2x2 *pm, const float2 *pv, float2 *out) noexcept {
    auto m = *reinterpret_cast<const llvm_float2x2 *>(pm);
    auto v = *reinterpret_cast<const llvm_float2 *>(pv);
    auto r = m.cols[0] * v.x + m.cols[1] * v.y;
    *reinterpret_cast<llvm_float2 *>(out) = r;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix2d_mul_matrix(const float2x2 *pm, const float2x2 *pn, float2x2 *out) noexcept {
    luisa_fallback_wrapper_matrix2d_mul_vector(pm, &pn->cols[0], &out->cols[0]);
    luisa_fallback_wrapper_matrix2d_mul_vector(pm, &pn->cols[1], &out->cols[1]);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix3d_mul_vector(const float3x3 *pm, const float3 *pv, float3 *out) noexcept {
    auto m = *reinterpret_cast<const llvm_float3x3 *>(pm);
    auto v = *reinterpret_cast<const llvm_float3 *>(pv);
    auto r = m.cols[0] * v.x + m.cols[1] * v.y + m.cols[2] * v.z;
    *reinterpret_cast<llvm_float3 *>(out) = r;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix3d_mul_matrix(const float3x3 *pm, const float3x3 *pn, float3x3 *out) noexcept {
    luisa_fallback_wrapper_matrix3d_mul_vector(pm, &pn->cols[0], &out->cols[0]);
    luisa_fallback_wrapper_matrix3d_mul_vector(pm, &pn->cols[1], &out->cols[1]);
    luisa_fallback_wrapper_matrix3d_mul_vector(pm, &pn->cols[2], &out->cols[2]);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix4d_mul_vector(const float4x4 *pm, const float4 *pv, float4 *out) noexcept {
    auto m = *reinterpret_cast<const llvm_float4x4 *>(pm);
    auto v = *reinterpret_cast<const llvm_float4 *>(pv);
    auto r = m.cols[0] * v.x + m.cols[1] * v.y + m.cols[2] * v.z + m.cols[3] * v.w;
    *reinterpret_cast<llvm_float4 *>(out) = r;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix4d_mul_matrix(const float4x4 *pm, const float4x4 *pn, float4x4 *out) noexcept {
    luisa_fallback_wrapper_matrix4d_mul_vector(pm, &pn->cols[0], &out->cols[0]);
    luisa_fallback_wrapper_matrix4d_mul_vector(pm, &pn->cols[1], &out->cols[1]);
    luisa_fallback_wrapper_matrix4d_mul_vector(pm, &pn->cols[2], &out->cols[2]);
    luisa_fallback_wrapper_matrix4d_mul_vector(pm, &pn->cols[3], &out->cols[3]);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix2d_transpose(const float2x2 *pm, float2x2 *p_out) noexcept {
    auto m = *reinterpret_cast<const llvm_float2x2 *>(pm);
    auto out = reinterpret_cast<llvm_float2x2 *>(p_out);
    out->cols[0] = {m.cols[0].x, m.cols[1].x};
    out->cols[1] = {m.cols[0].y, m.cols[1].y};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix3d_transpose(const float3x3 *pm, float3x3 *p_out) noexcept {
    auto m = *reinterpret_cast<const llvm_float3x3 *>(pm);
    auto out = reinterpret_cast<llvm_float3x3 *>(p_out);
    out->cols[0] = {m.cols[0].x, m.cols[1].x, m.cols[2].x};
    out->cols[1] = {m.cols[0].y, m.cols[1].y, m.cols[2].y};
    out->cols[2] = {m.cols[0].z, m.cols[1].z, m.cols[2].z};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix4d_transpose(const float4x4 *pm, float4x4 *p_out) noexcept {
    auto m = *reinterpret_cast<const llvm_float4x4 *>(pm);
    auto out = reinterpret_cast<llvm_float4x4 *>(p_out);
    out->cols[0] = {m.cols[0].x, m.cols[1].x, m.cols[2].x, m.cols[3].x};
    out->cols[1] = {m.cols[0].y, m.cols[1].y, m.cols[2].y, m.cols[3].y};
    out->cols[2] = {m.cols[0].z, m.cols[1].z, m.cols[2].z, m.cols[3].z};
    out->cols[3] = {m.cols[0].w, m.cols[1].w, m.cols[2].w, m.cols[3].w};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_vector2d_outer_product(const float2 *p_lhs, const float2 *p_rhs, float2x2 *p_out) noexcept {
    auto lhs = *reinterpret_cast<const llvm_float2 *>(p_lhs);
    auto rhs = *reinterpret_cast<const llvm_float2 *>(p_rhs);
    auto out = reinterpret_cast<llvm_float2x2 *>(p_out);
    out->cols[0] = lhs * rhs.x;
    out->cols[1] = lhs * rhs.y;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_vector3d_outer_product(const float3 *p_lhs, const float3 *p_rhs, float3x3 *p_out) noexcept {
    auto lhs = *reinterpret_cast<const llvm_float3 *>(p_lhs);
    auto rhs = *reinterpret_cast<const llvm_float3 *>(p_rhs);
    auto out = reinterpret_cast<llvm_float3x3 *>(p_out);
    out->cols[0] = lhs * rhs.x;
    out->cols[1] = lhs * rhs.y;
    out->cols[2] = lhs * rhs.z;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_vector4d_outer_product(const float4 *p_lhs, const float4 *p_rhs, float4x4 *p_out) noexcept {
    auto lhs = *reinterpret_cast<const llvm_float4 *>(p_lhs);
    auto rhs = *reinterpret_cast<const llvm_float4 *>(p_rhs);
    auto out = reinterpret_cast<llvm_float4x4 *>(p_out);
    out->cols[0] = lhs * rhs.x;
    out->cols[1] = lhs * rhs.y;
    out->cols[2] = lhs * rhs.z;
    out->cols[3] = lhs * rhs.w;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix2d_outer_product(const float2x2 *p_lhs, const float2x2 *p_rhs, float2x2 *p_out) noexcept {
    float2x2 rhs_T;
    luisa_fallback_wrapper_matrix2d_transpose(p_rhs, &rhs_T);
    luisa_fallback_wrapper_matrix2d_mul_matrix(p_lhs, &rhs_T, p_out);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix3d_outer_product(const float3x3 *p_lhs, const float3x3 *p_rhs, float3x3 *p_out) noexcept {
    float3x3 rhs_T;
    luisa_fallback_wrapper_matrix3d_transpose(p_rhs, &rhs_T);
    luisa_fallback_wrapper_matrix3d_mul_matrix(p_lhs, &rhs_T, p_out);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix4d_outer_product(const float4x4 *p_lhs, const float4x4 *p_rhs, float4x4 *p_out) noexcept {
    float4x4 rhs_T;
    luisa_fallback_wrapper_matrix4d_transpose(p_rhs, &rhs_T);
    luisa_fallback_wrapper_matrix4d_mul_matrix(p_lhs, &rhs_T, p_out);
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_matrix2d_determinant(const float2x2 *pm) noexcept {
    auto m = *reinterpret_cast<const llvm_float2x2 *>(pm);
    return m.cols[0].x * m.cols[1].y - m.cols[1].x * m.cols[0].y;
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_matrix3d_determinant(const float3x3 *pm) noexcept {
    auto m = *reinterpret_cast<const llvm_float3x3 *>(pm);
    return m.cols[0].x * (m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z) +
           m.cols[1].x * (m.cols[2].y * m.cols[0].z - m.cols[0].y * m.cols[2].z) +
           m.cols[2].x * (m.cols[0].y * m.cols[1].z - m.cols[1].y * m.cols[0].z);
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_matrix4d_determinant(const float4x4 *pm) noexcept {
    auto m = *reinterpret_cast<const llvm_float4x4 *>(pm);
    const auto coef00 = m.cols[2].z * m.cols[3].w - m.cols[3].z * m.cols[2].w;
    const auto coef02 = m.cols[1].z * m.cols[3].w - m.cols[3].z * m.cols[1].w;
    const auto coef03 = m.cols[1].z * m.cols[2].w - m.cols[2].z * m.cols[1].w;
    const auto coef04 = m.cols[2].y * m.cols[3].w - m.cols[3].y * m.cols[2].w;
    const auto coef06 = m.cols[1].y * m.cols[3].w - m.cols[3].y * m.cols[1].w;
    const auto coef07 = m.cols[1].y * m.cols[2].w - m.cols[2].y * m.cols[1].w;
    const auto coef08 = m.cols[2].y * m.cols[3].z - m.cols[3].y * m.cols[2].z;
    const auto coef10 = m.cols[1].y * m.cols[3].z - m.cols[3].y * m.cols[1].z;
    const auto coef11 = m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z;
    const auto coef12 = m.cols[2].x * m.cols[3].w - m.cols[3].x * m.cols[2].w;
    const auto coef14 = m.cols[1].x * m.cols[3].w - m.cols[3].x * m.cols[1].w;
    const auto coef15 = m.cols[1].x * m.cols[2].w - m.cols[2].x * m.cols[1].w;
    const auto coef16 = m.cols[2].x * m.cols[3].z - m.cols[3].x * m.cols[2].z;
    const auto coef18 = m.cols[1].x * m.cols[3].z - m.cols[3].x * m.cols[1].z;
    const auto coef19 = m.cols[1].x * m.cols[2].z - m.cols[2].x * m.cols[1].z;
    const auto coef20 = m.cols[2].x * m.cols[3].y - m.cols[3].x * m.cols[2].y;
    const auto coef22 = m.cols[1].x * m.cols[3].y - m.cols[3].x * m.cols[1].y;
    const auto coef23 = m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y;
    const auto fac0 = llvm_float4{coef00, coef00, coef02, coef03};
    const auto fac1 = llvm_float4{coef04, coef04, coef06, coef07};
    const auto fac2 = llvm_float4{coef08, coef08, coef10, coef11};
    const auto fac3 = llvm_float4{coef12, coef12, coef14, coef15};
    const auto fac4 = llvm_float4{coef16, coef16, coef18, coef19};
    const auto fac5 = llvm_float4{coef20, coef20, coef22, coef23};
    const auto Vec0 = llvm_float4{m.cols[1].x, m.cols[0].x, m.cols[0].x, m.cols[0].x};
    const auto Vec1 = llvm_float4{m.cols[1].y, m.cols[0].y, m.cols[0].y, m.cols[0].y};
    const auto Vec2 = llvm_float4{m.cols[1].z, m.cols[0].z, m.cols[0].z, m.cols[0].z};
    const auto Vec3 = llvm_float4{m.cols[1].w, m.cols[0].w, m.cols[0].w, m.cols[0].w};
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    constexpr auto sign_a = llvm_float4{+1.0f, -1.0f, +1.0f, -1.0f};
    constexpr auto sign_b = llvm_float4{-1.0f, +1.0f, -1.0f, +1.0f};
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m.cols[0] * llvm_float4{inv_0.x, inv_1.x, inv_2.x, inv_3.x};
    return dot0.x + dot0.y + dot0.z + dot0.w;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix2d_inverse(const float2x2 *pm, float2x2 *out) noexcept {
    auto m = *reinterpret_cast<const llvm_float2x2 *>(pm);
    const auto one_over_determinant = 1.0f / (m.cols[0].x * m.cols[1].y - m.cols[1].x * m.cols[0].y);
    out->cols[0].x = +m.cols[1].y * one_over_determinant;
    out->cols[0].y = -m.cols[0].y * one_over_determinant;
    out->cols[1].x = -m.cols[1].x * one_over_determinant;
    out->cols[1].y = +m.cols[0].x * one_over_determinant;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix3d_inverse(const float3x3 *pm, float3x3 *out) noexcept {
    auto m = *reinterpret_cast<const llvm_float3x3 *>(pm);
    const auto determinant = m.cols[0].x * (m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z) +
                             m.cols[1].x * (m.cols[2].y * m.cols[0].z - m.cols[0].y * m.cols[2].z) +
                             m.cols[2].x * (m.cols[0].y * m.cols[1].z - m.cols[1].y * m.cols[0].z);
    const auto one_over_determinant = 1.0f / determinant;
    out->cols[0].x = (m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z) * one_over_determinant;
    out->cols[0].y = (m.cols[2].y * m.cols[0].z - m.cols[0].y * m.cols[2].z) * one_over_determinant;
    out->cols[0].z = (m.cols[0].y * m.cols[1].z - m.cols[1].y * m.cols[0].z) * one_over_determinant;
    out->cols[1].x = (m.cols[2].x * m.cols[1].z - m.cols[1].x * m.cols[2].z) * one_over_determinant;
    out->cols[1].y = (m.cols[0].x * m.cols[2].z - m.cols[2].x * m.cols[0].z) * one_over_determinant;
    out->cols[1].z = (m.cols[1].x * m.cols[0].z - m.cols[0].x * m.cols[1].z) * one_over_determinant;
    out->cols[2].x = (m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y) * one_over_determinant;
    out->cols[2].y = (m.cols[2].x * m.cols[0].y - m.cols[0].x * m.cols[2].y) * one_over_determinant;
    out->cols[2].z = (m.cols[0].x * m.cols[1].y - m.cols[1].x * m.cols[0].y) * one_over_determinant;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_matrix4d_inverse(const float4x4 *pm, float4x4 *out) noexcept {
    auto m = *reinterpret_cast<const llvm_float4x4 *>(pm);
    const auto coef00 = m.cols[2].z * m.cols[3].w - m.cols[3].z * m.cols[2].w;
    const auto coef02 = m.cols[1].z * m.cols[3].w - m.cols[3].z * m.cols[1].w;
    const auto coef03 = m.cols[1].z * m.cols[2].w - m.cols[2].z * m.cols[1].w;
    const auto coef04 = m.cols[2].y * m.cols[3].w - m.cols[3].y * m.cols[2].w;
    const auto coef06 = m.cols[1].y * m.cols[3].w - m.cols[3].y * m.cols[1].w;
    const auto coef07 = m.cols[1].y * m.cols[2].w - m.cols[2].y * m.cols[1].w;
    const auto coef08 = m.cols[2].y * m.cols[3].z - m.cols[3].y * m.cols[2].z;
    const auto coef10 = m.cols[1].y * m.cols[3].z - m.cols[3].y * m.cols[1].z;
    const auto coef11 = m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z;
    const auto coef12 = m.cols[2].x * m.cols[3].w - m.cols[3].x * m.cols[2].w;
    const auto coef14 = m.cols[1].x * m.cols[3].w - m.cols[3].x * m.cols[1].w;
    const auto coef15 = m.cols[1].x * m.cols[2].w - m.cols[2].x * m.cols[1].w;
    const auto coef16 = m.cols[2].x * m.cols[3].z - m.cols[3].x * m.cols[2].z;
    const auto coef18 = m.cols[1].x * m.cols[3].z - m.cols[3].x * m.cols[1].z;
    const auto coef19 = m.cols[1].x * m.cols[2].z - m.cols[2].x * m.cols[1].z;
    const auto coef20 = m.cols[2].x * m.cols[3].y - m.cols[3].x * m.cols[2].y;
    const auto coef22 = m.cols[1].x * m.cols[3].y - m.cols[3].x * m.cols[1].y;
    const auto coef23 = m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y;
    const auto fac0 = llvm_float4{coef00, coef00, coef02, coef03};
    const auto fac1 = llvm_float4{coef04, coef04, coef06, coef07};
    const auto fac2 = llvm_float4{coef08, coef08, coef10, coef11};
    const auto fac3 = llvm_float4{coef12, coef12, coef14, coef15};
    const auto fac4 = llvm_float4{coef16, coef16, coef18, coef19};
    const auto fac5 = llvm_float4{coef20, coef20, coef22, coef23};
    const auto Vec0 = llvm_float4{m.cols[1].x, m.cols[0].x, m.cols[0].x, m.cols[0].x};
    const auto Vec1 = llvm_float4{m.cols[1].y, m.cols[0].y, m.cols[0].y, m.cols[0].y};
    const auto Vec2 = llvm_float4{m.cols[1].z, m.cols[0].z, m.cols[0].z, m.cols[0].z};
    const auto Vec3 = llvm_float4{m.cols[1].w, m.cols[0].w, m.cols[0].w, m.cols[0].w};
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    constexpr auto sign_a = llvm_float4{+1.0f, -1.0f, +1.0f, -1.0f};
    constexpr auto sign_b = llvm_float4{-1.0f, +1.0f, -1.0f, +1.0f};
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m.cols[0] * llvm_float4{inv_0.x, inv_1.x, inv_2.x, inv_3.x};
    const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    const auto one_over_determinant = 1.0f / dot1;
    auto result = reinterpret_cast<llvm_float4x4 *>(out);
    result->cols[0] = inv_0 * one_over_determinant;
    result->cols[1] = inv_1 * one_over_determinant;
    result->cols[2] = inv_2 * one_over_determinant;
    result->cols[3] = inv_3 * one_over_determinant;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture2d_read_int(const TextureView *handle, const uint2 *coord, int4 *out) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    *out = luisa_fallback_texture2d_read_int(packed_view->data, packed_view->extra, coord->x, coord->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture2d_read_uint(const TextureView *handle, const uint2 *coord, uint4 *out) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    *out = luisa_fallback_texture2d_read_uint(packed_view->data, packed_view->extra, coord->x, coord->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture2d_read_float(const TextureView *handle, const uint2 *coord, float4 *out) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    *out = luisa_fallback_texture2d_read_float(packed_view->data, packed_view->extra, coord->x, coord->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture2d_write_float(const TextureView *handle, const uint2 *coord, const float4 *value) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    luisa_fallback_texture2d_write_float(packed_view->data, packed_view->extra, coord->x, coord->y, *value);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture2d_write_uint(const TextureView *handle, const uint2 *coord, const uint4 *value) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    luisa_fallback_texture2d_write_uint(packed_view->data, packed_view->extra, coord->x, coord->y, *value);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture2d_write_int(const TextureView *handle, const uint2 *coord, const int4 *value) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    luisa_fallback_texture2d_write_int(packed_view->data, packed_view->extra, coord->x, coord->y, *value);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture3d_read_int(const TextureView *handle, const uint3 *coord, int4 *out) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    *out = luisa_fallback_texture3d_read_int(packed_view->data, packed_view->extra, coord->x, coord->y, coord->z);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture3d_read_uint(const TextureView *handle, const uint3 *coord, uint4 *out) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    *out = luisa_fallback_texture3d_read_uint(packed_view->data, packed_view->extra, coord->x, coord->y, coord->z);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture3d_read_float(const TextureView *handle, const uint3 *coord, float4 *out) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    *out = luisa_fallback_texture3d_read_float(packed_view->data, packed_view->extra, coord->x, coord->y, coord->z);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture3d_write_float(const TextureView *handle, const uint3 *coord, const float4 *value) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    luisa_fallback_texture3d_write_float(packed_view->data, packed_view->extra, coord->x, coord->y, coord->z, *value);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture3d_write_uint(const TextureView *handle, const uint3 *coord, const uint4 *value) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    luisa_fallback_texture3d_write_uint(packed_view->data, packed_view->extra, coord->x, coord->y, coord->z, *value);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture3d_write_int(const TextureView *handle, const uint3 *coord, const int4 *value) noexcept {
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    luisa_fallback_texture3d_write_int(packed_view->data, packed_view->extra, coord->x, coord->y, coord->z, *value);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture2d_size(const TextureView *handle, uint2 *out) noexcept {
    *out = {handle->_width, handle->_height};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture3d_size(const TextureView *handle, uint3 *out) noexcept {
    *out = {handle->_width, handle->_height, handle->_depth};
}

#define LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(bindless_handle, slot_index) \
    auto texture = bindless_handle->slots[slot_index].tex2d; \
    auto sampler = (bindless_handle->slots[slot_index]._compressed_buffer_size_sampler_2d_sampler_3d >> 4u) & 0x0fu;

#define LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(bindless_handle, slot_index) \
    auto texture = bindless_handle->slots[slot_index].tex3d; \
    auto sampler = bindless_handle->slots[slot_index]._compressed_buffer_size_sampler_2d_sampler_3d & 0x0fu;

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample(const BindlessArrayView *handle, uint slot_index, const float2 *uv, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    *out = luisa_fallback_bindless_texture2d_sample(texture, sampler, uv->x, uv->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_level(const BindlessArrayView *handle, uint slot_index, const float2 *uv, float level, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    *out = luisa_fallback_bindless_texture2d_sample_level(texture, sampler, uv->x, uv->y, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_grad(const BindlessArrayView *handle, uint slot_index, const float2 *uv, const float2 *ddx, const float2 *ddy, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    *out = luisa_fallback_bindless_texture2d_sample_grad(texture, sampler, uv->x, uv->y, ddx->x, ddx->y, ddy->x, ddy->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_grad_level(const BindlessArrayView *handle, uint slot_index, const float2 *uv, const float2 *ddx, const float2 *ddy, float level, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    *out = luisa_fallback_bindless_texture2d_sample_grad_level(texture, sampler, uv->x, uv->y, ddx->x, ddx->y, ddy->x, ddy->y, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_sample(const BindlessArrayView *handle, uint slot_index, const float3 *uvw, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(handle, slot_index)
    *out = luisa_fallback_bindless_texture3d_sample(texture, sampler, uvw->x, uvw->y, uvw->z);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_sample_level(const BindlessArrayView *handle, uint slot_index, const float3 *uvw, float level, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(handle, slot_index)
    *out = luisa_fallback_bindless_texture3d_sample_level(texture, sampler, uvw->x, uvw->y, uvw->z, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_sample_grad(const BindlessArrayView *handle, uint slot_index, const float3 *uvw, const float3 *ddx, const float3 *ddy, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(handle, slot_index)
    *out = luisa_fallback_bindless_texture3d_sample_grad(texture, sampler, uvw->x, uvw->y, uvw->z, ddx->x, ddx->y, ddx->z, ddy->x, ddy->y, ddy->z);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_sample_grad_level(const BindlessArrayView *handle, uint slot_index, const float3 *uvw, const float3 *ddx, const float3 *ddy, float level, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(handle, slot_index)
    *out = luisa_fallback_bindless_texture3d_sample_grad_level(texture, sampler, uvw->x, uvw->y, uvw->z, ddx->x, ddx->y, ddx->z, ddy->x, ddy->y, ddy->z, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_read(const BindlessArrayView *handle, uint slot_index, const uint2 *coord, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    *out = luisa_fallback_bindless_texture2d_read(texture, coord->x, coord->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_read(const BindlessArrayView *handle, uint slot_index, const uint3 *coord, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(handle, slot_index)
    *out = luisa_fallback_bindless_texture3d_read(texture, coord->x, coord->y, coord->z);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_read_level(const BindlessArrayView *handle, uint slot_index, const uint2 *coord, uint level, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    *out = luisa_fallback_bindless_texture2d_read_level(texture, coord->x, coord->y, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_read_level(const BindlessArrayView *handle, uint slot_index, const uint3 *coord, uint level, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(handle, slot_index)
    *out = luisa_fallback_bindless_texture3d_read_level(texture, coord->x, coord->y, coord->z, level);
}

struct alignas(16) TextureHeader {
    void *data;
    unsigned short size[3];
};

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_size_level(const BindlessArrayView *handle, uint slot_index, uint level, uint2 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    auto t = reinterpret_cast<const TextureHeader *>(texture);
    auto width = static_cast<uint>(t->size[0]) >> level;
    auto height = static_cast<uint>(t->size[1]) >> level;
    *out = {width > 0u ? width : 1u, height > 0u ? height : 1u};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_size_level(const BindlessArrayView *handle, uint slot_index, uint level, uint3 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(handle, slot_index)
    auto t = reinterpret_cast<const TextureHeader *>(texture);
    auto width = static_cast<uint>(t->size[0]) >> level;
    auto height = static_cast<uint>(t->size[1]) >> level;
    auto depth = static_cast<uint>(t->size[2]) >> level;
    *out = {width > 0u ? width : 1u, height > 0u ? height : 1u, depth > 0u ? depth : 1u};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_size(const BindlessArrayView *handle, uint slot_index, uint2 *out) noexcept {
    return luisa_fallback_wrapper_bindless_texture2d_size_level(handle, slot_index, 0u, out);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_size(const BindlessArrayView *handle, uint slot_index, uint3 *out) noexcept {
    return luisa_fallback_wrapper_bindless_texture3d_size_level(handle, slot_index, 0u, out);
}

/* Ray structure for a single ray */
struct alignas(16) EmbreeRay {
    llvm_float4 org_tnear;          // Ray origin and near t value
    llvm_float4 dir_time;           // Ray direction and far t value

    struct {
        float tfar;
        uint mask;
        uint id;
        uint flags;
    } extra;
};

/* Hit structure for a single ray */
struct alignas(16) EmbreeHit {
    float Ng_x;        // x coordinate of geometry normal
    float Ng_y;        // y coordinate of geometry normal
    float Ng_z;        // z coordinate of geometry normal
    float u;           // barycentric u coordinate of hit
    float v;           // barycentric v coordinate of hit
    uint primID;       // primitive ID
    uint geomID;       // geometry ID
    uint instID[1];    // instance ID
};

// Combined ray/hit structure for a single ray
// Note: embree 3 and 4 might have different size of EmbreeHit, so we add
//       the padding field to make sure out-of-bound access is safe.
struct alignas(16) EmbreeRayHit {
    EmbreeRay ray;
    EmbreeHit hit;
    llvm_uint4 padding;
};

struct alignas(16) LC_RayQueryObject {
    AccelView accel;
    RayQueryCandidate candidate;
    EmbreeRayHit ray_hit;
    Ray world_ray;
};

LUISA_FALLBACK_INTERNAL void luisa_fallback_create_embree_ray(EmbreeRay *embree_ray, const Ray *ray, float time, uint mask) noexcept {
    auto org_tnear = reinterpret_cast<const llvm_float4 *>(&ray->origin);
    auto dir_tfar = reinterpret_cast<const llvm_float4 *>(&ray->direction);
    embree_ray->org_tnear = *org_tnear;
    embree_ray->dir_time = {dir_tfar->x, dir_tfar->y, dir_tfar->z, time};
    embree_ray->extra = {ray->t_max, mask, 0u, 0u};
}

LUISA_FALLBACK_INTERNAL void luisa_fallback_create_embree_hit(EmbreeHit *embree_hit) noexcept {
    auto v = reinterpret_cast<llvm_uint4 *>(embree_hit);
    v[0] = {0u, 0u, 0u, 0u};
    v[1] = {~0u, ~0u, ~0u, ~0u};
}

LUISA_FALLBACK_INTERNAL void luisa_fallback_decode_surface_hit(SurfaceHit *surface_hit, const EmbreeRayHit *ray_hit) noexcept {
    *surface_hit = {.inst = ray_hit->hit.instID[0],
                    .prim = ray_hit->hit.primID,
                    .bary = {ray_hit->hit.u, ray_hit->hit.v},
                    .committed_ray_t = ray_hit->ray.extra.tfar};
}

LUISA_FALLBACK_INTERNAL void luisa_fallback_decode_committed_hit(CommittedHit *committed_hit, const EmbreeHit *hit) noexcept {
    *committed_hit = {.inst = hit->instID[0],
                      .prim = hit->primID,
                      .bary = {hit->u, hit->v},
                      .hit_type = hit->instID[0] == ~0u ? HitType::Miss :
                                  hit->u < 0.f          ? HitType::Procedural :
                                                          HitType::Surface,
                      .committed_ray_t = hit->Ng_z};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_traverse_motion(const AccelView *handle, const Ray *ray, float time, uint mask, LC_RayQueryObject *out) noexcept {
    out->accel = *handle;
    out->world_ray = *ray;
    luisa_fallback_create_embree_ray(&out->ray_hit.ray, ray, time, mask);
    luisa_fallback_create_embree_hit(&out->ray_hit.hit);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_traverse(const AccelView *handle, const Ray *ray, uint mask, LC_RayQueryObject *out) noexcept {
    luisa_fallback_wrapper_accel_traverse_motion(handle, ray, 0.f, mask, out);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_world_space_ray(const LC_RayQueryObject *q, Ray *out) noexcept {
    *out = q->world_ray;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_procedural_candidate_hit(const LC_RayQueryObject *q, AABBHit *out) noexcept {
    *out = {.inst = q->candidate.inst,
            .prim = q->candidate.prim};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_surface_candidate_hit(const LC_RayQueryObject *q, SurfaceHit *out) noexcept {
    *out = {.inst = q->candidate.inst,
            .prim = q->candidate.prim,
            .bary = q->candidate.bary,
            .committed_ray_t = q->candidate.t};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_committed_hit(const LC_RayQueryObject *q, CommittedHit *out) noexcept {
    luisa_fallback_decode_committed_hit(out, &q->ray_hit.hit);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_commit_surface_hit(LC_RayQueryObject *q) noexcept {
    q->candidate.committed = true;
    q->world_ray.t_max = q->candidate.t;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_commit_procedural_hit(LC_RayQueryObject *q, float t) noexcept {
    q->candidate.t = t;
    if (t >= q->world_ray.t_min &&
        t <= q->world_ray.t_max) {
        q->candidate.committed = true;
        q->world_ray.t_max = t;
    }
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_terminate(LC_RayQueryObject *q) noexcept {
    q->candidate.terminated = true;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_pipeline_all(LC_RayQueryObject *q, const void *capture, RayQueryOnSurfaceFunc *on_surface, RayQueryOnProceduralFunc *on_procedural) noexcept {
    luisa_fallback_ray_query_pipeline_all(q, capture, on_surface, on_procedural);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_pipeline_any(LC_RayQueryObject *q, const void *capture, RayQueryOnSurfaceFunc *on_surface, RayQueryOnProceduralFunc *on_procedural) noexcept {
    luisa_fallback_ray_query_pipeline_any(q, capture, on_surface, on_procedural);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_trace_closest_motion(const AccelView *handle, const Ray *ray, float time, uint mask, SurfaceHit *out) noexcept {
    EmbreeRayHit ray_hit;
    luisa_fallback_create_embree_ray(&ray_hit.ray, ray, time, mask);
    luisa_fallback_create_embree_hit(&ray_hit.hit);
    luisa_fallback_accel_trace_closest(handle->embree_scene, &ray_hit);
    luisa_fallback_decode_surface_hit(out, &ray_hit);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_trace_any_motion(const AccelView *handle, const Ray *ray, float time, uint mask, bool *out) noexcept {
    EmbreeRay ray_;
    luisa_fallback_create_embree_ray(&ray_, ray, time, mask);
    luisa_fallback_accel_trace_any(handle->embree_scene, &ray_);
    *out = ray_.extra.tfar < 0.f;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_trace_closest(const AccelView *handle, const Ray *ray, uint mask, SurfaceHit *out) noexcept {
    luisa_fallback_wrapper_accel_trace_closest_motion(handle, ray, 0.f, mask, out);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_trace_any(const AccelView *handle, const Ray *ray, uint mask, bool *out) noexcept {
    luisa_fallback_wrapper_accel_trace_any_motion(handle, ray, 0.f, mask, out);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_instance_transform(const AccelView *handle, uint instance_id, float4x4 *out) noexcept {
    auto rows = reinterpret_cast<const llvm_float4 *>(handle->instances[instance_id].affine);
    auto r0 = rows[0];
    auto r1 = rows[1];
    auto r2 = rows[2];
    auto m = reinterpret_cast<llvm_float4x4 *>(out);
    m->cols[0] = {r0.x, r1.x, r2.x, 0.f};
    m->cols[1] = {r0.y, r1.y, r2.y, 0.f};
    m->cols[2] = {r0.z, r1.z, r2.z, 0.f};
    m->cols[3] = {r0.w, r1.w, r2.w, 1.f};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_instance_user_id(const AccelView *handle, uint instance_id, uint *out) noexcept {
    *out = handle->instances[instance_id].user_id;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_instance_visibility_mask(const AccelView *handle, uint instance_id, uint *out) noexcept {
    *out = handle->instances[instance_id].mask;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_set_instance_transform(AccelView *handle, uint instance_id, const float4x4 *transform) noexcept {
    auto &instance = handle->instances[instance_id];
    auto rows = reinterpret_cast<llvm_float4 *>(instance.affine);
    auto m = *reinterpret_cast<const llvm_float4x4 *>(transform);
    rows[0] = {m.cols[0].x, m.cols[1].x, m.cols[2].x, m.cols[3].x};
    rows[1] = {m.cols[0].y, m.cols[1].y, m.cols[2].y, m.cols[3].y};
    rows[2] = {m.cols[0].z, m.cols[1].z, m.cols[2].z, m.cols[3].z};
    instance.dirty = true;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_set_instance_visibility_mask(AccelView *handle, uint instance_id, uint mask) noexcept {
    auto &instance = handle->instances[instance_id];
    instance.mask = mask;
    instance.dirty = true;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_set_instance_user_id(AccelView *handle, uint instance_id, uint user_id) noexcept {
    auto &instance = handle->instances[instance_id];
    instance.user_id = user_id;
    instance.dirty = true;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_set_instance_opacity(AccelView *handle, uint instance_id, bool opacity) noexcept {
    auto &instance = handle->instances[instance_id];
    instance.opaque = opacity;
    instance.dirty = true;
}

#define LUISA_FALLBACK_ATOMIC_MEMORY_ORDER __ATOMIC_RELAXED

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_exchange_int(int *a, int b) noexcept {
    return __atomic_exchange_n(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_exchange_uint(uint *a, uint b) noexcept {
    return __atomic_exchange_n(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_exchange_long(int64_t *a, int64_t b) noexcept {
    return __atomic_exchange_n(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_exchange_ulong(uint64_t *a, uint64_t b) noexcept {
    return __atomic_exchange_n(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_exchange_float(float *a, float b) noexcept {
    return __builtin_bit_cast(float, __atomic_exchange_n(reinterpret_cast<int *>(a), __builtin_bit_cast(int, b), LUISA_FALLBACK_ATOMIC_MEMORY_ORDER));
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_compare_exchange_int(int *a, int b, int c) noexcept {
    __atomic_compare_exchange_n(a, &b, c, true, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
    return b;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_compare_exchange_uint(uint *a, uint b, uint c) noexcept {
    __atomic_compare_exchange_n(a, &b, c, true, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
    return b;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_compare_exchange_long(int64_t *a, int64_t b, int64_t c) noexcept {
    __atomic_compare_exchange_n(a, &b, c, true, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
    return b;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_compare_exchange_ulong(uint64_t *a, uint64_t b, uint64_t c) noexcept {
    __atomic_compare_exchange_n(a, &b, c, true, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
    return b;
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_compare_exchange_float(float *a, float b, float c) noexcept {
    __atomic_compare_exchange_n(reinterpret_cast<int *>(a), reinterpret_cast<int *>(&b), __builtin_bit_cast(int, c), true, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
    return b;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_add_int(int *a, int b) noexcept {
    return __atomic_fetch_add(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_add_uint(uint *a, uint b) noexcept {
    return __atomic_fetch_add(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_add_long(int64_t *a, int64_t b) noexcept {
    return __atomic_fetch_add(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_add_ulong(uint64_t *a, uint64_t b) noexcept {
    return __atomic_fetch_add(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_fetch_add_float(float *a, float b) noexcept {
    return __atomic_fetch_add(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_sub_int(int *a, int b) noexcept {
    return __atomic_fetch_sub(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_sub_uint(uint *a, uint b) noexcept {
    return __atomic_fetch_sub(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_sub_long(int64_t *a, int64_t b) noexcept {
    return __atomic_fetch_sub(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_sub_ulong(uint64_t *a, uint64_t b) noexcept {
    return __atomic_fetch_sub(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_fetch_sub_float(float *a, float b) noexcept {
    return __atomic_fetch_sub(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_and_int(int *a, int b) noexcept {
    return __atomic_fetch_and(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_and_uint(uint *a, uint b) noexcept {
    return __atomic_fetch_and(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_and_long(int64_t *a, int64_t b) noexcept {
    return __atomic_fetch_and(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_and_ulong(uint64_t *a, uint64_t b) noexcept {
    return __atomic_fetch_and(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_or_int(int *a, int b) noexcept {
    return __atomic_fetch_or(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_or_uint(uint *a, uint b) noexcept {
    return __atomic_fetch_or(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_or_long(int64_t *a, int64_t b) noexcept {
    return __atomic_fetch_or(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_or_ulong(uint64_t *a, uint64_t b) noexcept {
    return __atomic_fetch_or(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_xor_int(int *a, int b) noexcept {
    return __atomic_fetch_xor(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_xor_uint(uint *a, uint b) noexcept {
    return __atomic_fetch_xor(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_xor_long(int64_t *a, int64_t b) noexcept {
    return __atomic_fetch_xor(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_xor_ulong(uint64_t *a, uint64_t b) noexcept {
    return __atomic_fetch_xor(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_min_int(int *a, int b) noexcept {
    return __atomic_fetch_min(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_min_uint(uint *a, uint b) noexcept {
    return __atomic_fetch_min(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_min_long(int64_t *a, int64_t b) noexcept {
    return __atomic_fetch_min(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_min_ulong(uint64_t *a, uint64_t b) noexcept {
    return __atomic_fetch_min(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_fetch_min_float(float *a, float b) noexcept {
    return __atomic_fetch_min(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_max_int(int *a, int b) noexcept {
    return __atomic_fetch_max(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_max_uint(uint *a, uint b) noexcept {
    return __atomic_fetch_max(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_max_long(int64_t *a, int64_t b) noexcept {
    return __atomic_fetch_max(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_max_ulong(uint64_t *a, uint64_t b) noexcept {
    return __atomic_fetch_max(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_fetch_max_float(float *a, float b) noexcept {
    return __atomic_fetch_max(a, b, LUISA_FALLBACK_ATOMIC_MEMORY_ORDER);
}

// For shared memory atomics, we don't actually need to use atomic operations as one thread block maps to one CPU thread.
LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_exchange_int_smem(int *a, int b) noexcept {
    auto tmp = *a;
    *a = b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_exchange_uint_smem(uint *a, uint b) noexcept {
    auto tmp = *a;
    *a = b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_exchange_long_smem(int64_t *a, int64_t b) noexcept {
    auto tmp = *a;
    *a = b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_exchange_ulong_smem(uint64_t *a, uint64_t b) noexcept {
    auto tmp = *a;
    *a = b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_exchange_float_smem(float *a, float b) noexcept {
    auto tmp = *a;
    *a = b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_compare_exchange_int_smem(int *a, int b, int c) noexcept {
    auto tmp = *a;
    if (*a == b) { *a = c; }
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_compare_exchange_uint_smem(uint *a, uint b, uint c) noexcept {
    auto tmp = *a;
    if (*a == b) { *a = c; }
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_compare_exchange_long_smem(int64_t *a, int64_t b, int64_t c) noexcept {
    auto tmp = *a;
    if (*a == b) { *a = c; }
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_compare_exchange_ulong_smem(uint64_t *a, uint64_t b, uint64_t c) noexcept {
    auto tmp = *a;
    if (*a == b) { *a = c; }
    return tmp;
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_compare_exchange_float_smem(float *a, float b, float c) noexcept {
    auto tmp = *a;
    if (*a == b) { *a = c; }
    return tmp;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_add_int_smem(int *a, int b) noexcept {
    auto tmp = *a;
    *a += b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_add_uint_smem(uint *a, uint b) noexcept {
    auto tmp = *a;
    *a += b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_add_long_smem(int64_t *a, int64_t b) noexcept {
    auto tmp = *a;
    *a += b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_add_ulong_smem(uint64_t *a, uint64_t b) noexcept {
    auto tmp = *a;
    *a += b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_fetch_add_float_smem(float *a, float b) noexcept {
    auto tmp = *a;
    *a += b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_sub_int_smem(int *a, int b) noexcept {
    auto tmp = *a;
    *a -= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_sub_uint_smem(uint *a, uint b) noexcept {
    auto tmp = *a;
    *a -= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_sub_long_smem(int64_t *a, int64_t b) noexcept {
    auto tmp = *a;
    *a -= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_sub_ulong_smem(uint64_t *a, uint64_t b) noexcept {
    auto tmp = *a;
    *a -= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_fetch_sub_float_smem(float *a, float b) noexcept {
    auto tmp = *a;
    *a -= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_and_int_smem(int *a, int b) noexcept {
    auto tmp = *a;
    *a &= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_and_uint_smem(uint *a, uint b) noexcept {
    auto tmp = *a;
    *a &= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_and_long_smem(int64_t *a, int64_t b) noexcept {
    auto tmp = *a;
    *a &= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_and_ulong_smem(uint64_t *a, uint64_t b) noexcept {
    auto tmp = *a;
    *a &= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_or_int_smem(int *a, int b) noexcept {
    auto tmp = *a;
    *a |= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_or_uint_smem(uint *a, uint b) noexcept {
    auto tmp = *a;
    *a |= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_or_long_smem(int64_t *a, int64_t b) noexcept {
    auto tmp = *a;
    *a |= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_or_ulong_smem(uint64_t *a, uint64_t b) noexcept {
    auto tmp = *a;
    *a |= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_xor_int_smem(int *a, int b) noexcept {
    auto tmp = *a;
    *a ^= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_xor_uint_smem(uint *a, uint b) noexcept {
    auto tmp = *a;
    *a ^= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_xor_long_smem(int64_t *a, int64_t b) noexcept {
    auto tmp = *a;
    *a ^= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_xor_ulong_smem(uint64_t *a, uint64_t b) noexcept {
    auto tmp = *a;
    *a ^= b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_min_int_smem(int *a, int b) noexcept {
    auto tmp = *a;
    *a = *a < b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_min_uint_smem(uint *a, uint b) noexcept {
    auto tmp = *a;
    *a = *a < b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_min_long_smem(int64_t *a, int64_t b) noexcept {
    auto tmp = *a;
    *a = *a < b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_min_ulong_smem(uint64_t *a, uint64_t b) noexcept {
    auto tmp = *a;
    *a = *a < b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_fetch_min_float_smem(float *a, float b) noexcept {
    auto tmp = *a;
    *a = *a < b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int luisa_fallback_wrapper_atomic_fetch_max_int_smem(int *a, int b) noexcept {
    auto tmp = *a;
    *a = *a > b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint luisa_fallback_wrapper_atomic_fetch_max_uint_smem(uint *a, uint b) noexcept {
    auto tmp = *a;
    *a = *a > b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER int64_t luisa_fallback_wrapper_atomic_fetch_max_long_smem(int64_t *a, int64_t b) noexcept {
    auto tmp = *a;
    *a = *a > b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER uint64_t luisa_fallback_wrapper_atomic_fetch_max_ulong_smem(uint64_t *a, uint64_t b) noexcept {
    auto tmp = *a;
    *a = *a > b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER float luisa_fallback_wrapper_atomic_fetch_max_float_smem(float *a, float b) noexcept {
    auto tmp = *a;
    *a = *a > b ? *a : b;
    return tmp;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_debug_break_trap() noexcept {
    __builtin_debugtrap();
}

}
