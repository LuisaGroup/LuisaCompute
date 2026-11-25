#define LUISA_COMPUTE_FALLBACK_DEVICE_LIB
#include "../fallback_device_api.h"

extern "C" {// wrappers

#define LUISA_FALLBACK_WRAPPER __attribute__((visibility("hidden"))) __attribute__((used)) __attribute__((always_inline))
#define LUISA_FALLBACK_INTERNAL __attribute__((visibility("hidden"))) __attribute__((always_inline)) inline

using llvm_uint2 = uint __attribute__((ext_vector_type(2)));
using llvm_uint3 = uint __attribute__((ext_vector_type(3)));
using llvm_uint4 = uint __attribute__((ext_vector_type(4)));

using llvm_uchar4 = uint8_t __attribute__((ext_vector_type(4)));
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
    const uint storage = handle->_storage;
    if ((storage == 2u) | (storage == 14u)) {
        const uint w = handle->_width;
        const uint h = handle->_height;
        const uint x = coord->x;
        const uint y = coord->y;
        if ((x >= w) | (y >= h)) {
            *reinterpret_cast<llvm_float4 *>(out) = {0.0f, 0.0f, 0.0f, 0.0f};
            return;
        }
        const size_t idx = static_cast<size_t>(y) * w + x;
        if (storage == 2u) {
            const auto *p = reinterpret_cast<const llvm_uchar4 *>(
                reinterpret_cast<const uint8_t *>(handle->_data) + (idx << 2u));
            *reinterpret_cast<llvm_float4 *>(out) =
                __builtin_convertvector(p[0], llvm_float4) * (1.0f / 255.0f);
        } else {
            const auto *p = reinterpret_cast<const llvm_float4 *>(handle->_data);
            *reinterpret_cast<llvm_float4 *>(out) = p[idx];
        }
        return;
    }
    auto packed_view = reinterpret_cast<const PackedTextureView *>(handle);
    *out = luisa_fallback_texture2d_read_float(packed_view->data, packed_view->extra, coord->x, coord->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_texture2d_write_float(const TextureView *handle, const uint2 *coord, const float4 *value) noexcept {
    const uint storage = handle->_storage;
    if ((storage == 2u) | (storage == 14u)) {
        const uint w = handle->_width;
        const uint h = handle->_height;
        const uint x = coord->x;
        const uint y = coord->y;
        if ((x >= w) | (y >= h)) { return; }
        const size_t idx = static_cast<size_t>(y) * w + x;
        const llvm_float4 v = *reinterpret_cast<const llvm_float4 *>(value);
        if (storage == 2u) {
            const bool ordered = (v.x == v.x) & (v.y == v.y) & (v.z == v.z) & (v.w == v.w);
            if (ordered) {
                const llvm_float4 zero{0.0f, 0.0f, 0.0f, 0.0f};
                const llvm_float4 one{1.0f, 1.0f, 1.0f, 1.0f};
                const llvm_float4 clamped_lo = v > zero ? v : zero;
                const llvm_float4 clamped = clamped_lo < one ? clamped_lo : one;
                const llvm_float4 scaled = clamped * 255.0f + 0.5f;
                auto *p = reinterpret_cast<llvm_uchar4 *>(
                    reinterpret_cast<uint8_t *>(handle->_data) + (idx << 2u));
                p[0] = __builtin_convertvector(scaled, llvm_uchar4);
                return;
            }
        } else {
            auto *p = reinterpret_cast<llvm_float4 *>(handle->_data);
            p[idx] = v;
            return;
        }
    }
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

#define LUISA_FALLBACK_BINDLESS_TEXTURE2D(bindless_handle, slot_index) \
    auto texture = bindless_handle->slots[slot_index].tex2d;

#define LUISA_FALLBACK_BINDLESS_TEXTURE3D(bindless_handle, slot_index) \
    auto texture = bindless_handle->slots[slot_index].tex3d;

#define LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(bindless_handle, slot_index) \
    LUISA_FALLBACK_BINDLESS_TEXTURE2D(bindless_handle, slot_index) \
    auto sampler = (bindless_handle->slots[slot_index]._compressed_buffer_size_sampler_2d_sampler_3d >> 4u) & 0x0fu;

#define LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE3D(bindless_handle, slot_index) \
    LUISA_FALLBACK_BINDLESS_TEXTURE3D(bindless_handle, slot_index) \
    auto sampler = bindless_handle->slots[slot_index]._compressed_buffer_size_sampler_2d_sampler_3d & 0x0fu;

[[nodiscard]] LUISA_FALLBACK_INTERNAL uint luisa_fallback_explicit_sampler(uint filter, uint address) noexcept {
    // Sampler::code() is a two-bit address lane below a two-bit filter lane.
    // Keep the operands dynamic: divergent shader lanes may legally select
    // different call-site samplers for the same bindless texture slot.
    return (filter << 2u) | address;
}

struct alignas(16) TextureHeader {
    const uint8_t *data;
    unsigned short size[3];
    unsigned short storage;
    uint8_t pixel_stride_shift;
    uint8_t mip_levels;
    uint8_t dimension;
    uint8_t is_external;
    uint32_t mip_offsets[3];
};
static_assert(sizeof(TextureHeader) == 32u);

[[nodiscard]] LUISA_FALLBACK_INTERNAL float luisa_fallback_tex_coord_point_1d(uint address, float u, float s) noexcept {
    constexpr float one_minus_eps = 0x1.fffffep-1f;
    switch (address) {
        case 0u: {
            const float c = u < 0.0f ? 0.0f : (u < one_minus_eps ? u : one_minus_eps);
            return c * s;
        }
        case 1u: {
            const float f = u - __builtin_floorf(u);
            return (f < one_minus_eps ? f : one_minus_eps) * s;
        }
        case 2u: {
            const float a = __builtin_fabsf(u);
            float m = a - 2.0f * __builtin_floorf(a * 0.5f);
            m = m < 1.0f ? m : (2.0f - m);
            return (m < one_minus_eps ? m : one_minus_eps) * s;
        }
        default:
            return ((u < 0.0f) | (u >= 1.0f)) ? 65536.0f : (u * s);
    }
}

LUISA_FALLBACK_INTERNAL void luisa_fallback_tex_coord_linear_1d(
    uint address, float u, uint size, uint &c0, uint &c1, float &t) noexcept {
    const float fs = static_cast<float>(size);
    const int isize = static_cast<int>(size);
    switch (address) {
        case 0u: {
            const float px = u * fs - 0.5f;
            if (px <= 0.0f) {
                c0 = 0u;
                c1 = 0u;
                t = 0.0f;
            } else if (px >= static_cast<float>(size - 1u)) {
                c0 = size - 1u;
                c1 = size - 1u;
                t = 0.0f;
            } else {
                const float fl = __builtin_floorf(px);
                const uint i0 = static_cast<uint>(fl);
                c0 = i0 < size ? i0 : (size - 1u);
                c1 = (i0 + 1u) < size ? (i0 + 1u) : (size - 1u);
                t = px - fl;
            }
            return;
        }
        case 1u: {
            float f = u - __builtin_floorf(u);
            if (!((f >= 0.0f) & (f < 1.0f))) { f = 0.0f; }
            const float px = f * fs - 0.5f;
            const float fl = __builtin_floorf(px);
            t = px - fl;
            const int i0 = static_cast<int>(fl);
            const int i1 = i0 + 1;
            const int w0 = i0 < 0 ? (isize - 1) : (i0 >= isize ? (isize - 1) : i0);
            const int w1 = i1 < 0 ? 0 : (i1 >= isize ? 0 : i1);
            c0 = static_cast<uint>(w0);
            c1 = static_cast<uint>(w1);
            return;
        }
        case 2u: {
            const float raw_a = __builtin_fabsf(u);
            float a = raw_a - 2.0f * __builtin_floorf(raw_a * 0.5f);
            if (!((a >= 0.0f) & (a < 2.0f))) { a = 0.0f; }
            const float folded = a < 1.0f ? a : (2.0f - a);
            const float px = folded * fs - 0.5f;
            const float fl = __builtin_floorf(px);
            t = px - fl;
            const int i0 = static_cast<int>(fl);
            const int i1 = i0 + 1;
            const int w0 = i0 < 0 ? 0 : (i0 >= isize ? (isize - 1) : i0);
            const int w1 = i1 < 0 ? 0 : (i1 >= isize ? (isize - 1) : i1);
            c0 = static_cast<uint>(w0);
            c1 = static_cast<uint>(w1);
            return;
        }
        default: {
            const float px = u * fs - 0.5f;
            if ((px <= -1.0f) | (px >= fs)) {
                c0 = size;
                c1 = size;
                t = 0.0f;
            } else {
                const float fl = __builtin_floorf(px);
                t = px - fl;
                const int i0 = static_cast<int>(fl);
                const int i1 = i0 + 1;
                c0 = ((i0 < 0) | (i0 >= isize)) ? size : static_cast<uint>(i0);
                c1 = ((i1 < 0) | (i1 >= isize)) ? size : static_cast<uint>(i1);
            }
            return;
        }
    }
}

[[nodiscard]] LUISA_FALLBACK_INTERNAL llvm_float4 luisa_fallback_tex2d_fetch_rgba(
    const TextureHeader *t, uint storage, uint w, uint h, uint x, uint y) noexcept {
    if ((x >= w) | (y >= h)) {
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }
    if (storage == 2u) {
        const auto *p = reinterpret_cast<const llvm_uchar4 *>(
            t->data + ((static_cast<size_t>(y) * w + x) << 2u));
        return __builtin_convertvector(p[0], llvm_float4) * (1.0f / 255.0f);
    }
    const auto *fraw = reinterpret_cast<const llvm_float4 *>(t->data);
    return fraw[static_cast<size_t>(y) * w + x];
}

static __attribute__((noinline)) bool luisa_fallback_tex2d_sample_level0_general(
    const TextureHeader *t, uint filter, uint address, float ux, float uy, float4 *out) noexcept {
    const uint w = t->size[0];
    const uint h = t->size[1];
    const uint storage = t->storage;
    if ((w == 0u) | (h == 0u)) { return false; }
    const uint ux_bits = __builtin_bit_cast(uint, ux);
    const uint uy_bits = __builtin_bit_cast(uint, uy);
    if (((ux_bits & 0x7f800000u) == 0x7f800000u) | ((uy_bits & 0x7f800000u) == 0x7f800000u)) {
        return false;
    }
    const float fw = static_cast<float>(w);
    const float fh = static_cast<float>(h);
    if (filter == 0u) {
        const uint cx = static_cast<uint>(luisa_fallback_tex_coord_point_1d(address, ux, fw));
        const uint cy = static_cast<uint>(luisa_fallback_tex_coord_point_1d(address, uy, fh));
        *reinterpret_cast<llvm_float4 *>(out) = luisa_fallback_tex2d_fetch_rgba(t, storage, w, h, cx, cy);
        return true;
    }
    uint c0x = 0u, c1x = 0u, c0y = 0u, c1y = 0u;
    float tx = 0.0f, ty = 0.0f;
    luisa_fallback_tex_coord_linear_1d(address, ux, w, c0x, c1x, tx);
    luisa_fallback_tex_coord_linear_1d(address, uy, h, c0y, c1y, ty);
    const llvm_float4 v00 = luisa_fallback_tex2d_fetch_rgba(t, storage, w, h, c0x, c0y);
    const llvm_float4 v01 = luisa_fallback_tex2d_fetch_rgba(t, storage, w, h, c1x, c0y);
    const llvm_float4 v10 = luisa_fallback_tex2d_fetch_rgba(t, storage, w, h, c0x, c1y);
    const llvm_float4 v11 = luisa_fallback_tex2d_fetch_rgba(t, storage, w, h, c1x, c1y);
    const llvm_float4 r0 = v00 + (v01 - v00) * tx;
    const llvm_float4 r1 = v10 + (v11 - v10) * tx;
    *reinterpret_cast<llvm_float4 *>(out) = r0 + (r1 - r0) * ty;
    return true;
}

LUISA_FALLBACK_INTERNAL bool luisa_fallback_try_tex2d_sample_level0_fast(
    const TextureHeader *t, uint filter, uint address, float ux, float uy, float4 *out) noexcept {
    const uint storage = t->storage;
    if ((storage != 2u) & (storage != 14u)) {
        return false;
    }
    const uint w = t->size[0];
    const uint h = t->size[1];
    if ((filter != 0u) & (address <= 1u) & (w >= 2u) & (h >= 2u)) {
        const uint ux_bits = __builtin_bit_cast(uint, ux);
        const uint uy_bits = __builtin_bit_cast(uint, uy);
        if (((ux_bits & 0x7f800000u) != 0x7f800000u) & ((uy_bits & 0x7f800000u) != 0x7f800000u)) {
            float u = ux;
            float v = uy;
            if (address == 1u) {
                u -= __builtin_floorf(u);
                v -= __builtin_floorf(v);
            }
            const float px = u * static_cast<float>(w) - 0.5f;
            const float py = v * static_cast<float>(h) - 0.5f;
            if ((px >= 0.0f) & (py >= 0.0f) &
                (px < static_cast<float>(w - 1u)) & (py < static_cast<float>(h - 1u))) {
                const uint x0 = static_cast<uint>(px);
                const uint y0 = static_cast<uint>(py);
                if ((x0 + 1u < w) & (y0 + 1u < h)) {
                    const float tx = px - static_cast<float>(x0);
                    const float ty = py - static_cast<float>(y0);
                    if (storage == 2u) {
                        const size_t stride = static_cast<size_t>(w) << 2u;
                        const auto *p0 = reinterpret_cast<const llvm_uchar4 *>(
                            t->data + static_cast<size_t>(y0) * stride + (static_cast<size_t>(x0) << 2u));
                        const auto *p1 = reinterpret_cast<const llvm_uchar4 *>(
                            reinterpret_cast<const uint8_t *>(p0) + stride);
                        const llvm_float4 f00 = __builtin_convertvector(p0[0], llvm_float4);
                        const llvm_float4 f01 = __builtin_convertvector(p0[1], llvm_float4);
                        const llvm_float4 f10 = __builtin_convertvector(p1[0], llvm_float4);
                        const llvm_float4 f11 = __builtin_convertvector(p1[1], llvm_float4);
                        const llvm_float4 r0 = f00 + (f01 - f00) * tx;
                        const llvm_float4 r1 = f10 + (f11 - f10) * tx;
                        *reinterpret_cast<llvm_float4 *>(out) = (r0 + (r1 - r0) * ty) * (1.0f / 255.0f);
                        return true;
                    }
                    const auto *fraw = reinterpret_cast<const llvm_float4 *>(t->data);
                    const size_t idx0 = static_cast<size_t>(y0) * w + x0;
                    const size_t idx1 = idx0 + w;
                    const llvm_float4 r0 = fraw[idx0] + (fraw[idx0 + 1u] - fraw[idx0]) * tx;
                    const llvm_float4 r1 = fraw[idx1] + (fraw[idx1 + 1u] - fraw[idx1]) * tx;
                    *reinterpret_cast<llvm_float4 *>(out) = r0 + (r1 - r0) * ty;
                    return true;
                }
            }
        }
    }
    return luisa_fallback_tex2d_sample_level0_general(t, filter, address, ux, uy, out);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample(const BindlessArrayView *handle, uint slot_index, const float2 *uv, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    const auto *t = reinterpret_cast<const TextureHeader *>(texture);
    if (luisa_fallback_try_tex2d_sample_level0_fast(t, sampler >> 2u, sampler & 0x03u, uv->x, uv->y, out)) { return; }
    *out = luisa_fallback_bindless_texture2d_sample(texture, sampler, uv->x, uv->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_level(const BindlessArrayView *handle, uint slot_index, const float2 *uv, float level, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    const auto *t = reinterpret_cast<const TextureHeader *>(texture);
    const uint filter = sampler >> 2u;
    if (((level <= 0.0f) | (t->mip_levels <= 1u)) &&
        luisa_fallback_try_tex2d_sample_level0_fast(t, filter, sampler & 0x03u, uv->x, uv->y, out)) {
        return;
    }
    *out = luisa_fallback_bindless_texture2d_sample_level(texture, sampler, uv->x, uv->y, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_grad(const BindlessArrayView *handle, uint slot_index, const float2 *uv, const float2 *ddx, const float2 *ddy, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    const auto *t = reinterpret_cast<const TextureHeader *>(texture);
    const uint filter = sampler >> 2u;
    if (((t->mip_levels <= 1u) | (filter <= 1u)) &&
        luisa_fallback_try_tex2d_sample_level0_fast(t, filter, sampler & 0x03u, uv->x, uv->y, out)) {
        return;
    }
    *out = luisa_fallback_bindless_texture2d_sample_grad(texture, sampler, uv->x, uv->y, ddx->x, ddx->y, ddy->x, ddy->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_grad_level(const BindlessArrayView *handle, uint slot_index, const float2 *uv, const float2 *ddx, const float2 *ddy, float level, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_DECODE_TEXTURE2D(handle, slot_index)
    const auto *t = reinterpret_cast<const TextureHeader *>(texture);
    const uint filter = sampler >> 2u;
    if (((level <= 0.0f) | (t->mip_levels <= 1u)) &&
        luisa_fallback_try_tex2d_sample_level0_fast(t, filter, sampler & 0x03u, uv->x, uv->y, out)) {
        return;
    }
    *out = luisa_fallback_bindless_texture2d_sample_grad_level(texture, sampler, uv->x, uv->y, ddx->x, ddx->y, ddy->x, ddy->y, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_sampler(const BindlessArrayView *handle, uint slot_index, const float2 *uv, uint filter, uint address, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_TEXTURE2D(handle, slot_index)
    const auto *t = reinterpret_cast<const TextureHeader *>(texture);
    if (luisa_fallback_try_tex2d_sample_level0_fast(t, filter, address, uv->x, uv->y, out)) { return; }
    auto sampler = luisa_fallback_explicit_sampler(filter, address);
    *out = luisa_fallback_bindless_texture2d_sample(texture, sampler, uv->x, uv->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_level_sampler(const BindlessArrayView *handle, uint slot_index, const float2 *uv, float level, uint filter, uint address, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_TEXTURE2D(handle, slot_index)
    const auto *t = reinterpret_cast<const TextureHeader *>(texture);
    if (((level <= 0.0f) | (t->mip_levels <= 1u)) &&
        luisa_fallback_try_tex2d_sample_level0_fast(t, filter, address, uv->x, uv->y, out)) {
        return;
    }
    auto sampler = luisa_fallback_explicit_sampler(filter, address);
    *out = luisa_fallback_bindless_texture2d_sample_level(texture, sampler, uv->x, uv->y, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_grad_sampler(const BindlessArrayView *handle, uint slot_index, const float2 *uv, const float2 *ddx, const float2 *ddy, uint filter, uint address, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_TEXTURE2D(handle, slot_index)
    const auto *t = reinterpret_cast<const TextureHeader *>(texture);
    if (((t->mip_levels <= 1u) | (filter <= 1u)) &&
        luisa_fallback_try_tex2d_sample_level0_fast(t, filter, address, uv->x, uv->y, out)) {
        return;
    }
    auto sampler = luisa_fallback_explicit_sampler(filter, address);
    *out = luisa_fallback_bindless_texture2d_sample_grad(texture, sampler, uv->x, uv->y, ddx->x, ddx->y, ddy->x, ddy->y);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture2d_sample_grad_level_sampler(const BindlessArrayView *handle, uint slot_index, const float2 *uv, const float2 *ddx, const float2 *ddy, float level, uint filter, uint address, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_TEXTURE2D(handle, slot_index)
    const auto *t = reinterpret_cast<const TextureHeader *>(texture);
    if (((level <= 0.0f) | (t->mip_levels <= 1u)) &&
        luisa_fallback_try_tex2d_sample_level0_fast(t, filter, address, uv->x, uv->y, out)) {
        return;
    }
    auto sampler = luisa_fallback_explicit_sampler(filter, address);
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

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_sample_sampler(const BindlessArrayView *handle, uint slot_index, const float3 *uvw, uint filter, uint address, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_TEXTURE3D(handle, slot_index)
    auto sampler = luisa_fallback_explicit_sampler(filter, address);
    *out = luisa_fallback_bindless_texture3d_sample(texture, sampler, uvw->x, uvw->y, uvw->z);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_sample_level_sampler(const BindlessArrayView *handle, uint slot_index, const float3 *uvw, float level, uint filter, uint address, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_TEXTURE3D(handle, slot_index)
    auto sampler = luisa_fallback_explicit_sampler(filter, address);
    *out = luisa_fallback_bindless_texture3d_sample_level(texture, sampler, uvw->x, uvw->y, uvw->z, level);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_sample_grad_sampler(const BindlessArrayView *handle, uint slot_index, const float3 *uvw, const float3 *ddx, const float3 *ddy, uint filter, uint address, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_TEXTURE3D(handle, slot_index)
    auto sampler = luisa_fallback_explicit_sampler(filter, address);
    *out = luisa_fallback_bindless_texture3d_sample_grad(texture, sampler, uvw->x, uvw->y, uvw->z, ddx->x, ddx->y, ddx->z, ddy->x, ddy->y, ddy->z);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_bindless_texture3d_sample_grad_level_sampler(const BindlessArrayView *handle, uint slot_index, const float3 *uvw, const float3 *ddx, const float3 *ddy, float level, uint filter, uint address, float4 *out) noexcept {
    LUISA_FALLBACK_BINDLESS_TEXTURE3D(handle, slot_index)
    auto sampler = luisa_fallback_explicit_sampler(filter, address);
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
    const EmbreeRay *active_rtc_ray;
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
    const uint inst = ray_hit->hit.instID[0] != ~0u ? ray_hit->hit.instID[0] : ray_hit->hit.geomID;
    *surface_hit = {.inst = inst,
                    .prim = ray_hit->hit.primID,
                    .bary = {ray_hit->hit.u, ray_hit->hit.v},
                    .committed_ray_t = ray_hit->ray.extra.tfar};
}

LUISA_FALLBACK_INTERNAL void luisa_fallback_decode_committed_hit(CommittedHit *committed_hit, const LC_RayQueryObject *q) noexcept {
    auto hit = &q->ray_hit.hit;
    const uint inst = hit->instID[0] != ~0u ? hit->instID[0] : hit->geomID;
    *committed_hit = {.inst = inst,
                      .prim = hit->primID,
                      .bary = {hit->u, hit->v},
                      .hit_type = q->candidate.committed_hit_type,
                      .committed_ray_t = hit->Ng_z};
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_traverse_motion(const AccelView *handle, const Ray *ray, float time, uint mask, LC_RayQueryObject *out) noexcept {
    out->accel = *handle;
    out->world_ray = *ray;
    out->active_rtc_ray = nullptr;
    out->candidate.committed_hit_type = HitType::Miss;
    luisa_fallback_create_embree_ray(&out->ray_hit.ray, ray, time, mask);
    luisa_fallback_create_embree_hit(&out->ray_hit.hit);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_traverse(const AccelView *handle, const Ray *ray, uint mask, LC_RayQueryObject *out) noexcept {
    luisa_fallback_wrapper_accel_traverse_motion(handle, ray, 0.f, mask, out);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_world_space_ray(const LC_RayQueryObject *q, Ray *out) noexcept {
    *out = q->world_ray;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_object_candidate_object_space_ray(const LC_RayQueryObject *q, Ray *out) noexcept {
    if (const auto *r = q->active_rtc_ray) {
        *out = {.origin = {r->org_tnear.x, r->org_tnear.y, r->org_tnear.z},
                .t_min = q->world_ray.t_min,
                .direction = {r->dir_time.x, r->dir_time.y, r->dir_time.z},
                .t_max = q->world_ray.t_max};
    } else {
        *out = q->world_ray;
    }
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
    luisa_fallback_decode_committed_hit(out, q);
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

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_pipeline_all_excl(LC_RayQueryObject *q, uint excl_inst0, uint excl_prim0, uint excl_inst1, uint excl_prim1, uint flags, const void *capture, RayQueryOnProceduralFunc *on_procedural) noexcept {
    luisa_fallback_ray_query_pipeline_all_excl(q, excl_inst0, excl_prim0, excl_inst1, excl_prim1, flags, capture, on_procedural);
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_ray_query_pipeline_any_excl(LC_RayQueryObject *q, uint excl_inst0, uint excl_prim0, uint excl_inst1, uint excl_prim1, uint flags, const void *capture, RayQueryOnProceduralFunc *on_procedural) noexcept {
    luisa_fallback_ray_query_pipeline_any_excl(q, excl_inst0, excl_prim0, excl_inst1, excl_prim1, flags, capture, on_procedural);
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
    auto instances = *handle->instances;
    auto rows = reinterpret_cast<const llvm_float4 *>(instances[instance_id].affine);
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
    *out = (*handle->instances)[instance_id].user_id;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_instance_visibility_mask(const AccelView *handle, uint instance_id, uint *out) noexcept {
    *out = (*handle->instances)[instance_id].mask;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_set_instance_transform(AccelView *handle, uint instance_id, const float4x4 *transform) noexcept {
    auto &instance = (*handle->instances)[instance_id];
    auto rows = reinterpret_cast<llvm_float4 *>(instance.affine);
    auto m = *reinterpret_cast<const llvm_float4x4 *>(transform);
    rows[0] = {m.cols[0].x, m.cols[1].x, m.cols[2].x, m.cols[3].x};
    rows[1] = {m.cols[0].y, m.cols[1].y, m.cols[2].y, m.cols[3].y};
    rows[2] = {m.cols[0].z, m.cols[1].z, m.cols[2].z, m.cols[3].z};
    instance.dirty = true;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_set_instance_visibility_mask(AccelView *handle, uint instance_id, uint mask) noexcept {
    auto &instance = (*handle->instances)[instance_id];
    instance.mask = mask;
    instance.dirty = true;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_set_instance_user_id(AccelView *handle, uint instance_id, uint user_id) noexcept {
    auto &instance = (*handle->instances)[instance_id];
    instance.user_id = user_id;
    instance.dirty = true;
}

LUISA_FALLBACK_WRAPPER void luisa_fallback_wrapper_accel_set_instance_opacity(AccelView *handle, uint instance_id, bool opacity) noexcept {
    auto &instance = (*handle->instances)[instance_id];
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
