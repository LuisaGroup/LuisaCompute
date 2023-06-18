#include <luisa/core/mathematics.h>
using namespace luisa;
constexpr float XM_PI = 3.141592654f;
constexpr float XM_2PI = 6.283185307f;
constexpr float XM_1DIVPI = 0.318309886f;
constexpr float XM_1DIV2PI = 0.159154943f;
constexpr float XM_PIDIV2 = 1.570796327f;
constexpr float XM_PIDIV4 = 0.785398163f;

constexpr uint32_t XM_SELECT_0 = 0x00000000;
constexpr uint32_t XM_SELECT_1 = 0xFFFFFFFF;

constexpr uint32_t XM_PERMUTE_0X = 0;
constexpr uint32_t XM_PERMUTE_0Y = 1;
constexpr uint32_t XM_PERMUTE_0Z = 2;
constexpr uint32_t XM_PERMUTE_0W = 3;
constexpr uint32_t XM_PERMUTE_1X = 4;
constexpr uint32_t XM_PERMUTE_1Y = 5;
constexpr uint32_t XM_PERMUTE_1Z = 6;
constexpr uint32_t XM_PERMUTE_1W = 7;

constexpr uint32_t XM_SWIZZLE_X = 0;
constexpr uint32_t XM_SWIZZLE_Y = 1;
constexpr uint32_t XM_SWIZZLE_Z = 2;
constexpr uint32_t XM_SWIZZLE_W = 3;

constexpr uint32_t XM_CRMASK_CR6 = 0x000000F0;
constexpr uint32_t XM_CRMASK_CR6TRUE = 0x00000080;
constexpr uint32_t XM_CRMASK_CR6FALSE = 0x00000020;
constexpr uint32_t XM_CRMASK_CR6BOUNDS = XM_CRMASK_CR6FALSE;
inline void ScalarSinCos(
    float *pSin,
    float *pCos,
    float Value) noexcept {

    // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
    float quotient = XM_1DIV2PI * Value;
    if (Value >= 0.0f) {
        quotient = static_cast<float>(static_cast<int>(quotient + 0.5f));
    } else {
        quotient = static_cast<float>(static_cast<int>(quotient - 0.5f));
    }
    float y = Value - XM_2PI * quotient;

    // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
    float sign;
    if (y > XM_PIDIV2) {
        y = XM_PI - y;
        sign = -1.0f;
    } else if (y < -XM_PIDIV2) {
        y = -XM_PI - y;
        sign = -1.0f;
    } else {
        sign = +1.0f;
    }

    float y2 = y * y;

    // 11-degree minimax approximation
    *pSin = (((((-2.3889859e-08f * y2 + 2.7525562e-06f) * y2 - 0.00019840874f) * y2 + 0.0083333310f) * y2 - 0.16666667f) * y2 + 1.0f) * y;

    // 10-degree minimax approximation
    float p = ((((-2.6051615e-07f * y2 + 2.4760495e-05f) * y2 - 0.0013888378f) * y2 + 0.041666638f) * y2 - 0.5f) * y2 + 1.0f;
    *pCos = sign * p;
}

inline float4x4 perspective_lh(
    float FovAngleY,
    float AspectRatio,
    float NearZ,
    float FarZ) {
    float SinFov;
    float CosFov;
    ScalarSinCos(&SinFov, &CosFov, 0.5f * FovAngleY);

    float Height = CosFov / SinFov;
    float Width = Height / AspectRatio;
    float fRange = FarZ / (FarZ - NearZ);
    float4x4 m;
    m[0][0] = Width;
    m[0][1] = 0.0f;
    m[0][2] = 0.0f;
    m[0][3] = 0.0f;

    m[1][0] = 0.0f;
    m[1][1] = Height;
    m[1][2] = 0.0f;
    m[1][3] = 0.0f;

    m[2][0] = 0.0f;
    m[2][1] = 0.0f;
    m[2][2] = fRange;
    m[2][3] = 1.0f;

    m[3][0] = 0.0f;
    m[3][1] = 0.0f;
    m[3][2] = -fRange * NearZ;
    m[3][3] = 0.0f;
    return m;
}
