// Device-side math builtins test.
// Computes math functions on GPU, reads results back, compares against CPU reference.

#include <cmath>
#include <numbers>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;
using namespace boost::ut;
using namespace boost::ut::literals;

enum Slot : uint {
    SIN_0,
    COS_0,
    TAN_0,
    SIN_PI6,
    COS_PI6,
    TAN_PI6,
    SIN_PI4,
    COS_PI4,
    TAN_PI4,
    SIN_PI3,
    COS_PI3,
    TAN_PI3,
    SIN_PI,
    COS_PI,
    TAN_PI,
    SIN_NPI4,
    COS_NPI4,
    TAN_NPI4,
    SIN_10,
    COS_10,
    TAN_10,
    ASIN_0,
    ACOS_0,
    ASIN_0P5,
    ACOS_0P5,
    ASIN_1,
    ACOS_1,
    ASIN_N1,
    ACOS_N1,
    ASIN_N0P5,
    ACOS_N0P5,
    ATAN_0,
    ATAN_0P5,
    ATAN_1,
    ATAN_N1,
    ATAN_10,
    ATAN_N10,
    ACOSH_1,
    ACOSH_2,
    ACOSH_10,
    ASINH_0,
    ASINH_1,
    ASINH_N1,
    ASINH_10,
    ASINH_N10,
    ATANH_0,
    ATANH_0P5,
    ATANH_N0P5,
    ATANH_0P99,
    ATANH_N0P99,
    EXP_0,
    EXP2_0,
    EXP_1,
    EXP2_1,
    EXP_N1,
    EXP2_N1,
    EXP_10,
    EXP2_10,
    EXP_N10,
    EXP2_N10,
    EXP10_0,
    EXP10_1,
    EXP10_N1,
    EXP10_5,
    EXP10_N5,
    LOG_1,
    LOG2_1,
    LOG10_1,
    LOG_E,
    LOG2_E,
    LOG10_E,
    LOG_2,
    LOG2_2,
    LOG10_2,
    LOG_10,
    LOG2_10,
    LOG10_10,
    LOG_100,
    LOG2_100,
    LOG10_100,
    LOG_0P1,
    LOG2_0P1,
    LOG10_0P1,
    SQRT_0,
    SQRT_1,
    SQRT_4,
    SQRT_0P25,
    SQRT_100,
    RSQRT_1,
    RSQRT_4,
    RSQRT_0P25,
    RSQRT_100,
    FLOOR_1P7,
    CEIL_1P7,
    TRUNC_1P7,
    FLOOR_1P3,
    CEIL_1P3,
    TRUNC_1P3,
    FLOOR_N1P7,
    CEIL_N1P7,
    TRUNC_N1P7,
    FLOOR_N1P3,
    CEIL_N1P3,
    TRUNC_N1P3,
    FLOOR_2,
    CEIL_2,
    TRUNC_2,
    FLOOR_2P5,
    CEIL_2P5,
    TRUNC_2P5,
    FLOOR_N2P5,
    CEIL_N2P5,
    TRUNC_N2P5,
    ROUND_1P7,
    ROUND_1P3,
    ROUND_N1P7,
    ROUND_N1P3,
    ROUND_2,
    ROUND_2P5,
    ROUND_N2P5,
    ROUND_3P5,
    FRACT_1P7,
    FRACT_1P3,
    FRACT_N1P7,
    FRACT_N1P3,
    FRACT_2,
    ABS_4P5,
    ABS_N4P5,
    ABS_0,
    ABS_N0,
    SIGN_3,
    SIGN_N2,
    SIGN_0,
    SIGN_N0,
    SATURATE_N0P5,
    SATURATE_0,
    SATURATE_0P5,
    SATURATE_1,
    SATURATE_1P5,
    POW_2_3,
    POW_0_0,
    POW_2P5_1P5,
    POW_9_0P5,
    ATAN2_0_1,
    ATAN2_1_0,
    ATAN2_0_N1,
    ATAN2_N1_0,
    ATAN2_1_1,
    ATAN2_N1_1,
    ATAN2_N1_N1,
    ATAN2_1_N1,
    ATAN2_0_0,
    MIN_3_5,
    MAX_3_5,
    MIN_5_3,
    MAX_5_3,
    MIN_N3_N5,
    MAX_N3_N5,
    MIN_4_4,
    MAX_4_4,
    FMA_2_3_4,
    FMA_N2_3_4,
    FMA_2_N3_N4,
    STEP_BELOW,
    STEP_AT,
    STEP_ABOVE,
    COPYSIGN_3_N1,
    COPYSIGN_N3_1,
    COPYSIGN_3_1,
    COPYSIGN_N3_N1,
    COPYSIGN_0_N1,
    MOD_P_P,
    FMOD_P_P,
    MOD_N_P,
    FMOD_N_P,
    MOD_P_N,
    FMOD_P_N,
    MOD_N_N,
    FMOD_N_N,
    SMOOTHSTEP_BELOW,
    SMOOTHSTEP_AT_LEFT,
    SMOOTHSTEP_MID,
    SMOOTHSTEP_AT_RIGHT,
    SMOOTHSTEP_ABOVE,
    DEGREES_0,
    DEGREES_PI2,
    DEGREES_PI,
    DEGREES_2PI,
    RADIANS_0,
    RADIANS_90,
    RADIANS_180,
    RADIANS_360,
    CLAMP_BELOW,
    CLAMP_IN,
    CLAMP_ABOVE,
    LERP_0,
    LERP_0P5,
    LERP_1,
    LERP_N0P5,
    LERP_1P5,
    DOT_ORTHO,
    DOT_STD,
    DOT_ANTI,
    CROSS_ORTHO_X,
    CROSS_ORTHO_Y,
    CROSS_ORTHO_Z,
    CROSS_STD_X,
    CROSS_STD_Y,
    CROSS_STD_Z,
    CROSS_PAR_X,
    CROSS_PAR_Y,
    CROSS_PAR_Z,
    LENGTH_UNIT,
    LENGTH_ZERO,
    LENGTH_STD,
    LENGTH_SQ_UNIT,
    LENGTH_SQ_ZERO,
    LENGTH_SQ_STD,
    NORM_STD_X,
    NORM_STD_Y,
    NORM_STD_Z,
    NORM_UNIT_X,
    NORM_UNIT_Y,
    NORM_UNIT_Z,
    DIST_SAME,
    DIST_DIFF,
    DIST_SQ_SAME,
    DIST_SQ_DIFF,
    REFLECT_1_X,
    REFLECT_1_Y,
    REFLECT_1_Z,
    REFLECT_2_X,
    REFLECT_2_Y,
    REFLECT_2_Z,
    FF_SAME_X,
    FF_SAME_Y,
    FF_SAME_Z,
    FF_OPP_X,
    FF_OPP_Y,
    FF_OPP_Z,
    RED_SUM_2,
    RED_SUM_3,
    RED_SUM_4,
    RED_PROD_2,
    RED_PROD_3,
    RED_PROD_4,
    RED_MIN_2,
    RED_MIN_3,
    RED_MIN_4,
    RED_MAX_2,
    RED_MAX_3,
    RED_MAX_4,
    ALL_TT,
    ALL_TFT,
    ANY_FF,
    ANY_FTF,
    NONE_FF,
    NONE_TFF,
    DET_ID,
    DET_SCALE,
    DET_SING,
    TRANS_0_1,
    TRANS_1_0,
    INV_SCALE_0_0,
    INV_SCALE_1_1,
    INV_SCALE_2_2,
    ABS_I_7,
    ABS_I_N7,
    ABS_I_0,
    ABS_I_N1000,
    MIN_I_3_5,
    MAX_I_3_5,
    MIN_I_5_3,
    MAX_I_5_3,
    MIN_I_N3_N5,
    MAX_I_N3_N5,
    CLAMP_I_ABOVE,
    CLAMP_I_BELOW,
    CLAMP_I_IN,
    CLZ_0,
    CTZ_0,
    POPCOUNT_0,
    REVERSE_0,
    CLZ_1,
    CTZ_1,
    POPCOUNT_1,
    REVERSE_1,
    CLZ_16,
    CTZ_16,
    POPCOUNT_16,
    REVERSE_16,
    CLZ_ALL1,
    CTZ_ALL1,
    POPCOUNT_ALL1,
    REVERSE_ALL1,
    CLZ_PATTERN,
    CTZ_PATTERN,
    POPCOUNT_PATTERN,
    REVERSE_PATTERN,
    ISINF_INF,
    ISINF_NINF,
    ISINF_NORMAL,
    ISINF_NAN,
    ISNAN_NAN,
    ISNAN_NORMAL,
    ISNAN_INF,
    SELECT_T,
    SELECT_F,
    ITE_T,
    ITE_F,
    SLOT_COUNT
};

int test_device_math(Device &device) {
    Stream stream = device.create_stream();

    Buffer<float> result_buf = device.create_buffer<float>(SLOT_COUNT);

    Kernel1D kernel = [&] {
        auto write = [&](uint idx, Float v) { result_buf->write(static_cast<UInt>(idx), v); };

        write(SIN_0, sin(def(0.0f)));
        write(COS_0, cos(def(0.0f)));
        write(TAN_0, tan(def(0.0f)));
        write(SIN_PI6, sin(def(0.5235987755982988f)));
        write(COS_PI6, cos(def(0.5235987755982988f)));
        write(TAN_PI6, tan(def(0.5235987755982988f)));
        write(SIN_PI4, sin(def(0.7853981633974483f)));
        write(COS_PI4, cos(def(0.7853981633974483f)));
        write(TAN_PI4, tan(def(0.7853981633974483f)));
        write(SIN_PI3, sin(def(1.0471975511965976f)));
        write(COS_PI3, cos(def(1.0471975511965976f)));
        write(TAN_PI3, tan(def(1.0471975511965976f)));
        write(SIN_PI, sin(def(3.141592653589793f)));
        write(COS_PI, cos(def(3.141592653589793f)));
        write(TAN_PI, tan(def(3.141592653589793f)));
        write(SIN_NPI4, sin(def(-0.7853981633974483f)));
        write(COS_NPI4, cos(def(-0.7853981633974483f)));
        write(TAN_NPI4, tan(def(-0.7853981633974483f)));
        write(SIN_10, sin(def(10.0f)));
        write(COS_10, cos(def(10.0f)));
        write(TAN_10, tan(def(10.0f)));
        write(ASIN_0, asin(def(0.0f)));
        write(ACOS_0, acos(def(0.0f)));
        write(ASIN_0P5, asin(def(0.5f)));
        write(ACOS_0P5, acos(def(0.5f)));
        write(ASIN_1, asin(def(1.0f)));
        write(ACOS_1, acos(def(1.0f)));
        write(ASIN_N1, asin(def(-1.0f)));
        write(ACOS_N1, acos(def(-1.0f)));
        write(ASIN_N0P5, asin(def(-0.5f)));
        write(ACOS_N0P5, acos(def(-0.5f)));
        write(ATAN_0, atan(def(0.0f)));
        write(ATAN_0P5, atan(def(0.5f)));
        write(ATAN_1, atan(def(1.0f)));
        write(ATAN_N1, atan(def(-1.0f)));
        write(ATAN_10, atan(def(10.0f)));
        write(ATAN_N10, atan(def(-10.0f)));
        write(ACOSH_1, acosh(def(1.0f)));
        write(ACOSH_2, acosh(def(2.0f)));
        write(ACOSH_10, acosh(def(10.0f)));
        write(ASINH_0, asinh(def(0.0f)));
        write(ASINH_1, asinh(def(1.0f)));
        write(ASINH_N1, asinh(def(-1.0f)));
        write(ASINH_10, asinh(def(10.0f)));
        write(ASINH_N10, asinh(def(-10.0f)));
        write(ATANH_0, atanh(def(0.0f)));
        write(ATANH_0P5, atanh(def(0.5f)));
        write(ATANH_N0P5, atanh(def(-0.5f)));
        write(ATANH_0P99, atanh(def(0.99f)));
        write(ATANH_N0P99, atanh(def(-0.99f)));
        write(EXP_0, exp(def(0.0f)));
        write(EXP2_0, exp2(def(0.0f)));
        write(EXP_1, exp(def(1.0f)));
        write(EXP2_1, exp2(def(1.0f)));
        write(EXP_N1, exp(def(-1.0f)));
        write(EXP2_N1, exp2(def(-1.0f)));
        write(EXP_10, exp(def(10.0f)));
        write(EXP2_10, exp2(def(10.0f)));
        write(EXP_N10, exp(def(-10.0f)));
        write(EXP2_N10, exp2(def(-10.0f)));
        write(EXP10_0, exp10(def(0.0f)));
        write(EXP10_1, exp10(def(1.0f)));
        write(EXP10_N1, exp10(def(-1.0f)));
        write(EXP10_5, exp10(def(5.0f)));
        write(EXP10_N5, exp10(def(-5.0f)));
        write(LOG_1, luisa::compute::log(def(1.0f)));
        write(LOG2_1, log2(def(1.0f)));
        write(LOG10_1, log10(def(1.0f)));
        write(LOG_E, luisa::compute::log(def(2.718281828459045f)));
        write(LOG2_E, log2(def(2.718281828459045f)));
        write(LOG10_E, log10(def(2.718281828459045f)));
        write(LOG_2, luisa::compute::log(def(2.0f)));
        write(LOG2_2, log2(def(2.0f)));
        write(LOG10_2, log10(def(2.0f)));
        write(LOG_10, luisa::compute::log(def(10.0f)));
        write(LOG2_10, log2(def(10.0f)));
        write(LOG10_10, log10(def(10.0f)));
        write(LOG_100, luisa::compute::log(def(100.0f)));
        write(LOG2_100, log2(def(100.0f)));
        write(LOG10_100, log10(def(100.0f)));
        write(LOG_0P1, luisa::compute::log(def(0.1f)));
        write(LOG2_0P1, log2(def(0.1f)));
        write(LOG10_0P1, log10(def(0.1f)));
        write(SQRT_0, sqrt(def(0.0f)));
        write(SQRT_1, sqrt(def(1.0f)));
        write(SQRT_4, sqrt(def(4.0f)));
        write(SQRT_0P25, sqrt(def(0.25f)));
        write(SQRT_100, sqrt(def(100.0f)));
        write(RSQRT_1, rsqrt(def(1.0f)));
        write(RSQRT_4, rsqrt(def(4.0f)));
        write(RSQRT_0P25, rsqrt(def(0.25f)));
        write(RSQRT_100, rsqrt(def(100.0f)));
        write(FLOOR_1P7, floor(def(1.7f)));
        write(CEIL_1P7, ceil(def(1.7f)));
        write(TRUNC_1P7, trunc(def(1.7f)));
        write(FLOOR_1P3, floor(def(1.3f)));
        write(CEIL_1P3, ceil(def(1.3f)));
        write(TRUNC_1P3, trunc(def(1.3f)));
        write(FLOOR_N1P7, floor(def(-1.7f)));
        write(CEIL_N1P7, ceil(def(-1.7f)));
        write(TRUNC_N1P7, trunc(def(-1.7f)));
        write(FLOOR_N1P3, floor(def(-1.3f)));
        write(CEIL_N1P3, ceil(def(-1.3f)));
        write(TRUNC_N1P3, trunc(def(-1.3f)));
        write(FLOOR_2, floor(def(2.0f)));
        write(CEIL_2, ceil(def(2.0f)));
        write(TRUNC_2, trunc(def(2.0f)));
        write(FLOOR_2P5, floor(def(2.5f)));
        write(CEIL_2P5, ceil(def(2.5f)));
        write(TRUNC_2P5, trunc(def(2.5f)));
        write(FLOOR_N2P5, floor(def(-2.5f)));
        write(CEIL_N2P5, ceil(def(-2.5f)));
        write(TRUNC_N2P5, trunc(def(-2.5f)));
        write(ROUND_1P7, round(def(1.7f)));
        write(ROUND_1P3, round(def(1.3f)));
        write(ROUND_N1P7, round(def(-1.7f)));
        write(ROUND_N1P3, round(def(-1.3f)));
        write(ROUND_2, round(def(2.0f)));
        write(ROUND_2P5, round(def(2.5f)));
        write(ROUND_N2P5, round(def(-2.5f)));
        write(ROUND_3P5, round(def(3.5f)));
        write(FRACT_1P7, fract(def(1.7f)));
        write(FRACT_1P3, fract(def(1.3f)));
        write(FRACT_N1P7, fract(def(-1.7f)));
        write(FRACT_N1P3, fract(def(-1.3f)));
        write(FRACT_2, fract(def(2.0f)));
        write(ABS_4P5, abs(def(4.5f)));
        write(ABS_N4P5, abs(def(-4.5f)));
        write(ABS_0, abs(def(0.0f)));
        write(ABS_N0, abs(def(-0.0f)));
        write(SIGN_3, sign(def(3.0f)));
        write(SIGN_N2, sign(def(-2.0f)));
        write(SIGN_0, sign(def(0.0f)));
        write(SIGN_N0, sign(def(-0.0f)));
        write(SATURATE_N0P5, saturate(def(-0.5f)));
        write(SATURATE_0, saturate(def(0.0f)));
        write(SATURATE_0P5, saturate(def(0.5f)));
        write(SATURATE_1, saturate(def(1.0f)));
        write(SATURATE_1P5, saturate(def(1.5f)));
        write(POW_2_3, pow(def(2.0f), def(3.0f)));
        write(POW_0_0, pow(def(0.0f), def(0.0f)));
        write(POW_2P5_1P5, pow(def(2.5f), def(1.5f)));
        write(POW_9_0P5, pow(def(9.0f), def(0.5f)));
        write(ATAN2_0_1, atan2(def(0.0f), def(1.0f)));
        write(ATAN2_1_0, atan2(def(1.0f), def(0.0f)));
        write(ATAN2_0_N1, atan2(def(0.0f), def(-1.0f)));
        write(ATAN2_N1_0, atan2(def(-1.0f), def(0.0f)));
        write(ATAN2_1_1, atan2(def(1.0f), def(1.0f)));
        write(ATAN2_N1_1, atan2(def(-1.0f), def(1.0f)));
        write(ATAN2_N1_N1, atan2(def(-1.0f), def(-1.0f)));
        write(ATAN2_1_N1, atan2(def(1.0f), def(-1.0f)));
        write(ATAN2_0_0, atan2(def(0.0f), def(0.0f)));
        write(MIN_3_5, min(def(3.0f), def(5.0f)));
        write(MAX_3_5, max(def(3.0f), def(5.0f)));
        write(MIN_5_3, min(def(5.0f), def(3.0f)));
        write(MAX_5_3, max(def(5.0f), def(3.0f)));
        write(MIN_N3_N5, min(def(-3.0f), def(-5.0f)));
        write(MAX_N3_N5, max(def(-3.0f), def(-5.0f)));
        write(MIN_4_4, min(def(4.0f), def(4.0f)));
        write(MAX_4_4, max(def(4.0f), def(4.0f)));
        write(FMA_2_3_4, fma(def(2.0f), def(3.0f), def(4.0f)));
        write(FMA_N2_3_4, fma(def(-2.0f), def(3.0f), def(4.0f)));
        write(FMA_2_N3_N4, fma(def(2.0f), def(-3.0f), def(-4.0f)));
        write(STEP_BELOW, step(def(0.5f), def(0.4f)));
        write(STEP_AT, step(def(0.5f), def(0.5f)));
        write(STEP_ABOVE, step(def(0.5f), def(0.6f)));
        write(COPYSIGN_3_N1, copysign(def(3.0f), def(-1.0f)));
        write(COPYSIGN_N3_1, copysign(def(-3.0f), def(1.0f)));
        write(COPYSIGN_3_1, copysign(def(3.0f), def(1.0f)));
        write(COPYSIGN_N3_N1, copysign(def(-3.0f), def(-1.0f)));
        write(COPYSIGN_0_N1, copysign(def(0.0f), def(-1.0f)));
        write(MOD_P_P, mod(def(5.2f), def(2.0f)));
        write(FMOD_P_P, fmod(def(5.2f), def(2.0f)));
        write(MOD_N_P, mod(def(-5.2f), def(2.0f)));
        write(FMOD_N_P, fmod(def(-5.2f), def(2.0f)));
        write(MOD_P_N, mod(def(5.2f), def(-2.0f)));
        write(FMOD_P_N, fmod(def(5.2f), def(-2.0f)));
        write(MOD_N_N, mod(def(-5.2f), def(-2.0f)));
        write(FMOD_N_N, fmod(def(-5.2f), def(-2.0f)));
        write(SMOOTHSTEP_BELOW, smoothstep(def(0.0f), def(1.0f), def(-0.5f)));
        write(SMOOTHSTEP_AT_LEFT, smoothstep(def(0.0f), def(1.0f), def(0.0f)));
        write(SMOOTHSTEP_MID, smoothstep(def(0.0f), def(1.0f), def(0.5f)));
        write(SMOOTHSTEP_AT_RIGHT, smoothstep(def(0.0f), def(1.0f), def(1.0f)));
        write(SMOOTHSTEP_ABOVE, smoothstep(def(0.0f), def(1.0f), def(1.5f)));
        write(DEGREES_0, degrees(def(0.0f)));
        write(DEGREES_PI2, degrees(def(1.5707963267948966f)));
        write(DEGREES_PI, degrees(def(3.141592653589793f)));
        write(DEGREES_2PI, degrees(def(6.283185307179586f)));
        write(RADIANS_0, radians(def(0.0f)));
        write(RADIANS_90, radians(def(90.0f)));
        write(RADIANS_180, radians(def(180.0f)));
        write(RADIANS_360, radians(def(360.0f)));
        write(CLAMP_BELOW, clamp(def(-1.0f), def(0.0f), def(1.0f)));
        write(CLAMP_IN, clamp(def(0.5f), def(0.0f), def(1.0f)));
        write(CLAMP_ABOVE, clamp(def(2.0f), def(0.0f), def(1.0f)));
        write(LERP_0, lerp(def(10.0f), def(20.0f), def(0.0f)));
        write(LERP_0P5, lerp(def(10.0f), def(20.0f), def(0.5f)));
        write(LERP_1, lerp(def(10.0f), def(20.0f), def(1.0f)));
        write(LERP_N0P5, lerp(def(10.0f), def(20.0f), def(-0.5f)));
        write(LERP_1P5, lerp(def(10.0f), def(20.0f), def(1.5f)));
        Float3 v1_0_0 = make_float3(1.0f, 0.0f, 0.0f);
        Float3 v0_1_0 = make_float3(0.0f, 1.0f, 0.0f);
        Float3 v1_2_3 = make_float3(1.0f, 2.0f, 3.0f);
        Float3 v4_5_6 = make_float3(4.0f, 5.0f, 6.0f);
        Float3 vn1_n2_n3 = make_float3(-1.0f, -2.0f, -3.0f);
        Float3 v2_4_6 = make_float3(2.0f, 4.0f, 6.0f);
        Float3 v0_0_0 = make_float3(0.0f, 0.0f, 0.0f);
        write(DOT_ORTHO, dot(v1_0_0, v0_1_0));
        write(DOT_STD, dot(v1_2_3, v4_5_6));
        write(DOT_ANTI, dot(v1_2_3, vn1_n2_n3));
        Float3 cr1 = cross(v1_0_0, v0_1_0);
        write(CROSS_ORTHO_X, cr1.x);
        write(CROSS_ORTHO_Y, cr1.y);
        write(CROSS_ORTHO_Z, cr1.z);
        Float3 cr2 = cross(v1_2_3, v4_5_6);
        write(CROSS_STD_X, cr2.x);
        write(CROSS_STD_Y, cr2.y);
        write(CROSS_STD_Z, cr2.z);
        Float3 cr3 = cross(v1_2_3, v2_4_6);
        write(CROSS_PAR_X, cr3.x);
        write(CROSS_PAR_Y, cr3.y);
        write(CROSS_PAR_Z, cr3.z);
        write(LENGTH_UNIT, length(v1_0_0));
        write(LENGTH_ZERO, length(v0_0_0));
        write(LENGTH_STD, length(v1_2_3));
        write(LENGTH_SQ_UNIT, length_squared(v1_0_0));
        write(LENGTH_SQ_ZERO, length_squared(v0_0_0));
        write(LENGTH_SQ_STD, length_squared(v1_2_3));
        Float3 n1 = normalize(v1_2_3);
        write(NORM_STD_X, n1.x);
        write(NORM_STD_Y, n1.y);
        write(NORM_STD_Z, n1.z);
        Float3 n2 = normalize(v1_0_0);
        write(NORM_UNIT_X, n2.x);
        write(NORM_UNIT_Y, n2.y);
        write(NORM_UNIT_Z, n2.z);
        write(DIST_SAME, distance(v1_2_3, v1_2_3));
        write(DIST_DIFF, distance(v1_2_3, v4_5_6));
        write(DIST_SQ_SAME, distance_squared(v1_2_3, v1_2_3));
        write(DIST_SQ_DIFF, distance_squared(v1_2_3, v4_5_6));
        Float3 i1 = make_float3(1.0f, -1.0f, 0.0f);
        Float3 n_ref1 = make_float3(0.0f, 1.0f, 0.0f);
        Float3 ref1 = reflect(i1, n_ref1);
        write(REFLECT_1_X, ref1.x);
        write(REFLECT_1_Y, ref1.y);
        write(REFLECT_1_Z, ref1.z);
        Float3 n_ref2 = make_float3(0.70710678f, 0.70710678f, 0.0f);
        Float3 ref2 = reflect(i1, n_ref2);
        write(REFLECT_2_X, ref2.x);
        write(REFLECT_2_Y, ref2.y);
        write(REFLECT_2_Z, ref2.z);
        Float3 ff_n = make_float3(0.0f, 1.0f, 0.0f);
        Float3 ff_i1 = make_float3(0.0f, -1.0f, 0.0f);
        Float3 ff_nref = make_float3(0.0f, 1.0f, 0.0f);
        Float3 ff1 = faceforward(ff_n, ff_i1, ff_nref);
        write(FF_SAME_X, ff1.x);
        write(FF_SAME_Y, ff1.y);
        write(FF_SAME_Z, ff1.z);
        Float3 ff_i2 = make_float3(0.0f, 1.0f, 0.0f);
        Float3 ff2 = faceforward(ff_n, ff_i2, ff_nref);
        write(FF_OPP_X, ff2.x);
        write(FF_OPP_Y, ff2.y);
        write(FF_OPP_Z, ff2.z);
        Float2 v2_1_2 = make_float2(1.0f, 2.0f);
        Float3 v3_1_2_3 = make_float3(1.0f, 2.0f, 3.0f);
        Float4 v4_1_2_3_4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        write(RED_SUM_2, reduce_sum(v2_1_2));
        write(RED_SUM_3, reduce_sum(v3_1_2_3));
        write(RED_SUM_4, reduce_sum(v4_1_2_3_4));
        write(RED_PROD_2, reduce_prod(v2_1_2));
        write(RED_PROD_3, reduce_prod(v3_1_2_3));
        write(RED_PROD_4, reduce_prod(v4_1_2_3_4));
        Float2 v2_2_1 = make_float2(2.0f, 1.0f);
        Float3 v3_3_1_2 = make_float3(3.0f, 1.0f, 2.0f);
        Float4 v4_4_1_3_2 = make_float4(4.0f, 1.0f, 3.0f, 2.0f);
        write(RED_MIN_2, reduce_min(v2_2_1));
        write(RED_MIN_3, reduce_min(v3_3_1_2));
        write(RED_MIN_4, reduce_min(v4_4_1_3_2));
        Float2 v2_1_2_max = make_float2(1.0f, 2.0f);
        Float3 v3_1_3_2 = make_float3(1.0f, 3.0f, 2.0f);
        Float4 v4_1_4_2_3 = make_float4(1.0f, 4.0f, 2.0f, 3.0f);
        write(RED_MAX_2, reduce_max(v2_1_2_max));
        write(RED_MAX_3, reduce_max(v3_1_3_2));
        write(RED_MAX_4, reduce_max(v4_1_4_2_3));
        Bool2 b2_tt = make_bool2(true, true);
        Bool2 b2_ff = make_bool2(false, false);
        Bool3 b3_tft = make_bool3(true, false, true);
        Bool3 b3_ftf = make_bool3(false, true, false);
        Bool3 b3_tff = make_bool3(true, false, false);
        write(ALL_TT, ite(all(b2_tt), 1.0f, 0.0f));
        write(ALL_TFT, ite(all(b3_tft), 1.0f, 0.0f));
        write(ANY_FF, ite(any(b2_ff), 1.0f, 0.0f));
        write(ANY_FTF, ite(any(b3_ftf), 1.0f, 0.0f));
        write(NONE_FF, ite(luisa::compute::none(b2_ff), 1.0f, 0.0f));
        write(NONE_TFF, ite(luisa::compute::none(b3_tff), 1.0f, 0.0f));
        Float3x3 m_id = make_float3x3(1.0f);
        Float3x3 m_scale = make_float3x3(make_float3(2.0f, 0.0f, 0.0f), make_float3(0.0f, 3.0f, 0.0f), make_float3(0.0f, 0.0f, 4.0f));
        Float3x3 m_sing = make_float3x3(make_float3(1.0f, 2.0f, 3.0f), make_float3(2.0f, 4.0f, 6.0f), make_float3(3.0f, 6.0f, 9.0f));
        write(DET_ID, determinant(m_id));
        write(DET_SCALE, determinant(m_scale));
        write(DET_SING, determinant(m_sing));
        Float3x3 m_nonsym = make_float3x3(make_float3(1.0f, 2.0f, 3.0f), make_float3(4.0f, 5.0f, 6.0f), make_float3(7.0f, 8.0f, 9.0f));
        Float3x3 m_trans = transpose(m_nonsym);
        write(TRANS_0_1, m_trans[0][1]);
        write(TRANS_1_0, m_trans[1][0]);
        Float3x3 m_inv_scale = inverse(m_scale);
        write(INV_SCALE_0_0, m_inv_scale[0][0]);
        write(INV_SCALE_1_1, m_inv_scale[1][1]);
        write(INV_SCALE_2_2, m_inv_scale[2][2]);
        write(ABS_I_7, cast<float>(abs(def(7))));
        write(ABS_I_N7, cast<float>(abs(def(-7))));
        write(ABS_I_0, cast<float>(abs(def(0))));
        write(ABS_I_N1000, cast<float>(abs(def(-1000))));
        write(MIN_I_3_5, cast<float>(min(def(3), def(5))));
        write(MAX_I_3_5, cast<float>(max(def(3), def(5))));
        write(MIN_I_5_3, cast<float>(min(def(5), def(3))));
        write(MAX_I_5_3, cast<float>(max(def(5), def(3))));
        write(MIN_I_N3_N5, cast<float>(min(def(-3), def(-5))));
        write(MAX_I_N3_N5, cast<float>(max(def(-3), def(-5))));
        write(CLAMP_I_ABOVE, cast<float>(clamp(def(10), def(0), def(5))));
        write(CLAMP_I_BELOW, cast<float>(clamp(def(-1), def(0), def(5))));
        write(CLAMP_I_IN, cast<float>(clamp(def(3), def(0), def(5))));
        write(CLZ_0, cast<float>(clz(def(0u))));
        write(CTZ_0, cast<float>(ctz(def(0u))));
        write(POPCOUNT_0, cast<float>(popcount(def(0u))));
        write(REVERSE_0, cast<float>(reverse(def(0u))));
        write(CLZ_1, cast<float>(clz(def(1u))));
        write(CTZ_1, cast<float>(ctz(def(1u))));
        write(POPCOUNT_1, cast<float>(popcount(def(1u))));
        write(REVERSE_1, cast<float>(reverse(def(1u))));
        write(CLZ_16, cast<float>(clz(def(16u))));
        write(CTZ_16, cast<float>(ctz(def(16u))));
        write(POPCOUNT_16, cast<float>(popcount(def(16u))));
        write(REVERSE_16, cast<float>(reverse(def(16u))));
        write(CLZ_ALL1, cast<float>(clz(def(4294967295u))));
        write(CTZ_ALL1, cast<float>(ctz(def(4294967295u))));
        write(POPCOUNT_ALL1, cast<float>(popcount(def(4294967295u))));
        write(REVERSE_ALL1, cast<float>(reverse(def(4294967295u))));
        write(CLZ_PATTERN, cast<float>(clz(def(252645135u))));
        write(CTZ_PATTERN, cast<float>(ctz(def(252645135u))));
        write(POPCOUNT_PATTERN, cast<float>(popcount(def(252645135u))));
        write(REVERSE_PATTERN, cast<float>(reverse(def(252645135u))));
        Float inf_val = def(1.0f) / def(0.0f);
        Float ninf_val = def(-1.0f) / def(0.0f);
        Float nan_val = sqrt(def(-1.0f));
        Float normal_val = def(1.0f);
        write(ISINF_INF, ite(dsl::isinf(inf_val), 1.0f, 0.0f));
        write(ISINF_NINF, ite(dsl::isinf(ninf_val), 1.0f, 0.0f));
        write(ISINF_NORMAL, ite(dsl::isinf(normal_val), 1.0f, 0.0f));
        write(ISINF_NAN, ite(dsl::isinf(nan_val), 1.0f, 0.0f));
        write(ISNAN_NAN, ite(dsl::isnan(nan_val), 1.0f, 0.0f));
        write(ISNAN_NORMAL, ite(dsl::isnan(normal_val), 1.0f, 0.0f));
        write(ISNAN_INF, ite(dsl::isnan(inf_val), 1.0f, 0.0f));
        write(SELECT_T, select(def(1.0f), def(2.0f), def(true)));
        write(SELECT_F, select(def(1.0f), def(2.0f), def(false)));
        write(ITE_T, ite(def(true), def(2.0f), def(1.0f)));
        write(ITE_F, ite(def(false), def(2.0f), def(1.0f)));
    };

    auto shader = device.compile(kernel);

    luisa::vector<float> results(SLOT_COUNT);
    stream << shader().dispatch(1u)
           << result_buf.copy_to(luisa::span{results})
           << synchronize();

    auto approx = [](float a, float b, float eps = 1e-4f) {
        if (std::isnan(a) && std::isnan(b)) return true;
        if (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)) return true;
        float abs_diff = std::abs(a - b);
        float rel_diff = abs_diff / std::max(std::abs(a), std::abs(b));
        return abs_diff < eps || rel_diff < eps;
    };

    auto check = [&](Slot s, float expected, const char *name) {
        bool ok = approx(results[s], expected);
        if (std::string(name) == "atan2_0_0" && std::isnan(results[s])) ok = true;
        expect(static_cast<bool>(ok))
            << name << ": got " << results[s] << " expected " << expected;
    };

    auto check_exact = [&](Slot s, float expected, const char *name) {
        bool ok = false;
        if (std::isnan(results[s]) && std::isnan(expected))
            ok = true;
        else if (std::isinf(results[s]) && std::isinf(expected) && (results[s] > 0) == (expected > 0))
            ok = true;
        else
            ok = (results[s] == expected);
        expect(static_cast<bool>(ok))
            << name << ": got " << results[s] << " expected " << expected;
    };

    check(SIN_0, std::sin(0.0f), "sin_0");
    check(COS_0, std::cos(0.0f), "cos_0");
    check(TAN_0, std::tan(0.0f), "tan_0");
    check(SIN_PI6, std::sin(0.5235987755982988f), "sin_pi6");
    check(COS_PI6, std::cos(0.5235987755982988f), "cos_pi6");
    check(TAN_PI6, std::tan(0.5235987755982988f), "tan_pi6");
    check(SIN_PI4, std::sin(0.7853981633974483f), "sin_pi4");
    check(COS_PI4, std::cos(0.7853981633974483f), "cos_pi4");
    check(TAN_PI4, std::tan(0.7853981633974483f), "tan_pi4");
    check(SIN_PI3, std::sin(1.0471975511965976f), "sin_pi3");
    check(COS_PI3, std::cos(1.0471975511965976f), "cos_pi3");
    check(TAN_PI3, std::tan(1.0471975511965976f), "tan_pi3");
    check(SIN_PI, std::sin(3.141592653589793f), "sin_pi");
    check(COS_PI, std::cos(3.141592653589793f), "cos_pi");
    check(TAN_PI, std::tan(3.141592653589793f), "tan_pi");
    check(SIN_NPI4, std::sin(-0.7853981633974483f), "sin_npi4");
    check(COS_NPI4, std::cos(-0.7853981633974483f), "cos_npi4");
    check(TAN_NPI4, std::tan(-0.7853981633974483f), "tan_npi4");
    check(SIN_10, std::sin(10.0f), "sin_10");
    check(COS_10, std::cos(10.0f), "cos_10");
    check(TAN_10, std::tan(10.0f), "tan_10");
    check(ASIN_0, std::asin(0.0f), "asin_0");
    check(ACOS_0, std::acos(0.0f), "acos_0");
    check(ASIN_0P5, std::asin(0.5f), "asin_0p5");
    check(ACOS_0P5, std::acos(0.5f), "acos_0p5");
    check(ASIN_1, std::asin(1.0f), "asin_1");
    check(ACOS_1, std::acos(1.0f), "acos_1");
    check(ASIN_N1, std::asin(-1.0f), "asin_n1");
    check(ACOS_N1, std::acos(-1.0f), "acos_n1");
    check(ASIN_N0P5, std::asin(-0.5f), "asin_n0p5");
    check(ACOS_N0P5, std::acos(-0.5f), "acos_n0p5");
    check(ATAN_0, std::atan(0.0f), "atan_0");
    check(ATAN_0P5, std::atan(0.5f), "atan_0p5");
    check(ATAN_1, std::atan(1.0f), "atan_1");
    check(ATAN_N1, std::atan(-1.0f), "atan_n1");
    check(ATAN_10, std::atan(10.0f), "atan_10");
    check(ATAN_N10, std::atan(-10.0f), "atan_n10");
    check(ACOSH_1, std::acosh(1.0f), "acosh_1");
    check(ACOSH_2, std::acosh(2.0f), "acosh_2");
    check(ACOSH_10, std::acosh(10.0f), "acosh_10");
    check(ASINH_0, std::asinh(0.0f), "asinh_0");
    check(ASINH_1, std::asinh(1.0f), "asinh_1");
    check(ASINH_N1, std::asinh(-1.0f), "asinh_n1");
    check(ASINH_10, std::asinh(10.0f), "asinh_10");
    check(ASINH_N10, std::asinh(-10.0f), "asinh_n10");
    check(ATANH_0, std::atanh(0.0f), "atanh_0");
    check(ATANH_0P5, std::atanh(0.5f), "atanh_0p5");
    check(ATANH_N0P5, std::atanh(-0.5f), "atanh_n0p5");
    check(ATANH_0P99, std::atanh(0.99f), "atanh_0p99");
    check(ATANH_N0P99, std::atanh(-0.99f), "atanh_n0p99");
    check(EXP_0, std::exp(0.0f), "exp_0");
    check(EXP2_0, std::exp2(0.0f), "exp2_0");
    check(EXP_1, std::exp(1.0f), "exp_1");
    check(EXP2_1, std::exp2(1.0f), "exp2_1");
    check(EXP_N1, std::exp(-1.0f), "exp_n1");
    check(EXP2_N1, std::exp2(-1.0f), "exp2_n1");
    check(EXP_10, std::exp(10.0f), "exp_10");
    check(EXP2_10, std::exp2(10.0f), "exp2_10");
    check(EXP_N10, std::exp(-10.0f), "exp_n10");
    check(EXP2_N10, std::exp2(-10.0f), "exp2_n10");
    check(EXP10_0, std::pow(10.0f, 0.0f), "exp10_0");
    check(EXP10_1, std::pow(10.0f, 1.0f), "exp10_1");
    check(EXP10_N1, std::pow(10.0f, -1.0f), "exp10_n1");
    check(EXP10_5, std::pow(10.0f, 5.0f), "exp10_5");
    check(EXP10_N5, std::pow(10.0f, -5.0f), "exp10_n5");
    check(LOG_1, std::log(1.0f), "log_1");
    check(LOG2_1, std::log2(1.0f), "log2_1");
    check(LOG10_1, std::log10(1.0f), "log10_1");
    check(LOG_E, std::log(2.718281828459045f), "log_e");
    check(LOG2_E, std::log2(2.718281828459045f), "log2_e");
    check(LOG10_E, std::log10(2.718281828459045f), "log10_e");
    check(LOG_2, std::log(2.0f), "log_2");
    check(LOG2_2, std::log2(2.0f), "log2_2");
    check(LOG10_2, std::log10(2.0f), "log10_2");
    check(LOG_10, std::log(10.0f), "log_10");
    check(LOG2_10, std::log2(10.0f), "log2_10");
    check(LOG10_10, std::log10(10.0f), "log10_10");
    check(LOG_100, std::log(100.0f), "log_100");
    check(LOG2_100, std::log2(100.0f), "log2_100");
    check(LOG10_100, std::log10(100.0f), "log10_100");
    check(LOG_0P1, std::log(0.1f), "log_0p1");
    check(LOG2_0P1, std::log2(0.1f), "log2_0p1");
    check(LOG10_0P1, std::log10(0.1f), "log10_0p1");
    check(SQRT_0, std::sqrt(0.0f), "sqrt_0");
    check(SQRT_1, std::sqrt(1.0f), "sqrt_1");
    check(SQRT_4, std::sqrt(4.0f), "sqrt_4");
    check(SQRT_0P25, std::sqrt(0.25f), "sqrt_0p25");
    check(SQRT_100, std::sqrt(100.0f), "sqrt_100");
    check(RSQRT_1, 1.0f / std::sqrt(1.0f), "rsqrt_1");
    check(RSQRT_4, 1.0f / std::sqrt(4.0f), "rsqrt_4");
    check(RSQRT_0P25, 1.0f / std::sqrt(0.25f), "rsqrt_0p25");
    check(RSQRT_100, 1.0f / std::sqrt(100.0f), "rsqrt_100");
    check_exact(FLOOR_1P7, std::floor(1.7f), "floor_1p7");
    check_exact(CEIL_1P7, std::ceil(1.7f), "ceil_1p7");
    check_exact(TRUNC_1P7, std::trunc(1.7f), "trunc_1p7");
    check_exact(FLOOR_1P3, std::floor(1.3f), "floor_1p3");
    check_exact(CEIL_1P3, std::ceil(1.3f), "ceil_1p3");
    check_exact(TRUNC_1P3, std::trunc(1.3f), "trunc_1p3");
    check_exact(FLOOR_N1P7, std::floor(-1.7f), "floor_n1p7");
    check_exact(CEIL_N1P7, std::ceil(-1.7f), "ceil_n1p7");
    check_exact(TRUNC_N1P7, std::trunc(-1.7f), "trunc_n1p7");
    check_exact(FLOOR_N1P3, std::floor(-1.3f), "floor_n1p3");
    check_exact(CEIL_N1P3, std::ceil(-1.3f), "ceil_n1p3");
    check_exact(TRUNC_N1P3, std::trunc(-1.3f), "trunc_n1p3");
    check_exact(FLOOR_2, std::floor(2.0f), "floor_2");
    check_exact(CEIL_2, std::ceil(2.0f), "ceil_2");
    check_exact(TRUNC_2, std::trunc(2.0f), "trunc_2");
    check_exact(FLOOR_2P5, std::floor(2.5f), "floor_2p5");
    check_exact(CEIL_2P5, std::ceil(2.5f), "ceil_2p5");
    check_exact(TRUNC_2P5, std::trunc(2.5f), "trunc_2p5");
    check_exact(FLOOR_N2P5, std::floor(-2.5f), "floor_n2p5");
    check_exact(CEIL_N2P5, std::ceil(-2.5f), "ceil_n2p5");
    check_exact(TRUNC_N2P5, std::trunc(-2.5f), "trunc_n2p5");
    check_exact(ROUND_1P7, std::round(1.7f), "round_1p7");
    check_exact(ROUND_1P3, std::round(1.3f), "round_1p3");
    check_exact(ROUND_N1P7, std::round(-1.7f), "round_n1p7");
    check_exact(ROUND_N1P3, std::round(-1.3f), "round_n1p3");
    check_exact(ROUND_2, std::round(2.0f), "round_2");
    check_exact(ROUND_2P5, std::round(2.5f), "round_2p5");
    check_exact(ROUND_N2P5, std::round(-2.5f), "round_n2p5");
    check_exact(ROUND_3P5, std::round(3.5f), "round_3p5");
    check(FRACT_1P7, 1.7f - std::floor(1.7f), "fract_1p7");
    check(FRACT_1P3, 1.3f - std::floor(1.3f), "fract_1p3");
    check(FRACT_N1P7, -1.7f - std::floor(-1.7f), "fract_n1p7");
    check(FRACT_N1P3, -1.3f - std::floor(-1.3f), "fract_n1p3");
    check(FRACT_2, 2.0f - std::floor(2.0f), "fract_2");
    check_exact(ABS_4P5, std::abs(4.5f), "abs_4p5");
    check_exact(ABS_N4P5, std::abs(-4.5f), "abs_n4p5");
    check_exact(ABS_0, std::abs(0.0f), "abs_0");
    check_exact(ABS_N0, std::abs(-0.0f), "abs_n0");
    check_exact(SIGN_3, (3.0f > 0.0f ? 1.0f : (3.0f < 0.0f ? -1.0f : 0.0f)), "sign_3");
    check_exact(SIGN_N2, (-2.0f > 0.0f ? 1.0f : (-2.0f < 0.0f ? -1.0f : 0.0f)), "sign_n2");
    check_exact(SIGN_0, (0.0f > 0.0f ? 1.0f : (0.0f < 0.0f ? -1.0f : 0.0f)), "sign_0");
    check_exact(SIGN_N0, (-0.0f > 0.0f ? 1.0f : (-0.0f < 0.0f ? -1.0f : 0.0f)), "sign_n0");
    check_exact(SATURATE_N0P5, std::clamp(-0.5f, 0.0f, 1.0f), "saturate_n0p5");
    check_exact(SATURATE_0, std::clamp(0.0f, 0.0f, 1.0f), "saturate_0");
    check_exact(SATURATE_0P5, std::clamp(0.5f, 0.0f, 1.0f), "saturate_0p5");
    check_exact(SATURATE_1, std::clamp(1.0f, 0.0f, 1.0f), "saturate_1");
    check_exact(SATURATE_1P5, std::clamp(1.5f, 0.0f, 1.0f), "saturate_1p5");
    check(POW_2_3, std::pow(2.0f, 3.0f), "pow_2_3");
    check(POW_0_0, std::pow(0.0f, 0.0f), "pow_0_0");
    check(POW_2P5_1P5, std::pow(2.5f, 1.5f), "pow_2p5_1p5");
    check(POW_9_0P5, std::pow(9.0f, 0.5f), "pow_9_0p5");
    check(ATAN2_0_1, std::atan2(0.0f, 1.0f), "atan2_0_1");
    check(ATAN2_1_0, std::atan2(1.0f, 0.0f), "atan2_1_0");
    check(ATAN2_0_N1, std::atan2(0.0f, -1.0f), "atan2_0_n1");
    check(ATAN2_N1_0, std::atan2(-1.0f, 0.0f), "atan2_n1_0");
    check(ATAN2_1_1, std::atan2(1.0f, 1.0f), "atan2_1_1");
    check(ATAN2_N1_1, std::atan2(-1.0f, 1.0f), "atan2_n1_1");
    check(ATAN2_N1_N1, std::atan2(-1.0f, -1.0f), "atan2_n1_n1");
    check(ATAN2_1_N1, std::atan2(1.0f, -1.0f), "atan2_1_n1");
    check(ATAN2_0_0, std::atan2(0.0f, 0.0f), "atan2_0_0");
    check_exact(MIN_3_5, std::min(3.0f, 5.0f), "min_3_5");
    check_exact(MAX_3_5, std::max(3.0f, 5.0f), "max_3_5");
    check_exact(MIN_5_3, std::min(5.0f, 3.0f), "min_5_3");
    check_exact(MAX_5_3, std::max(5.0f, 3.0f), "max_5_3");
    check_exact(MIN_N3_N5, std::min(-3.0f, -5.0f), "min_n3_n5");
    check_exact(MAX_N3_N5, std::max(-3.0f, -5.0f), "max_n3_n5");
    check_exact(MIN_4_4, std::min(4.0f, 4.0f), "min_4_4");
    check_exact(MAX_4_4, std::max(4.0f, 4.0f), "max_4_4");
    check(FMA_2_3_4, std::fma(2.0f, 3.0f, 4.0f), "fma_2_3_4");
    check(FMA_N2_3_4, std::fma(-2.0f, 3.0f, 4.0f), "fma_n2_3_4");
    check(FMA_2_N3_N4, std::fma(2.0f, -3.0f, -4.0f), "fma_2_n3_n4");
    check_exact(STEP_BELOW, (0.4f >= 0.5f ? 1.0f : 0.0f), "step_below");
    check_exact(STEP_AT, (0.5f >= 0.5f ? 1.0f : 0.0f), "step_at");
    check_exact(STEP_ABOVE, (0.6f >= 0.5f ? 1.0f : 0.0f), "step_above");
    check_exact(COPYSIGN_3_N1, std::copysign(3.0f, -1.0f), "copysign_3_n1");
    check_exact(COPYSIGN_N3_1, std::copysign(-3.0f, 1.0f), "copysign_n3_1");
    check_exact(COPYSIGN_3_1, std::copysign(3.0f, 1.0f), "copysign_3_1");
    check_exact(COPYSIGN_N3_N1, std::copysign(-3.0f, -1.0f), "copysign_n3_n1");
    check_exact(COPYSIGN_0_N1, std::copysign(0.0f, -1.0f), "copysign_0_n1");
    check(MOD_P_P, 5.2f - 2.0f * std::floor(5.2f / 2.0f), "mod_p_p");
    check(FMOD_P_P, std::fmod(5.2f, 2.0f), "fmod_p_p");
    check(MOD_N_P, -5.2f - 2.0f * std::floor(-5.2f / 2.0f), "mod_n_p");
    check(FMOD_N_P, std::fmod(-5.2f, 2.0f), "fmod_n_p");
    check(MOD_P_N, 5.2f - -2.0f * std::floor(5.2f / -2.0f), "mod_p_n");
    check(FMOD_P_N, std::fmod(5.2f, -2.0f), "fmod_p_n");
    check(MOD_N_N, -5.2f - -2.0f * std::floor(-5.2f / -2.0f), "mod_n_n");
    check(FMOD_N_N, std::fmod(-5.2f, -2.0f), "fmod_n_n");
    check(SMOOTHSTEP_BELOW, 0.0f, "smoothstep_below");
    check(SMOOTHSTEP_AT_LEFT, 0.0f, "smoothstep_at_left");
    check(SMOOTHSTEP_MID, 0.5f, "smoothstep_mid");
    check(SMOOTHSTEP_AT_RIGHT, 1.0f, "smoothstep_at_right");
    check(SMOOTHSTEP_ABOVE, 1.0f, "smoothstep_above");
    check(DEGREES_0, 0.0f * (180.0f / std::numbers::pi_v<float>), "degrees_0");
    check(DEGREES_PI2, 1.5707963267948966f * (180.0f / std::numbers::pi_v<float>), "degrees_pi2");
    check(DEGREES_PI, 3.141592653589793f * (180.0f / std::numbers::pi_v<float>), "degrees_pi");
    check(DEGREES_2PI, 6.283185307179586f * (180.0f / std::numbers::pi_v<float>), "degrees_2pi");
    check(RADIANS_0, 0.0f * (std::numbers::pi_v<float> / 180.0f), "radians_0");
    check(RADIANS_90, 90.0f * (std::numbers::pi_v<float> / 180.0f), "radians_90");
    check(RADIANS_180, 180.0f * (std::numbers::pi_v<float> / 180.0f), "radians_180");
    check(RADIANS_360, 360.0f * (std::numbers::pi_v<float> / 180.0f), "radians_360");
    check_exact(CLAMP_BELOW, std::clamp(-1.0f, 0.0f, 1.0f), "clamp_below");
    check_exact(CLAMP_IN, std::clamp(0.5f, 0.0f, 1.0f), "clamp_in");
    check_exact(CLAMP_ABOVE, std::clamp(2.0f, 0.0f, 1.0f), "clamp_above");
    check(LERP_0, std::lerp(10.0f, 20.0f, 0.0f), "lerp_0");
    check(LERP_0P5, std::lerp(10.0f, 20.0f, 0.5f), "lerp_0p5");
    check(LERP_1, std::lerp(10.0f, 20.0f, 1.0f), "lerp_1");
    check(LERP_N0P5, std::lerp(10.0f, 20.0f, -0.5f), "lerp_n0p5");
    check(LERP_1P5, std::lerp(10.0f, 20.0f, 1.5f), "lerp_1p5");
    check_exact(DOT_ORTHO, 0.0f, "dot_ortho");
    check_exact(DOT_STD, 32.0f, "dot_std");
    check_exact(DOT_ANTI, -14.0f, "dot_anti");
    check_exact(CROSS_ORTHO_X, 0.0f, "cross_ortho_x");
    check_exact(CROSS_ORTHO_Y, 0.0f, "cross_ortho_y");
    check_exact(CROSS_ORTHO_Z, 1.0f, "cross_ortho_z");
    check_exact(CROSS_STD_X, -3.0f, "cross_std_x");
    check_exact(CROSS_STD_Y, 6.0f, "cross_std_y");
    check_exact(CROSS_STD_Z, -3.0f, "cross_std_z");
    check_exact(CROSS_PAR_X, 0.0f, "cross_par_x");
    check_exact(CROSS_PAR_Y, 0.0f, "cross_par_y");
    check_exact(CROSS_PAR_Z, 0.0f, "cross_par_z");
    check_exact(LENGTH_UNIT, 1.0f, "length_unit");
    check_exact(LENGTH_ZERO, 0.0f, "length_zero");
    check(LENGTH_STD, std::sqrt(14.0f), "length_std");
    check_exact(LENGTH_SQ_UNIT, 1.0f, "length_sq_unit");
    check_exact(LENGTH_SQ_ZERO, 0.0f, "length_sq_zero");
    check_exact(LENGTH_SQ_STD, 14.0f, "length_sq_std");
    check(NORM_STD_X, 1.0f / std::sqrt(14.0f), "norm_std_x");
    check(NORM_STD_Y, 2.0f / std::sqrt(14.0f), "norm_std_y");
    check(NORM_STD_Z, 3.0f / std::sqrt(14.0f), "norm_std_z");
    check_exact(NORM_UNIT_X, 1.0f, "norm_unit_x");
    check_exact(NORM_UNIT_Y, 0.0f, "norm_unit_y");
    check_exact(NORM_UNIT_Z, 0.0f, "norm_unit_z");
    check_exact(DIST_SAME, 0.0f, "dist_same");
    check(DIST_DIFF, std::sqrt(27.0f), "dist_diff");
    check_exact(DIST_SQ_SAME, 0.0f, "dist_sq_same");
    check_exact(DIST_SQ_DIFF, 27.0f, "dist_sq_diff");
    check_exact(REFLECT_1_X, 1.0f, "reflect_1_x");
    check_exact(REFLECT_1_Y, 1.0f, "reflect_1_y");
    check_exact(REFLECT_1_Z, 0.0f, "reflect_1_z");
    check_exact(REFLECT_2_X, 1.0f, "reflect_2_x");
    check_exact(REFLECT_2_Y, -1.0f, "reflect_2_y");
    check_exact(REFLECT_2_Z, 0.0f, "reflect_2_z");
    check_exact(FF_SAME_X, 0.0f, "ff_same_x");
    check_exact(FF_SAME_Y, 1.0f, "ff_same_y");
    check_exact(FF_SAME_Z, 0.0f, "ff_same_z");
    check_exact(FF_OPP_X, 0.0f, "ff_opp_x");
    check_exact(FF_OPP_Y, -1.0f, "ff_opp_y");
    check_exact(FF_OPP_Z, 0.0f, "ff_opp_z");
    check_exact(RED_SUM_2, 3.0f, "red_sum_2");
    check_exact(RED_SUM_3, 6.0f, "red_sum_3");
    check_exact(RED_SUM_4, 10.0f, "red_sum_4");
    check_exact(RED_PROD_2, 2.0f, "red_prod_2");
    check_exact(RED_PROD_3, 6.0f, "red_prod_3");
    check_exact(RED_PROD_4, 24.0f, "red_prod_4");
    check_exact(RED_MIN_2, 1.0f, "red_min_2");
    check_exact(RED_MIN_3, 1.0f, "red_min_3");
    check_exact(RED_MIN_4, 1.0f, "red_min_4");
    check_exact(RED_MAX_2, 2.0f, "red_max_2");
    check_exact(RED_MAX_3, 3.0f, "red_max_3");
    check_exact(RED_MAX_4, 4.0f, "red_max_4");
    check_exact(ALL_TT, true ? 1.0f : 0.0f, "all_tt");
    check_exact(ALL_TFT, false ? 1.0f : 0.0f, "all_tft");
    check_exact(ANY_FF, false ? 1.0f : 0.0f, "any_ff");
    check_exact(ANY_FTF, true ? 1.0f : 0.0f, "any_ftf");
    check_exact(NONE_FF, true ? 1.0f : 0.0f, "none_ff");
    check_exact(NONE_TFF, false ? 1.0f : 0.0f, "none_tff");
    check_exact(DET_ID, 1.0f, "det_id");
    check_exact(DET_SCALE, 24.0f, "det_scale");
    check_exact(DET_SING, 0.0f, "det_sing");
    check_exact(TRANS_0_1, 4.0f, "trans_0_1");
    check_exact(TRANS_1_0, 2.0f, "trans_1_0");
    check_exact(INV_SCALE_0_0, 0.5f, "inv_scale_0_0");
    check(INV_SCALE_1_1, 1.0f / 3.0f, "inv_scale_1_1");
    check_exact(INV_SCALE_2_2, 0.25f, "inv_scale_2_2");
    check_exact(ABS_I_7, static_cast<float>(std::abs(7)), "abs_i_7");
    check_exact(ABS_I_N7, static_cast<float>(std::abs(-7)), "abs_i_n7");
    check_exact(ABS_I_0, static_cast<float>(std::abs(0)), "abs_i_0");
    check_exact(ABS_I_N1000, static_cast<float>(std::abs(-1000)), "abs_i_n1000");
    check_exact(MIN_I_3_5, static_cast<float>(std::min(3, 5)), "min_i_3_5");
    check_exact(MAX_I_3_5, static_cast<float>(std::max(3, 5)), "max_i_3_5");
    check_exact(MIN_I_5_3, static_cast<float>(std::min(5, 3)), "min_i_5_3");
    check_exact(MAX_I_5_3, static_cast<float>(std::max(5, 3)), "max_i_5_3");
    check_exact(MIN_I_N3_N5, static_cast<float>(std::min(-3, -5)), "min_i_n3_n5");
    check_exact(MAX_I_N3_N5, static_cast<float>(std::max(-3, -5)), "max_i_n3_n5");
    check_exact(CLAMP_I_ABOVE, static_cast<float>(std::clamp(10, 0, 5)), "clamp_i_above");
    check_exact(CLAMP_I_BELOW, static_cast<float>(std::clamp(-1, 0, 5)), "clamp_i_below");
    check_exact(CLAMP_I_IN, static_cast<float>(std::clamp(3, 0, 5)), "clamp_i_in");
    check_exact(CLZ_0, static_cast<float>(32), "clz_0");
    check_exact(CTZ_0, static_cast<float>(32), "ctz_0");
    check_exact(POPCOUNT_0, static_cast<float>(0), "popcount_0");
    check_exact(REVERSE_0, static_cast<float>(0u), "reverse_0");
    check_exact(CLZ_1, static_cast<float>(31), "clz_1");
    check_exact(CTZ_1, static_cast<float>(0), "ctz_1");
    check_exact(POPCOUNT_1, static_cast<float>(1), "popcount_1");
    check_exact(REVERSE_1, static_cast<float>(2147483648u), "reverse_1");
    check_exact(CLZ_16, static_cast<float>(27), "clz_16");
    check_exact(CTZ_16, static_cast<float>(4), "ctz_16");
    check_exact(POPCOUNT_16, static_cast<float>(1), "popcount_16");
    check_exact(REVERSE_16, static_cast<float>(134217728u), "reverse_16");
    check_exact(CLZ_ALL1, static_cast<float>(0), "clz_all1");
    check_exact(CTZ_ALL1, static_cast<float>(0), "ctz_all1");
    check_exact(POPCOUNT_ALL1, static_cast<float>(32), "popcount_all1");
    check_exact(REVERSE_ALL1, static_cast<float>(4294967295u), "reverse_all1");
    check_exact(CLZ_PATTERN, static_cast<float>(4), "clz_pattern");
    check_exact(CTZ_PATTERN, static_cast<float>(0), "ctz_pattern");
    check_exact(POPCOUNT_PATTERN, static_cast<float>(16), "popcount_pattern");
    check_exact(REVERSE_PATTERN, static_cast<float>(4042322160u), "reverse_pattern");
    check_exact(ISINF_INF, true ? 1.0f : 0.0f, "isinf_inf");
    check_exact(ISINF_NINF, true ? 1.0f : 0.0f, "isinf_ninf");
    check_exact(ISINF_NORMAL, false ? 1.0f : 0.0f, "isinf_normal");
    check_exact(ISINF_NAN, false ? 1.0f : 0.0f, "isinf_nan");
    check_exact(ISNAN_NAN, true ? 1.0f : 0.0f, "isnan_nan");
    check_exact(ISNAN_NORMAL, false ? 1.0f : 0.0f, "isnan_normal");
    check_exact(ISNAN_INF, false ? 1.0f : 0.0f, "isnan_inf");
    check_exact(SELECT_T, 2.0f, "select_t");
    check_exact(SELECT_F, 1.0f, "select_f");
    check_exact(ITE_T, 2.0f, "ite_t");
    check_exact(ITE_F, 1.0f, "ite_f");

    LUISA_INFO("All device math tests passed!");
    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_device_math(device);
}
