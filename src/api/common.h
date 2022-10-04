#pragma once
#include <core/platform.h>
#include <stdint.h>
#include <stddef.h>

// Uppercase names prefixed with underscores are reserved for the standard library.
#define LUISA_API_DECL_TYPE(TypeName) \
    typedef struct TypeName##_st {    \
        uint64_t __dummy;             \
    } TypeName##_st;                  \
    typedef TypeName##_st *TypeName

LUISA_API_DECL_TYPE(LCKernel);
LUISA_API_DECL_TYPE(LCFunction);
LUISA_API_DECL_TYPE(LCCallable);
LUISA_API_DECL_TYPE(LCType);
LUISA_API_DECL_TYPE(LCExpression);
LUISA_API_DECL_TYPE(LCConstantData);
LUISA_API_DECL_TYPE(LCStmt);

LUISA_API_DECL_TYPE(LCContext);
LUISA_API_DECL_TYPE(LCDevice);
LUISA_API_DECL_TYPE(LCShader);
LUISA_API_DECL_TYPE(LCBuffer);
LUISA_API_DECL_TYPE(LCTexture);
LUISA_API_DECL_TYPE(LCStream);
LUISA_API_DECL_TYPE(LCEvent);
LUISA_API_DECL_TYPE(LCCommandList);
LUISA_API_DECL_TYPE(LCCommand);
LUISA_API_DECL_TYPE(LCBindlessArray);
LUISA_API_DECL_TYPE(LCMesh);
LUISA_API_DECL_TYPE(LCAccel);

#undef LUISA_API_DECL_TYPE

typedef enum LCAccelUsageHint {
    LC_FAST_TRACE, // build with best quality
    LC_FAST_UPDATE,// optimize for frequent update, usually with compaction
    LC_FAST_BUILD  // optimize for frequent rebuild, maybe without compaction
} LCAccelUsageHint;
typedef enum LCAccelBuildRequest {
    LC_PREFER_UPDATE,
    LC_FORCE_BUILD,
} LCAccelBuildRequest;

typedef enum LCPixelStorage {

    LC_BYTE1,
    LC_BYTE2,
    LC_BYTE4,

    LC_SHORT1,
    LC_SHORT2,
    LC_SHORT4,

    LC_INT1,
    LC_INT2,
    LC_INT4,

    LC_HALF1,
    LC_HALF2,
    LC_HALF4,

    LC_FLOAT1,
    LC_FLOAT2,
    LC_FLOAT4
} LCPixelStorage;

typedef enum LCPixelFormat {

    LC_R8SInt,
    LC_R8UInt,
    LC_R8UNorm,

    LC_RG8SInt,
    LC_RG8UInt,
    LC_RG8UNorm,

    LC_RGBA8SInt,
    LC_RGBA8UInt,
    LC_RGBA8UNorm,

    LC_R16SInt,
    LC_R16UInt,
    LC_R16UNorm,

    LC_RG16SInt,
    LC_RG16UInt,
    LC_RG16UNorm,

    LC_RGBA16SInt,
    LC_RGBA16UInt,
    LC_RGBA16UNorm,

    LC_R32SInt,
    LC_R32UInt,

    LC_RG32SInt,
    LC_RG32UInt,

    LC_RGBA32SInt,
    LC_RGBA32UInt,

    LC_R16F,
    LC_RG16F,
    LC_RGBA16F,

    LC_R32F,
    LC_RG32F,
    LC_RGBA32F
} LCPixelFormat;

typedef struct lc_uint3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} lc_uint3;

typedef enum LCAccelBuildModficationFlags {
    LC_ACCEL_MESH = 1u << 0u,
    LC_ACCEL_TRANSFORM = 1u << 1u,
    LC_ACCEL_VISIBILITY_ON = 1u << 2u,
    LC_ACCEL_VISIBILITY_OFF = 1u << 3u,
    LC_ACCEL_VISIBILITY = LC_ACCEL_VISIBILITY_ON | LC_ACCEL_VISIBILITY_OFF
} LCAccelBuildModficationFlags;

typedef struct LCAccelBuildModification {
    uint32_t index;
    LCAccelBuildModficationFlags flags;
    uint64_t mesh;
    float affine[12];
} LCAccelBuildModification;

/**
 * @brief Enum of unary operations.
 * 
 * Note: We deliberately support *NO* pre and postfix inc/dec operators to avoid possible abuse
 */
typedef enum LCUnaryOp {
    LC_OP_PLUS,
    LC_OP_MINUS,  // +x, -x
    LC_OP_NOT,    // !x
    LC_OP_BIT_NOT,// ~x
} LCUnaryOp;

/**
 * @brief Enum of binary operations
 * 
 */
typedef enum LCBinaryOp {

    // arithmetic
    LC_OP_ADD,
    LC_OP_SUB,
    LC_OP_MUL,
    LC_OP_DIV,
    LC_OP_MOD,
    LC_OP_BIT_AND,
    LC_OP_BIT_OR,
    LC_OP_BIT_XOR,
    LC_OP_SHL,
    LC_OP_SHR,
    LC_OP_AND,
    LC_OP_OR,

    // relational
    LC_OP_LESS,
    LC_OP_GREATER,
    LC_OP_LESS_EQUAL,
    LC_OP_GREATER_EQUAL,
    LC_OP_EQUAL,
    LC_OP_NOT_EQUAL
} LCBinaryOp;

/**
 * @brief Enum of call operations.
 * 
 */
typedef enum LCCallOp {

    LC_OP_CUSTOM,

    LC_OP_ALL,
    LC_ANY,

    LC_OP_SELECT,
    LC_OP_CLAMP,
    LC_OP_LERP,
    LC_OP_STEP,

    LC_OP_ABS,
    LC_OP_MIN,
    LC_OP_MAX,

    LC_OP_CLZ,
    LC_OP_CTZ,
    LC_OP_POPCOUNT,
    LC_OP_REVERSE,

    LC_OP_ISINF,
    LC_OP_ISNAN,

    LC_OP_ACOS,
    LC_OP_ACOSH,
    LC_OP_ASIN,
    LC_OP_ASINH,
    LC_OP_ATAN,
    LC_OP_ATAN2,
    LC_OP_ATANH,

    LC_OP_COS,
    LC_OP_COSH,
    LC_OP_SIN,
    LC_OP_SINH,
    LC_OP_TAN,
    LC_OP_TANH,

    LC_OP_EXP,
    LC_OP_EXP2,
    LC_OP_EXP10,
    LC_OP_LOG,
    LC_OP_LOG2,
    LC_OP_LOG10,
    LC_OP_POW,

    LC_OP_SQRT,
    LC_OP_RSQRT,

    LC_OP_CEIL,
    LC_OP_FLOOR,
    LC_OP_FRACT,
    LC_OP_TRUNC,
    LC_OP_ROUND,

    LC_OP_FMA,
    LC_OP_COPYSIGN,

    LC_OP_CROSS,
    LC_OP_DOT,
    LC_OP_LENGTH,
    LC_OP_LENGTH_SQUARED,
    LC_OP_NORMALIZE,
    LC_OP_FACEFORWARD,

    LC_OP_DETERMINANT,
    LC_OP_TRANSPOSE,
    LC_OP_INVERSE,

    LC_OP_SYNCHRONIZE_BLOCK,

    LC_OP_ATOMIC_EXCHANGE,        /// [(atomic_ref, desired) -> old]: stores desired, returns old.
    LC_OP_ATOMIC_COMPARE_EXCHANGE,/// [(atomic_ref, expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
    LC_OP_ATOMIC_FETCH_ADD,       /// [(atomic_ref, val) -> old]: stores (old + val), returns old.
    LC_OP_ATOMIC_FETCH_SUB,       /// [(atomic_ref, val) -> old]: stores (old - val), returns old.
    LC_OP_ATOMIC_FETCH_AND,       /// [(atomic_ref, val) -> old]: stores (old & val), returns old.
    LC_OP_ATOMIC_FETCH_OR,        /// [(atomic_ref, val) -> old]: stores (old | val), returns old.
    LC_OP_ATOMIC_FETCH_XOR,       /// [(atomic_ref, val) -> old]: stores (old ^ val), returns old.
    LC_OP_ATOMIC_FETCH_MIN,       /// [(atomic_ref, val) -> old]: stores min(old, val), returns old.
    LC_OP_ATOMIC_FETCH_MAX,       /// [(atomic_ref, val) -> old]: stores max(old, val), returns old.

    LC_OP_BUFFER_READ,  /// [(buffer, index) -> value]: reads the index-th element in buffer
    LC_OP_BUFFER_WRITE, /// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
    LC_OP_TEXTURE_READ, /// [(texture, coord) -> value]
    LC_OP_TEXTURE_WRITE,/// [(texture, coord, value) -> void]

    LC_OP_BINDLESS_TEXTURE2D_SAMPLE,      //(bindless_array, index: uint, uv: float2): float4
    LC_OP_BINDLESS_TEXTURE2D_SAMPLE_LEVEL,//(bindless_array, index: uint, uv: float2, level: float): float4
    LC_OP_BINDLESS_TEXTURE2D_SAMPLE_GRAD, //(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    LC_OP_BINDLESS_TEXTURE3D_SAMPLE,      //(bindless_array, index: uint, uv: float3): float4
    LC_OP_BINDLESS_TEXTURE3D_SAMPLE_LEVEL,//(bindless_array, index: uint, uv: float3, level: float): float4
    LC_OP_BINDLESS_TEXTURE3D_SAMPLE_GRAD, //(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    LC_OP_BINDLESS_TEXTURE2D_READ,        //(bindless_array, index: uint, coord: uint2): float4
    LC_OP_BINDLESS_TEXTURE3D_READ,        //(bindless_array, index: uint, coord: uint3): float4
    LC_OP_BINDLESS_TEXTURE2D_READ_LEVEL,  //(bindless_array, index: uint, coord: uint2, level: uint): float4
    LC_OP_BINDLESS_TEXTURE3D_READ_LEVEL,  //(bindless_array, index: uint, coord: uint3, level: uint): float4
    LC_OP_BINDLESS_TEXTURE2D_SIZE,        //(bindless_array, index: uint): uint2
    LC_OP_BINDLESS_TEXTURE3D_SIZE,        //(bindless_array, index: uint): uint3
    LC_OP_BINDLESS_TEXTURE2D_SIZE_LEVEL,  //(bindless_array, index: uint, level: uint): uint2
    LC_OP_BINDLESS_TEXTURE3D_SIZE_LEVEL,  //(bindless_array, index: uint, level: uint): uint3

    LC_OP_BINDLESS_BUFFER_READ,//(bindless_array, index: uint): expr->type()

    LC_OP_MAKE_BOOL2,
    LC_OP_MAKE_BOOL3,
    LC_OP_MAKE_BOOL4,
    LC_OP_MAKE_INT2,
    LC_OP_MAKE_INT3,
    LC_OP_MAKE_INT4,
    LC_OP_MAKE_UINT2,
    LC_OP_MAKE_UINT3,
    LC_OP_MAKE_UINT4,
    LC_OP_MAKE_FLOAT2,
    LC_OP_MAKE_FLOAT3,
    LC_OP_MAKE_FLOAT4,

    LC_OP_MAKE_FLOAT2X2,
    LC_OP_MAKE_FLOAT3X3,
    LC_OP_MAKE_FLOAT4X4,

    // optimization hints
    LC_OP_ASSUME,
    LC_OP_UNREACHABLE,

    LC_OP_INSTANCE_TO_WORLD_MATRIX,
    LC_OP_SET_INSTANCE_TRANSFORM,
    LC_OP_SET_INSTANCE_VISIBILITY,
    LC_OP_TRACE_CLOSEST,
    LC_OP_TRACE_ANY
} LCCallOp;

typedef enum LCCastOp {
    LC_OP_STATIC,
    LC_OP_BITWISE
} LCCastOp;

typedef enum LCSamplerFilter {
    LC_POINT,
    LC_LINEAR_POINT,
    LC_LINEAR_LINEAR,
    LC_ANISOTROPIC
} LCSamplerFilter;

typedef enum LCSamplerAddress {
    LC_EDGE,
    LC_REPEAT,
    LC_MIRROR,
    LC_ZERO
} LCSamplerAddress;

typedef struct LCSampler {
    LCSamplerFilter filter;
    LCSamplerAddress address;
} LCSampler;

