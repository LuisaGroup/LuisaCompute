#pragma once

#include <bitset>

#include <luisa/core/stl/iterator.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/magic_enum.h>

namespace luisa::compute {

class Type;

/**
 * @brief Enum of unary operations.
 * 
 * Note: We deliberately support *NO* pre and postfix inc/dec operators to avoid possible abuse
 */
enum struct UnaryOp : uint32_t {
    PLUS,   // +x
    MINUS,  // -x
    NOT,    // !x
    BIT_NOT,// ~x
};

/**
 * @brief Enum of binary operations
 * 
 */
enum struct BinaryOp : uint32_t {

    // arithmetic
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    SHL,
    SHR,
    AND,
    OR,

    // relational
    LESS,
    GREATER,
    LESS_EQUAL,
    GREATER_EQUAL,
    EQUAL,
    NOT_EQUAL
};

struct TypePromotion {
    const Type *lhs{nullptr};
    const Type *rhs{nullptr};
    const Type *result{nullptr};
};

[[nodiscard]] LUISA_AST_API TypePromotion promote_types(BinaryOp op, const Type *lhs, const Type *rhs) noexcept;

[[nodiscard]] constexpr auto is_relational(BinaryOp op) noexcept {
    return op == BinaryOp::LESS ||
           op == BinaryOp::GREATER ||
           op == BinaryOp::LESS_EQUAL ||
           op == BinaryOp::GREATER_EQUAL ||
           op == BinaryOp::EQUAL ||
           op == BinaryOp::NOT_EQUAL;
}

[[nodiscard]] constexpr auto is_logical(BinaryOp op) noexcept {
    return op == BinaryOp::AND || op == BinaryOp::OR;
}

/**
 * @brief Enum of call operations.
 * 
 */
enum struct CallOp : uint32_t {

    CUSTOM,
    EXTERNAL,

    ALL,// (boolN)
    ANY,// (boolN)

    SELECT,  // (vecN, vecN, boolN)
    CLAMP,   // (vecN, vecN, vecN)
    SATURATE,// (vecN)
    LERP,    // (vecN, vecN, vecN)

    SMOOTHSTEP,// (vecN, vecN, vecN)
    STEP,      // (x, y): (x >= y) ? 1 : 0

    ABS,// (vecN)
    MIN,// (vecN)
    MAX,// (vecN)

    CLZ,     // (uintN)
    CTZ,     // (uintN)
    POPCOUNT,// (uintN)
    REVERSE, // (uintN)

    ISINF,// (floatN)
    ISNAN,// (floatN)

    ACOS, // (floatN)
    ACOSH,// (floatN)
    ASIN, // (floatN)
    ASINH,// (floatN)
    ATAN, // (floatN)
    ATAN2,// (floatN)
    ATANH,// (floatN)

    COS, // (floatN)
    COSH,// (floatN)
    SIN, // (floatN)
    SINH,// (floatN)
    TAN, // (floatN)
    TANH,// (floatN)

    EXP,  // (floatN)
    EXP2, // (floatN)
    EXP10,// (floatN)
    LOG,  // (floatN)
    LOG2, // (floatN)
    LOG10,// (floatN)
    POW,  // (floatN)

    SQRT, // (floatN)
    RSQRT,// (floatN)

    CEIL, // (floatN)
    FLOOR,// (floatN)
    FRACT,// (floatN)
    TRUNC,// (floatN)
    ROUND,// (floatN)

    FMA,     // (a: floatN, b: floatN, c: floatN): return a * b + c
    COPYSIGN,// (floatN, floatN)

    CROSS,         // (floatN, floatN)
    DOT,           // (floatN, floatN)
    LENGTH,        // (floatN)
    LENGTH_SQUARED,// (floatN)
    NORMALIZE,     // (floatN)
    FACEFORWARD,   // (floatN, floatN, floatN)
    REFLECT,       // (floatN, floatN)

    REDUCE_SUM,    // (floatN)
    REDUCE_PRODUCT,// (floatN)
    REDUCE_MIN,    // (floatN)
    REDUCE_MAX,    // (floatN)

    OUTER_PRODUCT,                       // (floatN | floatNxN)
    MATRIX_COMPONENT_WISE_MULTIPLICATION,// (floatNxN)
    DETERMINANT,                         // (floatNxN)
    TRANSPOSE,                           // (floatNxN)
    INVERSE,                             // (floatNxN)

    SYNCHRONIZE_BLOCK,// ()

    ATOMIC_EXCHANGE,        /// [(atomic_ref, desired) -> old]: stores desired, returns old.
    ATOMIC_COMPARE_EXCHANGE,/// [(atomic_ref, expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
    ATOMIC_FETCH_ADD,       /// [(atomic_ref, val) -> old]: stores (old + val), returns old.
    ATOMIC_FETCH_SUB,       /// [(atomic_ref, val) -> old]: stores (old - val), returns old.
    ATOMIC_FETCH_AND,       /// [(atomic_ref, val) -> old]: stores (old & val), returns old.
    ATOMIC_FETCH_OR,        /// [(atomic_ref, val) -> old]: stores (old | val), returns old.
    ATOMIC_FETCH_XOR,       /// [(atomic_ref, val) -> old]: stores (old ^ val), returns old.
    ATOMIC_FETCH_MIN,       /// [(atomic_ref, val) -> old]: stores min(old, val), returns old.
    ATOMIC_FETCH_MAX,       /// [(atomic_ref, val) -> old]: stores max(old, val), returns old.

    BUFFER_VOLATILE_READ, /// same as BUFFER_READ
    BUFFER_VOLATILE_WRITE,/// same as BUFFER_WRITE

    ADDRESS_OF,// (expr) -> uint64

    BUFFER_READ,   /// [(buffer, index) -> value]: reads the index-th element in buffer
    BUFFER_WRITE,  /// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
    BUFFER_SIZE,   /// [(buffer) -> size]
    BUFFER_ADDRESS,/// [(buffer) -> address]

    BYTE_BUFFER_READ,          /// [(buffer, byte_index) -> value]: reads the index-th element in buffer
    BYTE_BUFFER_VOLATILE_READ, /// [(buffer, byte_index) -> value]: reads the index-th element in buffer
    BYTE_BUFFER_WRITE,         /// [(buffer, byte_index, value) -> void]: writes value into the index-th element of buffer
    BYTE_BUFFER_VOLATILE_WRITE,/// [(buffer, byte_index, value) -> void]: writes value into the index-th element of buffer
    BYTE_BUFFER_SIZE,          /// [(buffer) -> size_bytes]

    TEXTURE_READ, /// [(texture, coord) -> value]
    TEXTURE_WRITE,/// [(texture, coord, value) -> void]
    TEXTURE_SIZE, /// [(texture) -> Vector<uint, dim>]

    BINDLESS_TEXTURE2D_SAMPLE,           // (bindless_array, index: uint, uv: float2): float4
    BINDLESS_TEXTURE2D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float2, level: float): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float): float4
    BINDLESS_TEXTURE3D_SAMPLE,           // (bindless_array, index: uint, uv: float3): float4
    BINDLESS_TEXTURE3D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float3, level: float): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float): float4

    BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float2, filter: uint, address: uint): float4
    BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float2, level: float, filter: uint, address: uint): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, filter: uint, address: uint): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float, filter: uint, address: uint): float4
    BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float3, filter: uint, address: uint): float4
    BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float3, level: float, filter: uint, address: uint): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3, filter: uint, address: uint): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float, filter: uint, address: uint): float4

    BINDLESS_TEXTURE2D_READ,      // (bindless_array, index: uint, coord: uint2): float4
    BINDLESS_TEXTURE3D_READ,      // (bindless_array, index: uint, coord: uint3): float4
    BINDLESS_TEXTURE2D_READ_LEVEL,// (bindless_array, index: uint, coord: uint2, level: uint): float4
    BINDLESS_TEXTURE3D_READ_LEVEL,// (bindless_array, index: uint, coord: uint3, level: uint): float4
    BINDLESS_TEXTURE2D_SIZE,      // (bindless_array, index: uint): uint2
    BINDLESS_TEXTURE3D_SIZE,      // (bindless_array, index: uint): uint3
    BINDLESS_TEXTURE2D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint2
    BINDLESS_TEXTURE3D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint3

    BINDLESS_BUFFER_READ,     // (bindless_array, index: uint, elem_index: uint): expr->type()
    BINDLESS_BUFFER_WRITE,    // (bindless_array, index: uint, elem_index: uint, value: expr): void
    BINDLESS_BYTE_BUFFER_READ,// (bindless_array, index: uint, offset_bytes: uint): expr->type()
    BINDLESS_BUFFER_SIZE,     // (bindless_array, index: uint, stride: uint) -> size
    BINDLESS_BUFFER_TYPE,     // (bindless_array, index: uint) -> uint64 (type id of the element); the returned value
                              // could be compared with the value of a TypeIDExpr to examine the type of the buffer
    BINDLESS_BUFFER_ADDRESS,  // (bindless_array, index: uint) -> uint64 (address of the buffer)

    // Block-uniform typed bindless
    UNIFORM_BINDLESS_TEXTURE2D_SAMPLE,           // (bindless_array, index: uint, uv: float2): float4
    UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float2, level: float): float4
    UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float): float4
    UNIFORM_BINDLESS_TEXTURE3D_SAMPLE,           // (bindless_array, index: uint, uv: float3): float4
    UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float3, level: float): float4
    UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float): float4

    UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float2, filter: uint, address: uint): float4
    UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float2, level: float, filter: uint, address: uint): float4
    UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, filter: uint, address: uint): float4
    UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float, filter: uint, address: uint): float4
    UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float3, filter: uint, address: uint): float4
    UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float3, level: float, filter: uint, address: uint): float4
    UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3, filter: uint, address: uint): float4
    UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float, filter: uint, address: uint): float4

    UNIFORM_BINDLESS_TEXTURE2D_READ,      // (bindless_array, index: uint, coord: uint2): float4
    UNIFORM_BINDLESS_TEXTURE3D_READ,      // (bindless_array, index: uint, coord: uint3): float4
    UNIFORM_BINDLESS_TEXTURE2D_READ_LEVEL,// (bindless_array, index: uint, coord: uint2, level: uint): float4
    UNIFORM_BINDLESS_TEXTURE3D_READ_LEVEL,// (bindless_array, index: uint, coord: uint3, level: uint): float4
    UNIFORM_BINDLESS_TEXTURE2D_SIZE,      // (bindless_array, index: uint): uint2
    UNIFORM_BINDLESS_TEXTURE3D_SIZE,      // (bindless_array, index: uint): uint3
    UNIFORM_BINDLESS_TEXTURE2D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint2
    UNIFORM_BINDLESS_TEXTURE3D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint3

    UNIFORM_BINDLESS_BUFFER_READ,     // (bindless_array, index: uint, elem_index: uint): expr->type()
    UNIFORM_BINDLESS_BUFFER_WRITE,    // (bindless_array, index: uint, elem_index: uint, value: expr): void
    UNIFORM_BINDLESS_BYTE_BUFFER_READ,// (bindless_array, index: uint, offset_bytes: uint): expr->type()
    UNIFORM_BINDLESS_BUFFER_SIZE,     // (bindless_array, index: uint, stride: uint) -> size
    UNIFORM_BINDLESS_BUFFER_TYPE,     // (bindless_array, index: uint) -> uint64 (type id of the element); the returned value
    UNIFORM_BINDLESS_BUFFER_ADDRESS,  // (bindless_array, index: uint) -> uint64 (address of the buffer)

    // Block-uniform typed bindless
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE,           // (bindless_array, index: uint, uv: float2): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float2, level: float): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE,           // (bindless_array, index: uint, uv: float3): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float3, level: float): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float): float4

    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float2, filter: uint, address: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float2, level: float, filter: uint, address: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, filter: uint, address: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float, filter: uint, address: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float3, filter: uint, address: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float3, level: float, filter: uint, address: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3, filter: uint, address: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float, filter: uint, address: uint): float4

    TYPED_UNIFORM_BINDLESS_TEXTURE2D_READ,      // (bindless_array, index: uint, coord: uint2): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_READ,      // (bindless_array, index: uint, coord: uint3): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_READ_LEVEL,// (bindless_array, index: uint, coord: uint2, level: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_READ_LEVEL,// (bindless_array, index: uint, coord: uint3, level: uint): float4
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SIZE,      // (bindless_array, index: uint): uint2
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SIZE,      // (bindless_array, index: uint): uint3
    TYPED_UNIFORM_BINDLESS_TEXTURE2D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint2
    TYPED_UNIFORM_BINDLESS_TEXTURE3D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint3

    TYPED_UNIFORM_BINDLESS_BUFFER_READ,     // (bindless_array, index: uint, elem_index: uint): expr->type()
    TYPED_UNIFORM_BINDLESS_BUFFER_WRITE,    // (bindless_array, index: uint, elem_index: uint, value: expr): void
    TYPED_UNIFORM_BINDLESS_BYTE_BUFFER_READ,// (bindless_array, index: uint, offset_bytes: uint): expr->type()
    TYPED_UNIFORM_BINDLESS_BUFFER_SIZE,     // (bindless_array, index: uint, stride: uint) -> size
    TYPED_UNIFORM_BINDLESS_BUFFER_TYPE,     // (bindless_array, index: uint) -> uint64 (type id of the element); the returned value
    TYPED_UNIFORM_BINDLESS_BUFFER_ADDRESS,  // (bindless_array, index: uint) -> uint64 (address of the buffer)

    // Non uniform typed
    TYPED_BINDLESS_TEXTURE2D_SAMPLE,           // (bindless_array, index: uint, uv: float2): float4
    TYPED_BINDLESS_TEXTURE2D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float2, level: float): float4
    TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float): float4
    TYPED_BINDLESS_TEXTURE3D_SAMPLE,           // (bindless_array, index: uint, uv: float3): float4
    TYPED_BINDLESS_TEXTURE3D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float3, level: float): float4
    TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float): float4

    TYPED_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float2, filter: uint, address: uint): float4
    TYPED_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float2, level: float, filter: uint, address: uint): float4
    TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, filter: uint, address: uint): float4
    TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float, filter: uint, address: uint): float4
    TYPED_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float3, filter: uint, address: uint): float4
    TYPED_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float3, level: float, filter: uint, address: uint): float4
    TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3, filter: uint, address: uint): float4
    TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float, filter: uint, address: uint): float4

    TYPED_BINDLESS_TEXTURE2D_READ,      // (bindless_array, index: uint, coord: uint2): float4
    TYPED_BINDLESS_TEXTURE3D_READ,      // (bindless_array, index: uint, coord: uint3): float4
    TYPED_BINDLESS_TEXTURE2D_READ_LEVEL,// (bindless_array, index: uint, coord: uint2, level: uint): float4
    TYPED_BINDLESS_TEXTURE3D_READ_LEVEL,// (bindless_array, index: uint, coord: uint3, level: uint): float4
    TYPED_BINDLESS_TEXTURE2D_SIZE,      // (bindless_array, index: uint): uint2
    TYPED_BINDLESS_TEXTURE3D_SIZE,      // (bindless_array, index: uint): uint3
    TYPED_BINDLESS_TEXTURE2D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint2
    TYPED_BINDLESS_TEXTURE3D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint3

    TYPED_BINDLESS_BUFFER_READ,     // (bindless_array, index: uint, elem_index: uint): expr->type()
    TYPED_BINDLESS_BUFFER_WRITE,    // (bindless_array, index: uint, elem_index: uint, value: expr): void
    TYPED_BINDLESS_BYTE_BUFFER_READ,// (bindless_array, index: uint, offset_bytes: uint): expr->type()
    TYPED_BINDLESS_BUFFER_SIZE,     // (bindless_array, index: uint, stride: uint) -> size
    TYPED_BINDLESS_BUFFER_TYPE,     // (bindless_array, index: uint) -> uint64 (type id of the element); the returned value
    TYPED_BINDLESS_BUFFER_ADDRESS,  // (bindless_array, index: uint) -> uint64 (address of the buffer)

    MAKE_BOOL2, // (bool, bool2)
    MAKE_BOOL3, // (bool, bool3)
    MAKE_BOOL4, // (bool, bool4)
    MAKE_INT2,  // (scalar, vec2)
    MAKE_INT3,  // (scalar, vec3)
    MAKE_INT4,  // (scalar, vec4)
    MAKE_UINT2, // (scalar, vec2)
    MAKE_UINT3, // (scalar, vec3)
    MAKE_UINT4, // (scalar, vec4)
    MAKE_FLOAT2,// (scalar, vec2)
    MAKE_FLOAT3,// (scalar, vec3)
    MAKE_FLOAT4,// (scalar, vec4)

    MAKE_SHORT2, // (scalar, vec2)
    MAKE_SHORT3, // (scalar, vec3)
    MAKE_SHORT4, // (scalar, vec4)
    MAKE_USHORT2,// (scalar, vec2)
    MAKE_USHORT3,// (scalar, vec3)
    MAKE_USHORT4,// (scalar, vec4)
    MAKE_LONG2,  // (scalar, vec2)
    MAKE_LONG3,  // (scalar, vec3)
    MAKE_LONG4,  // (scalar, vec4)
    MAKE_ULONG2, // (scalar, vec2)
    MAKE_ULONG3, // (scalar, vec3)
    MAKE_ULONG4, // (scalar, vec4)
    MAKE_HALF2,  // (scalar, vec2)
    MAKE_HALF3,  // (scalar, vec3)
    MAKE_HALF4,  // (scalar, vec4)
    MAKE_DOUBLE2,// (scalar, vec2)
    MAKE_DOUBLE3,// (scalar, vec3)
    MAKE_DOUBLE4,// (scalar, vec4)
    MAKE_BYTE2,  // (scalar, vec2)
    MAKE_BYTE3,  // (scalar, vec3)
    MAKE_BYTE4,  // (scalar, vec4)
    MAKE_UBYTE2, // (scalar, vec2)
    MAKE_UBYTE3, // (scalar, vec3)
    MAKE_UBYTE4, // (scalar, vec4)

    MAKE_FLOAT2X2,// (float2x2) / (float3x3) / (float4x4)
    MAKE_FLOAT3X3,// (float2x2) / (float3x3) / (float4x4)
    MAKE_FLOAT4X4,// (float2x2) / (float3x3) / (float4x4)

    // debugging
    ASSERT,// (bool) -> void

    // optimization hints
    ASSUME,     // ()
    UNREACHABLE,// ()
    FLATTEN,    // for if-statement
    BRANCH,     // for if-statement
    FORCE_CASE, // for switch-statement

    // used by the IR module
    ZERO,
    ONE,

    // Pack/unpack to array<uint, ceil(sizeof(T)/4))
    PACK,  // (T) -> array<uint, ceil(sizeof(T)/4))
    UNPACK,// (array<uint, ceil(sizeof(T)/4)) -> T

    // autodiff ops
    REQUIRES_GRADIENT,  // (expr) -> void
    GRADIENT,           // (expr) -> expr
    GRADIENT_MARKER,    // (ref, expr) -> void
    ACCUMULATE_GRADIENT,// (ref, expr) -> void
    BACKWARD,           // (expr) -> void
    DETACH,             // (expr) -> expr

    // ray tracing
    RAY_TRACING_INSTANCE_TRANSFORM,      // (Accel, uint)
    RAY_TRACING_INSTANCE_USER_ID,        // (Accel, uint)
    RAY_TRACING_INSTANCE_VISIBILITY_MASK,// (Accel, uint)

    RAY_TRACING_SET_INSTANCE_TRANSFORM, // (Accel, uint, float4x4)
    RAY_TRACING_SET_INSTANCE_VISIBILITY,// (Accel, uint, uint)
    RAY_TRACING_SET_INSTANCE_OPACITY,   // (Accel, uint, bool)
    RAY_TRACING_SET_INSTANCE_USER_ID,   // (Accel, uint, uint)

    RAY_TRACING_TRACE_CLOSEST,// (Accel, ray, mask: uint): TriangleHit
    RAY_TRACING_TRACE_ANY,    // (Accel, ray, mask: uint): bool
    RAY_TRACING_QUERY_ALL,    // (Accel, ray, mask: uint): RayQuery
    RAY_TRACING_QUERY_ANY,    // (Accel, ray, mask: uint): RayQuery

    // ray tracing with motion blur
    RAY_TRACING_INSTANCE_MOTION_MATRIX,    // (Accel, index: uint, key: uint): float4x4
    RAY_TRACING_INSTANCE_MOTION_SRT,       // (Accel, index: uint, key: uint): SRT
    RAY_TRACING_SET_INSTANCE_MOTION_MATRIX,// (Accel, index: uint, key: uint, transform: float4x4)
    RAY_TRACING_SET_INSTANCE_MOTION_SRT,   // (Accel, index: uint, key: uint, transform: SRT)

    RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR,// (Accel, ray, time: float, mask: uint): TriangleHit
    RAY_TRACING_TRACE_ANY_MOTION_BLUR,    // (Accel, ray, time: float, mask: uint): bool
    RAY_TRACING_QUERY_ALL_MOTION_BLUR,    // (Accel, ray, time: float, mask: uint): RayQuery
    RAY_TRACING_QUERY_ANY_MOTION_BLUR,    // (Accel, ray, time: float, mask: uint): RayQuery

    // ray query
    RAY_QUERY_WORLD_SPACE_RAY,         // (RayQuery): Ray
    RAY_QUERY_PROCEDURAL_CANDIDATE_HIT,// (RayQuery): ProceduralHit
    RAY_QUERY_TRIANGLE_CANDIDATE_HIT,  // (RayQuery): TriangleHit
    RAY_QUERY_COMMITTED_HIT,           // (RayQuery): CommittedHit
    RAY_QUERY_COMMIT_TRIANGLE,         // (RayQuery): void
    RAY_QUERY_COMMIT_PROCEDURAL,       // (RayQuery, float): void
    RAY_QUERY_TERMINATE,               // (RayQuery): void

    // For REAL rayquery
    RAY_QUERY_PROCEED,
    RAY_QUERY_IS_TRIANGLE_CANDIDATE,
    RAY_QUERY_IS_PROCEDURAL_CANDIDATE,

    // rasterization
    RASTER_DISCARD,    // (): void  discard-pixel (only in pixel shader)
    // set z_depth (only in pixel shader, may disable early-z)
    RASTER_SET_Z_DEPTH,// (float): void 
    // set z_depth that assumed greater or equal than origin (only in pixel shader, without disable early-z)
    RASTER_SET_Z_DEPTH_GREATER_EQUAL,// (float): void 
    // set z_depth that assumed less or equal than origin (only in pixel shader, without disable early-z)
    RASTER_SET_Z_DEPTH_LESS_EQUAL,// (float): void

    // Derivative Operations for 2x2 quad
    // partial derivative
    DDX,// (arg: float vector): float vector
    DDY,// (arg: float vector): float vector

    // Wave:
    WARP_IS_FIRST_ACTIVE_LANE,  // (): bool
    WARP_FIRST_ACTIVE_LANE,     // (): uint
    WARP_ACTIVE_ALL_EQUAL,      // (scalar/vector): boolN
    WARP_ACTIVE_BIT_AND,        // (intN): intN
    WARP_ACTIVE_BIT_OR,         // (intN): intN
    WARP_ACTIVE_BIT_XOR,        // (intN): intN
    WARP_ACTIVE_COUNT_BITS,     // (bool): uint
    WARP_ACTIVE_MAX,            // (type: scalar/vector): type
    WARP_ACTIVE_MIN,            // (type: scalar/vector): type
    WARP_ACTIVE_PRODUCT,        // (type: scalar/vector): type
    WARP_ACTIVE_SUM,            // (type: scalar/vector): type
    WARP_ACTIVE_ALL,            // (bool): bool
    WARP_ACTIVE_ANY,            // (bool): bool
    WARP_ACTIVE_BIT_MASK,       // (bool): uint4 (uint4 contained 128-bit)
    WARP_PREFIX_COUNT_BITS,     // (bool): uint (count bits before this lane)
    WARP_PREFIX_SUM,            // (type: scalar/vector): type (sum lanes before this lane)
    WARP_PREFIX_PRODUCT,        // (type: scalar/vector): type (multiply lanes before this lane)
    WARP_READ_LANE,             // (type: scalar/vector/matrix, index: uint): type (read this variable's value at this lane)
    WARP_READ_FIRST_ACTIVE_LANE,// (type: scalar/vector/matrix): type (read this variable's value at the first lane)

    // indirect
    INDIRECT_SET_DISPATCH_KERNEL,// (Buffer, uint offset, uint3 block_size, uint3 dispatch_size, uint kernel_id)
    INDIRECT_SET_DISPATCH_COUNT, // (Buffer, uint count)

    // texture direct sample

    TEXTURE2D_SAMPLE,           // (tex, uv: float2, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_LEVEL,     // (tex, uv: float2, level: float, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_GRAD,      // (tex, uv: float2, ddx: float2, ddy: float2, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_GRAD_LEVEL,// (tex, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE,           // (tex, uv: float3, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_LEVEL,     // (tex, uv: float3, level: float, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_GRAD,      // (tex, uv: float3, ddx: float3, ddy: float3, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_GRAD_LEVEL,// (tex, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float, filter: uint, address: uint): float4

    // SER
    SHADER_EXECUTION_REORDER,// (uint hint, uint hint_bits): void

    // cooperative
    COOPERATIVE_MUL_ADD,                 // (coop_vec<OutType, M> (matrix_buffer: byte_buffer, matrix_offset: coop_mat_ref<N, M, CoopRefType>, bias_buffer: byte_buffer, bias_offset: coop_vec_ref<M, CoopRefType>, input_vector: coop_vec<N>)
    BINDLESS_COOPERATIVE_MUL_ADD,        // (coop_vec<OutType, M> (bindless_array, matrix_buffer: uint, matrix_offset: coop_mat_ref<N, M, CoopRefType>, bias_buffer: uint, bias_offset: coop_vec_ref<M, CoopRefType>, input_vector: coop_vec<N>)
    TYPED_BINDLESS_COOPERATIVE_MUL_ADD,  // (coop_vec<OutType, M> (bindless_array, matrix_buffer: uint, matrix_offset: coop_mat_ref<N, M, CoopRefType>, bias_buffer: uint, bias_offset: coop_vec_ref<M, CoopRefType>, input_vector: coop_vec<N>)
    COOPERATIVE_MUL,                     // (coop_vec<OutType, M> (matrix_buffer: byte_buffer, matrix_offset: coop_mat_ref<N, M, CoopRefType>input_vector: coop_vec<N>)
    BINDLESS_COOPERATIVE_MUL,            // (coop_vec<OutType, M> (bindless_array, matrix_buffer: uint, matrix_offset: coop_mat_ref<N, M, CoopRefType>input_vector: coop_vec<N>)
    TYPED_BINDLESS_COOPERATIVE_MUL,      // (coop_vec<OutType, M> (bindless_array, matrix_buffer: uint, matrix_offset: coop_mat_ref<N, M, CoopRefType>input_vector: coop_vec<N>)
    COOPERATIVE_OUTER_PRODUCT_ACCUMULATE,// ResultMatrix += InputVector1 * Transpose(InputVector2);
    // void(matrix_buffer: byte_buffer, matrix_offset: coop_mat_ref, input_vec1 : coop_vector, input_vec2 : coop_vector, )
    COOPERATIVE_VECTOR_ACCUMULATE,// void(vector_buffer: byte_buffer, vector_offset: coop_vec_ref, input_vec: coop_vector)
    COOPERATIVE_VECTOR_LOAD,         // coop_vec<T,N> (byte_buffer, coop_vec_ref<N, CoopRefType>)
    COOPERATIVE_VECTOR_STORE,        // void (byte_buffer, coop_vec_ref<N, CoopRefType>, coop_vec<T,N>)
    COOPERATIVE_VECTOR_SPLAT,        // coop_vec<T,N> (T scalar)
    COOPERATIVE_VECTOR_CAST,         // coop_vec<ToT,N> (coop_vec<FromT,N>)
    BINDLESS_COOPERATIVE_VECTOR_LOAD,        // coop_vec<T,N> (bindless_array, buffer_handle: uint, offset: coop_vec_ref)
    TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD,  // typed variant
    BINDLESS_COOPERATIVE_VECTOR_STORE,       // void (bindless_array, buffer_handle: uint, offset: coop_vec_ref, coop_vec<T,N>)
    TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE, // typed variant
    COOPERATIVE_VECTOR_WORKGROUP_LOAD,  // coop_vec<T,N> (shared_buf: array<T>, index: uint)
    COOPERATIVE_VECTOR_WORKGROUP_STORE, // void (shared_buf: array<T>, index: uint, coop_vec<T,N>)

    // Async group copy
    ASYNC_COPY,/// [(uint scope, ref dst, ref src, uint elem_bytes, uint num, uint stride, uint event) -> uint]: async group copy

    // Async copy pipeline control (CUDA LDGSTS, CC 8.0+)
    PIPELINE_COMMIT,     // (): void — commit pending async copies
    PIPELINE_WAIT_PRIOR,  // (prior_stages: uint): void — wait for pipeline stages

    // Cluster Launch Control (CUDA Blackwell, SM 10.0+)
    CLUSTER_LAUNCH_CONTROL_TRY_CANCEL,      // (result: ref<uint4>, bar: ref<uint64>): void — single-block try_cancel
    CLUSTER_LAUNCH_CONTROL_TRY_CANCEL_MULTICAST, // (result: ref<uint4>, bar: ref<uint64>): void — cluster-wide multicast
    CLUSTER_LAUNCH_CONTROL_QUERY_IS_CANCELED,    // (result: uint4): bool
    CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_X,     // (result: uint4): int
    CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_Y,     // (result: uint4): int
    CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_Z,     // (result: uint4): int
    MBARRIER_INIT,                                // (bar: ref<uint64>, count: uint): void
    MBARRIER_ARRIVE_EXPECT_TX,                    // (bar: ref<uint64>, tx_bytes: uint): void
    MBARRIER_TRY_WAIT_PARITY,                     // (bar: ref<uint64>, phase: int): bool
    FENCE_PROXY_ASYNC_ACQUIRE,                    // (): void — fence.proxy.async with acquire
    FENCE_PROXY_ASYNC_RELEASE,                    // (): void — fence.proxy.async with release

    // Clock
    CLOCK,// (): uint64

    // Appended to preserve the numeric values of all existing public ops.
    // Unlike ROUND (half away from zero), RINT follows the target's
    // round-to-integral mode (round-to-nearest-even on supported GPU targets).
    RINT,// (floatN)
};

static constexpr size_t call_op_count = to_underlying(CallOp::RINT) + 1u;

[[nodiscard]] constexpr auto is_builtin_operation(CallOp op) noexcept {
    return op != CallOp::CUSTOM && op != CallOp::EXTERNAL;
}

[[nodiscard]] constexpr auto is_atomic_operation(CallOp op) noexcept {
    auto op_value = luisa::to_underlying(op);
    return op_value >= luisa::to_underlying(CallOp::ATOMIC_EXCHANGE) && op_value <= luisa::to_underlying(CallOp::ATOMIC_FETCH_MAX);
}

[[nodiscard]] constexpr auto is_autodiff_operation(CallOp op) noexcept {
    auto op_value = luisa::to_underlying(op);
    return op_value >= luisa::to_underlying(CallOp::REQUIRES_GRADIENT) && op_value <= luisa::to_underlying(CallOp::DETACH);
}

[[nodiscard]] constexpr auto is_vector_maker(CallOp op) noexcept {
    auto op_value = luisa::to_underlying(op);
    return op_value >= luisa::to_underlying(CallOp::MAKE_BOOL2) && op_value <= luisa::to_underlying(CallOp::MAKE_FLOAT4);
}

[[nodiscard]] constexpr auto is_matrix_maker(CallOp op) noexcept {
    return op == CallOp::MAKE_FLOAT2X2 ||
           op == CallOp::MAKE_FLOAT3X3 ||
           op == CallOp::MAKE_FLOAT4X4;
}

/// Returns whether the operation uses the descriptor layout of a typed
/// bindless resource array. The typed-uniform and typed-nonuniform resource
/// operations deliberately form one contiguous block in CallOp.
[[nodiscard]] constexpr auto is_typed_bindless_resource_call(CallOp op) noexcept {
    auto value = luisa::to_underlying(op);
    return value >= luisa::to_underlying(
                        CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE) &&
           value <= luisa::to_underlying(
                        CallOp::TYPED_BINDLESS_BUFFER_ADDRESS);
}

/// Returns whether the operation carries the caller's block-uniform bindless
/// index promise. Native compiler dialects must preserve this promise rather
/// than rediscovering (or silently discarding) it.
[[nodiscard]] constexpr auto is_uniform_bindless_resource_call(CallOp op) noexcept {
    auto value = luisa::to_underlying(op);
    return (value >= luisa::to_underlying(
                         CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE) &&
            value <= luisa::to_underlying(
                         CallOp::UNIFORM_BINDLESS_BUFFER_ADDRESS)) ||
           (value >= luisa::to_underlying(
                         CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE) &&
            value <= luisa::to_underlying(
                         CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_ADDRESS));
}

/// Returns whether the operation is a cluster launch control or mbarrier operation.
[[nodiscard]] constexpr auto uses_cluster_launch_control(CallOp op) noexcept {
    auto v = to_underlying(op);
    return v >= to_underlying(CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL) &&
           v <= to_underlying(CallOp::FENCE_PROXY_ASYNC_RELEASE);
}
class Expression;
LUISA_AST_API void check_builtin_call_valid(CallOp op, const Type *return_type, luisa::span<const Expression *const> args) noexcept;

/**
 * @brief Set of call operations.
 * 
 */
class CallableLibrary;

class LUISA_AST_API CallOpSet {

    friend class CallableLibrary;

public:
    using Bitset = std::bitset<call_op_count>;

    /// CallOpSet::Iterator
    class LUISA_AST_API Iterator {

    private:
        const CallOpSet &_set;
        uint _index{0u};

    private:
        friend class CallOpSet;
        Iterator(const CallOpSet &set) noexcept;

    public:
        [[nodiscard]] CallOp operator*() const noexcept;
        Iterator &operator++() noexcept;
        Iterator operator++(int) noexcept;
        [[nodiscard]] bool operator==(luisa::default_sentinel_t) const noexcept;
    };

private:
    Bitset _bits;

public:
    CallOpSet() noexcept : _bits{0} {}
    ~CallOpSet() noexcept = default;
    /// Mark given CallOp
    void mark(CallOp op) noexcept { _bits.set(to_underlying(op)); }
    /// Test given CallOp
    [[nodiscard]] auto test(CallOp op) const noexcept { return _bits.test(to_underlying(op)); }
    void propagate(CallOpSet other) noexcept { _bits |= other._bits; }
    [[nodiscard]] auto begin() const noexcept { return Iterator{*this}; }
    [[nodiscard]] auto end() const noexcept { return luisa::default_sentinel; }
    [[nodiscard]] auto uses_raytracing() const noexcept {
        return test(CallOp::RAY_TRACING_TRACE_CLOSEST) ||
               test(CallOp::RAY_TRACING_TRACE_ANY) ||
               test(CallOp::RAY_TRACING_QUERY_ALL) ||
               test(CallOp::RAY_TRACING_QUERY_ANY) ||
               test(CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR) ||
               test(CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR) ||
               test(CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR) ||
               test(CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR);
    }
    [[nodiscard]] auto uses_ray_query() const noexcept {
        return test(CallOp::RAY_TRACING_QUERY_ALL) ||
               test(CallOp::RAY_TRACING_QUERY_ANY) ||
               test(CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR) ||
               test(CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR);
    }
    [[nodiscard]] auto uses_raytracing_motion_blur() const noexcept {
        return test(CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR) ||
               test(CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR) ||
               test(CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR) ||
               test(CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR);
    }
    [[nodiscard]] auto uses_ray_query_motion_blur() const noexcept {
        return test(CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR) ||
               test(CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR);
    }
    [[nodiscard]] auto uses_atomic() const noexcept {
        return test(CallOp::ATOMIC_FETCH_ADD) ||
               test(CallOp::ATOMIC_FETCH_SUB) ||
               test(CallOp::ATOMIC_FETCH_MIN) ||
               test(CallOp::ATOMIC_FETCH_AND) ||
               test(CallOp::ATOMIC_FETCH_OR) ||
               test(CallOp::ATOMIC_FETCH_XOR) ||
               test(CallOp::ATOMIC_FETCH_MAX) ||
               test(CallOp::ATOMIC_EXCHANGE) ||
               test(CallOp::ATOMIC_COMPARE_EXCHANGE);
    }
    [[nodiscard]] auto uses_autodiff() const noexcept {
        return test(CallOp::REQUIRES_GRADIENT) ||
               test(CallOp::GRADIENT) ||
               test(CallOp::GRADIENT_MARKER) ||
               test(CallOp::ACCUMULATE_GRADIENT) ||
               test(CallOp::BACKWARD) ||
               test(CallOp::DETACH);
    }
    [[nodiscard]] auto uses_typed_bindless_resources() const noexcept {
        for (auto op : *this) {
            if (is_typed_bindless_resource_call(op)) { return true; }
        }
        return false;
    }
    [[nodiscard]] auto uses_uniform_bindless_resources() const noexcept {
        for (auto op : *this) {
            if (is_uniform_bindless_resource_call(op)) { return true; }
        }
        return false;
    }
    [[nodiscard]] auto uses_cooperative() const noexcept {
        return test(CallOp::COOPERATIVE_MUL_ADD) ||
               test(CallOp::BINDLESS_COOPERATIVE_MUL_ADD) ||
               test(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD) ||
               test(CallOp::COOPERATIVE_MUL) ||
               test(CallOp::BINDLESS_COOPERATIVE_MUL) ||
               test(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL) ||
               test(CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE) ||
               test(CallOp::COOPERATIVE_VECTOR_ACCUMULATE) ||
               test(CallOp::COOPERATIVE_VECTOR_LOAD) ||
               test(CallOp::COOPERATIVE_VECTOR_STORE) ||
               test(CallOp::COOPERATIVE_VECTOR_SPLAT) ||
               test(CallOp::COOPERATIVE_VECTOR_CAST) ||
               test(CallOp::BINDLESS_COOPERATIVE_VECTOR_LOAD) ||
               test(CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD) ||
               test(CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE) ||
               test(CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE) ||
               test(CallOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD) ||
               test(CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE);
    }
    [[nodiscard]] auto uses_cluster_launch_control() const noexcept {
        return test(CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL) ||
               test(CallOp::CLUSTER_LAUNCH_CONTROL_TRY_CANCEL_MULTICAST) ||
               test(CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_IS_CANCELED) ||
               test(CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_X) ||
               test(CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_Y) ||
               test(CallOp::CLUSTER_LAUNCH_CONTROL_QUERY_GET_CTAD_Z) ||
               test(CallOp::MBARRIER_INIT) ||
               test(CallOp::MBARRIER_ARRIVE_EXPECT_TX) ||
               test(CallOp::MBARRIER_TRY_WAIT_PARITY) ||
               test(CallOp::FENCE_PROXY_ASYNC_ACQUIRE) ||
               test(CallOp::FENCE_PROXY_ASYNC_RELEASE);
    }
};

}// namespace luisa::compute

LUISA_MAGIC_ENUM_RANGE(luisa::compute::CallOp, CUSTOM, CLOCK)
