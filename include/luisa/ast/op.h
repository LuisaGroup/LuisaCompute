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

[[nodiscard]] LC_AST_API TypePromotion promote_types(BinaryOp op, const Type *lhs, const Type *rhs) noexcept;

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

    ADDRESS_OF,// (expr) -> uint64

    BUFFER_READ,   /// [(buffer, index) -> value]: reads the index-th element in buffer
    BUFFER_WRITE,  /// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
    BUFFER_SIZE,   /// [(buffer) -> size]
    BUFFER_ADDRESS,/// [(buffer) -> address]

    BYTE_BUFFER_READ, /// [(buffer, byte_index) -> value]: reads the index-th element in buffer
    BYTE_BUFFER_WRITE,/// [(buffer, byte_index, value) -> void]: writes value into the index-th element of buffer
    BYTE_BUFFER_SIZE, /// [(buffer) -> size_bytes]

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
    BINDLESS_TEXTURE2D_READ,             // (bindless_array, index: uint, coord: uint2): float4
    BINDLESS_TEXTURE3D_READ,             // (bindless_array, index: uint, coord: uint3): float4
    BINDLESS_TEXTURE2D_READ_LEVEL,       // (bindless_array, index: uint, coord: uint2, level: uint): float4
    BINDLESS_TEXTURE3D_READ_LEVEL,       // (bindless_array, index: uint, coord: uint3, level: uint): float4
    BINDLESS_TEXTURE2D_SIZE,             // (bindless_array, index: uint): uint2
    BINDLESS_TEXTURE3D_SIZE,             // (bindless_array, index: uint): uint3
    BINDLESS_TEXTURE2D_SIZE_LEVEL,       // (bindless_array, index: uint, level: uint): uint2
    BINDLESS_TEXTURE3D_SIZE_LEVEL,       // (bindless_array, index: uint, level: uint): uint3

    BINDLESS_BUFFER_READ,     // (bindless_array, index: uint, elem_index: uint): expr->type()
    BINDLESS_BUFFER_WRITE,    // (bindless_array, index: uint, elem_index: uint, value: expr): void
    BINDLESS_BYTE_BUFFER_READ,// (bindless_array, index: uint, offset_bytes: uint): expr->type()
    BINDLESS_BUFFER_SIZE,     // (bindless_array, index: uint, stride: uint) -> size
    BINDLESS_BUFFER_TYPE,     // (bindless_array, index: uint) -> uint64 (type id of the element); the returned value
                              // could be compared with the value of a TypeIDExpr to examine the type of the buffer
    BINDLESS_BUFFER_ADDRESS,  // (bindless_array, index: uint) -> uint64 (address of the buffer)

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
    RASTER_DISCARD,// (): void

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
};

static constexpr size_t call_op_count = to_underlying(CallOp::SHADER_EXECUTION_REORDER) + 1u;

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

/**
 * @brief Set of call operations.
 * 
 */
class CallableLibrary;

class LC_AST_API CallOpSet {

    friend class CallableLibrary;

public:
    using Bitset = std::bitset<call_op_count>;

    /// CallOpSet::Iterator
    class Iterator {

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
};

}// namespace luisa::compute

LUISA_MAGIC_ENUM_RANGE(luisa::compute::CallOp, CUSTOM, SHADER_EXECUTION_REORDER)
