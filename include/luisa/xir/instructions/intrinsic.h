#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

enum struct IntrinsicOp {

    // no-op placeholder
    NOP,

    // unary operators
    UNARY_PLUS,   // +x
    UNARY_MINUS,  // -x
    UNARY_NOT,    // !x
    UNARY_BIT_NOT,// ~x

    // binary operators
    BINARY_ADD,
    BINARY_SUB,
    BINARY_MUL,
    BINARY_DIV,
    BINARY_MOD,

    BINARY_AND,
    BINARY_OR,

    BINARY_BIT_AND,
    BINARY_BIT_OR,
    BINARY_BIT_XOR,

    BINARY_SHIFT_LEFT,
    BINARY_SHIFT_RIGHT,
    BINARY_ROTATE_LEFT,
    BINARY_ROTATE_RIGHT,

    BINARY_LESS,
    BINARY_GREATER,
    BINARY_LESS_EQUAL,
    BINARY_GREATER_EQUAL,
    BINARY_EQUAL,
    BINARY_NOT_EQUAL,

    // optimization/debugging
    ASSUME,
    ASSERT,

    // hacking
    ADDRESS_OF,// (variable) -> uint64

    // thread coordination
    THREAD_ID,
    BLOCK_ID,
    WARP_SIZE,
    WARP_LANE_ID,
    DISPATCH_ID,
    DISPATCH_SIZE,

    // block synchronization
    SYNCHRONIZE_BLOCK,// ()

    // math
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
    ATAN2,// (floatN, floatN)
    ATANH,// (floatN)

    COS, // (floatN)
    COSH,// (floatN)
    SIN, // (floatN)
    SINH,// (floatN)
    TAN, // (floatN)
    TANH,// (floatN)

    EXP,    // (floatN)
    EXP2,   // (floatN)
    EXP10,  // (floatN)
    LOG,    // (floatN)
    LOG2,   // (floatN)
    LOG10,  // (floatN)
    POW,    // (floatN, floatN)
    POW_INT,// (floatN, intN)

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

    OUTER_PRODUCT,  // (floatN | floatNxN)
    MATRIX_COMP_MUL,// (floatNxN, floatNxN)
    DETERMINANT,    // (floatNxN)
    TRANSPOSE,      // (floatNxN)
    INVERSE,        // (floatNxN)

    // atomic operations
    ATOMIC_EXCHANGE,        /// [(atomic_ref, desired) -> old]: stores desired, returns old.
    ATOMIC_COMPARE_EXCHANGE,/// [(atomic_ref, expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
    ATOMIC_FETCH_ADD,       /// [(atomic_ref, val) -> old]: stores (old + val), returns old.
    ATOMIC_FETCH_SUB,       /// [(atomic_ref, val) -> old]: stores (old - val), returns old.
    ATOMIC_FETCH_AND,       /// [(atomic_ref, val) -> old]: stores (old & val), returns old.
    ATOMIC_FETCH_OR,        /// [(atomic_ref, val) -> old]: stores (old | val), returns old.
    ATOMIC_FETCH_XOR,       /// [(atomic_ref, val) -> old]: stores (old ^ val), returns old.
    ATOMIC_FETCH_MIN,       /// [(atomic_ref, val) -> old]: stores min(old, val), returns old.
    ATOMIC_FETCH_MAX,       /// [(atomic_ref, val) -> old]: stores max(old, val), returns old.

    // resource operations
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

    // bindless array operations
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

    // aggregate operations
    AGGREGATE,
    SHUFFLE,
    INSERT,
    EXTRACT,

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

    // ray query extensions for backends with native support
    RAY_QUERY_PROCEED,
    RAY_QUERY_IS_TRIANGLE_CANDIDATE,
    RAY_QUERY_IS_PROCEDURAL_CANDIDATE,

    // rasterization
    RASTER_DISCARD,// (): void
    DDX,           // (arg: float vector): float vector
    DDY,           // (arg: float vector): float vector

    // warp operations
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

    // indirect dispatch
    INDIRECT_SET_DISPATCH_KERNEL,// (Buffer, uint offset, uint3 block_size, uint3 dispatch_size, uint kernel_id)
    INDIRECT_SET_DISPATCH_COUNT, // (Buffer, uint count)

    // direct texture sampling
    TEXTURE2D_SAMPLE,           // (tex, uv: float2, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_LEVEL,     // (tex, uv: float2, level: float, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_GRAD,      // (tex, uv: float2, ddx: float2, ddy: float2, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_GRAD_LEVEL,// (tex, uv: float2, ddx: float2, ddy: float2,  mip_clamp: float, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE,           // (tex, uv: float3, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_LEVEL,     // (tex, uv: float3, level: float, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_GRAD,      // (tex, uv: float3, ddx: float3, ddy: float3, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_GRAD_LEVEL,// (tex, uv: float3, ddx: float3, ddy: float3,  mip_clamp: float, filter: uint, address: uint): float4

    // shader execution re-ordering
    SHADER_EXECUTION_REORDER,// (uint hint, uint hint_bits): void
};

class LC_XIR_API IntrinsicInst final : public Instruction {

private:
    IntrinsicOp _op;

public:
    explicit IntrinsicInst(Pool *pool, IntrinsicOp op = IntrinsicOp::NOP,
                           luisa::span<Value *const> operands = {},
                           const Type *type = nullptr,
                           const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::INTRINSIC;
    }

    [[nodiscard]] auto op() const noexcept { return _op; }
    void set_op(IntrinsicOp op) noexcept { _op = op; }
};

}// namespace luisa::compute::xir
