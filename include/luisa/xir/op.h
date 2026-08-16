//
// Created by mike on 3/20/25.
//

#pragma once

#ifndef LUISA_XIR_OP_H
#define LUISA_XIR_OP_H

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::xir {

// Bindless descriptor layout and index-uniformity are independent semantic
// axes. Keeping them as instruction attributes avoids multiplying every
// resource opcode into four variants while still making the distinction
// visible to verification, optimization, interchange, and code generation.
struct BindlessResourceAccess {
    uint8_t typed : 1 {false};
    uint8_t uniform : 1 {false};

    [[nodiscard]] constexpr uint8_t encoded() const noexcept {
        return static_cast<uint8_t>((typed ? 1u : 0u) |
                                    (uniform ? 2u : 0u));
    }

    [[nodiscard]] static constexpr BindlessResourceAccess
    decode(uint8_t value) noexcept {
        return {
            .typed = (value & 1u) != 0u,
            .uniform = (value & 2u) != 0u};
    }

    [[nodiscard]] constexpr bool is_default() const noexcept {
        return !typed && !uniform;
    }

    [[nodiscard]] constexpr bool operator==(
        const BindlessResourceAccess &) const noexcept = default;
};

static_assert(sizeof(BindlessResourceAccess) == 1u);

enum class AllocaOp {
    LOCAL,
    SHARED,
};

enum class ArithmeticOp {

    // unary operators
    UNARY_MINUS,  // -x
    UNARY_BIT_NOT,// ~x

    // binary operators
    BINARY_ADD,
    BINARY_SUB,
    BINARY_MUL,
    BINARY_DIV,
    BINARY_MOD,

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

    // math
    ALL,// (boolN)
    ANY,// (boolN)

    SELECT,  // (vecN, vecN, boolN)
    CLAMP,   // (vecN, vecN, vecN)
    SATURATE,// (vecN)
    LERP,    // (vecN, vecN, vecN)

    SMOOTHSTEP,// (edge0, edge1, x)
    STEP,      // (edge, x): (x >= edge) ? 1 : 0

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
    RINT, // (floatN)

    FMA,     // (a: floatN, b: floatN, c: floatN): return a * b + c
    COPYSIGN,// (floatN, floatN)

    CROSS,         // (floatN, floatN)
    DOT,           // (floatN, floatN)
    LENGTH,        // (floatN)
    LENGTH_SQUARED,// (floatN)
    NORMALIZE,     // (floatN)
    FACEFORWARD,   // (floatN, floatN, floatN)
    REFLECT,       // (floatN, floatN)

    REDUCE_SUM,    // (floatN) -> float
    REDUCE_PRODUCT,// (floatN) -> float
    REDUCE_MIN,    // (floatN) -> float
    REDUCE_MAX,    // (floatN) -> float

    OUTER_PRODUCT,// (floatN, floatN) -> floatNxN | (floatNxN, floatNxN) -> floatNxN

    MATRIX_COMP_NEG,   // (floatNxN) -> floatNxN
    MATRIX_COMP_ADD,   // (floatNxN, floatNxN) -> floatNxN | (floatNxN, float) -> floatNxN | (float, floatNxN) -> floatNxN
    MATRIX_COMP_SUB,   // (floatNxN, floatNxN) -> floatNxN | (floatNxN, float) -> floatNxN | (float, floatNxN) -> floatNxN
    MATRIX_COMP_MUL,   // (floatNxN, floatNxN) -> floatNxN | (floatNxN, float) -> floatNxN | (float, floatNxN) -> floatNxN
    MATRIX_COMP_DIV,   // (floatNxN, floatNxN) -> floatNxN | (floatNxN, float) -> floatNxN | (float, floatNxN) -> floatNxN
    MATRIX_LINALG_MUL, // (floatNxN, floatNxN) -> floatNxN | (floatNxN, floatN) -> floatN | (floatN, floatNxN) -> floatN
    MATRIX_DETERMINANT,// (floatNxN) -> float
    MATRIX_TRANSPOSE,  // (floatNxN) -> floatNxN
    MATRIX_INVERSE,    // (floatNxN) -> floatNxN

    // aggregate operations
    AGGREGATE,
    SHUFFLE,
    INSERT,
    EXTRACT,
};

enum class AtomicOp {
    EXCHANGE,        /// [(base, indices..., desired) -> old]: stores desired, returns old.
    COMPARE_EXCHANGE,/// [(base, indices..., expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
    FETCH_ADD,       /// [(base, indices..., val) -> old]: stores (old + val), returns old.
    FETCH_SUB,       /// [(base, indices..., val) -> old]: stores (old - val), returns old.
    FETCH_AND,       /// [(base, indices..., val) -> old]: stores (old & val), returns old.
    FETCH_OR,        /// [(base, indices..., val) -> old]: stores (old | val), returns old.
    FETCH_XOR,       /// [(base, indices..., val) -> old]: stores (old ^ val), returns old.
    FETCH_MIN,       /// [(base, indices..., val) -> old]: stores min(old, val), returns old.
    FETCH_MAX,       /// [(base, indices..., val) -> old]: stores max(old, val), returns old.
};

[[nodiscard]] constexpr auto atomic_op_value_count(AtomicOp op) noexcept {
    return op == AtomicOp::COMPARE_EXCHANGE ? 2u : 1u;
}

enum class AutodiffIntrinsicOp {
    AUTODIFF_REQUIRES_GRADIENT,  // (expr) -> void
    AUTODIFF_GRADIENT,           // (expr) -> expr
    AUTODIFF_GRADIENT_MARKER,    // (ref, expr) -> void
    AUTODIFF_ACCUMULATE_GRADIENT,// (ref, expr) -> void
    AUTODIFF_BACKWARD,           // (expr) -> void
    AUTODIFF_DETACH,             // (expr) -> expr
    AUTODIFF_PROPAGATE_GRADIENT, // (expr, grad...) -> void
    AUTODIFF_OUTPUT_GRADIENT,    // (expr, index) -> expr
};

enum class CastOp {
    STATIC_CAST,
    BITWISE_CAST,
};

enum class RayQueryObjectReadOp {
    RAY_QUERY_OBJECT_WORLD_SPACE_RAY,           // (RayQuery): Ray
    RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY,// (RayQuery): Ray
    RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT,  // (RayQuery): ProceduralHit
    RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT,    // (RayQuery): TriangleHit
    RAY_QUERY_OBJECT_COMMITTED_HIT,             // (RayQuery): CommittedHit

    // Maxwell's extensions
    RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE,  // (RayQuery): bool
    RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE,// (RayQuery): bool
    RAY_QUERY_OBJECT_IS_TERMINATED,          // (RayQuery): bool
};

enum class RayQueryObjectWriteOp {
    RAY_QUERY_OBJECT_COMMIT_TRIANGLE,  // (RayQuery): void
    RAY_QUERY_OBJECT_COMMIT_PROCEDURAL,// (RayQuery, float): void
    RAY_QUERY_OBJECT_TERMINATE,        // (RayQuery): void

    // Maxwell's extensions
    RAY_QUERY_OBJECT_PROCEED,// (RayQuery): void
};

enum class ResourceQueryOp {

    // buffer query operations
    BUFFER_SIZE,     /// [(buffer) -> size]
    BYTE_BUFFER_SIZE,/// [(buffer) -> size_bytes]

    // texture query operations
    TEXTURE2D_SIZE,/// [(texture) -> Vector<uint, dim>]
    TEXTURE3D_SIZE,/// [(texture) -> Vector<uint, dim>]

    // bindless array query operations
    BINDLESS_BUFFER_SIZE,     // (bindless_array, index: uint, stride: uint) -> size: uint32|uint64
    BINDLESS_BYTE_BUFFER_SIZE,// (bindless_array, index: uint) -> size: uint32|uint64

    BINDLESS_TEXTURE2D_SIZE,      // (bindless_array, index: uint): uint2
    BINDLESS_TEXTURE3D_SIZE,      // (bindless_array, index: uint): uint3
    BINDLESS_TEXTURE2D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint2
    BINDLESS_TEXTURE3D_SIZE_LEVEL,// (bindless_array, index: uint, level: uint): uint3

    // texture sampling (note: we assume texture sampling is not affected by resource writes in the same shader)
    TEXTURE2D_SAMPLE,           // (tex, uv: float2, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_LEVEL,     // (tex, uv: float2, level: float, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_GRAD,      // (tex, uv: float2, ddx: float2, ddy: float2, filter: uint, address: uint): float4
    TEXTURE2D_SAMPLE_GRAD_LEVEL,// (tex, uv: float2, ddx: float2, ddy: float2, mip_clamp: float, filter: uint, address: uint): float4

    TEXTURE3D_SAMPLE,           // (tex, uv: float3, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_LEVEL,     // (tex, uv: float3, level: float, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_GRAD,      // (tex, uv: float3, ddx: float3, ddy: float3, filter: uint, address: uint): float4
    TEXTURE3D_SAMPLE_GRAD_LEVEL,// (tex, uv: float3, ddx: float3, ddy: float3, mip_clamp: float, filter: uint, address: uint): float4

    BINDLESS_TEXTURE2D_SAMPLE,           // (bindless_array, index: uint, uv: float2): float4
    BINDLESS_TEXTURE2D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float2, level: float): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, mip_clamp: float): float4
    BINDLESS_TEXTURE3D_SAMPLE,           // (bindless_array, index: uint, uv: float3): float4
    BINDLESS_TEXTURE3D_SAMPLE_LEVEL,     // (bindless_array, index: uint, uv: float3, level: float): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3, mip_clamp: float): float4

    BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float2, filter: uint, address: uint): float4
    BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float2, level: float, filter: uint, address: uint): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, filter: uint, address: uint): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, mip_clamp: float, filter: uint, address: uint): float4
    BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,           // (bindless_array, index: uint, uv: float3, filter: uint, address: uint): float4
    BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER,     // (bindless_array, index: uint, uv: float3, level: float, filter: uint, address: uint): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER,      // (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3, filter: uint, address: uint): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER,// (bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3, mip_clamp: float, filter: uint, address: uint): float4

    // low-level pointer operations, for akari
    BUFFER_DEVICE_ADDRESS,         // (buffer) -> address: uint64
    BINDLESS_BUFFER_DEVICE_ADDRESS,// (bindless_array, index: uint) -> address: uint64

    // ray tracing
    RAY_TRACING_INSTANCE_TRANSFORM,      // (Accel, uint)
    RAY_TRACING_INSTANCE_USER_ID,        // (Accel, uint)
    RAY_TRACING_INSTANCE_VISIBILITY_MASK,// (Accel, uint)

    RAY_TRACING_TRACE_CLOSEST,// (Accel, ray, mask: uint): TriangleHit
    RAY_TRACING_TRACE_ANY,    // (Accel, ray, mask: uint): bool
    RAY_TRACING_QUERY_ALL,    // (Accel, ray, mask: uint): RayQuery
    RAY_TRACING_QUERY_ANY,    // (Accel, ray, mask: uint): RayQuery

    // ray tracing with motion blur
    RAY_TRACING_INSTANCE_MOTION_MATRIX,// (Accel, index: uint, key: uint): float4x4
    RAY_TRACING_INSTANCE_MOTION_SRT,   // (Accel, index: uint, key: uint): SRT

    RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR,// (Accel, ray, time: float, mask: uint): TriangleHit
    RAY_TRACING_TRACE_ANY_MOTION_BLUR,    // (Accel, ray, time: float, mask: uint): bool
    RAY_TRACING_QUERY_ALL_MOTION_BLUR,    // (Accel, ray, time: float, mask: uint): RayQuery
    RAY_TRACING_QUERY_ANY_MOTION_BLUR,    // (Accel, ray, time: float, mask: uint): RayQuery
};

enum class ResourceReadOp {

    BUFFER_READ,              /// [(buffer, index) -> value]: reads the index-th element in buffer
    BUFFER_VOLATILE_READ,     /// [(buffer, index) -> value]: reads the index-th element in buffer
    BYTE_BUFFER_READ,         /// [(buffer, byte_index) -> value]: reads the index-th element in buffer
    BYTE_BUFFER_VOLATILE_READ,/// [(buffer, byte_index) -> value]: reads the index-th element in buffer

    TEXTURE2D_READ,/// [(texture, coord) -> value]
    TEXTURE3D_READ,/// [(texture, coord) -> value]

    BINDLESS_BUFFER_READ,     // (bindless_array, index: uint, elem_index: uint) -> T
    BINDLESS_BYTE_BUFFER_READ,// (bindless_array, index: uint, offset_bytes: uint64) -> T

    BINDLESS_TEXTURE2D_READ,      // (bindless_array, index: uint, coord: uint2): float4
    BINDLESS_TEXTURE3D_READ,      // (bindless_array, index: uint, coord: uint3): float4
    BINDLESS_TEXTURE2D_READ_LEVEL,// (bindless_array, index: uint, coord: uint2, level: uint): float4
    BINDLESS_TEXTURE3D_READ_LEVEL,// (bindless_array, index: uint, coord: uint3, level: uint): float4

    DEVICE_ADDRESS_READ,// (address: uint64) -> value: T

    // cooperative vector operations. Cooperative vector/matrix references are
    // lowered to plain uint32 byte offsets in XIR; the buffer-data component
    // interpretation is carried by trailing constant uint32 operands.
    COOPERATIVE_MUL_ADD,        // (matrix_buffer, matrix_offset: uint, bias_buffer, bias_offset: uint, input: coopvec<K>, matrix_interp: const uint, bias_interp: const uint) -> coopvec<M>
    BINDLESS_COOPERATIVE_MUL_ADD,// (bindless_array, matrix_index: uint, matrix_offset: uint, bias_index: uint, bias_offset: uint, input: coopvec<K>, matrix_interp: const uint, bias_interp: const uint) -> coopvec<M>
    COOPERATIVE_MUL,            // (matrix_buffer, matrix_offset: uint, input: coopvec<K>, matrix_interp: const uint) -> coopvec<M>
    BINDLESS_COOPERATIVE_MUL,   // (bindless_array, matrix_index: uint, matrix_offset: uint, input: coopvec<K>, matrix_interp: const uint) -> coopvec<M>
    COOPERATIVE_VECTOR_LOAD,    // (byte_buffer, offset: uint) -> coopvec<T, N>
    BINDLESS_COOPERATIVE_VECTOR_LOAD,// (bindless_array, index: uint, offset: uint) -> coopvec<T, N>
    COOPERATIVE_VECTOR_SPLAT,   // (scalar: T) -> coopvec<T, N>
    COOPERATIVE_VECTOR_CAST,    // (coopvec<S, N>) -> coopvec<T, N>
    COOPERATIVE_VECTOR_WORKGROUP_LOAD,// (shared_array, index: uint) -> coopvec<T, N>
};

enum class ResourceWriteOp {

    // buffer write operations
    BUFFER_WRITE,              /// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
    BUFFER_VOLATILE_WRITE,     /// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
    BYTE_BUFFER_WRITE,         /// [(buffer, byte_index, value) -> void]: writes value into the index-th element of buffer
    BYTE_BUFFER_VOLATILE_WRITE,/// [(buffer, byte_index, value) -> void]: writes value into the index-th element of buffer

    // texture write operations
    TEXTURE2D_WRITE,/// [(texture, coord, value) -> void]
    TEXTURE3D_WRITE,/// [(texture, coord, value) -> void]

    // bindless array write operations
    BINDLESS_BUFFER_WRITE,     // (bindless_array, index: uint, elem_index: uint, value: T) -> void
    BINDLESS_BYTE_BUFFER_WRITE,// (bindless_array, index: uint, offset_bytes: uint64, value: T) -> void

    // low-level pointer operations, for akari
    DEVICE_ADDRESS_WRITE,// (address: uint64, value: T) -> void

    // ray tracing
    RAY_TRACING_SET_INSTANCE_TRANSFORM,      // (Accel, uint, float4x4)
    RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK,// (Accel, uint, uint)
    RAY_TRACING_SET_INSTANCE_OPACITY,        // (Accel, uint, bool)
    RAY_TRACING_SET_INSTANCE_USER_ID,        // (Accel, uint, uint)
    RAY_TRACING_SET_INSTANCE_MOTION_MATRIX,  // (Accel, index: uint, key: uint, transform: float4x4)
    RAY_TRACING_SET_INSTANCE_MOTION_SRT,     // (Accel, index: uint, key: uint, transform: SRT)

    // indirect dispatch
    INDIRECT_DISPATCH_SET_KERNEL,// (Buffer, uint offset, uint3 block_size, uint3 dispatch_size, uint kernel_id)
    INDIRECT_DISPATCH_SET_COUNT, // (Buffer, uint count)

    // cooperative vector operations (see ResourceReadOp for the XIR encoding
    // of references and interpretations)
    COOPERATIVE_OUTER_PRODUCT_ACCUMULATE,// (matrix_buffer, matrix_offset: uint, a: coopvec<M>, b: coopvec<N>, matrix_interp: const uint) -> void
    COOPERATIVE_VECTOR_ACCUMULATE,       // (byte_buffer, offset: uint, coopvec<T, N>) -> void
    COOPERATIVE_VECTOR_STORE,            // (byte_buffer, offset: uint, coopvec<T, N>) -> void
    BINDLESS_COOPERATIVE_VECTOR_STORE,   // (bindless_array, index: uint, offset: uint, coopvec<T, N>) -> void
    COOPERATIVE_VECTOR_WORKGROUP_STORE,  // (shared_array, index: uint, coopvec<T, N>) -> void
};

[[nodiscard]] constexpr bool is_bindless_resource_op(
    ResourceQueryOp op) noexcept {
    switch (op) {
        case ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS:
            return true;
        default: return false;
    }
}

[[nodiscard]] constexpr bool is_bindless_resource_op(
    ResourceReadOp op) noexcept {
    switch (op) {
        case ResourceReadOp::BINDLESS_BUFFER_READ:
        case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ:
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ:
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL:
        case ResourceReadOp::BINDLESS_COOPERATIVE_MUL_ADD:
        case ResourceReadOp::BINDLESS_COOPERATIVE_MUL:
        case ResourceReadOp::BINDLESS_COOPERATIVE_VECTOR_LOAD:
            return true;
        default: return false;
    }
}

[[nodiscard]] constexpr bool is_bindless_resource_op(
    ResourceWriteOp op) noexcept {
    switch (op) {
        case ResourceWriteOp::BINDLESS_BUFFER_WRITE:
        case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
        case ResourceWriteOp::BINDLESS_COOPERATIVE_VECTOR_STORE:
            return true;
        default: return false;
    }
}

enum class ThreadGroupOp {

    // shader execution re-ordering
    SHADER_EXECUTION_REORDER,// (uint hint, uint hint_bits): void

    RASTER_QUAD_DDX,// (arg: floatN): floatN
    RASTER_QUAD_DDY,// (arg: floatN): floatN

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

    // block synchronization
    SYNCHRONIZE_BLOCK,// ()
};

#include <luisa/xir/op_to_string.inl.h>

}// namespace luisa::compute::xir

#endif// LUISA_XIR_OP_H
