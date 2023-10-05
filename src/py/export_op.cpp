#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <luisa/ast/function.h>
#include <luisa/core/logging.h>

namespace py = pybind11;
using namespace luisa::compute;

void export_op(py::module &m) {

    py::enum_<CastOp>(m, "CastOp")
        .value("STATIC", CastOp::STATIC)
        .value("BITWISE", CastOp::BITWISE);

    py::enum_<UnaryOp>(m, "UnaryOp")
        .value("PLUS", UnaryOp::PLUS)
        .value("MINUS", UnaryOp::MINUS)
        .value("NOT", UnaryOp::NOT)
        .value("BIT_NOT", UnaryOp::BIT_NOT);

    py::enum_<BinaryOp>(m, "BinaryOp")
        // arithmetic
        .value("ADD", BinaryOp::ADD)
        .value("SUB", BinaryOp::SUB)
        .value("MUL", BinaryOp::MUL)
        .value("DIV", BinaryOp::DIV)
        .value("MOD", BinaryOp::MOD)
        .value("BIT_AND", BinaryOp::BIT_AND)
        .value("BIT_OR", BinaryOp::BIT_OR)
        .value("BIT_XOR", BinaryOp::BIT_XOR)
        .value("SHL", BinaryOp::SHL)
        .value("SHR", BinaryOp::SHR)
        .value("AND", BinaryOp::AND)
        .value("OR", BinaryOp::OR)
        // relational
        .value("LESS", BinaryOp::LESS)
        .value("GREATER", BinaryOp::GREATER)
        .value("LESS_EQUAL", BinaryOp::LESS_EQUAL)
        .value("GREATER_EQUAL", BinaryOp::GREATER_EQUAL)
        .value("EQUAL", BinaryOp::EQUAL)
        .value("NOT_EQUAL", BinaryOp::NOT_EQUAL);

    py::enum_<CallOp>(m, "CallOp")

        .value("CUSTOM", CallOp::CUSTOM)

        .value("ALL", CallOp::ALL)
        .value("ANY", CallOp::ANY)

        .value("SELECT", CallOp::SELECT)
        .value("CLAMP", CallOp::CLAMP)
        .value("SATURATE", CallOp::SATURATE)
        .value("LERP", CallOp::LERP)
        .value("STEP", CallOp::STEP)
        .value("SMOOTHSTEP", CallOp::SMOOTHSTEP)

        .value("ABS", CallOp::ABS)
        .value("MIN", CallOp::MIN)
        .value("MAX", CallOp::MAX)

        .value("CLZ", CallOp::CLZ)
        .value("CTZ", CallOp::CTZ)
        .value("POPCOUNT", CallOp::POPCOUNT)
        .value("REVERSE", CallOp::REVERSE)

        .value("ISINF", CallOp::ISINF)
        .value("ISNAN", CallOp::ISNAN)

        .value("ACOS", CallOp::ACOS)
        .value("ACOSH", CallOp::ACOSH)
        .value("ASIN", CallOp::ASIN)
        .value("ASINH", CallOp::ASINH)
        .value("ATAN", CallOp::ATAN)
        .value("ATAN2", CallOp::ATAN2)
        .value("ATANH", CallOp::ATANH)

        .value("COS", CallOp::COS)
        .value("COSH", CallOp::COSH)
        .value("SIN", CallOp::SIN)
        .value("SINH", CallOp::SINH)
        .value("TAN", CallOp::TAN)
        .value("TANH", CallOp::TANH)

        .value("EXP", CallOp::EXP)
        .value("EXP2", CallOp::EXP2)
        .value("EXP10", CallOp::EXP10)
        .value("LOG", CallOp::LOG)
        .value("LOG2", CallOp::LOG2)
        .value("LOG10", CallOp::LOG10)
        .value("POW", CallOp::POW)

        .value("SQRT", CallOp::SQRT)
        .value("RSQRT", CallOp::RSQRT)

        .value("CEIL", CallOp::CEIL)
        .value("FLOOR", CallOp::FLOOR)
        .value("FRACT", CallOp::FRACT)
        .value("TRUNC", CallOp::TRUNC)
        .value("ROUND", CallOp::ROUND)

        .value("FMA", CallOp::FMA)
        .value("COPYSIGN", CallOp::COPYSIGN)

        .value("CROSS", CallOp::CROSS)
        .value("DOT", CallOp::DOT)
        .value("DDX", CallOp::DDX)
        .value("DDY", CallOp::DDY)
        .value("LENGTH", CallOp::LENGTH)
        .value("LENGTH_SQUARED", CallOp::LENGTH_SQUARED)
        .value("NORMALIZE", CallOp::NORMALIZE)
        .value("FACEFORWARD", CallOp::FACEFORWARD)
        .value("REFLECT", CallOp::REFLECT)

        .value("DETERMINANT", CallOp::DETERMINANT)
        .value("TRANSPOSE", CallOp::TRANSPOSE)
        .value("INVERSE", CallOp::INVERSE)

        .value("SYNCHRONIZE_BLOCK", CallOp::SYNCHRONIZE_BLOCK)

        .value("ATOMIC_EXCHANGE", CallOp::ATOMIC_EXCHANGE)                /// [(atomic_ref, desired) -> old]: stores desired, returns old.
        .value("ATOMIC_COMPARE_EXCHANGE", CallOp::ATOMIC_COMPARE_EXCHANGE)/// [(atomic_ref, expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
        .value("ATOMIC_FETCH_ADD", CallOp::ATOMIC_FETCH_ADD)              /// [(atomic_ref, val) -> old]: stores (old + val), returns old.
        .value("ATOMIC_FETCH_SUB", CallOp::ATOMIC_FETCH_SUB)              /// [(atomic_ref, val) -> old]: stores (old - val), returns old.
        .value("ATOMIC_FETCH_AND", CallOp::ATOMIC_FETCH_AND)              /// [(atomic_ref, val) -> old]: stores (old & val), returns old.
        .value("ATOMIC_FETCH_OR", CallOp::ATOMIC_FETCH_OR)                /// [(atomic_ref, val) -> old]: stores (old | val), returns old.
        .value("ATOMIC_FETCH_XOR", CallOp::ATOMIC_FETCH_XOR)              /// [(atomic_ref, val) -> old]: stores (old ^ val), returns old.
        .value("ATOMIC_FETCH_MIN", CallOp::ATOMIC_FETCH_MIN)              /// [(atomic_ref, val) -> old]: stores min(old, val), returns old.
        .value("ATOMIC_FETCH_MAX", CallOp::ATOMIC_FETCH_MAX)              /// [(atomic_ref, val) -> old]: stores max(old, val), returns old.

        .value("BUFFER_READ", CallOp::BUFFER_READ)  /// [(buffer, index) -> value]: reads the index-th element in buffer
        .value("BUFFER_WRITE", CallOp::BUFFER_WRITE)/// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
        .value("BYTE_BUFFER_READ", CallOp::BYTE_BUFFER_READ)
        .value("BYTE_BUFFER_WRITE", CallOp::BYTE_BUFFER_WRITE)
        .value("TEXTURE_READ", CallOp::TEXTURE_READ)  /// [(texture, coord) -> value]
        .value("TEXTURE_WRITE", CallOp::TEXTURE_WRITE)/// [(texture, coord, value) -> void]
        .value("TEXTURE_SIZE", CallOp::TEXTURE_SIZE)

        .value("BINDLESS_TEXTURE2D_SAMPLE", CallOp::BINDLESS_TEXTURE2D_SAMPLE)                      //(bindless_array, index: uint, uv: float2): float4
        .value("BINDLESS_TEXTURE2D_SAMPLE_LEVEL", CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL)          //(bindless_array, index: uint, uv: float2, level: float): float4
        .value("BINDLESS_TEXTURE2D_SAMPLE_GRAD", CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD)            //(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
        .value("BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL", CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL)//(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
        .value("BINDLESS_TEXTURE3D_SAMPLE", CallOp::BINDLESS_TEXTURE3D_SAMPLE)                      //(bindless_array, index: uint, uv: float3): float4
        .value("BINDLESS_TEXTURE3D_SAMPLE_LEVEL", CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL)          //(bindless_array, index: uint, uv: float3, level: float): float4
        .value("BINDLESS_TEXTURE3D_SAMPLE_GRAD", CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD)            //(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
        .value("BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL", CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL)//(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
        .value("BINDLESS_TEXTURE2D_READ", CallOp::BINDLESS_TEXTURE2D_READ)                          //(bindless_array, index: uint, coord: uint2): float4
        .value("BINDLESS_TEXTURE3D_READ", CallOp::BINDLESS_TEXTURE3D_READ)                          //(bindless_array, index: uint, coord: uint3): float4
        .value("BINDLESS_TEXTURE2D_READ_LEVEL", CallOp::BINDLESS_TEXTURE2D_READ_LEVEL)              //(bindless_array, index: uint, coord: uint2, level: uint): float4
        .value("BINDLESS_TEXTURE3D_READ_LEVEL", CallOp::BINDLESS_TEXTURE3D_READ_LEVEL)              //(bindless_array, index: uint, coord: uint3, level: uint): float4
        .value("BINDLESS_BUFFER_SIZE", CallOp::BINDLESS_BUFFER_SIZE)                                //(bindless_array, index: uint): uint
        .value("BINDLESS_TEXTURE2D_SIZE", CallOp::BINDLESS_TEXTURE2D_SIZE)                          //(bindless_array, index: uint): uint2
        .value("BINDLESS_TEXTURE3D_SIZE", CallOp::BINDLESS_TEXTURE3D_SIZE)                          //(bindless_array, index: uint): uint3
        .value("BINDLESS_TEXTURE2D_SIZE_LEVEL", CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL)              //(bindless_array, index: uint, level: uint): uint2
        .value("BINDLESS_TEXTURE3D_SIZE_LEVEL", CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL)              //(bindless_array, index: uint, level: uint): uint3

        .value("BINDLESS_BUFFER_READ", CallOp::BINDLESS_BUFFER_READ)//(bindless_array, index: uint): expr->type()
        .value("BINDLESS_BYTE_BUFFER_READ", CallOp::BINDLESS_BYTE_BUFFER_READ)

        .value("MAKE_BOOL2", CallOp::MAKE_BOOL2)
        .value("MAKE_BOOL3", CallOp::MAKE_BOOL3)
        .value("MAKE_BOOL4", CallOp::MAKE_BOOL4)
        .value("MAKE_INT2", CallOp::MAKE_INT2)
        .value("MAKE_INT3", CallOp::MAKE_INT3)
        .value("MAKE_INT4", CallOp::MAKE_INT4)
        .value("MAKE_UINT2", CallOp::MAKE_UINT2)
        .value("MAKE_UINT3", CallOp::MAKE_UINT3)
        .value("MAKE_UINT4", CallOp::MAKE_UINT4)
        .value("MAKE_FLOAT2", CallOp::MAKE_FLOAT2)
        .value("MAKE_FLOAT3", CallOp::MAKE_FLOAT3)
        .value("MAKE_FLOAT4", CallOp::MAKE_FLOAT4)

        .value("MAKE_SHORT2", CallOp::MAKE_SHORT2)
        .value("MAKE_SHORT3", CallOp::MAKE_SHORT3)
        .value("MAKE_SHORT4", CallOp::MAKE_SHORT4)
        .value("MAKE_USHORT2", CallOp::MAKE_USHORT2)
        .value("MAKE_USHORT3", CallOp::MAKE_USHORT3)
        .value("MAKE_USHORT4", CallOp::MAKE_USHORT4)
        .value("MAKE_LONG2", CallOp::MAKE_LONG2)
        .value("MAKE_LONG3", CallOp::MAKE_LONG3)
        .value("MAKE_LONG4", CallOp::MAKE_LONG4)
        .value("MAKE_ULONG2", CallOp::MAKE_ULONG2)
        .value("MAKE_ULONG3", CallOp::MAKE_ULONG3)
        .value("MAKE_ULONG4", CallOp::MAKE_ULONG4)
        .value("MAKE_HALF2", CallOp::MAKE_HALF2)
        .value("MAKE_HALF3", CallOp::MAKE_HALF3)
        .value("MAKE_HALF4", CallOp::MAKE_HALF4)

        .value("MAKE_FLOAT2X2", CallOp::MAKE_FLOAT2X2)
        .value("MAKE_FLOAT3X3", CallOp::MAKE_FLOAT3X3)
        .value("MAKE_FLOAT4X4", CallOp::MAKE_FLOAT4X4)

        // optimization hints
        .value("ASSUME", CallOp::ASSUME)
        .value("UNREACHABLE", CallOp::UNREACHABLE)
        .value("RASTER_DISCARD", CallOp::RASTER_DISCARD)
        .value("INDIRECT_SET_DISPATCH_KERNEL", CallOp::INDIRECT_SET_DISPATCH_KERNEL)
        .value("INDIRECT_SET_DISPATCH_COUNT", CallOp::INDIRECT_SET_DISPATCH_COUNT)
        .value("RAY_QUERY_PROCEDURAL_CANDIDATE_HIT", CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT)
        .value("RAY_QUERY_WORLD_SPACE_RAY", CallOp::RAY_QUERY_WORLD_SPACE_RAY)
        .value("RAY_QUERY_TRIANGLE_CANDIDATE_HIT", CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT)
        .value("RAY_QUERY_COMMITTED_HIT", CallOp::RAY_QUERY_COMMITTED_HIT)
        .value("RAY_QUERY_COMMIT_TRIANGLE", CallOp::RAY_QUERY_COMMIT_TRIANGLE)
        .value("RAY_QUERY_COMMIT_PROCEDURAL", CallOp::RAY_QUERY_COMMIT_PROCEDURAL)
        .value("RAY_QUERY_TERMINATE", CallOp::RAY_QUERY_TERMINATE)

        .value("RAY_TRACING_INSTANCE_TRANSFORM", CallOp::RAY_TRACING_INSTANCE_TRANSFORM)
        .value("RAY_TRACING_INSTANCE_USER_ID", CallOp::RAY_TRACING_INSTANCE_USER_ID)
        .value("RAY_TRACING_SET_INSTANCE_TRANSFORM", CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM)
        .value("RAY_TRACING_SET_INSTANCE_VISIBILITY", CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY)
        .value("RAY_TRACING_SET_INSTANCE_OPACITY", CallOp::RAY_TRACING_SET_INSTANCE_OPACITY)
        .value("RAY_TRACING_SET_INSTANCE_USER_ID", CallOp::RAY_TRACING_SET_INSTANCE_USER_ID)
        .value("RAY_TRACING_TRACE_CLOSEST", CallOp::RAY_TRACING_TRACE_CLOSEST)
        .value("RAY_TRACING_TRACE_ANY", CallOp::RAY_TRACING_TRACE_ANY)
        .value("RAY_TRACING_QUERY_ALL", CallOp::RAY_TRACING_QUERY_ALL)
        .value("RAY_TRACING_QUERY_ANY", CallOp::RAY_TRACING_QUERY_ANY)

        .value("REQUIRES_GRADIENT", CallOp::REQUIRES_GRADIENT)
        .value("GRADIENT", CallOp::GRADIENT)
        .value("GRADIENT_MARKER", CallOp::GRADIENT_MARKER)
        .value("ACCUMULATE_GRADIENT", CallOp::ACCUMULATE_GRADIENT)
        .value("BACKWARD", CallOp::BACKWARD)
        .value("DETACH", CallOp::DETACH)
        .value("ZERO", CallOp::ZERO)
        .value("ONE", CallOp::ONE)
        .value("REDUCE_SUM", CallOp::REDUCE_SUM)
        .value("REDUCE_PRODUCT", CallOp::REDUCE_PRODUCT)
        .value("REDUCE_MIN", CallOp::REDUCE_MIN)
        .value("REDUCE_MAX", CallOp::REDUCE_MAX)
        .value("OUTER_PRODUCT", CallOp::OUTER_PRODUCT)
        .value("MATRIX_COMPONENT_WISE_MULTIPLICATION", CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION)

        .value("WARP_IS_FIRST_ACTIVE_LANE", CallOp::WARP_IS_FIRST_ACTIVE_LANE)
        .value("WARP_ACTIVE_ALL_EQUAL", CallOp::WARP_ACTIVE_ALL_EQUAL)
        .value("WARP_ACTIVE_BIT_AND", CallOp::WARP_ACTIVE_BIT_AND)
        .value("WARP_ACTIVE_BIT_OR", CallOp::WARP_ACTIVE_BIT_OR)
        .value("WARP_ACTIVE_BIT_XOR", CallOp::WARP_ACTIVE_BIT_XOR)
        .value("WARP_ACTIVE_COUNT_BITS", CallOp::WARP_ACTIVE_COUNT_BITS)
        .value("WARP_ACTIVE_MAX", CallOp::WARP_ACTIVE_MAX)
        .value("WARP_ACTIVE_MIN", CallOp::WARP_ACTIVE_MIN)
        .value("WARP_ACTIVE_PRODUCT", CallOp::WARP_ACTIVE_PRODUCT)
        .value("WARP_ACTIVE_SUM", CallOp::WARP_ACTIVE_SUM)
        .value("WARP_ACTIVE_ALL", CallOp::WARP_ACTIVE_ALL)
        .value("WARP_ACTIVE_ANY", CallOp::WARP_ACTIVE_ANY)
        .value("WARP_ACTIVE_BIT_MASK", CallOp::WARP_ACTIVE_BIT_MASK)
        .value("WARP_PREFIX_COUNT_BITS", CallOp::WARP_PREFIX_COUNT_BITS)
        .value("WARP_PREFIX_SUM", CallOp::WARP_PREFIX_SUM)
        .value("WARP_PREFIX_PRODUCT", CallOp::WARP_PREFIX_PRODUCT)
        .value("WARP_READ_LANE", CallOp::WARP_READ_LANE)
        .value("WARP_READ_FIRST_ACTIVE_LANE", CallOp::WARP_READ_FIRST_ACTIVE_LANE);
}
