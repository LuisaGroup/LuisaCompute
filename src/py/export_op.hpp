#pragma once

namespace py = pybind11;
using namespace luisa::compute;

void export_op(py::module &m) {

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
    	.value("LERP", CallOp::LERP)
    	.value("STEP", CallOp::STEP)

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
    	.value("LENGTH", CallOp::LENGTH)
    	.value("LENGTH_SQUARED", CallOp::LENGTH_SQUARED)
    	.value("NORMALIZE", CallOp::NORMALIZE)
    	.value("FACEFORWARD", CallOp::FACEFORWARD)

    	.value("DETERMINANT", CallOp::DETERMINANT)
    	.value("TRANSPOSE", CallOp::TRANSPOSE)
    	.value("INVERSE", CallOp::INVERSE)

    	.value("SYNCHRONIZE_BLOCK", CallOp::SYNCHRONIZE_BLOCK)

    	.value("ATOMIC_EXCHANGE", CallOp::ATOMIC_EXCHANGE)        /// [(atomic_ref, desired) -> old]: stores desired, returns old.
    	.value("ATOMIC_COMPARE_EXCHANGE", CallOp::ATOMIC_COMPARE_EXCHANGE)/// [(atomic_ref, expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
    	.value("ATOMIC_FETCH_ADD", CallOp::ATOMIC_FETCH_ADD)       /// [(atomic_ref, val) -> old]: stores (old + val), returns old.
    	.value("ATOMIC_FETCH_SUB", CallOp::ATOMIC_FETCH_SUB)       /// [(atomic_ref, val) -> old]: stores (old - val), returns old.
    	.value("ATOMIC_FETCH_AND", CallOp::ATOMIC_FETCH_AND)       /// [(atomic_ref, val) -> old]: stores (old & val), returns old.
    	.value("ATOMIC_FETCH_OR", CallOp::ATOMIC_FETCH_OR)        /// [(atomic_ref, val) -> old]: stores (old | val), returns old.
    	.value("ATOMIC_FETCH_XOR", CallOp::ATOMIC_FETCH_XOR)       /// [(atomic_ref, val) -> old]: stores (old ^ val), returns old.
    	.value("ATOMIC_FETCH_MIN", CallOp::ATOMIC_FETCH_MIN)       /// [(atomic_ref, val) -> old]: stores min(old, val), returns old.
    	.value("ATOMIC_FETCH_MAX", CallOp::ATOMIC_FETCH_MAX)       /// [(atomic_ref, val) -> old]: stores max(old, val), returns old.

    	.value("BUFFER_READ", CallOp::BUFFER_READ)  /// [(buffer, index) -> value]: reads the index-th element in buffer
    	.value("BUFFER_WRITE", CallOp::BUFFER_WRITE) /// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
    	.value("TEXTURE_READ", CallOp::TEXTURE_READ) /// [(texture, coord) -> value]
    	.value("TEXTURE_WRITE", CallOp::TEXTURE_WRITE)/// [(texture, coord, value) -> void]

    	.value("BINDLESS_TEXTURE2D_SAMPLE", CallOp::BINDLESS_TEXTURE2D_SAMPLE)      //(bindless_array, index: uint, uv: float2): float4
    	.value("BINDLESS_TEXTURE2D_SAMPLE_LEVEL", CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL)//(bindless_array, index: uint, uv: float2, level: float): float4
    	.value("BINDLESS_TEXTURE2D_SAMPLE_GRAD", CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD) //(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    	.value("BINDLESS_TEXTURE3D_SAMPLE", CallOp::BINDLESS_TEXTURE3D_SAMPLE)      //(bindless_array, index: uint, uv: float3): float4
    	.value("BINDLESS_TEXTURE3D_SAMPLE_LEVEL", CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL)//(bindless_array, index: uint, uv: float3, level: float): float4
    	.value("BINDLESS_TEXTURE3D_SAMPLE_GRAD", CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD) //(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    	.value("BINDLESS_TEXTURE2D_READ", CallOp::BINDLESS_TEXTURE2D_READ)        //(bindless_array, index: uint, coord: uint2): float4
    	.value("BINDLESS_TEXTURE3D_READ", CallOp::BINDLESS_TEXTURE3D_READ)        //(bindless_array, index: uint, coord: uint3): float4
    	.value("BINDLESS_TEXTURE2D_READ_LEVEL", CallOp::BINDLESS_TEXTURE2D_READ_LEVEL)  //(bindless_array, index: uint, coord: uint2, level: uint): float4
    	.value("BINDLESS_TEXTURE3D_READ_LEVEL", CallOp::BINDLESS_TEXTURE3D_READ_LEVEL)  //(bindless_array, index: uint, coord: uint3, level: uint): float4
    	.value("BINDLESS_TEXTURE2D_SIZE", CallOp::BINDLESS_TEXTURE2D_SIZE)        //(bindless_array, index: uint): uint2
    	.value("BINDLESS_TEXTURE3D_SIZE", CallOp::BINDLESS_TEXTURE3D_SIZE)        //(bindless_array, index: uint): uint3
    	.value("BINDLESS_TEXTURE2D_SIZE_LEVEL", CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL)  //(bindless_array, index: uint, level: uint): uint2
    	.value("BINDLESS_TEXTURE3D_SIZE_LEVEL", CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL)  //(bindless_array, index: uint, level: uint): uint3

    	.value("BINDLESS_BUFFER_READ", CallOp::BINDLESS_BUFFER_READ)//(bindless_array, index: uint): expr->type()

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

    	.value("MAKE_FLOAT2X2", CallOp::MAKE_FLOAT2X2)
    	.value("MAKE_FLOAT3X3", CallOp::MAKE_FLOAT3X3)
    	.value("MAKE_FLOAT4X4", CallOp::MAKE_FLOAT4X4)

    	// optimization hints
    	.value("ASSUME", CallOp::ASSUME)
    	.value("UNREACHABLE", CallOp::UNREACHABLE)

    	.value("INSTANCE_TO_WORLD_MATRIX", CallOp::INSTANCE_TO_WORLD_MATRIX)

    	.value("TRACE_CLOSEST", CallOp::TRACE_CLOSEST)
    	.value("TRACE_ANY", CallOp::TRACE_ANY);
}
