//
// Created by Mike Smith on 2021/6/30.
//

#pragma once

#include <bitset>
#include <iterator>

#include <core/basic_types.h>

namespace luisa::compute {

enum struct UnaryOp : uint32_t {
    PLUS,
    MINUS,  // +x, -x
    NOT,    // !x
    BIT_NOT,// ~x
    // Note: We deliberately support *NO* pre and postfix inc/dec operators to avoid possible abuse
};

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

enum struct CallOp : uint32_t {

    CUSTOM,

    ALL,
    ANY,

    SELECT,
    CLAMP,
    LERP,
    STEP,

    ABS,
    MIN,
    MAX,

    CLZ,
    CTZ,
    POPCOUNT,
    REVERSE,

    ISINF,
    ISNAN,

    ACOS,
    ACOSH,
    ASIN,
    ASINH,
    ATAN,
    ATAN2,
    ATANH,

    COS,
    COSH,
    SIN,
    SINH,
    TAN,
    TANH,

    EXP,
    EXP2,
    EXP10,
    LOG,
    LOG2,
    LOG10,
    POW,

    SQRT,
    RSQRT,

    CEIL,
    FLOOR,
    FRACT,
    TRUNC,
    ROUND,

    FMA,
    COPYSIGN,

    CROSS,
    DOT,
    LENGTH,
    LENGTH_SQUARED,
    NORMALIZE,
    FACEFORWARD,

    DETERMINANT,
    TRANSPOSE,
    INVERSE,

    SYNCHRONIZE_BLOCK,

    ATOMIC_EXCHANGE,
    ATOMIC_COMPARE_EXCHANGE,
    ATOMIC_FETCH_ADD,
    ATOMIC_FETCH_SUB,
    ATOMIC_FETCH_AND,
    ATOMIC_FETCH_OR,
    ATOMIC_FETCH_XOR,
    ATOMIC_FETCH_MIN,
    ATOMIC_FETCH_MAX,

    BUFFER_READ,
    BUFFER_WRITE,
    TEXTURE_READ,
    TEXTURE_WRITE,

    BINDLESS_TEXTURE2D_SAMPLE,
    BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
    BINDLESS_TEXTURE2D_SAMPLE_GRAD,
    BINDLESS_TEXTURE3D_SAMPLE,
    BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
    BINDLESS_TEXTURE3D_SAMPLE_GRAD,
    BINDLESS_TEXTURE2D_READ,
    BINDLESS_TEXTURE3D_READ,
    BINDLESS_TEXTURE2D_READ_LEVEL,
    BINDLESS_TEXTURE3D_READ_LEVEL,
    BINDLESS_TEXTURE2D_SIZE,
    BINDLESS_TEXTURE3D_SIZE,
    BINDLESS_TEXTURE2D_SIZE_LEVEL,
    BINDLESS_TEXTURE3D_SIZE_LEVEL,

    BINDLESS_BUFFER_READ,

    MAKE_BOOL2,
    MAKE_BOOL3,
    MAKE_BOOL4,
    MAKE_INT2,
    MAKE_INT3,
    MAKE_INT4,
    MAKE_UINT2,
    MAKE_UINT3,
    MAKE_UINT4,
    MAKE_FLOAT2,
    MAKE_FLOAT3,
    MAKE_FLOAT4,

    MAKE_FLOAT2X2,
    MAKE_FLOAT3X3,
    MAKE_FLOAT4X4,

    // optimization hints
    ASSUME,
    UNREACHABLE,

    INSTANCE_TO_WORLD_MATRIX,

    TRACE_CLOSEST,
    TRACE_ANY
};

static constexpr size_t call_op_count = to_underlying(CallOp::TRACE_ANY) + 1u;

class CallOpSet {

public:
    using Bitset = std::bitset<call_op_count>;

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
        [[nodiscard]] bool operator==(std::default_sentinel_t) const noexcept;
    };

private:
    Bitset _bits;

public:
    CallOpSet() noexcept = default;
    ~CallOpSet() noexcept = default;
    void mark(CallOp op) noexcept { _bits.set(to_underlying(op)); }
    [[nodiscard]] auto test(CallOp op) const noexcept { return _bits.test(to_underlying(op)); }
    [[nodiscard]] auto begin() const noexcept { return Iterator{*this}; }
    [[nodiscard]] auto end() const noexcept { return std::default_sentinel; }
};

enum struct AssignOp {
    ASSIGN,
    ADD_ASSIGN,
    SUB_ASSIGN,
    MUL_ASSIGN,
    DIV_ASSIGN,
    MOD_ASSIGN,
    BIT_AND_ASSIGN,
    BIT_OR_ASSIGN,
    BIT_XOR_ASSIGN,
    SHL_ASSIGN,
    SHR_ASSIGN
};

}// namespace luisa::compute
