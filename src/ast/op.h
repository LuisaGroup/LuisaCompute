//
// Created by Mike Smith on 2021/6/30.
//

#pragma once

#include <bitset>
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
    NONE,

    SELECT,

    CLAMP,
    LERP,
    SATURATE,
    SIGN,

    STEP,
    SMOOTHSTEP,

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
    MOD,
    FMOD,

    DEGREES,
    RADIANS,

    FMA,
    COPYSIGN,

    CROSS,
    DOT,
    DISTANCE,
    DISTANCE_SQUARED,
    LENGTH,
    LENGTH_SQUARED,
    NORMALIZE,
    FACEFORWARD,

    DETERMINANT,
    TRANSPOSE,
    INVERSE,

    BLOCK_BARRIER,
    DEVICE_BARRIER,
    ALL_BARRIER,

    ATOMIC_LOAD,
    ATOMIC_STORE,
    ATOMIC_EXCHANGE,
    ATOMIC_COMPARE_EXCHANGE,
    ATOMIC_FETCH_ADD,
    ATOMIC_FETCH_SUB,
    ATOMIC_FETCH_AND,
    ATOMIC_FETCH_OR,
    ATOMIC_FETCH_XOR,
    ATOMIC_FETCH_MIN,
    ATOMIC_FETCH_MAX,

    TEXTURE_READ,
    TEXTURE_WRITE,

    TEXTURE_HEAP_SAMPLE2D,
    TEXTURE_HEAP_SAMPLE2D_LEVEL,
    TEXTURE_HEAP_SAMPLE2D_GRAD,
    TEXTURE_HEAP_SAMPLE3D,
    TEXTURE_HEAP_SAMPLE3D_LEVEL,
    TEXTURE_HEAP_SAMPLE3D_GRAD,
    TEXTURE_HEAP_READ2D,
    TEXTURE_HEAP_READ3D,
    TEXTURE_HEAP_READ2D_LEVEL,
    TEXTURE_HEAP_READ3D_LEVEL,
    TEXTURE_HEAP_SIZE2D,
    TEXTURE_HEAP_SIZE3D,
    TEXTURE_HEAP_SIZE2D_LEVEL,
    TEXTURE_HEAP_SIZE3D_LEVEL,

    BUFFER_HEAP_READ,

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

    TRACE_CLOSEST,
    TRACE_ANY
};

static constexpr size_t call_op_count = to_underlying(CallOp::TRACE_ANY) + 1u;

class CallOpSet {

public:
    using Bitset = std::bitset<call_op_count>;

    class Iterator {

    public:
        struct End {};

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
        [[nodiscard]] bool operator==(End) const noexcept;
    };

private:
    Bitset _bits;

public:
    CallOpSet() noexcept = default;
    ~CallOpSet() noexcept = default;
    void mark(CallOp op) noexcept { _bits.set(to_underlying(op)); }
    [[nodiscard]] auto test(CallOp op) const noexcept { return _bits.test(to_underlying(op)); }
    [[nodiscard]] auto begin() const noexcept { return Iterator{*this}; }
    [[nodiscard]] auto end() const noexcept { return Iterator::End{}; }
};

}// namespace luisa::compute
