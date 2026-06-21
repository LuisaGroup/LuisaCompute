#pragma once
#include "hlsl_codegen.h"
namespace lc::hlsl {
class CodegenConstantPrinter final : public ConstantDecoder {

private:
    vstd::StringBuilder &_str;

public:
    CodegenConstantPrinter(CodegenUtility &codegen,
                           vstd::StringBuilder &str) noexcept
        : _str{str} {}

protected:
    void _decode_bool(bool x) noexcept override {
        PrintValue<bool>{}(x, _str);
    }
    void _decode_char(char x) noexcept override {
        PrintValue<luisa::byte>{}(static_cast<luisa::byte>(x), _str);
    }
    void _decode_uchar(uchar x) noexcept override {
        PrintValue<luisa::ubyte>{}(static_cast<luisa::ubyte>(x), _str);
    }
    void _decode_short(short x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_ushort(ushort x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_int(int x) noexcept override {
        PrintValue<int>{}(x, _str);
    }
    void _decode_uint(uint x) noexcept override {
        PrintValue<uint>{}(x, _str);
    }
    void _decode_long(slong x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_ulong(luisa::ulong x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_half(half x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_float(float x) noexcept override {
        PrintValue<float>{}(x, _str);
    }
    void _decode_double(double x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _vector_separator(const Type *type, uint index) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Should not be called.");
    }
    void _matrix_separator(const Type *type, uint index) noexcept override {
        LUISA_ERROR_WITH_LOCATION("Should not be called.");
    }
    void _decode_vector(const Type *type, const std::byte *data) noexcept override {
#define LUISA_HLSL_DECODE_CONST_VEC(T, N)                   \
    do {                                                    \
        if (type == Type::of<T##N>()) {                     \
            auto x = *reinterpret_cast<const T##N *>(data); \
            if constexpr (N == 3) { _str << "{"sv; }        \
            PrintValue<T##N>{}(x, _str);                    \
            if constexpr (N == 3) { _str << ",0}"sv; }      \
            return;                                         \
        }                                                   \
    } while (false)
#define LUISA_HLSL_DECODE_CONST(T)     \
    LUISA_HLSL_DECODE_CONST_VEC(T, 2); \
    LUISA_HLSL_DECODE_CONST_VEC(T, 3); \
    LUISA_HLSL_DECODE_CONST_VEC(T, 4)
        LUISA_HLSL_DECODE_CONST(bool);
        LUISA_HLSL_DECODE_CONST(byte);
        LUISA_HLSL_DECODE_CONST(ubyte);
        LUISA_HLSL_DECODE_CONST(int);
        LUISA_HLSL_DECODE_CONST(uint);
        LUISA_HLSL_DECODE_CONST(float);
        LUISA_ERROR_WITH_LOCATION(
            "Constant type '{}' is not supported yet.",
            type->description());
#undef LUISA_HLSL_DECODE_CONST_VEC
#undef LUISA_HLSL_DECODE_CONST
    }
    void _decode_matrix(const Type *type, const std::byte *data) noexcept override {
#define LUISA_HLSL_DECODE_CONST_MAT(N)                               \
    do {                                                             \
        using M = float##N##x##N;                                    \
        if (type == Type::of<M>()) {                                 \
            auto x = *reinterpret_cast<const M *>(data);             \
            _str << "float" << #N "x" << (N == 3 ? "4" : #N) << "("; \
            for (auto i = 0; i < N; i++) {                           \
                _str << "float" << (N == 3 ? "4" : #N) << "(";       \
                for (auto j = 0; j < 3; j++) {                       \
                    PrintValue<float>{}(x[i][j], _str);              \
                    if (j != N - 1) { _str << ","; }                 \
                }                                                    \
                if (N == 3) { _str << ",0"; }                        \
                _str << ")";                                         \
                if (i != N - 1) { _str << ","; }                     \
            }                                                        \
            _str << ")";                                             \
            return;                                                  \
        }                                                            \
    } while (false)
        LUISA_HLSL_DECODE_CONST_MAT(2);
        LUISA_HLSL_DECODE_CONST_MAT(3);
        LUISA_HLSL_DECODE_CONST_MAT(4);
        LUISA_ERROR_WITH_LOCATION(
            "Constant type '{}' is not supported yet.",
            type->description());
#undef LUISA_HLSL_DECODE_CONST_MAT
    }
    void _struct_separator(const Type *type, uint index) noexcept override {
        auto n = type->members().size();
        if (index == 0u) {
            _str << "{"sv;
        } else if (index == n) {
            _str << "}"sv;
        } else {
            _str << ',';
        }
    }
    void _array_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _str << "{{"sv;
        } else if (index == n) {
            _str << "}}"sv;
        } else {
            _str << ',';
        }
    }
};
template<typename T>
struct TypeNameStruct {
    void operator()(vstd::StringBuilder &str) {
        if constexpr (std::is_same_v<bool, T>) {
            str << "bool";
        } else if constexpr (std::is_same_v<int, T>) {
            str << "int";
        } else if constexpr (std::is_same_v<uint, T>) {
            str << "uint";
        } else if constexpr (std::is_same_v<float, T>) {
            str << "float";
        } else {
            static_assert(vstd::AlwaysFalse<T>, "illegal type");
        }
    }
};
template<typename T, size_t t>
struct TypeNameStruct<luisa::Vector<T, t>> {
    void operator()(vstd::StringBuilder &str) {
        TypeNameStruct<T>()(str);
        size_t n = (t == 3) ? 4 : t;
        str += ('0' + n);
    }
};
template<size_t t>
struct TypeNameStruct<luisa::Matrix<float, t>> {
    void operator()(vstd::StringBuilder &str) {
        TypeNameStruct<float>()(str);
        if constexpr (t == 2) {
            str << "2x2";
        } else if constexpr (t == 3) {
            str << "4x3";
        } else if constexpr (t == 4) {
            str << "4x4";
        } else {
            static_assert(vstd::AlwaysFalse<luisa::Matrix<float, t>>, "illegal type");
        }
    }
};


}// namespace lc::hlsl