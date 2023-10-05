#pragma once

#include <luisa/ast/type.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/concepts.h>

namespace luisa::compute {
class CallableLibrary;
class LC_AST_API ConstantDecoder {

protected:
    virtual void _decode_bool(bool x) noexcept = 0;
    virtual void _decode_short(short x) noexcept = 0;
    virtual void _decode_ushort(ushort x) noexcept = 0;
    virtual void _decode_int(int x) noexcept = 0;
    virtual void _decode_uint(uint x) noexcept = 0;
    virtual void _decode_long(slong x) noexcept = 0;
    virtual void _decode_ulong(ulong x) noexcept = 0;
    virtual void _decode_half(half x) noexcept = 0;
    virtual void _decode_float(float x) noexcept = 0;
    virtual void _decode_double(double x) noexcept = 0;
    virtual void _vector_separator(const Type *type, uint index) noexcept = 0;
    virtual void _matrix_separator(const Type *type, uint index) noexcept = 0;
    virtual void _struct_separator(const Type *type, uint index) noexcept = 0;
    virtual void _array_separator(const Type *type, uint index) noexcept = 0;
    virtual void _decode_vector(const Type *type, const std::byte *data) noexcept;
    virtual void _decode_matrix(const Type *type, const std::byte *data) noexcept;
    virtual void _decode_struct(const Type *type, const std::byte *data) noexcept;
    virtual void _decode_array(const Type *type, const std::byte *data) noexcept;
    virtual void _decode(const Type *type, const std::byte *data) noexcept;

public:
    virtual void decode(const Type *type, const std::byte *data) noexcept;
};

class LC_AST_API ConstantData {

    friend class CallableLibrary;

private:
    const Type *_type;
    const std::byte *_raw;
    uint64_t _hash;

private:
    ConstantData(const Type *type, const std::byte *data, uint64_t hash) noexcept;

public:
    ConstantData() noexcept = default;
    [[nodiscard]] static ConstantData create(const Type *type, const void *data, size_t size) noexcept;
    [[nodiscard]] auto raw() const noexcept { return _raw; }
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto hash() const noexcept { return _hash; }
    [[nodiscard]] bool operator==(const ConstantData &rhs) const noexcept { return _hash == rhs._hash; }
    void decode(ConstantDecoder &d) const noexcept { d.decode(_type, _raw); }
};

}// namespace luisa::compute
