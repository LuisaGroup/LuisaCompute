#include <luisa/core/stl/hash.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ast/constant_data.h>

namespace luisa::compute {

void ConstantDecoder::_decode_vector(const Type *type, const std::byte *data) noexcept {
    auto elem = type->element();
    auto elem_size = elem->size();
    auto n = type->dimension();
    for (auto i = 0u; i < n; i++) {
        _vector_separator(type, i);
        auto p_elem = data + i * elem_size;
        _decode(elem, p_elem);
    }
    _vector_separator(type, n);
}

void ConstantDecoder::_decode_matrix(const Type *type, const std::byte *data) noexcept {
    auto n = type->dimension();
    auto elem = Type::vector(type->element(), n);
    auto elem_size = elem->size();
    for (auto i = 0u; i < n; i++) {
        _matrix_separator(type, i);
        auto p_elem = data + i * elem_size;
        _decode(elem, p_elem);
    }
    _matrix_separator(type, n);
}

void ConstantDecoder::_decode_struct(const Type *type, const std::byte *data) noexcept {
    auto n = type->members().size();
    auto offset = 0u;
    for (auto i = 0u; i < n; i++) {
        _struct_separator(type, i);
        auto elem = type->members()[i];
        offset = luisa::align(offset, elem->alignment());
        auto p_elem = data + offset;
        offset += elem->size();
        _decode(elem, p_elem);
    }
    _struct_separator(type, n);
}

void ConstantDecoder::_decode_array(const Type *type, const std::byte *data) noexcept {
    auto elem = type->element();
    auto elem_size = elem->size();
    auto n = type->dimension();
    for (auto i = 0u; i < n; i++) {
        _array_separator(type, i);
        auto p_elem = data + i * elem_size;
        _decode(elem, p_elem);
    }
    _array_separator(type, n);
}

void ConstantDecoder::_decode(const Type *type, const std::byte *data) noexcept {
    switch (type->tag()) {
        case Type::Tag::BOOL: _decode_bool(*reinterpret_cast<const bool *>(data)); break;
        case Type::Tag::FLOAT32: _decode_float(*reinterpret_cast<const float *>(data)); break;
        case Type::Tag::INT32: _decode_int(*reinterpret_cast<const int *>(data)); break;
        case Type::Tag::UINT32: _decode_uint(*reinterpret_cast<const uint *>(data)); break;
        case Type::Tag::INT64: _decode_long(*reinterpret_cast<const slong *>(data)); break;
        case Type::Tag::UINT64: _decode_ulong(*reinterpret_cast<const ulong *>(data)); break;
        case Type::Tag::FLOAT16: _decode_half(*reinterpret_cast<const half *>(data)); break;
        case Type::Tag::INT16: _decode_short(*reinterpret_cast<const short *>(data)); break;
        case Type::Tag::UINT16: _decode_ushort(*reinterpret_cast<const ushort *>(data)); break;
        case Type::Tag::VECTOR: _decode_vector(type, data); break;
        case Type::Tag::MATRIX: _decode_matrix(type, data); break;
        case Type::Tag::ARRAY: _decode_array(type, data); break;
        case Type::Tag::STRUCTURE: _decode_struct(type, data); break;
        default: LUISA_ERROR_WITH_LOCATION(
            "Unsupported constant type: {}.",
            type->description());
    }
}

void ConstantDecoder::decode(const Type *type, const std::byte *data) noexcept {
    _decode(type, data);
}

namespace detail {

struct ConstantContainer {
    const Type *type;
    size_t hash;
    luisa::vector<std::byte> data;
};

[[nodiscard]] auto &constant_registry() noexcept {
    static luisa::vector<ConstantContainer> r;
    return r;
}

[[nodiscard]] auto &constant_registry_mutex() noexcept {
    static spin_mutex m;
    return m;
}

}// namespace detail

class ConstantSerializer final : public ConstantDecoder {

private:
    luisa::string _s;

public:
    void decode(const Type *type, const std::byte *data) noexcept override {
        ConstantDecoder::decode(type, data);
    }

protected:
    void _decode_bool(bool x) noexcept override { _s.append(luisa::format("bool({})", x)); }
    void _decode_short(short x) noexcept override { _s.append(luisa::format("short({})", x)); }
    void _decode_ushort(ushort x) noexcept override { _s.append(luisa::format("ushort({})", x)); }
    void _decode_int(int x) noexcept override { _s.append(luisa::format("int({})", x)); }
    void _decode_uint(uint x) noexcept override { _s.append(luisa::format("uint({})", x)); }
    void _decode_long(slong x) noexcept override { _s.append(luisa::format("long({})", x)); }
    void _decode_ulong(ulong x) noexcept override { _s.append(luisa::format("ulong({})", x)); }
    void _decode_half(half x) noexcept override { _s.append(luisa::format("half({})", static_cast<float>(x))); }
    void _decode_float(float x) noexcept override { _s.append(luisa::format("float({})", luisa::bit_cast<uint>(x))); }
    void _decode_double(double x) noexcept override { _s.append(luisa::format("double({})", luisa::bit_cast<ulong>(x))); }
    void _vector_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _s.append(luisa::format("{}{}(", type->element()->description(), n));
        } else if (index == n) {
            _s.append(")");
        } else {
            _s.append(", ");
        }
    }
    void _matrix_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _s.append(luisa::format("{}{}x{}(", type->element()->description(), n, n));
        } else if (index == n) {
            _s.append(")");
        } else {
            _s.append(", ");
        }
    }
    void _struct_separator(const Type *type, uint index) noexcept override {
        auto n = type->members().size();
        if (index == 0u) {
            _s.append(luisa::format("struct<{}>", type->alignment())).append("{");
        } else if (index == n) {
            _s.append("}");
        } else {
            _s.append(", ");
        }
    }
    void _array_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _s.append("array[");
        } else if (index == n) {
            _s.append("]");
        } else {
            _s.append(", ");
        }
    }

public:
    [[nodiscard]] auto s() const noexcept { return luisa::string_view{_s}; }
};

ConstantData::ConstantData(const Type *type, const std::byte *data, uint64_t hash) noexcept
    : _type{type}, _raw{data}, _hash{hash} {}

ConstantData ConstantData::create(const Type *type, const void *data, size_t size) noexcept {
    LUISA_ASSERT(type->size() == size,
                 "Size mismatch for constant data of type '{}'.",
                 type->description());
    ConstantSerializer serializer;
    serializer.decode(type, reinterpret_cast<const std::byte *>(data));
    auto hash = luisa::hash_value(serializer.s());
    std::scoped_lock lock{detail::constant_registry_mutex()};
    auto &registry = detail::constant_registry();
    for (auto &&c : registry) {
        if (c.hash == hash && c.type == type) {
            return ConstantData{type, c.data.data(), hash};
        }
    }
    luisa::vector<std::byte> buffer(size);
    memcpy(buffer.data(), data, size);
    registry.emplace_back(detail::ConstantContainer{
        type, hash, std::move(buffer)});
    auto &&c = registry.back();
    return ConstantData{c.type, c.data.data(), c.hash};
}

}// namespace luisa::compute
