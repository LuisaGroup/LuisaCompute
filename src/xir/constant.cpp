#include <luisa/core/stl/hash.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/module.h>
#include <luisa/xir/constant.h>

#include <utility>

namespace luisa::compute::xir {

namespace detail {

static void xir_constant_fill_data(const Type *t, const void *raw, void *data) noexcept {
    LUISA_DEBUG_ASSERT(t != nullptr && raw != nullptr && data != nullptr, "Invalid arguments.");
    if (t->is_bool()) {
        auto value = static_cast<uint8_t>(*static_cast<const bool *>(raw) ? 1u : 0u);
        memmove(data, &value, 1);
    } else if (t->is_scalar()) {
        memmove(data, raw, t->size());
    } else {
        switch (t->tag()) {
            case Type::Tag::VECTOR: {
                auto elem_type = t->element();
                auto dim = t->dimension();
                for (auto i = 0u; i < dim; i++) {
                    auto offset = i * elem_type->size();
                    auto raw_elem = static_cast<const std::byte *>(raw) + offset;
                    auto data_elem = static_cast<std::byte *>(data) + offset;
                    xir_constant_fill_data(elem_type, raw_elem, data_elem);
                }
                break;
            }
            case Type::Tag::MATRIX: {
                auto elem_type = t->element();
                auto dim = t->dimension();
                auto col_type = Type::vector(elem_type, dim);
                for (auto i = 0u; i < dim; i++) {
                    auto offset = i * col_type->size();
                    auto raw_col = static_cast<const std::byte *>(raw) + offset;
                    auto data_col = static_cast<std::byte *>(data) + offset;
                    xir_constant_fill_data(col_type, raw_col, data_col);
                }
                break;
            }
            case Type::Tag::ARRAY: {
                auto elem_type = t->element();
                auto dim = t->dimension();
                for (auto i = 0u; i < dim; i++) {
                    auto offset = i * elem_type->size();
                    auto raw_elem = static_cast<const std::byte *>(raw) + offset;
                    auto data_elem = static_cast<std::byte *>(data) + offset;
                    xir_constant_fill_data(elem_type, raw_elem, data_elem);
                }
                break;
            }
            case Type::Tag::STRUCTURE: {
                size_t offset = 0u;
                for (auto m : t->members()) {
                    offset = luisa::align(offset, m->alignment());
                    auto raw_member = static_cast<const std::byte *>(raw) + offset;
                    auto data_member = static_cast<std::byte *>(data) + offset;
                    xir_constant_fill_data(m, raw_member, data_member);
                    offset += m->size();
                }
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Unsupported constant type.");
        }
    }
}

static void xir_constant_fill_one(const Type *t, void *data) noexcept {
    LUISA_DEBUG_ASSERT(t != nullptr && data != nullptr, "Invalid arguments.");
    if (t->is_bool()) {
        auto value = static_cast<uint8_t>(1u);
        memmove(data, &value, 1);
    } else if (t->is_scalar()) {
        auto do_copy = [data, t]<typename T>(T x) noexcept {
            LUISA_DEBUG_ASSERT(Type::of<T>() == t, "Type mismatch.");
            memmove(data, &x, sizeof(T));
        };
        switch (t->tag()) {
            case Type::Tag::INT8: do_copy(static_cast<int8_t>(1)); break;
            case Type::Tag::UINT8: do_copy(static_cast<uint8_t>(1)); break;
            case Type::Tag::INT16: do_copy(static_cast<int16_t>(1)); break;
            case Type::Tag::UINT16: do_copy(static_cast<uint16_t>(1)); break;
            case Type::Tag::INT32: do_copy(static_cast<int32_t>(1)); break;
            case Type::Tag::UINT32: do_copy(static_cast<uint32_t>(1)); break;
            case Type::Tag::INT64: do_copy(static_cast<int64_t>(1)); break;
            case Type::Tag::UINT64: do_copy(static_cast<uint64_t>(1)); break;
            case Type::Tag::FLOAT16: do_copy(static_cast<half>(1.0f)); break;
            case Type::Tag::FLOAT32: do_copy(static_cast<float>(1.0f)); break;
            case Type::Tag::FLOAT64: do_copy(static_cast<double>(1.0)); break;
            default: LUISA_ERROR_WITH_LOCATION("Unsupported scalar type.");
        }
    } else {
        switch (t->tag()) {
            case Type::Tag::VECTOR: {
                auto elem_type = t->element();
                auto dim = t->dimension();
                for (auto i = 0u; i < dim; i++) {
                    auto offset = i * elem_type->size();
                    auto data_elem = static_cast<std::byte *>(data) + offset;
                    xir_constant_fill_one(elem_type, data_elem);
                }
                break;
            }
            case Type::Tag::MATRIX: {
                auto elem_type = t->element();
                auto dim = t->dimension();
                auto col_type = Type::vector(elem_type, dim);
                for (auto i = 0u; i < dim; i++) {
                    auto offset = i * col_type->size();
                    auto data_col = static_cast<std::byte *>(data) + offset;
                    xir_constant_fill_one(col_type, data_col);
                }
                break;
            }
            case Type::Tag::ARRAY: {
                auto elem_type = t->element();
                auto dim = t->dimension();
                for (auto i = 0u; i < dim; i++) {
                    auto offset = i * elem_type->size();
                    auto data_elem = static_cast<std::byte *>(data) + offset;
                    xir_constant_fill_one(elem_type, data_elem);
                }
                break;
            }
            case Type::Tag::STRUCTURE: {
                size_t offset = 0u;
                for (auto m : t->members()) {
                    offset = luisa::align(offset, m->alignment());
                    auto data_member = static_cast<std::byte *>(data) + offset;
                    xir_constant_fill_one(m, data_member);
                    offset += m->size();
                }
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Unsupported constant type.");
        }
    }
}

}// namespace detail

void *Constant::_data() noexcept {
    return _is_small() ? _small : _large;
}

void Constant::_check_reinterpret_cast_type_size(size_t size) const noexcept {
    LUISA_ASSERT(type()->size() == size, "Type size mismatch.");
}

void Constant::_update_hash(luisa::optional<uint64_t> hash) noexcept {
    static constexpr auto compute = [](const Type *type, const void *data) noexcept {
        auto hv = luisa::hash64(data, type->size(), luisa::hash64_default_seed);
        return luisa::hash_combine({type->hash(), hv});
    };
    if (hash.has_value()) {
        LUISA_DEBUG_ASSERT(compute(type(), _data()) == hash.value(), "Hash mismatch.");
        _hash = hash.value();
    } else {
        _hash = compute(type(), _data());
    }
}

Constant::Constant(Module *parent_module, const Type *type) noexcept
    : Super{parent_module, type} {
    LUISA_DEBUG_ASSERT(type != nullptr && !type->is_custom() && !type->is_resource(),
                       "Invalid constant type: {}.", type == nullptr ? "void" : type->description());
    if (!_is_small()) { _large = luisa::allocate_with_allocator<std::byte>(type->size()); }
    std::memset(_data(), 0, type->size());
}

bool Constant::_is_small() const noexcept {
    return type()->size() <= sizeof(void *);
}

Constant::Constant(Module *parent_module, const Type *type, const void *data,
                   luisa::optional<uint64_t> hash) noexcept
    : Constant{parent_module, type} {
    LUISA_DEBUG_ASSERT(data != nullptr, "Data must not be null.");
    detail::xir_constant_fill_data(type, data, _data());
    _update_hash(std::move(hash));
}

Constant::Constant(Module *parent_module, const Type *type, ctor_tag_zero,
                   luisa::optional<uint64_t> hash) noexcept
    : Constant{parent_module, type} {
    // already memset to zero in the delegate constructor
    _update_hash(std::move(hash));
}

Constant::Constant(Module *parent_module, const Type *type, ctor_tag_one,
                   luisa::optional<uint64_t> hash) noexcept
    : Constant{parent_module, type} {
    detail::xir_constant_fill_one(type, _data());
    _update_hash(std::move(hash));
}

Constant::Constant(Module *parent_module, ctor_tag_sentinel) noexcept
    : Super{parent_module, nullptr} {}

const void *Constant::data() const noexcept {
    return const_cast<Constant *>(this)->_data();
}

Constant::~Constant() noexcept {
    if (type() != nullptr && !_is_small()) {
        luisa::deallocate_with_allocator(static_cast<std::byte *>(_large));
    }
}

}// namespace luisa::compute::xir
