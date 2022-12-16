//
// Created by Mike Smith on 2021/12/31.
//

#include <bit>
#include <charconv>

#include <core/logging.h>
#include <ast/type_registry.h>

namespace luisa::compute::detail {

inline uint64_t compute::detail::TypeRegistry::_hash(std::string_view desc) noexcept {
    using namespace std::string_view_literals;
    return hash64(desc, hash64("__hash_type"sv));
}

const Type *TypeRegistry::_decode(std::string_view desc) noexcept {

    // TYPE := BASIC | ARRAY | VECTOR | MATRIX | STRUCT
    // BASIC := int | uint | bool | float
    // ARRAY := array<BASIC,N>
    // VECTOR := vector<BASIC,2> | vector<BASIC,3> | vector<BASIC,4>
    // MATRIX := matrix<2> | matrix<3> | matrix<4>
    // STRUCT := struct<4,TYPE+> | struct<8,TYPE+> | struct<16,TYPE+>

    // buffer<Type>
    // texture<n,int|uint|float>
    // bindless_array
    // accel

    auto hash = _hash(desc);
    if (auto iter = _type_set.find(hash); iter != _type_set.cend()) {
        return *iter;
    }

    using namespace std::string_view_literals;
    auto read_identifier = [&desc]() noexcept {
        auto i = 0u;
        for (; i < desc.size() && (isalpha(desc[i]) || desc[i] == '_'); i++) {}
        auto t = desc.substr(0u, i);
        desc = desc.substr(i);
        return t;
    };

    auto read_number = [&desc]() noexcept {
        size_t number;
        auto result = std::from_chars(desc.data(), desc.data() + desc.size(), number);
        if (result.ec != std::errc{}) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to parse number from type description: '{}'.",
                desc);
        }
        desc = desc.substr(result.ptr - desc.data());
        return number;
    };

    auto match = [&desc](char c) noexcept {
        if (!desc.starts_with(c)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Expected '{}' from type description: '{}'.",
                c, desc);
        }
        desc = desc.substr(1);
    };

    auto split = [&desc]() noexcept {
        auto balance = 0u;
        auto i = 0u;
        for (; i < desc.size(); i++) {
            if (auto c = desc[i]; c == '<') {
                balance++;
            } else if (c == '>') {
                if (balance == 0u) { break; }
                if (--balance == 0u) {
                    i++;
                    break;
                }
            } else if (c == ',' && balance == 0u) {
                break;
            }
        }
        if (balance != 0u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Unbalanced '<' and '>' in "
                "type description: {}.",
                desc);
        }
        auto t = desc.substr(0u, i);
        desc = desc.substr(i);
        return t;
    };

    auto info = luisa::make_unique<Type>();
    info->_description = desc;
    info->_hash = hash;

    auto type_identifier = read_identifier();

#define TRY_PARSE_SCALAR_TYPE(T, TAG)  \
    if (type_identifier == #T##sv) {   \
        info->_tag = Type::Tag::TAG;   \
        info->_size = sizeof(T);       \
        info->_alignment = alignof(T); \
    } else
    TRY_PARSE_SCALAR_TYPE(bool, BOOL)
    TRY_PARSE_SCALAR_TYPE(float, FLOAT)
    TRY_PARSE_SCALAR_TYPE(int, INT)
    TRY_PARSE_SCALAR_TYPE(uint, UINT)
#undef TRY_PARSE_SCALAR_TYPE
    if (type_identifier == "vector"sv) {
        info->_tag = Type::Tag::VECTOR;
        match('<');
        info->_members.emplace_back(_decode(split()));
        match(',');
        info->_dimension = read_number();
        match('>');
        auto elem = info->_members.front();
        if (!elem->is_scalar()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid vector element: {}.",
                elem->description());
        }
        if (info->_dimension != 2 &&
            info->_dimension != 3 &&
            info->_dimension != 4) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid vector dimension: {}.",
                info->dimension());
        }
        info->_size = info->_alignment = elem->size() * (info->_dimension == 3 ? 4 : info->_dimension);
    } else if (type_identifier == "matrix"sv) {
        info->_tag = Type::Tag::MATRIX;
        match('<');
        info->_dimension = read_number();
        match('>');
        info->_members.emplace_back(Type::of<float>());
        if (info->_dimension == 2) {
            info->_size = sizeof(float2x2);
            info->_alignment = alignof(float2x2);
        } else if (info->_dimension == 3) {
            info->_size = sizeof(float3x3);
            info->_alignment = alignof(float3x3);
        } else if (info->_dimension == 4) {
            info->_size = sizeof(float4x4);
            info->_alignment = alignof(float4x4);
        } else [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid matrix dimension: {}.",
                info->_dimension);
        }
    } else if (type_identifier == "array"sv) {
        info->_tag = Type::Tag::ARRAY;
        match('<');
        info->_members.emplace_back(_decode(split()));
        match(',');
        info->_dimension = read_number();
        match('>');
        if (info->_members.back()->is_buffer() ||
            info->_members.back()->is_texture()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Arrays are not allowed to "
                "hold buffers or images.");
        }
        info->_alignment = info->_members.front()->alignment();
        info->_size = info->_members.front()->size() * info->_dimension;
    } else if (type_identifier == "struct"sv) {
        info->_tag = Type::Tag::STRUCTURE;
        match('<');
        info->_alignment = read_number();
        while (desc.starts_with(',')) {
            desc = desc.substr(1);
            info->_members.emplace_back(_decode(split()));
        }
        match('>');
        info->_size = 0u;
        auto max_member_alignment = static_cast<size_t>(0u);
        for (auto member : info->_members) {
            if (member->is_buffer() || member->is_texture()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Structures are not allowed to have buffers or images as members.");
            }
            auto ma = member->alignment();
            max_member_alignment = std::max(ma, max_member_alignment);
            info->_size = (info->_size + ma - 1u) / ma * ma + member->size();
        }
        if (auto a = info->_alignment; a > 16u || std::bit_floor(a) != a) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid structure alignment {}.", a);
        } else if (a < max_member_alignment && a != 0u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Struct alignment {} is smaller than the largest member alignment {}.",
                info->_alignment, max_member_alignment);
        }
        info->_size = (info->_size + info->_alignment - 1u) / info->_alignment * info->_alignment;
    } else if (type_identifier == "buffer"sv) {
        info->_tag = Type::Tag::BUFFER;
        match('<');
        auto m = info->_members.emplace_back(_decode(split()));
        match('>');
        if (m->is_buffer() || m->is_texture()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Buffers are not allowed to "
                "hold buffers or images.");
        }
        info->_alignment = 8u;
        info->_size = 8u;
    } else if (type_identifier == "texture"sv) {
        info->_tag = Type::Tag::TEXTURE;
        match('<');
        info->_dimension = read_number();
        match(',');
        auto m = info->_members.emplace_back(_decode(split()));
        match('>');
        if (auto t = m->tag();
            t != Type::Tag::INT &&
            t != Type::Tag::UINT &&
            t != Type::Tag::FLOAT) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Images can only hold int, uint, or float.");
        }
        info->_size = 8u;
        info->_alignment = 8u;
    } else if (type_identifier == "bindless_array"sv) {
        info->_tag = Type::Tag::BINDLESS_ARRAY;
        info->_size = 8u;
        info->_alignment = 8u;
    } else if (type_identifier == "accel"sv) {
        info->_tag = Type::Tag::ACCEL;
        info->_size = 8u;
        info->_alignment = 8u;
    } else [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Unknown type identifier: {}.",
            type_identifier);
    }
    if (!desc.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Found junk after type description: {}.",
            desc);
    }

    info->_index = static_cast<uint32_t>(_types.size());
    auto [iter, not_present_before] = _type_set.emplace(info.get());
    if (not_present_before) [[likely]] { _types.emplace_back(std::move(info)); }
    return *iter;
}

TypeRegistry &TypeRegistry::instance() noexcept {
    static TypeRegistry r;
    return r;
}

const Type *TypeRegistry::type_from(luisa::string_view desc) noexcept {
    std::unique_lock lock{_mutex};
    if (desc == "void") { return nullptr; }
    return _decode(desc);
}

const Type *TypeRegistry::type_from(uint64_t hash) noexcept {
    std::unique_lock lock{_mutex};
    auto iter = _type_set.find(hash);
    if (iter == _type_set.end()) {
        LUISA_ERROR_WITH_LOCATION("Invalid type hash: {}.", hash);
    }
    return *iter;
}

const Type *TypeRegistry::type_at(size_t i) const noexcept {
    std::unique_lock lock{_mutex};
    if (i >= _types.size()) { LUISA_ERROR_WITH_LOCATION("Invalid type index: {}.", i); }
    return _types[i].get();
}

size_t TypeRegistry::type_count() const noexcept {
    std::unique_lock lock{_mutex};
    return _types.size();
}

void TypeRegistry::traverse(TypeVisitor &visitor) const noexcept {
    std::unique_lock lock{_mutex};
    for (auto &&t : _types) {
        visitor.visit(t.get());
    }
}

}// namespace luisa::compute::detail
