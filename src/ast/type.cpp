//
// Created by Mike Smith on 2021/2/6.
//

#include <charconv>
#include <vector>

#include <core/logging.h>
#include <core/hash.h>
#include <core/macro.h>
#include <ast/type.h>
#include <ast/type_registry.h>

namespace luisa::compute {

const Type *Type::from(std::string_view description) noexcept {

    static constexpr const Type *(*from_desc_impl)(std::string_view &) = [](std::string_view &s) noexcept -> const Type * {
        Type info;
        TypeData data;
        auto s_copy = s;

        using namespace std::string_view_literals;
        auto read_identifier = [&s] {
            auto p = s.cbegin();
            if (p == s.cend() || (*p != '_' && !std::isalpha(*p))) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Failed to parse identifier from '{}'.", s);
            }
            for (; p != s.cend() && (std::isalpha(*p) || std::isdigit(*p) || *p == '_'); p++) {}
            if (p == s.cbegin()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Failed to parse identifier from '{}'.", s);
            }
            auto identifier = s.substr(0, p - s.cbegin());
            s = s.substr(p - s.cbegin());
            return identifier;
        };

        auto read_number = [&s] {
            size_t number;
            auto result = std::from_chars(s.data(), s.data() + s.size(), number);
            if (result.ec != std::errc{}) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Failed to parse number from '{}'.", s);
            }
            s = s.substr(result.ptr - s.data());
            return number;
        };

        auto match = [&s](char c) {
            if (!s.starts_with(c)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Expected '{}' from '{}'.", c, s); }
            s = s.substr(1);
        };

        auto type_identifier = read_identifier();

#define TRY_PARSE_SCALAR_TYPE(T, TAG) \
    if (type_identifier == #T##sv) {  \
        info._tag = Tag::TAG;         \
        info._size = sizeof(T);       \
        info._alignment = alignof(T); \
    } else
        TRY_PARSE_SCALAR_TYPE(bool, BOOL)
        TRY_PARSE_SCALAR_TYPE(float, FLOAT)
        TRY_PARSE_SCALAR_TYPE(int, INT)
        TRY_PARSE_SCALAR_TYPE(uint, UINT)
#undef TRY_PARSE_SCALAR_TYPE

        if (type_identifier == "vector"sv) {
            info._tag = Tag::VECTOR;
            match('<');
            data.members.emplace_back(from_desc_impl(s));
            match(',');
            info._dimension = read_number();
            match('>');
            auto elem = data.members.front();
            if (!elem->is_scalar()) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Invalid vector element: {}.", elem->description()); }
            if (info._dimension != 2 && info._dimension != 3 && info._dimension != 4) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Invalid vector dimension: {}.", info.dimension());
            }
            info._size = info._alignment = elem->size() * (info._dimension == 3 ? 4 : info._dimension);
        } else if (type_identifier == "matrix"sv) {
            info._tag = Tag::MATRIX;
            match('<');
            data.members.emplace_back(Type::of<float>());
            info._dimension = read_number();
            match('>');
            if (info._dimension == 2) {
                info._size = sizeof(float2x2);
                info._alignment = alignof(float2x2);
            } else if (info._dimension == 3) {
                info._size = sizeof(float3x3);
                info._alignment = alignof(float3x3);
            } else if (info._dimension == 4) {
                info._size = sizeof(float4x4);
                info._alignment = alignof(float4x4);
            } else [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Invalid matrix dimension: {}.", info._dimension);
            }
        } else if (type_identifier == "array"sv) {
            info._tag = Tag::ARRAY;
            match('<');
            data.members.emplace_back(from_desc_impl(s));
            if (data.members.back()->is_buffer() || data.members.back()->is_texture()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Arrays are not allowed to hold buffers or images.");
            }
            match(',');
            info._dimension = read_number();
            match('>');
            info._alignment = data.members.front()->alignment();
            info._size = data.members.front()->size() * info._dimension;
        } else if (type_identifier == "struct"sv) {
            info._tag = Tag::STRUCTURE;
            match('<');
            info._alignment = read_number();
            while (s.starts_with(',')) {
                s = s.substr(1);
                data.members.emplace_back(from_desc_impl(s));
            }
            match('>');
            info._size = 0u;
            auto max_member_alignment = static_cast<size_t>(0u);
            for (auto member : data.members) {
                if (member->is_buffer() || member->is_texture()) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION(
                        "Structures are not allowed to have buffers or images as members.");
                }
                auto ma = member->alignment();
                max_member_alignment = std::max(ma, max_member_alignment);
                info._size = (info._size + ma - 1u) / ma * ma + member->size();
            }
            if (auto a = info._alignment; a > 16u || std::bit_floor(a) != a) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Invalid structure alignment {}.", a);
            } else if (a < max_member_alignment && a != 0u) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Struct alignment {} is smaller than the largest member alignment {}.",
                    info._alignment, max_member_alignment);
            }
            info._size = (info._size + info._alignment - 1u) / info._alignment * info._alignment;
        } else if (type_identifier == "buffer"sv) {
            info._tag = Tag::BUFFER;
            match('<');
            auto m = data.members.emplace_back(from_desc_impl(s));
            if (m->is_buffer() || m->is_texture()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Buffers are not allowed to hold buffers or images.");
            }
            match('>');
            info._alignment = 8u;
            info._size = 8u;
        } else if (type_identifier == "texture"sv) {
            info._tag = Tag::TEXTURE;
            match('<');
            info._dimension = read_number();
            match(',');
            auto m = data.members.emplace_back(from_desc_impl(s));
            if (!m->is_scalar()) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Images can only hold scalars."); }
            match('>');
            info._size = 8u;
            info._alignment = 8u;
        } else if (type_identifier == "heap"sv) {
            info._tag = Tag::HEAP;
            info._size = 8u;
            info._alignment = 8u;
        } else if (type_identifier == "accel"sv) {
            info._tag = Tag::ACCEL;
            info._size = 8u;
            info._alignment = 8u;
        } else [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Unknown type identifier: {}.", type_identifier);
        }

        auto description = s_copy.substr(0, s_copy.size() - s.size());
        auto hash = hash64(description);

        return _registry().with_types(
            [info = std::move(info), data = std::move(data), hash, description](auto &&types) mutable noexcept {
                if (auto iter = std::find_if(
                        types.cbegin(), types.cend(),
                        [hash](auto &&ptr) noexcept { return ptr->hash() == hash; });
                    iter != types.cend()) { return iter->get(); }
                info._hash = hash;
                info._index = types.size();
                data.description = description;
                info._data = luisa::make_unique<TypeData>(std::move(data));
                return types.emplace_back(luisa::make_unique<Type>(std::move(info))).get();
            });
    };

    auto info = from_desc_impl(description);
    if (!description.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Unexpected tokens after parsing type description: {}",
            description);
    }
    return info;
}

std::string_view Type::description() const noexcept { return _data->description; }

std::span<const Type *const> Type::members() const noexcept {
    assert(is_structure());
    return _data->members;
}

const Type *Type::element() const noexcept {
    assert(is_array() || is_vector() || is_matrix() || is_buffer() || is_texture());
    return _data->members.front();
}

const Type *Type::at(uint32_t uid) noexcept {
    return _registry().with_types([uid](auto &&types) {
        if (uid >= types.size()) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Invalid type uid {}.", uid); }
        return types[uid].get();
    });
}

TypeRegistry &Type::_registry() noexcept {
    static TypeRegistry r;
    return r;
}

size_t Type::count() noexcept {
    return _registry().with_types([](auto &&types) noexcept {
        return types.size();
    });
}

void Type::traverse(TypeVisitor &visitor) noexcept {
    _registry().with_types([&visitor](auto &&types) noexcept {
        for (auto &&t : types) { visitor.visit(t.get()); }
    });
}

}// namespace luisa::compute
