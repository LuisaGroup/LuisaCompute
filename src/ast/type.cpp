//
// Created by Mike Smith on 2021/2/6.
//

#include <charconv>
#include <vector>

#include <fmt/format.h>

#include <core/logging.h>
#include <core/hash.h>
#include <core/macro.h>
#include <ast/type.h>
#include <ast/type_registry.h>

namespace luisa::compute {

struct TypeData {
    std::string description;
    std::vector<const Type *> members;
};

const Type *Type::from(std::string_view description) noexcept {

    static constexpr const Type *(*from_desc_impl)(std::string_view &) = [](std::string_view &s) noexcept -> const Type * {
        Type info;
        TypeData data;
        auto s_copy = s;

        using namespace std::string_view_literals;
        auto read_identifier = [&s] {
            auto p = s.cbegin();
            if (p == s.cend() || (*p != '_' && !std::isalpha(*p))) {
                LUISA_ERROR_WITH_LOCATION("Failed to parse identifier from '{}'.", s);
            }
            for (; p != s.cend() && (std::isalpha(*p) || std::isdigit(*p) || *p == '_'); p++) {}
            if (p == s.cbegin()) {
                LUISA_ERROR_WITH_LOCATION("Failed to parse identifier from '{}'.", s);
            }
            auto identifier = s.substr(0, p - s.cbegin());
            s = s.substr(p - s.cbegin());
            return identifier;
        };

        auto read_number = [&s] {
            size_t number;
            auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), number);
            if (ec != std::errc{}) {
                LUISA_ERROR_WITH_LOCATION("Failed to parse number from '{}'.", s);
            }
            s = s.substr(p - s.data());
            return number;
        };

        auto match = [&s](char c) {
            if (!s.starts_with(c)) { LUISA_ERROR_WITH_LOCATION("Expected '{}' from '{}'.", c, s); }
            s = s.substr(1);
        };

        auto type_identifier = read_identifier();

#define TRY_PARSE_SCALAR_TYPE_CASE(T, TAG) \
    if (type_identifier == #T##sv) {       \
        info._tag = Tag::TAG;              \
        info._size = sizeof(T);            \
        info._alignment = alignof(T);      \
    } else

#define TRY_PARSE_SCALAR_TYPE(CASE) TRY_PARSE_SCALAR_TYPE_CASE CASE

        LUISA_MAP(TRY_PARSE_SCALAR_TYPE,
                  (bool, BOOL),
                  (float, FLOAT),
                  (char, INT8),
                  (uchar, UINT8),
                  (short, INT16),
                  (int, INT32),
                  (uint, UINT32))

#undef TRY_PARSE_SCALAR_TYPE
#undef TRY_PARSE_SCALAR_TYPE_CASE

        if (type_identifier == "atomic"sv) {
            info._tag = Tag::ATOMIC;
            match('<');
            data.members.emplace_back(from_desc_impl(s));
            match('>');
            info._alignment = data.members.front()->alignment();
            info._size = data.members.front()->size();
        } else if (type_identifier == "vector"sv) {
            info._tag = Tag::VECTOR;
            match('<');
            data.members.emplace_back(from_desc_impl(s));
            match(',');
            info._element_count = read_number();
            match('>');
            auto elem = data.members.front();
            if (!elem->is_scalar()) { LUISA_ERROR_WITH_LOCATION("Invalid vector element: {}.", elem->description()); }
            if (info._element_count != 2 && info._element_count != 3 && info._element_count != 4) {
                LUISA_ERROR_WITH_LOCATION("Invalid vector dimension: {}.", info.dimension());
            }
            info._size = info._alignment = elem->size() * (info._element_count == 3 ? 4 : info._element_count);
        } else if (type_identifier == "matrix"sv) {
            info._tag = Tag::MATRIX;
            match('<');
            data.members.emplace_back(Type::of<float>());
            info._element_count = read_number();
            match('>');
            if (info._element_count == 3) {
                info._size = sizeof(float3x3);
                info._alignment = alignof(float3x3);
            } else if (info._element_count == 4) {
                info._size = sizeof(float4x4);
                info._alignment = alignof(float4x4);
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid matrix dimension: {}.", info._element_count);
            }
        } else if (type_identifier == "array"sv) {
            info._tag = Tag::ARRAY;
            match('<');
            data.members.emplace_back(from_desc_impl(s));
            match(',');
            info._element_count = read_number();
            match('>');
            info._alignment = data.members.front()->alignment();
            info._size = data.members.front()->size() * info._element_count;
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
            for (auto member : data.members) {
                auto ma = member->alignment();
                info._size = (info._size + ma - 1u) / ma * ma + member->size();
            }
            info._size = (info._size + info._alignment - 1u) / info._alignment * info._alignment;
        } else {
            LUISA_ERROR_WITH_LOCATION("Unknown type identifier: {}.", type_identifier);
        }

        auto description = s_copy.substr(0, s_copy.size() - s.size());
        auto hash = xxh3_hash64(description.data(), description.size());

        return _registry().with_types(
            [info = std::move(info), data = std::move(data), hash, description](auto &&types) mutable noexcept {
                if (auto iter = std::find_if(
                        types.cbegin(), types.cend(),
                        [hash](auto &&ptr) noexcept { return ptr->hash() == hash; });
                    iter != types.cend()) { return iter->get(); }
                info._hash = hash;
                info._index = types.size();
                data.description = description;
                info._data = std::make_unique<TypeData>(std::move(data));
                return types.emplace_back(std::make_unique<Type>(std::move(info))).get();
            });
    };

    auto info = from_desc_impl(description);
    assert(description.empty());
    return info;
}

std::string_view Type::description() const noexcept { return _data->description; }

std::span<const Type *const> Type::members() const noexcept {
    assert(is_structure());
    return _data->members;
}

const Type *Type::element() const noexcept {
    assert(is_array() || is_atomic() || is_vector() || is_matrix());
    return _data->members.front();
}

const Type *Type::at(uint32_t uid) noexcept {
    return _registry().with_types([uid](auto &&types) {
        if (uid >= types.size()) { LUISA_ERROR_WITH_LOCATION("Invalid type uid {}.", uid); }
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

}// namespace luisa::compute
