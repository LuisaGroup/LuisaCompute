//
// Created by Mike Smith on 2021/2/6.
//

#include <charconv>
#include <vector>

#include <fmt/format.h>

#include <core/logging.h>
#include <core/hash.h>
#include <core/macro_map.h>
#include <ast/type.h>
#include <ast/type_registry.h>

namespace luisa::compute {

const Type *Type::from(std::string_view description) noexcept {

    static TypeRegistry registry;
    
    static thread_local Arena arena;
    static thread_local std::vector<const Type *> members;
    
    static constexpr const Type *(*from_desc_impl)(std::string_view &) = [](std::string_view &s) noexcept -> const Type * {

        Type info;
        auto s_copy = s;
        members.clear();

        using namespace std::string_view_literals;

        auto read_identifier = [&s] {
            auto p = s.cbegin();
            for (; p != s.cend() && std::isalpha(*p); p++) {}
            if (p == s.cbegin()) {
                LUISA_ERROR_WITH_LOCATION("Failed to parse type identifier from '{}'.", s);
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
        info._tag = Tag::TAG;             \
        info._size = sizeof(T);           \
        info._alignment = alignof(T);     \
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
            members.emplace_back(from_desc_impl(s));
            match('>');
            info._alignment = members.front()->alignment();
            info._size = members.front()->size();
        } else if (type_identifier == "vector"sv) {
            info._tag = Tag::VECTOR;
            match('<');
            members = {from_desc_impl(s)};
            match(',');
            info._element_count = read_number();
            match('>');
            auto elem = members.front();
            if (!elem->is_scalar()) { LUISA_ERROR_WITH_LOCATION("Invalid vector element: {}.", elem->description()); }
            if (info._element_count != 2 && info._element_count != 3 && info._element_count != 4) {
                LUISA_ERROR_WITH_LOCATION("Invalid vector dimension: {}.", info.element_count());
            }
            info._size = info._alignment = elem->size() * (info._element_count == 3 ? 4 : info._element_count);
        } else if (type_identifier == "matrix"sv) {
            info._tag = Tag::MATRIX;
            match('<');
            members.emplace_back(Type::from("float"));
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
            members.emplace_back(from_desc_impl(s));
            match(',');
            info._element_count = read_number();
            match('>');
            info._alignment = members.front()->alignment();
            info._size = members.front()->size() * info._element_count;
        } else if (type_identifier == "struct"sv) {
            info._tag = Tag::STRUCTURE;
            match('<');
            info._alignment = read_number();
            while (s.starts_with(',')) {
                s = s.substr(1);
                members.emplace_back(from_desc_impl(s));
            }
            match('>');
            info._size = 0u;
            for (auto member : members) {
                auto ma = member->alignment();
                info._size = (info._size + ma - 1u) / ma * ma + member->size();
            }
            info._size = (info._size + info._alignment - 1u) / info._alignment * info._alignment;
        } else if (type_identifier == "buffer"sv) {
            info._tag = Tag::BUFFER;
            match('<');
            members.emplace_back(from_desc_impl(s));
            match('>');
            info._alignment = 8;// same as pointer...
            info._size = 8;
        }

        auto description = s_copy.substr(0, s_copy.size() - s.size());
        auto hash = xxh3_hash64(description.data(), description.size());

        return registry.with_types(
            [&info, hash, description](auto &&types) noexcept {
                if (auto iter = std::find_if(
                        types.cbegin(), types.cend(), [hash](auto &&ptr) noexcept { return ptr->hash() == hash; });
                    iter != types.cend()) { return *iter; }
                info._hash = hash;
                info._index = types.size();
                info._description = ArenaString{arena, description};
                if (!members.empty()) {
                    ArenaVector m{arena, std::span{members}};
                    info._members = {m.data(), m.size()};
                }
                return types.emplace_back(arena.create<Type>(info));
            });
    };

    auto info = from_desc_impl(description);
    assert(description.empty());
    return info;
}

}// namespace luisa::compute
