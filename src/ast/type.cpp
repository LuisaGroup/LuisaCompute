//
// Created by Mike Smith on 2021/2/6.
//

#include <charconv>
#include <mutex>
#include <string>

#include <fmt/format.h>

#include <core/logging.h>
#include <core/hash.h>
#include <ast/type.h>

namespace luisa::compute {

std::mutex &Type::_register_mutex() noexcept {
    static std::mutex register_mutex;
    return register_mutex;
}

std::vector<std::unique_ptr<Type>> &Type::_registered_types() noexcept {
    static std::vector<std::unique_ptr<Type>> registered_types;
    return registered_types;
}

const Type *Type::from(std::string_view description) noexcept {
    auto info = _from_description_impl(description);
    assert(description.empty());
    return info;
}

const Type *Type::_from_description_impl(std::string_view &s) noexcept {

    Type info;
    auto s_copy = s;

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
        info._members.emplace_back(_from_description_impl(s));
        match('>');
        info._alignment = info._members.front()->alignment();
        info._size = info._members.front()->size();
    } else if (type_identifier == "vector"sv) {
        info._tag = Tag::VECTOR;
        match('<');
        info._members.emplace_back(_from_description_impl(s));
        match(',');
        info._element_count = read_number();
        match('>');
        auto elem = info._members.front();
        if (!elem->is_scalar()) { LUISA_ERROR_WITH_LOCATION("Invalid vector element: {}.", elem->description()); }
        if (info._element_count != 2 && info._element_count != 3 && info._element_count != 4) {
            LUISA_ERROR_WITH_LOCATION("Invalid vector dimension: {}.", info.element_count());
        }
        info._size = info._alignment = elem->size() * (info._element_count == 3 ? 4 : info._element_count);
    } else if (type_identifier == "matrix"sv) {
        info._tag = Tag::MATRIX;
        match('<');
        info._members.emplace_back(from("float"));
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
        info._members.emplace_back(_from_description_impl(s));
        match(',');
        info._element_count = read_number();
        match('>');
        info._alignment = info._members.front()->alignment();
        info._size = info._members.front()->size() * info._element_count;
    } else if (type_identifier == "struct"sv) {
        info._tag = Tag::STRUCTURE;
        match('<');
        info._alignment = read_number();
        while (s.starts_with(',')) {
            s = s.substr(1);
            info._members.emplace_back(_from_description_impl(s));
        }
        match('>');
        info._size = 0u;
        for (auto member : info._members) {
            auto ma = member->alignment();
            info._size = (info._size + ma - 1u) / ma * ma + member->size();
        }
        info._size = (info._size + info._alignment - 1u) / info._alignment * info._alignment;
    } else if (type_identifier == "buffer"sv) {
        info._tag = Tag::BUFFER;
        match('<');
        info._members.emplace_back(_from_description_impl(s));
        match('>');
        info._alignment = 8;  // same as pointer...
        info._size = 8;
    }

    auto description = s_copy.substr(0, s_copy.size() - s.size());
    auto hash = xxh3_hash64(description.data(), description.size());

    std::scoped_lock lock{_register_mutex()};
    auto &&types = _registered_types();
    if (auto iter = std::find_if(
            types.cbegin(), types.cend(), [hash](auto &&ptr) noexcept { return ptr->hash() == hash; });
        iter != types.cend()) { return iter->get(); }

    info._hash = hash;
    info._index = types.size();
    info._description = description;
    return types.emplace_back(std::make_unique<Type>(info)).get();
}

}// namespace luisa::compute
