//
// Created by Mike Smith on 2021/2/6.
//

#include <charconv>
#include <mutex>
#include <string>

#include <fmt/format.h>

#include <core/logging.h>
#include <core/hash.h>
#include <core/type_info.h>

namespace luisa {

std::mutex TypeInfo::_register_mutex;
std::vector<std::unique_ptr<TypeInfo>> TypeInfo::_registered_types;

const TypeInfo *TypeInfo::from(std::string_view description) noexcept {
    auto info = _from_description_impl(description);
    assert(description.empty());
    return info;
}

const TypeInfo *TypeInfo::_from_description_impl(std::string_view &s) noexcept {
    
    TypeInfo info;
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
        auto[p, ec] = std::from_chars(s.cbegin(), s.cend(), number);
        if (ec != std::errc{}) {
            LUISA_ERROR_WITH_LOCATION("Failed to parse number from '{}'.", s);
        }
        s = s.substr(p - s.cbegin());
        return number;
    };
    
    auto match = [&s](char c) {
        if (!s.starts_with(c)) { LUISA_ERROR_WITH_LOCATION("Expected '{}' from '{}'.", c, s); }
        s = s.substr(1);
    };
    
    auto type_identifier = read_identifier();

#define TRY_PARSE_SCALAR_TYPE(T, TAG)                         \
    if (type_identifier == #T ## sv) {                        \
        info._tag = TypeTag::TAG;                             \
        info._size = sizeof(T);                               \
        info._alignment = alignof(T);                         \
    }
    TRY_PARSE_SCALAR_TYPE(bool, BOOL)
    else TRY_PARSE_SCALAR_TYPE(float, FLOAT)
    else TRY_PARSE_SCALAR_TYPE(char, INT8)
    else TRY_PARSE_SCALAR_TYPE(uchar, UINT8)
    else TRY_PARSE_SCALAR_TYPE(short, INT16)
    else TRY_PARSE_SCALAR_TYPE(ushort, UINT16)
    else TRY_PARSE_SCALAR_TYPE(int, INT32)
    else TRY_PARSE_SCALAR_TYPE(uint, UINT32)
    else if (type_identifier == "atomic"sv) {
        info._tag = TypeTag::ATOMIC;
        match('<');
        info._members.emplace_back(_from_description_impl(s));
        match('>');
        info._alignment = info._members.front()->alignment();
        info._size = info._members.front()->size();
    } else if (type_identifier == "vector"sv) {
        info._tag = TypeTag::VECTOR;
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
        info._tag = TypeTag::MATRIX;
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
        info._tag = TypeTag::ARRAY;
        match('<');
        info._members.emplace_back(_from_description_impl(s));
        match(',');
        info._element_count = read_number();
        match('>');
        info._alignment = info._members.front()->alignment();
        info._size = info._members.front()->size() * info._element_count;
    } else if (type_identifier == "struct"sv) {
        info._tag = TypeTag::STRUCTURE;
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
    }
    
    auto description = s_copy.substr(0, s_copy.size() - s.size());
    auto hash = xxh3_hash64(description.data(), description.size());
    
    std::scoped_lock lock{_register_mutex};
    if (auto iter = std::find_if(_registered_types.cbegin(), _registered_types.cend(), [hash](auto &&ptr) noexcept {
            return ptr->hash() == hash;
        }); iter != _registered_types.cend()) { return iter->get(); }
    
    info._hash = hash;
    info._index = _registered_types.size();
    info._description = description;
    return _registered_types.emplace_back(std::make_unique<TypeInfo>(info)).get();
}

}
