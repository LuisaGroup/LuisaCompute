//
// Created by Mike Smith on 2021/2/6.
//

#include <charconv>
#include <mutex>

#include <fmt/format.h>

#include <core/logging.h>
#include <core/hash.h>
#include <core/type_info.h>

namespace luisa {

const TypeInfo *TypeInfo::from_description(std::string_view description) noexcept {
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
    // TODO: array, vector, matrix, atomic
    else if (type_identifier == "struct"sv) {
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
    
    std::string description{s_copy.substr(0, s_copy.size() - s.size())};
    
    static std::unordered_map<std::string, std::unique_ptr<TypeInfo>> description_to_info;
    
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    auto iter = description_to_info.find(description);
    if (iter == description_to_info.cend()) {
        info._hash = xxh3_hash64(description.data(), description.size());
        info._index = description_to_info.size();
        info._description = description;
        iter = description_to_info.emplace(std::move(description), std::make_unique<TypeInfo>(info)).first;
    }
    return iter->second.get();
}

}
