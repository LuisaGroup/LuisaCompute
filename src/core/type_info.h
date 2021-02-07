//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <cassert>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>
#include <sstream>

#include <fmt/format.h>
#include <core/macro_map.h>

namespace luisa {

enum struct TypeTag : uint32_t {
    
    BOOL,
    
    FLOAT,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    
    VECTOR,
    MATRIX,
    
    ARRAY,
    
    ATOMIC,
    STRUCTURE
};

[[nodiscard]] constexpr std::string_view type_tag_name(TypeTag tag) noexcept {
    using namespace std::string_view_literals;
    if (tag == TypeTag::BOOL) { return "bool"sv; }
    if (tag == TypeTag::FLOAT) { return "float"sv; }
    if (tag == TypeTag::INT8) { return "char"sv; }
    if (tag == TypeTag::UINT8) { return "uchar"sv; }
    if (tag == TypeTag::INT16) { return "short"sv; }
    if (tag == TypeTag::UINT16) { return "ushort"sv; }
    if (tag == TypeTag::INT32) { return "int"sv; }
    if (tag == TypeTag::UINT32) { return "uint"sv; }
    if (tag == TypeTag::VECTOR) { return "vector"sv; }
    if (tag == TypeTag::MATRIX) { return "matrix"sv; }
    if (tag == TypeTag::ARRAY) { return "array"sv; }
    if (tag == TypeTag::ATOMIC) { return "atomic"sv; }
    if (tag == TypeTag::STRUCTURE) { return "struct"sv; }
    return "unknown"sv;
}

class TypeInfo {

private:
    uint64_t _hash;
    size_t _size;
    size_t _index;
    size_t _alignment;
    TypeTag _tag;
    uint32_t _element_count;
    std::string _description;
    std::vector<const TypeInfo *> _members;
    
    [[nodiscard]] static const TypeInfo *_from_description_impl(std::string_view &s) noexcept;

public:
    template<typename T>
    [[nodiscard]] static const TypeInfo *of() noexcept;
    
    [[nodiscard]] static const TypeInfo *from_description(std::string_view description) noexcept;
    
    [[nodiscard]] bool operator==(const TypeInfo &rhs) const noexcept { return _hash == rhs._hash; }
    [[nodiscard]] bool operator!=(const TypeInfo &rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] bool operator<(const TypeInfo &rhs) const noexcept { return _index < rhs._index; }
    
    [[nodiscard]] constexpr auto hash() const noexcept { return _hash; }
    [[nodiscard]] constexpr auto index() const noexcept { return _index; }
    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto alignment() const noexcept { return _alignment; }
    [[nodiscard]] constexpr auto tag() const noexcept { return _tag; }
    [[nodiscard]] constexpr std::string_view description() const noexcept { return _description; }
    
    [[nodiscard]] constexpr size_t element_count() const noexcept {
        assert(is_array() || is_vector() || is_matrix());
        return _element_count;
    }
    
    [[nodiscard]] constexpr const auto &members() const noexcept {
        assert(is_structure());
        return _members;
    }
    
    [[nodiscard]] constexpr auto element() const noexcept {
        assert(is_array() || is_atomic() || is_vector() || is_matrix());
        return _members.front();
    }
    
    [[nodiscard]] constexpr bool is_scalar() const noexcept {
        return static_cast<uint16_t>(_tag) >= static_cast<uint16_t>(TypeTag::BOOL) &&
               static_cast<uint16_t>(_tag) <= static_cast<uint16_t>(TypeTag::UINT32);
    }
    
    [[nodiscard]] constexpr bool is_array() const noexcept { return _tag == TypeTag::ARRAY; }
    [[nodiscard]] constexpr bool is_vector() const noexcept { return _tag == TypeTag::VECTOR; }
    [[nodiscard]] constexpr bool is_matrix() const noexcept { return _tag == TypeTag::MATRIX; }
    [[nodiscard]] constexpr bool is_structure() const noexcept { return _tag == TypeTag::STRUCTURE; }
    [[nodiscard]] constexpr bool is_atomic() const noexcept { return _tag == TypeTag::ATOMIC; }
};

using uchar = uint8_t;
using ushort = uint16_t;
using uint = uint32_t;

namespace detail {

template<typename T>
struct Type {
    
    template<typename U>
    static constexpr auto always_false = false;
    
    static_assert(always_false<T>, "Unknown structure type.");
};

// scalar
#define LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(S, tag)    \
template<>                                               \
struct Type<S> {                                         \
    using T = S;                                         \
    static constexpr auto alignment = alignof(S);        \
    static constexpr auto size = sizeof(S);              \
    static constexpr auto tag = TypeTag::tag;            \
    static constexpr std::string_view description = #S;  \
};

LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(bool, BOOL)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(float, FLOAT)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(char, INT8)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(uchar, UINT8)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(short, INT16)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(ushort, UINT16)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(int, INT32)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(uint, UINT32)

#undef LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION

// TODO: array, vector, matrix, atomic

}

// struct
#define LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE(m) typename Type<decltype(std::declval<This>().m)>::T
#define LUISA_STRUCTURE_MAP_MEMBER_TO_DESC(m) Type<decltype(std::declval<This>().m)>::description
#define LUISA_STRUCTURE_MAP_MEMBER_TO_FMT(m) ",{}"

#define LUISA_MAKE_STRUCTURE_TYPE_SPECIALIZATION(S, ...)                                 \
namespace luisa::detail {                                                                \
    static_assert(std::is_standard_layout_v<S>);                                         \
    template<>                                                                           \
    struct Type<S> {                                                                     \
        using This = S;                                                                  \
        using Members = std::tuple<                                                      \
            LUISA_MAP_LIST(LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE, __VA_ARGS__)>;            \
        static constexpr auto alignment = alignof(S);                                    \
        static constexpr auto size = sizeof(S);                                          \
        static constexpr auto tag = TypeTag::STRUCTURE;                                  \
        using T = Type<S>;                                                               \
        inline static auto description = fmt::format(                                    \
            "struct<{}" LUISA_MAP(LUISA_STRUCTURE_MAP_MEMBER_TO_FMT, __VA_ARGS__) ">",   \
            alignof(S),                                                                  \
            LUISA_MAP_LIST(LUISA_STRUCTURE_MAP_MEMBER_TO_DESC, __VA_ARGS__));            \
    };                                                                                   \
}

template<typename T>
const TypeInfo *TypeInfo::of() noexcept {
    using Type = detail::Type<T>;
    static thread_local auto info = TypeInfo::from_description(Type::description);
    return info;
}

template<typename T>
[[nodiscard]] inline const TypeInfo *type_info() noexcept {
    return TypeInfo::of<T>();
}

}
