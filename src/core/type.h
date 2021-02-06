//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <cassert>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>

#include <core/macro_map.h>

namespace luisa {

enum struct TypeTag : uint16_t {
    
    UNKNOWN,
    
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

class TypeInfo {

private:
    uint32_t _hash;
    uint32_t _size;
    uint16_t _index;
    uint16_t _alignment;
    TypeTag _tag;
    uint16_t _count;
    std::vector<const TypeInfo *> _members;
    
    static std::unordered_map<std::string, std::unique_ptr<TypeInfo>> _description_to_info;
    [[nodiscard]] std::string _compute_description_string() const noexcept;
    [[nodiscard]] uint32_t _compute_hash() const noexcept;

public:
    [[nodiscard]] constexpr auto hash() const noexcept { return _hash; }
    [[nodiscard]] constexpr auto index() const noexcept { return _index; }
    [[nodiscard]] constexpr size_t size() const noexcept { return _size; }
    [[nodiscard]] constexpr size_t alignment() const noexcept { return _alignment; }
    [[nodiscard]] constexpr TypeTag tag() const noexcept { return _tag; }
    
    [[nodiscard]] constexpr size_t element_count() const noexcept {
        assert(is_array() || is_vector() || is_matrix());
        return _count;
    }
    
    [[nodiscard]] constexpr const auto &members() const noexcept {
        assert(is_structure());
        return _members;
    }
    
    [[nodiscard]] constexpr auto element() const noexcept {
        assert(is_array() || is_atomic() || is_vector() || is_matrix());
        return _members.front();
    }
    
    [[nodiscard]] constexpr auto is_unknown() const noexcept { return _tag == TypeTag::UNKNOWN; }
    
    [[nodiscard]] constexpr auto is_scalar() const noexcept {
        return static_cast<uint16_t>(_tag) >= static_cast<uint16_t>(TypeTag::BOOL) &&
               static_cast<uint16_t>(_tag) <= static_cast<uint16_t>(TypeTag::UINT32);
    }
    
    [[nodiscard]] constexpr auto is_array() const noexcept { return _tag == TypeTag::ARRAY; }
    [[nodiscard]] constexpr auto is_vector() const noexcept { return _tag == TypeTag::VECTOR; }
    [[nodiscard]] constexpr auto is_matrix() const noexcept { return _tag == TypeTag::MATRIX; }
    [[nodiscard]] constexpr auto is_structure() const noexcept { return _tag == TypeTag::STRUCTURE; }
    [[nodiscard]] constexpr auto is_atomic() const noexcept { return _tag == TypeTag::ATOMIC; }
};

template<typename T>
struct Type {
    
    template<typename U>
    static constexpr auto always_false = false;
    
    static_assert(always_false<T>, "Unknown structure type.");
};

#define LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(S, tag)  \
template<>                                             \
struct Type<S> {                                       \
    using T = S;                                       \
    static constexpr auto alignment = alignof(S);      \
    static constexpr auto size = sizeof(S);            \
    static constexpr auto tag = TypeTag::tag;          \
};

LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(bool, BOOL)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(float, FLOAT)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(int8_t, INT8)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(uint8_t, UINT8)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(int16_t, INT16)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(uint16_t, UINT16)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(int32_t, INT32)
LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION(uint32_t, UINT32)

#undef LUISA_MAKE_SCALAR_TYPE_SPECIALIZATION

#define LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE(m) typename Type<decltype(std::declval<This>().m)>::T

#define LUISA_MAKE_STRUCTURE_TYPE_SPECIALIZATION(S, ...)                            \
namespace luisa {                                                                   \
    static_assert(std::is_standard_layout_v<S>);                                    \
    template<>                                                                      \
    struct Type<S> {                                                                \
        using This = S;                                                             \
        using Members = std::tuple<                                                 \
            LUISA_MAP_LIST(LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE, __VA_ARGS__)>;       \
        static constexpr auto alignment = alignof(S);                               \
        static constexpr auto size = sizeof(S);                                     \
        static constexpr auto tag = TypeTag::STRUCTURE;                             \
        using T = Type<S>;                                                          \
    };                                                                              \
}

template<typename T>
[[nodiscard]] inline const TypeInfo *type_info() noexcept;

}
