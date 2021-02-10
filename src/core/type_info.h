//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <cassert>
#include <string>
#include <array>
#include <vector>
#include <unordered_set>
#include <string>
#include <string_view>
#include <sstream>
#include <atomic>
#include <mutex>

#include <fmt/format.h>
#include <core/macro_map.h>
#include <core/data_types.h>
#include <core/concepts.h>

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
    
    [[nodiscard]] static std::mutex &_register_mutex() noexcept;
    [[nodiscard]] static std::vector<std::unique_ptr<TypeInfo>> &_registered_types() noexcept;
    [[nodiscard]] static const TypeInfo *_from_description_impl(std::string_view &s) noexcept;

public:
    template<typename T>
    [[nodiscard]] static const TypeInfo *of() noexcept;
    
    template<typename T>
    [[nodiscard]] static auto of(T &&) noexcept { return of<std::remove_cvref_t<T>>(); }
    
    [[nodiscard]] static const TypeInfo *from(std::string_view description) noexcept;
    
    [[nodiscard]] bool operator==(const TypeInfo &rhs) const noexcept { return _hash == rhs._hash; }
    [[nodiscard]] bool operator!=(const TypeInfo &rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] bool operator<(const TypeInfo &rhs) const noexcept { return _index < rhs._index; }
    
    [[nodiscard]] constexpr auto hash() const noexcept { return _hash; }
    [[nodiscard]] constexpr auto index() const noexcept { return _index; }
    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto alignment() const noexcept { return _alignment; }
    [[nodiscard]] constexpr auto tag() const noexcept { return _tag; }
    [[nodiscard]] std::string_view description() const noexcept { return _description; }
    
    [[nodiscard]] constexpr size_t element_count() const noexcept {
        assert(is_array() || is_vector() || is_matrix());
        return _element_count;
    }
    
    [[nodiscard]] const auto &members() const noexcept {
        assert(is_structure());
        return _members;
    }
    
    [[nodiscard]] auto element() const noexcept {
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
    
    template<typename F, std::enable_if_t<std::is_invocable_v<F, const TypeInfo *>, int> = 0>
    static void traverse(F &&f) noexcept {
        std::scoped_lock lock{_register_mutex()};
        for (auto &&t : _registered_types()) { f(t.get()); }
    }
};

namespace detail {

template<typename T>
struct TypeDesc {
    static_assert(always_false<T>, "Invalid type.");
};

// scalar
#define LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(S, tag)    \
template<>                                                               \
struct TypeDesc<S> {                                                     \
    static constexpr std::string_view description() noexcept {           \
        using namespace std::string_view_literals;                       \
        return #S ## sv;                                                 \
    }                                                                    \
};                                                                       \
template<>                                                               \
struct TypeDesc<Vector<S, 2>> {                                          \
    static constexpr std::string_view description() noexcept {           \
        using namespace std::string_view_literals;                       \
        return "vector<" #S ",2>"sv;                                     \
    }                                                                    \
};                                                                       \
template<>                                                               \
struct TypeDesc<Vector<S, 3>> {                                          \
    static constexpr std::string_view description() noexcept {           \
        using namespace std::string_view_literals;                       \
        return "vector<" #S ",3>"sv;                                     \
    }                                                                    \
};                                                                       \
template<>                                                               \
struct TypeDesc<Vector<S, 4>> {                                          \
    static constexpr std::string_view description() noexcept {           \
        using namespace std::string_view_literals;                       \
        return "vector<" #S ",4>"sv;                                     \
    }                                                                    \
};

LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(bool, BOOL)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(float, FLOAT)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(char, INT8)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uchar, UINT8)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(short, INT16)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(ushort, UINT16)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(int, INT32)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uint, UINT32)

#undef LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION

// array
template<typename T, size_t N>
struct TypeDesc<std::array<T, N>> {
    static std::string_view description() noexcept {
        static auto s = fmt::format(FMT_STRING("array<{},{}>"), TypeDesc<T>::description(), N);
        return s;
    }
};

template<typename T, size_t N>
struct TypeDesc<T[N]> {
    static std::string_view description() noexcept {
        static auto s = fmt::format(FMT_STRING("array<{},{}>"), TypeDesc<T>::description(), N);
        return s;
    }
};

// atomics
template<>
struct TypeDesc<std::atomic<int>> {
    static constexpr std::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "atomic<int>";
    }
};

template<>
struct TypeDesc<std::atomic<uint>> {
    static constexpr std::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "atomic<uint>";
    }
};

template<>
struct TypeDesc<float3x3> {
    static constexpr std::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<3>";
    }
};

template<>
struct TypeDesc<float4x4> {
    static constexpr std::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<4>";
    }
};

}

// struct
#define LUISA_STRUCTURE_MAP_MEMBER_TO_DESC(m) TypeDesc<decltype(std::declval<This>().m)>::description()
#define LUISA_STRUCTURE_MAP_MEMBER_TO_FMT(m) ",{}"

#define LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, ...)                                                        \
namespace luisa::detail {                                                                                            \
    static_assert(std::is_standard_layout_v<S>);                                                                     \
    template<>                                                                                                       \
    struct TypeDesc<S> {                                                                                             \
        using This = S;                                                                                              \
        static std::string_view description() noexcept {                                                             \
            static auto s = fmt::format(                                                                             \
                FMT_STRING("struct<{}" LUISA_MAP(LUISA_STRUCTURE_MAP_MEMBER_TO_FMT, ##__VA_ARGS__) ">"),  \
                alignof(S),                                                                                          \
                LUISA_MAP_LIST(LUISA_STRUCTURE_MAP_MEMBER_TO_DESC, ##__VA_ARGS__));                       \
            return s;                                                                                                \
        }                                                                                                            \
    };                                                                                                               \
}

template<typename T>
const TypeInfo *TypeInfo::of() noexcept {
    static thread_local auto info = TypeInfo::from(detail::TypeDesc<T>::description());
    return info;
}

template<typename T>
[[nodiscard]] inline const TypeInfo *type_info() noexcept {
    return TypeInfo::of<T>();
}

}

#define LUISA_STRUCT(...) LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(__VA_ARGS__)
