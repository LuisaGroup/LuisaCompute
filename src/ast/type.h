//
// Created by Mike Smith on 2021/2/6.
//

#pragma once
#include <cassert>
#include <string>
#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <sstream>
#include <atomic>
#include <mutex>

#include <fmt/format.h>

#include <core/macro_map.h>
#include <core/data_types.h>
#include <core/concepts.h>

#include <runtime/buffer.h>
#include <runtime/texture.h>
#include "interface/itype.h"

namespace luisa::compute {

class Type : public IType{

public:

    
    [[nodiscard]] static constexpr std::string_view tag_name(Tag tag) noexcept {
        using namespace std::string_view_literals;
        if (tag == Tag::BOOL) { return "bool"sv; }
        if (tag == Tag::FLOAT) { return "float"sv; }
        if (tag == Tag::INT8) { return "char"sv; }
        if (tag == Tag::UINT8) { return "uchar"sv; }
        if (tag == Tag::INT16) { return "short"sv; }
        if (tag == Tag::UINT16) { return "ushort"sv; }
        if (tag == Tag::INT32) { return "int"sv; }
        if (tag == Tag::UINT32) { return "uint"sv; }
        if (tag == Tag::VECTOR) { return "vector"sv; }
        if (tag == Tag::MATRIX) { return "matrix"sv; }
        if (tag == Tag::ARRAY) { return "array"sv; }
        if (tag == Tag::ATOMIC) { return "atomic"sv; }
        if (tag == Tag::STRUCTURE) { return "struct"sv; }
        if (tag == Tag::BUFFER) { return "buffer"sv; }
        return "unknown"sv;
    }

private:
    uint64_t _hash;
    size_t _size;
    size_t _index;
    size_t _alignment;
    uint32_t _element_count;
    Tag _tag;
    bool _readonly;
    std::string _description;
    std::vector<const Type *> _members;

    [[nodiscard]] static std::mutex &_register_mutex() noexcept;
    [[nodiscard]] static std::vector<std::unique_ptr<Type>> &_registered_types() noexcept;
    [[nodiscard]] static const Type *_from_description_impl(std::string_view &s) noexcept;

public:
    template<typename T>
    [[nodiscard]] static const Type *of() noexcept;

    template<typename T>
    [[nodiscard]] static auto of(T &&) noexcept { return of<std::remove_cvref_t<T>>(); }

    [[nodiscard]] static const Type *from(std::string_view description) noexcept;

    [[nodiscard]] bool operator==(const Type &rhs) const noexcept { return _hash == rhs._hash; }
    [[nodiscard]] bool operator!=(const Type &rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] bool operator<(const Type &rhs) const noexcept { return _index < rhs._index; }

    [[nodiscard]] constexpr auto hash() const noexcept { return _hash; }
    [[nodiscard]] constexpr auto index() const noexcept { return _index; }
    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto alignment() const noexcept { return _alignment; }
    [[nodiscard]] constexpr auto tag() const noexcept { return _tag; }
    [[nodiscard]] constexpr auto readonly() const noexcept { return _readonly; }
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
        return static_cast<uint16_t>(_tag) >= static_cast<uint16_t>(Tag::BOOL) && static_cast<uint16_t>(_tag) <= static_cast<uint16_t>(Tag::UINT32);
    }

    [[nodiscard]] constexpr bool is_array() const noexcept { return _tag == Tag::ARRAY; }
    [[nodiscard]] constexpr bool is_vector() const noexcept { return _tag == Tag::VECTOR; }
    [[nodiscard]] constexpr bool is_matrix() const noexcept { return _tag == Tag::MATRIX; }
    [[nodiscard]] constexpr bool is_structure() const noexcept { return _tag == Tag::STRUCTURE; }
    [[nodiscard]] constexpr bool is_atomic() const noexcept { return _tag == Tag::ATOMIC; }
    [[nodiscard]] constexpr bool is_buffer() const noexcept { return _tag == Tag::BUFFER; }

    template<typename F, std::enable_if_t<std::is_invocable_v<F, const Type *>, int> = 0>
    static void for_each(F &&f) noexcept {
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
#define LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(S, tag) \
    template<>                                                        \
    struct TypeDesc<S> {                                              \
        static constexpr std::string_view description() noexcept {    \
            using namespace std::string_view_literals;                \
            return #S##sv;                                            \
        }                                                             \
    };                                                                \
    template<>                                                        \
    struct TypeDesc<Vector<S, 2>> {                                   \
        static constexpr std::string_view description() noexcept {    \
            using namespace std::string_view_literals;                \
            return "vector<" #S ",2>"sv;                              \
        }                                                             \
    };                                                                \
    template<>                                                        \
    struct TypeDesc<Vector<S, 3>> {                                   \
        static constexpr std::string_view description() noexcept {    \
            using namespace std::string_view_literals;                \
            return "vector<" #S ",3>"sv;                              \
        }                                                             \
    };                                                                \
    template<>                                                        \
    struct TypeDesc<Vector<S, 4>> {                                   \
        static constexpr std::string_view description() noexcept {    \
            using namespace std::string_view_literals;                \
            return "vector<" #S ",4>"sv;                              \
        }                                                             \
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

// matrices
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

// buffers
template<typename T>
struct TypeDesc<Buffer<T>> {
    static std::string_view description() noexcept {
        static auto s = fmt::format(FMT_STRING("buffer<{}>"), TypeDesc<T>::description());
        return s;
    }
};

template<typename T>
struct TypeDesc<BufferView<T>> {
    static std::string_view description() noexcept {
        static auto s = fmt::format(FMT_STRING("buffer<{}>"), TypeDesc<T>::description());
        return s;
    }
};

}// namespace detail

// struct
#define LUISA_STRUCTURE_MAP_MEMBER_TO_DESC(m) TypeDesc<decltype(std::declval<This>().m)>::description()
#define LUISA_STRUCTURE_MAP_MEMBER_TO_FMT(m) ",{}"

#define LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, ...)                                            \
    namespace luisa::compute::detail {                                                                   \
    static_assert(std::is_standard_layout_v<S>);                                                         \
    template<>                                                                                           \
    struct TypeDesc<S> {                                                                                 \
        using This = S;                                                                                  \
        static std::string_view description() noexcept {                                                 \
            static auto s = fmt::format(                                                                 \
                FMT_STRING("struct<{}" LUISA_MAP(LUISA_STRUCTURE_MAP_MEMBER_TO_FMT, ##__VA_ARGS__) ">"), \
                alignof(S),                                                                              \
                LUISA_MAP_LIST(LUISA_STRUCTURE_MAP_MEMBER_TO_DESC, ##__VA_ARGS__));                      \
            return s;                                                                                    \
        }                                                                                                \
    };                                                                                                   \
    }

template<typename T>
const Type *Type::of() noexcept {
    static thread_local auto info = Type::from(detail::TypeDesc<T>::description());
    return info;
}

}// namespace luisa::compute

#define LUISA_STRUCT(...) LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(__VA_ARGS__)
