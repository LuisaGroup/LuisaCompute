//
// Created by Mike Smith on 2021/2/22.
//

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <tuple>
#include <sstream>

#include <core/macro.h>
#include <core/stl.h>
#include <ast/type.h>

namespace luisa::compute {

template<typename T>
class Buffer;

template<typename T>
class BufferView;

template<typename T>
class Image;

template<typename T>
class ImageView;

template<typename T>
class Volume;

template<typename T>
class VolumeView;

class BindlessArray;
class Accel;

namespace detail {

class TypeRegistry {

private:
    struct TypePtrHash {
        [[nodiscard]] auto operator()(const Type *type) const noexcept { return type->hash(); }
        [[nodiscard]] auto operator()(uint64_t hash) const noexcept { return hash; }
    };
    struct TypePtrEqual {
        template<typename Lhs, typename Rhs>
        [[nodiscard]] auto operator()(Lhs &&lhs, Rhs &&rhs) const noexcept {
            constexpr TypePtrHash hash;
            return hash(std::forward<Lhs>(lhs)) == hash(std::forward<Rhs>(rhs));
        }
    };

private:
    luisa::vector<luisa::unique_ptr<Type>> _types;
    luisa::unordered_set<Type *, TypePtrHash, TypePtrEqual> _type_set;
    mutable std::recursive_mutex _mutex;

private:
    [[nodiscard]] static uint64_t _hash(std::string_view desc) noexcept;
    [[nodiscard]] const Type *_decode(std::string_view desc) noexcept;

public:
    [[nodiscard]] static TypeRegistry &instance() noexcept;
    [[nodiscard]] const Type *type_from(luisa::string_view desc) noexcept;
    [[nodiscard]] const Type *type_from(uint64_t hash) noexcept;
    [[nodiscard]] const Type *type_at(size_t i) const noexcept;
    [[nodiscard]] size_t type_count() const noexcept;
    void traverse(TypeVisitor &visitor) const noexcept;
};

template<typename T>
struct TypeDesc {
    static_assert(always_false_v<T>, "Invalid type.");
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
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(int, INT32)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uint, UINT32)

#undef LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION

// array
template<typename T, size_t N>
struct TypeDesc<std::array<T, N>> {
    static_assert(alignof(T) >= 4u);
    static std::string_view description() noexcept {
        static thread_local auto s = luisa::format(
            FMT_STRING("array<{},{}>"),
            TypeDesc<T>::description(), N);
        return s;
    }
};

template<typename T, size_t N>
struct TypeDesc<T[N]> {
    static std::string_view description() noexcept {
        static thread_local auto s = luisa::format(
            FMT_STRING("array<{},{}>"),
            TypeDesc<T>::description(), N);
        return s;
    }
};

template<typename T>
struct TypeDesc<Buffer<T>> {
    static std::string_view description() noexcept {
        static thread_local auto s = luisa::format(
            FMT_STRING("buffer<{}>"),
            TypeDesc<T>::description());
        return s;
    }
};

template<typename T>
struct TypeDesc<BufferView<T>> : TypeDesc<Buffer<T>> {};

template<typename T>
struct TypeDesc<Image<T>> {
    static std::string_view description() noexcept {
        static thread_local auto s = luisa::format(
            FMT_STRING("texture<2,{}>"),
            TypeDesc<T>::description());
        return s;
    }
};

template<typename T>
struct TypeDesc<ImageView<T>> : TypeDesc<Image<T>> {};

template<typename T>
struct TypeDesc<Volume<T>> {
    static std::string_view description() noexcept {
        static thread_local auto s = luisa::format(
            FMT_STRING("texture<3,{}>"),
            TypeDesc<T>::description());
        return s;
    }
};

template<>
struct TypeDesc<BindlessArray> {
    static constexpr std::string_view description() noexcept {
        return "bindless_array";
    }
};

template<>
struct TypeDesc<Accel> {
    static constexpr std::string_view description() noexcept {
        return "accel";
    }
};

template<typename T>
struct TypeDesc<VolumeView<T>> : TypeDesc<Volume<T>> {};

// matrices
template<>
struct TypeDesc<float2x2> {
    static constexpr std::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<2>"sv;
    }
};

template<>
struct TypeDesc<float3x3> {
    static constexpr std::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<3>"sv;
    }
};

template<>
struct TypeDesc<float4x4> {
    static constexpr std::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<4>"sv;
    }
};

template<typename... T>
struct TypeDesc<std::tuple<T...>> {
    static std::string_view description() noexcept {
        static thread_local auto s = [] {
            auto s = luisa::format("struct<{}", alignof(std::tuple<T...>));
            (s.append(",").append(TypeDesc<T>::description()), ...);
            s.append(">");
            return s;
        }();
        return s;
    }
};

}// namespace detail

template<typename T>
const Type *Type::of() noexcept {
    static thread_local auto info = Type::from(
        detail::TypeDesc<std::remove_cvref_t<T>>::description());
    return info;
}

namespace detail {

template<typename S, typename Members, typename offsets>
struct is_valid_reflection : std::false_type {};

template<typename S, typename... M, typename O, O... os>
struct is_valid_reflection<S, std::tuple<M...>, std::integer_sequence<O, os...>> {

    static_assert(((!is_struct_v<M> || alignof(M) >= 4u) && ...));
    static_assert((!is_bool_vector_v<M> && ...),
                  "Boolean vectors are not allowed in DSL "
                  "structures since their may have different "
                  "layouts on different platforms.");

private:
    [[nodiscard]] constexpr static auto _check() noexcept {
        constexpr auto count = sizeof...(M);
        static_assert(sizeof...(os) == count);
        constexpr std::array<size_t, count> sizes{sizeof(M)...};
        constexpr std::array<size_t, count> alignments{alignof(M)...};
        constexpr std::array<size_t, count> offsets{os...};
        auto current_offset = 0u;
        for (auto i = 0u; i < count; i++) {
            auto offset = offsets[i];
            auto size = sizes[i];
            auto alignment = alignments[i];
            current_offset = (current_offset + alignment - 1u) /
                             alignment *
                             alignment;
            if (current_offset != offset) { return false; }
            current_offset += size;
        }
        constexpr auto struct_size = sizeof(S);
        constexpr auto struct_alignment = alignof(S);
        current_offset = (current_offset + struct_alignment - 1u) /
                         struct_alignment *
                         struct_alignment;
        return current_offset == struct_size;
    };

public:
    static constexpr auto value = _check();
};

template<typename S, typename M, typename O>
constexpr auto is_valid_reflection_v = is_valid_reflection<S, M, O>::value;

}// namespace detail

}// namespace luisa::compute

// struct
#define LUISA_STRUCTURE_MAP_MEMBER_TO_DESC(m) \
    TypeDesc<std::remove_cvref_t<decltype(std::declval<this_type>().m)>>::description()
#define LUISA_STRUCTURE_MAP_MEMBER_TO_FMT(m) \
    ",{}"
#define LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE(m) \
    std::remove_cvref_t<decltype(std::declval<this_type>().m)>

#ifdef _MSC_VER// force the built-in offsetof(), otherwise clangd would complain that it's not constant
#define LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET(m) \
    __builtin_offsetof(this_type, m)
#else
#define LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET(m) \
    offsetof(this_type, m)
#endif

#define LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, ...) \
    namespace luisa::compute {                                \
    template<>                                                \
    struct is_struct<S> : std::true_type {};                  \
    template<>                                                \
    struct struct_member_tuple<S> {                           \
        using this_type = S;                                  \
        using type = std::tuple<                              \
            LUISA_MAP_LIST(                                   \
                LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE,           \
                ##__VA_ARGS__)>;                              \
        using offset = std::integer_sequence<                 \
            size_t,                                           \
            LUISA_MAP_LIST(                                   \
                LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET,         \
                ##__VA_ARGS__)>;                              \
        static_assert(detail::is_valid_reflection_v<          \
                      this_type, type, offset>);              \
    };                                                        \
    namespace detail {                                        \
    template<>                                                \
    struct TypeDesc<S> {                                      \
        using this_type = S;                                  \
        static std::string_view description() noexcept {      \
            static auto s = fmt::format(                      \
                FMT_STRING("struct<{}" LUISA_MAP(             \
                    LUISA_STRUCTURE_MAP_MEMBER_TO_FMT,        \
                    ##__VA_ARGS__) ">"),                      \
                alignof(S),                                   \
                LUISA_MAP_LIST(                               \
                    LUISA_STRUCTURE_MAP_MEMBER_TO_DESC,       \
                    ##__VA_ARGS__));                          \
            return s;                                         \
        }                                                     \
    };                                                        \
    }                                                         \
    }
#define LUISA_STRUCT_REFLECT(S, ...) \
    LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, __VA_ARGS__)
