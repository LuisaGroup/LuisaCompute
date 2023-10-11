#pragma once

#include <array>
#include <luisa/core/stl/memory.h>
#include <luisa/core/macro.h>
#include <luisa/ast/type.h>

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
class ByteBuffer;

namespace detail {

// TODO: is it possible to make the following functions constexpr?
[[nodiscard]] LC_AST_API luisa::string make_array_description(luisa::string_view elem, size_t dim) noexcept;
[[nodiscard]] LC_AST_API luisa::string make_struct_description(size_t alignment, std::initializer_list<luisa::string_view> members) noexcept;
[[nodiscard]] LC_AST_API luisa::string make_buffer_description(luisa::string_view elem) noexcept;

template<typename T>
struct TypeDesc;

#include <luisa/ast/member_reflect.inl.h>

template<typename T>
struct TypeDesc {
    [[nodiscard]] static luisa::string_view description() noexcept {
        static thread_local luisa::string desc = member_reflect<T>();
        return desc;
    }
};

// scalar
#define LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(S, desc) \
    template<>                                                              \
    struct TypeDesc<S> {                                                    \
        static constexpr luisa::string_view description() noexcept {        \
            using namespace std::string_view_literals;                      \
            return #desc##sv;                                               \
        }                                                                   \
    };                                                                      \
    template<>                                                              \
    struct TypeDesc<Vector<S, 2>> {                                         \
        static constexpr luisa::string_view description() noexcept {        \
            using namespace std::string_view_literals;                      \
            return "vector<" #desc ",2>"sv;                                 \
        }                                                                   \
    };                                                                      \
    template<>                                                              \
    struct TypeDesc<Vector<S, 3>> {                                         \
        static constexpr luisa::string_view description() noexcept {        \
            using namespace std::string_view_literals;                      \
            return "vector<" #desc ",3>"sv;                                 \
        }                                                                   \
    };                                                                      \
    template<>                                                              \
    struct TypeDesc<Vector<S, 4>> {                                         \
        static constexpr luisa::string_view description() noexcept {        \
            using namespace std::string_view_literals;                      \
            return "vector<" #desc ",4>"sv;                                 \
        }                                                                   \
    };

LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(bool, bool)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(byte, byte)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(ubyte, ubyte)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(short, short)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(ushort, ushort)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(int, int)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uint, uint)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(slong, long)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(ulong, ulong)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(half, half)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(float, float)
LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(double, double)

#undef LUISA_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION

template<typename T>
    requires std::is_enum_v<T>
struct TypeDesc<T>
    : TypeDesc<std::conditional_t<
          std::is_signed_v<std::underlying_type_t<T>>,
          int, uint>> {};

// array
template<typename T, size_t N>
struct TypeDesc<std::array<T, N>> {
    static_assert(alignof(T) >= 4u, "array element must be at least 4-byte aligned");
    static luisa::string_view description() noexcept {
        static thread_local auto s = make_array_description(
            TypeDesc<T>::description(), N);
        return s;
    }
};

template<typename T, size_t N>
struct TypeDesc<T[N]> {
    static luisa::string_view description() noexcept {
        static thread_local auto s = make_array_description(
            TypeDesc<T>::description(), N);
        return s;
    }
};

template<typename T>
struct TypeDesc<Buffer<T>> {
    static luisa::string_view description() noexcept {
        static thread_local auto s = make_buffer_description(
            TypeDesc<T>::description());
        return s;
    }
};

template<>
struct TypeDesc<ByteBuffer> {
    static luisa::string_view description() noexcept {
        return "buffer<void>";
    }
};

template<typename T>
struct TypeDesc<BufferView<T>> : TypeDesc<Buffer<T>> {};

template<>
struct TypeDesc<Image<float>> {
    static constexpr luisa::string_view description() noexcept {
        return "texture<2,float>";
    }
};

template<>
struct TypeDesc<Image<int>> {
    static constexpr luisa::string_view description() noexcept {
        return "texture<2,int>";
    }
};

template<>
struct TypeDesc<Image<uint>> {
    static constexpr luisa::string_view description() noexcept {
        return "texture<2,uint>";
    }
};

template<typename T>
struct TypeDesc<ImageView<T>> : TypeDesc<Image<T>> {};

template<>
struct TypeDesc<Volume<float>> {
    static constexpr luisa::string_view description() noexcept {
        return "texture<3,float>";
    }
};

template<>
struct TypeDesc<Volume<int>> {
    static constexpr luisa::string_view description() noexcept {
        return "texture<3,int>";
    }
};

template<>
struct TypeDesc<Volume<uint>> {
    static constexpr luisa::string_view description() noexcept {
        return "texture<3,uint>";
    }
};

template<>
struct TypeDesc<BindlessArray> {
    static constexpr luisa::string_view description() noexcept {
        return "bindless_array";
    }
};

template<>
struct TypeDesc<Accel> {
    static constexpr luisa::string_view description() noexcept {
        return "accel";
    }
};

template<typename T>
struct TypeDesc<VolumeView<T>> : TypeDesc<Volume<T>> {};

// matrices
template<>
struct TypeDesc<float2x2> {
    static constexpr luisa::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<2>"sv;
    }
};

template<>
struct TypeDesc<float3x3> {
    static constexpr luisa::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<3>"sv;
    }
};

template<>
struct TypeDesc<float4x4> {
    static constexpr luisa::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<4>"sv;
    }
};

template<typename... T>
struct TypeDesc<std::tuple<T...>> {
    static luisa::string_view description() noexcept {
        auto alignment = std::max({alignof(T)...});
        static thread_local auto s = make_struct_description(
            alignment, {TypeDesc<T>::description()...});
        return s;
    }
};

}// namespace detail

template<typename T>
const Type *Type::of() noexcept {
    if constexpr (std::is_same_v<T, void>) {
        return nullptr;
    } else {
        auto desc = detail::TypeDesc<std::remove_cvref_t<T>>::description();
        if constexpr (is_custom_struct_v<T>) {
            static thread_local auto t = Type::custom(desc);
            return t;
        } else {
            static thread_local auto t = Type::from(desc);
            return t;
        }
    }
}

namespace detail {

template<typename S, typename Members, typename offsets>
struct is_valid_reflection : std::false_type {};

template<typename S, typename... M, typename O, O... os>
struct is_valid_reflection<S, std::tuple<M...>, std::integer_sequence<O, os...>> {

    static_assert(alignof(S) >= 4u, "Structs must be aligned to at least 4 bytes.");

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
    luisa::compute::detail::TypeDesc<std::remove_cvref_t<decltype(std::declval<this_type>().m)>>::description()
#define LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE(m) \
    std::remove_cvref_t<decltype(std::declval<this_type>().m)>

#ifdef _MSC_VER// force the built-in offsetof(), otherwise clangd would complain that it's not constant
#define LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET(m) \
    __builtin_offsetof(this_type, m)
#else
#define LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET(m) \
    offsetof(this_type, m)
#endif

#define LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, ...)                \
    template<>                                                               \
    struct luisa::compute::struct_member_tuple<S> {                          \
        using this_type = S;                                                 \
        using type = std::tuple<                                             \
            LUISA_MAP_LIST(                                                  \
                LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE,                          \
                ##__VA_ARGS__)>;                                             \
        using offset = std::integer_sequence<                                \
            size_t,                                                          \
            LUISA_MAP_LIST(                                                  \
                LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET,                        \
                ##__VA_ARGS__)>;                                             \
        static_assert(alignof(this_type) >= 4);                              \
        static_assert(luisa::compute::detail::is_valid_reflection_v<         \
                      this_type, type, offset>);                             \
    };                                                                       \
    template<>                                                               \
    struct luisa::compute::detail::TypeDesc<S> {                             \
        using this_type = S;                                                 \
        static luisa::string_view description() noexcept {                   \
            static auto s = luisa::compute::detail::make_struct_description( \
                alignof(S),                                                  \
                {LUISA_MAP_LIST(LUISA_STRUCTURE_MAP_MEMBER_TO_DESC,          \
                                ##__VA_ARGS__)});                            \
            return s;                                                        \
        }                                                                    \
    };
#define LUISA_STRUCT_REFLECT(S, ...) \
    LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION(S, __VA_ARGS__)

#define LUISA_CUSTOM_STRUCT_REFLECT(S, name)                         \
    template<>                                                       \
    struct luisa::compute::is_custom_struct<S> : std::true_type {};  \
    template<>                                                       \
    struct luisa::compute::detail::TypeDesc<S> {                     \
        static constexpr luisa::string_view description() noexcept { \
            return name;                                             \
        }                                                            \
    };
