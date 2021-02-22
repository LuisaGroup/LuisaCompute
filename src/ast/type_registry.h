//
// Created by Mike Smith on 2021/2/22.
//

#pragma once

#include <vector>
#include <memory>
#include <mutex>

#include <fmt/format.h>

#include <core/memory.h>
#include <core/macro_map.h>
#include <runtime/buffer.h>
#include <ast/type.h>

namespace luisa::compute {

class TypeRegistry {

private:
    std::vector<const Type *> _types;
    std::mutex _types_mutex;

public:
    template<typename F>
    decltype(auto) with_types(F &&f) noexcept {
        std::scoped_lock lock{_types_mutex};
        return f(_types);
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
