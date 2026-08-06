#pragma once

#include <type_traits>

#if !defined(LUISA_USE_SYSTEM_STL) && defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <algorithm>// for std::copy

#if LUISA_USE_SYSTEM_SPDLOG
#include <spdlog/fmt/xchar.h>
#else
#include <spdlog/fmt/bundled/xchar.h>
#endif

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/platform.h>// for TraceItem

namespace luisa {

#ifndef FMT_STRING
#define FMT_STRING(...) __VA_ARGS__
#endif

using fmt::format_to;

#ifdef LUISA_USE_SYSTEM_STL
using fmt::format;
#else
template<typename String, typename Format, typename... Args>
[[nodiscard]] inline auto format(Format &&f, Args &&...args) noexcept {
    using char_type = typename String::value_type;
    using memory_buffer = fmt::basic_memory_buffer<char_type, fmt::inline_buffer_size, luisa::allocator<char_type>>;
    memory_buffer buffer;
    luisa::format_to(std::back_inserter(buffer), std::forward<Format>(f), std::forward<Args>(args)...);
    return String{buffer.data(), buffer.size()};
}
template<typename Format, typename... Args>
[[nodiscard]] inline auto format(Format &&f, Args &&...args) noexcept {
    return format<luisa::string>(std::forward<Format>(f), std::forward<Args>(args)...);
}
#endif

[[nodiscard]] inline auto hash_to_string(uint64_t hash) noexcept {
    return luisa::format(FMT_STRING("{:016X}"), hash);
}

};// namespace luisa

namespace fmt {

template<>
struct formatter<luisa::half> : formatter<float> {
    template<typename FormatContext>
    auto format(luisa::half value, FormatContext &ctx) const {
        return formatter<float>::format(static_cast<float>(value), ctx);
    }
};

template<typename T, size_t N>
struct formatter<luisa::Vector<T, N>> {
    constexpr auto parse(format_parse_context &ctx) const noexcept {
        return ctx.end();
    }
    template<typename FormatContext>
    auto format(const luisa::Vector<T, N> &v, FormatContext &ctx) const noexcept {
        using namespace std::string_view_literals;
        using luisa::uint;
        using luisa::ushort;
        using luisa::slong;
        using luisa::ulong;
        using luisa::half;
        constexpr auto type_name =
            std::is_same_v<T, bool>   ? "bool"sv :
            std::is_same_v<T, short>  ? "short"sv :
            std::is_same_v<T, ushort> ? "ushort"sv :
            std::is_same_v<T, int>    ? "int"sv :
            std::is_same_v<T, uint>   ? "uint"sv :
            std::is_same_v<T, slong>  ? "slong"sv :
            std::is_same_v<T, ulong>  ? "ulong"sv :
            std::is_same_v<T, half>   ? "half"sv :
            std::is_same_v<T, float>  ? "float"sv :
            std::is_same_v<T, double> ? "double"sv :
                                        "unknown"sv;
        if constexpr (N == 2u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("{}2({}, {})"),
                type_name, v.x, v.y);
        } else if constexpr (N == 3u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("{}3({}, {}, {})"),
                type_name, v.x, v.y, v.z);
        } else if constexpr (N == 4u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("{}4({}, {}, {}, {})"),
                type_name, v.x, v.y, v.z, v.w);
        } else {
            static_assert(luisa::always_false_v<T>);
        }
    }
};

template<typename T, size_t N>
struct formatter<luisa::Matrix<T, N>> {
    constexpr auto parse(format_parse_context &ctx) const noexcept {
        return ctx.end();
    }
    template<typename FormatContext>
    auto format(const luisa::Matrix<T, N> &m, FormatContext &ctx) const noexcept {
        if constexpr (N == 2u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("Matrix2x2("
                           "cols[0] = ({}, {}), "
                           "cols[1] = ({}, {}))"),
                m[0].x, m[0].y,
                m[1].x, m[1].y);
        } else if constexpr (N == 3u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("Matrix3x3("
                           "cols[0] = ({}, {}, {}), "
                           "cols[1] = ({}, {}, {}), "
                           "cols[2] = ({}, {}, {}))"),
                m[0].x, m[0].y, m[0].z,
                m[1].x, m[1].y, m[1].z,
                m[2].x, m[2].y, m[2].z);
        } else if constexpr (N == 4u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("Matrix4x4("
                           "cols[0] = ({}, {}, {}, {}), "
                           "cols[1] = ({}, {}, {}, {}), "
                           "cols[2] = ({}, {}, {}, {}), "
                           "cols[3] = ({}, {}, {}, {}))"),
                m[0].x, m[0].y, m[0].z, m[0].w,
                m[1].x, m[1].y, m[1].z, m[1].w,
                m[2].x, m[2].y, m[2].z, m[2].w,
                m[3].x, m[3].y, m[3].z, m[3].w);
        } else {
            static_assert(luisa::always_false_v<luisa::Matrix<T, N>>);
        }
    }
};

template<>
struct formatter<luisa::TraceItem> {
    constexpr auto parse(format_parse_context &ctx) const noexcept {
        return ctx.end();
    }
    template<typename FormatContext>
    auto format(const luisa::TraceItem &item, FormatContext &ctx) const noexcept {
        return luisa::format_to(ctx.out(), FMT_STRING("[0x{:012x}]: {} :: {} + {}"),
                                item.address, item.module, item.symbol, item.offset);
    }
};

}// namespace fmt

namespace luisa {

template<typename T, size_t N>
[[nodiscard]] auto to_string(Vector<T, N> v) noexcept {
    return luisa::format(FMT_STRING("({})"), v);
}

template<typename T, size_t N>
[[nodiscard]] auto to_string(Matrix<T, N> m) noexcept {
    return luisa::format(FMT_STRING("({})"), m);
}

template<typename T, size_t N>
[[nodiscard]] auto to_string(std::array<T, N> a) noexcept {
    return luisa::format(FMT_STRING("({})"), a);
}

}// namespace luisa

#if !defined(LUISA_USE_SYSTEM_STL) && defined(_MSC_VER)
#pragma warning(pop)
#endif
