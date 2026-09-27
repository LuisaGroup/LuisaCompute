#pragma once

#ifdef LUISA_USE_SYSTEM_STL
#include <optional>
#else

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <EASTL/optional.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif

namespace luisa {

#ifdef LUISA_USE_SYSTEM_STL
using std::make_optional;
using std::nullopt;
using std::nullopt_t;
using std::optional;
#else
using eastl::make_optional;
using eastl::nullopt;
using eastl::nullopt_t;
using eastl::optional;
#endif

}// namespace luisa

#ifndef LUISA_USE_SYSTEM_STL

// EASTL's optional only provides optional<T> vs optional<T> and optional<T> vs
// T (exact) comparisons, while std::optional also compares against arbitrary
// values (e.g. optional<size_t> == 0u or optional<std::string> == "literal").
// Provide the missing heterogeneous overloads so code written against
// std::optional semantics compiles with the bundled STL. They are defined in
// namespace eastl so argument-dependent lookup finds them like the std ones.
// Note: luisa::optional is an alias of eastl::optional in this configuration.
namespace eastl {

template<typename T, typename U>
    requires requires(const T &lhs, const U &rhs) { lhs == rhs; }
[[nodiscard]] constexpr bool operator==(const optional<T> &lhs, const U &rhs) {
    return lhs.has_value() && (*lhs == rhs);
}
template<typename T, typename U>
    requires requires(const U &lhs, const T &rhs) { lhs == rhs; }
[[nodiscard]] constexpr bool operator==(const U &lhs, const optional<T> &rhs) {
    return rhs.has_value() && (lhs == *rhs);
}
template<typename T, typename U>
    requires requires(const T &lhs, const U &rhs) { lhs != rhs; }
[[nodiscard]] constexpr bool operator!=(const optional<T> &lhs, const U &rhs) {
    return !lhs.has_value() || (*lhs != rhs);
}
template<typename T, typename U>
    requires requires(const U &lhs, const T &rhs) { lhs != rhs; }
[[nodiscard]] constexpr bool operator!=(const U &lhs, const optional<T> &rhs) {
    return !rhs.has_value() || (lhs != *rhs);
}

}// namespace eastl

#endif
