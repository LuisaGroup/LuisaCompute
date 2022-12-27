#pragma once

#include <EASTL/finally.h>
#include <EASTL/functional.h>
#include <EASTL/bonus/adaptors.h>

namespace luisa {

using eastl::equal_to;
using eastl::function;
using eastl::greater;
using eastl::greater_equal;
using eastl::less;
using eastl::less_equal;
using eastl::move_only_function;
using eastl::not_equal_to;

namespace detail {

template<typename F>
class LazyConstructor {
private:
    mutable F _ctor;

public:
    explicit LazyConstructor(F _ctor) noexcept : _ctor{_ctor} {}
    [[nodiscard]] operator auto() const noexcept { return _ctor(); }
};

}// namespace detail

template<typename F>
[[nodiscard]] inline auto lazy_construct(F ctor) noexcept {
    return detail::LazyConstructor<F>(ctor);
}

using eastl::make_finally;

// overloaded pattern
template<typename... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

// deduction guide (not needed as of C++20, but provided here for compatibility)
template<typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

template<typename... T>
[[nodiscard]] inline auto make_overloaded(T &&...t) noexcept {
    return overloaded{std::forward<T>(t)...};
}

}// namespace luisa
