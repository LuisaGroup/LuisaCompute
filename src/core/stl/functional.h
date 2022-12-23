#pragma once

#include <EASTL/functional.h>

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
[[nodiscard]] auto lazy_construct(F ctor) noexcept {
    return detail::LazyConstructor<F>(ctor);
}

}// namespace luisa
