#pragma once
#include "builtins.hpp"
#include "declval.hpp"

namespace detail {
template <class Func, class... Args, class = decltype(declval<Func>()(declval<Args>()...))>
true_type is_callable_(int);
template <class...>
false_type is_callable_(...);
}

template <class F, class... Args>
struct is_callable : decltype(detail::is_callable_<F, Args...>(0)) {};