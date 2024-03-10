#pragma once

namespace detail {
template<class T>
T &&declval_(int);
template<class T>
T declval_(long);
}// namespace detail

template<class T>
decltype(detail::declval_<T>(0)) declval() noexcept;