#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <utility>
#include <tuple>
namespace vstd {
template<typename T>
struct compare {
	template<typename V>
	int32_t operator()(T const& a, V const& b) const {
        if constexpr (std::is_arithmetic_v<T> || std::is_pointer_v<T>) {
            if (a == b)
                return 0;
            else if (a < b)
                return -1;
            return 1;
        } else {
            static_assert(sizeof(T) == sizeof(V), "Same size!");
            return memcmp(&a, &b, sizeof(T));
        }
	}
};
template<>
struct compare<std::nullptr_t> {
	template <typename T>
	int32_t operator()(std::nullptr_t, T* ptr) const {
		return (ptr == nullptr) ? 0 : -1;
	}
	template<typename T>
	int32_t operator()(T* ptr, std::nullptr_t) const {
		return (ptr == nullptr) ? 0 : 1;
	}
	int32_t operator()(std::nullptr_t, std::nullptr_t) const{
		return 0;
	}
};
namespace detail {
template<typename... Args>
struct MultiCompare {
};
template<typename T, typename... Args>
struct MultiCompare<T, Args...> {
	static int32_t Run(
		T const& a, Args const&... aArgs,
		T const& b, Args const&... bArgs) {
		compare<T> hs;
		auto value = hs(a, b);
		if (value == 0) {
			if constexpr (sizeof...(Args) > 0)
				return MultiCompare<Args...>::Run(aArgs..., bArgs...);
			else
				return 0;
		} else {
			return value;
		}
	}
};
template<size_t index, typename Tuple>
struct TupleCompare {};
template<size_t index, typename... T>
struct TupleCompare<index, std::tuple<T...>> {
	static int32_t Run(std::tuple<T...> const& a, std::tuple<T...> const& b) {
		if constexpr (index >= sizeof...(T)) {
			return 0;
		} else {
			using CurType = std::remove_cvref_t<decltype(std::get<index>(std::declval<std::tuple<T...>>()))>;
			compare<CurType> hs;
			auto value = hs(std::get<index>(a), std::get<index>(b));
			if (value == 0) {
				return TupleCompare<index + 1, std::tuple<T...>>::Run(a, b);
			} else
				return value;
		}
	}
};
}// namespace detail

template<typename A, typename B>
struct compare<std::pair<A, B>> {
	int32_t operator()(std::pair<A, B> const& a, std::pair<A, B> const& b) const {
		return detail::MultiCompare<A, B>::Run(a.first, a.second, b.first, b.second);
	}
};
template<typename... T>
struct compare<std::tuple<T...>> {
	int32_t operator()(std::tuple<T...> const& a, std::tuple<T...> const& b) const {
		return detail::TupleCompare<0, std::tuple<T...>>::Run(a, b);
	}
};
}// namespace vstd