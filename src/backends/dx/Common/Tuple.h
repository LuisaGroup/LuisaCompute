#pragma once
#include <type_traits>
#include <stdint.h>
//Declare Tuple
template <typename ... Types>
struct Tuple;
template<>
struct Tuple<> {};

//Declare TupleGetter
template <size_t index, typename Tup>
struct TupleGetter;

template <size_t index>
struct TupleGetter<index, Tuple<>>
{
	using Type = typename Tuple<>;
	static constexpr Type& Get(Tuple<>& value) noexcept
	{
		//Tuple Out of Range!
		static_assert(std::_Always_false<char>);
		return value;
	}
	static constexpr Type const& Get(Tuple<> const& value) noexcept
	{
		//Tuple Out of Range!
		static_assert(std::_Always_false<char>);
		return value;

	}
};


template <size_t index, typename T, typename ... Types>
struct TupleGetter<index, Tuple<T, Types...>>
{
	using Type = typename TupleGetter<index - 1, Tuple<Types...>>::Type;
	static constexpr typename Type& Get(Tuple<T, Types...>& value) noexcept;
	static constexpr typename Type const& Get(Tuple<T, Types...> const& value) noexcept;
};


template <typename T, typename ... Types>
struct TupleGetter<0, Tuple<T, Types...>>
{
	using Type = typename T;
	static constexpr T& Get(Tuple<T, Types...>& value) noexcept;
	static constexpr T const& Get(Tuple<T, Types...> const& value) noexcept;
};
//Body of Tuple
template <typename T, typename ... Types>
struct Tuple<T, Types...> : public Tuple<Types...>
{
	T value;
	template <typename F, typename ... FArgs>
	constexpr Tuple(F&& t, FArgs&&... types) noexcept : Tuple<Types...>(std::forward<FArgs>(types)...), value(std::forward<F>(t))
	{}
	constexpr Tuple() noexcept {}
public:
	template <size_t i>
	constexpr typename TupleGetter<i, Tuple<T, Types...>>::Type& Get() noexcept;
	template <size_t i>
	constexpr typename TupleGetter<i, Tuple<T, Types...>>::Type const& Get() const noexcept;
	template <typename F>
	constexpr F& GetFirst() noexcept;
	template <typename F>
	constexpr F& GetLast() noexcept;
	template <typename F>
	constexpr F const& GetFirst() const noexcept;
	template <typename F>
	constexpr F const& GetLast() const noexcept;
	template <typename F>
	static constexpr bool ContainedType() noexcept;
};

//Function of Tuple Getter
template <size_t index, typename T, typename ... Types>
constexpr typename TupleGetter<index, Tuple<T, Types...>>::Type const& TupleGetter<index, Tuple<T, Types...>>::Get(Tuple<T, Types...> const& value) noexcept
{
	return TupleGetter<index - 1, Tuple<Types...>>::Get(value);
}

template <typename T, typename ... Types>
constexpr T const& TupleGetter<0, Tuple<T, Types...>>::Get(Tuple<T, Types...> const& value) noexcept
{
	return value.value;
}
template <size_t index, typename T, typename ... Types>
constexpr typename TupleGetter<index, Tuple<T, Types...>>::Type& TupleGetter<index, Tuple<T, Types...>>::Get(Tuple<T, Types...>& value) noexcept
{
	return TupleGetter<index - 1, Tuple<Types...>>::Get(value);
}

template <typename T, typename ... Types>
constexpr T& TupleGetter<0, Tuple<T, Types...>>::Get(Tuple<T, Types...>& value) noexcept
{
	return value.value;
}

template <typename T, typename ... Types>
template <size_t i>
constexpr typename TupleGetter<i, Tuple<T, Types...>>::Type& Tuple<T, Types...>::Get() noexcept
{
	return TupleGetter<i, Tuple<T, Types...>>::Get(*this);
}
template <typename T, typename ... Types>
template <size_t i>
constexpr typename TupleGetter<i, Tuple<T, Types...>>::Type const& Tuple<T, Types...>::Get() const noexcept
{
	return TupleGetter<i, Tuple<T, Types...>>::Get(*this);
}
//Tuple Type Getter
template <typename A, typename B>
struct TypeEqual
{
	static constexpr bool value = false;
};

template <typename T>
struct TypeEqual<T, T>
{
	static constexpr bool value = true;
};


template <typename Tar, typename ... Args>
struct GetFirstFromTuple;
template <typename Tar>
struct GetFirstFromTuple<Tar>
{
	static constexpr bool containedType = false;
	static constexpr Tar& Run(Tuple<>& tuple) noexcept
	{
		//Can't find type
		static_assert(std::_Always_false<char>);
	}
	static constexpr Tar const& Run(Tuple<> const& tuple) noexcept
	{
		//Can't find type
		static_assert(std::_Always_false<char>);
	}
};
template <typename Tar, typename T, typename ... Args>
struct GetFirstFromTuple< Tar, T, Args...>
{
	static constexpr bool containedType = GetFirstFromTuple<Tar, Args...>::containedType || TypeEqual<Tar, T>::value;

	static constexpr Tar& Run(Tuple<T, Args...>& tuple) noexcept
	{
		if constexpr (TypeEqual<Tar, T>::value)
		{
			return tuple.value;
		}
		else
		{
			return GetFirstFromTuple<Tar, Args...>::Run(static_cast<Tuple<Args...>&>(tuple));
		}
	}
	static constexpr Tar const& Run(Tuple<T, Args...> const& tuple) noexcept
	{
		if constexpr (TypeEqual<Tar, T>::value)
		{
			return tuple.value;
		}
		else
		{
			return GetFirstFromTuple<Tar, Args...>::Run(static_cast<Tuple<Args...> const&>(tuple));
		}
	}
};

template <typename Tar, typename ... Args>
struct GetLastFromTuple;

template <typename Tar>
struct GetLastFromTuple<Tar>
{
	static constexpr bool matched = false;
	static constexpr Tar& Run(Tuple<>& tuple) noexcept
	{
		return *(Tar*)nullptr;
	}
	static constexpr Tar const& Run(Tuple<> const& tuple) noexcept
	{
		return *(Tar const*)nullptr;
	}
};

template <typename Tar, typename T, typename ... Args>
struct GetLastFromTuple< Tar, T, Args...>
{
	static constexpr bool lastMatched = GetLastFromTuple<Tar, Args...>::matched;
	static constexpr bool currentMatched = TypeEqual<Tar, T>::value;
	static constexpr bool matched = currentMatched || lastMatched;

	static constexpr Tar& Run(Tuple<T, Args...>& tuple) noexcept
	{
		if constexpr (lastMatched)
		{
			return GetLastFromTuple<Tar, Args...>::Run(static_cast<Tuple<Args...>&> (tuple));
		}
		if constexpr (currentMatched)
		{
			return tuple.value;
		}
		if constexpr (!lastMatched && !currentMatched)
		{
			//Can't find type
			static_assert(std::_Always_false<char>);
		}
	}
	static constexpr Tar const& Run(Tuple<T, Args...> const& tuple) noexcept
	{
		if constexpr (lastMatched)
		{
			return GetLastFromTuple<Tar, Args...>::Run(static_cast<Tuple<Args...> const&> (tuple));
		}
		if constexpr (currentMatched)
		{
			return tuple.value;
		}
		if constexpr (!lastMatched && !currentMatched)
		{
			//Can't find type
			static_assert(std::_Always_false<char>);
		}
	}
};
template <typename T, typename ... Types>
template <typename F>
constexpr F& Tuple<T, Types...>::GetFirst() noexcept
{
	return GetFirstFromTuple<F, T, Types...>::Run(*this);
}
template <typename T, typename ... Types>
template <typename F>
constexpr F& Tuple<T, Types...>::GetLast() noexcept
{
	return GetLastFromTuple<F, T, Types...>::Run(*this);
}
template <typename T, typename ... Types>
template <typename F>
constexpr F const& Tuple<T, Types...>::GetFirst() const noexcept
{
	return GetFirstFromTuple<F, T, Types...>::Run(*this);
}
template <typename T, typename ... Types>
template <typename F>
constexpr F const& Tuple<T, Types...>::GetLast() const noexcept
{
	return GetLastFromTuple<F, T, Types...>::Run(*this);
}

template <typename T, typename ... Types>
template <typename F>
constexpr bool Tuple<T, Types...>::ContainedType() noexcept
{
	return GetFirstFromTuple<F, T, Types...>::containedType;
}