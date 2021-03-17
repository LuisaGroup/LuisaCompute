#pragma once
#include <typeinfo>
#include <type_traits>
struct FunctorData
{
	void(*constructor)(void*, const void*);//arg0: placement ptr  arg1: copy source
	void(*disposer)(void*);
	void(*run)(void*);
};
template <typename Type>
FunctorData GetFunctor()
{
	using T = typename std::remove_cvref_t<Type>;
	FunctorData data;
	data.constructor = [](void* place, const void* source)->void
	{
		new (place)T(*((T*)source));
	};
	data.disposer = [](void* ptr)->void
	{
		((T*)ptr)->~T();
	};
	data.run = [](void* ptr)->void
	{
		(*((T*)ptr))();
	};
	return data;
}

template <typename Type, typename Ret, typename ... Args>
Ret(*GetFunctorPointer())(void*, Args...)
{
	using T = typename std::remove_cvref_t<Type>;
	return [](void* ptr, Args... args)->Ret
	{
		return (*((T*)ptr))(args...);
	};
}

template <typename Type, typename Ret, typename ... Args>
Ret(*GetFunctorPointer_Const())(void const*, Args...)
{
	using T = typename std::remove_cvref_t<Type>;
	return [](void const* ptr, Args... args)->Ret
	{
		return (*((T const*)ptr))(args...);
	};
}