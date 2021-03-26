#pragma once
#include <type_traits>
#include <typeinfo>
#include <stdint.h>
struct Type {
private:
	const std::type_info* typeEle;

public:
	Type() noexcept : typeEle(nullptr) {
	}
	Type(const Type& t) noexcept : typeEle(t.typeEle) {
	}
	Type(const std::type_info& info) noexcept : typeEle(&info) {
	}
	Type(std::nullptr_t) noexcept : typeEle(nullptr) {}
	bool operator==(const Type& t) const noexcept {
		size_t a = (size_t)(typeEle);
		size_t b = (size_t)(t.typeEle);
		//have nullptr
		if ((a & b) == 0) {
			return !(a | b);
		}
		return *t.typeEle == *typeEle;
	}
	bool operator!=(const Type& t) const noexcept {
		return !operator==(t);
	}
	void operator=(const Type& t) noexcept {
		typeEle = t.typeEle;
	}
	size_t HashCode() const noexcept {
		if (!typeEle) return 0;
		return typeEle->hash_code();
	}
	const std::type_info& GetType() const noexcept {
		return *typeEle;
	}
};

template<typename T>
struct funcPtr;

template<typename _Ret, typename... Args>
struct funcPtr<_Ret(Args...)> {
	using Type = _Ret (*)(Args...);
};

template<typename T>
using funcPtr_t = typename funcPtr<T>::Type;

namespace FunctionTemplateGlobal {

template<typename T, typename... Args>
struct FunctionRetAndArgs {
	static constexpr size_t ArgsCount = sizeof...(Args);
	using RetType = T;
	inline static const Type retTypes = typeid(T);
	inline static const Type argTypes[ArgsCount] =
		{
			typeid(Args)...};
};

template<typename T>
struct memFuncPtr;
template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...) const> {
	using RetAndArgsType = FunctionRetAndArgs<_Ret, Args...>;
	using FuncType = _Ret(Args...);
	using FuncPtrType = _Ret (*)(Args...);
};
template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...)> {
	using RetAndArgsType = FunctionRetAndArgs<_Ret, Args...>;
	using FuncType = _Ret(Args...);
	using FuncPtrType = _Ret (*)(Args...);
};

template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...) const noexcept> {
	using RetAndArgsType = FunctionRetAndArgs<_Ret, Args...>;
	using FuncType = _Ret(Args...);
	using FuncPtrType = _Ret (*)(Args...);
};
template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...) noexcept> {
	using RetAndArgsType = FunctionRetAndArgs<_Ret, Args...>;
	using FuncType = _Ret(Args...);
	using FuncPtrType = _Ret (*)(Args...);
};

template<typename T>
struct FunctionPointerData;

template<typename _Ret, typename... Args>
struct FunctionPointerData<_Ret(Args...)> {
	using RetAndArgsType = FunctionRetAndArgs<_Ret, Args...>;
};

template<typename T>
struct FunctionType {
	using RetAndArgsType = typename memFuncPtr<decltype(&T::operator())>::RetAndArgsType;
	using FuncType = typename memFuncPtr<decltype(&T::operator())>::FuncType;
	using FuncPtrType = typename memFuncPtr<decltype(&T::operator())>::FuncPtrType;
};

template<typename Ret, typename... Args>
struct FunctionType<Ret(Args...)> {
	using RetAndArgsType = typename FunctionPointerData<Ret(Args...)>::RetAndArgsType;
	using FuncType = Ret(Args...);
	using FuncPtrType = Ret (*)(Args...);
};
template<typename Ret, typename... Args>
struct FunctionType<Ret (*)(Args...)> {
	using RetAndArgsType = typename FunctionPointerData<Ret(Args...)>::RetAndArgsType;
	using FuncType = Ret(Args...);
	using FuncPtrType = Ret (*)(Args...);
};
}// namespace FunctionTemplateGlobal

template<typename T>
using FunctionDataType = typename FunctionTemplateGlobal::FunctionType<T>::RetAndArgsType;

template<typename T>
using MFunctorType = typename FunctionTemplateGlobal::FunctionType<T>::FuncType;

template<typename T>
using FuncPtrType = typename FunctionTemplateGlobal::FunctionType<T>::FuncPtrType;

template<typename A, typename B>
static constexpr bool IsFunctionTypeOf = std::is_same_v<typename FunctionTemplateGlobal::FunctionType<A>::FuncType, B>;
