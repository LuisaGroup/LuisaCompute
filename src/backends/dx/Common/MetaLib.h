#pragma once
#include <type_traits>
#include <stdint.h>
using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int32 = int32_t;
#include <typeinfo>
#include <new>
#include "../Common/Hash.h"
#include <mutex>
#include <atomic>
#include <thread>
class spin_mutex {
	std::atomic_flag flag;

public:
	spin_mutex() noexcept {
		flag.clear();
	}
	void lock() noexcept {
		while (flag.test_and_set(std::memory_order::acquire))
			std::this_thread::yield();
	}
	void unlock() noexcept {
		flag.clear(std::memory_order::release);
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

template<typename T, uint32_t size>
class Storage {
	alignas(T) char c[size * sizeof(T)];
};
template<typename T>
class Storage<T, 0> {};

using lockGuard = typename std::lock_guard<std::mutex>;

template<typename T, bool autoDispose = false>
class StackObject;

template<typename T>
class StackObject<T, false> {
private:
	alignas(T) bool storage[sizeof(T)];

public:
	template<typename... Args>
	inline void New(Args&&... args) noexcept {
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (storage) T(std::forward<Args>(args)...);
	}
	template<typename... Args>
	inline void InPlaceNew(Args&&... args) noexcept {
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (storage) T{std::forward<Args>(args)...};
	}
	inline void operator=(const StackObject<T>& value) {
		*(T*)storage = value.operator*();
	}
	inline void operator=(StackObject<T>&& value) {
		*(T*)storage = std::move(*value);
	}
	inline void Delete() noexcept {
		if constexpr (!std::is_trivially_destructible_v<T>)
			((T*)storage)->~T();
	}
	constexpr T& operator*() noexcept {
		return *(T*)storage;
	}
	constexpr T const& operator*() const noexcept {
		return *(T const*)storage;
	}
	constexpr T* operator->() noexcept {
		return (T*)storage;
	}
	constexpr T const* operator->() const noexcept {
		return (T const*)storage;
	}
	constexpr T* GetPtr() noexcept {
		return (T*)storage;
	}
	constexpr T const* GetPtr() const noexcept {
		return (T const*)storage;
	}
	constexpr operator T*() noexcept {
		return (T*)storage;
	}
	constexpr operator T const *() const noexcept {
		return (T const*)storage;
	}
	bool operator==(const StackObject<T>&) const noexcept = delete;
	bool operator!=(const StackObject<T>&) const noexcept = delete;
	StackObject() noexcept {}
	StackObject(const StackObject<T>& value) noexcept {
		New(value.operator*());
	}
};

template<typename T>
class StackObject<T, true> {
private:
	StackObject<T, false> stackObj;
	bool initialized = false;

public:
	template<typename... Args>
	inline void New(Args&&... args) noexcept {
		if (initialized) return;
		stackObj.New(std::forward<Args>(args)...);
		initialized = true;
	}
	template<typename... Args>
	inline void InPlaceNew(Args&&... args) noexcept {
		if (initialized) return;
		stackObj.InPlaceNew(std::forward<Args>(args)...);
		initialized = true;
	}
	constexpr operator bool() const noexcept {
		return initialized;
	}
	constexpr bool Initialized() const noexcept {
		return initialized;
	}
	inline void Delete() noexcept {
		if (initialized) {
			initialized = false;
			stackObj.Delete();
		}
	}
	inline void operator=(const StackObject<T, true>& value) noexcept {
		if (initialized) {
			stackObj.Delete();
		}
		initialized = value.initialized;
		if (initialized) {
			stackObj = value.stackObj;
		}
	}
	inline void operator=(StackObject<T>&& value) noexcept {
		if (initialized) {
			stackObj.Delete();
		}
		initialized = value.initialized;
		if (initialized) {
			stackObj = std::move(value.stackObj);
		}
	}
	constexpr T& operator*() noexcept {
		return *stackObj;
	}
	constexpr T const& operator*() const noexcept {
		return *stackObj;
	}
	constexpr T* operator->() noexcept {
		return stackObj.operator->();
	}
	constexpr T const* operator->() const noexcept {
		return stackObj.operator->();
	}
	constexpr T* GetPtr() noexcept {
		return stackObj.GetPtr();
	}
	constexpr T const* GetPtr() const noexcept {
		return stackObj.GetPtr();
	}
	constexpr operator T*() noexcept {
		return stackObj;
	}
	constexpr operator T const *() const noexcept {
		return stackObj;
	}
	bool operator==(const StackObject<T>&) const noexcept = delete;
	bool operator!=(const StackObject<T>&) const noexcept = delete;
	StackObject() noexcept {}
	StackObject(const StackObject<T, true>& value) noexcept {
		stackObj.New(value.operator*());
	}
	~StackObject() noexcept {
		if (initialized)
			stackObj.Delete();
	}
};
//Declare Tuple

template<typename T>
using PureType_t = typename std::remove_pointer_t<std::remove_cvref_t<T>>;

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

namespace vengine {
template<>
struct hash<Type> {
	size_t operator()(const Type& t) const noexcept {
		return t.HashCode();
	}
};
}// namespace vengine

static constexpr bool BinaryEqualTo_Size(void const* a, void const* b, uint64_t size) noexcept {
	const uint64_t bit64Count = size / sizeof(uint64_t);
	const uint64_t bit32Count = (size - bit64Count * sizeof(uint64_t)) / sizeof(uint32_t);
	const uint64_t bit16Count =
		(size - bit64Count * sizeof(uint64_t) - bit32Count * sizeof(uint32_t)) / sizeof(uint16_t);
	const uint64_t bit8Count =
		(size - bit64Count * sizeof(uint64_t) - bit32Count * sizeof(uint32_t) - bit16Count * sizeof(uint16_t)) / sizeof(uint8_t);
	if (bit64Count > 0) {
		uint64_t const*& aStartPtr = (uint64_t const*&)a;
		uint64_t const*& bStartPtr = (uint64_t const*&)b;
		uint64_t const* aEndPtr = aStartPtr + bit64Count;
		while (aStartPtr != aEndPtr) {
			if (*aStartPtr != *bStartPtr)
				return false;
			aStartPtr++;
			bStartPtr++;
		}
	}
	if (bit32Count > 0) {
		uint32_t const*& aStartPtr = (uint32_t const*&)a;
		uint32_t const*& bStartPtr = (uint32_t const*&)b;
		uint32_t const* aEndPtr = aStartPtr + bit32Count;
		while (aStartPtr != aEndPtr) {
			if (*aStartPtr != *bStartPtr)
				return false;
			aStartPtr++;
			bStartPtr++;
		}
	}
	if (bit16Count > 0) {
		uint16_t const*& aStartPtr = (uint16_t const*&)a;
		uint16_t const*& bStartPtr = (uint16_t const*&)b;
		uint16_t const* aEndPtr = aStartPtr + bit16Count;
		while (aStartPtr != aEndPtr) {
			if (*aStartPtr != *bStartPtr)
				return false;
			aStartPtr++;
			bStartPtr++;
		}
	}
	if (bit8Count > 0) {
		uint8_t const*& aStartPtr = (uint8_t const*&)a;
		uint8_t const*& bStartPtr = (uint8_t const*&)b;
		uint8_t const* aEndPtr = aStartPtr + bit8Count;
		while (aStartPtr != aEndPtr) {
			if (*aStartPtr != *bStartPtr)
				return false;
			aStartPtr++;
			bStartPtr++;
		}
	}
	return true;
}
template<typename T>
static constexpr bool BinaryEqualTo(T const* a, T const* b) {
	return BinaryEqualTo_Size(a, b, sizeof(T));
}
namespace FunctionTemplateGlobal {

template<typename T, typename... Args>
struct FunctionRetAndArgs {
	static constexpr size_t ArgsCount = sizeof...(Args);
	using RetType = typename T;
	inline static const Type retTypes = typeid(T);
	inline static const Type argTypes[ArgsCount] =
		{
			typeid(Args)...};
};

template<typename T>
struct memFuncPtr;
template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...) const> {
	using RetAndArgsType = typename FunctionRetAndArgs<_Ret, Args...>;
	using FuncType = typename _Ret(Args...);
};
template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...)> {
	using RetAndArgsType = typename FunctionRetAndArgs<_Ret, Args...>;
	using FuncType = typename _Ret(Args...);
};

template<typename T>
struct FunctionPointerData;

template<typename _Ret, typename... Args>
struct FunctionPointerData<_Ret(Args...)> {
	using RetAndArgsType = typename FunctionRetAndArgs<_Ret, Args...>;
	using FuncType = typename _Ret(Args...);
};

template<typename T>
struct FunctionType {
	using RetAndArgsType = typename memFuncPtr<decltype(&T::operator())>::RetAndArgsType;
	using FuncType = typename memFuncPtr<decltype(&T::operator())>::FuncType;
};

template<typename Ret, typename... Args>
struct FunctionType<Ret(Args...)> {
	using RetAndArgsType = typename FunctionPointerData<Ret(Args...)>::RetAndArgsType;
	using FuncType = typename FunctionPointerData<Ret(Args...)>::FuncType;
};
}// namespace FunctionTemplateGlobal

template<typename T>
using FunctionDataType = typename FunctionTemplateGlobal::FunctionType<T>::RetAndArgsType;

template<typename A, typename B>
static constexpr bool IsFunctionTypeOf = std::is_same_v<typename FunctionTemplateGlobal::FunctionType<A>::FuncType, B>;