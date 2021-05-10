#pragma once
#include <VEngineConfig.h>
#include <type_traits>
#include <stdint.h>
using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int32 = int32_t;
#include <typeinfo>
#include <new>
#include <Common/Hash.h>
#include <mutex>
#include <atomic>
#include <thread>

#ifdef MSVC
#define AUTO_FUNC template<typename TT = void> \
inline
#else
#define AUTO_FUNC inline
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#define VENGINE_INTRIN_PAUSE() _mm_pause()
#elif defined(_M_X64)
#include <windows.h>
#define VENGINE_INTRIN_PAUSE() YieldProcessor()
#elif defined(__aarch64__)
#define VENGINE_INTRIN_PAUSE() asm volatile("isb")
#else
#include <mutex>
#define VENGINE_INTRIN_PAUSE() std::this_thread::yield()
#endif

class spin_mutex {
	std::atomic_flag _flag;

public:
	spin_mutex() noexcept {
		_flag.clear();
	}
	void lock() noexcept {
		while (_flag.test_and_set(std::memory_order::acquire)) {// acquire lock
#ifdef __cpp_lib_atomic_flag_test
			while (_flag.test(std::memory_order::relaxed)) {// test lock
#endif
				VENGINE_INTRIN_PAUSE();
#ifdef __cpp_lib_atomic_flag_test
			}
#endif
		}
	}

	bool isLocked() const noexcept {
		return _flag.test(std::memory_order::relaxed);
	}
	void unlock() noexcept {
		_flag.clear(std::memory_order::release);
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

using lockGuard = std::lock_guard<std::mutex>;

template<typename T, bool autoDispose = false>
class StackObject;
template<typename T>
class StackObject<T, false> {
private:
	alignas(T) uint8_t storage[sizeof(T)];

public:
	template<typename... Args>
	inline void New(Args&&... args) noexcept {
		new (storage) T(std::forward<Args>(args)...);
	}
	template<typename... Args>
	inline void InPlaceNew(Args&&... args) noexcept {
		new (storage) T{std::forward<Args>(args)...};
	}
	/*
	inline void operator=(const StackObject<T>& value) {
		*reinterpret_cast<T*>(storage) = value.operator*();
	}
	inline void operator=(StackObject<T>&& value) {
		*reinterpret_cast<T*>(storage) = std::move(*value);
	}*/
	inline void Delete() noexcept {
		if constexpr (!std::is_trivially_destructible_v<T>)
			(reinterpret_cast<T*>(storage))->~T();
	}
	T& operator*() noexcept {
		return *reinterpret_cast<T*>(storage);
	}
	T const& operator*() const noexcept {
		return *reinterpret_cast<T const*>(storage);
	}
	T* operator->() noexcept {
		return reinterpret_cast<T*>(storage);
	}
	T const* operator->() const noexcept {
		return reinterpret_cast<T const*>(storage);
	}
	T* GetPtr() noexcept {
		return reinterpret_cast<T*>(storage);
	}
	T const* GetPtr() const noexcept {
		return reinterpret_cast<T const*>(storage);
	}
	operator T*() noexcept {
		return reinterpret_cast<T*>(storage);
	}
	operator T const *() const noexcept {
		return reinterpret_cast<T const*>(storage);
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
	std::atomic_flag initialized;

public:
	template<typename... Args>
	inline void New(Args&&... args) noexcept {
		if (initialized.test_and_set(std::memory_order_relaxed)) return;
		stackObj.New(std::forward<Args>(args)...);
	}
	template<typename... Args>
	inline void InPlaceNew(Args&&... args) noexcept {
		if (initialized.test_and_set(std::memory_order_relaxed)) return;
		stackObj.InPlaceNew(std::forward<Args>(args)...);
	}
	bool Initialized() const noexcept {
		return initialized.test(std::memory_order_relaxed);
	}
	operator bool() const noexcept {
		return Initialized();
	}
	operator bool() noexcept {
		return Initialized();
	}
	inline void Delete() noexcept {
		if (!Initialized()) return;
		initialized.clear();
		stackObj.Delete();
	}
	/*inline void operator=(const StackObject<T, true>& value) noexcept {
		if (Initialized()) {
			stackObj.Delete();
		}
		if (value.Initialized()) {
			initialized.test_and_set(std::memory_order_relaxed);
		} else {
			initialized.clear();
		}
		if (Initialized()) {
			stackObj = value.stackObj;
		}
	}
	inline void operator=(StackObject<T>&& value) noexcept {
		if (Initialized()) {
			stackObj.Delete();
		}
		if (value.Initialized()) {
			initialized.test_and_set(std::memory_order_relaxed);
		} else {
			initialized.clear();
		}
		if (Initialized()) {
			stackObj = std::move(value.stackObj);
		}
	}*/
	T& operator*() noexcept {
		return *stackObj;
	}
	T const& operator*() const noexcept {
		return *stackObj;
	}
	T* operator->() noexcept {
		return stackObj.operator->();
	}
	T const* operator->() const noexcept {
		return stackObj.operator->();
	}
	T* GetPtr() noexcept {
		return stackObj.GetPtr();
	}
	T const* GetPtr() const noexcept {
		return stackObj.GetPtr();
	}
	operator T*() noexcept {
		return stackObj;
	}
	operator T const *() const noexcept {
		return stackObj;
	}
	bool operator==(const StackObject<T>&) const noexcept = delete;
	bool operator!=(const StackObject<T>&) const noexcept = delete;
	StackObject() noexcept {
		initialized.clear();
	}
	StackObject(const StackObject<T, true>& value) noexcept {
		initialized.clear();
		stackObj.New(value.operator*());
	}
	~StackObject() noexcept {
		if (Initialized())
			stackObj.Delete();
	}
};
//Declare Tuple

template<typename T>
using PureType_t = std::remove_pointer_t<std::remove_cvref_t<T>>;

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
	using RetType = T;
	inline static const Type retTypes = typeid(T);
	inline static const Type argTypes[ArgsCount] =
		{
			typeid(Args)...};
};

template<typename T>
struct memFuncPtr;

template<typename T>
struct FunctionPointerData;

template<typename _Ret, typename... Args>
struct FunctionPointerData<_Ret(Args...)> {
	using RetAndArgsType = FunctionRetAndArgs<_Ret, Args...>;
	static constexpr size_t ArgsCount = sizeof...(Args);
};

template<typename T>
struct FunctionType {
	using Type = typename memFuncPtr<decltype(&T::operator())>::Type;
};

template<typename Ret, typename... Args>
struct FunctionType<Ret(Args...)> {
	using Type = FunctionType<Ret(Args...)>;
	using RetAndArgsType = typename FunctionPointerData<Ret(Args...)>::RetAndArgsType;
	using FuncType = Ret(Args...);
	using RetType = Ret;
	static constexpr size_t ArgsCount = sizeof...(Args);
	using FuncPtrType = Ret (*)(Args...);
};

template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...)> {
	using Type = FunctionType<_Ret(Args...)>;
};

template<typename Class, typename _Ret, typename... Args>
struct memFuncPtr<_Ret (Class::*)(Args...) const> {
	using Type = FunctionType<_Ret(Args...)>;
};

template<typename Ret, typename... Args>
struct FunctionType<Ret (*)(Args...)> {
	using Type = FunctionType<Ret(Args...)>;
};
}// namespace FunctionTemplateGlobal

template<typename T>
using FunctionDataType = typename FunctionTemplateGlobal::FunctionType<T>::Type::RetAndArgsType;

template<typename T>
using FuncPtrType = typename FunctionTemplateGlobal::FunctionType<T>::Type::FuncPtrType;

template<typename T>
using FuncType = typename FunctionTemplateGlobal::FunctionType<T>::Type::FuncType;

template<typename T>
using FuncRetType = typename FunctionTemplateGlobal::FunctionType<T>::Type::RetType;

template<typename T>
constexpr size_t FuncArgCount = FunctionTemplateGlobal::FunctionType<T>::Type::ArgsCount;

template<typename Func, typename Target>
static constexpr bool IsFunctionTypeOf = std::is_same_v<FuncType<Func>, Target>;
namespace vengine {
template<typename A, typename B, typename C, typename... Args>
decltype(auto) select(A&& a, B&& b, C&& c, Args&&... args) {
	return [&]() {
		if (c(std::forward<Args>(args)...)) {
			return b(std::forward<Args>(args)...);
		}
		return a(std::forward<Args>(args)...);
	};
}
template<typename A, typename B, typename C, typename... Args>
decltype(auto) range(A&& startIndex, B&& endIndex, C&& func, Args&&... args) {
	return [&]() {
		auto&& end = endIndex();
		for (auto v = std::move(startIndex()); v < end; ++v) {
			func(std::forward<Args>(args)...);
		}
	};
}
template<typename A, typename B, typename C, typename... Args>
decltype(auto) reverse_range(A&& startIndex, B&& endIndex, C&& func, Args&&... args) {
	return [&]() {
		auto&& start = startIndex();
		for (auto v = std::move(endIndex()); v > start; v--) {
			func(std::forward<Args>(args)...);
		}
	};
}
}// namespace vengine