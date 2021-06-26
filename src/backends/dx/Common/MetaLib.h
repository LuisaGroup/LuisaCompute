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

template<typename T>
struct funcPtr;

template<typename _Ret, typename... Args>
struct funcPtr<_Ret(Args...)> {
	using Type = _Ret (*)(Args...);
	using FuncType = _Ret(Args...);
};

template<typename _Ret, typename... Args>
struct funcPtr<_Ret (*)(Args...)> {
	using Type = _Ret (*)(Args...);
	using FuncType = _Ret(Args...);
};

template<typename T>
using funcPtr_t = typename funcPtr<T>::Type;
template<typename T>
using functor_t = typename funcPtr<T>::FuncType;
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
	bool initialized;

public:
	template<typename... Args>
	inline bool New(Args&&... args) noexcept {
		if (initialized) return false;
		initialized = true;
		stackObj.New(std::forward<Args>(args)...);
		return true;
	}
	template<typename... Args>
	inline bool InPlaceNew(Args&&... args) noexcept {
		if (initialized) return false;
		initialized = true;
		stackObj.InPlaceNew(std::forward<Args>(args)...);
		return true;
	}
	bool Initialized() const noexcept {
		return initialized;
	}
	operator bool() const noexcept {
		return Initialized();
	}
	operator bool() noexcept {
		return Initialized();
	}
	inline bool Delete() noexcept {
		if (!Initialized()) return false;
		initialized = false;
		stackObj.Delete();
		return true;
	}
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
		initialized = false;
	}
	StackObject(const StackObject<T, true>& value) noexcept {
		initialized = false;
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
template<typename T>
struct array_meta;
template<typename T, size_t N>
struct array_meta<T[N]> {
	static constexpr size_t array_size = N;
};

template<typename T>
static constexpr size_t array_count = array_meta<T>::array_size;
}// namespace vengine
#define VENGINE_ARRAY_COUNT(arr) (vengine::array_count<decltype(arr)>)

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
	if (c(std::forward<Args>(args)...)) {
		return b(std::forward<Args>(args)...);
	}
	return a(std::forward<Args>(args)...);
}
struct range {
public:
	struct rangeIte {
		int64 v;
		int64& operator++() {
			return ++v;
		}
		int64 operator++(int) {
			return v++;
		}
		int64 const* operator->() const {
			return &v;
		}
		int64 const& operator*() const {
			return v;
		}
		bool operator==(rangeIte r) const {
			return r.v == v;
		}
	};
	range(int64 b, int64 e) : b(b), e(e) {}
	range(int64 e) : b(0), e(e) {}
	rangeIte begin() const {
		return {b};
	}
	rangeIte end() const {
		return {e};
	}

private:
	int64 b;
	int64 e;
};

template<typename T>
struct ptr_range {
public:
	T* begin() const {
		return b;
	}
	T* end() const {
		return e;
	}
	ptr_range(T* b, T* e) : b(b), e(e) {}

private:
	T* b;
	T* e;
};
template<typename T>
decltype(auto) get_lvalue(T&& data) {
	return static_cast<std::remove_reference_t<T>&>(data);
}
template<typename T>
decltype(auto) get_const_lvalue(T&& data) {
	return static_cast<std::remove_reference_t<T> const&>(data);
}
template<typename A, typename B>
decltype(auto) array_same(A&& a, B&& b) {
	auto aSize = a.size();
	auto bSize = b.size();
	if (aSize != bSize) return false;
	auto ite = a.begin();
	auto end = a.end();
	auto oIte = b.begin();
	auto oEnd = b.end();
	while (ite != end && oIte != oEnd) {
		if (*ite != *oIte) return false;
		++ite;
		++oIte;
	}
	return true;
}
}// namespace vengine