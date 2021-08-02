#pragma once
#include "vstlconfig.h"
#include <type_traits>
#include <stdint.h>
using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int32 = int32_t;
#include <typeinfo>
#include <new>
#include "Hash.h"
#include <mutex>
#include <atomic>
#include <thread>
#include "AllocateType.h"
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
	using SelfType = StackObject<T, false>;
	template<typename... Args>
	inline SelfType& New(Args&&... args) & noexcept {
		new (storage) T(std::forward<Args>(args)...);
		return *this;
	}
	template<typename... Args>
	inline SelfType&& New(Args&&... args) && noexcept {
		return std::move(New(std::forward<Args>(args)...));
	}
	template<typename... Args>
	inline SelfType& InPlaceNew(Args&&... args) & noexcept {
		new (storage) T{std::forward<Args>(args)...};
		return *this;
	}
	template<typename... Args>
	inline SelfType&& InPlaceNew(Args&&... args) && noexcept {
		return std::move(InPlaceNew(std::forward<Args>(args)...));
	}
	inline void Delete() noexcept {
		if constexpr (!std::is_trivially_destructible_v<T>)
			(reinterpret_cast<T*>(storage))->~T();
	}
	T& operator*() & noexcept {
		return *reinterpret_cast<T*>(storage);
	}
	T&& operator*() && noexcept {
		return std::move(*reinterpret_cast<T*>(storage));
	}
	T const& operator*() const& noexcept {
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
	StackObject() noexcept {}
	StackObject(const SelfType& value) {
		if constexpr (std::is_copy_constructible_v<T>) {
			new (storage) T(*value);
		} else {
			VENGINE_EXIT;
		}
	}
	StackObject(SelfType&& value) {
		if constexpr (std::is_move_constructible_v<T>) {
			new (storage) T(std::move(*value));
		} else {
			VENGINE_EXIT;
		}
	}
	template<typename... Args>
	StackObject(Args&&... args) {
		new (storage) T(std::forward<Args>(args)...);
	}
	T& operator=(SelfType const& value) {
		if constexpr (std::is_copy_assignable_v<T>) {
			operator*() = *value;
		} else if constexpr (std::is_copy_constructible_v<T>) {
			Delete();
			New(*value);
		} else {
			VENGINE_EXIT;
		}
		return **this;
	}
	T& operator=(SelfType&& value) {
		if constexpr (std::is_move_assignable_v<T>) {
			operator*() = std::move(*value);
		} else if constexpr (std::is_move_constructible_v<T>) {
			Delete();
			New(std::move(*value));
		} else {
			VENGINE_EXIT;
		}
		return **this;
	}
	T& operator=(T const& value) {
		if constexpr (std::is_copy_assignable_v<T>) {
			operator*() = value;
		} else if constexpr (std::is_copy_constructible_v<T>) {
			Delete();
			New(value);
		} else {
			VENGINE_EXIT;
		}
		return **this;
	}
	T& operator=(T&& value) {
		if constexpr (std::is_move_assignable_v<T>) {
			operator*() = std::move(value);
		} else if constexpr (std::is_move_constructible_v<T>) {
			Delete();
			New(std::move(value));
		} else {
			VENGINE_EXIT;
		}
		return **this;
	}
};

template<typename T>
class StackObject<T, true> {
private:
	StackObject<T, false> stackObj;
	bool initialized;

public:
	using SelfType = StackObject<T, true>;
	template<typename... Args>
	inline SelfType& New(Args&&... args) & noexcept {
		if (initialized) return *this;
		initialized = true;
		stackObj.New(std::forward<Args>(args)...);
		return *this;
	}
	template<typename... Args>
	inline SelfType& InPlaceNew(Args&&... args) & noexcept {
		if (initialized) return *this;
		initialized = true;
		stackObj.InPlaceNew(std::forward<Args>(args)...);
		return *this;
	}
	template<typename... Args>
	inline SelfType&& New(Args&&... args) && noexcept {
		return std::move(New(std::forward<Args>(args)...));
	}
	template<typename... Args>
	inline SelfType&& InPlaceNew(Args&&... args) && noexcept {
		return std::move(InPlaceNew(std::forward<Args>(args)...));
	}
	bool hash_value() const noexcept {
		return initialized;
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
	void reset() const noexcept {
		Delete();
	}
	T& value() & noexcept {
		return *stackObj;
	}
	T const& value() const& noexcept {
		return *stackObj;
	}
	T&& value() && noexcept {
		return std::move(*stackObj);
	}
	template<class U>
	T value_or(U&& default_value) const& {
		if (initialized)
			return *stackObj;
		else
			return std::forward<U>(default_value);
	}
	template<class U>
	T value_or(U&& default_value) && {
		if (initialized)
			return std::move(*stackObj);
		else
			return std::forward<U>(default_value);
	}
	T& operator*() & noexcept {
		return *stackObj;
	}
	T&& operator*() && noexcept {
		return std::move(*stackObj);
	}
	T const& operator*() const& noexcept {
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
	StackObject() noexcept {
		initialized = false;
	}
	StackObject(std::nullptr_t) noexcept {
		initialized = false;
	}
	template<typename... Args>
	StackObject(Args&&... args)
		: stackObj(std::forward<Args>(args)...),
		  initialized(true) {
	}
	StackObject(const SelfType& value) noexcept {
		initialized = value.initialized;
		if (initialized) {
			if constexpr (std::is_copy_constructible_v<T>) {
				stackObj.New(*value);
			} else {
				VENGINE_EXIT;
			}
		}
	}
	StackObject(SelfType&& value) noexcept {
		initialized = value.initialized;
		if (initialized) {
			if constexpr (std::is_move_constructible_v<T>) {
				stackObj.New(std::move(*value));
			} else {
				VENGINE_EXIT;
			}
		}
	}
	~StackObject() noexcept {
		if (Initialized())
			stackObj.Delete();
	}
	T& operator=(SelfType const& value) {
		if (!initialized) {
			if (value.initialized) {
				if constexpr (std::is_copy_constructible_v<T>) {
					stackObj.New(*value);
				} else {
					VENGINE_EXIT;
				}
				initialized = true;
			}
		} else {
			if (value.initialized) {
				stackObj = value.stackObj;
			} else {
				stackObj.Delete();
				initialized = false;
			}
		}
		return *stackObj;
	}
	T& operator=(SelfType&& value) {
		if (!initialized) {
			if (value.initialized) {
				if constexpr (std::is_move_constructible_v<T>) {
					stackObj.New(std::move(*value));
				} else {
					VENGINE_EXIT;
				}
				initialized = true;
			}
		} else {
			if (value.initialized) {
				stackObj = std::move(value.stackObj);
			} else {
				stackObj.Delete();
				initialized = false;
			}
		}
		return *stackObj;
	}
	T& operator=(T const& value) {
		if (!initialized) {
			if constexpr (std::is_copy_constructible_v<T>) {
				stackObj.New(value);
			} else {
				VENGINE_EXIT;
			}
			initialized = true;

		} else {
			stackObj = value;
		}
		return *stackObj;
	}
	T& operator=(T&& value) {
		if (!initialized) {
			if constexpr (std::is_move_constructible_v<T>) {
				stackObj.New(std::move(value));
			} else {
				VENGINE_EXIT;
			}
			initialized = true;
		} else {
			stackObj = std::move(value);
		}
		return *stackObj;
	}
};
//Declare Tuple

template<typename T>
using PureType_t = std::remove_pointer_t<std::remove_cvref_t<T>>;

struct Type {
private:
	const std::type_info* typeEle;
	struct DefaultType {};

public:
	Type() noexcept : typeEle(&typeid(DefaultType)) {
	}
	Type(const Type& t) noexcept : typeEle(t.typeEle) {
	}
	Type(const std::type_info& info) noexcept : typeEle(&info) {
	}
	Type(std::nullptr_t) noexcept : typeEle(nullptr) {}
	bool operator==(const Type& t) const noexcept {
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

namespace vstd {
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
	static constexpr size_t byte_size = N * sizeof(T);
};

template<typename T>
static constexpr size_t array_count = array_meta<T>::array_size;

template<typename T>
static constexpr size_t array_size = array_meta<T>::byte_size;
}// namespace vstd
#define VENGINE_ARRAY_COUNT(arr) (vstd::array_count<decltype(arr)>)
#define VENGINE_ARRAY_SIZE(arr) (vstd::array_size<decltype(arr)>)

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
namespace vstd {
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
		int64 inc;
		int64& operator++() {
			v += inc;
			return v;
		}
		int64 operator++(int) {
			auto lastV = v;
			v += inc;
			return lastV;
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
	range(int64 b, int64 e, int64 inc = 1) : b(b), e(e), inc(inc) {}
	range(int64 e) : b(0), e(e), inc(1) {}
	rangeIte begin() const {
		return {b, inc};
	}
	rangeIte end() const {
		return {e};
	}

private:
	int64 b;
	int64 e;
	int64 inc;
};

template<typename T>
struct ptr_range {
public:
	struct rangeIte {
		T* v;
		int64 inc;
		T* operator++() {
			v += inc;
			return v;
		}
		T* operator++(int) {
			auto lastV = v;
			v += inc;
			return lastV;
		}
		T* operator->() const {
			return v;
		}
		T& operator*() const {
			return *v;
		}
		bool operator==(rangeIte r) const {
			return r.v == v;
		}
	};

	rangeIte begin() const {
		return {b, inc};
	}
	rangeIte end() const {
		return {e};
	}
	ptr_range(T* b, T* e, int64_t inc = 1) : b(b), e(e), inc(inc) {}

private:
	T* b;
	T* e;
	int64_t inc;
};
template<typename T>
struct disposer {
	T t;
	~disposer() {
		t();
	}
};

template<typename T>
disposer<T> create_disposer(T&& t) {
	return disposer<T>{std::forward<T>(t)};
}

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
template<typename... AA>
class variant {
	using FuncType = funcPtr_t<void(void*, void*)>;
	using FuncType_Const = funcPtr_t<void(void*, void const*)>;

	template<size_t maxSize, size_t... szs>
	struct MaxSize;
	template<size_t maxSize, size_t v, size_t... szs>
	struct MaxSize<maxSize, v, szs...> {
		static constexpr size_t MAX_SIZE = MaxSize<(maxSize > v ? maxSize : v), szs...>::MAX_SIZE;
	};
	template<size_t maxSize>
	struct MaxSize<maxSize> {
		static constexpr size_t MAX_SIZE = maxSize;
	};

	static constexpr size_t argSize = sizeof...(AA);

	template<typename T, typename Args>
	static void GetFuncPtr_Const(void* ptr, void const* arg) {
		(*reinterpret_cast<T*>(ptr))(*reinterpret_cast<Args const*>(arg));
	}

	template<size_t idx, size_t c, typename... Args>
	struct Iterator {
		template<typename... Funcs>
		static void Set(FuncType* funcPtr, void** funcP, Funcs&&... fs) {}

		template<typename... Funcs>
		static void Set_Const(FuncType_Const* funcPtr, void** funcP, Funcs&&... fs) {}
	};

	template<size_t idx, size_t c, typename T, typename... Args>
	struct Iterator<idx, c, T, Args...> {
		template<typename F, typename... Funcs>
		static void Set(FuncType* funcPtr, void** funcP, F&& f, Funcs&&... fs) {
			if constexpr (idx == c) return;
			*funcPtr = [](void* ptr, void* arg) {
				(*reinterpret_cast<std::remove_reference_t<F>*>(ptr))(*reinterpret_cast<std::remove_reference_t<T>*>(arg));
			};
			*funcP = &f;
			Iterator<idx + 1, c, Args...>::template Set<Funcs...>(funcPtr + 1, funcP + 1, std::forward<Funcs>(fs)...);
		}
		template<typename F, typename... Funcs>
		static void Set_Const(FuncType_Const* funcPtr, void** funcP, F&& f, Funcs&&... fs) {
			if constexpr (idx == c) return;
			*funcPtr = [](void* ptr, void const* arg) {
				(*reinterpret_cast<std::remove_reference_t<F>*>(ptr))(*reinterpret_cast<std::remove_reference_t<T> const*>(arg));
			};
			*funcP = &f;
			Iterator<idx + 1, c, Args...>::template Set_Const<Funcs...>(funcPtr + 1, funcP + 1, std::forward<Funcs>(fs)...);
		}
	};

	template<typename... Args>
	struct Constructor {
		template<typename A>
		static size_t CopyOrMoveConst(void*, size_t idx, A&&) {
			return idx;
		}
		template<typename... A>
		static size_t AnyConst(void*, size_t idx, A&&...) {
			return idx;
		}
		static void Dispose(size_t, void*) {}
		static void Copy(size_t, void*, void const*) {}
		static void Move(size_t, void*, void*) {}
		template<size_t v>
		static void Get(void*) {}
		static void Get(void const*) {}
	};
	template<typename B, typename... Args>
	struct Constructor<B, Args...> {
		template<typename A>
		static size_t CopyOrMoveConst(void* ptr, size_t idx, A&& a) {
			if constexpr (std::is_same_v<std::remove_cvref_t<B>, std::remove_cvref_t<A>>) {
				new (ptr) B(std::forward<A>(a));
				return idx;
			} else {
				return Constructor<Args...>::template CopyOrMoveConst<A>(ptr, idx + 1, std::forward<A>(a));
			}
		}
		template<typename... A>
		static size_t AnyConst(void* ptr, size_t idx, A&&... a) {
			if constexpr (std::is_constructible_v<B, A&&...>) {
				new (ptr) B(std::forward<A>(a)...);
				return idx;
			} else {
				return Constructor<Args...>::template AnyConst<A...>(ptr, idx + 1, std::forward<A>(a)...);
			}
		}
		static void Dispose(size_t v, void* ptr) {
			if (v == 0) {
				reinterpret_cast<B*>(ptr)->~B();
			} else {
				Constructor<Args...>::Dispose(v - 1, ptr);
			}
		}
		static void Copy(size_t v, void* ptr, void const* dstPtr) {
			if (v == 0) {
				new (ptr) B(*reinterpret_cast<B const*>(dstPtr));
			} else {
				Constructor<Args...>::Copy(v - 1, ptr, dstPtr);
			}
		}
		static void Move(size_t v, void* ptr, void* dstPtr) {
			if (v == 0) {
				new (ptr) B(std::move(*reinterpret_cast<B*>(dstPtr)));
			} else {
				Constructor<Args...>::Move(v - 1, ptr, dstPtr);
			}
		}
		template<size_t v>
		static decltype(auto) Get(void* ptr) {
			if constexpr (v == 0) {
				return vstd::get_lvalue(*reinterpret_cast<B*>(ptr));
			} else {
				return Constructor<Args...>::template Get<v - 1>(ptr);
			}
		}
		template<size_t v>
		static decltype(auto) Get(void const* ptr) {
			if constexpr (v == 0) {
				return vstd::get_lvalue(*reinterpret_cast<B const*>(ptr));
			} else {
				return Constructor<Args...>::template Get<v - 1>(ptr);
			}
		}
	};
	union {
		uint8_t placeHolder[MaxSize<0, sizeof(AA)...>::MAX_SIZE];
	};
	size_t switcher = 0;

public:
	variant() {
		switcher = sizeof...(AA);
	}
	template<typename T, typename... Arg>
	variant(T&& t, Arg&&... arg) {
		if constexpr (sizeof...(Arg) == 0) {
			switcher = Constructor<AA...>::template CopyOrMoveConst<T>(placeHolder, 0, std::forward<T>(t));
			if (switcher < sizeof...(AA)) return;
		}
		switcher = Constructor<AA...>::template AnyConst<T, Arg...>(placeHolder, 0, std::forward<T>(t), std::forward<Arg>(arg)...);
	}

	variant(variant const& v)
		: switcher(v.switcher) {
		Constructor<AA...>::Copy(switcher, placeHolder, v.placeHolder);
	}
	variant(variant&& v)
		: switcher(v.switcher) {
		Constructor<AA...>::Move(switcher, placeHolder, v.placeHolder);
	}
	variant(variant& v)
		: variant(static_cast<variant const&>(v)) {
	}
	variant(variant const&& v)
		: variant(v) {
	}
	~variant() {
		Constructor<AA...>::Dispose(switcher, placeHolder);
	}
	void* GetPlaceHolder() { return placeHolder; }
	void const* GetPlaceHolder() const { return placeHolder; }
	size_t GetType() const { return switcher; }

	template<size_t i>
	decltype(auto) get() {
#ifdef DEBUG
		if (i != switcher) {
			VENGINE_EXIT;
		}
#endif
		return Constructor<AA...>::template Get<i>(placeHolder);
	}
	template<size_t i>
	decltype(auto) get() const {
#ifdef DEBUG
		if (i != switcher) {
			VENGINE_EXIT;
		}
#endif
		return Constructor<AA...>::template Get<i>(placeHolder);
	}

	template<typename Arg>
	variant& operator=(Arg&& arg) {
		this->~variant();
		new (this) variant(std::forward<Arg>(arg));
		return *this;
	}

	template<typename... Funcs>
	void visit(Funcs&&... funcs) {
		static_assert(argSize == sizeof...(Funcs), "functor size not equal!");
		if (switcher >= argSize) return;
		FuncType ftype[argSize];
		void* funcPs[argSize];
		Iterator<0, argSize, AA...>::template Set<Funcs...>(ftype, funcPs, std::forward<Funcs>(funcs)...);
		ftype[switcher](funcPs[switcher], placeHolder);
	}

	template<typename... Funcs>
	void visit(Funcs&&... funcs) const {
		static_assert(argSize == sizeof...(Funcs), "functor size not equal!");
		if (switcher >= argSize) return;
		FuncType_Const ftype[argSize];
		void* funcPs[argSize];
		Iterator<0, argSize, AA const...>::template Set_Const<Funcs...>(ftype, funcPs, std::forward<Funcs>(funcs)...);
		ftype[switcher](funcPs[switcher], placeHolder);
	}
};

template<typename T>
using optional = StackObject<T, true>;
}// namespace vstd