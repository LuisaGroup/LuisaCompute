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
	alignas(T) uint8_t storage[sizeof(T)];

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

	inline void Delete() noexcept {
		if constexpr (!std::is_trivially_destructible_v<T>)
			((T*)storage)->~T();
	}
	T& operator*() noexcept {
		return *(T*)storage;
	}
	T const& operator*() const noexcept {
		return *(T const*)storage;
	}
	T* operator->() noexcept {
		return (T*)storage;
	}
	T const* operator->() const noexcept {
		return (T const*)storage;
	}
	T* GetPtr() noexcept {
		return (T*)storage;
	}
	T const* GetPtr() const noexcept {
		return (T const*)storage;
	}
	operator T*() noexcept {
		return (T*)storage;
	}
	operator T const *() const noexcept {
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
		-stackObj.InPlaceNew(std::forward<Args>(args)...);
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

template<typename F, typename T>
struct LoopClass;

template<typename F, size_t... idx>
struct LoopClass<F, std::index_sequence<idx...>> {
	static void Do(const F& f) noexcept {
		auto c = {(f(idx), 0)...};
	}
};

template<typename F, uint32_t count>
struct LoopClassEarlyBreak {
	static bool Do(const F& f) noexcept {
		if (!LoopClassEarlyBreak<F, count - 1>::Do((f))) return false;
		return f(count);
	}
};

template<typename F>
struct LoopClassEarlyBreak<F, 0> {
	static bool Do(const F& f) noexcept {
		return f(0);
	}
};

template<typename F, uint32_t count>
void InnerLoop(const F& function) noexcept {
	LoopClass<typename std::remove_cvref_t<F>, std::make_index_sequence<count>>::Do(function);
}

template<typename F, uint32_t count>
bool InnerLoopEarlyBreak(const F& function) noexcept {
	return LoopClassEarlyBreak<typename std::remove_cvref_t<F>, count - 1>::Do(function);
}

//Declare Tuple

template<typename T>
using PureType_t = typename std::remove_pointer_t<std::remove_cvref_t<T>>;

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
#include <Common/FunctorMeta.h>
namespace vengine {
template<>
struct hash<Type> {
	size_t operator()(const Type& t) const noexcept {
		return t.HashCode();
	}
};
}// namespace vengine