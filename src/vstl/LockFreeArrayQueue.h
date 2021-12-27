#pragma once
#include <thread>
#include <vstl/MetaLib.h>
#include <vstl/Memory.h>
#include <vstl/VAllocator.h>
#include <vstl/spin_mutex.h>
namespace vstd {
template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class LockFreeArrayQueue {
	using Allocator = VAllocHandle<allocType>;
	size_t head;
	size_t tail;
	size_t capacity;
	mutable spin_mutex mtx;
	T* arr;

	static constexpr size_t GetIndex(size_t index, size_t capacity) noexcept {
		return index & capacity;
	}
	using SelfType = LockFreeArrayQueue<T, allocType>;

public:
	LockFreeArrayQueue(size_t capacity) : head(0), tail(0) {
		if (capacity < 32) capacity = 32;
		capacity = [](size_t capacity) {
			size_t ssize = 1;
			while (ssize < capacity)
				ssize <<= 1;
			return ssize;
		}(capacity);
		this->capacity = capacity - 1;
		std::lock_guard<spin_mutex> lck(mtx);
		arr = (T*)Allocator().Malloc(sizeof(T) * capacity);
	}
	LockFreeArrayQueue(SelfType&& v)
		: head(v.head),
		  tail(v.tail),
		  capacity(v.capacity),
		  arr(v.arr) {
		v.arr = nullptr;
	}
	void operator=(SelfType&& v) {
		this->~SelfType();
		new (this) SelfType(std::move(v));
	}
	LockFreeArrayQueue() : LockFreeArrayQueue(64) {}

	template<typename... Args>
	void Push(Args&&... args) {
		std::lock_guard<spin_mutex> lck(mtx);
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)Allocator().Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s != index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(std::move(*ptr));
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			Allocator().Free(arr);
			arr = newArr;
			capacity = newCapa;
		}
		new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
	}
	template<typename... Args>
	bool TryPush(Args&&... args) {
		std::unique_lock<spin_mutex> lck(mtx, std::try_to_lock);
		if (!lck.owns_lock()) return false;
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)Allocator().Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s != index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(std::move(*ptr));
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			Allocator().Free(arr);
			arr = newArr;
			capacity = newCapa;
		}
		new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
		return true;
	}
	bool Pop(T* ptr) {
		constexpr bool isTrivial = std::is_trivially_destructible_v<T>;
		if constexpr (!isTrivial) {
			ptr->~T();
		}

		std::lock_guard<spin_mutex> lck(mtx);
		if (head == tail)
			return false;
		auto&& value = arr[GetIndex(tail++, capacity)];
		if (std::is_trivially_move_assignable_v<T>) {
			*ptr = std::move(value);
		} else {
			new (ptr) T(std::move(value));
		}
		if constexpr (!isTrivial) {
			value.~T();
		}
		return true;
	}
	optional<T> Pop() {
		mtx.lock();
		if (head == tail) {
			mtx.unlock();
			return optional<T>();
		}
		auto value = &arr[GetIndex(tail++, capacity)];
		auto disp = create_disposer([value, this]() {
			if constexpr (!std::is_trivially_destructible_v<T>) {
				value->~T();
			}
			mtx.unlock();
		});
		return optional<T>(std::move(*value));
	}
	optional<T> TryPop() {
		std::unique_lock<spin_mutex> lck(mtx, std::try_to_lock);
		if (!lck.owns_lock()
			|| head == tail) {
			return optional<T>();
		}
		auto value = &arr[GetIndex(tail++, capacity)];
		auto disp = create_disposer([value, this]() {
			if constexpr (!std::is_trivially_destructible_v<T>) {
				value->~T();
			}
		});
		return optional<T>(std::move(*value));
	}
	~LockFreeArrayQueue() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (size_t s = tail; s != head; ++s) {
				arr[GetIndex(s, capacity)].~T();
			}
		}
		Allocator().Free(arr);
	}
	size_t Length() const {
		std::lock_guard<spin_mutex> lck(mtx);
		return head - tail;
	}
};

template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class SingleThreadArrayQueue {
	size_t head;
	size_t tail;
	size_t capacity;
	T* arr;
    using Allocator = VAllocHandle<allocType>;

	static constexpr size_t GetIndex(size_t index, size_t capacity) noexcept {
		return index & capacity;
	}
	using SelfType = SingleThreadArrayQueue<T, allocType>;

public:
	size_t Length() const {
		return tail - head;
	}
	SingleThreadArrayQueue(SelfType&& v)
		: head(v.head),
		  tail(v.tail),
		  capacity(v.capacity),
		  arr(v.arr) {
		v.arr = nullptr;
	}
	SingleThreadArrayQueue(size_t capacity) : head(0), tail(0) {
		capacity = capacity = [](size_t capacity) {
			size_t ssize = 1;
			while (ssize < capacity)
				ssize <<= 1;
			return ssize;
		}(capacity);
		this->capacity = capacity - 1;
		arr = (T*)Allocator().Malloc(sizeof(T) * capacity);
	}
	SingleThreadArrayQueue() : SingleThreadArrayQueue(32) {}

	template<typename... Args>
	void Push(Args&&... args) {
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)Allocator().Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s != index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(std::move(*ptr));
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			Allocator().Free(arr);
			arr = newArr;
			capacity = newCapa;
		}
		new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
	}
	bool Pop(T* ptr) {
		constexpr bool isTrivial = std::is_trivially_destructible_v<T>;
		if constexpr (!isTrivial) {
			ptr->~T();
		}

		if (head == tail)
			return false;
		auto&& value = arr[GetIndex(tail++, capacity)];
		if (std::is_trivially_move_assignable_v<T>) {
			*ptr = std::move(value);
		} else {
			new (ptr) T(std::move(value));
		}
		if constexpr (!isTrivial) {
			value.~T();
		}
		return true;
	}
	~SingleThreadArrayQueue() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (size_t s = tail; s != head; ++s) {
				arr[GetIndex(s, capacity)].~T();
			}
		}
		Allocator().Free(arr);
	}
};
}// namespace vstd