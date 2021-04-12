#pragma once
#include <thread>
#include <Common/MetaLib.h>
#include <Common/Memory.h>

template<typename T, bool useVAlloc = true>
class LockFreeArrayQueue {
	size_t head;
	size_t tail;
	size_t capacity;
	spin_mutex mtx;
	T* arr;
	static inline void* Malloc(size_t i) noexcept {
		if constexpr (useVAlloc) {
			return vengine_malloc(i);
		} else
			return malloc(i);
	}
	static inline void Free(void* ptr) noexcept {
		if (!ptr) return;
		if constexpr (useVAlloc) {
			vengine_free(ptr);
		} else
			free(ptr);
	}
	static constexpr size_t GetPow2Size(size_t capacity) noexcept {
		size_t ssize = 1;
		while (ssize < capacity)
			ssize <<= 1;
		return ssize;
	}
	static constexpr size_t GetIndex(size_t index, size_t capacity) noexcept {
		return index & capacity;
	}

public:
	LockFreeArrayQueue(size_t capacity) : head(0), tail(0) {
		capacity = GetPow2Size(capacity);
		this->capacity = capacity - 1;
		arr = (T*)Malloc(sizeof(T) * capacity);
	}
	LockFreeArrayQueue(LockFreeArrayQueue<T, useVAlloc>&& v)
		: head(v.head),
		  tail(v.tail),
		  capacity(v.capacity),
		  arr(v.arr) {
		v.arr = nullptr;
	}
	void operator=(LockFreeArrayQueue<T, useVAlloc>&& v) {
		this->~LockFreeArrayQueue();
		new (this) LockFreeArrayQueue(std::move(v));
	}
	LockFreeArrayQueue() : LockFreeArrayQueue(32) {}

	template<typename... Args>
	void Push(Args&&... args) {
		std::lock_guard<spin_mutex> lck(mtx);
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s < index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(*ptr);
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			Free(arr);
			arr = newArr;
			capacity = newCapa;
		}
		new (arr + GetIndex(index, capacity)) T(std::forward<Args>(args)...);
	}
	template<typename... Args>
	void PushInPlaceNew(Args&&... args) {
		std::lock_guard<spin_mutex> lck(mtx);
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s < index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(*ptr);
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			Free(arr);
			arr = newArr;
			capacity = newCapa;
		}
		new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
	}
	bool Pop(T* ptr) {
		std::lock_guard<spin_mutex> lck(mtx);
		if (head - tail == 0)
			return false;
		auto&& value = arr[GetIndex(tail++, capacity)];
		constexpr bool isTrivial = std::is_trivially_destructible_v<T>;
		if constexpr (!isTrivial) {
			ptr->~T();
		}
		new (ptr) T(std::move(value));
		if constexpr (!isTrivial) {
			value.~T();
		}
		return true;
	}
	bool Pop() {
		std::lock_guard<spin_mutex> lck(mtx);
		if (head - tail == 0)
			return false;
		auto&& value = arr[GetIndex(tail++, capacity)];
		if constexpr (!std::is_trivially_destructible_v<T>) {
			value.~T();
		}
		return true;
	}
	~LockFreeArrayQueue() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (size_t s = tail; s < head; ++s) {
				arr[GetIndex(s, capacity)].~T();
			}
		}
		Free(arr);
	}
	size_t Length() const {
		return head - tail;
	}
};

template<typename T, bool useVAlloc = true>
class SingleThreadArrayQueue {
	size_t head;
	size_t tail;
	size_t capacity;
	T* arr;
	static inline void* Malloc(size_t i) noexcept {
		if constexpr (useVAlloc) {
			return vengine_malloc(i);
		} else
			return malloc(i);
	}
	static inline void Free(void* ptr) noexcept {
		if (!ptr) return;
		if constexpr (useVAlloc) {
			vengine_free(ptr);
		} else
			free(ptr);
	}
	static constexpr size_t GetPow2Size(size_t capacity) noexcept {
		size_t ssize = 1;
		while (ssize < capacity)
			ssize <<= 1;
		return ssize;
	}
	static constexpr size_t GetIndex(size_t index, size_t capacity) noexcept {
		return index & capacity;
	}

public:
	size_t Length() const {
		return tail - head;
	}
	SingleThreadArrayQueue(SingleThreadArrayQueue<T, useVAlloc>&& v)
		: head(v.head),
		  tail(v.tail),
		  capacity(v.capacity),
		  arr(v.arr) {
		v.arr = nullptr;
	}
	SingleThreadArrayQueue(size_t capacity) : head(0), tail(0) {
		capacity = GetPow2Size(capacity);
		this->capacity = capacity - 1;
		arr = (T*)Malloc(sizeof(T) * capacity);
	}
	SingleThreadArrayQueue() : SingleThreadArrayQueue(32) {}

	template<typename... Args>
	void Push(Args&&... args) {
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s < index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(*ptr);
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			Free(arr);
			arr = newArr;
			capacity = newCapa;
		}
		new (arr + GetIndex(index, capacity)) T(std::forward<Args>(args)...);
	}
	template<typename... Args>
	void PushInPlaceNew(Args&&... args) {
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s < index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(*ptr);
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			Free(arr);
			arr = newArr;
			capacity = newCapa;
		}
		new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
	}
	bool Pop(T* ptr) {
		if (head - tail == 0)
			return false;
		auto&& value = arr[GetIndex(tail++, capacity)];
		*ptr = value;
		if constexpr (!std::is_trivially_destructible_v<T>) {
			value.~T();
		}
		return true;
	}
	~SingleThreadArrayQueue() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (size_t s = tail; s < head; ++s) {
				arr[GetIndex(s, capacity)].~T();
			}
		}
		Free(arr);
	}
};