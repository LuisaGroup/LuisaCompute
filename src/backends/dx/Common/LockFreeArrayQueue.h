#pragma once
#include <thread>
#include <Common/MetaLib.h>
#include <Common/Memory.h>
#include <Common/VAllocator.h>
#include <Common/spin_mutex.h>

template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class LockFreeArrayQueue {
	size_t head;
	size_t tail;
	size_t capacity;
	mutable spin_mutex mtx;
	T* arr;
	VAllocHandle<allocType> allocHandle;
	
	static constexpr size_t GetPow2Size(size_t capacity) noexcept {
		size_t ssize = 1;
		while (ssize < capacity)
			ssize <<= 1;
		return ssize;
	}
	static constexpr size_t GetIndex(size_t index, size_t capacity) noexcept {
		return index & capacity;
	}
	using SelfType = LockFreeArrayQueue<T, allocType>;
public:
	LockFreeArrayQueue(size_t capacity) : head(0), tail(0) {
		capacity = GetPow2Size(capacity);
		this->capacity = capacity - 1;
		std::lock_guard<spin_mutex> lck(mtx);
		arr = (T*)allocHandle.Malloc(sizeof(T) * capacity);
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
	LockFreeArrayQueue() : LockFreeArrayQueue(32) {}

	template<typename... Args>
	void Push(Args&&... args) {
		std::lock_guard<spin_mutex> lck(mtx);
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)allocHandle.Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s < index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(*ptr);
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			allocHandle.Free(arr);
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
			T* newArr = (T*)allocHandle.Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s < index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(*ptr);
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			allocHandle.Free(arr);
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
		allocHandle.Free(arr);
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
	VAllocHandle<allocType> allocHandle;

	static constexpr size_t GetPow2Size(size_t capacity) noexcept {
		size_t ssize = 1;
		while (ssize < capacity)
			ssize <<= 1;
		return ssize;
	}
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
		capacity = GetPow2Size(capacity);
		this->capacity = capacity - 1;
		arr = (T*)allocHandle.Malloc(sizeof(T) * capacity);
	}
	SingleThreadArrayQueue() : SingleThreadArrayQueue(32) {}

	template<typename... Args>
	void Push(Args&&... args) {
		size_t index = head++;
		if (head - tail > capacity) {
			auto newCapa = (capacity + 1) * 2;
			T* newArr = (T*)allocHandle.Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s < index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(*ptr);
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			allocHandle.Free(arr);
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
			T* newArr = (T*)allocHandle.Malloc(sizeof(T) * newCapa);
			newCapa--;
			for (size_t s = tail; s < index; ++s) {
				T* ptr = arr + GetIndex(s, capacity);
				new (newArr + GetIndex(s, newCapa)) T(*ptr);
				if constexpr (!std::is_trivially_destructible_v<T>) {
					ptr->~T();
				}
			}
			allocHandle.Free(arr);
			arr = newArr;
			capacity = newCapa;
		}
		new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
	}
	bool Pop() {
		if (head - tail == 0)
			return false;
		auto&& value = arr[GetIndex(tail++, capacity)];
		if constexpr (!std::is_trivially_destructible_v<T>) {
			value.~T();
		}
		return true;
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
	bool GetLast(T* ptr) {
		if (head - tail == 0)
			return false;
		auto&& value = arr[GetIndex(tail, capacity)];
		*ptr = value;
		return true;
	}
	~SingleThreadArrayQueue() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (size_t s = tail; s < head; ++s) {
				arr[GetIndex(s, capacity)].~T();
			}
		}
		allocHandle.Free(arr);
	}
};