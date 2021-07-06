#pragma once
#include <thread>
#include <Common/MetaLib.h>
#include <Common/Memory.h>
#include <Common/VAllocator.h>
#include <Common/vector.h>
#include <Common/spin_mutex.h>
#include <utility>

template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class LockFreeArrayQueue {
	size_t head;
	size_t tail;
	size_t capacity;
	mutable spin_mutex mtx;
	VAllocHandle<allocType> allocHandle;
	size_t queueChunkSize;
	size_t queueMod;
	uint8_t rightMoveBit;
	struct Element {
		StackObject<T> obj;
		spin_mutex_init_lock mtx;
	};
	vstd::vector<Element*, allocType> chunks;
	vstd::vector<void*, allocType> allocatedMemory;

	Element* Get(size_t index) {
		size_t chunkIndex = (index & capacity) >> rightMoveBit;
		size_t localIndex = (index & queueMod);
		auto arr = chunks[chunkIndex];
		return arr + localIndex;
	}

	using SelfType = LockFreeArrayQueue<T, allocType>;
	void AddNewElementChunk(size_t count) {

		auto calcAlign = [](auto value, auto align) {
			return (value + (align - 1)) & ~(align - 1);
		};
		auto allocatedSize = queueChunkSize * count;
		Element* allocatedPtr = reinterpret_cast<Element*>(
			allocatedMemory.emplace_back(allocHandle.Malloc(
				allocatedSize * sizeof(Element))));
		for (auto&& i : vstd::ptr_range(allocatedPtr, allocatedPtr + allocatedSize)) {
			new (&i) Element();
		}

		chunks.reserve(chunks.size() + count);

		for (auto& i : vstd::range(count)) {
			auto newPtr = allocatedPtr;
			allocatedPtr += queueChunkSize;
			chunks.emplace_back(newPtr);
		}
	}

public:
	LockFreeArrayQueue(size_t capacity)
		: head(0), tail(0) {
		queueChunkSize =
			[&](size_t capacity) {
				if (capacity < 64) capacity = 64;
				rightMoveBit = 6;
				size_t ssize = 64;
				while (ssize < capacity) {
					ssize <<= 1;
					rightMoveBit++;
				}
				return ssize;
			}(capacity);

		this->capacity = queueChunkSize - 1;
		queueMod = this->capacity;
		AddNewElementChunk(1);
	}
	LockFreeArrayQueue(SelfType&& v)
		: head(v.head),
		  tail(v.tail),
		  capacity(v.capacity),
		  queueChunkSize(v.queueChunkSize),
		  queueMod(v.queueMod),
		  rightMoveBit(v.rightMoveBit),
		  chunks(std::move(v.chunks)),
		  allocatedMemory(std::move(allocatedMemory)) {
		v.head = 0;
		v.tail = 0;
	}
	void operator=(SelfType&& v) {
		this->~SelfType();
		new (this) SelfType(std::move(v));
	}
	LockFreeArrayQueue() : LockFreeArrayQueue(0) {}

	template<typename... Args>
	void Push(Args&&... args) {
		size_t index;
		Element* ele;
		spin_mutex* localMtx;
		{
			std::lock_guard<spin_mutex> lck(mtx);
			index = head++;
			if ((head - tail) > capacity) {
				auto newCapa = (capacity + 1) * 2;
				AddNewElementChunk(chunks.size());
				capacity = newCapa - 1;
			}
			ele = Get(index);
		}
		ele->obj.New(std::forward<Args>(args)...);
		ele->mtx.unlock();
	}
	bool Pop(T* ptr) {
		constexpr bool isTrivial = std::is_trivially_destructible_v<T>;
		if constexpr (!isTrivial) {
			ptr->~T();
		}
		Element* ele;
		{
			std::lock_guard<spin_mutex> lck(mtx);
			if (head == tail)
				return false;
			ele = Get(tail++);
		}
		ele->mtx.lock();
		if (std::is_trivially_move_assignable_v<T>) {
			*ptr = std::move(*ele->obj);
		} else {
			new (ptr) T(std::move(*ele->obj));
		}
		return true;
	}
	bool Pop(vstd::optional<T>& ptr) {
		if (ptr.initialized) {
			ptr.stackObj.Delete();
		}else ptr.initialized = true;
		Element* ele;
		{
			std::lock_guard<spin_mutex> lck(mtx);
			if (head == tail)
				return false;
			ele = Get(tail++);
		}
		ele->mtx.lock();
		ptr.stackObj.New(std::move(*ele->obj));
		return true;
	}
	bool Pop() {
		Element* ele;
		{
			std::lock_guard<spin_mutex> lck(mtx);
			if (head == tail)
				return false;
			ele = Get(tail++);
		}
		ele->mtx.lock();
		ele->obj.Delete();
		return true;
	}
	~LockFreeArrayQueue() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			for (size_t s = tail; s != head; ++s) {
				Element* ele = Get(s);
				ele->mtx.lock();
				ele->obj.Delete();
			}
		}
		for (auto& i : allocatedMemory) {
			allocHandle.Free(i);
		}
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
			for (size_t s = tail; s != index; ++s) {
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
			for (size_t s = tail; s != index; ++s) {
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
	bool Pop(vstd::optional<T>& ptr) {
		ptr.Delete();
		if (head == tail)
			return false;
		auto&& value = arr[GetIndex(tail++, capacity)];
		ptr.New(std::move(value));
		if constexpr (!std::is_trivially_destructible_v<T>) {
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
		allocHandle.Free(arr);
	}
};