#pragma once
#include <VEngineConfig.h>
#include <type_traits>
#include <stdint.h>

#include <atomic>
#include <type_traits>
#include <mutex>
#include "MetaLib.h"
#include "vector.h"
#include "Memory.h"
#include "RandomVector.h"
class PoolBase {
public:
	virtual void Delete(void* ptr) = 0;
	virtual void Delete_Lock(std::mutex& mtx, void* ptr) = 0;
	virtual ~PoolBase() {}
};

template<typename T, bool useVEngineMalloc = true, bool isTrivially = std::is_trivially_destructible<T>::value>
class Pool;

template<typename T, bool useVEngineMalloc>
class Pool<T, useVEngineMalloc, true> {
private:
	ArrayList<T*, useVEngineMalloc> allPtrs;
	ArrayList<void*, useVEngineMalloc> allocatedPtrs;
	size_t capacity;

	void* PoolMalloc(size_t size) {
		if constexpr (useVEngineMalloc) {
			return vengine_malloc(size);
		} else
			return malloc(size);
	}
	void PoolFree(void* ptr) {
		if constexpr (useVEngineMalloc) {
			return vengine_free(ptr);
		} else
			free(ptr);
	}
	inline void AllocateMemory() {
		if (!allPtrs.empty()) return;
		using StorageT = Storage<T, 1>;
		StorageT* ptr = reinterpret_cast<StorageT*>(PoolMalloc(sizeof(StorageT) * capacity));
		allPtrs.reserve(capacity + allPtrs.capacity());
		allPtrs.resize(capacity);
		for (size_t i = 0; i < capacity; ++i) {
			allPtrs[i] = reinterpret_cast<T*>(ptr + i);
		}
		allocatedPtrs.push_back(ptr);
		capacity = capacity * 2;
	}

public:
	Pool(Pool<T, useVEngineMalloc, true>&& o)
		: allPtrs(std::move(o.allPtrs)),
		  allocatedPtrs(std::move(o.allocatedPtrs)),
		  capacity(o.capacity) {
		o.capacity = 0;
	}
	 Pool(size_t capa, bool initialize = true) : capacity(capa) {
		if (initialize)
			AllocateMemory();
	}

	template<typename... Args>
	 T* New(Args&&... args) {
		AllocateMemory();
		T* value = allPtrs.erase_last();
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (value) T(std::forward<Args>(args)...);
		return value;
	}
	template<typename... Args>
	 T* PlaceNew(Args&&... args) {
		AllocateMemory();
		T* value = allPtrs.erase_last();
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (value) T{std::forward<Args>(args)...};
		return value;
	}
	template<typename... Args>
	T* New_Lock(std::mutex& mtx, Args&&... args) {
		T* value = nullptr;
		{
			std::lock_guard<std::mutex> lck(mtx);
			AllocateMemory();
			value = allPtrs.erase_last();
		}
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (value) T(std::forward<Args>(args)...);
		return value;
	}
	template<typename... Args>
	T* PlaceNew_Lock(std::mutex& mtx, Args&&... args) {
		T* value = nullptr;
		{
			std::lock_guard<std::mutex> lck(mtx);
			AllocateMemory();
			value = allPtrs.erase_last();
		}
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (value) T{std::forward<Args>(args)...};
		return value;
	}

	void Delete(void* pp) {
		T* ptr = (T*)pp;
		if constexpr (!std::is_trivially_destructible_v<T>)
			ptr->~T();
		allPtrs.push_back(ptr);
	}
	void Delete_Lock(std::mutex& mtx, void* pp) {
		T* ptr = (T*)pp;
		if constexpr (!std::is_trivially_destructible_v<T>)
			ptr->~T();
		std::lock_guard<std::mutex> lck(mtx);
		allPtrs.push_back(ptr);
	}

	void DeleteWithoutDestructor(void* pp) {
		T* ptr = (T*)pp;
		allPtrs.push_back(ptr);
	}

	~Pool() {
		for (size_t i = 0; i < allocatedPtrs.size(); ++i) {
			PoolFree(allocatedPtrs[i]);
		}
	}
};

template<typename T, bool useVEngineMalloc>
class Pool<T, useVEngineMalloc, false> {
private:
	struct TypeCollector {
		Storage<T, 1> t;
		uint index = -1;
	};
	ArrayList<T*, useVEngineMalloc> allPtrs;
	ArrayList<void*, useVEngineMalloc> allocatedPtrs;
	RandomVector<TypeCollector*, true, useVEngineMalloc> allocatedObjects;
	size_t capacity;
	void* PoolMalloc(size_t size) {
		if constexpr (useVEngineMalloc) {
			return vengine_malloc(size);
		} else
			return malloc(size);
	}
	void PoolFree(void* ptr) {
		if constexpr (useVEngineMalloc) {
			return vengine_free(ptr);
		} else
			free(ptr);
	}
	inline void AllocateMemory() {
		if (!allPtrs.empty()) return;
		TypeCollector* ptr = reinterpret_cast<TypeCollector*>(PoolMalloc(sizeof(TypeCollector) * capacity));
		allPtrs.reserve(capacity + allPtrs.capacity());
		allPtrs.resize(capacity);
		for (size_t i = 0; i < capacity; ++i) {
			allPtrs[i] = reinterpret_cast<T*>(ptr + i);
		}
		allocatedPtrs.push_back(ptr);
		capacity = capacity * 2;
	}
	void AddAllocatedObject(T* obj) {
		TypeCollector* col = reinterpret_cast<TypeCollector*>(obj);
		allocatedObjects.Add(col, &col->index);
	}
	void RemoveAllocatedObject(T* obj) {
		TypeCollector* col = reinterpret_cast<TypeCollector*>(obj);
		allocatedObjects.Remove(col->index);
	}

public:
	Pool(Pool<T, useVEngineMalloc, false>&& o)
		: allPtrs(std::move(o.allPtrs)),
		  allocatedPtrs(std::move(o.allocatedPtrs)),
		  capacity(o.capacity),
		  allocatedObjects(std::move(o.allocatedObjects)) {
		o.capacity = 0;
	}
	 Pool(size_t capa, bool initialize = true) : capacity(capa) {
		if (initialize)
			AllocateMemory();
	}

	template<typename... Args>
	 T* New(Args&&... args) {
		AllocateMemory();
		T* value = allPtrs.erase_last();
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (value) T(std::forward<Args>(args)...);
		AddAllocatedObject(value);
		return value;
	}
	template<typename... Args>
	 T* PlaceNew(Args&&... args) {
		AllocateMemory();
		T* value = allPtrs.erase_last();
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (value) T{std::forward<Args>(args)...};
		AddAllocatedObject(value);
		return value;
	}
	template<typename... Args>
	T* New_Lock(std::mutex& mtx, Args&&... args) {
		T* value = nullptr;
		{
			std::lock_guard<std::mutex> lck(mtx);
			AllocateMemory();
			value = allPtrs.erase_last();
			AddAllocatedObject(value);
		}
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (value) T(std::forward<Args>(args)...);
		return value;
	}
	template<typename... Args>
	T* PlaceNew_Lock(std::mutex& mtx, Args&&... args) {
		T* value = nullptr;
		{
			std::lock_guard<std::mutex> lck(mtx);
			AllocateMemory();
			value = allPtrs.erase_last();
			AddAllocatedObject(value);
		}
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (value) T{std::forward<Args>(args)...};
		return value;
	}

	void Delete(void* pp) {
		T* ptr = (T*)pp;
		if constexpr (!std::is_trivially_destructible_v<T>)
			ptr->~T();
		RemoveAllocatedObject(ptr);
		allPtrs.push_back(ptr);
	}
	void Delete_Lock(std::mutex& mtx, void* pp) {
		T* ptr = (T*)pp;
		if constexpr (!std::is_trivially_destructible_v<T>)
			ptr->~T();
		std::lock_guard<std::mutex> lck(mtx);
		RemoveAllocatedObject(ptr);
		allPtrs.push_back(ptr);
	}

	void DeleteWithoutDestructor(void* pp) {
		T* ptr = (T*)pp;
		allPtrs.push_back(ptr);
		RemoveAllocatedObject(ptr);
	}

	~Pool() {
		for (size_t i = 0; i < allocatedObjects.Length(); ++i) {
			(reinterpret_cast<T*>(allocatedObjects[i]))->~T();
		}
		for (size_t i = 0; i < allocatedPtrs.size(); ++i) {
			PoolFree(allocatedPtrs[i]);
		}
	}
};

template<typename T>
class ConcurrentPool {
private:
	typedef Storage<T, 1> StorageT;
	struct Array {
		StorageT** objs;
		std::atomic<int64_t> count;
		int64_t capacity;
	};

	Array unusedObjects[2];
	std::mutex mtx;
	bool objectSwitcher = true;

public:
	inline void UpdateSwitcher() {
		if (unusedObjects[objectSwitcher].count < 0) unusedObjects[objectSwitcher].count = 0;
		objectSwitcher = !objectSwitcher;
	}

	inline void Delete(T* targetPtr) {
		if constexpr (!std::is_trivially_destructible_v<T>)
			targetPtr->~T();
		Array* arr = unusedObjects + !objectSwitcher;
		int64_t currentCount = arr->count++;
		if (currentCount >= arr->capacity) {
			std::lock_guard<std::mutex> lck(mtx);
			if (currentCount >= arr->capacity) {
				int64_t newCapacity = arr->capacity * 2;
				StorageT** newArray = new StorageT*[newCapacity];
				memcpy(newArray, arr->objs, sizeof(StorageT*) * arr->capacity);
				delete arr->objs;
				arr->objs = newArray;
				arr->capacity = newCapacity;
			}
		}
		arr->objs[currentCount] = (StorageT*)targetPtr;
	}
	template<typename... Args>
	 T* New(Args&&... args) {
		Array* arr = unusedObjects + objectSwitcher;
		int64_t currentCount = --arr->count;
		T* t;
		if (currentCount >= 0) {
			t = (T*)arr->objs[currentCount];

		} else {
			t = (T*)vengine_malloc(sizeof(StorageT));
		}
		if constexpr (!std::is_trivially_constructible_v<T>)
			new (t) T(std::forward<Args>(args)...);
		return t;
	}

	 ConcurrentPool(size_t initCapacity) {
		if (initCapacity < 3) initCapacity = 3;
		unusedObjects[0].objs = new StorageT*[initCapacity];
		unusedObjects[0].capacity = initCapacity;
		unusedObjects[0].count = initCapacity / 2;
		for (size_t i = 0; i < unusedObjects[0].count; ++i) {
			unusedObjects[0].objs[i] = (StorageT*)vengine_malloc(sizeof(StorageT));
		}
		unusedObjects[1].objs = new StorageT*[initCapacity];
		unusedObjects[1].capacity = initCapacity;
		unusedObjects[1].count = initCapacity / 2;
		for (size_t i = 0; i < unusedObjects[1].count; ++i) {
			unusedObjects[1].objs[i] = (StorageT*)vengine_malloc(sizeof(StorageT));
		}
	}
	~ConcurrentPool() {
		for (int64_t i = 0; i < unusedObjects[0].count; ++i) {
			delete unusedObjects[0].objs[i];
		}
		delete unusedObjects[0].objs;
		for (int64_t i = 0; i < unusedObjects[1].count; ++i) {
			delete unusedObjects[1].objs[i];
		}
		delete unusedObjects[1].objs;
	}
};

template<typename T>
class JobPool {
private:
	ArrayList<T*> allocatedPool;
	ArrayList<T*> list[2];
	spin_mutex mtx;
	bool switcher = false;
	uint32_t capacity;
	void ReserveList(ArrayList<T*>& vec) {
		T* t = new T[capacity];
		allocatedPool.push_back(t);
		vec.resize(capacity);
		for (uint32_t i = 0; i < capacity; ++i) {
			vec[i] = t + i;
		}
	}

public:
	JobPool(uint32_t capacity) : capacity(capacity) {
		allocatedPool.reserve(10);
		list[0].reserve(capacity * 2);
		list[1].reserve(capacity * 2);
		ReserveList(list[0]);
		ReserveList(list[1]);
	}

	void UpdateSwitcher() {
		switcher = !switcher;
	}

	T* New() {
		ArrayList<T*>& lst = list[switcher];
		if (lst.empty()) ReserveList(lst);
		T* value = lst.erase_last();
		value->Reset();
		return value;
	}

	void Delete(T* value) {
		ArrayList<T*>& lst = list[!switcher];
		value->Dispose();
		std::lock_guard<spin_mutex> lck(mtx);
		lst.push_back(value);
	}

	~JobPool() {
		for (auto ite = allocatedPool.begin(); ite != allocatedPool.end(); ++ite) {
			delete[] * ite;
		}
	}
};