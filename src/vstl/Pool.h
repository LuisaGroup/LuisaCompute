#pragma once
#include <vstl/config.h>
#include <type_traits>
#include <stdint.h>

#include <atomic>
#include <type_traits>
#include <mutex>
#include <vstl/MetaLib.h>
#include <vstl/vector.h>
#include <vstl/Memory.h>
#include <vstl/VAllocator.h>
#include <vstl/spin_mutex.h>
namespace vstd {
template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine, bool noCheckBeforeDispose = std::is_trivially_destructible<T>::value>
class Pool;

template<typename T, VEngine_AllocType allocType>
class Pool<T, allocType, true> : public IOperatorNewBase {
private:
	ArrayList<T*, allocType> allPtrs;
	ArrayList<void*, allocType> allocatedPtrs;
	size_t capacity;
	VAllocHandle<allocType> allocHandle;
	void* PoolMalloc(size_t size) {
		return allocHandle.Malloc(size);
	}
	void PoolFree(void* ptr) {
		return allocHandle.Free(ptr);
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
	Pool(size_t capa, bool initialize = true) : capacity(capa) {
		if (initialize)
			AllocateMemory();
	}
	Pool(Pool&& o) = default;
	Pool(Pool const& o) = delete;
	VAllocHandle<allocType> const& GetAllocator() const { return allocHandle; }
	template<typename... Args>
	T* New(Args&&... args) {
		AllocateMemory();
		T* value = allPtrs.erase_last();
		new (value) T(std::forward<Args>(args)...);
		return value;
	}
	template<typename... Args>
	T* PlaceNew(Args&&... args) {
		AllocateMemory();
		T* value = allPtrs.erase_last();
		new (value) T{std::forward<Args>(args)...};
		return value;
	}
	template<typename Mutex, typename... Args>
	T* New_Lock(Mutex& mtx, Args&&... args) {
		T* value = nullptr;
		{
			std::lock_guard lck(mtx);
			AllocateMemory();
			value = allPtrs.erase_last();
		}
		new (value) T(std::forward<Args>(args)...);
		return value;
	}
	template<typename Mutex, typename... Args>
	T* PlaceNew_Lock(Mutex& mtx, Args&&... args) {
		T* value = nullptr;
		{
			std::lock_guard lck(mtx);
			AllocateMemory();
			value = allPtrs.erase_last();
		}
		new (value) T{std::forward<Args>(args)...};
		return value;
	}

	void Delete(T* ptr) {
		if constexpr (!std::is_trivially_destructible_v<T>)
			ptr->~T();
		allPtrs.push_back(ptr);
	}
	template<typename Mutex>
	void Delete_Lock(Mutex& mtx, T* ptr) {
		if constexpr (!std::is_trivially_destructible_v<T>)
			ptr->~T();
		std::lock_guard lck(mtx);
		allPtrs.push_back(ptr);
	}

	void DeleteWithoutDestructor(void* pp) {
		T* ptr = (T*)pp;
		allPtrs.push_back(ptr);
	}

	~Pool() {
		for (auto&& i : allocatedPtrs) {
			PoolFree(i);
		}
	}
};

template<typename T, VEngine_AllocType allocType>
class Pool<T, allocType, false> : public IOperatorNewBase {
private:
	struct TypeCollector {
		Storage<T, 1> t;
		size_t index = std::numeric_limits<size_t>::max();
	};
	ArrayList<T*, allocType> allPtrs;
	ArrayList<void*, allocType> allocatedPtrs;
	vector<TypeCollector*, allocType> allocatedObjects;
	size_t capacity;
	VAllocHandle<allocType> allocHandle;
	void* PoolMalloc(size_t size) {
		return allocHandle.Malloc(size);
	}
	void PoolFree(void* ptr) {
		return allocHandle.Free(ptr);
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
		col->index = allocatedObjects.size();
		allocatedObjects.push_back(col);
	}
	void RemoveAllocatedObject(T* obj) {
		TypeCollector* col = reinterpret_cast<TypeCollector*>(obj);
		if (col->index != allocatedObjects.size() - 1) {
			auto&& v = allocatedObjects[col->index];
			v = allocatedObjects.erase_last();
			v->index = col->index;
		} else {
			allocatedObjects.erase_last();
		}
	}

public:
	VAllocHandle<allocType> const& GetAllocator() const { return allocHandle; }
	struct PoolIterator {
	private:
		TypeCollector** beg;
		TypeCollector** ed;
		Pool const* ptr;

	public:
		PoolIterator(Pool const* ptr) : ptr(ptr) {
			beg = ptr->allocatedObjects.begin();
			ed = ptr->allocatedObjects.end();
		}
		bool operator==(IteEndTag) const {
			return beg == ed;
		}
		T* operator*() const {
			return reinterpret_cast<T*>(&(*beg)->t);
		}
		void operator++() {
			++beg;
		}
	};
	struct PoolIteratorMother {
		Pool const* ptr;
		PoolIterator begin() const {
			return PoolIterator(ptr);
		}
		IteEndTag end() const {
			return {};
		}
		size_t size() const {
			return ptr->allocatedObjects.size();
		}
	};
	PoolIteratorMother Iterator() const {
		return {this};
	}
	Pool(Pool&& o) = default;
	Pool(Pool const& o) = delete;
	Pool(size_t capa, bool initialize = true) : capacity(capa) {
		if (initialize)
			AllocateMemory();
	}

	template<typename... Args>
	T* New(Args&&... args) {
		AllocateMemory();
		T* value = allPtrs.erase_last();
		new (value) T(std::forward<Args>(args)...);
		AddAllocatedObject(value);
		return value;
	}
	template<typename... Args>
	T* PlaceNew(Args&&... args) {
		AllocateMemory();
		T* value = allPtrs.erase_last();
		new (value) T{std::forward<Args>(args)...};
		AddAllocatedObject(value);
		return value;
	}
	template<typename Mutex, typename... Args>
	T* New_Lock(Mutex& mtx, Args&&... args) {
		T* value = nullptr;
		{
			std::lock_guard lck(mtx);
			AllocateMemory();
			value = allPtrs.erase_last();
			AddAllocatedObject(value);
		}
		new (value) T(std::forward<Args>(args)...);
		return value;
	}
	template<typename Mutex, typename... Args>
	T* PlaceNew_Lock(Mutex& mtx, Args&&... args) {
		T* value = nullptr;
		{
			std::lock_guard lck(mtx);
			AllocateMemory();
			value = allPtrs.erase_last();
			AddAllocatedObject(value);
		}
		new (value) T{std::forward<Args>(args)...};
		return value;
	}

	void Delete(T* ptr) {
		if constexpr (!std::is_trivially_destructible_v<T>)
			ptr->~T();
		RemoveAllocatedObject(ptr);
		allPtrs.push_back(ptr);
	}
	void DeleteAll() {
		for (auto&& ptr : allocatedObjects) {
			allPtrs.push_back(reinterpret_cast<T*>(ptr));
		}
		allocatedObjects.clear();
	}
	template<typename Mutex>
	void Delete_Lock(Mutex& mtx, T* ptr) {
		if constexpr (!std::is_trivially_destructible_v<T>)
			ptr->~T();
		std::lock_guard lck(mtx);
		RemoveAllocatedObject(ptr);
		allPtrs.push_back(ptr);
	}

	void DeleteWithoutDestructor(void* pp) {
		T* ptr = (T*)pp;
		allPtrs.push_back(ptr);
		RemoveAllocatedObject(ptr);
	}

	~Pool() {
		for (auto&& i : allocatedObjects) {
			(reinterpret_cast<T*>(i))->~T();
		}
		for (auto&& i : allocatedPtrs) {
			PoolFree(i);
		}
	}
};

template<typename T>
class ConcurrentPool : public IOperatorNewBase {
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
			std::lock_guard lck(mtx);
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
class JobPool : public IOperatorNewBase {
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
}// namespace vstd