#pragma once

#include <type_traits>
#include <cstdint>
#include <atomic>
#include <type_traits>
#include <mutex>
#include <vector>

#include <core/spin_mutex.h>
#include <vstl/config.h>
#include <vstl/MetaLib.h>
#include <vstl/Memory.h>
#include <vstl/VAllocator.h>

namespace vstd {

template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine, bool noCheckBeforeDispose = std::is_trivially_destructible<T>::value>
class Pool;
template<typename Vec>
decltype(auto) VectorEraseLast(Vec &&vec) {
    auto ite = vec.end() - 1;
    auto value = std::move(*ite);
    vec.erase(ite);
    return value;
}

template<typename T, VEngine_AllocType allocType>
class Pool<T, allocType, true> : public vstd::IOperatorNewBase {
private:
    std::vector<T *> allPtrs;
    std::vector<void *> allocatedPtrs;
    size_t capacity;
    VAllocHandle<allocType> allocHandle;
    void *PoolMalloc(size_t size) {
        return allocHandle.Malloc(size);
    }
    void PoolFree(void *ptr) {
        return allocHandle.Free(ptr);
    }
    inline void AllocateMemory() {
        if (!allPtrs.empty()) return;
        using StorageT = Storage<T, 1>;
        StorageT *ptr = reinterpret_cast<StorageT *>(PoolMalloc(sizeof(StorageT) * capacity));
        allPtrs.reserve(capacity + allPtrs.capacity());
        allPtrs.resize(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            allPtrs[i] = reinterpret_cast<T *>(ptr + i);
        }
        allocatedPtrs.push_back(ptr);
        capacity = capacity * 2;
    }

public:
    Pool(size_t capa, bool initialize = true) : capacity(capa) {
        if (initialize)
            AllocateMemory();
    }
    Pool(Pool &&o) = default;
    Pool(Pool const &o) = delete;
    template<typename... Args>
    T *New(Args &&...args) {
        AllocateMemory();
        T *value = VectorEraseLast(allPtrs);
        if constexpr (!std::is_trivially_constructible_v<T>)
            new (value) T(std::forward<Args>(args)...);
        return value;
    }
    template<typename... Args>
    T *PlaceNew(Args &&...args) {
        AllocateMemory();
        T *value = VectorEraseLast(allPtrs);
        if constexpr (!std::is_trivially_constructible_v<T>)
            new (value) T{std::forward<Args>(args)...};
        return value;
    }
    template<typename Mutex, typename... Args>
    T *New_Lock(Mutex &mtx, Args &&...args) {
        T *value = nullptr;
        {
            std::lock_guard lck(mtx);
            AllocateMemory();
            value = VectorEraseLast(allPtrs);
        }
        if constexpr (!std::is_trivially_constructible_v<T>)
            new (value) T(std::forward<Args>(args)...);
        return value;
    }
    template<typename Mutex, typename... Args>
    T *PlaceNew_Lock(Mutex &mtx, Args &&...args) {
        T *value = nullptr;
        {
            std::lock_guard lck(mtx);
            AllocateMemory();
            value = VectorEraseLast(allPtrs);
        }
        if constexpr (!std::is_trivially_constructible_v<T>)
            new (value) T{std::forward<Args>(args)...};
        return value;
    }

    void Delete(T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            ptr->~T();
        allPtrs.push_back(ptr);
    }
    template<typename Mutex>
    void Delete_Lock(Mutex &mtx, T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            ptr->~T();
        std::lock_guard lck(mtx);
        allPtrs.push_back(ptr);
    }

    void DeleteWithoutDestructor(void *pp) {
        T *ptr = (T *)pp;
        allPtrs.push_back(ptr);
    }

    ~Pool() {
        for (auto &&i : allocatedPtrs) {
            PoolFree(i);
        }
    }
};

template<typename T, VEngine_AllocType allocType>
class Pool<T, allocType, false> : public vstd::IOperatorNewBase {
private:
    struct TypeCollector {
        Storage<T, 1> t;
        size_t index = std::numeric_limits<size_t>::max();
    };
    std::vector<T *> allPtrs;
    std::vector<void *> allocatedPtrs;
    std::vector<TypeCollector *> allocatedObjects;
    size_t capacity;
    VAllocHandle<allocType> allocHandle;
    void *PoolMalloc(size_t size) {
        return allocHandle.Malloc(size);
    }
    void PoolFree(void *ptr) {
        return allocHandle.Free(ptr);
    }
    inline void AllocateMemory() {
        if (!allPtrs.empty()) return;
        TypeCollector *ptr = reinterpret_cast<TypeCollector *>(PoolMalloc(sizeof(TypeCollector) * capacity));
        allPtrs.reserve(capacity + allPtrs.capacity());
        allPtrs.resize(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            allPtrs[i] = reinterpret_cast<T *>(ptr + i);
        }
        allocatedPtrs.push_back(ptr);
        capacity = capacity * 2;
    }
    void AddAllocatedObject(T *obj) {
        TypeCollector *col = reinterpret_cast<TypeCollector *>(obj);
        col->index = allocatedObjects.size();
        allocatedObjects.push_back(col);
    }
    void RemoveAllocatedObject(T *obj) {
        TypeCollector *col = reinterpret_cast<TypeCollector *>(obj);
        if (col->index != allocatedObjects.size() - 1) {
            auto &&v = allocatedObjects[col->index];
            v = allocatedObjects.erase_last();
            v->index = col->index;
        } else {
            allocatedObjects.erase_last();
        }
    }

public:
    Pool(Pool &&o) = default;
    Pool(Pool const &o) = delete;
    Pool(size_t capa, bool initialize = true) : capacity(capa) {
        if (initialize)
            AllocateMemory();
    }

    template<typename... Args>
    T *New(Args &&...args) {
        AllocateMemory();
        T *value = VectorEraseLast(allPtrs);
        if constexpr (!std::is_trivially_constructible_v<T>)
            new (value) T(std::forward<Args>(args)...);
        AddAllocatedObject(value);
        return value;
    }
    template<typename... Args>
    T *PlaceNew(Args &&...args) {
        AllocateMemory();
        T *value = VectorEraseLast(allPtrs);
        if constexpr (!std::is_trivially_constructible_v<T>)
            new (value) T{std::forward<Args>(args)...};
        AddAllocatedObject(value);
        return value;
    }
    template<typename Mutex, typename... Args>
    T *New_Lock(Mutex &mtx, Args &&...args) {
        T *value = nullptr;
        {
            std::lock_guard lck(mtx);
            AllocateMemory();
            value = VectorEraseLast(allPtrs);
            AddAllocatedObject(value);
        }
        if constexpr (!std::is_trivially_constructible_v<T>)
            new (value) T(std::forward<Args>(args)...);
        return value;
    }
    template<typename Mutex, typename... Args>
    T *PlaceNew_Lock(Mutex &mtx, Args &&...args) {
        T *value = nullptr;
        {
            std::lock_guard lck(mtx);
            AllocateMemory();
            value = VectorEraseLast(allPtrs);
            AddAllocatedObject(value);
        }
        if constexpr (!std::is_trivially_constructible_v<T>)
            new (value) T{std::forward<Args>(args)...};
        return value;
    }

    void Delete(T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            ptr->~T();
        RemoveAllocatedObject(ptr);
        allPtrs.push_back(ptr);
    }
    template<typename Mutex>
    void Delete_Lock(Mutex &mtx, T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            ptr->~T();
        std::lock_guard lck(mtx);
        RemoveAllocatedObject(ptr);
        allPtrs.push_back(ptr);
    }

    void DeleteWithoutDestructor(void *pp) {
        T *ptr = (T *)pp;
        allPtrs.push_back(ptr);
        RemoveAllocatedObject(ptr);
    }

    ~Pool() {
        for (auto &&i : allocatedObjects) {
            (reinterpret_cast<T *>(i))->~T();
        }
        for (auto &&i : allocatedPtrs) {
            PoolFree(i);
        }
    }
};
}// namespace vstd