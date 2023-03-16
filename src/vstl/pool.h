#pragma once
#include <vstl/config.h>
#include <type_traits>
#include <stdint.h>

#include <atomic>
#include <type_traits>
#include <mutex>
#include <vstl/meta_lib.h>
#include <vstl/memory.h>
#include <vstl/spin_mutex.h>
#include <vstl/vector.h>

namespace vstd {

template<typename T, bool noCheckBeforeDispose = std::is_trivially_destructible<T>::value>
class Pool;

template<typename T>
class Pool<T, true> {

private:
    vector<T *> allPtrs;
    vector<void *> allocatedPtrs;
    size_t capacity;
    static void *PoolMalloc(size_t size) {
        return vengine_malloc(size);
    }
    static void PoolFree(void *ptr) {
        return vengine_free(ptr);
    }
    inline void AllocateMemory() {
        if (!allPtrs.empty()) return;
        using StorageT = Storage<T, 1>;
        StorageT *ptr = reinterpret_cast<StorageT *>(PoolMalloc(sizeof(StorageT) * capacity));
        allPtrs.reserve(capacity + allPtrs.capacity());
        push_back_func(
            allPtrs,
            capacity,
            [&](size_t i) {
                return (T *)(ptr + i);
            });

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
        requires(std::is_constructible_v<T, Args && ...>)
    T *create(Args &&...args) {
        AllocateMemory();
        T *value = allPtrs.back();
        allPtrs.pop_back();
        new (value) T(std::forward<Args>(args)...);
        return value;
    }
    template<typename Mutex, typename... Args>
        requires(std::is_constructible_v<T, Args && ...>)
    T *create_lock(Mutex &mtx, Args &&...args) {
        T *value = nullptr;
        {
            std::lock_guard lck(mtx);
            AllocateMemory();
            value = allPtrs.back();
            allPtrs.pop_back();
        }
        new (value) T(std::forward<Args>(args)...);
        return value;
    }
    void destroy(T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            vstd::destruct(ptr);
        allPtrs.push_back(ptr);
    }
    template<typename Mutex>
    void destroy_lock(Mutex &mtx, T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            vstd::destruct(ptr);
        std::lock_guard lck(mtx);
        allPtrs.push_back(ptr);
    }

    ~Pool() {
        for (auto &&i : allocatedPtrs) {
            PoolFree(i);
        }
    }
};

template<typename T>
class Pool<T, false> {
private:
    struct TypeCollector {
        Storage<T, 1> t;
        size_t index = std::numeric_limits<size_t>::max();
    };
    vector<T *> allPtrs;
    vector<void *> allocatedPtrs;
    vector<TypeCollector *> allocatedObjects;
    size_t capacity;
    static void *PoolMalloc(size_t size) {
        return vengine_malloc(size);
    }
    static void PoolFree(void *ptr) {
        return vengine_free(ptr);
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
            v = allocatedObjects.back();
            allocatedObjects.pop_back();
            v->index = col->index;
        } else {
            allocatedObjects.pop_back();
        }
    }

public:
    struct PoolIterator {
    private:
        TypeCollector **beg;
        TypeCollector **ed;
        Pool const *ptr;

    public:
        PoolIterator(Pool const *ptr) : ptr(ptr) {
            beg = ptr->allocatedObjects.begin();
            ed = ptr->allocatedObjects.end();
        }
        bool operator==(IteEndTag) const {
            return beg == ed;
        }
        T *operator*() const {
            return reinterpret_cast<T *>(&(*beg)->t);
        }
        void operator++() {
            ++beg;
        }
    };
    struct PoolIteratorMother {
        Pool const *ptr;
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
    PoolIteratorMother iterator() const {
        return {this};
    }
    Pool(Pool &&o) = default;
    Pool(Pool const &o) = delete;
    Pool(size_t capa, bool initialize = true) : capacity(capa) {
        if (initialize)
            AllocateMemory();
    }

    template<typename... Args>
        requires(std::is_constructible_v<T, Args && ...>)
    T *create(Args &&...args) {
        AllocateMemory();
        T *value = allPtrs.back();
        allPtrs.pop_back();
        new (value) T(std::forward<Args>(args)...);
        AddAllocatedObject(value);
        return value;
    }
    template<typename Mutex, typename... Args>
        requires(std::is_constructible_v<T, Args && ...>)
    T *create_lock(Mutex &mtx, Args &&...args) {
        T *value = nullptr;
        {
            std::lock_guard lck(mtx);
            AllocateMemory();
            value = allPtrs.back();
            allPtrs.pop_back();
            AddAllocatedObject(value);
        }
        new (value) T(std::forward<Args>(args)...);
        return value;
    }

    void destroy(T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            vstd::destruct(ptr);
        RemoveAllocatedObject(ptr);
        allPtrs.push_back(ptr);
    }
    void destroy_all() {
        for (auto &&ptr : allocatedObjects) {
            allPtrs.push_back(reinterpret_cast<T *>(ptr));
        }
        allocatedObjects.clear();
    }
    template<typename Mutex>
    void destroy_lock(Mutex &mtx, T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            vstd::destruct(ptr);
        std::lock_guard lck(mtx);
        RemoveAllocatedObject(ptr);
        allPtrs.push_back(ptr);
    }

    ~Pool() {
        for (auto &&i : allocatedObjects) {
            vstd::destruct(reinterpret_cast<T *>(i));
        }
        for (auto &&i : allocatedPtrs) {
            PoolFree(i);
        }
    }
};
}// namespace vstd