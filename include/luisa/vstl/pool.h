#pragma once
#include <luisa/vstl/config.h>
#include <type_traits>

#include <atomic>
#include <type_traits>
#include <mutex>
#include <luisa/core/stl/type_traits.h>
#include <luisa/vstl/meta_lib.h>
#include <luisa/vstl/memory.h>
#include <luisa/vstl/vector.h>

namespace vstd {

template<typename T, bool noCheckBeforeDispose = std::is_trivially_destructible<T>::value>
class Pool;

template<typename T>
class Pool<T, true> {

private:
    // Freed slots are threaded into an intrusive singly-linked free list that
    // lives inside the slots themselves. That needs every slot to be large and
    // aligned enough to hold a link; otherwise fall back to the vector stack.
    static constexpr bool useIntrusiveFreeList = sizeof(T) >= sizeof(void *) && alignof(T) >= alignof(void *);
    vector<T *> allPtrs;// free stack used by the fallback path only
    T *freeHead = nullptr;// head of the intrusive free list (nullptr when empty)
    vector<std::pair<void *, size_t>> allocatedPtrs;
    size_t capacity;
    static void *PoolMalloc(size_t size) {
        return vengine_malloc(size);
    }
    static void PoolFree(void *ptr) {
        return vengine_free(ptr);
    }
    inline void AllocateMemory() {
        if constexpr (useIntrusiveFreeList) {
            if (freeHead) return;
        } else {
            if (!allPtrs.empty()) return;
        }
        using StorageT = Storage<T, 1>;
        StorageT *ptr = reinterpret_cast<StorageT *>(PoolMalloc(sizeof(StorageT) * capacity));
        if constexpr (useIntrusiveFreeList) {
            for (size_t i = 0; i < capacity; ++i) {
                T *slot = reinterpret_cast<T *>(ptr + i);
                *reinterpret_cast<T **>(slot) = freeHead;
                freeHead = slot;
            }
        } else {
            allPtrs.reserve(capacity + allPtrs.capacity());
            push_back_func(
                allPtrs,
                capacity,
                [&](size_t i) {
                    return (T *)(ptr + i);
                });
        }
        allocatedPtrs.emplace_back(ptr, capacity);
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
        requires(luisa::is_constructible_v<T, Args && ...>)
    T *create(Args &&...args) {
        AllocateMemory();
        T *value;
        if constexpr (useIntrusiveFreeList) {
            value = freeHead;
            freeHead = *reinterpret_cast<T **>(value);
        } else {
            value = allPtrs.back();
            allPtrs.pop_back();
        }
        new (value) T(std::forward<Args>(args)...);
        return value;
    }
    void destroy_all() {
        if constexpr (useIntrusiveFreeList) {
            freeHead = nullptr;
            for (auto &i : allocatedPtrs) {
                using StorageT = Storage<T, 1>;
                auto ptr = reinterpret_cast<StorageT *>(i.first);
                for (size_t idx = 0; idx < i.second; ++idx) {
                    T *slot = reinterpret_cast<T *>(ptr + idx);
                    *reinterpret_cast<T **>(slot) = freeHead;
                    freeHead = slot;
                }
            }
        } else {
            allPtrs.clear();
            for (auto &i : allocatedPtrs) {
                using StorageT = Storage<T, 1>;
                auto ptr = reinterpret_cast<StorageT *>(i.first);
                push_back_func(
                    allPtrs,
                    i.second,
                    [&](size_t idx) {
                        return (T *)(ptr + idx);
                    });
            }
        }
    }
    template<typename Mutex, typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    T *create_lock(Mutex &mtx, Args &&...args) {
        T *value = nullptr;
        {
            std::lock_guard lck(mtx);
            AllocateMemory();
            if constexpr (useIntrusiveFreeList) {
                value = freeHead;
                freeHead = *reinterpret_cast<T **>(value);
            } else {
                value = allPtrs.back();
                allPtrs.pop_back();
            }
        }
        new (value) T(std::forward<Args>(args)...);
        return value;
    }
    void destroy(T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            std::destroy_at(ptr);
        if constexpr (useIntrusiveFreeList) {
            *reinterpret_cast<T **>(ptr) = freeHead;
            freeHead = ptr;
        } else {
            allPtrs.push_back(ptr);
        }
    }
    template<typename Mutex>
    void destroy_lock(Mutex &mtx, T *ptr) {
        if constexpr (!std::is_trivially_destructible_v<T>)
            std::destroy_at(ptr);
        std::lock_guard lck(mtx);
        if constexpr (useIntrusiveFreeList) {
            *reinterpret_cast<T **>(ptr) = freeHead;
            freeHead = ptr;
        } else {
            allPtrs.push_back(ptr);
        }
    }

    ~Pool() {
        for (auto &&i : allocatedPtrs) {
            PoolFree(i.first);
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
    // Freed slots are threaded into an intrusive free list living inside the
    // slots themselves, unless a TypeCollector slot cannot hold a void* link
    // (in which case the vector free stack below is used instead).
    static constexpr bool useIntrusiveFreeList = sizeof(TypeCollector) >= sizeof(void *) && alignof(TypeCollector) >= alignof(void *);
    vector<T *> allPtrs;// free stack used by the fallback path only
    T *freeHead = nullptr;// head of the intrusive free list (nullptr when empty)
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
        if constexpr (useIntrusiveFreeList) {
            if (freeHead) return;
        } else {
            if (!allPtrs.empty()) return;
        }
        TypeCollector *ptr = reinterpret_cast<TypeCollector *>(PoolMalloc(sizeof(TypeCollector) * capacity));
        if constexpr (useIntrusiveFreeList) {
            for (size_t i = 0; i < capacity; ++i) {
                T *slot = reinterpret_cast<T *>(ptr + i);
                *reinterpret_cast<T **>(slot) = freeHead;
                freeHead = slot;
            }
        } else {
            allPtrs.reserve(capacity + allPtrs.capacity());
            allPtrs.resize(capacity);
            for (size_t i = 0; i < capacity; ++i) {
                allPtrs[i] = reinterpret_cast<T *>(ptr + i);
            }
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
            auto v = allocatedObjects.back();
            allocatedObjects[col->index] = v;
            v->index = col->index;
        }
        allocatedObjects.pop_back();
    }

public:
    struct PoolIterator {
    private:
        typename vector<TypeCollector *>::const_iterator beg;
        typename vector<TypeCollector *>::const_iterator ed;
        Pool const *ptr;

    public:
        PoolIterator(Pool const *ptr) : ptr(ptr) {
            beg = ptr->allocatedObjects.cbegin();
            ed = ptr->allocatedObjects.cend();
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
        requires(luisa::is_constructible_v<T, Args && ...>)
    T *create(Args &&...args) {
        AllocateMemory();
        T *value;
        if constexpr (useIntrusiveFreeList) {
            value = freeHead;
            freeHead = *reinterpret_cast<T **>(value);
        } else {
            value = allPtrs.back();
            allPtrs.pop_back();
        }
        new (value) T(std::forward<Args>(args)...);
        AddAllocatedObject(value);
        return value;
    }
    template<typename Mutex, typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    T *create_lock(Mutex &mtx, Args &&...args) {
        T *value = nullptr;
        {
            std::lock_guard lck(mtx);
            AllocateMemory();
            if constexpr (useIntrusiveFreeList) {
                value = freeHead;
                freeHead = *reinterpret_cast<T **>(value);
            } else {
                value = allPtrs.back();
                allPtrs.pop_back();
            }
            AddAllocatedObject(value);
        }
        new (value) T(std::forward<Args>(args)...);
        return value;
    }

    void destroy(T *ptr) {
        RemoveAllocatedObject(ptr);
        if constexpr (!std::is_trivially_destructible_v<T>)
            std::destroy_at(ptr);
        if constexpr (useIntrusiveFreeList) {
            *reinterpret_cast<T **>(ptr) = freeHead;
            freeHead = ptr;
        } else {
            allPtrs.push_back(ptr);
        }
    }
    void destroy_all() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (auto &&ptr : allocatedObjects) {
                std::destroy_at(reinterpret_cast<T *>(ptr));
            }
        }
        if constexpr (useIntrusiveFreeList) {
            for (auto &&ptr : allocatedObjects) {
                T *slot = reinterpret_cast<T *>(ptr);
                *reinterpret_cast<T **>(slot) = freeHead;
                freeHead = slot;
            }
        } else {
            vstd::push_back_all(
                allPtrs,
                reinterpret_cast<T **>(allocatedObjects.data()),
                allocatedObjects.size());
        }
        allocatedObjects.clear();
    }
    template<typename Mutex>
    void destroy_lock(Mutex &mtx, T *ptr) {
        std::lock_guard lck(mtx);
        RemoveAllocatedObject(ptr);
        if constexpr (!std::is_trivially_destructible_v<T>)
            std::destroy_at(ptr);
        if constexpr (useIntrusiveFreeList) {
            *reinterpret_cast<T **>(ptr) = freeHead;
            freeHead = ptr;
        } else {
            allPtrs.push_back(ptr);
        }
    }

    ~Pool() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (auto &&i : allocatedObjects) {
                std::destroy_at(reinterpret_cast<T *>(i));
            }
        }
        for (auto &&i : allocatedPtrs) {
            PoolFree(i);
        }
    }
};
}// namespace vstd
