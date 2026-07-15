#pragma once

#include <thread>
#include <luisa/vstl/meta_lib.h>
#include <luisa/vstl/memory.h>
#include <luisa/vstl/v_allocator.h>
#include <luisa/vstl/spin_mutex.h>

namespace vstd {
template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class LockFreeArrayQueue {
    using Allocator = VAllocHandle<allocType>;
    struct alignas(64) ProducerData {
        std::atomic<size_t> head{0};
        mutable spin_mutex mtx;
    };
    struct alignas(64) ConsumerData {
        std::atomic<size_t> tail{0};
        mutable spin_mutex mtx;
    };
    ProducerData prod;
    ConsumerData cons;
    size_t capacity;
    T *arr;

    static constexpr size_t GetIndex(size_t index, size_t capacity) noexcept {
        return index & capacity;
    }
    using SelfType = LockFreeArrayQueue<T, allocType>;

public:
    LockFreeArrayQueue(size_t capacity) : capacity(0), arr(nullptr) {
        if (capacity < 32) capacity = 32;
        capacity = [](size_t capacity) {
            size_t ssize = 1;
            while (ssize < capacity)
                ssize <<= 1;
            return ssize;
        }(capacity);
        this->capacity = capacity - 1;
        arr = (T *)Allocator().Malloc(sizeof(T) * capacity);
    }
    LockFreeArrayQueue(SelfType &&v)
        : prod{v.prod.head.load(std::memory_order_relaxed)},
          cons{v.cons.tail.load(std::memory_order_relaxed)},
          capacity(v.capacity),
          arr(v.arr) {
        v.arr = nullptr;
    }
    void operator=(SelfType &&v) {
        this->~SelfType();
        new (this) SelfType(std::move(v));
    }
    LockFreeArrayQueue() : LockFreeArrayQueue(64) {}
    void reserve(size_t newCapa) {
        std::lock_guard<spin_mutex> lck_head(prod.mtx);
        std::lock_guard<spin_mutex> lck_tail(cons.mtx);
        size_t h = prod.head.load(std::memory_order_relaxed);
        size_t t = cons.tail.load(std::memory_order_relaxed);
        if (newCapa > capacity) {
            auto newCapa = (capacity + 1) * 2;
            T *newArr = (T *)Allocator().Malloc(sizeof(T) * newCapa);
            newCapa--;
            for (size_t s = t; s != h; ++s) {
                T *ptr = arr + GetIndex(s, capacity);
                new (newArr + GetIndex(s, newCapa)) T(std::move(*ptr));
                std::destroy_at(ptr);
            }
            Allocator().Free(arr);
            arr = newArr;
            capacity = newCapa;
        }
    }
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    void enqueue(Args &&...args) {
        std::lock_guard<spin_mutex> lck(prod.mtx);
        size_t index = prod.head.load(std::memory_order_relaxed);
        size_t new_head = index + 1;
        if (new_head - cons.tail.load(std::memory_order_relaxed) > capacity) {
            std::lock_guard<spin_mutex> lck_tail(cons.mtx);
            size_t t = cons.tail.load(std::memory_order_relaxed);
            if (new_head - t > capacity) {
                auto newCapa = (capacity + 1) * 2;
                T *newArr = (T *)Allocator().Malloc(sizeof(T) * newCapa);
                newCapa--;
                for (size_t s = t; s != index; ++s) {
                    T *ptr = arr + GetIndex(s, capacity);
                    new (newArr + GetIndex(s, newCapa)) T(std::move(*ptr));
                    std::destroy_at(ptr);
                }
                Allocator().Free(arr);
                arr = newArr;
                capacity = newCapa;
            }
        }
        new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
        prod.head.store(new_head, std::memory_order_release);
    }
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    [[deprecated("please use enqueue instead")]] void push(Args &&...args) {
        enqueue(std::forward<Args>(args)...);
    }
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    bool try_push(Args &&...args) {
        std::unique_lock<spin_mutex> lck(prod.mtx, std::try_to_lock);
        if (!lck.owns_lock()) return false;
        size_t index = prod.head.load(std::memory_order_relaxed);
        size_t new_head = index + 1;
        if (new_head - cons.tail.load(std::memory_order_relaxed) > capacity) {
            std::lock_guard<spin_mutex> lck_tail(cons.mtx);
            size_t t = cons.tail.load(std::memory_order_relaxed);
            if (new_head - t > capacity) {
                auto newCapa = (capacity + 1) * 2;
                T *newArr = (T *)Allocator().Malloc(sizeof(T) * newCapa);
                newCapa--;
                for (size_t s = t; s != index; ++s) {
                    T *ptr = arr + GetIndex(s, capacity);
                    new (newArr + GetIndex(s, newCapa)) T(std::move(*ptr));
                    std::destroy_at(ptr);
                }
                Allocator().Free(arr);
                arr = newArr;
                capacity = newCapa;
            }
        }
        new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
        prod.head.store(new_head, std::memory_order_release);
        return true;
    }
    bool pop(T *ptr) {
        std::destroy_at(ptr);
        size_t h = prod.head.load(std::memory_order_acquire);
        size_t t = cons.tail.load(std::memory_order_relaxed);
        if (h == t)
            return false;
        std::lock_guard<spin_mutex> lck(cons.mtx);
        h = prod.head.load(std::memory_order_acquire);
        t = cons.tail.load(std::memory_order_relaxed);
        if (h == t)
            return false;
        auto &&value = arr[GetIndex(t, capacity)];
        cons.tail.store(t + 1, std::memory_order_relaxed);
        if (std::is_trivially_move_assignable_v<T>) {
            *ptr = std::move(value);
        } else {
            new (ptr) T(std::move(value));
        }
        std::destroy_at(std::addressof(value));
        return true;
    }
    optional<T> dequeue() {
        size_t h = prod.head.load(std::memory_order_acquire);
        size_t t = cons.tail.load(std::memory_order_relaxed);
        if (h == t) {
            return optional<T>();
        }
        std::lock_guard<spin_mutex> lck(cons.mtx);
        h = prod.head.load(std::memory_order_acquire);
        t = cons.tail.load(std::memory_order_relaxed);
        if (h == t) {
            return optional<T>();
        }
        auto value = &arr[GetIndex(t, capacity)];
        cons.tail.store(t + 1, std::memory_order_relaxed);
        auto disp = scope_exit([value]() {
            std::destroy_at(value);
        });
        return optional<T>(std::move(*value));
    }
    [[deprecated("please use dequeue instead")]]
    optional<T> pop() {
        return dequeue();
    }
    optional<T> try_pop() {
        size_t h = prod.head.load(std::memory_order_acquire);
        size_t t = cons.tail.load(std::memory_order_relaxed);
        if (h == t) {
            return optional<T>();
        }
        std::unique_lock<spin_mutex> lck(cons.mtx, std::try_to_lock);
        if (!lck.owns_lock()) return optional<T>();
        h = prod.head.load(std::memory_order_acquire);
        t = cons.tail.load(std::memory_order_relaxed);
        if (h == t) {
            return optional<T>();
        }
        auto value = &arr[GetIndex(t, capacity)];
        cons.tail.store(t + 1, std::memory_order_relaxed);
        auto disp = scope_exit([value]() {
            std::destroy_at(value);
        });
        return optional<T>(std::move(*value));
    }
    ~LockFreeArrayQueue() {
        if (!arr) return;
        size_t h = prod.head.load(std::memory_order_relaxed);
        size_t t = cons.tail.load(std::memory_order_relaxed);
        for (size_t s = t; s != h; ++s) {
            std::destroy_at(std::addressof(arr[GetIndex(s, capacity)]));
        }
        Allocator().Free(arr);
    }
    size_t length() const {
        return prod.head.load(std::memory_order_acquire) - cons.tail.load(std::memory_order_acquire);
    }
};

template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class SingleThreadArrayQueue {
    using Allocator = VAllocHandle<allocType>;
    std::atomic_size_t head;
    std::atomic_size_t tail;
    size_t capacity;
    T *arr;

    static constexpr size_t GetIndex(size_t index, size_t capacity) noexcept {
        return index & capacity;
    }
    using SelfType = SingleThreadArrayQueue<T, allocType>;

public:
    SingleThreadArrayQueue(size_t capacity) : head(0), tail(0) {
        if (capacity < 32) capacity = 32;
        capacity = [](size_t capacity) {
            size_t ssize = 1;
            while (ssize < capacity)
                ssize <<= 1;
            return ssize;
        }(capacity);
        this->capacity = capacity - 1;
        arr = (T *)Allocator().Malloc(sizeof(T) * capacity);
    }
    SingleThreadArrayQueue(SelfType &&v)
        : head(v.head.load()),
          tail(v.tail.load()),
          capacity(v.capacity),
          arr(v.arr) {
        v.arr = nullptr;
    }
    void operator=(SelfType &&v) {
        this->~SelfType();
        new (this) SelfType(std::move(v));
    }
    SingleThreadArrayQueue() : SingleThreadArrayQueue(64) {}
    void reserve(size_t newCapa) {
        size_t index = head;
        if (newCapa > capacity) {
            auto newCapa = (capacity + 1) * 2;
            T *newArr = (T *)Allocator().Malloc(sizeof(T) * newCapa);
            newCapa--;
            for (size_t s = tail; s != index; ++s) {
                T *ptr = arr + GetIndex(s, capacity);
                new (newArr + GetIndex(s, newCapa)) T(std::move(*ptr));
                std::destroy_at(ptr);
            }
            Allocator().Free(arr);
            arr = newArr;
            capacity = newCapa;
        }
    }
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    T *enqueue(Args &&...args) {
        size_t index = head++;
        if (head - tail > capacity) {
            auto newCapa = (capacity + 1) * 2;
            T *newArr = (T *)Allocator().Malloc(sizeof(T) * newCapa);
            newCapa--;
            for (size_t s = tail; s != index; ++s) {
                T *ptr = arr + GetIndex(s, capacity);
                new (newArr + GetIndex(s, newCapa)) T(std::move(*ptr));
                std::destroy_at(ptr);
            }
            Allocator().Free(arr);
            arr = newArr;
            capacity = newCapa;
        }
        return new (arr + GetIndex(index, capacity)) T{std::forward<Args>(args)...};
    }
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args && ...>)
    [[deprecated("please use enqueue instead")]] T *push(Args &&...args) {
        return enqueue(std::forward<Args>(args)...);
    }
    T *front() {
        if (head == tail)
            return nullptr;
        auto &&value = arr[GetIndex(tail, capacity)];
        return &value;
    }
    bool pop(T *ptr) {
        std::destroy_at(ptr);
        if (head == tail)
            return false;
        auto &&value = arr[GetIndex(tail++, capacity)];
        if (std::is_trivially_move_assignable_v<T>) {
            *ptr = std::move(value);
        } else {
            new (ptr) T(std::move(value));
        }
        std::destroy_at(std::addressof(value));
        return true;
    }
    optional<T> dequeue() {
        if (head == tail) {
            return optional<T>();
        }
        auto value = &arr[GetIndex(tail++, capacity)];
        auto disp = scope_exit([value]() {
            std::destroy_at(value);
        });
        return optional<T>(std::move(*value));
    }
    [[deprecated("please use dequeue instead")]]
    optional<T> pop() {
        return dequeue();
    }
    void pop_discard() {
        if (head == tail) {
            return;
        }
        auto value = &arr[GetIndex(tail++, capacity)];
        std::destroy_at(value);
    }
    ~SingleThreadArrayQueue() {
        for (size_t s = tail; s != head; ++s) {
            std::destroy_at(std::addressof(arr[GetIndex(s, capacity)]));
        }
        Allocator().Free(arr);
    }
    size_t length() const {
        return head - tail;
    }
};
}// namespace vstd
