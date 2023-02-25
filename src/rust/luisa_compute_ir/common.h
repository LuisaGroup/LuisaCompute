#pragma once

#include <cstdint>
#include <cstddef>// size_t
#include <core/stl.h>
#include <atomic>

const static inline size_t usize_MAX = (size_t)-1;

#ifdef __cplusplus

namespace luisa::compute::ir {
struct VectorType;
struct Type;

template<typename T>
struct CArcSharedBlock {
    T ptr;
    std::atomic<size_t> ref_count;
    void (*destructor)(CArcSharedBlock<T> *);

    void release() {
        ref_count.fetch_sub(1, std::memory_order_release);
        if (ref_count.load(std::memory_order_acquire) == 0) {
            destructor(this);
        }
    }
    void retain() { ref_count.fetch_add(1, std::memory_order_release); }
};
static_assert(sizeof(CArcSharedBlock<int32_t>) == 24);

template<typename T>
struct CArc {
    CArc() : inner(nullptr) {}
    CArc(const CArc &) = delete;
    CArc &operator=(const CArc &) = delete;
    CArc(CArc &&other) noexcept : inner(other.inner) { other.inner = nullptr; }
    CArc &operator=(CArc &&other) noexcept {
        if (this != &other) {
            if (inner) inner->release();
            inner = other.inner;
            other.inner = nullptr;
        }
        return *this;
    }
    CArc clone() const noexcept {
        if (inner) inner->retain();
        return CArc(inner);
    }
    template<class... Args>
    friend CArc<T> make_arc<T>(Args &&...args) {
        auto *block = new CArcSharedBlock<T>{T(std::forward<Args>(args)...), 1, [](CArcSharedBlock<T> *block) {
                                                 delete block;
                                             }};
        return CArc<T>(block);
    }
    bool is_null() const noexcept { return inner == nullptr; }
    T *operator->() const noexcept { return &inner->ptr; }
    T &operator*() const noexcept { return inner->ptr; }
    ~CArc() {
        if (inner) inner->release();
    }

private:
    CArcSharedBlock<T> *inner;
    explicit CArc(CArcSharedBlock<T> *inner) : inner(inner) {}
};

}// namespace luisa::compute::ir

#else

struct VectorType;
struct Type;
typedef struct VectorType VectorType;
typedef struct Type Type;

#endif
