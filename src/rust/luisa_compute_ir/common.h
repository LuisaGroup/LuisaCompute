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
using AtomicUsize = std::atomic<size_t>;

template<typename T>
struct CArcSharedBlock {
    T* ptr;
    AtomicUsize ref_count;
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
    CArcSharedBlock<T> *inner;
    CArc(CArcSharedBlock<T> *block = nullptr) noexcept: inner{block} {}
    [[nodiscard]] bool is_null() const noexcept { return inner == nullptr; }
    [[nodiscard]] T *operator->() const noexcept { return inner->ptr; }
    [[nodiscard]] T &operator*() const noexcept { return *inner->ptr; }
    [[nodiscard]] T* get() const noexcept { return this->inner->ptr; }
    [[nodiscard]] CArc<T> clone() const noexcept {
        retain();
        return CArc<T>{this->inner};
    }
    void retain() const noexcept {
        if (this->inner) this->inner->retain();
    }
    void release() noexcept {
        if (this->inner) {
            this->inner->release();
            this->inner = nullptr;
        }
    }
};

template<typename T>
struct CppOwnedCArc : CArc<T> {
    CppOwnedCArc() : CArc<T>{nullptr} {}
    CppOwnedCArc(CArc<T>&& other) noexcept : CArc<T>{other.inner} { other.inner = nullptr; }
    explicit CppOwnedCArc(CArcSharedBlock<T> *inner) noexcept : CArc<T>{inner} {}
    CppOwnedCArc(const CppOwnedCArc &other) noexcept : CArc<T>{other.inner} {
        if (this->inner) this->retain();
    }
    CppOwnedCArc &operator=(const CppOwnedCArc & other) noexcept {
        if (this != &other) {
            if (this->inner) this->release();
            this->inner = other.inner;
            if (this->inner) this->retain();
        }
        return *this;
    }
    CppOwnedCArc(CppOwnedCArc &&other) noexcept : CArc<T>{other.inner} { other.inner = nullptr; }
    CppOwnedCArc &operator=(CppOwnedCArc &&other) noexcept {
        if (this != &other) {
            if (this->inner) this->inner->release();
            this->inner = other.inner;
            other.inner = nullptr;
        }
        return *this;
    }
    template<class... Args>
    [[nodiscard]] friend CppOwnedCArc<T> make_arc(Args &&...args) {
        auto *block = new CArcSharedBlock<T>{T(std::forward<Args>(args)...), 1, [](CArcSharedBlock<T> *block) {
                                                 delete block;
                                             }};
        return CppOwnedCArc<T>(block);
    }

    ~CppOwnedCArc() {
        if (this->inner) this->release();
    }

};
template<typename T>
struct Pooled {
    T* get() const noexcept { return ptr; }
    T* operator->() const noexcept { return ptr; }
    T& operator*() const noexcept { return *ptr; }
    T *ptr;
};
}// namespace luisa::compute::ir

#else

struct VectorType;
struct Type;
typedef struct VectorType VectorType;
typedef struct Type Type;

#endif
