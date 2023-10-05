#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>

const static inline size_t usize_MAX = (size_t)-1;

#ifdef __cplusplus

#include <luisa/core/stl/memory.h>

namespace luisa::compute::ir {

struct c_half {
    uint16_t bits;
};

struct VectorType;
struct Type;
using AtomicUsize = std::atomic<size_t>;
struct CallableModuleRef;
template<typename T>
struct CArcSharedBlock {
    T *ptr;
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
    CArc() = default;
    CArc(CArcSharedBlock<T> *block) noexcept : inner{block} {}
    [[nodiscard]] bool is_null() const noexcept { return inner == nullptr; }
    [[nodiscard]] T *operator->() const noexcept { return inner->ptr; }
    [[nodiscard]] T &operator*() const noexcept { return *inner->ptr; }
    [[nodiscard]] T *get() const noexcept { return this->inner->ptr; }
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
    CppOwnedCArc() : CArc<T> { nullptr }
    {}
    CppOwnedCArc(CArc<T> &&other) noexcept : CArc<T> { other.inner }
    { other.inner = nullptr; }
    explicit CppOwnedCArc(CArcSharedBlock<T> *inner) noexcept : CArc<T> { inner }
    {}
    CppOwnedCArc(const CppOwnedCArc &other) noexcept : CArc<T> { other.inner }
    {
        if (this->inner) this->retain();
    }
    CppOwnedCArc &operator=(const CppOwnedCArc &other) noexcept {
        if (this != &other) {
            if (this->inner) this->release();
            this->inner = other.inner;
            if (this->inner) this->retain();
        }
        return *this;
    }
    CppOwnedCArc(CppOwnedCArc &&other) noexcept : CArc<T> { other.inner }
    { other.inner = nullptr; }
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
    T *get() const noexcept { return ptr; }
    T *operator->() const noexcept { return ptr; }
    T &operator*() const noexcept { return *ptr; }
    T *ptr;
};

// forward decl for CBoxedSlice
template<typename T>
struct CBoxedSlice;

template<typename T>
[[nodiscard]] inline auto create_boxed_slice(size_t n) noexcept -> CBoxedSlice<T> {
    if (n == 0u) {
        return {.ptr = nullptr,
                .len = 0u,
                .destructor = [](T *, size_t) noexcept {}};
    }
    return {.ptr = luisa::allocate_with_allocator<T>(n),
            .len = n,
            .destructor = [](T *ptr, size_t) noexcept {
                luisa::deallocate_with_allocator(ptr);
            }};
}

template<typename T>
inline void destroy_boxed_slice(CBoxedSlice<T> slice) noexcept {
    if (slice.ptr != nullptr && slice.len > 0u) {
        slice.destructor(slice.ptr, slice.len);
    }
}

}// namespace luisa::compute::ir

#else

struct VectorType;
struct Type;
typedef struct VectorType VectorType;
typedef struct Type Type;

#endif
