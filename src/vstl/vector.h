#pragma once
#include <vstl/config.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <initializer_list>
#include <type_traits>
#include <vstl/Memory.h>
#include <vstl/VAllocator.h>
#include <span>
namespace vstd {
namespace detail {
template<typename T, size_t stackCount>
struct vector_stack_obj {
    T *arr;
    std::aligned_storage_t<sizeof(T) * stackCount, alignof(T)> value;
};
template<typename T>
struct vector_stack_obj<T, 0> {
    T *arr;
};
}// namespace detail
template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine, size_t stackCount = 0>
class vector : public IOperatorNewBase {
private:
    //	template<bool v>
    //	struct EraseLastType;
    //	template<>
    //	struct EraseLastType<true> {
    //		using Type = T&&;
    //	};
    //	template<>
    //	struct EraseLastType<false> {
    //		using Type = T;
    //	};
    static_assert(!std::is_const_v<T>, "vector element cannot be constant!");
    detail::vector_stack_obj<T, stackCount> vec;
    size_t mSize;
    size_t mCapacity;
    VAllocHandle<allocType> allocHandle;
    static size_t GetNewVectorSize(size_t oldSize) {
        return oldSize * 1.5 + 8;
    }
    T *Allocate(size_t const &capacity) noexcept {
        return reinterpret_cast<T *>(allocHandle.Malloc(sizeof(T) * capacity));
    }

    void Free(T *ptr) noexcept {
        if constexpr (stackCount > 0) {
            if (mCapacity > stackCount)
                allocHandle.Free(ptr);
        } else {
            allocHandle.Free(ptr);
        }
    }
    void ResizeRange(size_t count) {
        if (mSize + count > mCapacity) {
            size_t newCapacity = GetNewVectorSize(mCapacity);
            size_t values[2] = {
                mCapacity + 1, count + mSize};
            newCapacity = newCapacity > values[0] ? newCapacity : values[0];
            newCapacity = newCapacity > values[1] ? newCapacity : values[1];
            reserve(newCapacity);
        }
    }
    void InitCapacity(size_t capacity) {
        if constexpr (stackCount > 0) {
            if (capacity < stackCount) {
                mCapacity = stackCount;
                vec.arr = reinterpret_cast<T *>(&vec.value);
            } else {
                mCapacity = capacity;
                vec.arr = Allocate(capacity);
            }
        } else {
            mCapacity = capacity;
            vec.arr = Allocate(capacity);
        }
    }

public:
    using Iterator = T *;
    void reserve(size_t newCapacity) noexcept {
        if (newCapacity <= mCapacity) return;
        T *newArr = Allocate(newCapacity);
        if (vec.arr) {
            if constexpr (std::is_trivially_move_constructible_v<T>) {
                memcpy(newArr, vec.arr, sizeof(T) * mSize);
                if constexpr (!(std::is_trivially_destructible_v<T>)) {
                    auto ee = end();
                    for (auto i = begin(); i != ee; ++i) {
                        i->~T();
                    }
                }
            } else {
                for (size_t i = 0; i < mSize; ++i) {
                    auto oldT = vec.arr + i;
                    new (newArr + i) T(std::move(*oldT));
                    if constexpr (!(std::is_trivially_destructible_v<T>)) {
                        oldT->~T();
                    }
                }
            }
            auto tempArr = vec.arr;
            vec.arr = newArr;
            Free(tempArr);
        } else {
            vec.arr = newArr;
        }
        mCapacity = newCapacity;
    }
    T *data() const noexcept { return vec.arr; }
    size_t size() const noexcept { return mSize; }
    size_t capacity() const noexcept { return mCapacity; }
    vector(size_t mSize) noexcept : mSize(mSize) {
        InitCapacity(mSize);
        if constexpr (!(std::is_trivially_constructible_v<T>)) {
            auto ee = end();
            for (auto i = begin(); i != ee; ++i) {
                new (i) T();
            }
        }
    }
    vector(std::initializer_list<T> const &lst) : mSize(lst.size()) {
        InitCapacity(lst.size());
        auto ptr = &static_cast<T const &>(*lst.begin());
        if constexpr (!(std::is_trivially_copy_constructible_v<T>)) {
            for (size_t i = 0; i < mSize; ++i) {
                new (vec.arr + i) T(ptr[i]);
            }
        } else {
            memcpy(vec.arr, ptr, sizeof(T) * mSize);
        }
    }
    vector(std::span<T> const &lst) : mSize(lst.size()) {
        InitCapacity(lst.size());
        if constexpr (!(std::is_trivially_copy_constructible_v<T>)) {
            for (size_t i = 0; i < mSize; ++i) {
                new (vec.arr + i) T(lst[i]);
            }
        } else {
            memcpy(vec.arr, &static_cast<T const &>(*lst.begin()), sizeof(T) * mSize);
        }
    }
    vector(const vector &another) noexcept
        : mSize(another.mSize),
          mCapacity(another.mCapacity) {
        vec.arr = Allocate(mCapacity);
        if constexpr (!(std::is_trivially_copy_constructible_v<T>)) {
            for (size_t i = 0; i < mSize; ++i) {
                new (vec.arr + i) T(another.vec.arr[i]);
            }
        } else
            memcpy(vec.arr, another.vec.arr, sizeof(T) * mSize);
    }
    //TODO
    vector(vector &&another) noexcept
        : mSize(another.mSize),
          mCapacity(another.mCapacity) {
        if constexpr (stackCount > 0) {
            //Local
            if (mCapacity <= stackCount) {
                vec.arr = reinterpret_cast<T *>(&vec.value);
                constexpr bool Trivial = std::is_trivially_move_constructible_v<T> && std::is_trivially_destructible_v<T>;
                if constexpr (!Trivial) {
                    for (size_t i = 0; i < another.mSize; ++i) {
                        T *arr = &another.vec.arr[i];
                        new (vec.arr + i) T(std::move(*arr));
                        if constexpr (!std::is_trivially_destructible_v<T>)
                            arr->~T();
                    }
                } else {
                    memcpy(vec.arr, another.vec.arr, sizeof(T) * another.mSize);
                }
            } else {
                vec.arr = another.vec.arr;
            }
        } else {
            vec.arr = another.vec.arr;
        }
        another.vec.arr = nullptr;
        another.mSize = 0;
        another.mCapacity = std::numeric_limits<size_t>::max();
    }
    void operator=(const vector &another) noexcept {
        clear();
        reserve(another.mSize);
        mSize = another.mSize;
        if constexpr (!(std::is_trivially_copy_constructible_v<T>)) {
            for (size_t i = 0; i < mSize; ++i) {
                new (vec.arr + i) T(another.vec.arr[i]);
            }
        } else {
            memcpy(vec.arr, another.vec.arr, sizeof(T) * mSize);
        }
    }
    void operator=(vector &&another) noexcept {
        this->~vector();
        new (this) vector(std::move(another));
    }

    void push_back_all(const T *values, size_t count) noexcept {
        ResizeRange(count);
        if constexpr (!(std::is_trivially_copy_constructible_v<T>)) {
            auto endPtr = vec.arr + mSize;
            for (size_t i = 0; i < count; ++i) {
                T *ptr = endPtr + i;
                new (ptr) T(values[i]);
            }
        } else {
            memcpy(vec.arr + mSize, values, count * sizeof(T));
        }
        mSize += count;
    }
    void push_back_all(std::span<T> sp) noexcept {
        push_back_all(sp.data(), sp.size());
    }
    void push_back_all(std::span<T const> sp) noexcept {
        push_back_all(sp.data(), sp.size());
    }
    template<typename Func>
    void push_back_func(Func &&f, size_t count) {
        ResizeRange(count);
        auto endPtr = vec.arr + mSize;
        size_t index = 0;
        for (auto &&i : ptr_range(endPtr, endPtr + count)) {
            T *ptr = &i;
            if constexpr (std::is_invocable_v<std::remove_cvref_t<Func>, size_t>) {
                new (ptr) T(std::move(f(index)));
            } else if constexpr (std::is_invocable_v<std::remove_cvref_t<Func>>) {
                new (ptr) T(std::move(f()));
            } else {
                static_assert(AlwaysFalse<Func>, "Invalid Function Type!");
            }
            ++index;
        }
        mSize += count;
    }
    template<typename Func>
        requires(std::is_invocable_v<Func, T &>)
    void compact(Func &&f) {
        T *curPtr = begin();
        T *lastPtr;
        T *ed = end();
        while (curPtr != ed) {
            if (f(*curPtr))
                curPtr++;
            else {
                lastPtr = curPtr;
                curPtr++;
                while (curPtr != ed) {
                    if (f(*curPtr)) {
                        if constexpr (std::is_move_assignable_v<T>) {
                            *lastPtr = std::move(*curPtr);
                        } else if constexpr (std::is_move_constructible_v<T>) {
                            if constexpr (!std::is_trivially_destructible_v<T>)
                                lastPtr->~T();
                            new (lastPtr) T(std::move(*curPtr));
                        } else {
                            static_assert(AlwaysFalse<Func>, "Element not movable!");
                        }
                        lastPtr++;
                    }
                    curPtr++;
                }
                size_t newSize = lastPtr - begin();
                if constexpr (!(std::is_trivially_destructible_v<T>)) {
                    for (auto i = begin() + newSize; i != ed; ++i) {
                        i->~T();
                    }
                }
                mSize = newSize;
                return;
            }
        }
    }
    operator std::span<T>() const {
        return std::span<T>(begin(), end());
    }
    operator std::span<T const>() const {
        return std::span<T const>(begin(), end());
    }
    void push_back_all(const std::initializer_list<T> &list) noexcept {
        push_back_all(&static_cast<T const &>(*list.begin()), list.size());
    }
    void SetZero() const noexcept {
        if constexpr (!(std::is_trivial_v<T>)) {
            static_assert(AlwaysFalse<T>, "Non-Trivial data cannot be setted");
        } else {
            if (vec.arr) memset(vec.arr, 0, sizeof(T) * mSize);
        }
    }
    void operator=(std::initializer_list<T> const &list) noexcept {
        clear();
        reserve(list.size());
        mSize = list.size();
        auto ptr = &static_cast<T const &>(*list.begin());
        if constexpr (!(std::is_trivially_copy_constructible_v<T>)) {
            for (size_t i = 0; i < mSize; ++i) {
                new (vec.arr + i) T(ptr[i]);
            }
        } else {
            memcpy(vec.arr, ptr, sizeof(T) * mSize);
        }
    }
    vector() noexcept : mCapacity(stackCount), mSize(0) {
        if constexpr (stackCount > 0) {
            vec.arr = reinterpret_cast<T *>(&vec.value);
        } else {
            vec.arr = nullptr;
        }
    }
    ~vector() noexcept {
        if (vec.arr) {
            clear();
            Free(vec.arr);
        }
    }
    bool empty() const noexcept {
        return mSize == 0;
    }

    template<typename... Args>
        requires(std::is_constructible_v<T, Args...>)
    T &emplace_back(Args &&...args) noexcept {
        if (mSize >= mCapacity) {
            size_t newCapacity = GetNewVectorSize(mCapacity);
            reserve(newCapacity);
        }
        T *ptr = vec.arr + mSize;
        new (ptr) T(std::forward<Args>(args)...);
        mSize++;
        return *ptr;
    }

    void push_back(const T &value) noexcept {
        emplace_back(value);
    }
    void push_back(T &value) noexcept {
        emplace_back(value);
    }
    void push_back(T &&value) noexcept {
        emplace_back(static_cast<T &&>(value));
    }

    T *begin() const noexcept {
        return vec.arr;
    }
    T *end() const noexcept {
        return vec.arr + mSize;
    }

    void erase(T *ite) noexcept {
        size_t index = reinterpret_cast<size_t>(ite) - reinterpret_cast<size_t>(vec.arr);
        index /= sizeof(T);
#if defined(DEBUG)
        if (index >= mSize) throw "Out of Range!";
#endif
        if constexpr (!(std::is_trivially_move_constructible_v<T>)) {
            if (index < mSize - 1) {
                for (auto &&i : ptr_range(ite, end() - 1)) {
                    i = std::move((&i)[1]);
                }
            }
            if constexpr (!(std::is_trivially_destructible_v<T>))
                (vec.arr + mSize - 1)->~T();
        } else {
            if (index < mSize - 1) {
                memmove(ite, ite + 1, (mSize - index - 1) * sizeof(T));
            }
        }
        mSize--;
    }

    T erase_last() noexcept {
        mSize--;
        if constexpr (!(std::is_trivially_destructible_v<T>)) {
            auto disp = create_disposer([this]() {
                (vec.arr + mSize)->~T();
            });
            return std::move(vec.arr[mSize]);
        } else {
            return std::move(vec.arr[mSize]);
        }
    }
    void clear() noexcept {
        if constexpr (!(std::is_trivially_destructible_v<T>)) {
            for (auto &&i : *this) {
                i.~T();
            }
        }
        mSize = 0;
    }
    void dispose() noexcept {
        clear();
        mCapacity = stackCount;
        if (vec.arr) {
            Free(vec.arr);
            vec.arr = nullptr;
        }
    }
    void resize(size_t newSize) noexcept {
        reserve(newSize);
        if constexpr (!((std::is_trivially_constructible_v<T> && std::is_trivially_destructible_v<T>))) {
            if (mSize < newSize) {
                if constexpr (!std::is_trivially_constructible_v<T>) {
                    auto bb = begin() + mSize;
                    auto ee = begin() + newSize;
                    for (auto i = bb; i != ee; ++i) {
                        new (i) T();
                    }
                }
            } else {
                if constexpr (!std::is_trivially_destructible_v<T>) {
                    if (mSize > newSize) {
                        auto bb = begin() + newSize;
                        auto ee = begin() + mSize;
                        for (auto i = bb; i != ee; ++i) {
                            i->~T();
                        }
                    }
                }
            }
        }
        mSize = newSize;
    }
    T &operator[](size_t index) noexcept {
#if defined(DEBUG)
        if (index >= mSize) throw "Out of Range!";
#endif
        return vec.arr[index];
    }
    const T &operator[](size_t index) const noexcept {
#if defined(DEBUG)
        if (index >= mSize) throw "Out of Range!";
#endif
        return vec.arr[index];
    }
};
}// namespace vstd