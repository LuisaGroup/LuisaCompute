#pragma once
#include <vstl/Common.h>
#include <EASTL/vector.h>
namespace vstd {
namespace detail {
template<typename T>
class SpanIterator : public IEnumerable<T>, public IOperatorNewBase {
    T *ptr;
    T *ed;
    size_t mSize;

public:
    SpanIterator(T *ptr, size_t mSize)
        : ptr(ptr), ed(ptr + mSize), mSize(mSize) {}
    T GetValue() override {
        return *ptr;
    }
    bool End() override {
        return ptr == ed;
    }
    void GetNext() override {
        ++ptr;
    }
    optional<size_t> Length() override {
        return mSize;
    }
    ~SpanIterator() {}
};
}// namespace detail
template<typename T>
Iterator<T> GetIterator(vstd::span<T> const &sp) {
    return [&](void *ptr) {
        return new (ptr) detail::SpanIterator<T>{sp.data(), sp.size()};
    };
}
template<typename T>
Iterator<T> GetIterator(eastl::vector<T> const &sp) {
    return [&](void *ptr) {
        return new (ptr) detail::SpanIterator<T>{sp.data(), sp.size()};
    };
}
template<typename T, VEngine_AllocType alloc, size_t stackCount>
Iterator<T> GetIterator(vstd::vector<T, alloc, stackCount> &vec) {
    return [&](void *ptr) {
        return new (ptr) detail::SpanIterator<T>{vec.data(), vec.size()};
    };
}
}// namespace vstd