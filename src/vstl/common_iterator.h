#pragma once
#include <vstl/common.h>
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
template<typename K, typename V, typename Iterator>
class PairIterator : public IEnumerable<std::pair<K, V>>, public IOperatorNewBase {
    Iterator ptr;
    Iterator end;
    size_t size;

public:
    PairIterator(
        Iterator const &cur,
        Iterator const &end,
        size_t size)
        : ptr(cur), end(end), size(size) {}
    std::pair<K, V> GetValue() override {
        return {ptr->first, ptr->second};
    }
    bool End() override {
        return ptr == end;
    }
    void GetNext() override {
        ++ptr;
    }
    optional<size_t> Length() override {
        return size;
    }
};
}// namespace detail
template<typename T>
Iterator<T> GetIterator(vstd::span<T> const &sp) {
    return [&](void *ptr) {
        return new (ptr) detail::SpanIterator<T>{sp.data(), sp.size()};
    };
}
template<typename T>
Iterator<T> GetIterator(vstd::vector<T> const &sp) {
    return [&](void *ptr) {
        return new (ptr) detail::SpanIterator<T>{sp.data(), sp.size()};
    };
}
template<typename T, size_t node>
Iterator<T> GetIterator(vstd::fixed_vector<T, node> const &sp) {
    return [&](void *ptr) {
        return new (ptr) detail::SpanIterator<T>{sp.data(), sp.size()};
    };
}
template<typename K, typename V, typename Hash, typename Compare, VEngine_AllocType allocType>
Iterator<std::pair<K, V>> GetIterator(vstd::HashMap<K, V, Hash, Compare, allocType> const &map) {
    using IteratorType = decltype(std::declval(map).begin());
    return [&](void *ptr) {
        return new (ptr) detail::PairIterator<K, V, IteratorType>(
            map.begin(),
            map.end(),
            map.size());
    };
}
}// namespace vstd