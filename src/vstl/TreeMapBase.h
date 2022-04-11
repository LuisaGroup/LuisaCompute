#pragma once
#include <vstl/Compare.h>
#include <vstl/Pool.h>
namespace vstd {
namespace detail {
class LC_VSTL_API TreeMapUtility {
public:
    static void fixInsert(void *k, void *&root);
    static void *getNext(void *ptr);
    static void *getLast(void *ptr);
    static void deleteOneNode(void *z, void *&root);
};
}// namespace detail

template<typename K, typename V>
struct TreeElement {
    K first;
    V second;
    template<typename A, typename... B>
    TreeElement(A &&a, B &&...b)
        : first(std::forward<A>(a)),
          second(std::forward<B>(b)...) {
    }
};
template<typename K, typename V>
struct ConstTreeElement {
    const K first;
    V second;
    template<typename A, typename... B>
    ConstTreeElement(A &&a, B &&...b)
        : first(std::forward<A>(a)),
          second(std::forward<B>(b)...) {
    }
};
template<typename K>
struct TreeElement<K, void> {
    K first;
    template<typename A>
    TreeElement(A &&a)
        : first(std::forward<A>(a)) {
    }
};
template<typename K>
struct ConstTreeElement<K, void> {
    const K first;
    template<typename A>
    ConstTreeElement(A &&a)
        : first(std::forward<A>(a)) {
    }
};
}// namespace vstd