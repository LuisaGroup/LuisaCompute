#pragma once
#include <vstl/Compare.h>
#include <vstl/Pool.h>
namespace vstd {
namespace detail {
class VENGINE_DLL_COMMON TreeMapUtility {
public:
	static void fixInsert(void* k, void*& root);
	static void* getNext(void* ptr);
	static void* getLast(void* ptr);
	static void deleteOneNode(void* z, void*& root);
};
}// namespace detail
template<typename K, typename V>
struct TreeElement {
	const K first;
	V second;
	template<typename A, typename... B>
	TreeElement(A&& a, B&&... b)
		: first(std::forward<A>(a)),
		  second(std::forward<B>(b)...) {
	}
};
template<typename K>
struct TreeElement<K, void> {
	const K first;
	template<typename A>
	TreeElement(A&& a)
		: first(std::forward<A>(a)) {
	}
};
}// namespace vstd