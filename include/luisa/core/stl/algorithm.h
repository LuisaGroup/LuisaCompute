#pragma once
#include <EASTL/algorithm.h>
#include <luisa/core/stl/pdqsort.h>
namespace luisa {
using eastl::transform;
using eastl::swap;
template<pdqsort_detail::LinearIterable Iter>
inline void sort(Iter begin, Iter end) {
    pdqsort(begin, end);
}
template<pdqsort_detail::LinearIterable Iter, pdqsort_detail::CompareFunc<Iter> Compare>
inline void sort(Iter begin, Iter end, Compare comp) {
    pdqsort(begin, end, comp);
}
}// namespace luisa