#pragma once

#include <algorithm>
#include <utility>

#include <luisa/core/stl/pdqsort.h>

namespace luisa {

using std::swap;
using std::transform;
using std::binary_search;

template<typename Begin, typename End>
void sort(Begin &&begin, End &&end) noexcept {
    pdqsort(std::forward<Begin>(begin),
            std::forward<End>(end));
}

template<typename Begin, typename End, typename Compare>
void sort(Begin &&begin, End &&end, Compare &&comp) noexcept {
    pdqsort(std::forward<Begin>(begin),
            std::forward<End>(end),
            std::forward<Compare>(comp));
}

}// namespace luisa
