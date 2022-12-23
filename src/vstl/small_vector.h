#pragma once
#include <vstl/vector.h>
namespace vstd {
namespace detail {
constexpr size_t SMALL_VECTOR_STACK_SIZE = 64;
template <typename T>
constexpr size_t SmallVecEleCount(){
    if constexpr(sizeof(T) > SMALL_VECTOR_STACK_SIZE){
        return (sizeof(T) > SMALL_VECTOR_STACK_SIZE * 2) ? 0 : 1;
    }
    return SMALL_VECTOR_STACK_SIZE / sizeof(T);
}
}
template<typename T>
using small_vector = fixed_vector<T, (detail::SmallVecEleCount<T>())>;
}// namespace vstd