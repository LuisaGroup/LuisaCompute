#pragma once

#include <cstdint>
#include <cstddef>// size_t

const static inline size_t usize_MAX = (size_t)-1;

#ifdef __cplusplus

namespace luisa::compute::ir {
struct VectorType;
struct Type;

template<class T>
struct Gc{
    const T ** ptr = nullptr;
    const T * operator->() const { return *ptr; }
};

}// namespace luisa::compute::ir

#else

struct VectorType;
struct Type;
typedef struct VectorType VectorType;
typedef struct Type Type;

#endif
