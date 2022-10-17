#pragma once
#include <cstdint>
#include <cstddef> // size_t
const static inline size_t usize_MAX = (size_t)-1;
#ifdef __cplusplus

struct VectorType;
struct Type;
#else
struct VectorType;
struct Type;
typedef struct VectorType VectorType;
typedef struct Type Type;
#endif