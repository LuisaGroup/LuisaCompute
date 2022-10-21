#pragma once

#include <cstdint>
#include <cstddef>// size_t

const static inline size_t usize_MAX = (size_t)-1;

#ifdef __cplusplus

namespace luisa::compute::ir {
struct VectorType;
struct Type;

using GcTraceFunc = void (*)(uint8_t *);
using GcDeleteFunc = void (*)(uint8_t *);
struct GcHeader {
    uint8_t *data = nullptr;
    GcHeader *next = nullptr;
    GcTraceFunc trace = nullptr;
    GcDeleteFunc del = nullptr;
    bool mark = false;
};
template<class T>
struct GcObject {
    GcHeader header;
    T data;
};
template<class T>
class Gc {
    GcObject<T> *object;

public:
    Gc() : object(nullptr) {}
    T *operator->() const { return &object->data; }
};
template<class T, class... Args>
inline Gc<T> make_gc(Args &&...args) {
    auto *object = new GcObject<T>();
    new (&object->data) T(std::forward<Args>(args)...);
    object->header.data = (uint8_t *)&object->data;
    object->header.trace = [](uint8_t *data) {
        auto *object = (GcObject<T> *)data;
        trace(object->data); // luisa fix this pls
    };
    object->header.del = [](uint8_t *data) {
        auto *object = (GcObject<T> *)data;
        delete object;
    };
    return Gc<T>{object};
}

class GcContext;
template<class F>
inline void mark_sweep(F &&f) {
    luisa_compute_gc_clear_marks();
    f();
    luisa_compute_gc_collect();
}
}// namespace luisa::compute::ir

#else

struct VectorType;
struct Type;
typedef struct VectorType VectorType;
typedef struct Type Type;

#endif
