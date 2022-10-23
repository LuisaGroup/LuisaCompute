#pragma once

#include <cstdint>
#include <cstddef>// size_t
#include <core/stl.h>

const static inline size_t usize_MAX = (size_t)-1;

#ifdef __cplusplus

namespace luisa::compute::ir {
struct VectorType;
struct Type;

using GcTraceFunc = void (*)(uint8_t *);
using GcDeleteFunc = void (*)(uint8_t *);

struct GcHeader {
    uint8_t *data;
    GcHeader *next;
    GcTraceFunc trace;
    GcDeleteFunc del;
    bool mark = false;
    bool root = false;
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
    Gc() noexcept = default;
    [[nodiscard]] auto get() noexcept { return &object->data; }
    [[nodiscard]] auto get() const noexcept { return const_cast<const T *>(&object->data); }
    [[nodiscard]] T *operator->() noexcept { return get(); }
    [[nodiscard]] const T *operator->() const noexcept { return get(); }
    void set_root(bool root) const noexcept {
        object->header.root = root;
    }
    [[nodiscard]] bool is_root() const noexcept {
        return object->header.root;
    }
};

template<class T, class... Args>
inline Gc<T> make_gc(Args &&...args) {
    auto *object = new_with_allocator<GcObject<T>>();
    new (&object->data) T(std::forward<Args>(args)...);
    object->header.data = (uint8_t *)&object->data;
    object->header.trace = [](uint8_t *data) {
        auto object = reinterpret_cast<GcObject<T> *>(data);
        trace(object->data);// luisa fix this pls
    };
    object->header.del = [](uint8_t *data) {
        auto object = reinterpret_cast<GcObject<T> *>(data);
        delete_with_allocator(object);
    };
    luisa_compute_gc_append_object(&object->header);
    return Gc<T>{object};
}

extern "C" {
void luisa_compute_gc_collect();
}

class GcContext;

inline void collect() {
    void luisa_compute_gc_collect();
}

}// namespace luisa::compute::ir

#else

struct VectorType;
struct Type;
typedef struct VectorType VectorType;
typedef struct Type Type;

#endif
