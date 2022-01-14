#pragma once
#include <vstl/MetaLib.h>
#include <vstl/Memory.h>
#include <EASTL/unique_ptr.h>
namespace vstd {
struct unique_ptr_deleter {
    template<typename T>
    void operator()(T *ptr) const noexcept {
        if constexpr (std::is_base_of_v<IDisposable, T>) {
            ptr->Dispose();
        } else if constexpr (std::is_base_of_v<ISelfPtr, T>) {
            auto selfPtr = ptr->SelfPtr();
            ptr->~T();
            vengine_free(selfPtr);
        } else {
            ptr->~T();
            vengine_free(ptr);
        }
    }
};
template<typename T>
using unique_ptr = eastl::unique_ptr<T, unique_ptr_deleter>;
template<typename T>
unique_ptr<T> make_unique(T *ptr) {
    return unique_ptr<T>(ptr);
}
template<typename T>
unique_ptr<T> create_unique(T *ptr) {
    return unique_ptr<T>(ptr);
}
}// namespace vstd