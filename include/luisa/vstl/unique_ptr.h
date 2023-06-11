#pragma once
#include <luisa/vstl/meta_lib.h>
#include <luisa/vstl/memory.h>
#include <EASTL/unique_ptr.h>
#include <EASTL/shared_ptr.h>
namespace vstd {
struct unique_ptr_deleter {
    template<typename T>
    void operator()(T *ptr) const noexcept {
        if constexpr (std::is_base_of_v<IDisposable, T>) {
            ptr->Dispose();
        } else {
            vstd::destruct(ptr);
            vengine_free(ptr);
        }
    }
};
template<typename T, typename Deleter = unique_ptr_deleter>
using unique_ptr = eastl::unique_ptr<T, Deleter>;
template<typename T>
unique_ptr<T> create_unique(T *ptr) {
    return unique_ptr<T>(ptr);
}
template<typename T>
using shared_ptr = eastl::shared_ptr<T>;
template<typename T>
shared_ptr<T> create_shared(T *ptr) {
    return shared_ptr<T>(ptr);
}

using luisa::make_shared;
template<typename T, typename... Args>
    requires(std::is_constructible_v<T, Args &&...>)
unique_ptr<T> make_unique(Args &&...args) {
    return unique_ptr<T>(new (vengine_malloc(sizeof(T))) T(std::forward<Args>(args)...));
}
}// namespace vstd
