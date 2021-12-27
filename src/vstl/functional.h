#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <vstl/Hash.h>
#include <vstl/Memory.h>
#include <type_traits>
#include <new>
#include <vstl/VAllocator.h>
#include <vector>
#include <EASTL/functional.h>
namespace vstd {
template<typename T>
using function = eastl::function<T>;

template<typename T>
decltype(auto) MakeRunnable(T &&functor) {
    return function<FuncType<std::remove_cvref_t<T>>>(functor);
}

}// namespace vstd