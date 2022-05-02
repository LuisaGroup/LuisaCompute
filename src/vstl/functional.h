#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <vstl/Hash.h>
#include <vstl/Memory.h>
#include <type_traits>
#include <new>
#include <vstl/VAllocator.h>
#include <EASTL/functional.h>
namespace vstd {
template<typename T>
using function = luisa::function<T>;
template<typename T>
using move_only_func = luisa::move_only_function<T>;

}// namespace vstd