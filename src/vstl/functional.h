#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <vstl/hash.h>
#include <vstl/memory.h>
#include <type_traits>
#include <new>
#include <vstl/v_allocator.h>
#include <EASTL/functional.h>
#include <core/stl/functional.h>
namespace vstd {
template<typename T>
using function = luisa::move_only_function<T>;
}// namespace vstd