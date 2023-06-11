#pragma once
#include <luisa/vstl/config.h>
#include <stdint.h>
#include <luisa/vstl/hash.h>
#include <luisa/vstl/memory.h>
#include <type_traits>
#include <new>
#include <luisa/vstl/v_allocator.h>
#include <EASTL/functional.h>
#include <luisa/core/stl/functional.h>
namespace vstd {
template<typename T>
using function = luisa::move_only_function<T>;
}// namespace vstd
