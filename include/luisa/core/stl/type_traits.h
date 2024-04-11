#pragma once
#include <EASTL/internal/type_pod.h>
namespace luisa {
template<class T, class... Args>
constexpr bool is_constructible_v = eastl::is_constructible_v<T, Args...>;
}// namespace luisa