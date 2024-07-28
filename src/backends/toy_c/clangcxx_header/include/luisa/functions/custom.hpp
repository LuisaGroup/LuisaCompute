#pragma once
#include "math.hpp"

namespace luisa::shader {

template<typename T>
auto sum(T v) { return v; }

template<typename T, typename... Args>
auto sum(T v, Args... args) {
	return v + sum(args...);
}
[[ext_call("lc_memcmp")]] int memcmp(uint64 a, uint64 b, uint64 size);
[[ext_call("lc_memcpy")]] void memcpy(uint64 dst, uint64 src, uint64 size);
[[ext_call("lc_memmove")]] void memmove(uint64 dst, uint64 src, uint64 size);
namespace detail {
template<typename Ret, typename... Args>
[[ext_call("invoke")]] Ret __lc_builtin_invoke__(uint64 func_ptr, Args... args);
}// namespace detail
[[ext_call("persist_malloc")]] uint64 persist_malloc(uint64 size);
[[ext_call("temp_malloc")]] uint64 temp_malloc(uint64 size);
[[ext_call("persist_free")]] void persist_free(uint64 ptr);
namespace detail {
template<typename T>
[[ext_call("dispose")]] void __lc_dispose__(T& value);
}// namespace detail
template<typename T>
inline void dispose(T& vec) {
	detail::__lc_dispose__(vec);
}
template<typename T>
[[ext_call("is_trivial")]] bool is_trivial(T& value);
}// namespace luisa::shader