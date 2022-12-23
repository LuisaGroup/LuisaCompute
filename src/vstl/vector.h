#pragma once
#include <vstl/meta_lib.h>
#include <core/stl/vector.h>
namespace vstd {
template<typename T, typename Alloc = eastl::allocator>
using vector = luisa::vector<T, Alloc>;
template<typename T, size_t node_count, typename Alloc = eastl::allocator>
using fixed_vector = luisa::fixed_vector<T, node_count, Alloc>;
template<typename T>
using span = luisa::span<T>;
namespace detail {
template<typename F>
constexpr auto VectorFuncReturnType() {
    if constexpr (std::is_invocable_v<F>) {
        return TypeOf<std::invoke_result_t<F>>{};
    } else if constexpr (std::is_invocable_v<F, size_t>) {
        return TypeOf<std::invoke_result_t<F, size_t>>{};
    } else {
        return TypeOf<void>{};
    }
}
template<typename F>
using VectorFuncReturnType_t = typename decltype(VectorFuncReturnType<F>())::Type;
}// namespace detail
template<typename T, typename... Func>
    requires(std::is_constructible_v<T, detail::VectorFuncReturnType_t<Func> &&...>)
void push_back_func(vector<T> &vec, size_t n, Func &&...f) EA_NOEXCEPT {
    size_t index = 0;
    auto CallFunc = [&]<typename FT>(FT &&f) -> decltype(auto) {
        if constexpr (std::is_invocable_v<FT, size_t>) {
            return f(index);
        } else {
            return f();
        }
    };
    auto ptr = reinterpret_cast<T *>(vec.push_back_uninitialized(n));
    while (index != n) {
        ::new (ptr) T(CallFunc(f)...);
        ++index;
        ++ptr;
    }
}
template<typename T, size_t node, typename... Func>
    requires(std::is_constructible_v<T, detail::VectorFuncReturnType_t<Func> &&...>)
void push_back_func(fixed_vector<T, node> &vec, size_t n, Func &&...f) EA_NOEXCEPT {
    size_t index = 0;
    auto CallFunc = [&]<typename FT>(FT &&f) -> decltype(auto) {
        if constexpr (std::is_invocable_v<FT, size_t>) {
            return f(index);
        } else {
            return f();
        }
    };
    auto ptr = reinterpret_cast<T *>(vec.push_back_uninitialized(n));
    while (index != n) {
        ::new (ptr) T(CallFunc(f)...);
        ++index;
        ++ptr;
    }
}

template<typename T>
void push_back_all(vector<T> &vec, T const *t, size_t n) {
    auto ptr = reinterpret_cast<T *>(vec.push_back_uninitialized(n));
    auto end = vec.end();
    while (ptr != end) {
        ::new (ptr) T(*t);
        ++ptr;
        ++t;
    }
}
template<typename T, size_t node>
void push_back_all(fixed_vector<T, node> &vec, T const *t, size_t n) {
    auto ptr = reinterpret_cast<T *>(vec.push_back_uninitialized(n));
    auto end = vec.end();
    while (ptr != end) {
        ::new (ptr) T(*t);
        ++ptr;
        ++t;
    }
}
template<typename T, typename... Args>
    requires(std::is_constructible_v<T, Args &&...>)
void push_back_all(vector<T> &vec, size_t n, Args &&...args) {
    auto ptr = reinterpret_cast<T *>(vec.push_back_uninitialized(n));
    auto end = vec.end();
    while (ptr != end) {
        ::new (ptr) T(std::forward<Args>(args)...);
        ++ptr;
    }
}
template<typename T, size_t node, typename... Args>
    requires(std::is_constructible_v<T, Args &&...>)
void push_back_all(fixed_vector<T, node> &vec, size_t n, Args &&...args) {
    auto ptr = reinterpret_cast<T *>(vec.push_back_uninitialized(n));
    auto end = vec.end();
    while (ptr != end) {
        ::new (ptr) T(std::forward<Args>(args)...);
        ++ptr;
    }
}
template<typename T>
void push_back_all(vector<T> &vec, std::initializer_list<T> list) {
    push_back_all(vec, list.begin(), list.size());
}

template<typename T, size_t node>
void push_back_all(fixed_vector<T, node> &vec, std::initializer_list<T> list) {
    push_back_all(vec, list.begin(), list.size());
}
template<typename T>
void push_back_all(vector<T> &vec, span<T> list) {
    push_back_all(vec, list.data(), list.size());
}

template<typename T, size_t node>
void push_back_all(fixed_vector<T, node> &vec, span<T> list) {
    push_back_all(vec, list.data(), list.size());
}
template<typename T>
void push_back_all(vector<T> &vec, span<const T> list) {
    push_back_all(vec, list.data(), list.size());
}

template<typename T, size_t node>
void push_back_all(fixed_vector<T, node> &vec, span<const T> list) {
    push_back_all(vec, list.data(), list.size());
}

}// namespace vstd