#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <vstl/Hash.h>
#include <vstl/Memory.h>
#include <type_traits>
#include <new>
#include <vstl/VAllocator.h>
#include <functional>
#include <vector>
namespace vstd {

template<class Func>
class LazyEval {
private:
    Func func;

public:
    LazyEval(Func &&func)
        : func(std::move(func)) {}
    LazyEval(Func const &func)
        : func(func) {
    }
    operator std::invoke_result_t<Func>() const {
        return func();
    }
};

template<typename T>
using function = std::function<T>;
template<class Func>
LazyEval<std::remove_cvref_t<Func>> MakeLazyEval(Func &&func) {
    return std::forward<Func>(func);
}

template<typename T>
decltype(auto) MakeRunnable(T &&functor) {
    return function<FuncType<std::remove_cvref_t<T>>>(functor);
}
template<typename Vec, typename Func>
void push_back_func(Vec &&vec, Func &&func, size_t count) {
    std::fill_n(
        std::back_inserter(vec),
        count,
        LazyEval<std::remove_cvref_t<Func>>(std::forward<Func>(func)));
}

}// namespace vstd