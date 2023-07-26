#pragma once

#include "metal-cpp/SingleHeader/Metal.hpp"

namespace luisa::compute::metal {

namespace detail {

class ScopedAutoreleasePool {

private:
    NS::AutoreleasePool *_pool;

public:
    ScopedAutoreleasePool() noexcept
        : _pool{NS::AutoreleasePool::alloc()->init()} {}
    ~ScopedAutoreleasePool() noexcept { _pool->release(); }
    ScopedAutoreleasePool(ScopedAutoreleasePool &&) = delete;
    ScopedAutoreleasePool(const ScopedAutoreleasePool &) = delete;
    ScopedAutoreleasePool &operator=(ScopedAutoreleasePool &&) = delete;
    ScopedAutoreleasePool &operator=(const ScopedAutoreleasePool &) = delete;
};

}// namespace detail

template<typename F>
decltype(auto) with_autorelease_pool(F &&f) noexcept {
    detail::ScopedAutoreleasePool pool;
    return std::forward<F>(f)();
}

}// namespace luisa::compute::metal

