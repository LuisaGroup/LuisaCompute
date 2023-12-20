#pragma once
#include <type_traits>

namespace skr
{
namespace detail
{
template <typename F>
class defer_raii
{
public:
    // copy/move construction and any kind of assignment would lead to the cleanup function getting
    // called twice. We can't have that.
    defer_raii(defer_raii&&) = delete;
    defer_raii(const defer_raii&) = delete;
    defer_raii& operator=(const defer_raii&) = delete;
    defer_raii& operator=(defer_raii&&) = delete;

    // construct the object from the given callable
    template <typename FF>
    defer_raii(FF&& f)
        : cleanup_function(std::forward<FF>(f))
    {
    }

    // when the object goes out of scope call the cleanup function
    ~defer_raii() { cleanup_function(); }

private:
    F cleanup_function;
};
} // namespace detail

template <typename F>
detail::defer_raii<F> defer(F&& f)
{
    return { std::forward<F>(f) };
}

#define SKR_DEFER_ACTUALLY_JOIN(x, y) x##y
#define SKR_DEFER_JOIN(x, y) SKR_DEFER_ACTUALLY_JOIN(x, y)
#ifdef __COUNTER__
    #define SKR_DEFER_UNIQUE_VARNAME(x) SKR_DEFER_JOIN(x, __COUNTER__)
#else
    #define SKR_DEFER_UNIQUE_VARNAME(x) SKR_DEFER_JOIN(x, __LINE__)
#endif

#define SKR_DEFER(lambda__) [[maybe_unused]] const auto& SKR_DEFER_UNIQUE_VARNAME(defer_object) = skr::defer([&]() lambda__)
} // namespace skr