//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <cstdlib>
#include <type_traits>

#include <luisa/core/basic_types.h>
#include <luisa/core/concepts.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/functional.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

template<typename... Args>
class CoroScheduler;

namespace detail {

// A scheduler expands one logical coroutine into several physical shaders.
// Every generated shader must inherit the caller's compilation semantics
// (fast-math, debug information, register limits, driver optimization, ...).
// A user-provided AOT name must additionally be made unique per stage; an
// empty name stays empty so ordinary hash-based shader caching remains active.
[[nodiscard]] inline ShaderOption coro_scheduler_shader_option(
    const ShaderOption &base,
    luisa::string_view stage) noexcept {
    auto result = base;
    if (!result.name.empty()) {
        result.name = luisa::format("{}_{}", result.name, stage);
    }
    return result;
}

// ShaderOption::name selects the AOT/cache identity and therefore cannot be
// used merely to make scheduler stages visible in backend profilers. Apply a
// runtime resource label after compilation instead. Keep this diagnostic-only
// so ordinary scheduler dispatches do not pay backend encoder-label overhead.
template<typename Shader>
[[nodiscard]] inline Shader coro_scheduler_label_shader(
    Shader shader, luisa::string_view stage) noexcept {
    if (std::getenv("LUISA_CORO_SHADER_MAP") != nullptr) {
        shader.set_name(stage);
    }
    return shader;
}

template<typename... Args>
class CoroSchedulerInvoke;

/// Dispatch object returned by CoroSchedulerInvoke::dispatch().
/// Move-only; holds the concrete dispatch logic.  The primary usage is
///   auto disp = scheduler(args...).dispatch(size);
///   stream << disp;
/// Direct invocation with `disp(stream)` remains available for scheduler
/// implementations that need explicit submission.
class CoroSchedulerDispatch : public concepts::Noncopyable {

private:
    luisa::move_only_function<void(Stream &)> _impl;

private:
    template<typename... Args>
    friend class CoroSchedulerInvoke;
    explicit CoroSchedulerDispatch(luisa::move_only_function<void(Stream &)> impl) noexcept
        : _impl{std::move(impl)} {}

public:
    /// Submit the dispatch to a stream (explicit form).
    void operator()(Stream &stream) noexcept { _impl(stream); }
    /// Stream-pipe syntax:  `stream << dispatch`
    friend Stream &operator<<(Stream &stream, const CoroSchedulerDispatch &d) noexcept {
        d._impl(stream);
        return stream;
    }
};

template<typename... Args>
class CoroSchedulerInvoke : public concepts::Noncopyable {

private:
    using Scheduler = CoroScheduler<Args...>;
    template<typename T>
    using InvocationArgument = compute::detail::prototype_to_shader_invocation_t<T>;
    // Lazy dispatch owns ordinary scalar/aggregate snapshots. Move-only
    // resources such as Accel retain the invocation API's reference lifetime.
    template<typename T>
    using StoredArgument = std::conditional_t<
        std::is_copy_constructible_v<std::decay_t<InvocationArgument<T>>>,
        std::decay_t<InvocationArgument<T>>,
        InvocationArgument<T>>;
    Scheduler *_scheduler;
    std::tuple<StoredArgument<Args>...> _args;

private:
    friend Scheduler;
    explicit CoroSchedulerInvoke(Scheduler *scheduler,
                                 compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept
        : _scheduler{scheduler}, _args{args...} {}

public:
    [[nodiscard]] auto dispatch(uint3 size) && noexcept {
        return CoroSchedulerDispatch{[s = _scheduler, args = std::move(_args), size](Stream &stream) noexcept {
            std::apply(
                [s, size, &stream]<typename... A>(A &&...a) noexcept {
                    s->_dispatch(stream, size, std::forward<A>(a)...);
                },
                args);
        }};
    }
    [[nodiscard]] auto dispatch(uint nx, uint ny, uint nz) && noexcept {
        return std::move(*this).dispatch(luisa::make_uint3(nx, ny, nz));
    }
    [[nodiscard]] auto dispatch(uint nx, uint ny) && noexcept {
        return std::move(*this).dispatch(luisa::make_uint3(nx, ny, 1u));
    }
    [[nodiscard]] auto dispatch(uint2 size) && noexcept {
        return std::move(*this).dispatch(luisa::make_uint3(size, 1u));
    }
    [[nodiscard]] auto dispatch(uint nx) && noexcept {
        return std::move(*this).dispatch(luisa::make_uint3(nx, 1u, 1u));
    }
};

}// namespace detail

template<typename... Args>
class CoroScheduler {

private:
    friend class detail::CoroSchedulerInvoke<Args...>;

    /// Subclasses implement this to submit dispatch commands to the stream.
    virtual void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept = 0;

public:
    virtual ~CoroScheduler() noexcept = default;

    /// Binds coroutine arguments. Returns an invoker on which
    /// `.dispatch(size)` is called, yielding a `CoroSchedulerDispatch`
    /// that can be submitted to a stream:
    ///   stream << scheduler(args...).dispatch(size);
    [[nodiscard]] auto operator()(
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept {
        return detail::CoroSchedulerInvoke<Args...>{this, args...};
    }
};

}// namespace luisa::compute::coro
