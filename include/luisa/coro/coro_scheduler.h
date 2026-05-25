//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/runtime/shader.h>

namespace luisa::compute::coroutine {

template<typename... Args>
class CoroScheduler;

namespace detail {

template<typename... Args>
class CoroSchedulerInvoke;

class CoroSchedulerDispatch : public concepts::Noncopyable {

private:
    luisa::move_only_function<void(Stream &)> _impl;

private:
    template<typename... Args>
    friend class CoroSchedulerInvoke;
    explicit CoroSchedulerDispatch(luisa::move_only_function<void(Stream &)> impl) noexcept
        : _impl{std::move(impl)} {}

public:
    void operator()(Stream &stream) && noexcept { _impl(stream); }
};

template<typename... Args>
class CoroSchedulerInvoke : public concepts::Noncopyable {

private:
    using Scheduler = CoroScheduler<Args...>;
    Scheduler *_scheduler;
    std::tuple<compute::detail::prototype_to_shader_invocation_t<Args>...> _args;

private:
    friend Scheduler;
    explicit CoroSchedulerInvoke(Scheduler *scheduler, compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept
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
    virtual void _dispatch(Stream &stream, uint3 dispatch_size,
                           compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept = 0;

public:
    virtual ~CoroScheduler() noexcept = default;
    [[nodiscard]] auto operator()(compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept {
        return detail::CoroSchedulerInvoke<Args...>{this, args...};
    }
};

}// namespace luisa::compute::coroutine

LUISA_MARK_STREAM_EVENT_TYPE(luisa::compute::coroutine::detail::CoroSchedulerDispatch)
