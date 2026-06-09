#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/dsl/coro_frame.h>

#include <tuple>
#include <utility>

namespace luisa::compute {
class Stream;
}

namespace luisa::compute::coro {

/// Abstract base class for all coroutine schedulers.
///
/// Derived scheduler types (StateMachine, Wavefront, Persistent, etc.)
/// implement the pure virtual `_dispatch()` method to issue device-side
/// coroutine launches via the Stream API.
///
/// Usage:
///   stream << scheduler(arg1, arg2).dispatch(width, height);
///
/// @tparam Args  Coroutine input parameter types.
template<typename... Args>
class CoroScheduler {

public:
    /// @brief Construct with references to the coroutine graph and frame descriptor.
    ///
    /// The scheduler does NOT own these objects; the caller must ensure
    /// they outlive the scheduler.
    CoroScheduler(const CoroGraph &graph, const CoroFrameDesc &frame_desc) noexcept
        : _graph{graph}, _frame_desc{frame_desc} {}

    virtual ~CoroScheduler() noexcept = default;

    // Non-copyable, non-movable (holds references to graph & frame_desc)
    CoroScheduler(const CoroScheduler &) = delete;
    CoroScheduler &operator=(const CoroScheduler &) = delete;
    CoroScheduler(CoroScheduler &&) = delete;
    CoroScheduler &operator=(CoroScheduler &&) = delete;

    /// Pure virtual dispatch method.
    /// Implemented by derived schedulers to launch the coroutine shader.
    virtual void _dispatch(Stream &stream, uint3 dispatch_size, const Args &...args) noexcept = 0;

    /// Intermediate object returned by `operator()`.
    /// Binds the scheduler and arguments; `.dispatch()` creates the actual
    /// command callable that is submitted to a Stream.
    struct CoroTaskSubmitter {
    private:
        friend class CoroScheduler;
        CoroScheduler *_scheduler;
        std::tuple<const Args &...> _args;

        explicit CoroTaskSubmitter(CoroScheduler *s, const Args &...a) noexcept
            : _scheduler{s}, _args{a...} {}

    public:
        [[nodiscard]] auto dispatch(uint3 size) && noexcept {
            return [s = _scheduler, args = _args, size](Stream &stream) noexcept {
                std::apply([&](const auto &...a) noexcept {
                    s->_dispatch(stream, size, a...);
                },
                           args);
            };
        }

        /// Convenience: 1D dispatch.
        [[nodiscard]] auto dispatch(uint x) && noexcept {
            return std::move(*this).dispatch(make_uint3(x, 1u, 1u));
        }

        /// Convenience: 2D dispatch.
        [[nodiscard]] auto dispatch(uint x, uint y) && noexcept {
            return std::move(*this).dispatch(make_uint3(x, y, 1u));
        }

        /// Convenience: 3D dispatch with individual dimensions.
        [[nodiscard]] auto dispatch(uint x, uint y, uint z) && noexcept {
            return std::move(*this).dispatch(make_uint3(x, y, z));
        }
    };

    /// Binds coroutine arguments for a subsequent dispatch.
    ///
    /// Returns a CoroTaskSubmitter on which `.dispatch(size)` must be called
    /// to obtain the actual dispatch callable.
    /// Accepts by const reference to support move-only types (e.g. Buffer<T>).
    [[nodiscard]] auto operator()(const Args &...args) noexcept {
        return CoroTaskSubmitter{this, args...};
    }

    // --- Accessors ---

    [[nodiscard]] const CoroGraph &graph() const noexcept { return _graph; }
    [[nodiscard]] const CoroFrameDesc &frame_desc() const noexcept { return _frame_desc; }

private:
    const CoroGraph &_graph;
    const CoroFrameDesc &_frame_desc;
};

}// namespace luisa::compute::coro
