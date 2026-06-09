#pragma once

#include <luisa/ast/function_builder.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

/// Megakernel coroutine scheduler: one thread per coroutine instance.
///
/// Each device thread:
///  1. Creates a thread-local CoroFrame
///  2. Calls the entry subroutine (scope 0)
///  3. Enters a switch-based state machine that polls target_token
///     and invokes the matching continuation subroutine
///  4. Exits when target_token == TERMINAL_TOKEN (0xFFFFFFFF)
///
/// The kernel is compiled eagerly in the constructor using the supplied Device.
template<typename... Args>
class StateMachineCoroScheduler : public CoroScheduler<Args...> {

    using Coro = Coroutine<void(Args...)>;
    using ShaderType = Shader<1, Args...>;
    using Base = CoroScheduler<Args...>;

    const Coro &_coro;
    ShaderType _shader;

public:
    /// Construct the scheduler and compile the state-machine kernel.
    ///
    /// @param device   Device used to compile the kernel.
    /// @param coro     The compiled coroutine (provides graph, frame_desc, subroutines).
    StateMachineCoroScheduler(Device &device, const Coro &coro) noexcept
        : Base{coro.graph(), coro.frame_desc()},
          _coro{coro},
          _shader{_compile(device, coro)} {}

    // Non-copyable, non-movable
    StateMachineCoroScheduler(const StateMachineCoroScheduler &) = delete;
    StateMachineCoroScheduler &operator=(const StateMachineCoroScheduler &) = delete;
    StateMachineCoroScheduler(StateMachineCoroScheduler &&) = delete;
    StateMachineCoroScheduler &operator=(StateMachineCoroScheduler &&) = delete;

    /// Dispatch the pre-compiled state-machine kernel.
    void _dispatch(Stream &stream, uint3 dispatch_size, const Args &...args) noexcept override {
        stream << _shader(args...).dispatch(dispatch_size.x);
    }

    [[nodiscard]] const Coro &coroutine() const noexcept { return _coro; }

private:
    /// Compile the state-machine kernel from the coroutine's subroutines.
    [[nodiscard]] static ShaderType _compile(Device &device, const Coro &coro) noexcept {
        return device.compile(Kernel1D{[&coro](Var<std::remove_cvref_t<Args>>... k_args) noexcept {
            const auto &graph = coro.graph();
            const auto *frame_desc = &coro.frame_desc();

            // Create thread-local frame
            auto frame = CoroFrame::create(frame_desc);
            frame.coro_id = dispatch_id();

            // Helper: call a subroutine (entry or continuation) with frame + args
            auto call_sub = [&](const auto &sub) noexcept {
                const Expression *call_args[1u + sizeof...(Args)];
                call_args[0] = frame.expression();
                size_t ai = 1u;
                ((call_args[ai++] = detail::extract_expression(k_args)), ...);
                detail::FunctionBuilder::current()->call(
                    sub->function(),
                    luisa::span<const Expression *const>{call_args, 1u + sizeof...(Args)});
            };

            // --- Phase 1: invoke the entry subroutine (scope 0) ---
            if (auto entry_sub = coro[0u]) {
                call_sub(entry_sub);
            }

            // --- Phase 2: call continuations sequentially ---
            // Each continuation has a skip-flag guard added by coro-split,
            // so calling them unconditionally is safe - they become no-ops
            // after the first execution.
            for (size_t i = 1u; i < graph.node_count(); ++i) {
                if (auto cont_sub = coro[i]) {
                    call_sub(cont_sub);
                }
            }
        }});
    }
};

// CTAD deduction guide
template<typename... Args>
StateMachineCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> StateMachineCoroScheduler<Args...>;

}// namespace luisa::compute::coro
