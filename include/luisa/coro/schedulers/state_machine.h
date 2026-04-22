//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/coro/coro_func.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/coro/coro_frame_smem.h>
#include <luisa/coro/coro_scheduler.h>

namespace luisa::compute::coroutine {

namespace detail {
LUISA_CORO_API void coro_scheduler_state_machine_impl(
    CoroFrame &frame, uint state_count,
    luisa::move_only_function<void(CoroToken)> node) noexcept;
LUISA_CORO_API void coro_scheduler_state_machine_smem_impl(
    Shared<CoroFrame> &smem, const CoroGraph *graph,
    luisa::move_only_function<void(CoroToken, CoroFrame &frame)> node) noexcept;
}// namespace detail

struct StateMachineCoroSchedulerConfig {
    uint3 block_size = luisa::make_uint3(128, 1, 1);
    bool shared_memory = false;
    bool shared_memory_soa = true;
};

template<typename... Args>
class StateMachineCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = StateMachineCoroSchedulerConfig;

private:
    Shader3D<Args...> _shader;

private:
    void _create_shader(Device &device, const Coro &coroutine, const Config &config) noexcept {
        Kernel3D kernel = [&coroutine, &config](Var<Args>... args) noexcept {
            set_block_size(config.block_size);
            if (config.shared_memory) {
                auto n = config.block_size.x * config.block_size.y * config.block_size.z;
                Shared<CoroFrame> sm{coroutine.shared_frame(), n, config.shared_memory_soa, std::array{0u, 1u}};
                detail::coro_scheduler_state_machine_smem_impl(
                    sm, coroutine.graph(),
                    [&](CoroToken token, CoroFrame &frame) noexcept {
                        coroutine.subroutine(token)(frame, args...);
                    });
            } else {
                auto frame = coroutine.instantiate(dispatch_id());
                detail::coro_scheduler_state_machine_impl(
                    frame, coroutine.subroutine_count(),
                    [&](CoroToken token) noexcept {
                        coroutine.subroutine(token)(frame, args...);
                    });
            }
        };
        _shader = device.compile(kernel);
    }

    void _dispatch(Stream &stream, uint3 dispatch_size,
                   compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        stream << _shader(args...).dispatch(dispatch_size);
    }

public:
    StateMachineCoroScheduler(Device &device, const Coro &coro, const Config &config) noexcept {
        _create_shader(device, coro, config);
    }
    StateMachineCoroScheduler(Device &device, const Coro &coro) noexcept
        : StateMachineCoroScheduler{device, coro, Config{}} {}
};

// User-defined CTAD guides
template<typename... Args>
StateMachineCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> StateMachineCoroScheduler<Args...>;

template<typename... Args>
StateMachineCoroScheduler(Device &, const Coroutine<void(Args...)> &,
                          const StateMachineCoroSchedulerConfig &)
    -> StateMachineCoroScheduler<Args...>;

}// namespace luisa::compute::coroutine
