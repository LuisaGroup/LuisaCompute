#pragma once

#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

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

    void _create_shader(Device &device, const Coro &coroutine, const Config &config) noexcept {
        Kernel3D kernel = [&coroutine, &config](Var<Args>... args) noexcept {
            set_block_size(config.block_size);
            auto frame = coroutine.instantiate(dispatch_id());
            frame.target_token = 0u;
            coroutine[0u](frame, args...);
            $while (!frame.is_terminated()) {
                for (size_t i = 1u; i < coroutine.subroutine_count(); ++i) {
                    frame.skip_flag = 0u;
                    coroutine[i](frame, args...);
                }
            };
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

template<typename... Args>
StateMachineCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> StateMachineCoroScheduler<Args...>;

template<typename... Args>
StateMachineCoroScheduler(Device &, const Coroutine<void(Args...)> &,
                          const StateMachineCoroSchedulerConfig &)
    -> StateMachineCoroScheduler<Args...>;

}// namespace luisa::compute::coro
