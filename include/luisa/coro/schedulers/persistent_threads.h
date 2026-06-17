//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/dsl/coro_func.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute::coro {

struct PersistentThreadsCoroSchedulerConfig {
    uint thread_count = 65536u;// 64K threads
    uint block_size = 128u;    // threads per block
    uint fetch_size = 4u;      // blocks per atomic fetch
    bool shared_memory_soa = false;
    bool global_memory_ext = false;
};

template<typename... Args>
class PersistentThreadsCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = PersistentThreadsCoroSchedulerConfig;

private:
    Config _config;
    Shader1D<Buffer<uint>, uint3, Args...> _pt_shader;
    Shader1D<Buffer<uint>> _clear_shader;
    Buffer<uint> _global;

private:
    void _prepare(Device &device, const Coro &coro) noexcept {
        _global = device.create_buffer<uint>(1u);

        Kernel1D main_kernel = [this, &coro](BufferUInt global, UInt3 dispatch_size_prefix_product, Var<Args>... args) noexcept {
            set_block_size(_config.block_size, 1u, 1u);
            $loop {
                auto global_index = global.atomic(0u).fetch_add(1u);
                $if (global_index >= dispatch_size_prefix_product.z) { $break; };
                auto index_z = global_index / dispatch_size_prefix_product.y;
                auto index_xy = global_index - index_z * dispatch_size_prefix_product.y;
                auto index_y = index_xy / dispatch_size_prefix_product.x;
                auto index_x = index_xy - index_y * dispatch_size_prefix_product.x;

                auto frame = coro.instantiate(make_uint3(index_x, index_y, index_z));
                frame.target_token = 0u;
                frame.skip_flag = 0u;
                coro.entry()(frame, args...);
                $while (!frame.is_terminated()) {
                    for (size_t i = 1u; i < coro.subroutine_count(); ++i) {
                        $if (frame.target_token == coro.trigger_token(i)) {
                            frame.skip_flag = 0u;
                            coro[i](frame, args...);
                        };
                    }
                };
            };
        };
        _pt_shader = device.compile(main_kernel);

        _clear_shader = device.compile<1>([](BufferUInt g) {
            g.write(dispatch_x(), 0u);
        });
    }

    void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        auto dispatch_size_prefix_product = make_uint3(
            dispatch_size.x,
            dispatch_size.x * dispatch_size.y,
            dispatch_size.x * dispatch_size.y * dispatch_size.z);
        stream << _clear_shader(_global).dispatch(1u);
        stream << _pt_shader(_global, dispatch_size_prefix_product, args...).dispatch(_config.thread_count);
    }

public:
    [[nodiscard]] const Config &config() const noexcept { return _config; }

    PersistentThreadsCoroScheduler(Device &device, const Coro &coro, const Config &config) noexcept
        : _config{config} {
        _config.thread_count = luisa::align(_config.thread_count, _config.block_size);
        _prepare(device, coro);
    }
    PersistentThreadsCoroScheduler(Device &device, const Coro &coro) noexcept
        : PersistentThreadsCoroScheduler{device, coro, Config{}} {}
};

template<typename... Args>
PersistentThreadsCoroScheduler(Device &device, const Coroutine<void(Args...)> &coro,
                               const PersistentThreadsCoroSchedulerConfig &config) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

template<typename... Args>
PersistentThreadsCoroScheduler(Device &device, const Coroutine<void(Args...)> &coro) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

}// namespace luisa::compute::coro
