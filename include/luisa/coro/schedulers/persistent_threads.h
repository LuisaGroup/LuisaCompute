//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/coro/coro_func.h>
#include <luisa/coro/coro_scheduler.h>

namespace luisa::compute::coroutine {

struct PersistentThreadsCoroSchedulerConfig {
    uint thread_count = 64_k;
    uint block_size = 128;
    uint fetch_size = 4;
    bool shared_memory_soa = false;
    bool global_memory_ext = false;
};

namespace detail {
LUISA_CORO_API void persistent_threads_coro_scheduler_main_kernel_impl(
    const PersistentThreadsCoroSchedulerConfig &config,
    uint q_fac, uint g_fac, uint shared_queue_size, uint global_queue_size,
    const CoroGraph *graph, Shared<CoroFrame> &frames, Expr<uint3> dispatch_size_prefix_product,
    Expr<Buffer<uint>> global, const Buffer<CoroFrame> &global_frames,
    luisa::move_only_function<void(CoroFrame &, CoroToken)> call_subroutine) noexcept;
}// namespace detail

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
    Buffer<CoroFrame> _global_frames;
    Shader1D<uint> _initialize_shader;

private:
    void _prepare(Device &device, const Coro &coro) noexcept {
        _global = device.create_buffer<uint>(1);
        auto q_fac = 1u;
        auto g_fac = coro.subroutine_count() - q_fac;
        if (_config.global_memory_ext) {
            auto global_ext_size = _config.thread_count * g_fac;
            _global_frames = device.create<Buffer<CoroFrame>>(coro.shared_frame(), global_ext_size);
        }
        Kernel1D main_kernel = [this, q_fac, g_fac, &coro, graph = coro.graph()](BufferUInt global, UInt3 dispatch_size_prefix_product, Var<Args>... args) noexcept {
            set_block_size(_config.block_size, 1u, 1u);
            auto global_queue_size = _config.block_size * g_fac;
            auto shared_queue_size = _config.block_size * q_fac;
            auto call_subroutine = [&](CoroFrame &frame, CoroToken token) noexcept { coro[token](frame, args...); };
            Shared<CoroFrame> frames{graph->shared_frame(), shared_queue_size, _config.shared_memory_soa};
            detail::persistent_threads_coro_scheduler_main_kernel_impl(
                _config, q_fac, g_fac, shared_queue_size, global_queue_size, graph,
                frames, dispatch_size_prefix_product, global, _global_frames, call_subroutine);
        };
        _pt_shader = device.compile(main_kernel);
        _clear_shader = device.compile<1>([](BufferUInt global) {
            global->write(dispatch_x(), 0u);
        });
        if (_config.global_memory_ext) {
            _initialize_shader = device.compile<1>([&](UInt n) noexcept {
                auto x = dispatch_x();
                $if (x < n) {
                    _global_frames->write(x, coro.instantiate());
                };
            });
        }
    }

    void _dispatch(Stream &stream, uint3 dispatch_size,
                   compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        auto dispatch_size_prefix_product = make_uint3(
            dispatch_size.x,
            dispatch_size.x * dispatch_size.y,
            dispatch_size.x * dispatch_size.y * dispatch_size.z);
        if (_config.global_memory_ext) {
            auto n = static_cast<uint>(_global_frames.size());
            stream << _clear_shader(_global).dispatch(1u)
                   << _initialize_shader(n).dispatch(n)
                   << _pt_shader(_global, dispatch_size_prefix_product, args...).dispatch(_config.thread_count);
        } else {
            stream << _clear_shader(_global).dispatch(1u)
                   << _pt_shader(_global, dispatch_size_prefix_product, args...).dispatch(_config.thread_count);
        }
    }

public:
    PersistentThreadsCoroScheduler(Device &device, const Coro &coro, const Config &config) noexcept
        : _config{config} {
        _config.thread_count = luisa::align(_config.thread_count, _config.block_size);
        _prepare(device, coro);
    }
    PersistentThreadsCoroScheduler(Device &device, const Coro &coro) noexcept
        : PersistentThreadsCoroScheduler{device, coro, Config{}} {}
};

// User-defined CTAD guides
template<typename... Args>
PersistentThreadsCoroScheduler(Device &device, const Coroutine<void(Args...)> &coro,
                               const PersistentThreadsCoroSchedulerConfig &config) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

template<typename... Args>
PersistentThreadsCoroScheduler(Device &device, const Coroutine<void(Args...)> &coro) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

}// namespace luisa::compute::coroutine
