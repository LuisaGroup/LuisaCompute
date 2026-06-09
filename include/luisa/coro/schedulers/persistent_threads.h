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
    Shader1D<Buffer<uint>, Args...> _pt_shader;
    Shader1D<Buffer<uint>> _clear_shader;
    Buffer<uint> _global;
    Buffer<uint> _global_frames;
    Shader1D<uint> _initialize_shader;

private:
    void _prepare(Device &device, const Coro &coro) noexcept {
        _global = device.create_buffer<uint>(1u);

        if (_config.global_memory_ext) {
            auto frame_bytes = coro.frame().total_size();
            if (frame_bytes < sizeof(uint)) { frame_bytes = sizeof(uint); }
            auto g_fac = static_cast<uint>(coro.subroutine_count() - 1u);
            auto uint_count = (frame_bytes * _config.thread_count * g_fac + sizeof(uint) - 1u) / sizeof(uint);
            if (uint_count > 0u) {
                _global_frames = device.create_buffer<uint>(uint_count);
            }
        }

        Kernel1D main_kernel = [this, &coro](BufferUInt global, Var<Args>... args) noexcept {
            set_block_size(_config.block_size, 1u, 1u);
            auto tid = thread_x();
            // Each thread initializes a frame from the entry subroutine
            Shared<uint> shm_tokens{_config.block_size};
            Shared<bool> shm_active{_config.block_size};
            for (uint i = 0u; i < _config.block_size / _config.block_size; i++) {
                shm_tokens[tid] = 0u;
                shm_active[tid] = false;
            }
            sync_block();

            Shared<bool> shm_have_work{1u};
            Shared<uint> shm_work_start{1u};
            $if (tid == 0u) {
                UInt claimed = global.atomic(0u).fetch_add(_config.block_size * _config.fetch_size);
                shm_work_start[0u] = claimed;
                shm_have_work[0u] = true;
            };
            sync_block();

            $if (shm_have_work[0u]) {
                if (auto entry_sub = coro[0u]) {
                    auto frame = coro.instantiate();
                    frame.coro_id = make_uint3(tid, 0u, 0u);
                    entry_sub(frame, args...);
                    shm_tokens[tid] = frame.target_token;
                    shm_active[tid] = !frame.is_terminated();
                }
            };
        };
        _pt_shader = device.compile(main_kernel);

        _clear_shader = device.compile<1>([](BufferUInt g) {
            g.write(dispatch_x(), 0u);
        });

        if (_config.global_memory_ext) {
            _initialize_shader = device.compile<1>([&](UInt n) noexcept {
                auto x = dispatch_x();
                $if (x < n) { _global_frames->write(x, 0u); };
            });
        }
    }

    void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        stream << _clear_shader(_global).dispatch(1u);
        if (_config.global_memory_ext) {
            auto n = static_cast<uint>(_global_frames.size());
            stream << _initialize_shader(n).dispatch(n);
        }
        stream << _pt_shader(_global, args...).dispatch(_config.thread_count);
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

template<typename... Args>
PersistentThreadsCoroScheduler(Device &device, const Coroutine<void(Args...)> &coro,
                               const PersistentThreadsCoroSchedulerConfig &config) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

template<typename... Args>
PersistentThreadsCoroScheduler(Device &device, const Coroutine<void(Args...)> &coro) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

}// namespace luisa::compute::coro
