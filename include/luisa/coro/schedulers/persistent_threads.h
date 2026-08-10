//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/coro/coro_frame_storage.h>
#include <luisa/coro/schedulers/detail/token_index.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

#include <limits>

namespace luisa::compute::coro {

struct PersistentThreadsCoroSchedulerConfig {
    uint thread_count = 65536u;// 64K threads
    uint block_size = 128u;    // threads per block
    uint fetch_size = 4u;      // blocks per atomic fetch
    bool shared_memory_soa = false;
    bool global_memory_ext = false;
    ShaderOption shader_option{};
};

template<typename... Args>
class PersistentThreadsCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = PersistentThreadsCoroSchedulerConfig;

private:
    Config _config;
    Shader1D<Buffer<uint>, ByteBuffer, uint3, Args...> _pt_shader;
    Shader1D<Buffer<uint>> _clear_shader;
    Buffer<uint> _global;
    ByteBuffer _global_frames;
    CoroFrameStorageLayout _global_frame_layout;
    luisa::vector<luisa::vector<size_t>> _input_fields;
    luisa::vector<luisa::vector<size_t>> _output_fields;

private:
    [[nodiscard]] static Config _normalize_config(
        Config config, size_t subroutine_count) noexcept {
        constexpr auto uint_max = std::numeric_limits<uint>::max();
        LUISA_ASSERT(config.thread_count != 0u,
                     "Persistent coroutine worker count must be positive.");
        LUISA_ASSERT(config.block_size != 0u,
                     "Persistent coroutine block size must be positive.");
        LUISA_ASSERT(config.fetch_size != 0u,
                     "Persistent coroutine fetch size must be positive.");
        LUISA_ASSERT(subroutine_count != 0u && subroutine_count <= uint_max,
                     "Persistent coroutine subroutine count ({}) is outside the supported uint range.",
                     subroutine_count);

        auto aligned_thread_count =
            (static_cast<uint64_t>(config.thread_count) + config.block_size - 1u) /
            config.block_size * config.block_size;
        LUISA_ASSERT(aligned_thread_count <= uint_max,
                     "Persistent coroutine worker count ({}) cannot be aligned to block size ({}) "
                     "without overflowing uint.",
                     config.thread_count, config.block_size);
        config.thread_count = static_cast<uint>(aligned_thread_count);

        auto fetch_count =
            static_cast<uint64_t>(config.block_size) * config.fetch_size;
        LUISA_ASSERT(fetch_count <= uint_max,
                     "Persistent coroutine fetch batch ({} * {}) overflows uint.",
                     config.block_size, config.fetch_size);
        if (config.global_memory_ext) {
            auto total_queue_size =
                static_cast<uint64_t>(config.block_size) * subroutine_count;
            auto global_frame_capacity =
                static_cast<uint64_t>(config.thread_count) * (subroutine_count - 1u);
            LUISA_ASSERT(total_queue_size <= uint_max,
                         "Persistent coroutine block queue size ({} * {}) overflows uint.",
                         config.block_size, subroutine_count);
            LUISA_ASSERT(global_frame_capacity <= uint_max,
                         "Persistent coroutine global frame capacity ({} * {}) overflows uint.",
                         config.thread_count, subroutine_count - 1u);
        }
        return config;
    }

private:
    void _prepare(Device &device, const Coro &coro) noexcept {
        _global = device.create_buffer<uint>(1u);
        auto q_fac = 1u;
        auto coro_g_fac = static_cast<uint>(coro.subroutine_count() > q_fac ? coro.subroutine_count() - q_fac : 0u);
        auto g_fac = coro_g_fac;
        auto global_frame_capacity = _config.global_memory_ext ?
                                         std::max<uint>(1u, _config.thread_count * g_fac) :
                                         1u;
        _global_frame_layout = CoroFrameStorageLayout::make_aos(coro.frame(), global_frame_capacity);
        _global_frames = device.create_byte_buffer(_global_frame_layout.size_bytes);
        _input_fields.resize(coro.subroutine_count());
        _output_fields.resize(coro.subroutine_count());
        for (auto i = 0u; i < coro.subroutine_count(); i++) {
            _input_fields[i] = coro_frame_collect_input_fields(coro.graph(), i);
            _output_fields[i] = coro_frame_collect_output_fields(coro.graph(), i);
        }
        auto token_to_index = detail::make_coro_token_index_callable(coro);

        Kernel1D main_kernel = [this, &coro, &token_to_index, q_fac, g_fac,
                                input_fields = _input_fields,
                                output_fields = _output_fields,
                                global_layout = _global_frame_layout](
                                   BufferUInt global, ByteBufferVar global_frames,
                                   UInt3 dispatch_size_prefix_product, Var<Args>... args) noexcept {
            set_block_size(_config.block_size, 1u, 1u);
            auto subroutine_count = static_cast<uint>(coro.subroutine_count());
            auto shared_queue_size = _config.block_size * q_fac;
            auto global_queue_size = _config.block_size * g_fac;
            auto total_queue_size = _config.global_memory_ext ?
                                        shared_queue_size + global_queue_size :
                                        shared_queue_size;
            auto dispatch_id_from_linear = [&](UInt global_index) noexcept {
                auto index_z = global_index / dispatch_size_prefix_product.y;
                auto index_xy = global_index - index_z * dispatch_size_prefix_product.y;
                auto index_y = index_xy / dispatch_size_prefix_product.x;
                auto index_x = index_xy - index_y * dispatch_size_prefix_product.x;
                return make_uint3(index_x, index_y, index_z);
            };
            auto logical_dispatch_size = make_uint3(
                dispatch_size_prefix_product.x,
                dispatch_size_prefix_product.y / dispatch_size_prefix_product.x,
                dispatch_size_prefix_product.z / dispatch_size_prefix_product.y);
            CoroFrameSharedStorage frames{&coro.frame(), shared_queue_size, _config.shared_memory_soa};
            Shared<uint> all_token{total_queue_size};
            Shared<uint> path_id{shared_queue_size};
            Shared<uint> work_counter{subroutine_count};
            Shared<uint> work_offset{2u};
            Shared<uint> workload{2u};
            Shared<uint> work_stat{2u};
            Shared<uint> rem_global{1u};
            Shared<uint> rem_local{1u};

            for (auto index = 0u; index < q_fac; index++) {
                auto s = index * _config.block_size + thread_x();
                all_token[s] = 0u;
                if (_config.shared_memory_soa) {
                    auto frame = CoroFrame::create(&coro.frame());
                    frames.write(s, frame, luisa::span{input_fields[0u]});
                }
            }
            if (_config.global_memory_ext) {
                $for (index, 0u, g_fac) {
                    auto s = shared_queue_size + index * _config.block_size + thread_x();
                    all_token[s] = 0u;
                };
            }
            // A scheduler is not restricted to at most one continuation per
            // worker in a block. Initialize and reduce the complete counter
            // domain with a block-strided loop.
            $for (i, thread_x(), subroutine_count, _config.block_size) {
                work_counter[i] = 0u;
            };
            $if (thread_x() == 0u) {
                work_counter[0u] = total_queue_size;
            };
            $if (thread_x() == 0u) {
                workload[0u] = 0u;
                workload[1u] = 0u;
                rem_global[0u] = 1u;
                rem_local[0u] = 0u;
            };
            sync_block();

            $while (rem_global[0u] != 0u | rem_local[0u] != 0u) {
                sync_block();
                $if (thread_x() == 0u) {
                    rem_local[0u] = 0u;
                    work_stat[0u] = 0u;
                    work_stat[1u] = 0xffffffffu;
                };
                sync_block();

                $if (thread_x() == _config.block_size - 1u) {
                    $if (workload[0u] >= workload[1u] & rem_global[0u] != 0u) {
                        auto fetch_count = _config.block_size * _config.fetch_size;
                        auto st = global.atomic(0u).fetch_add(fetch_count);
                        workload[0u] = st;
                        workload[1u] = min(st + fetch_count, dispatch_size_prefix_product.z);
                        $if (st >= dispatch_size_prefix_product.z) {
                            rem_global[0u] = 0u;
                        };
                    };
                };
                sync_block();

                $for (i, thread_x(), subroutine_count, _config.block_size) {
                    $if (workload[0u] < workload[1u] | i != 0u) {
                        auto count = work_counter[i];
                        $if (count != 0u) {
                            rem_local.atomic(0u).fetch_or(1u);
                            work_stat.atomic(0u).fetch_max(count);
                        };
                    };
                };
                sync_block();
                $for (i, thread_x(), subroutine_count, _config.block_size) {
                    auto count = work_counter[i];
                    $if (work_stat[0u] == count & (workload[0u] < workload[1u] | i != 0u)) {
                        // Equal-size classes are semantically interchangeable.
                        // Pick the lowest index atomically to avoid a shared-memory
                        // data race and make the schedule deterministic.
                        work_stat.atomic(1u).fetch_min(i);
                    };
                };
                sync_block();
                $if (thread_x() == 0u) {
                    $if (work_stat[0u] == 0u & rem_global[0u] != 0u) {
                        rem_local[0u] = 1u;
                    };
                };
                sync_block();

                $if (thread_x() == 0u) {
                    work_offset[0u] = 0u;
                    work_offset[1u] = 0u;
                };
                sync_block();

                if (!_config.global_memory_ext) {
                    for (auto index = 0u; index < q_fac; index++) {
                        auto frame_token = def(all_token[index * _config.block_size + thread_x()]);
                        $if (frame_token == work_stat[1u]) {
                            auto id = work_offset.atomic(0u).fetch_add(1u);
                            path_id[id] = index * _config.block_size + thread_x();
                        };
                    }
                } else {
                    for (auto index = 0u; index < q_fac; index++) {
                        auto frame_token = def(all_token[index * _config.block_size + thread_x()]);
                        $if (frame_token != work_stat[1u]) {
                            auto id = work_offset.atomic(0u).fetch_add(1u);
                            path_id[id] = index * _config.block_size + thread_x();
                        };
                    }
                    sync_block();
                    $if (shared_queue_size - work_offset[0u] < _config.block_size) {
                        // `g_fac` grows with the continuation count. This is a
                        // device loop by design: host unrolling duplicates the
                        // complete frame spill/restore path once per continuation.
                        $for (index, 0u, g_fac) {
                            auto global_queue_id = index * _config.block_size + thread_x();
                            auto global_token_index = shared_queue_size + global_queue_id;
                            auto coro_token = def(all_token[global_token_index]);
                            $if (coro_token == work_stat[1u]) {
                                auto id = work_offset.atomic(1u).fetch_add(1u);
                                $if (id < work_offset[0u]) {
                                    auto dst = path_id[id];
                                    auto global_id = block_x() * global_queue_size + global_queue_id;
                                    auto frame_token = def(all_token[dst]);
                                    $if (coro_token != 0u) {
                                        auto global_frame = coro_frame_load(
                                            &coro.frame(), global_frames, global_id,
                                            global_layout, false, luisa::nullopt, true);
                                        $if (frame_token != 0u) {
                                            auto frame = frames.read(dst);
                                            coro_frame_store(global_frames, global_id, frame, global_layout, false, luisa::nullopt, true);
                                        };
                                        frames.write(dst, global_frame);
                                        all_token[global_token_index] = frame_token;
                                        all_token[dst] = coro_token;
                                    }
                                    $elif (frame_token != 0u) {
                                        auto frame = frames.read(dst);
                                        coro_frame_store(global_frames, global_id, frame, global_layout, false, luisa::nullopt, true);
                                        $if (frame_token != 0u) {
                                            all_token[global_token_index] = frame_token;
                                            all_token[dst] = coro_token;
                                        };
                                    };
                                };
                            };
                        };
                    };
                }

                auto gen_start = workload[0u];
                sync_block();
                auto pid = def(_config.global_memory_ext ? thread_x() : 0u);
                auto do_work = def(false);
                if (_config.global_memory_ext) {
                    do_work = all_token[pid] == work_stat[1u];
                } else {
                    do_work = thread_x() < work_offset[0u];
                    $if (do_work) {
                        pid = path_id[thread_x()];
                    };
                }
                $if (do_work) {
                    auto current_token = def(all_token[pid]);
                    $if (current_token == 0u) {
                        auto global_index = _config.global_memory_ext ?
                                                gen_start + thread_x() :
                                                workload.atomic(0u).fetch_add(1u);
                        $if (global_index < workload[1u]) {
                            work_counter.atomic(0u).fetch_sub(1u);
                            auto frame = coro.instantiate(dispatch_id_from_linear(global_index), logical_dispatch_size);
                            frame.target_token = 0u;
                            coro.entry()(frame, args...);
                            auto next = token_to_index(frame.target_token);
                            $if (frame.target_token == CoroFrame::TERMINAL_TOKEN) {
                                frame.target_token = 0u;
                            };
                            frames.write(pid, frame, luisa::span{output_fields[0u]});
                            all_token[pid] = next;
                            work_counter.atomic(next).fetch_add(1u);
                            if (_config.global_memory_ext) {
                                workload.atomic(0u).fetch_add(1u);
                            }
                        };
                    };
                    for (size_t i = 1u; i < coro.subroutine_count(); ++i) {
                        $if (current_token == static_cast<uint>(i)) {
                            work_counter.atomic(static_cast<uint>(i)).fetch_sub(1u);
                            auto frame = frames.read(pid, luisa::span{input_fields[i]});
                            coro[i](frame, args...);
                            auto next = token_to_index(frame.target_token);
                            $if (frame.target_token == CoroFrame::TERMINAL_TOKEN) {
                                frame.target_token = 0u;
                            };
                            frames.write(pid, frame, luisa::span{output_fields[i]});
                            all_token[pid] = next;
                            work_counter.atomic(next).fetch_add(1u);
                        };
                    }
                };
                sync_block();
            };
            sync_block();
        };
        auto main_shader_option =
            detail::coro_scheduler_shader_option(
                _config.shader_option, "persistent_main");
        _pt_shader = device.compile(main_kernel, main_shader_option);

        _clear_shader = device.compile<1>([](BufferUInt g) {
            g.write(dispatch_x(), 0u);
        }, detail::coro_scheduler_shader_option(
               _config.shader_option, "persistent_clear"));
    }

    void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        auto prefix_y =
            static_cast<uint64_t>(dispatch_size.x) * dispatch_size.y;
        auto logical_dispatch_size = prefix_y * dispatch_size.z;
        if (logical_dispatch_size == 0u) { return; }
        LUISA_ASSERT(logical_dispatch_size <= std::numeric_limits<uint>::max(),
                     "Persistent coroutine logical dispatch size ({} x {} x {}) exceeds uint capacity.",
                     dispatch_size.x, dispatch_size.y, dispatch_size.z);
        auto dispatch_size_prefix_product = make_uint3(
            dispatch_size.x,
            static_cast<uint>(prefix_y),
            static_cast<uint>(logical_dispatch_size));
        stream << _clear_shader(_global).dispatch(1u);
        // Global frames are reachable only through this dispatch's shared
        // `all_token` table, which is initialized to the empty token. Every
        // non-empty token is published only after its frame has been stored, so
        // stale bytes from a previous dispatch are unobservable and need not be
        // cleared.
        // The configured thread count is the scheduler's worker-capacity ceiling.
        // Launching more workers than logical coroutine instances only creates idle
        // workgroups contending on the global work counter. Keep a complete final
        // workgroup because the persistent kernel contains block barriers.
        auto worker_count = _config.thread_count;
        if (dispatch_size_prefix_product.z < worker_count) {
            worker_count = static_cast<uint>(
                luisa::align(dispatch_size_prefix_product.z, _config.block_size));
        }
        stream << _pt_shader(_global, _global_frames, dispatch_size_prefix_product, args...).dispatch(worker_count);
    }

public:
    [[nodiscard]] const Config &config() const noexcept { return _config; }

    PersistentThreadsCoroScheduler(Device &device, const Coro &coro, const Config &config) noexcept
        : _config{_normalize_config(config, coro.subroutine_count())} {
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
