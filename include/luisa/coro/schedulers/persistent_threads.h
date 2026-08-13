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
#include <utility>

namespace luisa::compute::coro {

struct PersistentThreadsCoroSchedulerConfig {
    uint thread_count = 65536u;// 64K threads
    uint block_size = 128u;    // threads per block
    uint fetch_size = 4u;      // blocks per atomic fetch
    bool shared_memory_soa = false;
    bool global_memory_ext = false;
    // Store every queue slot's frame in global memory. The token table and
    // scheduling counters remain workgroup-local, so this preserves the same
    // slot-state machine while removing the O(block_size * frame_size) shared
    // memory term. Implies global_memory_ext. The scheduler may enable this
    // automatically when even one portable scheduler quantum of shared frames
    // cannot fit.
    bool global_memory_frames = false;
    // Optional portable cap for tests or applications that need to reserve
    // workgroup memory for backend-specific instrumentation. Zero uses the
    // device-reported limit; if neither is available, no automatic fitting is
    // attempted.
    size_t shared_memory_limit_bytes = 0u;
    ShaderOption shader_option{};
};

template<typename... Args>
class PersistentThreadsCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = PersistentThreadsCoroSchedulerConfig;

private:
    Config _config;
    Shader1D<Buffer<uint>, ByteBuffer, uint3, uint, Args...> _pt_shader;
    Shader1D<Buffer<uint>> _clear_shader;
    Buffer<uint> _global;
    ByteBuffer _global_frames;
    CoroFrameStorageLayout _global_frame_layout;
    luisa::vector<luisa::vector<size_t>> _input_fields;
    luisa::vector<luisa::vector<size_t>> _output_fields;
    luisa::vector<luisa::vector<size_t>> _relocation_fields;
    size_t _static_shared_memory_size_bytes{};
    uint64_t _main_shader_structure_hash{};

private:
    [[nodiscard]] static Config _normalize_config(
        Config config, size_t subroutine_count) noexcept {
        constexpr auto uint_max = std::numeric_limits<uint>::max();
        LUISA_ASSERT(config.thread_count != 0u,
                     "Persistent coroutine worker count must be positive.");
        LUISA_ASSERT(config.block_size >= 32u &&
                         config.block_size <= 1024u &&
                         config.block_size % 32u == 0u,
                     "Persistent coroutine block size must be a multiple of "
                     "32 in [32, 1024], but got {}.",
                     config.block_size);
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
        if (config.global_memory_frames) {
            config.global_memory_ext = true;
        }
        if (config.global_memory_ext) {
            auto total_queue_size =
                static_cast<uint64_t>(config.block_size) * subroutine_count;
            auto global_frame_capacity =
                static_cast<uint64_t>(config.thread_count) *
                (subroutine_count - (config.global_memory_frames ? 0u : 1u));
            LUISA_ASSERT(total_queue_size <= uint_max,
                         "Persistent coroutine block queue size ({} * {}) overflows uint.",
                         config.block_size, subroutine_count);
            LUISA_ASSERT(global_frame_capacity <= uint_max,
                         "Persistent coroutine global frame capacity ({} * {}) overflows uint.",
                         config.thread_count, subroutine_count - 1u);
        }
        return config;
    }

    template<typename K>
    [[nodiscard]] static size_t _shared_memory_size(const K &kernel) noexcept {
        auto size = size_t{0u};
        for (auto variable : kernel.function()->shared_variables()) {
            auto alignment = std::max(variable.type()->alignment(), size_t{1u});
            auto remainder = size % alignment;
            auto padding = remainder == 0u ? 0u : alignment - remainder;
            LUISA_ASSERT(
                padding <= std::numeric_limits<size_t>::max() - size,
                "Persistent coroutine shared-memory alignment overflows size_t.");
            size += padding;
            LUISA_ASSERT(
                variable.type()->size() <= std::numeric_limits<size_t>::max() - size,
                "Persistent coroutine shared-memory size overflows size_t.");
            size += variable.type()->size();
        }
        return size;
    }

    [[nodiscard]] static uint _next_block_size(
        uint requested, uint current, uint warp_size) noexcept {
        if (current <= warp_size) { return 0u; }
        auto candidate = current - warp_size;
        while (candidate >= warp_size) {
            // Keeping a divisor of the originally requested block preserves
            // the already-normalized worker-count alignment.
            if (requested % candidate == 0u) { return candidate; }
            if (candidate < warp_size * 2u) { break; }
            candidate -= warp_size;
        }
        return 0u;
    }

private:
    void _prepare(Device &device, const Coro &coro) noexcept {
        _global = device.create_buffer<uint>(1u);
        auto update_global_frame_layout = [&]() noexcept {
            auto shared_queue_factor =
                _config.global_memory_frames ? 0u : 1u;
            auto global_queue_factor = _config.global_memory_ext ?
                                           static_cast<uint>(coro.subroutine_count()) - shared_queue_factor :
                                           0u;
            auto global_frame_capacity_u64 = _config.global_memory_ext ?
                                                 static_cast<uint64_t>(_config.thread_count) * global_queue_factor :
                                                 1u;
            LUISA_ASSERT(
                global_frame_capacity_u64 <= std::numeric_limits<uint>::max(),
                "Persistent coroutine global frame capacity ({}) exceeds the "
                "current uint queue-index ABI.",
                global_frame_capacity_u64);
            auto global_frame_capacity =
                static_cast<size_t>(std::max<uint64_t>(
                    global_frame_capacity_u64, 1u));
            auto frame_stride = coro.frame().frame_type()->size();
            LUISA_ASSERT(
                frame_stride == 0u ||
                    global_frame_capacity <=
                        std::numeric_limits<uint>::max() / frame_stride,
                "Persistent coroutine global frame storage ({} frames * {} "
                "bytes) exceeds the current 32-bit byte-buffer offset ABI.",
                global_frame_capacity, frame_stride);
            _global_frame_layout =
                CoroFrameStorageLayout::make_aos(
                    coro.frame(), global_frame_capacity);
        };
        update_global_frame_layout();
        _input_fields.resize(coro.subroutine_count());
        _output_fields.resize(coro.subroutine_count());
        for (auto i = 0u; i < coro.subroutine_count(); i++) {
            _input_fields[i] = coro_frame_collect_input_fields(coro.graph(), i);
            _output_fields[i] = coro_frame_collect_output_fields(coro.graph(), i);
        }
        _relocation_fields = coro_frame_collect_relocation_fields(
            coro.graph(), coro.frame().frame_field_count());
        auto token_to_index = detail::make_coro_token_index_callable(coro);

        auto make_main_kernel = [&]() noexcept {
            auto shared_queue_factor =
                _config.global_memory_frames ? 0u : 1u;
            auto global_queue_factor = _config.global_memory_ext ?
                                           static_cast<uint>(coro.subroutine_count()) - shared_queue_factor :
                                           0u;
            auto global_memory_frames =
                _config.global_memory_frames;
            luisa::vector<size_t> common_relocation_fields;
            if (coro.subroutine_count() > 1u) {
                common_relocation_fields = _relocation_fields[1u];
                for (auto i = 2u; i < coro.subroutine_count(); ++i) {
                    auto &fields = _relocation_fields[i];
                    common_relocation_fields.erase(
                        std::remove_if(
                            common_relocation_fields.begin(),
                            common_relocation_fields.end(),
                            [&](auto field) noexcept {
                                return std::find(
                                           fields.begin(), fields.end(),
                                           field) == fields.end();
                            }),
                        common_relocation_fields.end());
                }
            }
            auto residual_relocation_fields = _relocation_fields;
            for (auto i = 1u; i < coro.subroutine_count(); ++i) {
                auto &fields = residual_relocation_fields[i];
                fields.erase(
                    std::remove_if(
                        fields.begin(), fields.end(),
                        [&](auto field) noexcept {
                            return std::find(
                                       common_relocation_fields.begin(),
                                       common_relocation_fields.end(),
                                       field) !=
                                   common_relocation_fields.end();
                        }),
                    fields.end());
            }
            return Kernel1D{[this, &coro, &token_to_index,
                             shared_queue_factor, global_queue_factor,
                             global_memory_frames,
                             common_relocation_fields,
                             residual_relocation_fields,
                             input_fields = _input_fields,
                             output_fields = _output_fields,
                             global_layout = _global_frame_layout](
                                BufferUInt global, ByteBufferVar global_frames,
                                UInt3 dispatch_size_prefix_product,
                                UInt fetch_size, Var<Args>... args) noexcept {
                set_block_size(_config.block_size, 1u, 1u);
                auto subroutine_count = static_cast<uint>(coro.subroutine_count());
                auto shared_queue_size =
                    _config.block_size * shared_queue_factor;
                auto global_queue_size =
                    _config.block_size * global_queue_factor;
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
                luisa::optional<CoroFrameSharedStorage> shared_frames;
                if (!global_memory_frames) {
                    shared_frames.emplace(
                        &coro.frame(), shared_queue_size,
                        _config.shared_memory_soa);
                }
                Shared<uint> all_token{total_queue_size};
                Shared<uint> path_id{_config.block_size};
                Shared<uint> work_counter{subroutine_count};
                Shared<uint> work_offset{2u};
                Shared<uint> workload{2u};
                Shared<uint> work_stat{2u};
                Shared<uint> rem_global{1u};
                Shared<uint> rem_local{1u};

                // A queued frame is a token-indexed sum type. Relocation must
                // preserve not only fields read by continuation t, but fields
                // live through t and first consumed later. CoroGraph projects
                // cfg-distill's least-fixed-point live_begin certificate onto
                // these physical relocation fields.
                auto load_global_frame = [&](UInt token,
                                             UInt global_id) noexcept {
                    auto result = CoroFrame::create(&coro.frame());
                    if (coro.subroutine_count() <= 1u) {
                        return result;
                    }
                    if (!common_relocation_fields.empty()) {
                        coro_frame_load_into(
                            result, global_frames, global_id,
                            global_layout, false,
                            luisa::span<const size_t>{common_relocation_fields}, true);
                    }
                    auto load = switch_(token);
                    for (size_t i = 1u;
                         i < coro.subroutine_count(); ++i) {
                        load = std::move(load).case_(
                            static_cast<uint>(i), [&, i] {
                                if (!residual_relocation_fields[i].empty()) {
                                    coro_frame_load_into(
                                        result, global_frames, global_id,
                                        global_layout, false,
                                        luisa::span<const size_t>{
                                            residual_relocation_fields[i]},
                                        true, false);
                                }
                            });
                    }
                    std::move(load).default_([] {});
                    return result;
                };
                auto store_shared_frame = [&](UInt token, UInt shared_id,
                                              const CoroFrame &frame) noexcept {
                    if (coro.subroutine_count() <= 1u) { return; }
                    if (!common_relocation_fields.empty()) {
                        shared_frames->write(
                            shared_id, frame,
                            luisa::span<const size_t>{common_relocation_fields});
                    }
                    auto store = switch_(token);
                    for (size_t i = 1u;
                         i < coro.subroutine_count(); ++i) {
                        store = std::move(store).case_(
                            static_cast<uint>(i), [&, i] {
                                if (!residual_relocation_fields[i].empty()) {
                                    shared_frames->write(
                                        shared_id, frame,
                                        luisa::span<const size_t>{
                                            residual_relocation_fields[i]},
                                        false);
                                }
                            });
                    }
                    std::move(store).default_([] {});
                };
                auto spill_shared_frame = [&](UInt token, UInt shared_id,
                                              UInt global_id) noexcept {
                    if (coro.subroutine_count() <= 1u) { return; }
                    auto frame = CoroFrame::create(&coro.frame());
                    if (!common_relocation_fields.empty()) {
                        shared_frames->read_into(
                            shared_id, frame,
                            luisa::span<const size_t>{common_relocation_fields});
                        coro_frame_store(
                            global_frames, global_id, frame,
                            global_layout, false,
                            luisa::span<const size_t>{common_relocation_fields}, true);
                    }
                    auto spill = switch_(token);
                    for (size_t i = 1u;
                         i < coro.subroutine_count(); ++i) {
                        spill = std::move(spill).case_(
                            static_cast<uint>(i), [&, i] {
                                if (!residual_relocation_fields[i].empty()) {
                                    shared_frames->read_into(
                                        shared_id,
                                        frame,
                                        luisa::span<const size_t>{
                                            residual_relocation_fields[i]},
                                        false);
                                    coro_frame_store(
                                        global_frames, global_id, frame,
                                        global_layout, false,
                                        luisa::span<const size_t>{
                                            residual_relocation_fields[i]},
                                        true, false);
                                }
                            });
                    }
                    std::move(spill).default_([] {});
                };

                for (auto index = 0u;
                     index < shared_queue_factor; index++) {
                    auto s = index * _config.block_size + thread_x();
                    all_token[s] = 0u;
                    if (_config.shared_memory_soa) {
                        auto frame = CoroFrame::create(&coro.frame());
                        shared_frames->write(
                            s, frame,
                            luisa::span{input_fields[0u]});
                    }
                }
                if (_config.global_memory_ext) {
                    $for (index, 0u, global_queue_factor) {
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
                            // Fetch granularity changes only the partition of the
                            // monotonically allocated global task interval. It is
                            // therefore a dispatch policy, not shader structure.
                            auto fetch_count = _config.block_size * fetch_size;
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

                    if (global_memory_frames) {
                        // The global-frame representation is the same finite
                        // slot state machine as the shared/hybrid path. Gather
                        // at most one block of slots in the selected token
                        // class; unselected slots retain both token and frame.
                        $for (index, 0u, global_queue_factor) {
                            auto queue_id =
                                index * _config.block_size + thread_x();
                            auto frame_token = def(all_token[queue_id]);
                            $if (frame_token == work_stat[1u]) {
                                auto id =
                                    work_offset.atomic(0u).fetch_add(1u);
                                $if (id < _config.block_size) {
                                    path_id[id] = queue_id;
                                };
                            };
                        };
                    } else if (!_config.global_memory_ext) {
                        for (auto index = 0u;
                             index < shared_queue_factor; index++) {
                            auto frame_token = def(all_token[index * _config.block_size + thread_x()]);
                            $if (frame_token == work_stat[1u]) {
                                auto id = work_offset.atomic(0u).fetch_add(1u);
                                path_id[id] = index * _config.block_size + thread_x();
                            };
                        }
                    } else {
                        for (auto index = 0u;
                             index < shared_queue_factor; index++) {
                            auto frame_token = def(all_token[index * _config.block_size + thread_x()]);
                            $if (frame_token != work_stat[1u]) {
                                auto id = work_offset.atomic(0u).fetch_add(1u);
                                path_id[id] = index * _config.block_size + thread_x();
                            };
                        }
                        sync_block();
                        $if (shared_queue_size - work_offset[0u] < _config.block_size) {
                            // The global queue factor grows with the continuation count. This is a
                            // device loop by design: host unrolling duplicates the
                            // complete frame spill/restore path once per continuation.
                            $for (index, 0u, global_queue_factor) {
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
                                            // This is an in-place exchange:
                                            // load the incoming value before
                                            // overwriting its global slot.
                                            auto global_frame =
                                                load_global_frame(
                                                    coro_token, global_id);
                                            $if (frame_token != 0u) {
                                                spill_shared_frame(
                                                    frame_token, dst,
                                                    global_id);
                                            };
                                            store_shared_frame(
                                                coro_token, dst,
                                                global_frame);
                                            all_token[global_token_index] = frame_token;
                                            all_token[dst] = coro_token;
                                        }
                                        $elif (frame_token != 0u) {
                                            spill_shared_frame(
                                                frame_token, dst,
                                                global_id);
                                            all_token[global_token_index] = frame_token;
                                            all_token[dst] = coro_token;
                                        };
                                    };
                                };
                            };
                        };
                    }

                    auto gen_start = workload[0u];
                    sync_block();
                    auto pid = def(0u);
                    auto do_work = def(false);
                    if (global_memory_frames) {
                        do_work = thread_x() <
                                  min(work_offset[0u],
                                      _config.block_size);
                        $if (do_work) {
                            pid = path_id[thread_x()];
                        };
                    } else if (_config.global_memory_ext) {
                        pid = thread_x();
                        do_work = all_token[pid] == work_stat[1u];
                    } else {
                        do_work = thread_x() < work_offset[0u];
                        $if (do_work) {
                            pid = path_id[thread_x()];
                        };
                    }
                    $if (do_work) {
                        auto current_token = def(all_token[pid]);
                        // The queue token is a sum type: empty/generate (zero) or
                        // exactly one continuation index. Encode that partition as
                        // a switch so continuation-local opaque state cannot leak
                        // through correlated sibling `if` regions after CFG
                        // restructuring.
                        auto dispatch = switch_(current_token);
                        dispatch = std::move(dispatch).case_(0u, [&] {
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
                                if (global_memory_frames) {
                                    auto global_frame_id =
                                        block_x() * global_queue_size + pid;
                                    coro_frame_store(
                                        global_frames, global_frame_id,
                                        frame, global_layout, false,
                                        luisa::span{output_fields[0u]}, true);
                                } else {
                                    shared_frames->write(
                                        pid, frame,
                                        luisa::span{output_fields[0u]});
                                }
                                all_token[pid] = next;
                                work_counter.atomic(next).fetch_add(1u);
                                if (_config.global_memory_ext) {
                                    workload.atomic(0u).fetch_add(1u);
                                }
                            };
                        });
                        for (size_t i = 1u; i < coro.subroutine_count(); ++i) {
                            dispatch = std::move(dispatch).case_(
                                static_cast<uint>(i), [&, i] {
                                    work_counter.atomic(static_cast<uint>(i)).fetch_sub(1u);
                                    auto frame = [&]() noexcept {
                                        if (global_memory_frames) {
                                            auto global_frame_id =
                                                block_x() * global_queue_size + pid;
                                            return coro_frame_load(
                                                &coro.frame(), global_frames,
                                                global_frame_id,
                                                global_layout, false,
                                                luisa::span{input_fields[i]}, true);
                                        }
                                        return shared_frames->read(
                                            pid,
                                            luisa::span{input_fields[i]});
                                    }();
                                    coro[i](frame, args...);
                                    auto next = token_to_index(frame.target_token);
                                    $if (frame.target_token == CoroFrame::TERMINAL_TOKEN) {
                                        frame.target_token = 0u;
                                    };
                                    if (global_memory_frames) {
                                        auto global_frame_id =
                                            block_x() * global_queue_size + pid;
                                        coro_frame_store(
                                            global_frames, global_frame_id,
                                            frame, global_layout, false,
                                            luisa::span{output_fields[i]}, true);
                                    } else {
                                        shared_frames->write(
                                            pid, frame,
                                            luisa::span{output_fields[i]});
                                    }
                                    all_token[pid] = next;
                                    work_counter.atomic(next).fetch_add(1u);
                                });
                        }
                        std::move(dispatch).default_([] {});
                    };
                    sync_block();
                };
                sync_block();
            }};
        };
        auto main_kernel = make_main_kernel();
        auto shared_memory_size = _shared_memory_size(main_kernel);
        auto device_limit = device.compute_max_shared_memory_size();
        auto shared_memory_limit = device_limit;
        if (_config.shared_memory_limit_bytes != 0u) {
            shared_memory_limit = device_limit == 0u ?
                                      _config.shared_memory_limit_bytes :
                                      std::min(device_limit, _config.shared_memory_limit_bytes);
        }
        if (shared_memory_limit != 0u && shared_memory_size > shared_memory_limit) {
            const auto requested_block_size = _config.block_size;
            const auto requested_shared_memory_size = shared_memory_size;
            // DSL kernels require a workgroup size divisible by 32 even when a
            // scalar backend reports a logical wave size of one. Supported
            // wave sizes are powers of two, so max(wave, 32) is their LCM and
            // is the true portable resource-fitting quantum.
            constexpr auto dsl_block_granularity = 32u;
            const auto device_wave_size =
                std::max(device.compute_warp_size(), 1u);
            const auto block_quantum =
                std::max(device_wave_size,
                         dsl_block_granularity);
            LUISA_ASSERT(
                requested_block_size % block_quantum == 0u,
                "Persistent coroutine block size ({}) must contain a whole number "
                "of scheduler quanta (device wave {}, DSL granularity {}, "
                "effective quantum {}).",
                requested_block_size, device_wave_size,
                dsl_block_granularity, block_quantum);
            const auto requested_global_memory_frames =
                _config.global_memory_frames;
            while (shared_memory_size > shared_memory_limit) {
                auto next_block_size = _next_block_size(
                    requested_block_size, _config.block_size,
                    block_quantum);
                if (next_block_size != 0u) {
                    _config.block_size = next_block_size;
                } else if (_config.global_memory_ext &&
                           !_config.global_memory_frames) {
                    // A shared-frame workgroup has the irreducible resource
                    // lower bound wave_size * frame_size. Crossing that bound
                    // is not a reason for a backend-specific exception: switch
                    // representation while preserving the queue-slot state
                    // machine, then retry the original requested block size.
                    _config.global_memory_frames = true;
                    _config.block_size = requested_block_size;
                    update_global_frame_layout();
                } else {
                    LUISA_ERROR_WITH_LOCATION(
                        "Persistent coroutine scheduler requires {} bytes of "
                        "shared memory for a {}-thread block, exceeding the "
                        "device limit of {} bytes; even one {}-thread "
                        "scheduler quantum "
                        "cannot fit after {} frame storage. Reduce the live "
                        "coroutine frame or the number of continuation queues.",
                        shared_memory_size, _config.block_size,
                        shared_memory_limit, block_quantum,
                        _config.global_memory_frames ?
                            "global" :
                            "shared");
                }
                main_kernel = make_main_kernel();
                shared_memory_size = _shared_memory_size(main_kernel);
            }
            LUISA_WARNING(
                "Persistent coroutine resources fitted: block {} -> {}, "
                "frame storage {} -> {}, static shared memory {} -> {} bytes "
                "({} bytes available).",
                requested_block_size, _config.block_size,
                requested_global_memory_frames ? "global" : "shared",
                _config.global_memory_frames ? "global" : "shared",
                requested_shared_memory_size, shared_memory_size,
                shared_memory_limit);
        }
        _global_frames =
            device.create_byte_buffer(
                _global_frame_layout.size_bytes);
        _static_shared_memory_size_bytes = shared_memory_size;
        _main_shader_structure_hash =
            main_kernel.function()->function().hash();
        auto main_shader_option =
            detail::coro_scheduler_shader_option(
                _config.shader_option, "persistent_main");
        _pt_shader = device.compile(main_kernel, main_shader_option);

        _clear_shader = device.compile<1>([](BufferUInt g) {
            g.write(dispatch_x(), 0u);
        },
                                          detail::coro_scheduler_shader_option(_config.shader_option, "persistent_clear"));
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
        stream << _pt_shader(
                      _global, _global_frames,
                      dispatch_size_prefix_product,
                      _config.fetch_size, args...)
                      .dispatch(worker_count);
    }

public:
    [[nodiscard]] const Config &config() const noexcept { return _config; }
    [[nodiscard]] size_t static_shared_memory_size_bytes() const noexcept {
        return _static_shared_memory_size_bytes;
    }
    /// Structural identity of the persistent state-machine kernel. Worker
    /// capacity and task-fetch granularity are dispatch policy and therefore
    /// deliberately excluded; block size and storage representation remain
    /// structural.
    [[nodiscard]] uint64_t main_shader_structure_hash() const noexcept {
        return _main_shader_structure_hash;
    }

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
