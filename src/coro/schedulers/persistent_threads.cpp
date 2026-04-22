//
// Created by Mike on 2024/5/10.
//

#include <luisa/dsl/sugar.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/coro/coro_frame_smem.h>
#include <luisa/coro/coro_frame_buffer.h>
#include <luisa/coro/schedulers/persistent_threads.h>

namespace luisa::compute::coroutine::detail {

void persistent_threads_coro_scheduler_main_kernel_impl(
    const PersistentThreadsCoroSchedulerConfig &config,
    uint q_fac, uint g_fac, uint shared_queue_size, uint global_queue_size,
    const CoroGraph *graph, Shared<CoroFrame> &frames, Expr<uint3> dispatch_size_prefix_product,
    Expr<Buffer<uint>> global, const Buffer<CoroFrame> &global_frames,
    luisa::move_only_function<void(CoroFrame &, CoroToken)> call_subroutine) noexcept {

    auto subroutine_count = static_cast<uint>(graph->nodes().size());
    Shared<uint> path_id{shared_queue_size};
    Shared<uint> work_counter{subroutine_count};
    Shared<uint> work_offset{2u};
    Shared<uint> all_token{config.global_memory_ext ?
                               shared_queue_size + global_queue_size :
                               shared_queue_size};
    Shared<uint> workload{2};
    Shared<uint> work_stat{2};// work_state[0] for max_count, [1] for max_id
    for (auto index = 0u; index < q_fac; index++) {
        auto s = index * config.block_size + thread_x();
        all_token[s] = 0u;
        if (config.shared_memory_soa) {
            frames.write(s, CoroFrame::create(graph->shared_frame()), std::array{0u, 1u});
        } else {
            frames[s].coro_id = make_uint3();
            frames[s].target_token = 0u;
        }
    }
    for (auto index = 0u; index < g_fac; index++) {
        auto s = index * config.block_size + thread_x();
        all_token[shared_queue_size + s] = 0u;
    }
    $if (thread_x() < subroutine_count) {
        $if (thread_x() == 0u) {
            work_counter[thread_x()] =
                config.global_memory_ext ?
                    shared_queue_size + global_queue_size :
                    shared_queue_size;
        }
        $else {
            work_counter[thread_x()] = 0u;
        };
    };
    workload[0] = 0u;
    workload[1] = 0u;
    Shared<uint> rem_global{1};
    Shared<uint> rem_local{1};
    rem_global[0] = 1u;
    rem_local[0] = 0u;
    sync_block();
    auto count = def(0u);
    auto count_limit = def(std::numeric_limits<uint>::max());
    $while ((rem_global[0] != 0u | rem_local[0] != 0u) & (count != count_limit)) {
        sync_block();//very important, synchronize for condition
        rem_local[0] = 0u;
        count += 1;
        work_stat[0] = 0;
        work_stat[1] = -1;
        sync_block();
        $if (thread_x() == config.block_size - 1) {
            $if (workload[0] >= workload[1] & rem_global[0] == 1u) {//fetch new workload
                workload[0] = global.atomic(0u).fetch_add(config.block_size * config.fetch_size);
                workload[1] = min(workload[0] + config.block_size * config.fetch_size, dispatch_size_prefix_product.z);
                $if (workload[0] >= dispatch_size_prefix_product.z) {
                    rem_global[0] = 0u;
                };
            };
        };
        sync_block();
        $if (thread_x() < subroutine_count) {//get max
            $if (workload[0] < workload[1] | thread_x() != 0u) {
                $if (work_counter[thread_x()] != 0) {
                    rem_local[0] = 1u;
                    work_stat.atomic(0).fetch_max(work_counter[thread_x()]);
                };
            };
        };
        sync_block();
        $if (thread_x() < subroutine_count) {//get argmax
            $if (work_stat[0] == work_counter[thread_x()] & (workload[0] < workload[1] | thread_x() != 0u)) {
                work_stat[1] = thread_x();
            };
        };
        sync_block();
        work_offset[0] = 0;
        work_offset[1] = 0;
        sync_block();
        if (!config.global_memory_ext) {
            for (auto index = 0u; index < q_fac; index++) {//collect indices
                auto frame_token = all_token[index * config.block_size + thread_x()];
                $if (frame_token == work_stat[1]) {
                    auto id = work_offset.atomic(0).fetch_add(1u);
                    path_id[id] = index * config.block_size + thread_x();
                };
            }
        } else {
            for (auto index = 0u; index < q_fac; index++) {//collect switch out indices
                auto frame_token = all_token[index * config.block_size + thread_x()];
                $if (frame_token != work_stat[1]) {
                    auto id = work_offset.atomic(0).fetch_add(1u);
                    path_id[id] = index * config.block_size + thread_x();
                };
            }
            sync_block();
            $if (shared_queue_size - work_offset[0] < config.block_size) {//no enough work
                $for (index, 0u, g_fac) {                                 //swap frames
                    auto global_id = block_x() * global_queue_size + index * config.block_size + thread_x();
                    auto g_queue_id = index * config.block_size + thread_x();
                    auto coro_token = all_token[shared_queue_size + g_queue_id];
                    $if (coro_token == work_stat[1]) {
                        auto id = work_offset.atomic(1).fetch_add(1u);
                        $if (id < work_offset[0]) {
                            auto dst = path_id[id];
                            auto frame_token = all_token[dst];
                            $if (coro_token != 0u) {
                                auto g_state = global_frames->read(global_id);
                                $if (frame_token != 0u) {
                                    if (config.shared_memory_soa) {
                                        global_frames->write(global_id, frames.read(dst));
                                    } else {
                                        global_frames->write(global_id, frames[dst]);
                                    }
                                };
                                if (config.shared_memory_soa) {
                                    frames.write(dst, g_state);
                                } else {
                                    frames[dst] = g_state;
                                }
                                all_token[shared_queue_size + g_queue_id] = frame_token;
                                all_token[dst] = coro_token;
                            }
                            $elif (frame_token != 0u) {
                                if (config.shared_memory_soa) {
                                    auto frame = frames.read(dst);
                                    global_frames->write(global_id, frame);
                                } else {
                                    global_frames->write(global_id, frames[dst]);
                                }
                                all_token[shared_queue_size + g_queue_id] = frame_token;
                                all_token[dst] = coro_token;
                            };
                        };
                    };
                };
            };
        }
        auto gen_st = workload[0];
        sync_block();
        auto pid = config.global_memory_ext ? thread_x() : path_id[thread_x()];
        $if (config.global_memory_ext ? (all_token[pid] == work_stat[1]) : (thread_x() < work_offset[0])) {
            $switch (all_token[pid]) {
                $case (0u) {
                    $if (gen_st + thread_x() < workload[1]) {
                        work_counter.atomic(0u).fetch_sub(1u);
                        auto global_index = gen_st + thread_x();
                        auto index_z = global_index / dispatch_size_prefix_product.y;
                        auto index_xy = global_index % dispatch_size_prefix_product.y;
                        auto index_x = index_xy % dispatch_size_prefix_product.x;
                        auto index_y = index_xy / dispatch_size_prefix_product.x;
                        auto next = def(0u);
                        if (config.shared_memory_soa) {
                            auto frame = CoroFrame::create(graph->shared_frame(), make_uint3(index_x, index_y, index_z));
                            call_subroutine(frame, coro_token_entry);
                            next = frame.target_token & coro_token_valid_mask;
                            frames.write(pid, frame, graph->entry().output_fields());
                        } else {
                            frames[pid].coro_id = make_uint3(index_x, index_y, index_z);
                            frames[pid].target_token = coro_token_entry;
                            call_subroutine(frames[pid], coro_token_entry);
                            next = frames[pid].target_token & coro_token_valid_mask;
                        }
                        all_token[pid] = next;
                        work_counter.atomic(next).fetch_add(1u);
                        workload.atomic(0).fetch_add(1u);
                    };
                };
                for (auto i = 1u; i < subroutine_count; i++) {
                    $case (i) {
                        work_counter.atomic(i).fetch_sub(1u);
                        auto next = def(0u);
                        if (config.shared_memory_soa) {
                            auto frame = frames.read(pid, graph->node(i).input_fields());
                            call_subroutine(frame, i);
                            next = frame.target_token & coro_token_valid_mask;
                            frames.write(pid, frame, graph->node(i).output_fields());
                        } else {
                            call_subroutine(frames[pid], i);
                            next = frames[pid].target_token & coro_token_valid_mask;
                        }
                        all_token[pid] = next;
                        work_counter.atomic(next).fetch_add(1u);
                    };
                }
                $default { dsl::unreachable(); };
            };
        };
        sync_block();
    };
#ifndef NDEBUG
    $if (count >= count_limit) {
        device_log("block_id{},thread_id {}, loop not break! local:{}, global:{}", block_x(), thread_x(), rem_local[0], rem_global[0]);
        $if (thread_x() < subroutine_count) {
            device_log("work rem: id {}, size {}", thread_x(), work_counter[thread_x()]);
        };
    };
#endif
}

}// namespace luisa::compute::coroutine::detail
