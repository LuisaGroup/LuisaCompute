#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/vector.h>
#include <luisa/coro/coro_frame_storage.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/coro/schedulers/detail/token_index.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

struct GraphWavefrontCoroSchedulerConfig {
    uint thread_count = static_cast<uint>(2_M);
    bool global_memory_soa = true;
    uint execution_block_size = 256u;
    uint counter_readback_batch_size = 4u;
    // Number of readback slots kept in flight. A value greater than one lets
    // the GPU execute later fixed sweeps while the host waits for an older
    // counter snapshot. This host scheduling choice never enters a shader AST.
    uint counter_readback_pipeline_depth = 2u;
    // Once all logical invocations have been generated, a small residual set
    // is cheaper to finish in one graph-derived state-machine kernel than in
    // one guarded launch per continuation. This is a host policy parameter:
    // it does not participate in shader construction or cache identity. Zero
    // disables the hybrid tail drain.
    uint tail_megakernel_threshold = 4096u;
    bool report_stats = false;
    ShaderOption shader_option{};
};

struct GraphWavefrontCoroDispatchStats {
    bool collected{false};
    uint64_t sweep_count{0u};
    uint64_t counter_snapshot_count{0u};
    uint64_t counter_readback_count{0u};
    uint64_t counter_readback_bytes{0u};
    uint64_t host_wait_count{0u};
    uint64_t generated_count{0u};
    uint max_live_count{0u};
    uint max_readbacks_in_flight{0u};
    uint tail_dispatch_count{0u};
    uint tail_instance_count{0u};
    double elapsed_ms{0.0};
};

/// A CoroGraph-driven wavefront scheduler with explicit continuation queues.
///
/// Queue zero is the free-frame/entry queue. Every other queue corresponds
/// one-to-one with a non-entry CoroGraph node; the node's materialized
/// subroutine is its consumer and the graph's target token selects the output
/// queue. The renderer never supplies a consumer callback.
///
/// Non-entry queue storage is double-buffered. A graph sweep reads frame-slot
/// indices exclusively from one active bank and appends them exclusively to
/// the other, so self-edges and back-edges are race-free without copying frame
/// storage. Queue zero is a single stable free-index stack: entry pops from its
/// tail and termination pushes back to it. Every graph-provided consumer uses
/// a fixed, guarded launch.
///
/// Frame storage is stable: queues contain only uint indices into the frame
/// buffer, and no scheduler operation relocates a frame. One source index
/// produces exactly one successor index or returns to queue zero. Thus the sum
/// of all queue cardinalities is invariant and equal to the active frame
/// capacity. Giving every queue that capacity proves that no append can
/// overflow. Queue counters therefore affect scheduling and termination only,
/// never correctness. A batch consists of complete graph sweeps. Each sweep
/// writes all aggregate counters to a device snapshot ring; the entire ring is
/// copied to one host slot at the batch boundary. Multiple slots may remain in
/// flight, hiding an older readback behind later GPU work.
template<typename... Args>
class GraphWavefrontCoroScheduler final : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = GraphWavefrontCoroSchedulerConfig;

private:
    Config _config;
    CoroFrameStorageLayout _frame_layout;
    ByteBuffer _frame_buffer;
    Buffer<uint> _queue_indices;
    Buffer<uint> _queue_counts;
    Buffer<uint> _work_state;
    Buffer<uint> _dispatch_state;
    Buffer<uint> _counter_snapshots;
    Buffer<uint> _tail_offsets;
    Stream _counter_readback_stream;
    Event _counter_snapshot_ready_event;
    Event _counter_readback_done_event;
    Shader1D<uint, uint> _initialize_shader;
    Shader1D<uint> _clear_counts_shader;
    Shader1D<uint, uint, uint> _prepare_entry_shader;
    Shader1D<uint, uint, uint3, Args...> _entry_shader;
    luisa::vector<Shader1D<uint, uint, uint, Args...>>
        _continuation_shaders;
    Shader1D<uint, uint> _prepare_returns_shader;
    Shader1D<uint> _copy_returns_shader;
    Shader1D<> _commit_returns_shader;
    Shader1D<uint, uint> _snapshot_shader;
    Shader1D<uint> _prepare_tail_offsets_shader;
    Shader1D<uint, uint, Args...> _tail_shader;
    luisa::vector<uint> _host_snapshots;
    GraphWavefrontCoroDispatchStats _last_dispatch_stats;
    uint _node_count{};
    uint _snapshot_stride{};
    uint _active_frame_capacity{};

private:
    [[nodiscard]] static auto _dispatch_id_from_linear_index(
        UInt global_index, UInt3 dispatch_shape) noexcept {
        auto index_z = global_index /
                       (dispatch_shape.x * dispatch_shape.y);
        auto index_xy = global_index -
                        index_z * dispatch_shape.x * dispatch_shape.y;
        auto index_y = index_xy / dispatch_shape.x;
        auto index_x = index_xy - index_y * dispatch_shape.x;
        return make_uint3(index_x, index_y, index_z);
    }

    template<typename Kernel>
    [[nodiscard]] auto _compile(
        Device &device, const Kernel &kernel,
        luisa::string_view stage) const noexcept {
        return device.compile(
            kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, stage));
    }

    void _create_shaders(Device &device, const Coro &coro) {
        auto *frame_buffer = &_frame_buffer;
        auto *queue_indices = &_queue_indices;
        auto *queue_counts = &_queue_counts;
        auto *work_state = &_work_state;
        auto *dispatch_state = &_dispatch_state;
        auto *counter_snapshots = &_counter_snapshots;
        auto *tail_offsets = &_tail_offsets;
        auto block_size = _config.execution_block_size;
        auto node_count = _node_count;
        auto active_node_count = node_count - 1u;
        auto return_queue = 1u + 2u * active_node_count;
        auto snapshot_stride = _snapshot_stride;
        auto layout = _frame_layout;
        auto soa = _config.global_memory_soa;

        Kernel1D initialize = [queue_indices, queue_counts, dispatch_state,
                               active_node_count](
                                  UInt active_capacity,
                                  UInt clear_count) noexcept {
            auto x = dispatch_x();
            auto indices = Expr<Buffer<uint>>{*queue_indices};
            auto counts = Expr<Buffer<uint>>{*queue_counts};
            auto state = Expr<Buffer<uint>>{*dispatch_state};
            $if (x < clear_count) {
                $if (x < 2u + 2u * active_node_count) {
                    counts.write(x, 0u);
                };
                $if (x < active_capacity) { indices.write(x, x); };
                $if (x == 0u) {
                    counts.write(0u, active_capacity);
                    state.write(0u, 0u);// next logical invocation
                };
            };
        };
        _initialize_shader = _compile(
            device, initialize, "graph_wavefront_initialize");

        Kernel1D clear_counts = [queue_counts, active_node_count,
                                 return_queue](
                                    UInt bank) noexcept {
            auto x = dispatch_x();
            $if (x < active_node_count) {
                Expr<Buffer<uint>>{*queue_counts}.write(
                    1u + bank * active_node_count + x, 0u);
            };
            $if (x == 0u) {
                Expr<Buffer<uint>>{*queue_counts}.write(return_queue, 0u);
            };
        };
        _clear_counts_shader = _compile(
            device, clear_counts, "graph_wavefront_clear_counts");

        Kernel1D prepare_entry =
            [queue_counts, work_state, dispatch_state,
             active_node_count](UInt source_bank,
                                UInt logical_count,
                                UInt active_capacity) noexcept {
                auto counts = Expr<Buffer<uint>>{*queue_counts};
                auto work = Expr<Buffer<uint>>{*work_state};
                auto state = Expr<Buffer<uint>>{*dispatch_state};
                auto free_count = counts.read(0u);
                auto remaining_capacity = def(active_capacity);
                auto ownership_valid = def(free_count <= active_capacity);
                for (auto node = 1u; node <= active_node_count; ++node) {
                    auto source_queue = 1u + source_bank * active_node_count +
                                        node - 1u;
                    auto source_count = counts.read(source_queue);
                    ownership_valid &= source_count <= active_capacity;
                    ownership_valid &= source_count <= remaining_capacity;
                    remaining_capacity -= min(
                        source_count, remaining_capacity);
                    // Every consumer is gated by a count published by this
                    // pre-execution ownership certificate, never by a queue
                    // counter that might already exceed its capacity.
                    work.write(5u + node - 1u, 0u);
                }
                ownership_valid &= free_count == remaining_capacity;
                auto logical_first = state.read(0u);
                auto remaining = ite(
                    logical_first < logical_count,
                    logical_count - logical_first, 0u);
                work.write(0u, logical_first);
                work.write(1u, 0u);
                work.write(2u, free_count);
                $if (ownership_valid) {
                    auto admit_count = min(free_count, remaining);
                    auto free_first = free_count - admit_count;
                    counts.write(0u, free_first);
                    state.write(0u, logical_first + admit_count);
                    work.write(1u, admit_count);
                    work.write(2u, free_first);
                    for (auto node = 1u; node <= active_node_count; ++node) {
                        auto source_queue =
                            1u + source_bank * active_node_count + node - 1u;
                        work.write(5u + node - 1u,
                                   counts.read(source_queue));
                    }
                }
                $else {
                    // Gates were cleared above, so no later callable in this
                    // sweep can advance or perform side effects. This is a failed
                    // scheduler invariant, not a recoverable renderer condition.
                    unreachable(
                        "Graph wavefront ownership certificate failed before "
                        "consumer execution.");
                };
            };
        _prepare_entry_shader = _compile(
            device, prepare_entry, "graph_wavefront_prepare_entry");

        auto token_to_index = detail::make_coro_token_index_callable(coro);
        auto push_frame = Callable<void(uint, uint, uint, uint, uint)>{
            [queue_indices, queue_counts](
                UInt destination_bank, UInt queue, UInt frame_index,
                UInt storage_capacity, UInt active_queue_count) noexcept {
                auto counts = Expr<Buffer<uint>>{*queue_counts};
                auto return_queue = 1u + 2u * active_queue_count;
                auto absolute_queue = ite(
                    queue == 0u, return_queue,
                    1u + destination_bank * active_queue_count + queue - 1u);
                auto slot = counts.atomic(absolute_queue).fetch_add(1u);
                $if (slot < storage_capacity) {
                    Expr<Buffer<uint>>{*queue_indices}.write(
                        absolute_queue * storage_capacity + slot,
                        frame_index);
                }
                $else {
                    unreachable(
                        "Graph wavefront queue exceeded the proven frame-pool bound.");
                };
            }};

        auto entry_output_fields = luisa::vector<luisa::vector<size_t>>{};
        entry_output_fields.resize(_node_count);
        for (auto target = 0u; target < _node_count; ++target) {
            entry_output_fields[target] = coro_frame_collect_output_fields(
                coro.graph(), 0u, target);
        }
        Kernel1D entry = [&coro, frame_buffer, queue_indices, work_state,
                          block_size, active_node_count,
                          layout, soa,
                          output_fields = std::move(entry_output_fields),
                          token_to_index, push_frame](
                             UInt destination_bank,
                             UInt storage_capacity,
                             UInt3 dispatch_shape,
                             Var<Args>... args) noexcept {
            set_block_size(block_size);
            auto x = dispatch_x();
            auto admit_count =
                Expr<Buffer<uint>>{*work_state}.read(1u);
            $if (x < admit_count) {
                auto free_first =
                    Expr<Buffer<uint>>{*work_state}.read(2u);
                auto frame_index =
                    Expr<Buffer<uint>>{*queue_indices}.read(
                        free_first + x);
                auto logical_first =
                    Expr<Buffer<uint>>{*work_state}.read(0u);
                auto logical_id = _dispatch_id_from_linear_index(
                    logical_first + x, dispatch_shape);
                auto frame = coro.instantiate(logical_id, dispatch_shape);
                frame.target_token = 0u;
                coro.entry()(frame, args...);
                auto next = token_to_index(frame.target_token);
                frame.target_token = next;
                for (auto target = 1u;
                     target < output_fields.size(); ++target) {
                    $if (next == static_cast<uint>(target)) {
                        coro_frame_store(
                            Expr<ByteBuffer>{*frame_buffer}, frame_index,
                            storage_capacity, frame, layout, soa,
                            luisa::span{output_fields[target]});
                    };
                };
                push_frame(destination_bank, next, frame_index,
                           storage_capacity, active_node_count);
            };
        };
        _entry_shader = _compile(
            device, entry, "graph_wavefront_entry");

        _continuation_shaders.resize(_node_count);
        for (auto node_index = 1u; node_index < _node_count; ++node_index) {
            auto subroutine = coro[node_index];
            LUISA_ASSERT(subroutine,
                         "CoroGraph node {} has no materialized consumer.",
                         node_index);
            auto input_fields = coro_frame_collect_input_fields(
                coro.graph(), node_index);
            auto output_fields = luisa::vector<luisa::vector<size_t>>{};
            output_fields.resize(_node_count);
            for (auto target = 0u; target < _node_count; ++target) {
                output_fields[target] = coro_frame_collect_output_fields(
                    coro.graph(), node_index, target);
            }
            Kernel1D continuation =
                [&coro, frame_buffer, queue_indices, queue_counts, work_state,
                 block_size,
                 active_node_count, layout, soa, node_index, subroutine,
                 input_fields = std::move(input_fields),
                 output_fields = std::move(output_fields), token_to_index,
                 push_frame](UInt source_bank, UInt destination_bank,
                             UInt storage_capacity,
                             Var<Args>... args) noexcept {
                    set_block_size(block_size);
                    auto x = dispatch_x();
                    auto source_queue = 1u + source_bank * active_node_count +
                                        node_index - 1u;
                    auto count = Expr<Buffer<uint>>{*work_state}.read(
                        5u + node_index - 1u);
                    $if (x < count) {
                        auto frame_index =
                            Expr<Buffer<uint>>{*queue_indices}.read(
                                source_queue * storage_capacity + x);
                        auto frame = coro_frame_load(
                            &coro.frame(), Expr<ByteBuffer>{*frame_buffer},
                            frame_index, storage_capacity, layout, soa,
                            luisa::span{input_fields});
                        frame.target_token = CoroFrame::TERMINAL_TOKEN;
                        subroutine(frame, args...);
                        auto next = token_to_index(frame.target_token);
                        frame.target_token = next;
                        for (auto target = 1u;
                             target < output_fields.size(); ++target) {
                            $if (next == static_cast<uint>(target)) {
                                coro_frame_store(
                                    Expr<ByteBuffer>{*frame_buffer},
                                    frame_index, storage_capacity, frame,
                                    layout, soa,
                                    luisa::span{output_fields[target]});
                            };
                        }
                        push_frame(destination_bank, next, frame_index,
                                   storage_capacity, active_node_count);
                    };
                };
            _continuation_shaders[node_index] = _compile(
                device, continuation,
                luisa::format("graph_wavefront_node_{}", node_index));
        }

        // Returning frame indices to the free stack is a publication
        // transaction. The old free count and the number of returns must be
        // frozen before the parallel copy; publishing the new free count in
        // the copy kernel would race with other lanes reading the old count.
        Kernel1D prepare_returns = [queue_counts, work_state, return_queue](
                                       UInt storage_capacity,
                                       UInt returned_capacity) noexcept {
            auto counts = Expr<Buffer<uint>>{*queue_counts};
            auto work = Expr<Buffer<uint>>{*work_state};
            auto returned_count = counts.read(return_queue);
            auto free_first = counts.read(0u);
            $if (returned_count <= returned_capacity) {
                $if (returned_count <= storage_capacity) {
                    $if (free_first <= storage_capacity - returned_count) {
                        work.write(3u, free_first);
                        work.write(4u, returned_count);
                    }
                    $else {
                        unreachable(
                            "Graph wavefront stable free stack overflowed "
                            "while preparing returned frame-slot indices.");
                    };
                }
                $else {
                    unreachable(
                        "Graph wavefront returned frame-slot count exceeds "
                        "the storage capacity.");
                };
            }
            $else {
                unreachable(
                    "Graph wavefront returned frame-slot count exceeds "
                    "the active dispatch capacity.");
            };
        };
        _prepare_returns_shader = _compile(
            device, prepare_returns, "graph_wavefront_prepare_returns");

        Kernel1D copy_returns = [queue_indices, work_state, return_queue](
                                    UInt storage_capacity) noexcept {
            auto x = dispatch_x();
            auto indices = Expr<Buffer<uint>>{*queue_indices};
            auto work = Expr<Buffer<uint>>{*work_state};
            auto free_first = work.read(3u);
            auto returned_count = work.read(4u);
            $if (x < returned_count) {
                auto frame_index = indices.read(
                    return_queue * storage_capacity + x);
                indices.write(free_first + x, frame_index);
            };
        };
        _copy_returns_shader = _compile(
            device, copy_returns, "graph_wavefront_copy_returns");

        Kernel1D commit_returns = [queue_counts, work_state]() noexcept {
            auto work = Expr<Buffer<uint>>{*work_state};
            Expr<Buffer<uint>>{*queue_counts}.write(
                0u, work.read(3u) + work.read(4u));
        };
        _commit_returns_shader = _compile(
            device, commit_returns, "graph_wavefront_commit_returns");

        Kernel1D snapshot = [queue_counts, dispatch_state, counter_snapshots,
                             node_count, active_node_count, snapshot_stride](
                                UInt snapshot_index, UInt bank) noexcept {
            auto x = dispatch_x();
            auto output = Expr<Buffer<uint>>{*counter_snapshots};
            auto offset = snapshot_index * snapshot_stride;
            $if (x < node_count) {
                auto queue = ite(
                    x == 0u, 0u,
                    1u + bank * active_node_count + x - 1u);
                output.write(
                    offset + x,
                    Expr<Buffer<uint>>{*queue_counts}.read(queue));
            };
            $if (x == 0u) {
                output.write(
                    offset + node_count,
                    Expr<Buffer<uint>>{*dispatch_state}.read(0u));
            };
        };
        _snapshot_shader = _compile(
            device, snapshot, "graph_wavefront_snapshot");

        if (_config.tail_megakernel_threshold != 0u) {
            Kernel1D prepare_tail_offsets =
                [queue_counts, tail_offsets, active_node_count](
                    UInt source_bank) noexcept {
                    auto counts = Expr<Buffer<uint>>{*queue_counts};
                    auto offsets = Expr<Buffer<uint>>{*tail_offsets};
                    auto prefix = def(0u);
                    offsets.write(0u, 0u);
                    for (auto node = 1u; node <= active_node_count; ++node) {
                        auto queue = 1u + source_bank * active_node_count +
                                     node - 1u;
                        prefix += counts.read(queue);
                        offsets.write(node, prefix);
                    }
                };
            _prepare_tail_offsets_shader = _compile(
                device, prepare_tail_offsets,
                "graph_wavefront_prepare_tail_offsets");

            auto relocation_fields = coro_frame_collect_relocation_fields(
                coro.graph(), coro.frame().frame_field_count());
            Kernel1D tail =
                [&coro, frame_buffer, queue_indices, tail_offsets,
                 block_size, active_node_count, node_count,
                 layout, soa,
                 relocation_fields = std::move(relocation_fields),
                 token_to_index](UInt source_bank,
                                 UInt storage_capacity,
                                 Var<Args>... args) noexcept {
                    set_block_size(block_size);
                    auto x = dispatch_x();
                    auto offsets = Expr<Buffer<uint>>{*tail_offsets};

                    // Prefix ends are monotone. Locate the unique continuation queue
                    // containing this flattened tail instance without specializing
                    // the shader on the runtime tail size.
                    auto first = def(1u);
                    auto last = def(node_count);
                    $while (first < last) {
                        auto middle = first + (last - first) / 2u;
                        $if (x < offsets.read(middle)) {
                            last = middle;
                        }
                        $else {
                            first = middle + 1u;
                        };
                    };
                    auto current = def(first);
                    $if (current < node_count) {
                        auto local_index = x - offsets.read(current - 1u);
                        auto source_queue = 1u + source_bank * active_node_count +
                                            current - 1u;
                        auto frame_index =
                            Expr<Buffer<uint>>{*queue_indices}.read(
                                source_queue * storage_capacity + local_index);
                        auto frame = CoroFrame::create(&coro.frame());

                        // A queued frame is a token-indexed sum type. Load the exact
                        // live payload certified by CoroGraph for the current token;
                        // loading only immediate callable inputs would lose dormant
                        // values consumed by a later continuation.
                        auto load = switch_(current);
                        for (auto node = 1u; node < node_count; ++node) {
                            load = std::move(load).case_(node, [&, node] {
                                coro_frame_load_into(
                                    frame, Expr<ByteBuffer>{*frame_buffer},
                                    frame_index, storage_capacity, layout, soa,
                                    luisa::span{relocation_fields[node]});
                            });
                        }
                        std::move(load).default_([] {
                            unreachable(
                                "Graph wavefront tail received an unknown token.");
                        });

                        // This is generated from the same materialized subroutines as
                        // the wavefront consumers. It is not a second renderer path:
                        // only the scheduling policy changes for the residual set.
                        $while (current != 0u) {
                            auto execute = switch_(current);
                            for (auto node = 1u; node < node_count; ++node) {
                                auto subroutine = coro[node];
                                execute = std::move(execute).case_(
                                    node, [&, subroutine] {
                                        frame.target_token =
                                            CoroFrame::TERMINAL_TOKEN;
                                        subroutine(frame, args...);
                                        current = token_to_index(
                                            frame.target_token);
                                    });
                            }
                            std::move(execute).default_([] {
                                unreachable(
                                    "Graph wavefront tail state machine reached an "
                                    "unknown continuation.");
                            });
                        };
                    };
                };
            _tail_shader = _compile(
                device, tail, "graph_wavefront_tail_state_machine");
        }
    }

    void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        auto logical_count_64 =
            static_cast<uint64_t>(dispatch_size.x) * dispatch_size.y *
            dispatch_size.z;
        LUISA_ASSERT(
            logical_count_64 <= std::numeric_limits<uint>::max(),
            "Graph wavefront dispatch size {} x {} x {} exceeds uint.",
            dispatch_size.x, dispatch_size.y, dispatch_size.z);
        auto logical_count = static_cast<uint>(logical_count_64);
        if (logical_count == 0u) { return; }
        _active_frame_capacity =
            std::min(_config.thread_count, logical_count);
        auto active_node_count = _node_count - 1u;
        auto queue_count = 2u + 2u * active_node_count;
        auto initialize_count =
            std::max(_active_frame_capacity, queue_count);
        stream << _initialize_shader(
                      _active_frame_capacity, initialize_count)
                      .dispatch(initialize_count);

        auto report_stats = _config.report_stats ||
                            std::getenv("LUISA_CORO_WAVEFRONT_STATS") !=
                                nullptr;
        _last_dispatch_stats = {};
        _last_dispatch_stats.collected = report_stats;
        Clock clock;
        struct PendingReadback {
            uint slot{};
            uint64_t done_fence{};
        };
        auto pipeline_depth =
            _config.counter_readback_pipeline_depth;
        auto snapshots_per_slot =
            _config.counter_readback_batch_size;
        auto elements_per_slot =
            static_cast<size_t>(_snapshot_stride) * snapshots_per_slot;
        auto pending = luisa::vector<PendingReadback>(pipeline_depth);
        auto submitted = uint64_t{0u};
        auto consumed = uint64_t{0u};
        auto observed_sweeps = uint64_t{0u};
        auto latest_live_count = 0u;
        auto latest_generated_count = 0u;
        auto done = false;
        auto tail_candidate = false;

        auto consume_oldest = [&](bool allow_wait) noexcept {
            LUISA_ASSERT(consumed < submitted,
                         "No graph-wavefront readback is pending.");
            auto &p = pending[consumed % pipeline_depth];
            if (!_counter_readback_done_event.is_completed(p.done_fence)) {
                if (!allow_wait) { return false; }
                _last_dispatch_stats.host_wait_count++;
                _counter_readback_done_event.synchronize(p.done_fence);
            }
            auto slot_offset = static_cast<size_t>(p.slot) * elements_per_slot;
            for (auto epoch = 0u; epoch < snapshots_per_slot; ++epoch) {
                auto offset = slot_offset +
                              static_cast<size_t>(epoch) * _snapshot_stride;
                auto live_count = 0u;
                for (auto node = 1u; node < _node_count; ++node) {
                    live_count += _host_snapshots[offset + node];
                }
                _last_dispatch_stats.max_live_count = std::max(
                    _last_dispatch_stats.max_live_count, live_count);
                auto generated = _host_snapshots[offset + _node_count];
                observed_sweeps++;
                latest_live_count = live_count;
                latest_generated_count = generated;
                _last_dispatch_stats.generated_count = generated;
                done |= generated == logical_count && live_count == 0u;
                tail_candidate =
                    _config.tail_megakernel_threshold != 0u &&
                    generated == logical_count && live_count != 0u &&
                    live_count <= _config.tail_megakernel_threshold;
            }
            consumed++;
            return true;
        };

        auto submit_batch = [&](uint slot) noexcept {
            for (auto epoch = 0u; epoch < snapshots_per_slot; ++epoch) {
                auto source_bank = static_cast<uint>(
                    _last_dispatch_stats.sweep_count & 1u);
                auto destination_bank = 1u - source_bank;
                stream << _clear_counts_shader(destination_bank)
                              .dispatch(active_node_count)
                       << _prepare_entry_shader(
                              source_bank, logical_count,
                              _active_frame_capacity)
                              .dispatch(1u)
                       << _entry_shader(destination_bank,
                                        _config.thread_count, dispatch_size,
                                        args...)
                              .dispatch(_active_frame_capacity);
                for (auto node = 1u; node < _node_count; ++node) {
                    stream << _continuation_shaders[node](
                                  source_bank, destination_bank,
                                  _config.thread_count, args...)
                                  .dispatch(_active_frame_capacity);
                }
                stream << _prepare_returns_shader(
                              _config.thread_count, _active_frame_capacity)
                              .dispatch(1u)
                       << _copy_returns_shader(_config.thread_count)
                              .dispatch(_active_frame_capacity)
                       << _commit_returns_shader().dispatch(1u);
                auto snapshot_index = slot * snapshots_per_slot + epoch;
                stream << _snapshot_shader(snapshot_index, destination_bank)
                              .dispatch(_snapshot_stride);
                _last_dispatch_stats.sweep_count++;
                _last_dispatch_stats.counter_snapshot_count++;
            }

            auto ready = _counter_snapshot_ready_event.signal();
            auto ready_fence = ready.fence;
            stream << std::move(ready);
            auto done_signal = _counter_readback_done_event.signal();
            auto done_fence = done_signal.fence;
            auto slot_offset = static_cast<size_t>(slot) * elements_per_slot;
            _counter_readback_stream
                << _counter_snapshot_ready_event.wait(ready_fence)
                << _counter_snapshots.view()
                       .subview(slot_offset, elements_per_slot)
                       .copy_to(luisa::span{
                           _host_snapshots.data() + slot_offset,
                           elements_per_slot})
                << std::move(done_signal);
            pending[submitted % pipeline_depth] = PendingReadback{
                .slot = slot,
                .done_fence = done_fence};
            submitted++;
            _last_dispatch_stats.counter_readback_count++;
            _last_dispatch_stats.counter_readback_bytes +=
                elements_per_slot * sizeof(uint);
            _last_dispatch_stats.max_readbacks_in_flight = std::max(
                _last_dispatch_stats.max_readbacks_in_flight,
                static_cast<uint>(submitted - consumed));
        };

        while (!done) {
            tail_candidate = false;
            while (consumed < submitted && consume_oldest(false)) {}
            if (!done && !tail_candidate) {
                if (submitted - consumed == pipeline_depth) {
                    consume_oldest(true);
                    if (!done && !tail_candidate) { continue; }
                } else {
                    submit_batch(static_cast<uint>(submitted % pipeline_depth));
                    continue;
                }
            }

            // A tail decision is valid only for the newest completed sweep.
            // Stop speculation and drain every newer snapshot before taking
            // ownership of its active bank. With one-successor coroutine
            // transitions, the live cardinality cannot increase after all
            // logical instances have been generated.
            while (consumed < submitted) { consume_oldest(true); }
            if (latest_generated_count == logical_count &&
                latest_live_count == 0u) {
                done = true;
                break;
            }
            if (_config.tail_megakernel_threshold != 0u &&
                latest_generated_count == logical_count &&
                latest_live_count <= _config.tail_megakernel_threshold) {
                auto source_bank = static_cast<uint>(observed_sweeps & 1u);
                stream << _prepare_tail_offsets_shader(source_bank)
                              .dispatch(1u)
                       << _tail_shader(source_bank, _config.thread_count,
                                       args...)
                              .dispatch(latest_live_count);
                _last_dispatch_stats.tail_dispatch_count++;
                _last_dispatch_stats.tail_instance_count +=
                    latest_live_count;
                done = true;
            }
        }
        // A delayed observation can report completion while newer batches are
        // already executing. Once all logical invocations have terminated,
        // these sweeps only transfer stable uint frame-slot indices between
        // queue banks; they never access frame storage. Their readback buffers
        // still belong to this scheduler, so drain them before reuse/destruction.
        while (consumed < submitted) { consume_oldest(true); }
        if (report_stats) {
            _last_dispatch_stats.elapsed_ms = clock.toc();
            LUISA_INFO(
                "Graph wavefront stats: sweeps={} snapshots={} "
                "readbacks={} bytes={} waits={} max_in_flight={} generated={} "
                "max_live={} tail_dispatches={} tail_instances={} "
                "elapsed_ms={:.3f}.",
                _last_dispatch_stats.sweep_count,
                _last_dispatch_stats.counter_snapshot_count,
                _last_dispatch_stats.counter_readback_count,
                _last_dispatch_stats.counter_readback_bytes,
                _last_dispatch_stats.host_wait_count,
                _last_dispatch_stats.max_readbacks_in_flight,
                _last_dispatch_stats.generated_count,
                _last_dispatch_stats.max_live_count,
                _last_dispatch_stats.tail_dispatch_count,
                _last_dispatch_stats.tail_instance_count,
                _last_dispatch_stats.elapsed_ms);
        }
    }

public:
    GraphWavefrontCoroScheduler(
        Device &device, const Coro &coro, const Config &config) noexcept
        : _config{config} {
        LUISA_ASSERT(_config.thread_count != 0u,
                     "Graph wavefront frame capacity must be positive.");
        LUISA_ASSERT(
            _config.execution_block_size >= 32u &&
                _config.execution_block_size <= 1024u &&
                _config.execution_block_size % 32u == 0u,
            "Graph wavefront execution block size must be a multiple of 32 "
            "in [32, 1024], but got {}.",
            _config.execution_block_size);
        LUISA_ASSERT(
            _config.counter_readback_batch_size != 0u,
            "Graph wavefront counter readback batch must be positive.");
        LUISA_ASSERT(
            _config.counter_readback_pipeline_depth != 0u,
            "Graph wavefront counter readback pipeline must contain at least "
            "one slot.");
        LUISA_ASSERT(
            coro.subroutine_count() > 1u &&
                coro.subroutine_count() <=
                    std::numeric_limits<uint>::max(),
            "Graph wavefront requires at least one continuation and a uint "
            "node count (got {}).",
            coro.subroutine_count());
        _node_count = static_cast<uint>(coro.subroutine_count());
        _snapshot_stride = _node_count + 1u;
        auto active_node_count = _node_count - 1u;
        auto queue_count = 2ull + 2ull * active_node_count;
        auto queue_element_count =
            queue_count * _config.thread_count;
        LUISA_ASSERT(
            queue_element_count <= std::numeric_limits<uint>::max(),
            "Graph wavefront indirect queue index domain (2 + 2 x {}) x {} "
            "exceeds uint.",
            active_node_count, _config.thread_count);
        _frame_layout = _config.global_memory_soa ?
                            CoroFrameStorageLayout::make_runtime_soa(
                                coro.frame(), _config.thread_count) :
                            CoroFrameStorageLayout::make_aos(
                                coro.frame(), _config.thread_count);
        _frame_buffer = device.create_byte_buffer(_frame_layout.size_bytes);
        _queue_indices = device.create_buffer<uint>(queue_element_count);
        _queue_counts = device.create_buffer<uint>(queue_count);
        _work_state = device.create_buffer<uint>(5u + active_node_count);
        _dispatch_state = device.create_buffer<uint>(1u);
        auto snapshot_count =
            static_cast<uint64_t>(_snapshot_stride) *
            _config.counter_readback_batch_size *
            _config.counter_readback_pipeline_depth;
        LUISA_ASSERT(
            snapshot_count <= std::numeric_limits<uint>::max(),
            "Graph wavefront counter snapshot ring is too large.");
        _counter_snapshots = device.create_buffer<uint>(snapshot_count);
        if (_config.tail_megakernel_threshold != 0u) {
            _tail_offsets = device.create_buffer<uint>(_node_count);
        }
        _host_snapshots.resize(snapshot_count);
        _counter_readback_stream = device.create_stream(StreamTag::COPY);
        _counter_snapshot_ready_event = device.create_event();
        _counter_readback_done_event = device.create_event();
        _create_shaders(device, coro);
    }

    GraphWavefrontCoroScheduler(Device &device, const Coro &coro) noexcept
        : GraphWavefrontCoroScheduler{
              device, coro, GraphWavefrontCoroSchedulerConfig{}} {}

    [[nodiscard]] const Config &config() const noexcept { return _config; }
    [[nodiscard]] uint node_count() const noexcept { return _node_count; }
    [[nodiscard]] uint active_frame_capacity() const noexcept {
        return _active_frame_capacity;
    }
    [[nodiscard]] const GraphWavefrontCoroDispatchStats &
    last_dispatch_stats() const noexcept {
        return _last_dispatch_stats;
    }
};

template<typename... Args>
GraphWavefrontCoroScheduler(
    Device &, const Coroutine<void(Args...)> &,
    const GraphWavefrontCoroSchedulerConfig &)
    -> GraphWavefrontCoroScheduler<Args...>;

template<typename... Args>
GraphWavefrontCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> GraphWavefrontCoroScheduler<Args...>;

}// namespace luisa::compute::coro
