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
#include <luisa/coro/radix_sort.h>
#include <luisa/coro/schedulers/detail/token_index.h>
#include <luisa/coro/schedulers/graph_wavefront_policy.h>
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
    // Maximum logical lanes launched for one queue consumer. A zero value
    // preserves the full-frame-capacity grid. A smaller runtime value uses a
    // grid-stride bijection over the actual device queue count, so it changes
    // scheduling cost but neither shader identity nor coroutine semantics.
    uint worker_count = 0u;
    // Consume only one continuation queue per action, selected from exact
    // host-observed populations. This is a scheduling-order policy; device
    // ownership and backpressure remain exact. Initially restricted to one
    // snapshot per readback so no action is chosen from stale state.
    bool selective_scheduling = false;
    // Entry refill threshold and optional CoroGraph node names. Zero means
    // half the active frame capacity. Empty means every selected node is
    // aligned for refill.
    uint refill_threshold = 0u;
    luisa::vector<luisa::string> refill_continuations;
    // Largest-queue-first can otherwise starve a sparse self-loop behind a
    // permanently populated hot queue. A nonzero limit adds a host-only
    // bounded-service rule. It changes scheduling order, never shader AST or
    // cache identity. Zero preserves unrestricted greedy selection.
    uint64_t max_queue_wait_actions = 32u;
    uint counter_readback_batch_size = 4u;
    // Number of readback slots kept in flight. A value greater than one lets
    // the GPU execute later fixed sweeps while the host waits for an older
    // counter snapshot. This host scheduling choice never enters a shader AST.
    uint counter_readback_pipeline_depth = 2u;
    // Once all logical invocations have been generated, a small residual set
    // is cheaper to finish in one graph-derived state-machine kernel than in
    // one guarded launch per continuation. This is a host policy parameter:
    // its nonzero magnitude does not participate in shader construction or
    // cache identity. Zero disables the hybrid tail drain and avoids compiling
    // the optional state-machine shader altogether.
    uint tail_megakernel_threshold = 4096u;
    bool report_stats = false;
    ShaderOption shader_option{};
    // Optional queue-local coherence key. `hint_fields` names CoroGraph
    // continuations whose frame indices should be sorted by the explicitly
    // exported uint `coro_hint` before consumption. These host/JIT choices do
    // not change coroutine semantics or the consuming continuation shader
    // identity; only the optional sorting shaders depend on the hint range.
    // Exact queue cardinality is required, so hint sorting is currently
    // defined only for non-speculative selective scheduling.
    uint hint_range = 0xffffffffu;
    luisa::vector<luisa::string> hint_fields;
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
    uint worker_count{0u};
    uint64_t entry_dispatch_count{0u};
    uint64_t fairness_dispatch_count{0u};
    // Host-observed destination populations for each CoroGraph node. Node
    // zero is always zero here because free frames are reported separately.
    // In a non-tail dispatch, summing q[t + 1] over all snapshots is exactly
    // the number of later consumer executions: every queued frame is consumed
    // once in the following sweep.
    luisa::vector<uint64_t> queued_count_sum;
    luisa::vector<uint64_t> nonempty_snapshot_count;
    luisa::vector<uint> peak_queued_count;
    luisa::vector<uint> input_field_count;
    luisa::vector<uint> max_transition_output_field_count;
    luisa::vector<uint64_t> continuation_dispatch_count;
    luisa::vector<uint64_t> continuation_executed_count;
    luisa::vector<uint64_t> continuation_hint_sort_count;
    luisa::vector<uint64_t> continuation_max_wait_actions;
    luisa::vector<luisa::string> continuation_names;
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
    // Device work-state ABI. Keep scheduler scalars named: these indices are
    // shared by several independently compiled kernels and therefore form a
    // real ABI rather than disposable implementation magic numbers.
    static constexpr uint _work_logical_first = 0u;
    static constexpr uint _work_admit_count = 1u;
    static constexpr uint _work_free_first = 2u;
    static constexpr uint _work_return_free_first = 3u;
    static constexpr uint _work_return_count = 4u;
    static constexpr uint _work_carry_count = 5u;
    static constexpr uint _work_node_count_base = 6u;

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
    Shader1D<uint, uint, uint, uint, uint, uint> _prepare_action_shader;
    Shader1D<uint, uint, uint, uint> _carry_queues_shader;
    Shader1D<uint, uint, uint3, Args...> _entry_shader;
    luisa::vector<Shader1D<uint, Buffer<uint>, uint, uint, Args...>>
        _continuation_shaders;
    Shader1D<uint, uint> _prepare_returns_shader;
    Shader1D<uint> _copy_returns_shader;
    Shader1D<> _commit_returns_shader;
    Shader1D<uint, uint> _snapshot_shader;
    Shader1D<uint> _prepare_tail_offsets_shader;
    Shader1D<uint, uint, Args...> _tail_shader;
    Buffer<uint> _sort_key[2];
    Buffer<uint> _sort_index;
    radix_sort::temp_storage _sort_temp_storage;
    radix_sort::instance<Buffer<uint>, ByteBuffer, uint> _sort_hint;
    luisa::vector<uint> _host_snapshots;
    luisa::vector<uint64_t> _shader_structure_hashes;
    GraphWavefrontCoroDispatchStats _last_dispatch_stats;
    luisa::vector<uint> _input_field_count;
    luisa::vector<uint> _max_transition_output_field_count;
    luisa::vector<luisa::string> _continuation_names;
    luisa::vector<uint> _refill_nodes;
    luisa::vector<bool> _have_hint;
    size_t _hint_field_index{static_cast<size_t>(-1)};
    uint _node_count{};
    uint _snapshot_stride{};
    uint _active_frame_capacity{};
    bool _has_hint_sort{false};

private:
    [[nodiscard]] static auto _find_frame_field_index(
        const CoroFrameDesc &desc, luisa::string_view name) noexcept {
        auto index = desc.field_index(name);
        return index == static_cast<size_t>(-1) ?
                   index :
                   CoroFrameDesc::reserved_field_count + index;
    }

    [[nodiscard]] auto _valid_hint_field_count() const noexcept {
        auto count = 0u;
        for (auto hint : _have_hint) {
            if (hint) { count++; }
        }
        return count;
    }

    [[nodiscard]] auto _sort_hint_range(
        Stream &stream, BufferView<uint> source_indices,
        uint count) noexcept {
        LUISA_ASSERT(
            _has_hint_sort && count != 0u &&
                count <= _config.thread_count &&
                source_indices.size() >= count,
            "Invalid graph-wavefront hint-sort range: enabled={} count={} "
            "capacity={} source_size={}.",
            _has_hint_sort, count, _config.thread_count,
            source_indices.size());
        BufferView<uint> indices[2] = {
            source_indices.subview(0u, count),
            _sort_index.view().subview(0u, count)};
        BufferView<uint> keys[2] = {
            _sort_key[0].view().subview(0u, count),
            _sort_key[1].view().subview(0u, count)};
        auto output = _sort_hint.sort_switch(
            stream, keys, indices, count,
            source_indices.subview(0u, count),
            _frame_buffer, _config.thread_count);
        return indices[output];
    }

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
        luisa::string_view stage) noexcept {
        _shader_structure_hashes.emplace_back(
            kernel.function()->function().hash());
        return device.compile(
            kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, stage));
    }

    void _create_shaders(Device &device, const Coro &coro) {
        _shader_structure_hashes.clear();
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
                    work.write(_work_node_count_base + node - 1u, 0u);
                }
                ownership_valid &= free_count == remaining_capacity;
                auto logical_first = state.read(0u);
                auto remaining = ite(
                    logical_first < logical_count,
                    logical_count - logical_first, 0u);
                work.write(_work_logical_first, logical_first);
                work.write(_work_admit_count, 0u);
                work.write(_work_free_first, free_count);
                work.write(_work_carry_count, 0u);
                $if (ownership_valid) {
                    auto admit_count = min(free_count, remaining);
                    auto free_first = free_count - admit_count;
                    counts.write(0u, free_first);
                    state.write(0u, logical_first + admit_count);
                    work.write(_work_admit_count, admit_count);
                    work.write(_work_free_first, free_first);
                    for (auto node = 1u; node <= active_node_count; ++node) {
                        auto source_queue =
                            1u + source_bank * active_node_count + node - 1u;
                        work.write(_work_node_count_base + node - 1u,
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

        // A selective action consumes at most one continuation. Every other
        // queue is carried verbatim into the destination bank. Its copied
        // prefix is published before producers append, so transitions into an
        // unselected queue append after that prefix without overlap.
        Kernel1D prepare_action =
            [queue_counts, work_state, dispatch_state,
             active_node_count](UInt source_bank, UInt destination_bank,
                                UInt selected_node, UInt admit_entry,
                                UInt logical_count,
                                UInt active_capacity) noexcept {
                auto counts = Expr<Buffer<uint>>{*queue_counts};
                auto work = Expr<Buffer<uint>>{*work_state};
                auto state = Expr<Buffer<uint>>{*dispatch_state};
                auto free_count = counts.read(0u);
                auto remaining_capacity = def(active_capacity);
                auto ownership_valid = def(free_count <= active_capacity);
                auto carry_count = def(0u);
                remaining_capacity -= min(free_count, remaining_capacity);
                for (auto node = 1u; node <= active_node_count; ++node) {
                    auto source_queue = 1u + source_bank * active_node_count +
                                        node - 1u;
                    auto source_count = counts.read(source_queue);
                    ownership_valid &= source_count <= active_capacity;
                    ownership_valid &= source_count <= remaining_capacity;
                    remaining_capacity -= min(
                        source_count, remaining_capacity);
                    auto destination_queue =
                        1u + destination_bank * active_node_count + node - 1u;
                    auto carried_count = ite(
                        selected_node == node, 0u, source_count);
                    counts.write(destination_queue, carried_count);
                    work.write(_work_node_count_base + node - 1u,
                               ite(selected_node == node,
                                   source_count, 0u));
                    carry_count = max(carry_count, carried_count);
                }
                ownership_valid &= remaining_capacity == 0u;
                auto logical_first = state.read(0u);
                auto remaining = ite(
                    logical_first < logical_count,
                    logical_count - logical_first, 0u);
                auto admit_count = min(
                    free_count, remaining) * min(admit_entry, 1u);
                work.write(_work_logical_first, logical_first);
                work.write(_work_admit_count, 0u);
                work.write(_work_free_first, free_count);
                work.write(_work_carry_count, carry_count);
                $if (ownership_valid) {
                    auto free_first = free_count - admit_count;
                    counts.write(0u, free_first);
                    state.write(0u, logical_first + admit_count);
                    work.write(_work_admit_count, admit_count);
                    work.write(_work_free_first, free_first);
                }
                $else {
                    unreachable(
                        "Graph wavefront selective-action ownership "
                        "certificate failed before execution.");
                };
            };
        _prepare_action_shader = _compile(
            device, prepare_action, "graph_wavefront_prepare_action");

        Kernel1D carry_queues =
            [queue_indices, queue_counts, work_state, active_node_count](
                UInt source_bank, UInt destination_bank,
                UInt selected_node, UInt storage_capacity) noexcept {
                auto indices = Expr<Buffer<uint>>{*queue_indices};
                auto counts = Expr<Buffer<uint>>{*queue_counts};
                auto work = Expr<Buffer<uint>>{*work_state};
                auto x = def(dispatch_x());
                // Let C_i be the source count of unselected queue i and
                // M=max_i C_i. Copying queue i visits exactly [0,C_i), which
                // is a subset of [0,M). The selected queue has C_i=0 for this
                // transfer. Therefore [0,M) is both a sufficient and the
                // smallest common rectangular launch domain; scanning the
                // complete frame capacity adds no observable work.
                auto carry_count = work.read(_work_carry_count);
                $while (x < carry_count) {
                    for (auto node = 1u; node <= active_node_count; ++node) {
                        auto source_queue =
                            1u + source_bank * active_node_count + node - 1u;
                        auto destination_queue =
                            1u + destination_bank * active_node_count + node - 1u;
                        auto count = counts.read(source_queue);
                        $if ((selected_node != node) & (x < count)) {
                            indices.write(
                                destination_queue * storage_capacity + x,
                                indices.read(
                                    source_queue * storage_capacity + x));
                        };
                    }
                    x += dispatch_size_x();
                };
            };
        _carry_queues_shader = _compile(
            device, carry_queues, "graph_wavefront_carry_queues");

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
            auto admit_count =
                Expr<Buffer<uint>>{*work_state}.read(_work_admit_count);
            auto x = def(dispatch_x());
            $while (x < admit_count) {
                auto free_first =
                    Expr<Buffer<uint>>{*work_state}.read(_work_free_first);
                auto frame_index =
                    Expr<Buffer<uint>>{*queue_indices}.read(
                        free_first + x);
                auto logical_first =
                    Expr<Buffer<uint>>{*work_state}.read(_work_logical_first);
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
                x += dispatch_size_x();
            };
        };
        _entry_shader = _compile(
            device, entry, "graph_wavefront_entry");

        _continuation_shaders.resize(_node_count);
        _input_field_count.resize(_node_count, 0u);
        _max_transition_output_field_count.resize(_node_count, 0u);
        _continuation_names.resize(_node_count);
        _continuation_names[0u] = "<entry>";
        for (auto node_index = 1u; node_index < _node_count; ++node_index) {
            _continuation_names[node_index] =
                coro.graph().node(node_index).name;
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
                _max_transition_output_field_count[node_index] = std::max(
                    _max_transition_output_field_count[node_index],
                    static_cast<uint>(output_fields[target].size()));
            }
            _input_field_count[node_index] =
                static_cast<uint>(input_fields.size());
            Kernel1D continuation =
                [&coro, frame_buffer, work_state, block_size,
                 active_node_count, layout, soa, node_index, subroutine,
                 input_fields = std::move(input_fields),
                 output_fields = std::move(output_fields), token_to_index,
                 push_frame](UInt destination_bank,
                             BufferUInt resume_indices,
                             UInt resume_offset, UInt storage_capacity,
                             Var<Args>... args) noexcept {
                    set_block_size(block_size);
                    auto count = Expr<Buffer<uint>>{*work_state}.read(
                        _work_node_count_base + node_index - 1u);
                    auto x = def(dispatch_x());
                    $while (x < count) {
                        auto frame_index = resume_indices.read(
                            resume_offset + x);
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
                        x += dispatch_size_x();
                    };
                };
            _continuation_shaders[node_index] = _compile(
                device, continuation,
                luisa::format("graph_wavefront_node_{}", node_index));
        }

        // Returning frame indices to the free stack is a publication
        // transaction. Entry lanes may still be reading the popped stack
        // suffix while other entry lanes terminate, so returns must remain in
        // a disjoint queue until every producer kernel has completed.
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
                        work.write(_work_return_free_first, free_first);
                        work.write(_work_return_count, returned_count);
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
            auto indices = Expr<Buffer<uint>>{*queue_indices};
            auto work = Expr<Buffer<uint>>{*work_state};
            auto free_first = work.read(_work_return_free_first);
            auto returned_count = work.read(_work_return_count);
            auto x = def(dispatch_x());
            $while (x < returned_count) {
                auto frame_index = indices.read(
                    return_queue * storage_capacity + x);
                indices.write(free_first + x, frame_index);
                x += dispatch_size_x();
            };
        };
        _copy_returns_shader = _compile(
            device, copy_returns, "graph_wavefront_copy_returns");

        Kernel1D commit_returns = [queue_counts, work_state]() noexcept {
            auto work = Expr<Buffer<uint>>{*work_state};
            Expr<Buffer<uint>>{*queue_counts}.write(
                0u, work.read(_work_return_free_first) +
                        work.read(_work_return_count));
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
        auto worker_count = _config.worker_count == 0u ?
                                _active_frame_capacity :
                                std::min(_config.worker_count,
                                         _active_frame_capacity);
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
        _last_dispatch_stats.worker_count = worker_count;
        _last_dispatch_stats.queued_count_sum.resize(_node_count, 0u);
        _last_dispatch_stats.nonempty_snapshot_count.resize(_node_count, 0u);
        _last_dispatch_stats.peak_queued_count.resize(_node_count, 0u);
        _last_dispatch_stats.continuation_dispatch_count.resize(
            _node_count, 0u);
        _last_dispatch_stats.continuation_executed_count.resize(
            _node_count, 0u);
        _last_dispatch_stats.continuation_hint_sort_count.resize(
            _node_count, 0u);
        _last_dispatch_stats.continuation_max_wait_actions.resize(
            _node_count, 0u);
        _last_dispatch_stats.continuation_names = _continuation_names;
        _last_dispatch_stats.input_field_count = _input_field_count;
        _last_dispatch_stats.max_transition_output_field_count =
            _max_transition_output_field_count;
        Clock clock;
        struct PendingReadback {
            uint slot{};
            uint64_t done_fence{};
        };
        auto pipeline_depth =
            _config.counter_readback_pipeline_depth;
        auto snapshots_per_slot =
            _config.counter_readback_batch_size;
        LUISA_ASSERT(
            !_config.selective_scheduling ||
                (pipeline_depth == 1u && snapshots_per_slot == 1u),
            "Exact selective graph-wavefront scheduling currently requires "
            "one snapshot and one readback slot; delayed actions require the "
            "Markov predictor.");
        auto elements_per_slot =
            static_cast<size_t>(_snapshot_stride) * snapshots_per_slot;
        auto pending = luisa::vector<PendingReadback>(pipeline_depth);
        auto submitted = uint64_t{0u};
        auto consumed = uint64_t{0u};
        auto observed_sweeps = uint64_t{0u};
        auto latest_live_count = 0u;
        auto latest_generated_count = 0u;
        auto latest_population = GraphWavefrontPopulation{
            .queues = luisa::vector<double>(_node_count, 0.0),
            .generated_count = 0.0};
        latest_population.queues[0u] = _active_frame_capacity;
        auto queue_wait_actions = luisa::vector<uint64_t>(_node_count, 0u);
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
                    auto queued = _host_snapshots[offset + node];
                    latest_population.queues[node] = queued;
                    live_count += queued;
                    _last_dispatch_stats.queued_count_sum[node] += queued;
                    _last_dispatch_stats.nonempty_snapshot_count[node] +=
                        queued != 0u;
                    _last_dispatch_stats.peak_queued_count[node] = std::max(
                        _last_dispatch_stats.peak_queued_count[node], queued);
                }
                _last_dispatch_stats.max_live_count = std::max(
                    _last_dispatch_stats.max_live_count, live_count);
                auto generated = _host_snapshots[offset + _node_count];
                latest_population.queues[0u] =
                    _active_frame_capacity - live_count;
                latest_population.generated_count = generated;
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
                if (_config.selective_scheduling) {
                    for (auto node = 1u; node < _node_count; ++node) {
                        _last_dispatch_stats
                            .continuation_max_wait_actions[node] = std::max(
                                _last_dispatch_stats
                                    .continuation_max_wait_actions[node],
                                queue_wait_actions[node]);
                    }
                    auto action = graph_wavefront_select_action(
                        latest_population, logical_count,
                        _active_frame_capacity, _config.refill_threshold,
                        luisa::span{_refill_nodes},
                        luisa::span{queue_wait_actions},
                        _config.max_queue_wait_actions);
                    if (action.forced_by_fairness) {
                        _last_dispatch_stats.fairness_dispatch_count++;
                    }
                    graph_wavefront_advance_wait_actions(
                        latest_population, action.selected_node,
                        luisa::span{queue_wait_actions});
                    stream << _clear_counts_shader(destination_bank)
                                  .dispatch(active_node_count)
                           << _prepare_action_shader(
                                  source_bank, destination_bank,
                                  action.selected_node,
                                  static_cast<uint>(action.admit_entry),
                                  logical_count, _active_frame_capacity)
                                  .dispatch(1u)
                           << _carry_queues_shader(
                                  source_bank, destination_bank,
                                  action.selected_node,
                                  _config.thread_count)
                                  .dispatch(worker_count);
                    if (action.admit_entry) {
                        stream << _entry_shader(
                                      destination_bank,
                                      _config.thread_count, dispatch_size,
                                      args...)
                                      .dispatch(worker_count);
                        _last_dispatch_stats.entry_dispatch_count++;
                    }
                    if (action.selected_node != 0u) {
                        auto source_queue =
                            1u + source_bank * active_node_count +
                            action.selected_node - 1u;
                        auto source_indices = _queue_indices.view().subview(
                            source_queue * _config.thread_count,
                            _config.thread_count);
                        auto resume_indices = source_indices;
                        auto selected_count = static_cast<uint>(
                            latest_population.queues[action.selected_node]);
                        if (_has_hint_sort &&
                            _have_hint[action.selected_node]) {
                            LUISA_ASSERT(
                                selected_count != 0u &&
                                    selected_count <= _active_frame_capacity,
                                "Graph-wavefront selected queue {} has invalid "
                                "exact population {} for active capacity {}.",
                                action.selected_node, selected_count,
                                _active_frame_capacity);
                            resume_indices = _sort_hint_range(
                                stream, source_indices, selected_count);
                            _last_dispatch_stats
                                .continuation_hint_sort_count[
                                    action.selected_node]++;
                        }
                        stream << _continuation_shaders[action.selected_node](
                                      destination_bank, resume_indices,
                                      0u, _config.thread_count, args...)
                                      .dispatch(worker_count);
                        _last_dispatch_stats
                            .continuation_dispatch_count[action.selected_node]++;
                        _last_dispatch_stats
                            .continuation_executed_count[action.selected_node] +=
                            static_cast<uint64_t>(latest_population.queues[
                                action.selected_node]);
                    }
                } else {
                    stream << _clear_counts_shader(destination_bank)
                                  .dispatch(active_node_count)
                           << _prepare_entry_shader(
                                  source_bank, logical_count,
                                  _active_frame_capacity)
                                  .dispatch(1u)
                           << _entry_shader(
                                  destination_bank, _config.thread_count,
                                  dispatch_size, args...)
                                  .dispatch(worker_count);
                    _last_dispatch_stats.entry_dispatch_count++;
                    for (auto node = 1u; node < _node_count; ++node) {
                        auto source_queue =
                            1u + source_bank * active_node_count + node - 1u;
                        auto source_indices = _queue_indices.view().subview(
                            source_queue * _config.thread_count,
                            _config.thread_count);
                        stream << _continuation_shaders[node](
                                      destination_bank, source_indices,
                                      0u, _config.thread_count, args...)
                                      .dispatch(worker_count);
                        _last_dispatch_stats
                            .continuation_dispatch_count[node]++;
                        _last_dispatch_stats
                            .continuation_executed_count[node] +=
                            static_cast<uint64_t>(
                                latest_population.queues[node]);
                    }
                }
                stream << _prepare_returns_shader(
                              _config.thread_count, _active_frame_capacity)
                              .dispatch(1u)
                       << _copy_returns_shader(_config.thread_count)
                              .dispatch(worker_count)
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
        if (!_config.selective_scheduling) {
            for (auto node = 1u; node < _node_count; ++node) {
                _last_dispatch_stats.continuation_dispatch_count[node] =
                    _last_dispatch_stats.sweep_count;
                _last_dispatch_stats.continuation_executed_count[node] =
                    _last_dispatch_stats.queued_count_sum[node];
            }
        }
        if (report_stats) {
            _last_dispatch_stats.elapsed_ms = clock.toc();
            LUISA_INFO(
                "Graph wavefront stats: sweeps={} snapshots={} "
                "readbacks={} bytes={} waits={} max_in_flight={} generated={} "
                "max_live={} workers={} tail_dispatches={} tail_instances={} "
                "fairness_dispatches={} elapsed_ms={:.3f}.",
                _last_dispatch_stats.sweep_count,
                _last_dispatch_stats.counter_snapshot_count,
                _last_dispatch_stats.counter_readback_count,
                _last_dispatch_stats.counter_readback_bytes,
                _last_dispatch_stats.host_wait_count,
                _last_dispatch_stats.max_readbacks_in_flight,
                _last_dispatch_stats.generated_count,
                _last_dispatch_stats.max_live_count,
                _last_dispatch_stats.worker_count,
                _last_dispatch_stats.tail_dispatch_count,
                _last_dispatch_stats.tail_instance_count,
                _last_dispatch_stats.fairness_dispatch_count,
                _last_dispatch_stats.elapsed_ms);
            for (auto node = 1u; node < _node_count; ++node) {
                LUISA_INFO(
                    "Graph wavefront queue: index={} name='{}' queued_sum={} "
                    "nonempty_snapshots={} peak_queued={} input_fields={} "
                    "max_transition_output_fields={} dispatches={} "
                    "executed={} hint_sorts={} max_wait_actions={}.",
                    node, _last_dispatch_stats.continuation_names[node],
                    _last_dispatch_stats.queued_count_sum[node],
                    _last_dispatch_stats.nonempty_snapshot_count[node],
                    _last_dispatch_stats.peak_queued_count[node],
                    _last_dispatch_stats.input_field_count[node],
                    _last_dispatch_stats
                        .max_transition_output_field_count[node],
                    _last_dispatch_stats
                        .continuation_dispatch_count[node],
                    _last_dispatch_stats
                        .continuation_executed_count[node],
                    _last_dispatch_stats
                        .continuation_hint_sort_count[node],
                    _last_dispatch_stats
                        .continuation_max_wait_actions[node]);
            }
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
            _config.max_queue_wait_actions == 0u ||
                _config.max_queue_wait_actions <=
                    std::numeric_limits<uint64_t>::max() -
                        coro.subroutine_count(),
            "Graph-wavefront fairness horizon plus node count overflows "
            "uint64.");
        _refill_nodes.reserve(_config.refill_continuations.size());
        for (auto &&name : _config.refill_continuations) {
            auto *node = coro.graph().node_by_name(name);
            LUISA_ASSERT(node != nullptr && node->index != 0u,
                         "Graph-wavefront refill continuation '{}' does not "
                         "name a materialized non-entry CoroGraph node.",
                         name);
            _refill_nodes.emplace_back(static_cast<uint>(node->index));
        }
        std::sort(_refill_nodes.begin(), _refill_nodes.end());
        _refill_nodes.erase(
            std::unique(_refill_nodes.begin(), _refill_nodes.end()),
            _refill_nodes.end());
        LUISA_ASSERT(
            coro.subroutine_count() > 1u &&
                coro.subroutine_count() <=
                    std::numeric_limits<uint>::max(),
            "Graph wavefront requires at least one continuation and a uint "
            "node count (got {}).",
            coro.subroutine_count());
        _node_count = static_cast<uint>(coro.subroutine_count());
        _have_hint.resize(_node_count, false);
        luisa::vector<luisa::string> valid_hint_fields;
        valid_hint_fields.reserve(_config.hint_fields.size());
        for (auto &&name : _config.hint_fields) {
            if (auto *node = coro.graph().node_by_name(name)) {
                if (node->index == 0u || node->index >= _node_count) {
                    LUISA_WARNING(
                        "Graph-wavefront hint continuation '{}' resolves to "
                        "invalid node {}; hint disabled.",
                        name, node->index);
                } else {
                    _have_hint[node->index] = true;
                    valid_hint_fields.emplace_back(name);
                }
            } else {
                LUISA_WARNING(
                    "Graph-wavefront hint continuation '{}' does not match a "
                    "suspend name; hint disabled.",
                    name);
            }
        }
        _config.hint_fields = std::move(valid_hint_fields);
        if (_valid_hint_field_count() != 0u) {
            _hint_field_index = _find_frame_field_index(
                coro.frame(), "coro_hint");
            if (_hint_field_index == static_cast<size_t>(-1)) {
                LUISA_WARNING(
                    "GraphWavefrontCoroSchedulerConfig::hint_fields requires "
                    "a uint frame value explicitly exported as 'coro_hint'; "
                    "hint sorting is disabled.");
                std::fill(_have_hint.begin(), _have_hint.end(), false);
                _config.hint_fields.clear();
            } else if (coro.frame().frame_field_type(_hint_field_index) !=
                       Type::of<uint>()) {
                LUISA_WARNING(
                    "Graph-wavefront coroutine frame export 'coro_hint' must "
                    "be uint; hint sorting is disabled.");
                std::fill(_have_hint.begin(), _have_hint.end(), false);
                _config.hint_fields.clear();
            } else {
                // A scheduler observes a queued frame by target token. The
                // exported field is valid for that token iff CoroGraph's
                // relocation certificate contains it; cfg distillation has
                // already proved this as a must property over every incoming
                // edge. Merely finding the same physical field globally would
                // permit a misconfigured continuation to read a stale slot.
                luisa::vector<luisa::string> frame_valid_hint_fields;
                frame_valid_hint_fields.reserve(_config.hint_fields.size());
                for (auto &&name : _config.hint_fields) {
                    auto *node = coro.graph().node_by_name(name);
                    LUISA_ASSERT(node != nullptr && node->index < _node_count,
                                 "Validated graph-wavefront hint node '{}' "
                                 "disappeared from CoroGraph.",
                                 name);
                    auto &&fields = node->relocation_fields;
                    if (std::find(fields.begin(), fields.end(),
                                  _hint_field_index) != fields.end()) {
                        frame_valid_hint_fields.emplace_back(name);
                    } else {
                        _have_hint[node->index] = false;
                        LUISA_WARNING(
                            "Graph-wavefront hint continuation '{}' does not "
                            "carry the exported 'coro_hint' field on every "
                            "incoming edge; hint disabled for this node.",
                            name);
                    }
                }
                _config.hint_fields =
                    std::move(frame_valid_hint_fields);
            }
        }
        // Queue-local sorting is semantics preserving, but encoding a sort
        // requires the exact source cardinality on the host. The selective
        // policy's one-snapshot/one-slot contract supplies that cardinality;
        // speculative or all-node sweeps deliberately do not synchronize for
        // it. Disable only the optimization when this proof precondition is
        // absent.
        if (_valid_hint_field_count() != 0u &&
            (!_config.selective_scheduling ||
             _config.counter_readback_batch_size != 1u ||
             _config.counter_readback_pipeline_depth != 1u)) {
            LUISA_WARNING(
                "Graph-wavefront hint sorting requires exact selective "
                "scheduling with one snapshot and one readback slot; hint "
                "sorting is disabled.");
            std::fill(_have_hint.begin(), _have_hint.end(), false);
            _config.hint_fields.clear();
        }
        // Small ranges use subgroup-independent bucket sorting. Larger ranges
        // use one-sweep radix ranking, whose contract requires 32-lane waves.
        if (_valid_hint_field_count() != 0u &&
            _config.hint_range > radix_sort::hist_block_size &&
            device.compute_warp_size() != radix_sort::warp_size) {
            LUISA_WARNING(
                "Graph-wavefront hint sorting over range {} requires "
                "{}-lane subgroups, but the device reports {}; hint sorting "
                "is disabled.",
                _config.hint_range, radix_sort::warp_size,
                device.compute_warp_size());
            std::fill(_have_hint.begin(), _have_hint.end(), false);
            _config.hint_fields.clear();
        }
        _has_hint_sort = _valid_hint_field_count() != 0u;
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
        if (_has_hint_sort) {
            _sort_index = device.create_buffer<uint>(_config.thread_count);
            _sort_key[0] = device.create_buffer<uint>(_config.thread_count);
            _sort_key[1] = device.create_buffer<uint>(_config.thread_count);
            auto max_digit = std::max(
                1u, std::min(_config.hint_range,
                             radix_sort::hist_block_size));
            _sort_temp_storage = radix_sort::temp_storage{
                device, _config.thread_count, max_digit};
            Callable<uint(uint, Buffer<uint>, ByteBuffer, uint)> keep_index =
                [](UInt index, BufferUInt values, ByteBufferVar,
                   UInt) noexcept { return values.read(index); };
            Callable<uint(uint, Buffer<uint>, ByteBuffer, uint)> get_hint =
                [layout = _frame_layout,
                 soa = _config.global_memory_soa,
                 hint_field_index = static_cast<uint>(_hint_field_index)](
                    UInt index, BufferUInt values, ByteBufferVar frame_buffer,
                    UInt frame_capacity) noexcept {
                    auto frame_index = values.read(index);
                    return coro_frame_read_field<uint>(
                        frame_buffer, frame_index, frame_capacity,
                        layout, soa, hint_field_index);
                };
            if (_config.hint_range <= radix_sort::hist_block_size) {
                auto hint_digit = std::max(_config.hint_range, 1u);
                _sort_hint = radix_sort::instance<
                    Buffer<uint>, ByteBuffer, uint>{
                    device, _config.thread_count, _sort_temp_storage,
                    &get_hint, &keep_index, &get_hint, 1u, hint_digit};
            } else {
                auto high_bit = 0u;
                while ((_config.hint_range >> high_bit) != 1u) {
                    high_bit++;
                }
                _sort_hint = radix_sort::instance<
                    Buffer<uint>, ByteBuffer, uint>{
                    device, _config.thread_count, _sort_temp_storage,
                    &get_hint, &keep_index, &get_hint, 0u,
                    radix_sort::hist_block_size, 0u, high_bit};
            }
        }
        _queue_indices = device.create_buffer<uint>(queue_element_count);
        _queue_counts = device.create_buffer<uint>(queue_count);
        _work_state = device.create_buffer<uint>(
            _work_node_count_base + active_node_count);
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
    /// Structural hashes of all scheduler-owned kernels. Runtime launch
    /// policy (frame capacity, worker count, readback batching, and pipeline
    /// depth) is deliberately absent from these hashes.
    [[nodiscard]] luisa::span<const uint64_t>
    shader_structure_hashes() const noexcept {
        return {_shader_structure_hashes.data(),
                _shader_structure_hashes.size()};
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
