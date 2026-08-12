//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <cstdlib>
#include <utility>

#include <luisa/coro/coro_frame_storage.h>
#include <luisa/coro/schedulers/detail/token_index.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/coro/radix_sort.h>
#include <luisa/core/clock.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute::coro {

struct WavefrontCoroSchedulerConfig {
    uint thread_count = static_cast<uint>(2_M);
    bool global_memory_soa = true;
    bool gather_by_sorting = true;
    bool frame_buffer_compaction = true;
    uint hint_range = 0xffffffffu;
    luisa::vector<luisa::string> hint_fields;
    bool report_stats = false;
    ShaderOption shader_option{};
    // Workgroup size for the thread-local generate/resume kernels. This is a
    // shader-structure choice and intentionally independent of the runtime
    // frame-pool capacity. Queue-management kernels retain their own block
    // sizes because some of them use block-local collectives. Kept last to
    // preserve the meaning of existing positional aggregate initializers.
    uint execution_block_size = 256u;
    // Cycles-style greedy scheduling executes only the largest non-empty
    // continuation queue in each host iteration. The default drains every
    // queue, preserving the original scheduler policy. This option changes
    // scheduling order only: coroutine transition semantics and frame layout
    // are identical. Kept after the legacy fields so positional aggregate
    // initializers retain their meaning.
    bool largest_continuation_first = false;
    // If non-empty, new entry work may be admitted alongside live frames only
    // when the largest continuation queue has one of these suspend names.
    // This models schedulers such as Cycles, which align old and new paths at
    // INTERSECT_CLOSEST before filling idle state slots. An empty list retains
    // the legacy unrestricted refill policy.
    luisa::vector<luisa::string> refill_continuations;
    // Minimum live-frame population below which refill is considered. Zero
    // selects half the active frame capacity, matching the legacy scheduler
    // and Cycles' upper bound for its device-derived busy-state threshold.
    uint refill_threshold = 0u;
    // Maintain the number of frames in each nonterminal continuation
    // incrementally at coroutine transitions, then scan only for the
    // continuation selected by the host. This replaces a complete token
    // multi-split per scheduling iteration with one selected-token gather.
    // LUISA_CORO_WAVEFRONT_VERIFY_QUEUES=1 independently materializes and
    // checks the invariant at scheduler boundaries. This strategy requires
    // greedy one-continuation-at-a-time scheduling. Kept last for positional
    // source compatibility.
    bool incremental_continuation_counts = false;
};

/// Host-observed work executed by one coroutine graph node during the most
/// recent scheduler dispatch. Node zero is the entry generator; for every
/// other node, `executed_count` is exactly the number of frames submitted to
/// that continuation and `peak_queued_count` is the maximum materialized
/// queue cardinality observed at a scheduler boundary.
struct WavefrontCoroContinuationStats {
    size_t index{0u};
    size_t token{0u};
    luisa::string name;
    uint64_t dispatch_count{0u};
    uint64_t executed_count{0u};
    uint peak_queued_count{0u};
};

/// Diagnostics for the most recent dispatch. Collection is enabled by
/// WavefrontCoroSchedulerConfig::report_stats or
/// LUISA_CORO_WAVEFRONT_STATS. It is a host-only observation: enabling it
/// does not change generated shaders, coroutine frames, or queue semantics.
struct WavefrontCoroDispatchStats {
    bool collected{false};
    uint64_t iteration_count{0u};
    uint64_t generated_count{0u};
    uint64_t resumed_count{0u};
    uint64_t gather_scan_count{0u};
    uint64_t compact_scan_count{0u};
    uint max_scan_count{0u};
    uint max_active_count{0u};
    double elapsed_ms{0.0};
    luisa::vector<WavefrontCoroContinuationStats> continuations;
};

template<typename... Args>
class WavefrontCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = WavefrontCoroSchedulerConfig;

private:
    Config _config;
    ByteBuffer _frame_buffer;
    Shader1D<uint, Buffer<uint>, uint, uint, uint, uint, uint3, Args...>
        _gen_kernel;
    luisa::vector<Shader1D<uint, Buffer<uint>, uint, uint, Args...>>
        _resume_kernels;
    Shader1D<uint, uint> _initialize_shader;
    Shader1D<Buffer<uint>, uint> _clear_count_shader;
    Shader1D<uint, Buffer<uint>, uint> _count_shader;
    Shader1D<uint, Buffer<uint>, Buffer<uint>, uint> _gather_shader;
    Shader1D<uint, Buffer<uint>, Buffer<uint>, uint, uint>
        _gather_selected_shader;
    Shader1D<uint, Buffer<uint>, Buffer<uint>, uint, uint, uint> _compact_shader;
    Buffer<uint> _resume_index;
    Buffer<uint> _resume_count;
    Buffer<uint> _resume_offset;
    Buffer<uint> _global_buffer;
    Buffer<uint> _sort_key[2];
    Buffer<uint> _sort_index;
    radix_sort::temp_storage _sort_temp_storage;
    radix_sort::instance<ByteBuffer, uint> _sort_token;
    radix_sort::instance<Buffer<uint>, ByteBuffer, uint> _sort_hint;
    luisa::vector<uint> _host_count;
    luisa::vector<uint> _host_offset;
    luisa::vector<bool> _have_hint;
    luisa::vector<bool> _refill_at;
    CoroFrameStorageLayout _frame_layout;
    luisa::vector<luisa::vector<size_t>> _input_fields;
    luisa::vector<luisa::vector<size_t>> _output_fields;
    luisa::vector<luisa::vector<luisa::vector<size_t>>> _transition_output_fields;
    luisa::vector<uint64_t> _shader_structure_hashes;
    WavefrontCoroDispatchStats _last_dispatch_stats;
    size_t _hint_field_index{static_cast<size_t>(-1)};
    uint _used_frame_count{0u};
    uint _active_frame_capacity{0u};
    bool _has_hint_sort{false};

private:
    [[nodiscard]] static auto _linear_dispatch_index() noexcept {
        auto id = dispatch_id();
        auto size = dispatch_size();
        return id.x + id.y * size.x + id.z * size.x * size.y;
    }

    [[nodiscard]] static auto _dispatch_id_from_linear_index(UInt global_index, UInt3 dispatch_shape) noexcept {
        auto index_z = global_index / (dispatch_shape.x * dispatch_shape.y);
        auto index_xy = global_index - index_z * dispatch_shape.x * dispatch_shape.y;
        auto index_y = index_xy / dispatch_shape.x;
        auto index_x = index_xy - index_y * dispatch_shape.x;
        return make_uint3(index_x, index_y, index_z);
    }

    [[nodiscard]] static auto _next_power_of_two(uint value) noexcept {
        if (value <= 1u) { return 1u; }
        value--;
        value |= value >> 1u;
        value |= value >> 2u;
        value |= value >> 4u;
        value |= value >> 8u;
        value |= value >> 16u;
        return value + 1u;
    }

    [[nodiscard]] static auto _find_frame_field_index(const CoroFrameDesc &desc, luisa::string_view name) noexcept {
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

    template<typename Kernel>
    [[nodiscard]] auto _compile_shader(
        Device &device, const Kernel &kernel,
        ShaderOption option) noexcept {
        _shader_structure_hashes.emplace_back(
            kernel.function()->function().hash());
        return device.compile(kernel, std::move(option));
    }

    void _create_shader(Device &device, const Coro &coro) {
        _shader_structure_hashes.clear();
        size_t nc = coro.subroutine_count();
        _frame_layout = _config.global_memory_soa ?
                            CoroFrameStorageLayout::make_runtime_soa(coro.frame(), _config.thread_count) :
                            CoroFrameStorageLayout::make_aos(coro.frame(), _config.thread_count);
        _input_fields.resize(nc);
        _output_fields.resize(nc);
        _transition_output_fields.resize(nc);
        _last_dispatch_stats.continuations.clear();
        _last_dispatch_stats.continuations.reserve(nc);
        for (auto i = 0u; i < nc; i++) {
            auto &&node = coro.graph().node(i);
            _last_dispatch_stats.continuations.emplace_back(
                WavefrontCoroContinuationStats{
                    .index = node.index,
                    .token = node.token,
                    .name = node.index == 0u ?
                                luisa::string{"<entry>"} :
                                node.name});
            _input_fields[i] = coro_frame_collect_input_fields(coro.graph(), i);
            _output_fields[i] = coro_frame_collect_output_fields(coro.graph(), i);
            _transition_output_fields[i].resize(nc);
            for (auto j = 0u; j < nc; j++) {
                _transition_output_fields[i][j] = coro_frame_collect_output_fields(coro.graph(), i, j);
            }
        }
        _resume_kernels.resize(nc);
        _host_count.resize(nc);
        _host_offset.resize(nc);
        _have_hint.resize(nc, false);
        _refill_at.resize(nc, false);

        for (auto &name : _config.refill_continuations) {
            auto node = coro.graph().node_by_name(name);
            LUISA_ASSERT(node != nullptr && node->index != 0u &&
                             node->index < nc,
                         "Wavefront refill continuation '{}' does not name a "
                         "valid non-entry coroutine suspension.",
                         name);
            _refill_at[node->index] = true;
        }

        luisa::vector<luisa::string> valid_hint_fields;
        valid_hint_fields.reserve(_config.hint_fields.size());
        for (auto &name : _config.hint_fields) {
            if (auto node = coro.graph().node_by_name(name)) {
                if (node->index == 0u || node->index >= nc) {
                    LUISA_WARNING("Coroutine hint field '{}' resolves to invalid node {}; hint disabled.", name, node->index);
                } else {
                    _have_hint[node->index] = true;
                    valid_hint_fields.emplace_back(name);
                }
            } else {
                LUISA_WARNING("Coroutine hint field '{}' does not match a suspend name; hint disabled.", name);
            }
        }
        _config.hint_fields = std::move(valid_hint_fields);
        if (auto hint_count = _valid_hint_field_count(); hint_count != 0u) {
            _hint_field_index = _find_frame_field_index(coro.frame(), "coro_hint");
            if (_hint_field_index == static_cast<size_t>(-1)) {
                LUISA_WARNING("WavefrontCoroSchedulerConfig::hint_fields requires a uint frame value explicitly exported as 'coro_hint'; hint sorting is disabled.");
                std::fill(_have_hint.begin(), _have_hint.end(), false);
                _config.hint_fields.clear();
            } else if (coro.frame().frame_field_type(_hint_field_index) != Type::of<uint>()) {
                LUISA_WARNING("Coroutine frame export 'coro_hint' must be uint; hint sorting is disabled.");
                std::fill(_have_hint.begin(), _have_hint.end(), false);
                _config.hint_fields.clear();
            }
        }
        // Small hint ranges use the subgroup-independent bucket path. Larger
        // ranges use one-sweep radix sorting, whose rank construction is
        // explicitly defined for 32-lane subgroups. Hint sorting is only a
        // scheduling optimization, so disable that hint when the device does
        // not satisfy the algorithmic precondition; coroutine semantics and
        // token gathering remain unchanged.
        if (_valid_hint_field_count() != 0u &&
            _config.hint_range > radix_sort::hist_block_size &&
            device.compute_warp_size() != radix_sort::warp_size) {
            LUISA_WARNING(
                "Wavefront coroutine hint sorting over range {} requires "
                "{}-lane subgroups, but the device reports {}; hint sorting "
                "is disabled.",
                _config.hint_range, radix_sort::warp_size,
                device.compute_warp_size());
            std::fill(_have_hint.begin(), _have_hint.end(), false);
            _config.hint_fields.clear();
        }
        _has_hint_sort = _valid_hint_field_count() != 0u;
        auto use_token_sort =
            _config.gather_by_sorting &&
            !_config.incremental_continuation_counts;
        auto use_sort = use_token_sort || _has_hint_sort;

        _frame_buffer = device.create_byte_buffer(_frame_layout.size_bytes);
        // The scheduler owns this complete allocation for the lifetime of all
        // generated shaders. Capturing it makes the zero-offset binding part
        // of the shader ABI (but not its structural hash), allowing backends
        // to prove the frame base alignment without weakening arbitrary
        // ByteBufferView semantics for user arguments.
        auto *frame_buffer = &_frame_buffer;
        _resume_index = device.create_buffer<uint>(_config.thread_count);
        _resume_count = device.create_buffer<uint>(nc);
        // Incremental queue counts are scheduler-owned captured state. For
        // every nonterminal continuation t, C[t] is the cardinality of the
        // active-prefix frames whose target token is t. Queue zero is derived
        // as prefix_size - sum(C[1..]) and is deliberately not maintained.
        // Keeping the buffer captured avoids changing every coroutine user's
        // argument ABI; the host policy still enters the shader structure
        // through its compile-time C++ branch below.
        auto *queue_count = &_resume_count;
        _resume_offset = device.create_buffer<uint>(nc);
        _global_buffer = device.create_buffer<uint>(1u);
        if (use_sort) {
            _sort_index = device.create_buffer<uint>(_config.thread_count);
            _sort_key[0] = device.create_buffer<uint>(_config.thread_count);
            _sort_key[1] = device.create_buffer<uint>(_config.thread_count);
            auto max_digit = std::max<uint>(static_cast<uint>(nc),
                                            std::min<uint>(_config.hint_range, radix_sort::hist_block_size));
            _sort_temp_storage = radix_sort::temp_storage{device, _config.thread_count, max_digit};
        }

        auto token_to_index = detail::make_coro_token_index_callable(coro);
        Callable<uint(uint, ByteBuffer, uint)> read_scheduler_token = [layout = _frame_layout, soa = _config.global_memory_soa](
                                                                          UInt index, ByteBufferVar frame_buf,
                                                                          UInt frame_capacity) noexcept {
            return coro_frame_read_field<uint>(
                frame_buf, index, frame_capacity, layout, soa, 6u);
        };

        Callable<uint(uint, ByteBuffer, uint)> get_scheduler_token = [read_scheduler_token](
                                                                         UInt index, ByteBufferVar frame_buf,
                                                                         UInt frame_capacity) noexcept {
            return read_scheduler_token(index, frame_buf, frame_capacity);
        };
        Callable<uint(uint, ByteBuffer, uint)> identity_index = [](
                                                                    UInt index, ByteBufferVar,
                                                                    UInt) noexcept {
            return index;
        };
        if (use_token_sort) {
            _sort_token = radix_sort::instance<ByteBuffer, uint>{
                device, _config.thread_count, _sort_temp_storage,
                &get_scheduler_token, &identity_index, &get_scheduler_token,
                1u, static_cast<uint>(nc)};
        }
        if (_has_hint_sort) {
            Callable<uint(uint, Buffer<uint>, ByteBuffer, uint)> keep_index = [](
                                                                               UInt index, BufferUInt values,
                                                                               ByteBufferVar, UInt) noexcept {
                return values.read(index);
            };
            Callable<uint(uint, Buffer<uint>, ByteBuffer, uint)> get_coro_hint = [layout = _frame_layout, soa = _config.global_memory_soa,
                                                                                  hint_field_index = static_cast<uint>(_hint_field_index)](
                                                                                     UInt index, BufferUInt values,
                                                                                     ByteBufferVar frame_buf,
                                                                                     UInt frame_capacity) noexcept {
                auto frame_index = values.read(index);
                return coro_frame_read_field<uint>(
                    frame_buf, frame_index, frame_capacity,
                    layout, soa, hint_field_index);
            };
            if (_config.hint_range <= radix_sort::hist_block_size) {
                auto hint_digit = std::max<uint>(_config.hint_range, 1u);
                _sort_hint = radix_sort::instance<Buffer<uint>, ByteBuffer, uint>{
                    device, _config.thread_count, _sort_temp_storage,
                    &get_coro_hint, &keep_index, &get_coro_hint,
                    1u, hint_digit};
            } else {
                auto high_bit = 0u;
                while ((_config.hint_range >> high_bit) != 1u) { high_bit++; }
                _sort_hint = radix_sort::instance<Buffer<uint>, ByteBuffer, uint>{
                    device, _config.thread_count, _sort_temp_storage,
                    &get_coro_hint, &keep_index, &get_coro_hint,
                    0u, radix_sort::hist_block_size, 0u, high_bit};
            }
        }

        if (auto entry_sub = coro[0u]) {
            Kernel1D k_gen = [&coro, frame_buffer, queue_count, layout = _frame_layout, output_fields = _transition_output_fields[0u],
                              soa = _config.global_memory_soa, compact = _config.frame_buffer_compaction,
                              incremental = _config.incremental_continuation_counts,
                              execution_block_size = _config.execution_block_size,
                              token_to_index](
                                 UInt frame_capacity,
                                 BufferUInt resume_index,
                                 UInt index_offset, UInt frame_offset, UInt global_start,
                                 UInt count, UInt3 dispatch_shape,
                                 Var<Args>... k_args) noexcept {
                set_block_size(execution_block_size);
                auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto frame_id = compact ? frame_offset + x : resume_index.read(index_offset + x);
                auto logical_id = _dispatch_id_from_linear_index(global_start + x, dispatch_shape);
                auto frame = coro.instantiate(logical_id, dispatch_shape);
                frame.target_token = 0u;
                coro.entry()(frame, k_args...);
                auto next = token_to_index(frame.target_token);
                frame.target_token = next;
                if (incremental) {
                    auto counts = Expr<Buffer<uint>>{*queue_count};
                    $if(next != 0u) {
                        counts.atomic(next).fetch_add(1u);
                    };
                }
                for (size_t target = 0u; target < output_fields.size(); ++target) {
                    $if (next == static_cast<uint>(target)) {
                        coro_frame_store(
                            frame_buf, frame_id, frame_capacity, frame,
                            layout, soa, luisa::span{output_fields[target]});
                    };
                }
            };
            _gen_kernel = _compile_shader(
                device,
                k_gen,
                detail::coro_scheduler_shader_option(
                    _config.shader_option, "wavefront_generate"));
        }

        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = coro[i];
            if (!cont_sub) continue;
            Kernel1D k_cont = [&coro, frame_buffer, queue_count, layout = _frame_layout, input_fields = _input_fields[i], output_fields = _transition_output_fields[i],
                               soa = _config.global_memory_soa, i,
                               incremental = _config.incremental_continuation_counts,
                               execution_block_size = _config.execution_block_size,
                               read_scheduler_token, token_to_index](
                                  UInt frame_capacity,
                                  BufferUInt resume_index,
                                  UInt resume_offset, UInt count,
                                  Var<Args>... k_args) noexcept {
                set_block_size(execution_block_size);
                auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto idx = resume_index.read(resume_offset + x);
                auto tok = read_scheduler_token(idx, frame_buf, frame_capacity);
                $if (tok != static_cast<uint>(i)) { $return(); };
                if (incremental) {
                    auto counts = Expr<Buffer<uint>>{*queue_count};
                    counts.atomic(static_cast<uint>(i)).fetch_sub(1u);
                }
                auto frame = coro_frame_load(
                    &coro.frame(), frame_buf, idx, frame_capacity,
                    layout, soa, luisa::span{input_fields});
                frame.target_token = CoroFrame::TERMINAL_TOKEN;
                coro[i](frame, k_args...);
                auto next = token_to_index(frame.target_token);
                frame.target_token = next;
                if (incremental) {
                    auto counts = Expr<Buffer<uint>>{*queue_count};
                    $if(next != 0u) {
                        counts.atomic(next).fetch_add(1u);
                    };
                }
                for (size_t target = 0u; target < output_fields.size(); ++target) {
                    $if (next == static_cast<uint>(target)) {
                        coro_frame_store(
                            frame_buf, idx, frame_capacity, frame,
                            layout, soa, luisa::span{output_fields[target]});
                    };
                }
            };
            _resume_kernels[i] = _compile_shader(
                device,
                k_cont,
                detail::coro_scheduler_shader_option(
                    _config.shader_option,
                    luisa::format("wavefront_resume_{}", i)));
        }

        Kernel1D initialize_kernel = [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa](
                                         UInt frame_capacity,
                                         UInt n) {
            auto buf = Expr<ByteBuffer>{*frame_buffer};
            auto x = dispatch_x();
            $if (x < n) {
                coro_frame_write_field(
                    buf, x, frame_capacity, layout, soa, 6u, 0u);
            };
        };
        _initialize_shader = _compile_shader(
            device, initialize_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_initialize"));

        Kernel1D clear_count_kernel = [](BufferUInt buffer, UInt n) {
            auto x = dispatch_x();
            $if (x < n) { buffer.write(x, 0u); };
        };
        _clear_count_shader = _compile_shader(
            device, clear_count_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_clear_count"));

        Kernel1D count_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, node_count = static_cast<uint>(nc)](
                UInt frame_capacity,
                BufferUInt count, UInt n) noexcept {
            auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
            auto x = dispatch_x();
            $if (x >= n) { $return(); };
            auto tok = read_scheduler_token(x, frame_buf, frame_capacity);
            $if (tok < node_count) {
                count.atomic(tok).fetch_add(1u);
            };
        };
        _count_shader = _compile_shader(
            device, count_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_count"));

        Kernel1D gather_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, node_count = static_cast<uint>(nc)](
                UInt frame_capacity,
                BufferUInt index, BufferUInt offset, UInt n) noexcept {
            auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
            auto x = dispatch_x();
            $if (x >= n) { $return(); };
            auto tok = read_scheduler_token(x, frame_buf, frame_capacity);
            $if (tok < node_count) {
                auto slot = offset.atomic(tok).fetch_add(1u);
                index.write(slot, x);
            };
        };
        _gather_shader = _compile_shader(
            device, gather_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_gather"));

        if (_config.incremental_continuation_counts) {
            Kernel1D gather_selected_kernel =
                [frame_buffer, layout = _frame_layout,
                 soa = _config.global_memory_soa,
                 read_scheduler_token](
                    UInt frame_capacity, BufferUInt index,
                    BufferUInt write_count, UInt selected, UInt n) noexcept {
                    auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                    auto x = dispatch_x();
                    $if(x >= n) { $return(); };
                    auto tok = read_scheduler_token(
                        x, frame_buf, frame_capacity);
                    $if(tok == selected) {
                        auto slot = write_count.atomic(0u).fetch_add(1u);
                        index.write(slot, x);
                    };
                };
            _gather_selected_shader = _compile_shader(
                device, gather_selected_kernel,
                detail::coro_scheduler_shader_option(
                    _config.shader_option, "wavefront_gather_selected"));
        }

        Kernel1D compact_kernel =
            [frame_buffer, layout = _frame_layout, soa = _config.global_memory_soa,
             read_scheduler_token, desc = &coro.frame()](
                UInt frame_capacity,
                BufferUInt index, BufferUInt global,
                UInt active_count, UInt empty_count, UInt scan_count) noexcept {
                auto frame_buf = Expr<ByteBuffer>{*frame_buffer};
                auto x = dispatch_x();
                auto src = active_count + x;
                $if (src >= scan_count) { $return(); };
                auto tok = read_scheduler_token(src, frame_buf, frame_capacity);
                $if (tok != 0u) {
                    auto dst = def(0u);
                    auto found_dst = def(false);
                    $while (!found_dst) {
                        auto res = global.atomic(0u).fetch_add(1u);
                        $if (res >= empty_count) { $break; };
                        dst = index.read(res);
                        found_dst = dst < active_count;
                    };
                    $if (found_dst) {
                        auto frame = coro_frame_load(
                            desc, frame_buf, src, frame_capacity, layout, soa);
                        coro_frame_store(
                            frame_buf, dst, frame_capacity, frame, layout, soa);
                        coro_frame_write_field(
                            frame_buf, src, frame_capacity,
                            layout, soa, 6u, 0u);
                    };
                };
            };
        _compact_shader = _compile_shader(
            device, compact_kernel,
            detail::coro_scheduler_shader_option(
                _config.shader_option, "wavefront_compact"));
    }

    void _sort_token_buckets(Stream &stream, uint count) noexcept {
        if (count == 0u) {
            std::fill(_host_count.begin(), _host_count.end(), 0u);
            std::fill(_host_offset.begin(), _host_offset.end(), 0u);
            return;
        }
        _sort_token.sort(stream, _sort_key[0].view(), _resume_index.view(),
                         _sort_key[1].view(), _resume_index.view(),
                         count, _frame_buffer, _config.thread_count);
        stream << _sort_temp_storage.hist_buffer.view()
                      .subview(0u, _host_offset.size())
                      .copy_to(luisa::span{_host_offset.data(), _host_offset.size()})
               << synchronize();
        for (auto i = 0u; i < _host_count.size(); i++) {
            auto next = i + 1u == _host_count.size() ? count : _host_offset[i + 1u];
            _host_count[i] = next - _host_offset[i];
        }
    }

    [[nodiscard]] auto _sort_hint_range(Stream &stream, uint offset, uint count) noexcept {
        BufferView<uint> indices[2] = {
            _resume_index.view().subview(offset, count),
            _sort_index.view().subview(offset, count)};
        BufferView<uint> keys[2] = {
            _sort_key[0].view().subview(offset, count),
            _sort_key[1].view().subview(offset, count)};
        return _sort_hint.sort_switch(stream, keys, indices, count,
                                      _resume_index.view().subview(offset, count),
                                      _frame_buffer, _config.thread_count);
    }

    void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        uint N = dispatch_size.x * dispatch_size.y * dispatch_size.z;
        auto report_stats = _config.report_stats ||
                            std::getenv("LUISA_CORO_WAVEFRONT_STATS") != nullptr;
        _last_dispatch_stats.collected = report_stats;
        _last_dispatch_stats.iteration_count = 0u;
        _last_dispatch_stats.generated_count = 0u;
        _last_dispatch_stats.resumed_count = 0u;
        _last_dispatch_stats.gather_scan_count = 0u;
        _last_dispatch_stats.compact_scan_count = 0u;
        _last_dispatch_stats.max_scan_count = 0u;
        _last_dispatch_stats.max_active_count = 0u;
        _last_dispatch_stats.elapsed_ms = 0.0;
        for (auto &continuation : _last_dispatch_stats.continuations) {
            continuation.dispatch_count = 0u;
            continuation.executed_count = 0u;
            continuation.peak_queued_count = 0u;
        }
        // `thread_count` is the allocated frame-pool ceiling, not a mandate to
        // initialize and scan every slot. A paper-scale pool (2^24 frames) is
        // intentionally much larger than many tiled or diagnostic dispatches;
        // only logical instances that can exist in this dispatch are active.
        _active_frame_capacity = std::min(_config.thread_count, N);
        if (_active_frame_capacity == 0u) { return; }
        stream << _initialize_shader(
                      _config.thread_count,
                      _active_frame_capacity)
                      .dispatch(_active_frame_capacity);
        _used_frame_count = 0u;

        auto nc = _resume_kernels.size();
        for (size_t i = 0u; i < nc; ++i) {
            _host_count[i] =
                _config.incremental_continuation_counts ?
                    0u :
                    (i == 0u ? _active_frame_capacity : 0u);
            _host_offset[i] = i == 0u ? 0u : _active_frame_capacity;
        }
        stream << _resume_count.copy_from(luisa::span{_host_count.data(), _host_count.size()});

        auto dispatch_counter = 0u;
        auto trace_iterations =
            std::getenv("LUISA_CORO_WAVEFRONT_TRACE_ITERATIONS") != nullptr;
        auto verify_queues =
            std::getenv("LUISA_CORO_WAVEFRONT_VERIFY_QUEUES") != nullptr;
        auto iteration_count = uint64_t{0u};
        auto gather_scan_count = uint64_t{0u};
        auto compact_scan_count = uint64_t{0u};
        auto generated_count = uint64_t{0u};
        auto resumed_count = uint64_t{0u};
        auto max_scan_count = 0u;
        auto max_active_count = 0u;
        Clock dispatch_clock;
        while (true) {
            iteration_count++;
            auto scan_count = _config.frame_buffer_compaction ? _used_frame_count : _active_frame_capacity;
            // A non-draining queue may execute indefinitely. Keep explicit
            // diagnostics useful without turning the failure mode into an
            // unbounded log producer: retain the prefix and then sample at
            // powers of two.
            auto trace_this_iteration =
                trace_iterations &&
                (iteration_count <= 64u ||
                 (iteration_count & (iteration_count - 1u)) == 0u);
            if (trace_this_iteration) {
                LUISA_INFO(
                    "Wavefront iteration {} begin: dispatched={} used={} scan={} capacity={}.",
                    iteration_count, dispatch_counter, _used_frame_count,
                    scan_count, _active_frame_capacity);
            }
            // The incremental policy defers its one selected-token scan until
            // after host selection. Legacy policies materialize every token
            // bucket here.
            if (!_config.incremental_continuation_counts) {
                gather_scan_count += scan_count;
            }
            max_scan_count = std::max(max_scan_count, scan_count);
            if (_config.incremental_continuation_counts) {
                stream << _resume_count.copy_to(
                              luisa::span{_host_count.data(),
                                          _host_count.size()})
                       << synchronize();
                if (verify_queues && scan_count != 0u) {
                    stream << _clear_count_shader(
                                  _resume_offset, static_cast<uint>(nc))
                                  .dispatch(static_cast<uint>(nc));
                    stream << _count_shader(
                                  _config.thread_count,
                                  _resume_offset, scan_count)
                                  .dispatch(scan_count);
                    luisa::vector<uint> actual(nc);
                    stream << _resume_offset.copy_to(luisa::span{actual})
                           << synchronize();
                    for (size_t i = 1u; i < nc; ++i) {
                        LUISA_ASSERT(
                            actual[i] == _host_count[i],
                            "Incremental wavefront queue invariant violation "
                            "at iteration {}, continuation {}: maintained "
                            "count {} differs from materialized count {}.",
                            iteration_count, i, _host_count[i], actual[i]);
                    }
                }
            } else if (_config.gather_by_sorting) {
                _sort_token_buckets(stream, scan_count);
                if (verify_queues && scan_count != 0u) {
                    luisa::vector<uint> indices(scan_count);
                    stream << _resume_index.view().subview(0u, scan_count)
                                  .copy_to(luisa::span{indices})
                           << synchronize();
                    luisa::vector<uint> seen(scan_count, 0u);
                    for (auto slot = 0u; slot < scan_count; slot++) {
                        auto index = indices[slot];
                        LUISA_ASSERT(
                            index < scan_count,
                            "Wavefront sorted queue invariant violation at iteration {}, slot {}: "
                            "frame index {} exceeds scanned prefix {}.",
                            iteration_count, slot, index, scan_count);
                        LUISA_ASSERT(
                            seen[index] == 0u,
                            "Wavefront sorted queue invariant violation at iteration {}, slot {}: "
                            "frame index {} appears more than once.",
                            iteration_count, slot, index);
                        seen[index] = 1u;
                    }
                }
            } else {
                stream << _clear_count_shader(_resume_count, static_cast<uint>(nc)).dispatch(static_cast<uint>(nc));
                if (scan_count != 0u) {
                    stream << _count_shader(
                                  _config.thread_count,
                                  _resume_count, scan_count)
                                  .dispatch(scan_count);
                }
                stream << _resume_count.copy_to(luisa::span{_host_count.data(), _host_count.size()})
                       << synchronize();
            }

            auto active_count = 0u;
            for (size_t i = 1u; i < nc; ++i) {
                active_count += _host_count[i];
                if (report_stats) {
                    auto &continuation =
                        _last_dispatch_stats.continuations[i];
                    continuation.peak_queued_count = std::max(
                        continuation.peak_queued_count,
                        _host_count[i]);
                }
            }
            max_active_count = std::max(max_active_count, active_count);
            LUISA_ASSERT(active_count <= scan_count,
                         "Wavefront coroutine queue invariant violation: active frames ({}) exceed scanned frame prefix ({}).",
                         active_count, scan_count);
            auto empty_count = _config.frame_buffer_compaction ?
                                   _active_frame_capacity - active_count :
                                   _host_count[0u];
            auto compact_empty_count = _config.frame_buffer_compaction ?
                                           scan_count - active_count :
                                           empty_count;
            _host_count[0u] = compact_empty_count;
            if (trace_this_iteration) {
                LUISA_INFO(
                    "Wavefront iteration {} gathered: active={} empty={} compact_empty={}.",
                    iteration_count, active_count, empty_count,
                    compact_empty_count);
            }

            auto active_offset = 0u;
            for (size_t i = 0u; i < nc; ++i) {
                _host_offset[i] = active_offset;
                active_offset += _host_count[i];
            }
            if (dispatch_counter == N && active_count == 0u) { break; }

            auto selected = nc;
            auto selected_count = 0u;
            for (size_t i = 1u; i < nc; ++i) {
                // Strict comparison makes the lowest continuation index the
                // deterministic winner for equal populations, matching
                // Cycles' DeviceKernel scan order.
                if (_host_count[i] > selected_count) {
                    selected = i;
                    selected_count = _host_count[i];
                }
            }

            // The legacy counter/gather path materializes every queue before
            // the refill decision because compaction may need queue zero's
            // empty-slot indices. Keep that established ordering unchanged;
            // only the incremental policy defers a selected-token gather.
            if (!_config.incremental_continuation_counts &&
                !_config.gather_by_sorting && scan_count != 0u) {
                stream << _resume_offset.copy_from(
                              luisa::span{_host_offset.data(),
                                          _host_offset.size()});
                stream << _gather_shader(
                              _config.thread_count,
                              _resume_index, _resume_offset, scan_count)
                              .dispatch(scan_count);
            }

            auto refill_threshold = _config.refill_threshold == 0u ?
                                        _active_frame_capacity / 2u :
                                        std::min(_config.refill_threshold,
                                                 _active_frame_capacity);
            auto refill_aligned = active_count == 0u ||
                                  _config.refill_continuations.empty() ||
                                  (selected < _refill_at.size() &&
                                   _refill_at[selected]);
            // An empty scheduler must always admit work. In particular,
            // floor(capacity / 2) is zero for a one-frame pool, so using only
            // the threshold inequality would leave the empty state as a
            // fixed point and make forward progress impossible.
            auto should_refill = active_count == 0u ||
                                 active_count < refill_threshold;
            if (should_refill && refill_aligned &&
                dispatch_counter < N) {
                auto gen_count = std::min(N - dispatch_counter, empty_count);
                auto frame_offset = active_count;
                if (_config.frame_buffer_compaction && active_count != 0u && compact_empty_count != 0u) {
                    if (_config.incremental_continuation_counts) {
                        // Compaction needs the empty frame indices. They are
                        // scheduler queue zero, so collect only that queue on
                        // the uncommon refill/relocation path.
                        stream << _clear_count_shader(_global_buffer, 1u)
                                      .dispatch(1u);
                        stream << _gather_selected_shader(
                                      _config.thread_count, _resume_index,
                                      _global_buffer, 0u, scan_count)
                                      .dispatch(scan_count);
                        gather_scan_count += scan_count;
                        _host_offset[0u] = 0u;
                        if (verify_queues) {
                            uint gathered_empty_count = 0u;
                            stream << _global_buffer.copy_to(
                                          luisa::span{
                                              &gathered_empty_count, 1u})
                                   << synchronize();
                            LUISA_ASSERT(
                                gathered_empty_count == compact_empty_count,
                                "Incremental wavefront empty-queue gather "
                                "violation at iteration {}: gathered {} "
                                "frames, expected {}.",
                                iteration_count, gathered_empty_count,
                                compact_empty_count);
                        }
                    }
                    stream << _clear_count_shader(_global_buffer, 1u).dispatch(1u);
                    stream << _compact_shader(_config.thread_count,
                                              _resume_index, _global_buffer,
                                              frame_offset, compact_empty_count, scan_count)
                                  .dispatch(scan_count - frame_offset);
                    compact_scan_count += scan_count - frame_offset;
                }
                stream << _gen_kernel(_config.thread_count,
                                      _resume_index,
                                      _host_offset[0u], frame_offset,
                                      dispatch_counter, gen_count, dispatch_size, args...)
                              .dispatch(gen_count);
                if (report_stats) {
                    auto &entry =
                        _last_dispatch_stats.continuations[0u];
                    entry.dispatch_count++;
                    entry.executed_count += gen_count;
                }
                dispatch_counter += gen_count;
                generated_count += gen_count;
                if (_config.frame_buffer_compaction) {
                    _used_frame_count = frame_offset + gen_count;
                }
            } else {
                if (_config.incremental_continuation_counts &&
                    selected < nc && selected_count != 0u &&
                    scan_count != 0u) {
                    stream << _clear_count_shader(_global_buffer, 1u)
                                  .dispatch(1u);
                    stream << _gather_selected_shader(
                                  _config.thread_count, _resume_index,
                                  _global_buffer,
                                  static_cast<uint>(selected), scan_count)
                                  .dispatch(scan_count);
                    if (verify_queues) {
                        uint gathered_count = 0u;
                        stream << _global_buffer.copy_to(
                                      luisa::span{&gathered_count, 1u})
                               << synchronize();
                        LUISA_ASSERT(
                            gathered_count == selected_count,
                            "Incremental wavefront selected-queue gather "
                            "violation at iteration {}, continuation {}: "
                            "gathered {} frames, expected {}.",
                            iteration_count, selected, gathered_count,
                            selected_count);
                    }
                    gather_scan_count += scan_count;
                    _host_offset[selected] = 0u;
                }
                for (size_t i = 1u; i < nc; ++i) {
                    if (_config.largest_continuation_first && i != selected) {
                        continue;
                    }
                    auto count = _host_count[i];
                    if (count == 0u) { continue; }
                    resumed_count += count;
                    if (report_stats) {
                        auto &continuation =
                            _last_dispatch_stats.continuations[i];
                        continuation.dispatch_count++;
                        continuation.executed_count += count;
                    }
                    if (_has_hint_sort && _have_hint[i]) {
                        auto sorted_index = _sort_hint_range(stream, _host_offset[i], count);
                        BufferView<uint> indices[2] = {
                            _resume_index.view().subview(_host_offset[i], count),
                            _sort_index.view().subview(_host_offset[i], count)};
                        stream << _resume_kernels[i](_config.thread_count,
                                                     indices[sorted_index],
                                                     0u, count, args...)
                                      .dispatch(count);
                    } else {
                        stream << _resume_kernels[i](_config.thread_count,
                                                     _resume_index,
                                                     _host_offset[i], count, args...)
                                      .dispatch(count);
                    }
                }
            }
        }
        if (report_stats) {
            _last_dispatch_stats.iteration_count = iteration_count;
            _last_dispatch_stats.generated_count = generated_count;
            _last_dispatch_stats.resumed_count = resumed_count;
            _last_dispatch_stats.gather_scan_count = gather_scan_count;
            _last_dispatch_stats.compact_scan_count = compact_scan_count;
            _last_dispatch_stats.max_scan_count = max_scan_count;
            _last_dispatch_stats.max_active_count = max_active_count;
            _last_dispatch_stats.elapsed_ms = dispatch_clock.toc();
            LUISA_INFO("Wavefront stats: iterations={} generated={} resumed={} gather_scan={} compact_scan={} max_scan={} max_active={} elapsed_ms={:.3f}",
                       iteration_count, generated_count, resumed_count,
                       gather_scan_count, compact_scan_count,
                       max_scan_count, max_active_count,
                       _last_dispatch_stats.elapsed_ms);
            for (auto &&continuation :
                 _last_dispatch_stats.continuations) {
                LUISA_INFO(
                    "Wavefront continuation: index={} token={} name='{}' "
                    "dispatches={} executed={} peak_queued={}.",
                    continuation.index, continuation.token,
                    continuation.name, continuation.dispatch_count,
                    continuation.executed_count,
                    continuation.peak_queued_count);
            }
        }
    }

public:
    [[nodiscard]] const Config &config() const noexcept { return _config; }
    [[nodiscard]] uint active_frame_capacity() const noexcept { return _active_frame_capacity; }
    [[nodiscard]] const WavefrontCoroDispatchStats &
    last_dispatch_stats() const noexcept {
        return _last_dispatch_stats;
    }
    /// Structural hashes of the scheduler-owned generate, continuation, and
    /// queue-management kernels. These exclude allocation sizes and provide a
    /// direct cache-identity diagnostic for scheduler configuration changes.
    [[nodiscard]] luisa::span<const uint64_t>
    shader_structure_hashes() const noexcept {
        return {_shader_structure_hashes.data(),
                _shader_structure_hashes.size()};
    }

    WavefrontCoroScheduler(Device &device, const Coro &coro, const Config &config) noexcept
        : _config{config} {
        LUISA_ASSERT(_config.thread_count != 0u,
                     "Wavefront coroutine frame capacity must be positive.");
        LUISA_ASSERT(_config.execution_block_size >= 32u &&
                         _config.execution_block_size <= 1024u &&
                         _config.execution_block_size % 32u == 0u,
                     "Wavefront coroutine execution block size must be a "
                     "multiple of 32 in [32, 1024], but got {}.",
                     _config.execution_block_size);
        LUISA_ASSERT(
            !_config.incremental_continuation_counts ||
                _config.largest_continuation_first,
            "Incremental selected-queue scheduling requires greedy "
            "largest-continuation-first execution.");
        LUISA_ASSERT(
            !_config.incremental_continuation_counts ||
                _config.frame_buffer_compaction,
            "Incremental selected-queue scheduling currently requires "
            "frame-buffer compaction so refill can use the compact active "
            "prefix without a persistent free-list.");
        _create_shader(device, coro);
    }
    WavefrontCoroScheduler(Device &device, const Coro &coro) noexcept
        : WavefrontCoroScheduler{device, coro, Config{}} {}
};

template<typename... Args>
WavefrontCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> WavefrontCoroScheduler<Args...>;

template<typename... Args>
WavefrontCoroScheduler(Device &, const Coroutine<void(Args...)> &,
                       const WavefrontCoroSchedulerConfig &)
    -> WavefrontCoroScheduler<Args...>;

}// namespace luisa::compute::coro
