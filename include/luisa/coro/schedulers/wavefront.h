//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <cstdlib>

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
};

template<typename... Args>
class WavefrontCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = WavefrontCoroSchedulerConfig;

private:
    Config _config;
    ByteBuffer _frame_buffer;
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint, uint, uint, uint3, Args...> _gen_kernel;
    luisa::vector<Shader1D<ByteBuffer, Buffer<uint>, uint, uint, Args...>> _resume_kernels;
    Shader1D<ByteBuffer, uint> _initialize_shader;
    Shader1D<Buffer<uint>, uint> _clear_count_shader;
    Shader1D<ByteBuffer, Buffer<uint>, uint> _count_shader;
    Shader1D<ByteBuffer, Buffer<uint>, Buffer<uint>, uint> _gather_shader;
    Shader1D<ByteBuffer, Buffer<uint>, Buffer<uint>, uint, uint, uint> _compact_shader;
    Buffer<uint> _resume_index;
    Buffer<uint> _resume_count;
    Buffer<uint> _resume_offset;
    Buffer<uint> _global_buffer;
    Buffer<uint> _sort_key[2];
    Buffer<uint> _sort_index;
    radix_sort::temp_storage _sort_temp_storage;
    radix_sort::instance<ByteBuffer> _sort_token;
    radix_sort::instance<Buffer<uint>, ByteBuffer> _sort_hint;
    luisa::vector<uint> _host_count;
    luisa::vector<uint> _host_offset;
    luisa::vector<bool> _have_hint;
    CoroFrameStorageLayout _frame_layout;
    luisa::vector<luisa::vector<size_t>> _input_fields;
    luisa::vector<luisa::vector<size_t>> _output_fields;
    luisa::vector<luisa::vector<luisa::vector<size_t>>> _transition_output_fields;
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
        for (auto i = 0u; i < desc.field_count(); i++) {
            if (desc.field(i).name == name) {
                return CoroFrameDesc::reserved_field_count + i;
            }
        }
        return static_cast<size_t>(-1);
    }

    [[nodiscard]] auto _valid_hint_field_count() const noexcept {
        auto count = 0u;
        for (auto hint : _have_hint) {
            if (hint) { count++; }
        }
        return count;
    }

    void _create_shader(Device &device, const Coro &coro) {
        size_t nc = coro.subroutine_count();
        _frame_layout = _config.global_memory_soa ?
                            CoroFrameStorageLayout::make_soa(coro.frame(), _config.thread_count) :
                            CoroFrameStorageLayout::make_aos(coro.frame(), _config.thread_count);
        _input_fields.resize(nc);
        _output_fields.resize(nc);
        _transition_output_fields.resize(nc);
        for (auto i = 0u; i < nc; i++) {
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
                LUISA_WARNING("WavefrontCoroSchedulerConfig::hint_fields requires a uint variable named 'coro_hint'; hint sorting is disabled.");
                std::fill(_have_hint.begin(), _have_hint.end(), false);
                _config.hint_fields.clear();
            } else if (coro.frame().frame_field_type(_hint_field_index) != Type::of<uint>()) {
                LUISA_WARNING("Coroutine frame field 'coro_hint' must be uint; hint sorting is disabled.");
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
        auto use_sort = _config.gather_by_sorting || _has_hint_sort;

        _frame_buffer = device.create_byte_buffer(_frame_layout.size_bytes);
        _resume_index = device.create_buffer<uint>(_config.thread_count);
        _resume_count = device.create_buffer<uint>(nc);
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
        Callable<uint(uint, ByteBuffer)> read_scheduler_token = [layout = _frame_layout, soa = _config.global_memory_soa](
                                                                    UInt index, ByteBufferVar frame_buf) noexcept {
            return coro_frame_read_field<uint>(frame_buf, index, layout, soa, 6u);
        };

        Callable<uint(uint, ByteBuffer)> get_scheduler_token = [read_scheduler_token](
                                                                   UInt index, ByteBufferVar frame_buf) noexcept {
            return read_scheduler_token(index, frame_buf);
        };
        Callable<uint(uint, ByteBuffer)> identity_index = [](UInt index, ByteBufferVar) noexcept {
            return index;
        };
        if (_config.gather_by_sorting) {
            _sort_token = radix_sort::instance<ByteBuffer>{
                device, _config.thread_count, _sort_temp_storage,
                &get_scheduler_token, &identity_index, &get_scheduler_token,
                1u, static_cast<uint>(nc)};
        }
        if (_has_hint_sort) {
            Callable<uint(uint, Buffer<uint>, ByteBuffer)> keep_index = [](UInt index, BufferUInt values, ByteBufferVar) noexcept {
                return values.read(index);
            };
            Callable<uint(uint, Buffer<uint>, ByteBuffer)> get_coro_hint = [layout = _frame_layout, soa = _config.global_memory_soa,
                                                                            hint_field_index = static_cast<uint>(_hint_field_index)](
                                                                               UInt index, BufferUInt values, ByteBufferVar frame_buf) noexcept {
                auto frame_index = values.read(index);
                return coro_frame_read_field<uint>(frame_buf, frame_index, layout, soa, hint_field_index);
            };
            if (_config.hint_range <= radix_sort::hist_block_size) {
                auto hint_digit = std::max<uint>(_config.hint_range, 1u);
                _sort_hint = radix_sort::instance<Buffer<uint>, ByteBuffer>{
                    device, _config.thread_count, _sort_temp_storage,
                    &get_coro_hint, &keep_index, &get_coro_hint,
                    1u, hint_digit};
            } else {
                auto high_bit = 0u;
                while ((_config.hint_range >> high_bit) != 1u) { high_bit++; }
                _sort_hint = radix_sort::instance<Buffer<uint>, ByteBuffer>{
                    device, _config.thread_count, _sort_temp_storage,
                    &get_coro_hint, &keep_index, &get_coro_hint,
                    0u, radix_sort::hist_block_size, 0u, high_bit};
            }
        }

        if (auto entry_sub = coro[0u]) {
            Kernel1D k_gen = [&coro, layout = _frame_layout, output_fields = _transition_output_fields[0u],
                              soa = _config.global_memory_soa, compact = _config.frame_buffer_compaction,
                              token_to_index](
                                 ByteBufferVar frame_buf, BufferUInt resume_index,
                                 UInt index_offset, UInt frame_offset, UInt global_start,
                                 UInt count, UInt3 dispatch_shape,
                                 Var<Args>... k_args) noexcept {
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto frame_id = compact ? frame_offset + x : resume_index.read(index_offset + x);
                auto logical_id = _dispatch_id_from_linear_index(global_start + x, dispatch_shape);
                auto frame = coro.instantiate(logical_id, dispatch_shape);
                frame.target_token = 0u;
                coro.entry()(frame, k_args...);
                auto next = token_to_index(frame.target_token);
                frame.target_token = next;
                for (size_t target = 0u; target < output_fields.size(); ++target) {
                    $if (next == static_cast<uint>(target)) {
                        coro_frame_store(frame_buf, frame_id, frame, layout, soa, luisa::span{output_fields[target]});
                    };
                }
            };
            _gen_kernel = device.compile(
                k_gen,
                detail::coro_scheduler_shader_option(
                    _config.shader_option, "wavefront_generate"));
        }

        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = coro[i];
            if (!cont_sub) continue;
            Kernel1D k_cont = [&coro, layout = _frame_layout, input_fields = _input_fields[i], output_fields = _transition_output_fields[i],
                               soa = _config.global_memory_soa, i,
                               read_scheduler_token, token_to_index](
                                  ByteBufferVar frame_buf, BufferUInt resume_index,
                                  UInt resume_offset, UInt count,
                                  Var<Args>... k_args) noexcept {
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto idx = resume_index.read(resume_offset + x);
                auto tok = read_scheduler_token(idx, frame_buf);
                $if (tok != static_cast<uint>(i)) { $return(); };
                auto frame = coro_frame_load(&coro.frame(), frame_buf, idx, layout, soa, luisa::span{input_fields});
                frame.target_token = CoroFrame::TERMINAL_TOKEN;
                coro[i](frame, k_args...);
                auto next = token_to_index(frame.target_token);
                frame.target_token = next;
                for (size_t target = 0u; target < output_fields.size(); ++target) {
                    $if (next == static_cast<uint>(target)) {
                        coro_frame_store(frame_buf, idx, frame, layout, soa, luisa::span{output_fields[target]});
                    };
                }
            };
            _resume_kernels[i] = device.compile(
                k_cont,
                detail::coro_scheduler_shader_option(
                    _config.shader_option,
                    luisa::format("wavefront_resume_{}", i)));
        }

        _initialize_shader = device.compile<1>([layout = _frame_layout, soa = _config.global_memory_soa](ByteBufferVar buf, UInt n) {
            auto x = dispatch_x();
            $if (x < n) {
                coro_frame_write_field(buf, x, layout, soa, 6u, 0u);
            };
        }, detail::coro_scheduler_shader_option(
               _config.shader_option, "wavefront_initialize"));

        _clear_count_shader = device.compile<1>([](BufferUInt buffer, UInt n) {
            auto x = dispatch_x();
            $if (x < n) { buffer.write(x, 0u); };
        }, detail::coro_scheduler_shader_option(
               _config.shader_option, "wavefront_clear_count"));

        _count_shader = device.compile<1>(
            [layout = _frame_layout, soa = _config.global_memory_soa, read_scheduler_token,
             node_count = static_cast<uint>(nc)](
                ByteBufferVar frame_buf, BufferUInt count, UInt n) noexcept {
                auto x = dispatch_x();
                $if (x >= n) { $return(); };
                auto tok = read_scheduler_token(x, frame_buf);
                $if (tok < node_count) {
                    count.atomic(tok).fetch_add(1u);
                };
            }, detail::coro_scheduler_shader_option(
                   _config.shader_option, "wavefront_count"));

        _gather_shader = device.compile<1>(
            [layout = _frame_layout, soa = _config.global_memory_soa, read_scheduler_token,
             node_count = static_cast<uint>(nc)](
                ByteBufferVar frame_buf, BufferUInt index, BufferUInt offset, UInt n) noexcept {
                auto x = dispatch_x();
                $if (x >= n) { $return(); };
                auto tok = read_scheduler_token(x, frame_buf);
                $if (tok < node_count) {
                    auto slot = offset.atomic(tok).fetch_add(1u);
                    index.write(slot, x);
                };
            }, detail::coro_scheduler_shader_option(
                   _config.shader_option, "wavefront_gather"));

        _compact_shader = device.compile<1>(
            [layout = _frame_layout, soa = _config.global_memory_soa, read_scheduler_token, desc = &coro.frame()](
                ByteBufferVar frame_buf, BufferUInt index, BufferUInt global,
                UInt active_count, UInt empty_count, UInt scan_count) noexcept {
                auto x = dispatch_x();
                auto src = active_count + x;
                $if (src >= scan_count) { $return(); };
                auto tok = read_scheduler_token(src, frame_buf);
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
                        auto frame = coro_frame_load(desc, frame_buf, src, layout, soa);
                        coro_frame_store(frame_buf, dst, frame, layout, soa);
                        coro_frame_write_field(frame_buf, src, layout, soa, 6u, 0u);
                    };
                };
            }, detail::coro_scheduler_shader_option(
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
                         count, _frame_buffer);
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
                                      _frame_buffer);
    }

    void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        uint N = dispatch_size.x * dispatch_size.y * dispatch_size.z;
        // `thread_count` is the allocated frame-pool ceiling, not a mandate to
        // initialize and scan every slot. A paper-scale pool (2^24 frames) is
        // intentionally much larger than many tiled or diagnostic dispatches;
        // only logical instances that can exist in this dispatch are active.
        _active_frame_capacity = std::min(_config.thread_count, N);
        if (_active_frame_capacity == 0u) { return; }
        stream << _initialize_shader(_frame_buffer, _active_frame_capacity).dispatch(_active_frame_capacity);
        _used_frame_count = 0u;

        auto nc = _resume_kernels.size();
        for (size_t i = 0u; i < nc; ++i) {
            _host_count[i] = i == 0u ? _active_frame_capacity : 0u;
            _host_offset[i] = i == 0u ? 0u : _active_frame_capacity;
        }
        stream << _resume_count.copy_from(luisa::span{_host_count.data(), _host_count.size()});

        auto dispatch_counter = 0u;
        auto report_stats = _config.report_stats || std::getenv("LUISA_CORO_WAVEFRONT_STATS") != nullptr;
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
            gather_scan_count += scan_count;
            max_scan_count = std::max(max_scan_count, scan_count);
            if (_config.gather_by_sorting) {
                _sort_token_buckets(stream, scan_count);
            } else {
                stream << _clear_count_shader(_resume_count, static_cast<uint>(nc)).dispatch(static_cast<uint>(nc));
                if (scan_count != 0u) {
                    stream << _count_shader(_frame_buffer, _resume_count, scan_count).dispatch(scan_count);
                }
                stream << _resume_count.copy_to(luisa::span{_host_count.data(), _host_count.size()})
                       << synchronize();
            }

            auto active_count = 0u;
            for (size_t i = 1u; i < nc; ++i) {
                active_count += _host_count[i];
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

            auto active_offset = 0u;
            for (size_t i = 0u; i < nc; ++i) {
                _host_offset[i] = active_offset;
                active_offset += _host_count[i];
            }
            if (dispatch_counter == N && active_count == 0u) { break; }

            if (!_config.gather_by_sorting && scan_count != 0u) {
                stream << _resume_offset.copy_from(luisa::span{_host_offset.data(), _host_offset.size()});
                stream << _gather_shader(_frame_buffer, _resume_index, _resume_offset, scan_count).dispatch(scan_count);
            }

            if (empty_count > _active_frame_capacity / 2u && dispatch_counter < N) {
                auto gen_count = std::min(N - dispatch_counter, empty_count);
                auto frame_offset = active_count;
                if (_config.frame_buffer_compaction && active_count != 0u && compact_empty_count != 0u) {
                    stream << _clear_count_shader(_global_buffer, 1u).dispatch(1u);
                    stream << _compact_shader(_frame_buffer, _resume_index, _global_buffer,
                                              frame_offset, compact_empty_count, scan_count)
                                  .dispatch(scan_count - frame_offset);
                    compact_scan_count += scan_count - frame_offset;
                }
                stream << _gen_kernel(_frame_buffer, _resume_index,
                                      _host_offset[0u], frame_offset,
                                      dispatch_counter, gen_count, dispatch_size, args...)
                              .dispatch(gen_count);
                dispatch_counter += gen_count;
                generated_count += gen_count;
                if (_config.frame_buffer_compaction) {
                    _used_frame_count = frame_offset + gen_count;
                }
            } else {
                for (size_t i = 1u; i < nc; ++i) {
                    auto count = _host_count[i];
                    if (count == 0u) { continue; }
                    resumed_count += count;
                    if (_has_hint_sort && _have_hint[i]) {
                        auto sorted_index = _sort_hint_range(stream, _host_offset[i], count);
                        BufferView<uint> indices[2] = {
                            _resume_index.view().subview(_host_offset[i], count),
                            _sort_index.view().subview(_host_offset[i], count)};
                        stream << _resume_kernels[i](_frame_buffer, indices[sorted_index],
                                                     0u, count, args...)
                                      .dispatch(count);
                    } else {
                        stream << _resume_kernels[i](_frame_buffer, _resume_index,
                                                     _host_offset[i], count, args...)
                                      .dispatch(count);
                    }
                }
            }
            if (!_config.gather_by_sorting) {
            }
        }
        if (report_stats) {
            LUISA_INFO("Wavefront stats: iterations={} generated={} resumed={} gather_scan={} compact_scan={} max_scan={} max_active={} elapsed_ms={:.3f}",
                       iteration_count, generated_count, resumed_count,
                       gather_scan_count, compact_scan_count,
                       max_scan_count, max_active_count, dispatch_clock.toc());
        }
    }

public:
    [[nodiscard]] const Config &config() const noexcept { return _config; }
    [[nodiscard]] uint active_frame_capacity() const noexcept { return _active_frame_capacity; }

    WavefrontCoroScheduler(Device &device, const Coro &coro, const Config &config) noexcept
        : _config{config} {
        LUISA_ASSERT(_config.thread_count != 0u,
                     "Wavefront coroutine frame capacity must be positive.");
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
