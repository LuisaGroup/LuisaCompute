//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/coro/coro_frame_storage.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/coro/radix_sort.h>
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
};

template<typename... Args>
class WavefrontCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = WavefrontCoroSchedulerConfig;

private:
    Config _config;
    ByteBuffer _frame_buffer;
    Shader1D<ByteBuffer, Buffer<uint>, Buffer<uint>, uint, uint, uint, uint, uint3, Args...> _gen_kernel;
    luisa::vector<Shader1D<ByteBuffer, Buffer<uint>, Buffer<uint>, uint, uint, Args...>> _resume_kernels;
    Shader<1, ByteBuffer, uint> _clear_shader;
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
    size_t _hint_field_index{static_cast<size_t>(-1)};
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
        for (auto i = 0u; i < nc; i++) {
            _input_fields[i] = coro_frame_collect_input_fields(coro.graph(), i);
            _output_fields[i] = coro_frame_collect_output_fields(coro.graph(), i);
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

        auto token_to_index = [&coro](UInt target_token) noexcept {
            auto next = def(0u);
            $if (target_token != CoroFrame::TERMINAL_TOKEN) {
                for (size_t i = 1u; i < coro.subroutine_count(); ++i) {
                    $if (target_token == coro.trigger_token(i)) {
                        next = static_cast<uint>(i);
                    };
                }
            };
            return next;
        };

        Callable<uint(uint, ByteBuffer)> get_coro_token = [layout = _frame_layout, soa = _config.global_memory_soa,
                                                           token_to_index](UInt index, ByteBufferVar frame_buf) noexcept {
            auto token = coro_frame_read_field<uint>(frame_buf, index, layout, soa, 1u);
            return token_to_index(token);
        };
        Callable<uint(uint, ByteBuffer)> identity_index = [](UInt index, ByteBufferVar) noexcept {
            return index;
        };
        if (_config.gather_by_sorting) {
            _sort_token = radix_sort::instance<ByteBuffer>{
                device, _config.thread_count, _sort_temp_storage,
                &get_coro_token, &identity_index, &get_coro_token,
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
            Kernel1D k_gen = [&coro, layout = _frame_layout, output_fields = _output_fields[0u],
                              soa = _config.global_memory_soa, compact = _config.frame_buffer_compaction,
                              sorted = _config.gather_by_sorting, token_to_index](
                                 ByteBufferVar frame_buf, BufferUInt resume_index,
                                 BufferUInt resume_count,
                                 UInt index_offset, UInt frame_offset, UInt global_start,
                                 UInt count, UInt3 dispatch_shape,
                                 Var<Args>... k_args) noexcept {
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto frame_id = compact ? frame_offset + x : resume_index.read(index_offset + x);
                if (!sorted) {
                    resume_count.atomic(0u).fetch_sub(1u);
                }
                auto logical_id = _dispatch_id_from_linear_index(global_start + x, dispatch_shape);
                auto frame = coro.instantiate(logical_id);
                frame.target_token = 0u;
                frame.skip_flag = 0u;
                coro.entry()(frame, k_args...);
                coro_frame_store(frame_buf, frame_id, frame, layout, soa, luisa::span{output_fields});
                if (!sorted) {
                    auto next = token_to_index(frame.target_token);
                    resume_count.atomic(next).fetch_add(1u);
                }
            };
            _gen_kernel = device.compile(k_gen);
        }

        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = coro[i];
            if (!cont_sub) continue;
            uint my_token = static_cast<uint>(coro.graph().node(i).token);
            Kernel1D k_cont = [&coro, layout = _frame_layout, input_fields = _input_fields[i], output_fields = _output_fields[i],
                               soa = _config.global_memory_soa, my_token, i,
                               sorted = _config.gather_by_sorting, token_to_index](
                                  ByteBufferVar frame_buf, BufferUInt resume_index,
                                  BufferUInt resume_count,
                                  UInt resume_offset, UInt count,
                                  Var<Args>... k_args) noexcept {
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto idx = resume_index.read(resume_offset + x);
                auto tok = coro_frame_read_field<uint>(frame_buf, idx, layout, soa, 1u);
                $if (tok != my_token) { $return(); };
                if (!sorted) {
                    resume_count.atomic(static_cast<uint>(i)).fetch_sub(1u);
                }
                auto frame = coro_frame_load(&coro.frame(), frame_buf, idx, layout, soa, luisa::span{input_fields});
                frame.skip_flag = 0u;
                coro[i](frame, k_args...);
                coro_frame_store(frame_buf, idx, frame, layout, soa, luisa::span{output_fields});
                if (!sorted) {
                    auto next = token_to_index(frame.target_token);
                    resume_count.atomic(next).fetch_add(1u);
                }
            };
            _resume_kernels[i] = device.compile(k_cont);
        }

        _clear_shader = device.compile<1>([](ByteBufferVar buf, UInt n) {
            auto x = dispatch_x();
            $if (x < n) { buf.write(x * 4u, 0u); };
        });

        _clear_count_shader = device.compile<1>([](BufferUInt buffer, UInt n) {
            auto x = dispatch_x();
            $if (x < n) { buffer.write(x, 0u); };
        });

        _count_shader = device.compile<1>(
            [layout = _frame_layout, soa = _config.global_memory_soa, token_to_index](
                ByteBufferVar frame_buf, BufferUInt count, UInt n) noexcept {
                auto x = dispatch_x();
                $if (x >= n) { $return(); };
                auto tok = coro_frame_read_field<uint>(frame_buf, x, layout, soa, 1u);
                auto bucket = token_to_index(tok);
                count.atomic(bucket).fetch_add(1u);
            });

        _gather_shader = device.compile<1>(
            [layout = _frame_layout, soa = _config.global_memory_soa, token_to_index](
                ByteBufferVar frame_buf, BufferUInt index, BufferUInt offset, UInt n) noexcept {
                auto x = dispatch_x();
                $if (x >= n) { $return(); };
                auto tok = coro_frame_read_field<uint>(frame_buf, x, layout, soa, 1u);
                auto bucket = token_to_index(tok);
                auto slot = offset.atomic(bucket).fetch_add(1u);
                index.write(slot, x);
            });

        _compact_shader = device.compile<1>(
            [layout = _frame_layout, soa = _config.global_memory_soa, desc = &coro.frame()](
                ByteBufferVar frame_buf, BufferUInt index, BufferUInt global,
                UInt empty_offset, UInt capacity, UInt sorted) noexcept {
                auto x = dispatch_x();
                auto src = empty_offset + x;
                $if (src >= capacity) { $return(); };
                auto tok = coro_frame_read_field<uint>(frame_buf, src, layout, soa, 1u);
                $if (tok != 0u & tok != CoroFrame::TERMINAL_TOKEN) {
                    auto res = global.atomic(0u).fetch_add(1u);
                    auto dst = index.read(res);
                    $while (sorted == 0u & dst >= empty_offset) {
                        res = global.atomic(0u).fetch_add(1u);
                        dst = index.read(res);
                    };
                    auto frame = coro_frame_load(desc, frame_buf, src, layout, soa);
                    coro_frame_store(frame_buf, dst, frame, layout, soa);
                    coro_frame_write_field(frame_buf, src, layout, soa, 1u, 0u);
                };
            });
    }

    void _sort_token_buckets(Stream &stream) noexcept {
        _sort_token.sort(stream, _sort_key[0].view(), _resume_index.view(),
                         _sort_key[1].view(), _resume_index.view(),
                         _config.thread_count, _frame_buffer);
        stream << _sort_temp_storage.hist_buffer.view()
                      .subview(0u, _host_offset.size())
                      .copy_to(luisa::span{_host_offset.data(), _host_offset.size()})
               << synchronize();
        for (auto i = 0u; i < _host_count.size(); i++) {
            auto next = i + 1u == _host_count.size() ? _config.thread_count : _host_offset[i + 1u];
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
        auto uint_count = static_cast<uint>((_frame_buffer.size_bytes() + sizeof(uint) - 1u) / sizeof(uint));
        stream << _clear_shader(_frame_buffer, uint_count).dispatch(uint_count);

        auto nc = _resume_kernels.size();
        for (size_t i = 0u; i < nc; ++i) {
            _host_count[i] = i == 0u ? _config.thread_count : 0u;
            _host_offset[i] = i == 0u ? 0u : _config.thread_count;
        }
        stream << _resume_count.copy_from(luisa::span{_host_count.data(), _host_count.size()});

        auto dispatch_counter = 0u;
        while (true) {
            if (_config.gather_by_sorting) {
                _sort_token_buckets(stream);
            }

            auto active_count = 0u;
            auto active_offset = 0u;
            for (size_t i = 0u; i < nc; ++i) {
                _host_offset[i] = active_offset;
                active_offset += _host_count[i];
                if (i != 0u) { active_count += _host_count[i]; }
            }
            if (dispatch_counter == N && active_count == 0u) { break; }

            if (!_config.gather_by_sorting) {
                stream << _resume_offset.copy_from(luisa::span{_host_offset.data(), _host_offset.size()});
                stream << _gather_shader(_frame_buffer, _resume_index, _resume_offset, _config.thread_count).dispatch(_config.thread_count);
            }

            auto empty_count = _host_count[0u];
            if (empty_count > _config.thread_count / 2u && dispatch_counter < N) {
                auto gen_count = std::min(N - dispatch_counter, empty_count);
                auto frame_offset = _config.thread_count - empty_count;
                if (_config.frame_buffer_compaction && empty_count != _config.thread_count) {
                    stream << _clear_count_shader(_global_buffer, 1u).dispatch(1u);
                    stream << _compact_shader(_frame_buffer, _resume_index, _global_buffer,
                                              frame_offset, _config.thread_count,
                                              _config.gather_by_sorting ? 1u : 0u)
                                  .dispatch(empty_count);
                }
                stream << _gen_kernel(_frame_buffer, _resume_index, _resume_count,
                                      _host_offset[0u], frame_offset,
                                      dispatch_counter, gen_count, dispatch_size, args...)
                              .dispatch(gen_count);
                dispatch_counter += gen_count;
            } else {
                for (size_t i = 1u; i < nc; ++i) {
                    auto count = _host_count[i];
                    if (count == 0u) { continue; }
                    if (_has_hint_sort && _have_hint[i]) {
                        auto sorted_index = _sort_hint_range(stream, _host_offset[i], count);
                        BufferView<uint> indices[2] = {
                            _resume_index.view().subview(_host_offset[i], count),
                            _sort_index.view().subview(_host_offset[i], count)};
                        stream << _resume_kernels[i](_frame_buffer, indices[sorted_index], _resume_count,
                                                     0u, count, args...)
                                      .dispatch(count);
                    } else {
                        stream << _resume_kernels[i](_frame_buffer, _resume_index, _resume_count,
                                                     _host_offset[i], count, args...)
                                      .dispatch(count);
                    }
                }
            }
            if (!_config.gather_by_sorting) {
                stream << _resume_count.copy_to(_host_count.data()) << synchronize();
            }
        }
    }

public:
    [[nodiscard]] const Config &config() const noexcept { return _config; }

    WavefrontCoroScheduler(Device &device, const Coro &coro, const Config &config) noexcept
        : _config{config} {
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
