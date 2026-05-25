//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/coro/coro_scheduler.h>
#include <luisa/coro/coro_func.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/coro/coro_frame_soa.h>
#include <luisa/coro/coro_frame_buffer.h>

namespace luisa::compute::coroutine {

struct WavefrontCoroSchedulerConfig {
    // uint3 block_size = make_uint3(8u, 8u, 1u);
    uint thread_count = 2_M;
    bool global_memory_soa = true;
    bool gather_by_sorting = true;// use sort for coro token gathering
    bool frame_buffer_compaction = true;
    uint hint_range = 0xffff'ffff;
    luisa::vector<luisa::string> hint_fields;
};

template<typename... Args>
class WavefrontCoroScheduler : public CoroScheduler<Args...> {

public:
    using Config = WavefrontCoroSchedulerConfig;

private:
    Config _config;
    using ArgPack = std::tuple<compute::detail::prototype_to_shader_invocation_t<Args>...>;
    luisa::optional<ArgPack> _args;
    SOA<CoroFrame> _frame_soa;
    Buffer<CoroFrame> _frame_buffer;
    Shader1D<Buffer<uint>, Buffer<uint>, uint3, uint, uint, uint, Args...> _gen_shader;
    luisa::vector<Shader1D<Buffer<uint>, Buffer<uint>, uint, Args...>> _resume_shaders;
    Shader1D<Buffer<uint>, Buffer<uint>, uint> _count_prefix_shader;
    Shader1D<Buffer<uint>, Buffer<uint>, uint> _gather_shader;
    Shader1D<Buffer<uint>, uint> _initialize_shader;
    Shader1D<Buffer<uint>, uint, uint> _compact_shader;
    Shader1D<Buffer<uint>, uint> _clear_shader;
    Buffer<uint> _resume_index;
    Buffer<uint> _resume_count;
    ///offset calculate from count, will be end after gathering
    Buffer<uint> _resume_offset;
    Buffer<uint> _global_buffer;
    luisa::vector<uint> _host_count;
    luisa::vector<uint> _host_offset;
    uint3 _dispatch_shape;
    bool _host_empty;
    uint _dispatch_counter;
    uint _max_sub_coro;
    uint _dispatch_size;
    radix_sort::temp_storage _sort_temp_storage;
    radix_sort::instance<> _sort_token;
    radix_sort::instance<Buffer<uint>> _sort_hint;
    luisa::vector<bool> _have_hint;
    Buffer<uint> _temp_key[2];
    Buffer<uint> _temp_index;

private:
    void _dispatch(Stream &stream, uint3 dispatch_size,
                   compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        _args.emplace(std::forward<compute::detail::prototype_to_shader_invocation_t<Args>>(args)...);
        _dispatch_shape = dispatch_size;
        _dispatch_size = dispatch_size.x * dispatch_size.y * dispatch_size.z;// TODO
        _dispatch_counter = 0;
        _host_empty = true;
        for (auto i = 0u; i < _max_sub_coro; i++) {
            if (i) {
                _host_count[i] = 0;
                _host_offset[i] = _config.thread_count;
            } else {
                _host_count[i] = _config.thread_count;
                _host_offset[i] = 0;
            }
        }
        stream << _initialize_shader(_resume_count, _config.thread_count).dispatch(_config.thread_count);
        this->_await_all(stream);
    }

    template<typename ShaderTytpe, typename... PrefixArgsType>
    [[nodiscard]] auto _invoke(ShaderTytpe &shader, PrefixArgsType &&...prefix_args) const noexcept {
        return std::apply(
            [&]<typename... PostfixArgsType>(PostfixArgsType &&...postfix_args) {
                return shader(std::forward<PrefixArgsType>(prefix_args)...,
                              std::forward<PostfixArgsType>(postfix_args)...);
            },
            _args.value());
    }

    void _create_shader(Device &device, const Coroutine<void(Args...)> &coroutine,
                        const WavefrontCoroSchedulerConfig &config) noexcept {
        _config = config;
        if (_config.global_memory_soa) {
            _frame_soa = SOA<CoroFrame>{device, coroutine.shared_frame(), _config.thread_count};
        } else {
            _frame_buffer = device.create<Buffer<CoroFrame>>(coroutine.shared_frame(), _config.thread_count);
        }
        bool use_sort = _config.gather_by_sorting || !_config.hint_fields.empty();
        _max_sub_coro = coroutine.subroutine_count();
        _resume_index = device.create_buffer<uint>(_config.thread_count);
        if (use_sort) {
            _temp_index = device.create_buffer<uint>(_config.thread_count);
            _temp_key[0] = device.create_buffer<uint>(_config.thread_count);
            _temp_key[1] = device.create_buffer<uint>(_config.thread_count);
        }
        _resume_count = device.create_buffer<uint>(_max_sub_coro);
        _resume_offset = device.create_buffer<uint>(_max_sub_coro);
        _global_buffer = device.create_buffer<uint>(1);
        _host_empty = true;
        _dispatch_counter = 0;
        _host_offset.resize(_max_sub_coro);
        _host_count.resize(_max_sub_coro);
        _have_hint.resize(_max_sub_coro, false);
        for (auto &token : _config.hint_fields) {
            auto id = coroutine.graph()->named_tokens().find(token);
            if (id != coroutine.graph()->named_tokens().end()) {
                LUISA_ASSERT(id->second < _max_sub_coro,
                             "coroutine token {} of id {} out of range {}", token, id->second, _max_sub_coro);
                _have_hint[id->second] = true;
            } else {
                LUISA_WARNING("coroutine token {} not found, hint disabled", token);
            }
        }
        for (auto i = 0u; i < _max_sub_coro; i++) {
            if (i) {
                _host_count[i] = 0;
                _host_offset[i] = _config.thread_count;
            } else {
                _host_count[i] = _config.thread_count;
                _host_offset[i] = 0;
            }
        }
        Callable get_coro_token = [&](UInt index) {
            $if (index > _config.thread_count) {
                device_log("Index out of range {}/{}", index, _config.thread_count);
            };
            if (_config.global_memory_soa) {
                return _frame_soa->read_field<uint>(index, "target_token") & coro_token_valid_mask;
            } else {
                CoroFrame frame = _frame_buffer->read(index);
                return frame.target_token & coro_token_valid_mask;
            }
        };
        Callable identical = [](UInt index) {
            return index;
        };

        Callable keep_index = [](UInt index, BufferUInt val) {
            return val.read(index);
        };
        Callable get_coro_hint = [&](UInt index, BufferUInt val) {
            if (!_config.hint_fields.empty()) {
                auto id = keep_index(index, val);
                if (_config.global_memory_soa) {
                    return _frame_soa->read_field<uint>(id, "coro_hint");
                } else {
                    CoroFrame frame = _frame_buffer->read(id);
                    return frame.get<uint>("coro_hint");
                }
            }
            return def<uint>(0u);
        };
        if (use_sort) {
            _sort_temp_storage = radix_sort::temp_storage(
                device, _config.thread_count, std::max(std::min(_config.hint_range, 128u), _max_sub_coro));
        }
        if (_config.gather_by_sorting) {
            _sort_token = radix_sort::instance<>(
                device, _config.thread_count, _sort_temp_storage, &get_coro_token, &identical,
                &get_coro_token, 1, _max_sub_coro);
        }
        if (!_config.hint_fields.empty()) {
            if (_config.hint_range <= 128) {
                _sort_hint = radix_sort::instance<Buffer<uint>>(
                    device, _config.thread_count, _sort_temp_storage, &get_coro_hint, &keep_index,
                    &get_coro_hint, 1, _config.hint_range);
            } else {
                auto highbit = 0;
                while ((_config.hint_range >> highbit) != 1) {
                    highbit++;
                }
                _sort_hint = radix_sort::instance<Buffer<uint>>(
                    device, _config.thread_count, _sort_temp_storage, &get_coro_hint, &keep_index,
                    &get_coro_hint, 0, 128, 0, highbit);
            }
        }
        Kernel1D gen_kernel = [&](BufferUInt index, BufferUInt count, UInt3 dispatch_shape, UInt offset, UInt st_task_id, UInt n, Var<Args>... args) {
            auto x = dispatch_x();
            $if (x >= n) {
                $return();
            };
            UInt frame_id;
            if (!_config.frame_buffer_compaction) {
                frame_id = index->read(x);
            } else {
                frame_id = offset + x;
            }
            if (!_config.gather_by_sorting) {
                count.atomic(0u).fetch_add(-1u);
            }
            auto global_id = st_task_id + x;
            auto image_size = dispatch_shape.x * dispatch_shape.y;
            auto global_id_z = global_id / image_size;
            auto global_id_xy = global_id % image_size;
            auto global_id_x = global_id_xy % dispatch_shape.x;
            auto global_id_y = global_id_xy / dispatch_shape.x;
            CoroFrame frame = coroutine.instantiate(make_uint3(global_id_x, global_id_y, global_id_z));
            coroutine.entry()(frame, args...);
            if (_config.global_memory_soa) {
                _frame_soa->write(frame_id, frame, coroutine.graph()->node(0u).output_fields());
            } else {
                _frame_buffer->write(frame_id, frame);
            }
            if (!_config.gather_by_sorting) {
                auto nxt = frame.target_token & coro_token_valid_mask;
                count.atomic(nxt).fetch_add(1u);
            }
        };
        ShaderOption o{};
        _gen_shader = device.compile(gen_kernel, o);
        _gen_shader.set_name("gen");
        _resume_shaders.resize(_max_sub_coro);

        for (auto i = 1u; i < _max_sub_coro; ++i) {
            Kernel1D resume_kernel = [&](BufferUInt index, BufferUInt count, UInt n, Var<Args>... args) {
                auto x = dispatch_x();
                $if (x >= n) {
                    $return();
                };
                auto frame_id = index.read(x);
                CoroFrame frame = coroutine.instantiate();
                if (_config.global_memory_soa) {
                    //frame = frame_buffer.read(frame_id);
                    frame = _frame_soa->read(frame_id, coroutine.graph()->node(i).input_fields());
                } else {
                    frame = _frame_buffer->read(frame_id);
                }
                if (!_config.gather_by_sorting) {
                    count.atomic(i).fetch_sub(1u);
                }
                coroutine[i](frame, args...);
                if (_config.global_memory_soa) {
                    _frame_soa->write(frame_id, frame, coroutine.graph()->node(i).output_fields());
                } else {
                    _frame_buffer->write(frame_id, frame);
                }

                if (!_config.gather_by_sorting) {
                    auto nxt = frame.target_token & coro_token_valid_mask;
                    $if (nxt < _max_sub_coro) {
                        count.atomic(nxt).fetch_add(1u);
                    };
                }
            };
            _resume_shaders[i] = device.compile(resume_kernel, o);
            _resume_shaders[i].set_name("resume" + std::to_string(i));
        }

        Kernel1D _prefix_kernel = [&](BufferUInt count, BufferUInt prefix, UInt n) {
            $if (dispatch_x() == 0) {
                auto pre = def(0u);
                for (auto i = 0u; i < _max_sub_coro; ++i) {
                    auto val = count.read(i);
                    prefix.write(i, pre);
                    pre = pre + val;
                }
            };
        };
        _count_prefix_shader = device.compile(_prefix_kernel);

        Kernel1D _gather_kernel = [&](BufferUInt index, BufferUInt prefix, UInt n) {
            auto x = dispatch_x();
            UInt r_id;
            if (_config.global_memory_soa) {
                r_id = _frame_soa->read_field<uint>(x, "target_token") & coro_token_valid_mask;
            } else {
                auto frame = _frame_buffer->read(x);
                r_id = frame.target_token & coro_token_valid_mask;
            }
            auto q_id = prefix.atomic(r_id).fetch_add(1u);
            index.write(q_id, x);
        };
        _gather_shader = device.compile(_gather_kernel);

        Kernel1D _compact_kernel_2 = [&](BufferUInt index, UInt empty_offset, UInt n) {
            //_global_buffer->write(0u, 0u);
            auto x = dispatch_x();
            $if (empty_offset + x < n) {
                UInt token;
                if (_config.global_memory_soa) {
                    token = _frame_soa->read_field<uint>(empty_offset + x, "target_token");
                } else {
                    CoroFrame frame = _frame_buffer->read(empty_offset + x);
                    token = frame.target_token;
                }
                $if ((token & coro_token_valid_mask) != 0u) {
                    auto res = _global_buffer->atomic(0u).fetch_add(1u);
                    auto slot = index.read(res);
                    if (!_config.gather_by_sorting) {
                        $while (slot >= empty_offset) {
                            res = _global_buffer->atomic(0u).fetch_add(1u);
                            slot = index.read(res);
                        };
                    }
                    if (_config.global_memory_soa) {
                        auto frame = _frame_soa->read(empty_offset + x);
                        _frame_soa->write(slot, frame);
                    } else {
                        auto frame = _frame_buffer->read(empty_offset + x);
                        _frame_buffer->write(slot, frame);
                    }
                    if (_config.global_memory_soa) {
                        _frame_soa->write_field(empty_offset + x, 0u, "target_token");
                    } else {
                        CoroFrame empty_frame = coroutine.instantiate();
                        _frame_buffer->write(empty_offset + x, empty_frame);
                    }
                };
            };
        };
        _compact_shader = device.compile(_compact_kernel_2);
        _compact_shader.set_name("compact");

        Kernel1D _initialize_kernel = [&](BufferUInt count, UInt n) {
            auto x = dispatch_x();
            $if (x < n) {
                CoroFrame frame = coroutine.instantiate();
                if (_config.global_memory_soa) {
                    _frame_soa->write(x, frame, std::array{0u, 1u});
                } else {
                    _frame_buffer->write(x, frame);
                }
            };
            $if (x < _max_sub_coro) {
                count.write(x, ite(x == 0u, _config.thread_count, 0u));
            };
        };
        _initialize_shader = device.compile(_initialize_kernel);

        Kernel1D clear = [&](BufferUInt buffer, UInt n) {
            auto x = dispatch_x();
            $if (x < n) {
                buffer.write(x, 0u);
            };
        };
        _clear_shader = device.compile(clear);
    }

    [[nodiscard]] bool _all_dispatched() const noexcept {
        return _dispatch_counter == _dispatch_size;
    }
    [[nodiscard]] bool _all_done() const noexcept {
        return this->_all_dispatched() && _host_empty;
    }

    void _await_all(Stream &stream) noexcept {
        while (!this->_all_done()) {
            this->_await_step(stream);
        }
    }
    void _await_step(Stream &stream) noexcept {
        if (_config.gather_by_sorting) {
            auto host_update = [&] {
                _host_empty = true;
                for (uint i = 0u; i < _max_sub_coro; i++) {
                    _host_count[i] = (i + 1u == _max_sub_coro ? _config.thread_count : _host_offset[i + 1u]) - _host_offset[i];
                    _host_empty = _host_empty && (i == 0u || _host_count[i] == 0u);
                }
            };
            _sort_token.sort(stream, _temp_key[0], _resume_index, _temp_key[1],
                             _resume_index, _config.thread_count);

            stream << _sort_temp_storage.hist_buffer.view(0u, _max_sub_coro).copy_to(_host_offset.data())
                   << host_update
                   << synchronize();

            if (_host_count[0] > _config.thread_count * 0.5f && !this->_all_dispatched()) {
                auto gen_count = std::min(_dispatch_size - _dispatch_counter, _host_count[0]);
                if (_host_count[0] != _config.thread_count && _config.frame_buffer_compaction) {
                    stream << _clear_shader(_global_buffer, 1).dispatch(1u);
                    stream << _compact_shader(_resume_index, _config.thread_count - _host_count[0], _config.thread_count).dispatch(_host_count[0]);
                }
                stream << _invoke(_gen_shader, _resume_index.view(_host_offset[0], _host_count[0]),
                                  _resume_count, _dispatch_shape, _config.thread_count - _host_count[0], _dispatch_counter,
                                  _config.thread_count)
                              .dispatch(gen_count);
                _dispatch_counter += gen_count;
                _host_empty = false;
            } else {
                for (uint i = 1; i < _max_sub_coro; i++) {
                    if (_host_count[i] > 0) {
                        if (_have_hint[i]) {
                            BufferView<uint> _index[2] = {_resume_index.view(_host_offset[i], _host_count[i]), _temp_index.view(_host_offset[i], _host_count[i])};
                            BufferView<uint> _key[2] = {_temp_key[1].view(_host_offset[i], _host_count[i]), _temp_key[0].view(_host_offset[i], _host_count[i])};
                            uint out = _sort_hint.sort_switch(stream, _key, _index, _host_count[i], _resume_index.view(_host_offset[i], _host_count[i]));
                            stream << _invoke(_resume_shaders[i], _index[out], _resume_count, _config.thread_count)
                                          .dispatch(_host_count[i]);
                        } else {
                            stream << _invoke(_resume_shaders[i], _resume_index.view(_host_offset[i], _host_count[i]),
                                              _resume_count, _config.thread_count)
                                          .dispatch(_host_count[i]);
                        }
                    }
                }
            }
            stream << synchronize();
        } else {
            stream << _count_prefix_shader(_resume_count, _resume_offset, _max_sub_coro).dispatch(1u);
            stream << _gather_shader(_resume_index, _resume_offset, _config.thread_count).dispatch(_config.thread_count);
            if (_host_count[0] > _config.thread_count / 2 && !_all_dispatched()) {
                auto gen_count = std::min(_dispatch_size - _dispatch_counter, _host_count[0]);
                if (_host_count[0] != _config.thread_count && _config.frame_buffer_compaction) {
                    stream << _clear_shader(_global_buffer, 1).dispatch(1u);
                    stream
                        << _compact_shader(_resume_index.view(_host_offset[0], _host_count[0]),
                                           _config.thread_count - _host_count[0], _config.thread_count)
                               .dispatch(_host_count[0]);
                }
                stream << _invoke(_gen_shader, _resume_index.view(_host_offset[0], _host_count[0]),
                                  _resume_count, _dispatch_shape, _config.thread_count - _host_count[0], _dispatch_counter, _config.thread_count)
                              .dispatch(gen_count);
                _dispatch_counter += gen_count;
                _host_empty = false;
            } else {
                for (uint i = 1; i < _max_sub_coro; i++) {
                    if (_host_count[i] > 0) {
                        if (_have_hint[i]) {
                            BufferView<uint> _index[2] = {_resume_index.view(_host_offset[i], _host_count[i]), _temp_index.view(_host_offset[i], _host_count[i])};
                            BufferView<uint> _key[2] = {_temp_key[0].view(_host_offset[i], _host_count[i]), _temp_key[1].view(_host_offset[i], _host_count[i])};
                            uint out = _sort_hint.sort_switch(stream, _key, _index, _host_count[i], _resume_index.view(_host_offset[i], _host_count[i]));
                            stream << _invoke(_resume_shaders[i], _index[out], _resume_count, _config.thread_count)
                                          .dispatch(_host_count[i]);
                        } else {
                            stream << _invoke(_resume_shaders[i], _resume_index.view(_host_offset[i], _host_count[i]),
                                              _resume_count, _config.thread_count)
                                          .dispatch(_host_count[i]);
                        }
                    }
                }
            }
            auto host_update = [&] {
                _host_empty = true;
                auto sum = 0u;
                for (uint i = 0; i < _max_sub_coro; i++) {
                    _host_offset[i] = sum;
                    sum += _host_count[i];
                    _host_empty = _host_empty && (i == 0 || _host_count[i] == 0);
                }
            };
            stream << _resume_count.view(0, _max_sub_coro).copy_to(_host_count.data())
                   << host_update;
            stream << synchronize();
        }
    }

public:
    WavefrontCoroScheduler(Device &device, const Coroutine<void(Args...)> &coro,
                           const WavefrontCoroSchedulerConfig &config) noexcept {
        _create_shader(device, coro, config);
    }
    WavefrontCoroScheduler(Device &device, const Coroutine<void(Args...)> &coro) noexcept
        : WavefrontCoroScheduler{device, coro, WavefrontCoroSchedulerConfig{}} {}
};

template<typename... Args>
WavefrontCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> WavefrontCoroScheduler<Args...>;

template<typename... Args>
WavefrontCoroScheduler(Device &, const Coroutine<void(Args...)> &,
                       const WavefrontCoroSchedulerConfig &)
    -> WavefrontCoroScheduler<Args...>;

}// namespace luisa::compute::coroutine
