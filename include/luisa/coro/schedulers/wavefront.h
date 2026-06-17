//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute::coro {

struct WavefrontCoroSchedulerConfig {
    uint thread_count = 131072u;// 128K threads
    bool global_memory_soa = true;
    bool gather_by_sorting = true;
    bool frame_buffer_compaction = true;
};

template<typename... Args>
class WavefrontCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = WavefrontCoroSchedulerConfig;

private:
    Config _config;
    ByteBuffer _frame_buffer;
    Shader3D<ByteBuffer, uint, Args...> _entry_kernel;
    luisa::vector<Shader1D<ByteBuffer, Buffer<uint>, uint, uint, Args...>> _resume_kernels;
    Shader<1, ByteBuffer, uint> _clear_shader;
    Shader1D<Buffer<uint>, uint> _clear_count_shader;
    Shader1D<ByteBuffer, Buffer<uint>, uint> _count_shader;
    Shader1D<ByteBuffer, Buffer<uint>, Buffer<uint>, uint> _gather_shader;
    Buffer<uint> _resume_index;
    Buffer<uint> _resume_count;
    Buffer<uint> _resume_offset;
    luisa::vector<uint> _host_count;
    luisa::vector<uint> _host_offset;
    luisa::vector<size_t> _frame_member_offsets;
    size_t _frame_stride{0u};

private:
    [[nodiscard]] static auto _frame_type(const CoroFrameDesc &desc) noexcept {
        luisa::vector<const Type *> members;
        members.reserve(desc.field_count() + 3u);
        members.emplace_back(Type::of<uint3>());
        members.emplace_back(Type::of<uint>());
        members.emplace_back(Type::of<uint>());
        for (auto i = 0u; i < desc.field_count(); i++) {
            members.emplace_back(desc.field(i).type);
        }
        return Type::structure(members);
    }

    [[nodiscard]] static auto _linear_dispatch_index() noexcept {
        auto id = dispatch_id();
        auto size = dispatch_size();
        return id.x + id.y * size.x + id.z * size.x * size.y;
    }

    static void _store_frame(const Var<ByteBuffer> &frame_buf, UInt base,
                             const CoroFrame &frame, luisa::span<const size_t> offsets) noexcept {
        frame_buf.write(base + static_cast<uint>(offsets[0u]), frame.coro_id);
        frame_buf.write(base + static_cast<uint>(offsets[1u]), frame.target_token);
        frame_buf.write(base + static_cast<uint>(offsets[2u]), frame.skip_flag);
        auto *fb = luisa::compute::detail::FunctionBuilder::current();
        for (auto i = 0u; i < frame.desc()->field_count(); i++) {
            auto *type = frame.desc()->field(i).type;
            auto *member = fb->member(type, frame.expression(), i + 3u);
            fb->call(CallOp::BYTE_BUFFER_WRITE,
                     {frame_buf.expression(),
                      luisa::compute::detail::extract_expression(base + static_cast<uint>(offsets[i + 3u])),
                      member});
        }
    }

    [[nodiscard]] static auto _load_frame(const Var<ByteBuffer> &frame_buf, UInt base,
                                          const Coro &coro,
                                          luisa::span<const size_t> offsets) noexcept {
        auto frame = coro.instantiate(frame_buf.read<uint3>(base + static_cast<uint>(offsets[0u])));
        auto *fb = luisa::compute::detail::FunctionBuilder::current();
        frame.target_token = frame_buf.read<uint>(base + static_cast<uint>(offsets[1u]));
        frame.skip_flag = frame_buf.read<uint>(base + static_cast<uint>(offsets[2u]));
        for (auto i = 0u; i < frame.desc()->field_count(); i++) {
            auto *type = frame.desc()->field(i).type;
            auto *value = fb->call(type, CallOp::BYTE_BUFFER_READ,
                                   {frame_buf.expression(),
                                    luisa::compute::detail::extract_expression(base + static_cast<uint>(offsets[i + 3u]))});
            auto *member = fb->member(type, frame.expression(), i + 3u);
            fb->assign(member, value);
        }
        return frame;
    }

    void _create_shader(Device &device, const Coro &coro) {
        auto *frame_type = _frame_type(coro.frame());
        _frame_stride = frame_type->size();
        _frame_member_offsets.clear();
        _frame_member_offsets.reserve(frame_type->members().size());
        size_t offset = 0u;
        for (auto *member : frame_type->members()) {
            auto alignment = member->alignment();
            offset = (offset + alignment - 1u) / alignment * alignment;
            _frame_member_offsets.emplace_back(offset);
            offset += member->size();
        }

        size_t nc = coro.subroutine_count();
        _resume_kernels.resize(nc);
        _host_count.resize(nc);
        _host_offset.resize(nc);

        if (auto entry_sub = coro[0u]) {
            Kernel3D k_entry = [&coro, offsets = _frame_member_offsets, frame_stride = static_cast<uint>(_frame_stride)](
                                   ByteBufferVar frame_buf, UInt N,
                                   Var<Args>... k_args) noexcept {
                auto idx = _linear_dispatch_index();
                $if (idx >= N) { $return(); };
                auto base = idx * frame_stride;
                auto frame = coro.instantiate(dispatch_id());
                frame.target_token = 0u;
                frame.skip_flag = 0u;
                coro.entry()(frame, k_args...);
                _store_frame(frame_buf, base, frame, offsets);
            };
            _entry_kernel = device.compile(k_entry);
        }

        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = coro[i];
            if (!cont_sub) continue;
            uint my_token = static_cast<uint>(coro.graph().node(i).token);
            Kernel1D k_cont = [&coro, offsets = _frame_member_offsets, frame_stride = static_cast<uint>(_frame_stride), my_token, i](
                                  ByteBufferVar frame_buf, BufferUInt resume_index,
                                  UInt resume_offset, UInt count,
                                  Var<Args>... k_args) noexcept {
                auto x = dispatch_x();
                $if (x >= count) { $return(); };
                auto idx = resume_index.read(resume_offset + x);
                auto base = idx * frame_stride;
                auto tok = frame_buf.read<uint>(base + static_cast<uint>(offsets[1u]));
                $if (tok != my_token) { $return(); };
                auto frame = _load_frame(frame_buf, base, coro, offsets);
                frame.skip_flag = 0u;
                coro[i](frame, k_args...);
                _store_frame(frame_buf, base, frame, offsets);
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
            [&coro, offsets = _frame_member_offsets, frame_stride = static_cast<uint>(_frame_stride)](
                ByteBufferVar frame_buf, BufferUInt count, UInt n) noexcept {
                auto x = dispatch_x();
                $if (x >= n) { $return(); };
                auto base = x * frame_stride;
                auto tok = frame_buf.read<uint>(base + static_cast<uint>(offsets[1u]));
                $if (tok != CoroFrame::TERMINAL_TOKEN) {
                    for (size_t i = 1u; i < coro.subroutine_count(); ++i) {
                        $if (tok == coro.trigger_token(i)) {
                            count.atomic(static_cast<uint>(i)).fetch_add(1u);
                        };
                    }
                };
            });

        _gather_shader = device.compile<1>(
            [&coro, offsets = _frame_member_offsets, frame_stride = static_cast<uint>(_frame_stride)](
                ByteBufferVar frame_buf, BufferUInt index, BufferUInt offset, UInt n) noexcept {
                auto x = dispatch_x();
                $if (x >= n) { $return(); };
                auto base = x * frame_stride;
                auto tok = frame_buf.read<uint>(base + static_cast<uint>(offsets[1u]));
                $if (tok != CoroFrame::TERMINAL_TOKEN) {
                    for (size_t i = 1u; i < coro.subroutine_count(); ++i) {
                        $if (tok == coro.trigger_token(i)) {
                            auto slot = offset.atomic(static_cast<uint>(i)).fetch_add(1u);
                            index.write(slot, x);
                        };
                    }
                };
        });

        _frame_buffer = device.create_byte_buffer(
            static_cast<size_t>(_config.thread_count) * _frame_stride);
        _resume_index = device.create_buffer<uint>(_config.thread_count);
        _resume_count = device.create_buffer<uint>(nc);
        _resume_offset = device.create_buffer<uint>(nc);
    }

    void _dispatch(
        Stream &stream, uint3 dispatch_size,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept override {
        uint N = dispatch_size.x * dispatch_size.y * dispatch_size.z;
        if (!_frame_buffer || _frame_buffer.size_bytes() < static_cast<size_t>(N) * _frame_stride) {
            LUISA_ERROR(
                "WavefrontCoroScheduler dispatch size {} exceeds configured frame capacity {}. "
                "Increase WavefrontCoroSchedulerConfig::thread_count.",
                N, _config.thread_count);
            return;
        }

        auto uint_count = static_cast<uint>((static_cast<size_t>(N) * _frame_stride + sizeof(uint) - 1u) / sizeof(uint));
        stream << _clear_shader(_frame_buffer, uint_count).dispatch(uint_count);
        stream << _entry_kernel(_frame_buffer, N, args...).dispatch(dispatch_size);

        auto nc = _resume_kernels.size();
        while (true) {
            stream << _clear_count_shader(_resume_count, static_cast<uint>(nc)).dispatch(static_cast<uint>(nc));
            stream << _count_shader(_frame_buffer, _resume_count, N).dispatch(N);
            stream << _resume_count.copy_to(_host_count.data()) << synchronize();

            auto active_count = 0u;
            auto active_offset = 0u;
            for (size_t i = 0u; i < nc; ++i) {
                _host_offset[i] = active_offset;
                active_offset += _host_count[i];
                if (i != 0u) { active_count += _host_count[i]; }
            }
            if (active_count == 0u) { break; }

            stream << _resume_offset.copy_from(luisa::span{_host_offset.data(), _host_offset.size()});
            stream << _gather_shader(_frame_buffer, _resume_index, _resume_offset, N).dispatch(N);
            for (size_t i = 1u; i < nc; ++i) {
                auto count = _host_count[i];
                if (count == 0u) { continue; }
                stream << _resume_kernels[i](_frame_buffer, _resume_index, _host_offset[i], count, args...).dispatch(count);
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
