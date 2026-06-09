#pragma once

#include <luisa/ast/function_builder.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

/// Frame memory layout used by the wavefront scheduler.
enum class FrameLayout : uint8_t {
    /// Array-of-Structures: all frame fields interleaved in one buffer.
    AoS = 0,
    /// Structure-of-Arrays: each frame field gets its own buffer.
    SoA = 1,
};

template<typename... Args>
class WavefrontCoroScheduler : public CoroScheduler<Args...> {

    using Coro = Coroutine<void(Args...)>;
    using Base = CoroScheduler<Args...>;
    using AoSSubKernel = Shader<1, Buffer<uint>, uint, Args...>;
    // SoA with 0 user fields: token + skip only
    using SoASubKernelSimple = Shader<1, Buffer<uint>, Buffer<uint>, uint, Args...>;

    const Coro &_coro;
    Device &_device;
    FrameLayout _layout{FrameLayout::AoS};

    // --- AoS data ---
    Buffer<uint> _frame_buffer;
    luisa::vector<AoSSubKernel> _kernels;
    uint _uints_per_frame{2u};
    uint _current_capacity{0u};

    // --- SoA data (0 user fields: token + skip) ---
    Buffer<uint> _token_buffer;
    Buffer<uint> _skip_buffer;
    luisa::vector<SoASubKernelSimple> _soa_kernels_simple;
    uint _soa_capacity{0u};

    // --- SoA data (general: >0 user fields) ---
    // Uses a single Buffer<uint> with SoA layout internally
    // (same kernel type as AoS for simplicity).
    Buffer<uint> _soa_frame_buffer;
    luisa::vector<AoSSubKernel> _soa_kernels;
    luisa::vector<uint> _soa_field_uints;
    uint _soa_stride{0u};
    uint _soa_capacity_general{0u};

public:
    /// Construct a WavefrontCoroScheduler.
    /// @param device  Device used to compile shaders and create buffers.
    /// @param coro    The coroutine to schedule.
    /// @param layout  Frame memory layout (default: AoS).
    WavefrontCoroScheduler(Device &device, const Coro &coro,
                           FrameLayout layout = FrameLayout::AoS) noexcept
        : Base{coro.graph(), coro.frame_desc()},
          _coro{coro},
          _device{device},
          _layout{layout} {
        _uints_per_frame += static_cast<uint>(
            coro.frame_desc().total_size() / sizeof(uint));
        if (_layout == FrameLayout::AoS) {
            _compile_kernels(device);
        } else {
            _compile_kernels_soa(device);
        }
    }

    WavefrontCoroScheduler(const WavefrontCoroScheduler &) = delete;
    WavefrontCoroScheduler &operator=(const WavefrontCoroScheduler &) = delete;
    WavefrontCoroScheduler(WavefrontCoroScheduler &&) = delete;
    WavefrontCoroScheduler &operator=(WavefrontCoroScheduler &&) = delete;

    [[nodiscard]] auto layout() const noexcept { return _layout; }

    void _dispatch(Stream &stream, uint3 dispatch_size,
                   const Args &...args) noexcept override {
        uint N = dispatch_size.x * dispatch_size.y * dispatch_size.z;

        if (_layout == FrameLayout::AoS) {
            _dispatch_aos(stream, N, args...);
        } else {
            _dispatch_soa(stream, N, args...);
        }
    }

    [[nodiscard]] const Coro &coroutine() const noexcept { return _coro; }

private:
    // ===================================================================
    // AoS dispatch
    // ===================================================================
    void _dispatch_aos(Stream &stream, uint N, const Args &...args) noexcept {
        uint needed = N * _uints_per_frame;
        if (_current_capacity < needed) {
            _frame_buffer = _device.create_buffer<uint>(needed);
            _current_capacity = needed;
        }

        luisa::vector<uint> zeros(needed, 0u);
        stream << _frame_buffer.copy_from(zeros.data()) << synchronize();

        stream << _kernels[0u](_frame_buffer, N, args...).dispatch(N)
               << synchronize();

        size_t max_iters = this->graph().node_count();
        for (size_t iter = 0u; iter < max_iters; ++iter) {
            for (size_t i = 1u; i < this->graph().node_count(); ++i) {
                stream << _kernels[i](_frame_buffer, N, args...).dispatch(N);
            }
            stream << synchronize();
        }
    }

    // ===================================================================
    // SoA dispatch (0 user fields: separate token + skip buffers)
    // ===================================================================
    void _dispatch_soa_simple(Stream &stream, uint N, const Args &...args) noexcept {
        uint needed = N;
        if (_soa_capacity < needed) {
            _token_buffer = _device.create_buffer<uint>(N);
            _skip_buffer = _device.create_buffer<uint>(N);
            _soa_capacity = needed;
        }

        luisa::vector<uint> zeros(N, 0u);
        stream << _token_buffer.copy_from(zeros.data())
               << _skip_buffer.copy_from(zeros.data())
               << synchronize();

        stream << _soa_kernels_simple[0u](_token_buffer, _skip_buffer, N, args...)
                      .dispatch(N)
               << synchronize();

        size_t max_iters = this->graph().node_count();
        for (size_t iter = 0u; iter < max_iters; ++iter) {
            for (size_t i = 1u; i < this->graph().node_count(); ++i) {
                stream << _soa_kernels_simple[i](_token_buffer, _skip_buffer, N, args...)
                              .dispatch(N);
            }
            stream << synchronize();
        }
    }

    // ===================================================================
    // SoA dispatch (general: >0 user fields, flat buffer SoA layout)
    // ===================================================================
    void _dispatch_soa_general(Stream &stream, uint N, const Args &...args) noexcept {
        uint needed = N * _soa_stride;
        if (_soa_capacity_general < needed) {
            _soa_frame_buffer = _device.create_buffer<uint>(needed);
            _soa_capacity_general = needed;
        }

        luisa::vector<uint> zeros(needed, 0u);
        stream << _soa_frame_buffer.copy_from(zeros.data()) << synchronize();

        stream << _soa_kernels[0u](_soa_frame_buffer, N, args...).dispatch(N)
               << synchronize();

        size_t max_iters = this->graph().node_count();
        for (size_t iter = 0u; iter < max_iters; ++iter) {
            for (size_t i = 1u; i < this->graph().node_count(); ++i) {
                stream << _soa_kernels[i](_soa_frame_buffer, N, args...).dispatch(N);
            }
            stream << synchronize();
        }
    }

    void _dispatch_soa(Stream &stream, uint N, const Args &...args) noexcept {
        if (_soa_field_uints.empty()) {
            _dispatch_soa_simple(stream, N, args...);
        } else {
            _dispatch_soa_general(stream, N, args...);
        }
    }

    // ===================================================================
    // Kernel compilation — AoS
    // ===================================================================
    void _compile_kernels(Device &device) noexcept {
        const auto *frame_desc = &_coro.frame_desc();
        size_t nc = this->graph().node_count();
        uint uints_per = _uints_per_frame;

        if (auto entry_sub = _coro[0u]) {
            Kernel1D k_entry = [entry_sub, frame_desc, uints_per](
                                   BufferUInt frame_buf, UInt N,
                                   Var<std::remove_cvref_t<Args>>... k_args) noexcept {
                auto idx = dispatch_x();
                $if(idx >= N) { $return(); };
                auto frame = CoroFrame::create(frame_desc);
                frame.coro_id = make_uint3(idx, 0u, 0u);
                {
                    const Expression *call_args[1u + sizeof...(Args)];
                    call_args[0] = frame.expression();
                    size_t ai = 1u;
                    ((call_args[ai++] = detail::extract_expression(k_args)), ...);
                    detail::FunctionBuilder::current()->call(
                        entry_sub->function(),
                        luisa::span<const Expression *const>{
                            call_args, 1u + sizeof...(Args)});
                }
                auto base = idx * uints_per;
                frame_buf.write(base + 0u, frame.target_token);
            };
            _kernels.emplace_back(device.compile(k_entry));
        } else {
            _kernels.emplace_back();
        }

        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = _coro[i];
            if (!cont_sub) { _kernels.emplace_back(); continue; }
            uint my_token = static_cast<uint>(this->graph().node(i).token);
            Kernel1D k_cont = [cont_sub, frame_desc, uints_per, my_token](
                                  BufferUInt frame_buf, UInt N,
                                  Var<std::remove_cvref_t<Args>>... k_args) noexcept {
                auto idx = dispatch_x();
                $if(idx >= N) { $return(); };
                auto base = idx * uints_per;
                auto tok = frame_buf.read(base + 0u);
                $if(tok != my_token) { $return(); };
                auto frame = CoroFrame::create(frame_desc);
                frame.coro_id = make_uint3(idx, 0u, 0u);
                frame.target_token = tok;
                {
                    const Expression *call_args[1u + sizeof...(Args)];
                    call_args[0] = frame.expression();
                    size_t ai = 1u;
                    ((call_args[ai++] = detail::extract_expression(k_args)), ...);
                    detail::FunctionBuilder::current()->call(
                        cont_sub->function(),
                        luisa::span<const Expression *const>{
                            call_args, 1u + sizeof...(Args)});
                }
                frame_buf.write(base + 0u, frame.target_token);
            };
            _kernels.emplace_back(device.compile(k_cont));
        }
    }

    // ===================================================================
    // Kernel compilation — SoA
    // ===================================================================
    void _compile_kernels_soa(Device &device) noexcept {
        const auto *frame_desc = &_coro.frame_desc();
        size_t n_fields = frame_desc->field_count();

        // Build field uint sizes
        _soa_field_uints.clear();
        _soa_field_uints.reserve(n_fields);
        for (size_t i = 0u; i < n_fields; ++i) {
            auto &f = frame_desc->field(i);
            _soa_field_uints.push_back(
                static_cast<uint>((f.size + sizeof(uint) - 1u) / sizeof(uint)));
        }

        if (n_fields == 0u) {
            _compile_kernels_soa_simple(device, frame_desc);
        } else {
            _compile_kernels_soa_general(device, frame_desc);
        }
    }

    // SoA with 0 user fields: each subkernel takes token_buf + skip_buf only.
    void _compile_kernels_soa_simple(Device &device,
                                     const CoroFrameDesc *frame_desc) noexcept {
        size_t nc = this->graph().node_count();

        if (auto entry_sub = _coro[0u]) {
            Kernel1D k_entry = [entry_sub, frame_desc](
                                   BufferUInt token_buf, BufferUInt skip_buf,
                                   UInt N,
                                   Var<std::remove_cvref_t<Args>>... k_args) noexcept {
                auto idx = dispatch_x();
                $if(idx >= N) { $return(); };
                auto frame = CoroFrame::create(frame_desc);
                frame.coro_id = make_uint3(idx, 0u, 0u);
                {
                    const Expression *call_args[1u + sizeof...(Args)];
                    call_args[0] = frame.expression();
                    size_t ai = 1u;
                    ((call_args[ai++] = detail::extract_expression(k_args)), ...);
                    detail::FunctionBuilder::current()->call(
                        entry_sub->function(),
                        luisa::span<const Expression *const>{
                            call_args, 1u + sizeof...(Args)});
                }
                token_buf.write(idx, frame.target_token);
            };
            _soa_kernels_simple.emplace_back(device.compile(k_entry));
        } else {
            _soa_kernels_simple.emplace_back();
        }

        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = _coro[i];
            if (!cont_sub) { _soa_kernels_simple.emplace_back(); continue; }
            uint my_token = static_cast<uint>(this->graph().node(i).token);
            Kernel1D k_cont = [cont_sub, frame_desc, my_token](
                                  BufferUInt token_buf, BufferUInt skip_buf,
                                  UInt N,
                                  Var<std::remove_cvref_t<Args>>... k_args) noexcept {
                auto idx = dispatch_x();
                $if(idx >= N) { $return(); };
                auto tok = token_buf.read(idx);
                $if(tok != my_token) { $return(); };
                auto frame = CoroFrame::create(frame_desc);
                frame.coro_id = make_uint3(idx, 0u, 0u);
                frame.target_token = tok;
                {
                    const Expression *call_args[1u + sizeof...(Args)];
                    call_args[0] = frame.expression();
                    size_t ai = 1u;
                    ((call_args[ai++] = detail::extract_expression(k_args)), ...);
                    detail::FunctionBuilder::current()->call(
                        cont_sub->function(),
                        luisa::span<const Expression *const>{
                            call_args, 1u + sizeof...(Args)});
                }
                token_buf.write(idx, frame.target_token);
            };
            _soa_kernels_simple.emplace_back(device.compile(k_cont));
        }
    }

    // SoA with >0 user fields: single Buffer<uint> with SoA layout.
    // The kernel computes per-field offsets from the SoA stride.
    void _compile_kernels_soa_general(Device &device,
                                      const CoroFrameDesc *frame_desc) noexcept {
        size_t nc = this->graph().node_count();
        size_t n_fields = _soa_field_uints.size();

        // Compute SoA stride: token(1) + skip(1) + sum of field uints
        uint soa_stride = 2u;
        for (auto u : _soa_field_uints) { soa_stride += u; }
        _soa_stride = soa_stride;

        // Pre-compute per-field SoA offsets (in uints within a stripe)
        luisa::vector<uint> soa_offsets;
        soa_offsets.reserve(n_fields + 2u);
        soa_offsets.push_back(0u);// token offset
        soa_offsets.push_back(1u);// skip offset
        uint accum = 2u;
        for (auto u : _soa_field_uints) {
            soa_offsets.push_back(accum);
            accum += u;
        }

        auto make_entry_kernel =
            [](auto entry_sub, const CoroFrameDesc *fd,
               luisa::vector<uint> offsets, uint stride, size_t nf) {
                return Kernel1D{
                    [entry_sub, fd,
                     offsets = std::move(offsets), stride, nf](
                        BufferUInt frame_buf, UInt N,
                        Var<std::remove_cvref_t<Args>>... k_args) noexcept {
                        auto idx = dispatch_x();
                        $if(idx >= N) { $return(); };
                        auto base = idx * stride;
                        auto frame = CoroFrame::create(fd);
                        frame.coro_id = make_uint3(idx, 0u, 0u);
                        {
                            const Expression *call_args[1u + sizeof...(Args)];
                            call_args[0] = frame.expression();
                            size_t ai = 1u;
                            ((call_args[ai++] =
                                  detail::extract_expression(k_args)),
                             ...);
                            detail::FunctionBuilder::current()->call(
                                entry_sub->function(),
                                luisa::span<const Expression *const>{
                                    call_args, 1u + sizeof...(Args)});
                        }
                        // Write token back (offset 0 in SoA stripe)
                        frame_buf.write(base + offsets[0u], frame.target_token);
                    }};
            };

        auto make_cont_kernel =
            [](auto cont_sub, const CoroFrameDesc *fd,
               luisa::vector<uint> offsets, uint stride,
               size_t nf, uint my_token) {
                return Kernel1D{
                    [cont_sub, fd,
                     offsets = std::move(offsets), stride, nf, my_token](
                        BufferUInt frame_buf, UInt N,
                        Var<std::remove_cvref_t<Args>>... k_args) noexcept {
                        auto idx = dispatch_x();
                        $if(idx >= N) { $return(); };
                        auto base = idx * stride;
                        auto tok = frame_buf.read(base + offsets[0u]);
                        $if(tok != my_token) { $return(); };
                        auto frame = CoroFrame::create(fd);
                        frame.coro_id = make_uint3(idx, 0u, 0u);
                        frame.target_token = tok;
                        {
                            const Expression *call_args[1u + sizeof...(Args)];
                            call_args[0] = frame.expression();
                            size_t ai = 1u;
                            ((call_args[ai++] =
                                  detail::extract_expression(k_args)),
                             ...);
                            detail::FunctionBuilder::current()->call(
                                cont_sub->function(),
                                luisa::span<const Expression *const>{
                                    call_args, 1u + sizeof...(Args)});
                        }
                        // Write token back
                        frame_buf.write(base + offsets[0u], frame.target_token);
                    }};
            };

        if (auto entry_sub = _coro[0u]) {
            _soa_kernels.emplace_back(
                device.compile(make_entry_kernel(
                    entry_sub, frame_desc, soa_offsets, soa_stride, n_fields)));
        } else {
            _soa_kernels.emplace_back();
        }

        for (size_t i = 1u; i < nc; ++i) {
            auto cont_sub = _coro[i];
            if (!cont_sub) { _soa_kernels.emplace_back(); continue; }
            uint my_token = static_cast<uint>(this->graph().node(i).token);
            _soa_kernels.emplace_back(
                device.compile(make_cont_kernel(
                    cont_sub, frame_desc, soa_offsets, soa_stride,
                    n_fields, my_token)));
        }
    }
};

template<typename... Args>
WavefrontCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> WavefrontCoroScheduler<Args...>;

}// namespace luisa::compute::coro
