//
// Created by Mike on 7/28/2021.
//

#pragma once

#include <cuda.h>

#include <runtime/device.h>
#include <backends/cuda/cuda_error.h>

namespace luisa::compute::cuda {

class CUDADevice : public Device::Interface {

    class ContextGuard {

    private:
        CUcontext _ctx;

    public:
        explicit ContextGuard(CUcontext ctx) noexcept : _ctx{ctx} {
            LUISA_CHECK_CUDA(cuCtxPushCurrent(_ctx));
        }
        ~ContextGuard() noexcept {
            CUcontext ctx = nullptr;
            LUISA_CHECK_CUDA(cuCtxPopCurrent(&ctx));
            if (ctx != _ctx) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid CUDA context {} (expected {}).",
                    fmt::ptr(ctx), fmt::ptr(_ctx));
            }
        }
    };

public:
    class Handle {

    private:
        CUcontext _context{nullptr};
        CUdevice _device{0};
        uint32_t _compute_capability{};

    public:
        explicit Handle(uint index) noexcept;
        ~Handle() noexcept;
        Handle(Handle &&) noexcept = delete;
        Handle(const Handle &) noexcept = delete;
        Handle &operator=(Handle &&) noexcept = delete;
        Handle &operator=(const Handle &) noexcept = delete;
        [[nodiscard]] std::string_view name() const noexcept;
        [[nodiscard]] auto device() const noexcept { return _device; }
        [[nodiscard]] auto context() const noexcept { return _context; }
        [[nodiscard]] auto compute_capability() const noexcept { return _compute_capability; }
    };

private:
    Handle _handle;

public:
    CUDADevice(const Context &ctx, uint device_id) noexcept;
    ~CUDADevice() noexcept override = default;
    [[nodiscard]] auto &handle() const noexcept { return _handle; }
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    uint64_t create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    uint64_t create_stream() noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList list) noexcept override;
    uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    uint64_t create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;
    uint64_t create_mesh() noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    uint64_t create_accel() noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    uint64_t create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override;
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    bool is_buffer_in_bindless_array(uint64_t array, uint64_t handle) noexcept override;
    bool is_texture_in_bindless_array(uint64_t array, uint64_t handle) noexcept override;
    void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept override;

    // TODO...
    void *native_handle() const noexcept override {
        return nullptr;
    }
    void *buffer_native_handle(uint64_t handle) const noexcept override {
        return nullptr;
    }
    void *texture_native_handle(uint64_t handle) const noexcept override {
        return nullptr;
    }
    void *stream_native_handle(uint64_t handle) const noexcept override {
        return nullptr;
    }

    template<typename F>
    decltype(auto) with_handle(F &&f) const noexcept {
        ContextGuard guard{_handle.context()};
        return f();
    }
};

}// namespace luisa::compute::cuda
