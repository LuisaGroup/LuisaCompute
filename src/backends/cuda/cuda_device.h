//
// Created by Mike on 7/28/2021.
//

#pragma once

#include <cuda.h>
#include <optix.h>

#include <runtime/device.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_heap.h>

namespace luisa::compute::cuda {

class CUDADevice final : public Device::Interface {

    class ContextGuard {

    private:
        CUcontext _ctx;

    public:
        explicit ContextGuard(CUcontext ctx) noexcept : _ctx{ctx} {
            static std::once_flag flag;
            std::call_once(flag, [ctx]{
                LUISA_CHECK_CUDA(cuCtxPushCurrent(ctx));
            });
//            LUISA_CHECK_CUDA(cuCtxPushCurrent(_ctx));
        }
        ~ContextGuard() noexcept {
//            CUcontext ctx = nullptr;
//            LUISA_CHECK_CUDA(cuCtxPopCurrent(&ctx));
//            if (ctx != _ctx) [[unlikely]] {
//                LUISA_ERROR_WITH_LOCATION(
//                    "Invalid CUDA context {} (expected {}).",
//                    fmt::ptr(ctx), fmt::ptr(_ctx));
//            }
        }
    };

public:
    class Handle {

    private:
        CUcontext _context{nullptr};
        OptixDeviceContext _optix_context{nullptr};
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
        [[nodiscard]] auto optix_context() const noexcept { return _optix_context; }
        [[nodiscard]] auto compute_capability() const noexcept { return _compute_capability; }
    };

private:
    Handle _handle;
    luisa::unique_ptr<CUDAHeap> _heap;

public:
    CUDADevice(const Context &ctx, uint device_id) noexcept;
    ~CUDADevice() noexcept override;
    [[nodiscard]] auto &handle() const noexcept { return _handle; }
    [[nodiscard]] auto heap() noexcept { return _heap.get(); }
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
    uint64_t create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint)  noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    uint64_t get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    uint64_t get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    uint64_t create_accel(AccelBuildHint hint) noexcept override;
    void emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, float4x4 transform, bool visible) noexcept override;
    void set_instance_transform_in_accel(uint64_t accel, size_t index, float4x4 transform) noexcept override;
    bool is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept override;
    bool is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    uint64_t create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override;
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    bool is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    bool is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept override;

    void *native_handle() const noexcept override { return _handle.context(); }
    void *buffer_native_handle(uint64_t handle) const noexcept override {
        return reinterpret_cast<void *>(handle);
    }
    void *texture_native_handle(uint64_t handle) const noexcept override {
        return reinterpret_cast<void *>(reinterpret_cast<CUDAMipmapArray *>(handle)->handle());
    }
    void *stream_native_handle(uint64_t handle) const noexcept override {
        return reinterpret_cast<CUDAStream *>(handle)->handle();
    }

    template<typename F>
    decltype(auto) with_handle(F &&f) const noexcept {
        ContextGuard guard{_handle.context()};
        return f();
    }
    void pop_back_instance_from_accel(uint64_t accel) noexcept override;
    void set_instance_in_accel(uint64_t accel, size_t index, uint64_t mesh, float4x4 transform, bool visible) noexcept override;
    void set_instance_visibility_in_accel(uint64_t accel, size_t index, bool visible) noexcept override;
};

}// namespace luisa::compute::cuda
