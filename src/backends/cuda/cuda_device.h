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

/**
 * @brief CUDA device
 * 
 */
class CUDADevice final : public Device::Interface {

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
    /**
     * @brief Device handle of CUDA
     * 
     */
    class Handle {

    private:
        CUcontext _context{nullptr};
        OptixDeviceContext _optix_context{nullptr};
        CUdevice _device{0};
        uint32_t _compute_capability{};

    public:
        /**
         * @brief Construct a new Handle object
         * 
         * @param index index of CUDA device
         */
        explicit Handle(uint index) noexcept;
        ~Handle() noexcept;
        Handle(Handle &&) noexcept = delete;
        Handle(const Handle &) noexcept = delete;
        Handle &operator=(Handle &&) noexcept = delete;
        Handle &operator=(const Handle &) noexcept = delete;
        /**
         * @brief Return name of device
         * 
         * @return std::string_view 
         */
        [[nodiscard]] std::string_view name() const noexcept;
        /**
         * @brief Return handle of device
         * 
         * @return CUdevice
         */
        [[nodiscard]] auto device() const noexcept { return _device; }
        /**
         * @brief Return handle of context
         * 
         * @return CUcontext
         */
        [[nodiscard]] auto context() const noexcept { return _context; }
        /**
         * @brief Return handle of Optix context
         * 
         * @return OptixDeviceContext
         */
        [[nodiscard]] auto optix_context() const noexcept { return _optix_context; }
        /**
         * @brief Return compute capability
         * 
         * @return uint32
         */
        [[nodiscard]] auto compute_capability() const noexcept { return _compute_capability; }
    };

private:
    Handle _handle;
    luisa::unique_ptr<CUDAHeap> _heap;

public:
    /**
     * @brief Construct a new CUDADevice object
     * 
     * @param ctx context
     * @param device_id device id
     */
    CUDADevice(const Context &ctx, uint device_id) noexcept;
    ~CUDADevice() noexcept override;
    /**
     * @brief Return handle of device
     * 
     * @return Handle
     */
    [[nodiscard]] auto &handle() const noexcept { return _handle; }
    /**
     * @brief Return address of CUDAHeap
     * 
     * @return CUDAHeap*
     */
    [[nodiscard]] auto heap() noexcept { return _heap.get(); }
    /**
     * @brief Create a buffer on device
     * 
     * @param size_bytes size of buffer
     * @return handle of buffer
     */
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    /**
     * @brief Destroy a buffer on device
     * 
     * @param handle handle of buffer
     */
    void destroy_buffer(uint64_t handle) noexcept override;
    /**
     * @brief Create a texture on device
     * 
     * @param format pixel format
     * @param dimension dimension
     * @param width width
     * @param height height
     * @param depth depth
     * @param mipmap_levels mipmap levels
     * @return handle of texture
     */
    uint64_t create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept override;
    /**
     * @brief Destroy a texture on device
     * 
     * @param handle handle of texture
     */
    void destroy_texture(uint64_t handle) noexcept override;
    /**
     * @brief Create a CUDAStream object
     * 
     * @return address of CUDAStream
     */
    uint64_t create_stream() noexcept override;
    /**
     * @brief Destroy a CUDAStream object
     * 
     * @param handle address of CUDAStream
     */
    void destroy_stream(uint64_t handle) noexcept override;
    /**
     * @brief Synchronize a CUDAStream
     * 
     * @param stream_handle address of CUDAStream
     */
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    /**
     * @brief Dispatch commands to a stream
     * 
     * @param stream_handle address of CUDAStream
     * @param list list of commands
     */
    void dispatch(uint64_t stream_handle, const CommandList &list) noexcept override;
    /**
     * @brief Dispatch multiple command lists
     *
     * @param stream_handle address of CUDAStream
     * @param lists vector of lists of commands
     */
    void dispatch(uint64_t stream_handle, luisa::span<const CommandList> lists) noexcept override;
    /**
     * @brief Create a shader on device
     * 
     * @param kernel kernel function
     * @param meta_options meta options
     * @return handle of shader
     */
    uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept override;
    /**
     * @brief Destroy a shader on device
     * 
     * @param handle handle of shader
     */
    void destroy_shader(uint64_t handle) noexcept override;
    /**
     * @brief Create a event on device
     * 
     * @return handle of event
     */
    uint64_t create_event() noexcept override;
    /**
     * @brief Destroy a event on device
     * 
     * @param handle handle of event
     */
    void destroy_event(uint64_t handle) noexcept override;
    /**
     * @brief Signal a event on device
     * 
     * @param handle handle of event
     * @param stream_handle handle of stream
     */
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    /**
     * @brief Wait a event on device
     * 
     * @param handle handle of event
     * @param stream_handle handle of stream
     */
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    /**
     * @brief Synchronize a event on device
     * 
     * @param handle handle of event
     */
    void synchronize_event(uint64_t handle) noexcept override;
    /**
     * @brief Create a mesh
     * 
     * @param v_buffer handle of vertex buffer
     * @param v_offset offset of vertex buffer
     * @param v_stride stride of vertex buffer
     * @param v_count count of vertices
     * @param t_buffer handle of triangle buffer
     * @param t_offset offset of triangle buffer
     * @param t_count count of triangles
     * @param hint build hint
     * @return handle of CUDAMesh object
     */
    uint64_t create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint)  noexcept override;
    /**
     * @brief Destroy a mesh
     * 
     * @param handle handle of CUDAMesh object
     */
    void destroy_mesh(uint64_t handle) noexcept override;
    /**
     * @brief Get the vertex buffer from mesh object
     * 
     * @param mesh_handle handle of mesh
     * @return handle of vertex buffer
     */
    uint64_t get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    /**
     * @brief Get the triangle buffer from mesh object
     * 
     * @param mesh_handle handle of mesh
     * @return handle of triangle buffer
     */
    uint64_t get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    /**
     * @brief Create an accel object
     * 
     * @param hint build hint
     * @return handle of CUDAAccel object
     */
    uint64_t create_accel(AccelBuildHint hint) noexcept override;
    /**
     * @brief Emplace back an instnce in accel
     * 
     * @param accel handle of accel
     * @param mesh handle of mesh
     * @param transform mesh's transform
     * @param visible mesh's visibility 
     */
    void emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, float4x4 transform, bool visible) noexcept override;
    /**
     * @brief Set tranfrom of instance
     * 
     * @param accel handle of accel
     * @param index place to set
     * @param transform new transform
     */
    void set_instance_transform_in_accel(uint64_t accel, size_t index, float4x4 transform) noexcept override;
    /**
     * @brief If buffer is in accel
     * 
     * @param accel handle of accel
     * @param buffer handle of buffer
     * @return true 
     * @return false 
     */
    bool is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept override;
    /**
     * @brief If mesh is in accel
     * 
     * @param accel handle of accel
     * @param mesh handle of mesh
     * @return true 
     * @return false 
     */
    bool is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept override;
    /**
     * @brief Destroy an accel
     * 
     * @param handle handle of accel
     */
    void destroy_accel(uint64_t handle) noexcept override;
    /**
     * @brief Create a bindless array
     * 
     * @param size size of bindless array
     * @return handle of CUDABindlessArray object
     */
    uint64_t create_bindless_array(size_t size) noexcept override;
    /**
     * @brief Destroy a bindless array
     * 
     * @param handle handle of bindless array
     */
    void destroy_bindless_array(uint64_t handle) noexcept override;
    /**
     * @brief Emplace buffer in bindless array
     * 
     * @param array handle of bindless array
     * @param index index to emplace
     * @param handle handle of buffer
     * @param offset_bytes offset of buffer
     */
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override;
    /**
     * @brief Emplace 2D texture in bindless array
     * 
     * @param array handle of bindless array
     * @param index place to emplace
     * @param handle handle of 2D texture
     * @param sampler sampler of 2D texture
     */
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    /**
     * @brief Emplace 3D texture in bindless array
     * 
     * @param array handle of bindless array
     * @param index place to emplace
     * @param handle handle of 3D texture
     * @param sampler sampler of 3D texture
     */
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    /**
     * @brief If buffer is in bindless array
     * 
     * @param array handle of bindless array
     * @param handle handle of buffer
     * @return true 
     * @return false 
     */
    bool is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    /**
     * @brief If texture is in bindless array
     * 
     * @param array handle of bidnless array
     * @param handle handle of texture
     * @return true 
     * @return false 
     */
    bool is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    /**
     * @brief Remove buffer from bidnless array
     * 
     * @param array handle of bindless array
     * @param index place to remove
     */
    void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept override;
    /**
     * @brief Remove 2D texture from bidnless array
     * 
     * @param array handle of bindless array
     * @param index place to remove
     */
    void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    /**
     * @brief Remove 3D texture from bidnless array
     * 
     * @param array handle of bindless array
     * @param index place to remove
     */
    void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept override;

    /**
     * @brief Return CUDA handle of context
     * 
     * @return CUcontext
     */
    void *native_handle() const noexcept override { return _handle.context(); }
    /**
     * @brief Return native handle of buffer
     * 
     * @param handle handle of buffer
     * @return void* 
     */
    void *buffer_native_handle(uint64_t handle) const noexcept override {
        return reinterpret_cast<void *>(handle);
    }
    /**
     * @brief Return native handle of texture
     * 
     * @param handle handle of texture(CUDAMipmapArray)
     * @return void* 
     */
    void *texture_native_handle(uint64_t handle) const noexcept override {
        return reinterpret_cast<void *>(reinterpret_cast<CUDAMipmapArray *>(handle)->handle());
    }
    /**
     * @brief Return native handle of stream
     * 
     * @param handle handle of CUDAStream
     * @return void* 
     */
    void *stream_native_handle(uint64_t handle) const noexcept override {
        return reinterpret_cast<CUDAStream *>(handle)->handle();
    }

    /**
     * @brief Run function with ContextGuard
     * 
     * @tparam F function type
     * @param f function
     * @return return of function f 
     */
    template<typename F>
    decltype(auto) with_handle(F &&f) const noexcept {
        ContextGuard guard{_handle.context()};
        return f();
    }
    /**
     * @brief Pop back instance from accel
     * 
     * @param accel handle of accel
     */
    void pop_back_instance_from_accel(uint64_t accel) noexcept override;
    /**
     * @brief Set instance in accel
     * 
     * @param accel handle of accel
     * @param index place to set
     * @param mesh new mesh
     * @param transform new transform
     * @param visible new visibility
     */
    void set_instance_in_accel(uint64_t accel, size_t index, uint64_t mesh, float4x4 transform, bool visible) noexcept override;
    /**
     * @brief Set instance visibility in accel
     * 
     * @param accel handle of accel
     * @param index place to set
     * @param visible new visibility
     */
    void set_instance_visibility_in_accel(uint64_t accel, size_t index, bool visible) noexcept override;
    /**
     * @brief If requires command reordering
     * 
     * @return true
     */
    bool requires_command_reordering() const noexcept override { return true; }
};

}// namespace luisa::compute::cuda
