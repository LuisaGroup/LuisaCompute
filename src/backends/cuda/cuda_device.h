//
// Created by Mike on 7/28/2021.
//

#pragma once

#include <cuda.h>

#include <runtime/device_interface.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/optix_api.h>

namespace luisa::compute::cuda {

/**
 * @brief CUDA device
 * 
 */
class CUDADevice final : public DeviceInterface {

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
        optix::DeviceContext _optix_context{nullptr};
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
    CUmodule _accel_update_module{nullptr};
    CUfunction _accel_update_function{nullptr};
    CUfunction _stream_wait_value_function{nullptr};
    BinaryIO *_io{nullptr};

public:
    CUDADevice(Context &&ctx, uint device_id) noexcept;
    ~CUDADevice() noexcept override;
    [[nodiscard]] auto &handle() const noexcept { return _handle; }
    template<typename F>
    decltype(auto) with_handle(F &&f) const noexcept {
        ContextGuard guard{_handle.context()};
        return f();
    }
    void *native_handle() const noexcept override { return _handle.context(); }
    [[nodiscard]] auto accel_update_function() const noexcept { return _accel_update_function; }
    [[nodiscard]] auto io() const noexcept { return _io; }
    void set_io(BinaryIO *binary_io) noexcept override { _io = binary_io; }
    bool is_c_api() const noexcept override { return false; }
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    ResourceCreationInfo create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept override;
    void destroy_depth_buffer(uint64_t handle) noexcept override;
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;
    SwapChainCreationInfo create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override;
    ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    ResourceCreationInfo create_raster_shader(const MeshFormat &mesh_format, const RasterState &raster_state, luisa::span<const PixelFormat> rtv_format, DepthFormat dsv_format, Function vert, Function pixel, ShaderOption shader_option) noexcept override;
    void save_raster_shader(const MeshFormat &mesh_format, Function vert, Function pixel, luisa::string_view name, bool enable_debug_info, bool enable_fast_math) noexcept override;
    ResourceCreationInfo load_raster_shader(const MeshFormat &mesh_format, const RasterState &raster_state, luisa::span<const PixelFormat> rtv_format, DepthFormat dsv_format, luisa::span<const Type *const> types, luisa::string_view ser_path) noexcept override;
    void destroy_raster_shader(uint64_t handle) noexcept override;
    ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;
    ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    string query(luisa::string_view property) noexcept override;
    DeviceExtension *extension(luisa::string_view name) noexcept override;
};

}// namespace luisa::compute::cuda
