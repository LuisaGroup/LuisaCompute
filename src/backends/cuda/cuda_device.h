#pragma once

#include <cuda.h>

#include <luisa/runtime/rhi/device_interface.h>
#include "../common/default_binary_io.h"
#include "cuda_error.h"
#include "cuda_texture.h"
#include "cuda_stream.h"
#include "cuda_compiler.h"
#include "optix_api.h"
#include "cuda_shader_metadata.h"

namespace luisa::compute::cuda {

class CUDAOldDenoiserExt;
class CUDADenoiserExt;
class CUDADStorageExt;
class CUDAPinnedMemoryExt;

#ifdef LUISA_COMPUTE_ENABLE_NVTT
class CUDATexCompressExt;
#endif

class CUDATimelineEventPool;
class CUDAEventManager;

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
        CUdevice _device{0};
        uint32_t _device_index{};
        uint32_t _compute_capability{};
        uint32_t _driver_version{};
        CUuuid _uuid{};
        // will be lazily initialized
        mutable optix::DeviceContext _optix_context{nullptr};
        mutable spin_mutex _mutex{};

    public:
        explicit Handle(size_t index) noexcept;
        ~Handle() noexcept;
        Handle(Handle &&) noexcept = delete;
        Handle(const Handle &) noexcept = delete;
        Handle &operator=(Handle &&) noexcept = delete;
        Handle &operator=(const Handle &) noexcept = delete;
        [[nodiscard]] std::string_view name() const noexcept;
        [[nodiscard]] auto index() const noexcept { return _device_index; }
        [[nodiscard]] auto uuid() const noexcept { return _uuid; }
        [[nodiscard]] auto device() const noexcept { return _device; }
        [[nodiscard]] auto context() const noexcept { return _context; }
        [[nodiscard]] auto driver_version() const noexcept { return _driver_version; }
        void force_compute_capability(uint32_t cc) noexcept { _compute_capability = cc; }
        [[nodiscard]] auto compute_capability() const noexcept { return _compute_capability; }
        [[nodiscard]] optix::DeviceContext optix_context() const noexcept;
    };

private:
    Handle _handle;
    CUmodule _builtin_kernel_module{nullptr};
    CUfunction _accel_update_function{nullptr};
    CUfunction _instance_handle_update_function{nullptr};
    CUfunction _bindless_array_update_function{nullptr};
    luisa::unique_ptr<CUDACompiler> _compiler;
    luisa::unique_ptr<DefaultBinaryIO> _default_io;
    const BinaryIO *_io{nullptr};
    luisa::string _cudadevrt_library;

    mutable spin_mutex _event_manager_mutex;
    mutable luisa::unique_ptr<CUDAEventManager> _event_manager;

private:
    // extensions
    std::mutex _ext_mutex;
    luisa::unique_ptr<CUDADStorageExt> _dstorage_ext;
    luisa::unique_ptr<CUDAPinnedMemoryExt> _pinned_memory_ext;

#if LUISA_BACKEND_ENABLE_OIDN
    luisa::unique_ptr<CUDADenoiserExt> _denoiser_ext;
#endif

#ifdef LUISA_COMPUTE_ENABLE_NVTT
    luisa::unique_ptr<CUDATexCompressExt> _tex_comp_ext;
#endif

private:
    [[nodiscard]] ShaderCreationInfo _create_shader(luisa::string name,
                                                    const string &source, const ShaderOption &option,
                                                    luisa::span<const char *const> nvrtc_options,
                                                    const CUDAShaderMetadata &expected_metadata,
                                                    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept;

public:
    CUDADevice(Context &&ctx, size_t device_id, const BinaryIO *io) noexcept;
    ~CUDADevice() noexcept override;
    [[nodiscard]] auto &handle() const noexcept { return _handle; }
    template<typename F>
    decltype(auto) with_handle(F &&f) const noexcept {
        ContextGuard guard{_handle.context()};
        return f();
    }
    void *native_handle() const noexcept override { return _handle.context(); }
    [[nodiscard]] uint compute_warp_size() const noexcept override { return 32u; }

public:
    [[nodiscard]] auto accel_update_function() const noexcept { return _accel_update_function; }
    [[nodiscard]] auto instance_handle_update_function() const noexcept { return _instance_handle_update_function; }
    [[nodiscard]] auto bindless_array_update_function() const noexcept { return _bindless_array_update_function; }
    [[nodiscard]] auto cudadevrt_library() const noexcept { return luisa::string_view{_cudadevrt_library}; }
    [[nodiscard]] auto compiler() const noexcept { return _compiler.get(); }
    [[nodiscard]] auto io() const noexcept { return _io; }
    [[nodiscard]] CUDAEventManager *event_manager() const noexcept;

public:
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count, void *external_memory) noexcept override;
    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count, void *external_memory) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    ResourceCreationInfo create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;
    SwapchainCreationInfo create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept override;
    void destroy_swap_chain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override;
    ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override;
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept override;
    bool is_event_completed(uint64_t handle, uint64_t value) const noexcept override;
    void synchronize_event(uint64_t handle, uint64_t value) noexcept override;
    ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_curve(const AccelOption &option) noexcept override;
    void destroy_curve(uint64_t handle) noexcept override;
    ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_motion_instance(const AccelMotionOption &option) noexcept override;
    void destroy_motion_instance(uint64_t handle) noexcept override;
    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    string query(luisa::string_view property) noexcept override;
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
    DeviceExtension *extension(luisa::string_view name) noexcept override;
};

}// namespace luisa::compute::cuda

