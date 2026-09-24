#pragma once

#include <cuda.h>

#include <luisa/core/stl/functional.h>
#include <luisa/runtime/rhi/device_interface.h>
#include "../common/default_binary_io.h"
#include "../common/rtx/fallback_rtx.h"
#include "cuda_error.h"
#include "cuda_texture.h"
#include "cuda_stream.h"
#include "cuda_compiler.h"
#include "optix_api.h"
#include "cuda_shader_metadata.h"
#include "extensions/cuda_external_ext.h"

namespace luisa::compute::cuda {

class CUDAOldDenoiserExt;
class CUDADenoiserExt;
class CUDADStorageExt;
class CUDAPinnedMemoryExt;
class CudaGraphExtImpl;
class CUDANativeShaderExt;

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

    /**
     * @brief Binds this device's CUDA context to the calling thread for the
     * duration of a call, without switching contexts between consecutive calls.
     *
     * A `cuCtxPushCurrent` / `cuCtxPopCurrent` pair around every single call -
     * which is what this guard used to do - is not free on the CUDA driver: a
     * context switch acts as a submission boundary and closes the driver's
     * current submission batch. Two kernels launched on two different streams
     * with a switch in between therefore end up in two batches, and the driver
     * executes those batches one after the other. That throws away the entire
     * point of spreading a batch of independent commands over several streams:
     * on an RTX 4060 with driver 595.71, 16 launches of the same kernel on 16
     * (non-blocking) CUDA streams take ~14.8 ms with a push/pop around every
     * launch and ~1.0 ms without one. See group F of
     * benchmark_cuda_vs_vk_cuda_reorder and its section 4.5.
     *
     * This guard therefore only switches when the calling thread is not already
     * running on this context, and it decides with `cuCtxGetCurrent`, which is
     * cheap and - unlike a context switch - not a submission boundary. Querying
     * on every call also means that a context change made behind our back (say,
     * interop code popping the context) is always noticed and never ignored.
     *
     * A thread that was running on a *different* context gets that context back
     * on destruction, exactly like the old push/pop did. A thread that had *no*
     * context keeps this one, and that is deliberate: it is what lets
     * back-to-back submissions of this device (typically one per CUDA stream)
     * run without a switch in between, and it matches the usual "bind the
     * engine's context once per thread" pattern. The binding is dropped again
     * when the device is destroyed.
     */
    class ContextGuard {

    private:
        CUcontext _ctx{};
        CUcontext _previous{};
        bool _restore_previous{false};

    public:
        explicit ContextGuard(CUcontext ctx) noexcept : _ctx{ctx} {
            CUcontext current = nullptr;
            LUISA_CHECK_CUDA(cuCtxGetCurrent(&current));
            if (current == _ctx) { return; }// already bound: no switch at all
            LUISA_CHECK_CUDA(cuCtxSetCurrent(_ctx));
            _previous = current;
            _restore_previous = current != nullptr;
        }
        ~ContextGuard() noexcept {
            if (!_restore_previous) { return; }
            CUcontext current = nullptr;
            LUISA_CHECK_CUDA(cuCtxGetCurrent(&current));
            if (current != _ctx) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid CUDA context {} (expected {}).",
                    fmt::ptr(current), fmt::ptr(_ctx));
            }
            LUISA_CHECK_CUDA(cuCtxSetCurrent(_previous));
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
        [[nodiscard]] auto handle_uuid() const noexcept { return _uuid; }
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
    luisa::string_view _cudadevrt_library;
    uint64_t _sparse_granularity{};
    mutable spin_mutex _event_manager_mutex;
    mutable luisa::unique_ptr<CUDAEventManager> _event_manager;
    // Software ray-tracing fallback.  `_use_fallback_rtx` is decided once, in the
    // constructor (see `DeviceConfigExt::use_fallback_rtx()` and
    // `optix::available()`); `_fallback_rtx` is created lazily on the first
    // acceleration-structure request, so a device that does not use the fallback
    // never allocates anything for it.  When `_use_fallback_rtx` is false the
    // whole path is inert and OptiX is used exactly as before.
    bool _use_fallback_rtx{false};
    mutable spin_mutex _fallback_rtx_mutex;
    mutable luisa::unique_ptr<lc::fallback_rtx::FallbackRtxDevice> _fallback_rtx;

private:
    // extensions
    //
    // NOTE: the extensions are declared *after* `_handle` on purpose: members
    // are destroyed in reverse declaration order, so every extension (and
    // therefore every GPU/API object it owns, e.g. the modules of the native
    // shaders) dies while this device's CUDA context is still alive.
    std::mutex _ext_mutex;
    luisa::unique_ptr<DeviceConfigExt> _device_config_ext;
    luisa::unique_ptr<CUDADStorageExt> _dstorage_ext;
    luisa::unique_ptr<CUDAPinnedMemoryExt> _pinned_memory_ext;
    luisa::unique_ptr<CudaGraphExtImpl> _cuda_graph_ext;
    luisa::unique_ptr<CUDAExternalExt> _external_ext;
    luisa::unique_ptr<CUDANativeShaderExt> _native_shader_ext;
#if LUISA_BACKEND_ENABLE_OIDN
    luisa::unique_ptr<CUDADenoiserExt> _denoiser_ext;
#endif

#ifdef LUISA_COMPUTE_ENABLE_NVTT
    luisa::unique_ptr<CUDATexCompressExt> _tex_comp_ext;
#endif

private:
    [[nodiscard]] ShaderCreationInfo _load_or_compile_shader(
        luisa::string name,
        const string &source, const ShaderOption &option,
        luisa::span<const char *const> nvrtc_options,
        const CUDAShaderMetadata &expected_metadata,
        luisa::vector<ShaderDispatchCommand::Argument> bound_arguments,
        luisa::function<luisa::string()> generate_ptx = {}) noexcept;

public:
    CUDADevice(Context &&ctx, size_t device_id, const BinaryIO *io, bool use_lmdb,
               luisa::unique_ptr<DeviceConfigExt> device_config_ext, bool headless) noexcept;
    ~CUDADevice() noexcept override;
    [[nodiscard]] const auto &handle() const noexcept { return _handle; }
    template<typename F>
    decltype(auto) with_handle(F &&f) const noexcept {
        ContextGuard guard{_handle.context()};
        return f();
    }
    void *native_handle() const noexcept override { return _handle.context(); }
    [[nodiscard]] uint compute_warp_size() const noexcept override { return 32u; }
    [[nodiscard]] size_t compute_max_shared_memory_size() const noexcept override {
        return with_handle([this] {
            int bytes = 0;
            LUISA_CHECK_CUDA(cuDeviceGetAttribute(
                &bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                _handle.device()));
            return static_cast<size_t>(bytes);
        });
    }
    [[nodiscard]] uint64_t memory_granularity() const noexcept override { return _sparse_granularity; }
    [[nodiscard]] uint64_t sparse_granularity() const noexcept { return _sparse_granularity; }

public:
    [[nodiscard]] auto accel_update_function() const noexcept { return _accel_update_function; }
    [[nodiscard]] auto instance_handle_update_function() const noexcept { return _instance_handle_update_function; }
    [[nodiscard]] auto bindless_array_update_function() const noexcept { return _bindless_array_update_function; }
    [[nodiscard]] auto cudadevrt_library() const noexcept { return luisa::string_view{_cudadevrt_library}; }
    [[nodiscard]] auto compiler() const noexcept { return _compiler.get(); }
    // Whether this device answers ray tracing with the software fallback (and
    // therefore never initialises OptiX).
    [[nodiscard]] bool use_fallback_rtx() const noexcept { return _use_fallback_rtx; }
    // The fallback device, or `nullptr` when the hardware path is in use.  It is
    // created on the first call; the object lives as long as the device does.
    [[nodiscard]] lc::fallback_rtx::FallbackRtxDevice *fallback_rtx() noexcept;
    // Whether `handle` belongs to the fallback, i.e. whether an acceleration
    // structure has to be routed into it instead of into the OptiX path.
    [[nodiscard]] bool owns_fallback_blas(uint64_t handle) noexcept {
        auto fallback = fallback_rtx();
        return fallback != nullptr && fallback->owns_blas(handle);
    }
    [[nodiscard]] bool owns_fallback_accel(uint64_t handle) noexcept {
        auto fallback = fallback_rtx();
        return fallback != nullptr && fallback->owns_accel(handle);
    }
    [[nodiscard]] auto io() const noexcept { return _io; }
    [[nodiscard]] CUDAEventManager *event_manager() const noexcept;

public:
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count, void *external_memory) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    ResourceCreationInfo create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels,
                                        void *external_native_handle, bool simultaneous_access, bool allow_raster_target) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    ResourceCreationInfo create_bindless_array(size_t size, BindlessSlotType type) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;
    SwapchainCreationInfo create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept override;
    void destroy_swapchain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_tile_kernel(const ShaderOption &option, const tile::Function &kernel,
                                          const tile::CompileOptions &tile_options,
                                          tile::KernelMetadata &metadata) noexcept override;
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
    // sparse
    SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept override;
    ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override;
    void update_sparse_resources(
        uint64_t stream_handle,
        luisa::vector<SparseUpdateTile> &&textures_update) noexcept override;
    void destroy_sparse_buffer(uint64_t handle) noexcept override;
};

}// namespace luisa::compute::cuda
