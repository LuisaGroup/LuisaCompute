//
// Created by mike on 12/25/25.
//

#pragma once

#include <mutex>

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>
#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>
#include "../common/default_binary_io.h"

namespace luisa::compute::hip {

class HIPPinnedMemoryExt;
class HIPMotionMeshBuiltin;

class HIPDevice final : public luisa::compute::DeviceInterface {

private:
    luisa::unique_ptr<DefaultBinaryIO> _default_io{nullptr};
    const BinaryIO *_io{nullptr};
    int _device_id;
    hipDevice_t _hip_device{};
    hipCtx_t _hip_context{nullptr};
    luisa::string _amdgpu_arch;
    mutable std::mutex _hiprt_mutex;
    mutable hiprtContext _hiprt_context{nullptr};
    mutable hiprtGlobalStackBuffer _hiprt_global_stack_buffer{};
    mutable bool _hiprt_global_stack_buffer_initialized{false};
    std::mutex _extension_mutex;
    luisa::unique_ptr<HIPPinnedMemoryExt> _pinned_memory_ext{nullptr};
    mutable std::mutex _motion_mesh_builtin_mutex;
    mutable luisa::unique_ptr<HIPMotionMeshBuiltin> _motion_mesh_builtin{nullptr};

    template<typename F>
    decltype(auto) with_device(F &&f) const noexcept;
    [[nodiscard]] hiprtContext _ensure_hiprt_context_locked() const noexcept;

public:
    HIPDevice(Context &&ctx, const DeviceConfig *config) noexcept;
    ~HIPDevice() noexcept override;
    [[nodiscard]] auto device_id() const noexcept { return _device_id; }
    [[nodiscard]] auto amdgpu_arch() const noexcept { return luisa::string_view{_amdgpu_arch}; }
    [[nodiscard]] auto uses_hardware_rt_stack() const noexcept {
        return _amdgpu_arch == "gfx1200" || _amdgpu_arch == "gfx1201";
    }
    [[nodiscard]] hiprtContext hiprt_context() const noexcept;
    [[nodiscard]] hiprtGlobalStackBuffer hiprt_global_stack_buffer() const noexcept;
    [[nodiscard]] HIPMotionMeshBuiltin &motion_mesh_builtin() const noexcept;
    [[nodiscard]] hipUUID_t device_uuid() const noexcept;
    [[nodiscard]] hipUUID_t device_uuid_for_vulkan() const noexcept;
    [[nodiscard]] void *native_handle() const noexcept override;
    [[nodiscard]] uint compute_warp_size() const noexcept override;
    [[nodiscard]] uint64_t memory_granularity() const noexcept override;
    [[nodiscard]] BufferCreationInfo create_buffer(const Type *element, size_t elem_count, void *external_memory) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, void *external_native_handle, bool simultaneous_access, bool allow_raster_target) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_bindless_array(size_t size, BindlessSlotType type) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;
    [[nodiscard]] SwapchainCreationInfo create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept override;
    void destroy_swapchain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
    [[nodiscard]] ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    [[nodiscard]] ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<Type const *const> arg_types) noexcept override;
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    bool is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept override;
    void synchronize_event(uint64_t handle, uint64_t fence_value) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    void set_name(Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
    void set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_curve(const AccelOption &option) noexcept override;
    void destroy_curve(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_motion_instance(const AccelMotionOption &option) noexcept override;
    void destroy_motion_instance(uint64_t handle) noexcept override;
    [[nodiscard]] luisa::string query(luisa::string_view property) noexcept override;
    [[nodiscard]] DeviceExtension *extension(luisa::string_view name) noexcept override;
    [[nodiscard]] luisa::string_view get_name(uint64_t resource_handle) const noexcept override;
    [[nodiscard]] SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept override;
    [[nodiscard]] ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override;
    void update_sparse_resources(uint64_t stream_handle, luisa::vector<SparseUpdateTile> &&textures_update) noexcept override;
    void destroy_sparse_buffer(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_texture_heap(uint64_t handle) noexcept override;
    [[nodiscard]] SparseTextureCreationInfo create_sparse_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, bool simultaneous_access) noexcept override;
    void destroy_sparse_texture(uint64_t handle) noexcept override;
};

}// namespace luisa::compute::hip
