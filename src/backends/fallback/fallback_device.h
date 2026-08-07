//
// Created by Mike Smith on 2022/5/23.
//

#pragma once

#include <luisa/runtime/device.h>
#include <luisa/core/spin_mutex.h>

#include "../common/default_binary_io.h"
#include "fallback_embree.h"

namespace llvm {
class TargetMachine;
}// namespace llvm

namespace llvm::orc {
class LLJIT;
}// namespace llvm::orc

namespace luisa::compute::fallback {

class FallbackTexCompressInterface;

class FallbackDevice : public DeviceInterface {

private:
    RTCDevice _rtc_device{nullptr};
    luisa::unique_ptr<DefaultBinaryIO> _default_io{};
    const BinaryIO *_io{};

private:
    luisa::spin_mutex _ext_mutex;
    luisa::unique_ptr<FallbackTexCompressInterface> _tex_compress_ext{};

public:
    explicit FallbackDevice(Context &&ctx, const DeviceConfig *config) noexcept;
    ~FallbackDevice() noexcept override;
    [[nodiscard]] auto binary_io() const noexcept { return _io; }
    void *native_handle() const noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void destroy_swapchain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    uint compute_warp_size() const noexcept override;
    uint64_t memory_granularity() const noexcept override;
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count, void *external_memory) noexcept override;
    ResourceCreationInfo create_texture(PixelFormat format, uint dimension,
                                        uint width, uint height, uint depth,
                                        uint mipmap_levels, void *external_native_handle,
                                        bool simultaneous_access, bool allow_raster_target) noexcept override;
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;
    void set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept override;
    SwapchainCreationInfo create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override;
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    bool is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept override;
    void synchronize_event(uint64_t handle, uint64_t fence_value) noexcept override;
    ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;
    ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    ResourceCreationInfo create_curve(const AccelOption &option) noexcept override;
    void destroy_curve(uint64_t handle) noexcept override;
    ResourceCreationInfo create_motion_instance(const AccelMotionOption &option) noexcept override;
    void destroy_motion_instance(uint64_t handle) noexcept override;
    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    string query(luisa::string_view property) noexcept override;
    DeviceExtension *extension(luisa::string_view name) noexcept override;
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
    SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept override;
    ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override;
    void update_sparse_resources(uint64_t stream_handle, vector<luisa::compute::SparseUpdateTile> &&textures_update) noexcept override;
    void destroy_sparse_buffer(uint64_t handle) noexcept override;
    ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_texture_heap(uint64_t handle) noexcept override;
    SparseTextureCreationInfo create_sparse_texture(PixelFormat format, uint dimension,
                                                    uint width, uint height, uint depth,
                                                    uint mipmap_levels, bool simultaneous_access) noexcept override;
    void destroy_sparse_texture(uint64_t handle) noexcept override;
    ResourceCreationInfo create_bindless_array(size_t size, BindlessSlotType type) noexcept override;
    ResourceCreationInfo create_event() noexcept override;
};

}// namespace luisa::compute::fallback
