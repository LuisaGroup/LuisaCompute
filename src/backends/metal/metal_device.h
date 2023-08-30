#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include "../common/default_binary_io.h"
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalCompiler;
class MetalDStorageExt;
class MetalDebugCaptureExt;
class MetalPinnedMemoryExt;

class MetalDevice : public DeviceInterface {

public:
    static constexpr auto update_bindless_slots_block_size = 256u;
    static constexpr auto update_accel_instances_block_size = 256u;
    static constexpr auto prepare_indirect_dispatches_block_size = 64u;

private:
    MTL::Device *_handle{nullptr};
    MTL::ComputePipelineState *_builtin_update_bindless_slots{nullptr};
    MTL::ComputePipelineState *_builtin_update_accel_instances{nullptr};
    MTL::ComputePipelineState *_builtin_prepare_indirect_dispatches{nullptr};
    MTL::RenderPipelineState *_builtin_swapchain_present_ldr{nullptr};
    MTL::RenderPipelineState *_builtin_swapchain_present_hdr{nullptr};
    luisa::unique_ptr<DefaultBinaryIO> _default_io;
    const BinaryIO *_io{nullptr};
    luisa::unique_ptr<MetalCompiler> _compiler;
    bool _inqueue_buffer_limit;

private:
    std::mutex _ext_mutex;
    luisa::unique_ptr<MetalDStorageExt> _dstorage_ext;
    luisa::unique_ptr<MetalPinnedMemoryExt> _pinned_memory_ext;
    luisa::unique_ptr<MetalDebugCaptureExt> _debug_capture_ext;

public:
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto io() const noexcept { return _io; }
    [[nodiscard]] auto builtin_update_bindless_slots() const noexcept { return _builtin_update_bindless_slots; }
    [[nodiscard]] auto builtin_update_accel_instances() const noexcept { return _builtin_update_accel_instances; }
    [[nodiscard]] auto builtin_prepare_indirect_dispatches() const noexcept { return _builtin_prepare_indirect_dispatches; }
    [[nodiscard]] auto builtin_swapchain_present_ldr() const noexcept { return _builtin_swapchain_present_ldr; }
    [[nodiscard]] auto builtin_swapchain_present_hdr() const noexcept { return _builtin_swapchain_present_hdr; }

public:
    MetalDevice(Context &&ctx, const DeviceConfig *config) noexcept;
    ~MetalDevice() noexcept override;
    void *native_handle() const noexcept override;
    uint compute_warp_size() const noexcept override;
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override;
    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    ResourceCreationInfo create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, bool simultaneous_access) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    ResourceCreationInfo create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override;
    SwapchainCreationInfo create_swapchain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) noexcept override;
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
    bool is_event_completed(uint64_t handle, uint64_t value) const noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept override;
    void synchronize_event(uint64_t handle, uint64_t value) noexcept override;
    ResourceCreationInfo create_mesh(const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    string query(luisa::string_view property) noexcept override;
    DeviceExtension *extension(luisa::string_view name) noexcept override;
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
};

}// namespace luisa::compute::metal

