#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/runtime/device.h>

namespace luisa::compute::simd {

class SIMDThreadPool;

class SIMDDevice final : public DeviceInterface {

private:
    uint _warp_width{8u};
    luisa::unique_ptr<SIMDThreadPool> _thread_pool;

public:
    explicit SIMDDevice(Context &&context, const DeviceConfig *config) noexcept;
    ~SIMDDevice() noexcept override;

    [[nodiscard]] void *native_handle() const noexcept override;
    [[nodiscard]] uint compute_warp_size() const noexcept override;
    [[nodiscard]] uint64_t memory_granularity() const noexcept override;

    [[nodiscard]] BufferCreationInfo create_buffer(
        const Type *element, size_t elem_count,
        void *external_memory) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, void *external_native_handle,
        bool simultaneous_access,
        bool allow_raster_target) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_bindless_array(
        size_t size, BindlessSlotType type) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    [[nodiscard]] ResourceCreationInfo create_stream(
        StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(
        uint64_t stream_handle, CommandList &&list) noexcept override;
    void set_stream_log_callback(
        uint64_t stream_handle,
        const StreamLogCallback &callback) noexcept override;

    [[nodiscard]] SwapchainCreationInfo create_swapchain(
        const SwapchainOption &option,
        uint64_t stream_handle) noexcept override;
    void destroy_swapchain(uint64_t handle) noexcept override;
    void present_display_in_stream(
        uint64_t stream_handle, uint64_t swapchain_handle,
        uint64_t image_handle) noexcept override;

    [[nodiscard]] ShaderCreationInfo create_shader(
        const ShaderOption &option, Function kernel) noexcept override;
    [[nodiscard]] ShaderCreationInfo load_shader(
        luisa::string_view name,
        luisa::span<const Type *const> arg_types) noexcept override;
    [[nodiscard]] Usage shader_argument_usage(
        uint64_t handle, size_t index) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    [[nodiscard]] ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(
        uint64_t handle, uint64_t stream_handle,
        uint64_t fence_value) noexcept override;
    void wait_event(
        uint64_t handle, uint64_t stream_handle,
        uint64_t fence_value) noexcept override;
    [[nodiscard]] bool is_event_completed(
        uint64_t handle, uint64_t fence_value) const noexcept override;
    void synchronize_event(
        uint64_t handle, uint64_t fence_value) noexcept override;

    [[nodiscard]] ResourceCreationInfo create_mesh(
        const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_accel(
        const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    void set_name(
        Resource::Tag resource_tag, uint64_t resource_handle,
        luisa::string_view name) noexcept override;
};

}// namespace luisa::compute::simd
