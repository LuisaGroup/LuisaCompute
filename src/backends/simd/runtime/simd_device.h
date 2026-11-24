#pragma once

#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/device.h>

#include "../../common/default_binary_io.h"
#include "simd_embree.h"

namespace luisa::compute {
class EmbreeAccelArchiveExt;
}

namespace luisa::compute::simd {

class SIMDThreadPool;

class SIMDDevice final : public DeviceInterface {

private:
#if defined(__aarch64__) || defined(_M_ARM64)
    uint _warp_width{1u};
#else
    uint _warp_width{8u};
#endif
    RTCDevice _rtc_device{nullptr};
    luisa::unique_ptr<DefaultBinaryIO> _default_io{};
    const BinaryIO *_io{nullptr};
    std::unique_ptr<SIMDThreadPool> _thread_pool;
    mutable luisa::spin_mutex _ext_mutex;
    luisa::unique_ptr<EmbreeAccelArchiveExt> _accel_archive_ext;

public:
    explicit SIMDDevice(Context &&context, const DeviceConfig *config) noexcept;
    ~SIMDDevice() noexcept override;

    [[nodiscard]] auto binary_io() const noexcept { return _io; }
    [[nodiscard]] void *native_handle() const noexcept override;
    [[nodiscard]] uint compute_warp_size() const noexcept override;
    [[nodiscard]] size_t compute_max_shared_memory_size() const noexcept override;
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
    [[nodiscard]] ShaderCreationInfo create_tile_kernel(
        const ShaderOption &option, const tile::Function &kernel,
        const tile::CompileOptions &tile_options, tile::KernelMetadata &metadata) noexcept override;
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
    [[nodiscard]] ResourceCreationInfo create_curve(
        const AccelOption &option) noexcept override;
    void destroy_curve(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_motion_instance(
        const AccelMotionOption &option) noexcept override;
    void destroy_motion_instance(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_accel(
        const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    [[nodiscard]] DeviceExtension *extension(luisa::string_view name) noexcept override;
    void set_name(
        Resource::Tag resource_tag, uint64_t resource_handle,
        luisa::string_view name) noexcept override;
};

}// namespace luisa::compute::simd
