#pragma once
#include <vstl/Common.h>
#include <runtime/ext/dx_device.h>
#include <DXRuntime/Device.h>
using namespace luisa::compute;
namespace toolhub::directx {
using LCDeviceInterface = luisa::compute::DxDevice;
class LCDevice : public LCDeviceInterface, public vstd::IOperatorNewBase {
public:
    Device nativeDevice;
    static constexpr size_t maxAllocatorCount = 2;
    //std::numeric_limits<size_t>::max();
    LCDevice(const Context &ctx);
    void *native_handle() const noexcept override;

    // buffer
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    void *buffer_native_handle(uint64_t handle) const noexcept override;

    // texture
    uint64_t create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;
    void *texture_native_handle(uint64_t handle) const noexcept override;

    // bindless array
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
    void present_display_stream(uint64_t stream_handle, uint64_t texture) noexcept override;

    // stream
    uint64_t create_stream() noexcept override;
    uint64_t create_display_stream(
        uint64 window_handle,
        uint width,
        uint height) noexcept override;

    void destroy_stream(uint64_t handle) noexcept override;
    void destroy_display_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void synchronize_display_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList const &) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList const &, luisa::move_only_function<void()> &&callback) noexcept override;
    void dispatch(uint64_t stream_handle, vstd::span<const CommandList> lists) noexcept override;
    void dispatch(uint64_t stream_handle, vstd::span<const CommandList> lists, luisa::move_only_function<void()> &&callback) noexcept override;
    void *stream_native_handle(uint64_t handle) const noexcept override;

    // kernel
    uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // event
    uint64_t create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;
    void signal_display_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_display_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    // accel
    uint64_t create_mesh(
        uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;

    uint64_t create_accel(AccelBuildHint hint) noexcept override;
    void emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, luisa::float4x4 transform, bool visible) noexcept override;
    void pop_back_instance_from_accel(uint64_t accel) noexcept override;
    void set_instance_mesh_in_accel(uint64_t accel, uint64_t index, uint64_t mesh) noexcept override;
    bool is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept override;
    bool is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept override;
    uint64_t get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    uint64_t get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
};
}// namespace toolhub::directx