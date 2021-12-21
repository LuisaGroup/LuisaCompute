#pragma once

#include <runtime/device.h>
#include <vstl/ThreadPool.h>

using namespace luisa;
using namespace luisa::compute;

namespace lc::ispc {
class ISPCDevice final : public Device::Interface {

private:
    ThreadPool tPool;
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

public:
    uint64_t create_bindless_array(size_t size) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override;
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    bool is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    bool is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;

    // stream
    uint64_t create_stream() noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList) noexcept override;
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

    // accel
    void destroy_mesh(uint64_t handle) noexcept override;
    uint64_t create_mesh(
        uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept override;
    uint64_t get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    uint64_t get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    uint64_t create_accel(AccelBuildHint hint) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
    void emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, float4x4 transform, bool visible) noexcept override;
    void set_instance_transform_in_accel(uint64_t accel, size_t index, float4x4 transform) noexcept override;
    bool is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept override;
    bool is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept override;

public:
    ISPCDevice(const luisa::compute::Context &ctx, uint32_t index /* TODO */) noexcept
        : Device::Interface(ctx), tPool(std::thread::hardware_concurrency() + 1u) {}
    ~ISPCDevice() noexcept override = default;
    void pop_back_instance_from_accel(uint64_t accel) noexcept override;
    void set_instance_in_accel(uint64_t accel, size_t index, uint64_t mesh, float4x4 transform, bool visible) noexcept override;
    void set_instance_visibility_in_accel(uint64_t accel, size_t index, bool visible) noexcept override;
};

}// namespace lc::ispc
