//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#include <runtime/device.h>
#include <runtime/context.h>

namespace luisa::compute {

class FakeDevice : public Device::Interface {

private:
    uint64_t _handle{0u};

public:
    explicit FakeDevice(const Context &ctx) noexcept
        : Device::Interface{ctx} {}
    ~FakeDevice() noexcept override = default;
    uint64_t create_buffer(size_t) noexcept override { return _handle++; }
    void destroy_buffer(uint64_t) noexcept override {}
    uint64_t create_stream() noexcept override { return _handle++; }
    void destroy_stream(uint64_t) noexcept override {}
    void synchronize_stream(uint64_t stream_handle) noexcept override {}
    void dispatch(uint64_t stream_handle, const CommandList &list) noexcept override {}
    uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept override { return _handle++; }
    void destroy_shader(uint64_t handle) noexcept override {}
    uint64_t create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept override { return _handle++; }
    void destroy_texture(uint64_t handle) noexcept override {}
    uint64_t create_event() noexcept override { return _handle++; }
    void synchronize_event(uint64_t handle) noexcept override {}
    void destroy_event(uint64_t handle) noexcept override {}
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override {}
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override {}
    uint64_t create_mesh(
        uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept override { return 0; }
    void destroy_mesh(uint64_t handle) noexcept override {}
    uint64_t create_accel(AccelBuildHint hint) noexcept override { return _handle++; }
    void destroy_accel(uint64_t handle) noexcept override {}
    uint64_t create_bindless_array(size_t size) noexcept override { return _handle++; }
    void destroy_bindless_array(uint64_t handle) noexcept override {}
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override {}
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override {}
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override {}
    void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept override {}
    void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept override {}
    void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept override {}
    bool is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override { return false; }
    bool is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override { return false; }
    void emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, float4x4 transform, bool) noexcept override {}
    void pop_back_instance_from_accel(uint64_t accel) noexcept override {}
    bool is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept override { return false; }
    bool is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept override { return false; }
    uint64_t get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept override { return 0; }
    uint64_t get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept override { return 0; }
    [[nodiscard]] static auto create(const Context &ctx) noexcept {
        auto deleter = [](Device::Interface *d) { delete d; };
        return Device{Device::Handle{new FakeDevice{ctx}, deleter}};
    }
    void *buffer_native_handle(uint64_t handle) const noexcept override { return nullptr; }
    void *texture_native_handle(uint64_t handle) const noexcept override { return nullptr; }
    void *native_handle() const noexcept override { return nullptr; }
    void *stream_native_handle(uint64_t) const noexcept override { return nullptr; }
};

}// namespace luisa::compute
