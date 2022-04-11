//
// Created by Mike on 2021/12/6.
//

#include <backends/remote/remote_device.h>

namespace luisa::compute::remote {

void *RemoteDevice::native_handle() const noexcept {
    return nullptr;
}

uint64_t RemoteDevice::create_buffer(size_t size_bytes) noexcept {
    return 0;
}

void RemoteDevice::destroy_buffer(uint64_t handle) noexcept {
}

void *RemoteDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}

uint64_t RemoteDevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    return 0;
}

void RemoteDevice::destroy_texture(uint64_t handle) noexcept {
}

void *RemoteDevice::texture_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}

uint64_t RemoteDevice::create_bindless_array(size_t size) noexcept {
    return 0;
}

void RemoteDevice::destroy_bindless_array(uint64_t handle) noexcept {
}

void RemoteDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
}

void RemoteDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
}

void RemoteDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
}

bool RemoteDevice::is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return false;
}

bool RemoteDevice::is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return false;
}

void RemoteDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
}

void RemoteDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
}

void RemoteDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
}

uint64_t RemoteDevice::create_stream() noexcept {
    return 0;
}

void RemoteDevice::destroy_stream(uint64_t handle) noexcept {
}

void RemoteDevice::synchronize_stream(uint64_t stream_handle) noexcept {
}

void RemoteDevice::dispatch(uint64_t stream_handle, const CommandList &list) noexcept {
}

void *RemoteDevice::stream_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}

uint64_t RemoteDevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    return 0;
}

void RemoteDevice::destroy_shader(uint64_t handle) noexcept {
}

uint64_t RemoteDevice::create_event() noexcept {
    return 0;
}

void RemoteDevice::destroy_event(uint64_t handle) noexcept {
}

void RemoteDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
}

void RemoteDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
}

void RemoteDevice::synchronize_event(uint64_t handle) noexcept {
}

uint64_t RemoteDevice::create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept {
    return 0;
}

void RemoteDevice::destroy_mesh(uint64_t handle) noexcept {
}

uint64_t RemoteDevice::create_accel(AccelUsageHint hint) noexcept {
    return 0;
}

uint64_t RemoteDevice::get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
    return 0;
}

uint64_t RemoteDevice::get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
    return 0;
}

void RemoteDevice::destroy_accel(uint64_t handle) noexcept {
}

RemoteDevice::RemoteDevice(const Context &ctx, std::string_view properties) noexcept
    : Device::Interface{ctx} {

    LUISA_INFO_WITH_LOCATION(
        "Creating remote device with properties: {}.",
        properties);
}

void RemoteDevice::dispatch(uint64_t stream_handle, move_only_function<void()> &&func) noexcept {
}

uint64_t RemoteDevice::create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, uint back_buffer_size) noexcept {
    return 0;
}

void RemoteDevice::destroy_swap_chain(uint64_t handle) noexcept {
}

PixelStorage RemoteDevice::swap_chain_pixel_storage(uint64_t handle) noexcept {
    return PixelStorage::BYTE2;
}

void RemoteDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
}

}

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, std::string_view properties) noexcept {
    return luisa::new_with_allocator<luisa::compute::remote::RemoteDevice>(ctx, properties);// TODO: decode properties
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    luisa::delete_with_allocator(device);
}
