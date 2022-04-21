//
// Created by Mike Smith on 2022/2/7.
//

#include <backends/ispc/ispc_stream.h>
#include <backends/ispc/ispc_device.h>
#include <backends/ispc/ispc_event.h>
#include <backends/ispc/ispc_shader.h>
#include <backends/ispc/ispc_mesh.h>
#include <backends/ispc/ispc_accel.h>
#include <backends/ispc/ispc_texture.h>
#include <backends/ispc/ispc_bindless_array.h>

namespace luisa::compute::ispc {

void *ISPCDevice::native_handle() const noexcept {
    return reinterpret_cast<void *>(reinterpret_cast<uint64_t>(this));
}

uint64_t ISPCDevice::create_buffer(size_t size_bytes) noexcept {
    return reinterpret_cast<uint64_t>(luisa::allocate<std::byte>(size_bytes));
}

void ISPCDevice::destroy_buffer(uint64_t handle) noexcept {
    luisa::deallocate(reinterpret_cast<std::byte *>(handle));
}

void *ISPCDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

uint64_t ISPCDevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    auto texture = luisa::new_with_allocator<ISPCTexture>(
        format, dimension, make_uint3(width, height, depth), mipmap_levels);
    return reinterpret_cast<uint64_t>(texture);
}

void ISPCDevice::destroy_texture(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<ISPCTexture *>(handle));
}

void *ISPCDevice::texture_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

uint64_t ISPCDevice::create_bindless_array(size_t size) noexcept {
    auto array = luisa::new_with_allocator<ISPCBindlessArray>(size);
    return reinterpret_cast<uint64_t>(array);
}

void ISPCDevice::destroy_bindless_array(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<ISPCBindlessArray *>(handle));
}

void ISPCDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    reinterpret_cast<ISPCBindlessArray *>(array)->emplace_buffer(
        index, reinterpret_cast<const void *>(handle), offset_bytes);
}

void ISPCDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    reinterpret_cast<ISPCBindlessArray *>(array)->emplace_tex2d(
        index, reinterpret_cast<const ISPCTexture *>(handle), sampler);
}

void ISPCDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    reinterpret_cast<ISPCBindlessArray *>(array)->emplace_tex3d(
        index, reinterpret_cast<const ISPCTexture *>(handle), sampler);
}

bool ISPCDevice::is_resource_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return reinterpret_cast<ISPCBindlessArray *>(array)->uses_resource(handle);
}

void ISPCDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<ISPCBindlessArray *>(array)->remove_buffer(index);
}

void ISPCDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<ISPCBindlessArray *>(array)->remove_tex2d(index);
}

void ISPCDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<ISPCBindlessArray *>(array)->remove_tex3d(index);
}

uint64_t ISPCDevice::create_stream() noexcept {
    return reinterpret_cast<uint64_t>(luisa::new_with_allocator<ISPCStream>());
}

void ISPCDevice::destroy_stream(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<ISPCStream *>(handle));
}

void ISPCDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    reinterpret_cast<ISPCStream *>(stream_handle)->synchronize();
}

void ISPCDevice::dispatch(uint64_t stream_handle, const CommandList &list) noexcept {
    auto stream = reinterpret_cast<ISPCStream *>(stream_handle);
    stream->dispatch(list);
}

void ISPCDevice::dispatch(uint64_t stream_handle, move_only_function<void()> &&func) noexcept {
    auto stream = reinterpret_cast<ISPCStream *>(stream_handle);
    stream->dispatch(std::move(func));
}

void *ISPCDevice::stream_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

uint64_t ISPCDevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    auto shader = luisa::new_with_allocator<ISPCShader>(context(), kernel);
    return reinterpret_cast<uint64_t>(shader);
}

void ISPCDevice::destroy_shader(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<ISPCShader *>(handle));
}

uint64_t ISPCDevice::create_event() noexcept {
    return reinterpret_cast<uint64_t>(luisa::new_with_allocator<ISPCEvent>());
}

void ISPCDevice::destroy_event(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<ISPCEvent *>(handle));
}

void ISPCDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    reinterpret_cast<ISPCStream *>(stream_handle)->signal(reinterpret_cast<ISPCEvent *>(handle));
}

void ISPCDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    reinterpret_cast<ISPCStream *>(stream_handle)->wait(reinterpret_cast<ISPCEvent *>(handle));
}

void ISPCDevice::synchronize_event(uint64_t handle) noexcept {
    reinterpret_cast<ISPCEvent *>(handle)->wait();
}

uint64_t ISPCDevice::create_mesh(
    uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept {
    auto mesh = luisa::new_with_allocator<ISPCMesh>(
        _rtc_device, hint,
        v_buffer, v_offset, v_stride, v_count,
        t_buffer, t_offset, t_count);
    return reinterpret_cast<uint64_t>(mesh);
}

void ISPCDevice::destroy_mesh(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<ISPCMesh *>(handle));
}

uint64_t ISPCDevice::create_accel(AccelUsageHint hint) noexcept {
    auto accel = luisa::new_with_allocator<ISPCAccel>(_rtc_device, hint);
    return reinterpret_cast<uint64_t>(accel);
}

void ISPCDevice::destroy_accel(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<ISPCAccel *>(handle));
}

ISPCDevice::ISPCDevice(const Context &ctx) noexcept
    : Device::Interface{ctx}, _rtc_device{rtcNewDevice(nullptr)} {}

ISPCDevice::~ISPCDevice() noexcept { rtcReleaseDevice(_rtc_device); }

uint64_t ISPCDevice::create_swap_chain(
    uint64_t window_handle, uint64_t stream_handle, uint width, uint height,
    bool allow_hdr, uint back_buffer_count) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void ISPCDevice::destroy_swap_chain(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

PixelStorage ISPCDevice::swap_chain_pixel_storage(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void ISPCDevice::present_display_in_stream(
    uint64_t stream_handle, uint64_t swap_chain_handle, uint64_t image_handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

}// namespace luisa::compute::ispc

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, std::string_view /* properties */) noexcept {
    return luisa::new_with_allocator<luisa::compute::ispc::ISPCDevice>(ctx);
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    luisa::delete_with_allocator(device);
}
