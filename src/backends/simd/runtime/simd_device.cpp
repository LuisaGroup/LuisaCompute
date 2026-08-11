#include "simd_device.h"

#include <luisa/ast/type_registry.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/core/stl/memory.h>

#include "simd_buffer.h"
#include "simd_event.h"
#include "simd_shader.h"
#include "simd_stream.h"

namespace luisa::compute::simd {

SIMDDevice::SIMDDevice(
    Context &&context, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(context)} {
    static_cast<void>(config);
}

void *SIMDDevice::native_handle() const noexcept {
    return const_cast<SIMDDevice *>(this);
}

uint SIMDDevice::compute_warp_size() const noexcept { return 8u; }

uint64_t SIMDDevice::memory_granularity() const noexcept { return 1u; }

BufferCreationInfo SIMDDevice::create_buffer(
    const Type *element, size_t elem_count,
    void *external_memory) noexcept {
    BufferCreationInfo info{};
    info.element_stride = element == Type::of<void>() ?
                              1u : element->size();
    info.total_size_bytes = info.element_stride * elem_count;
    auto *buffer = external_memory == nullptr ?
        luisa::new_with_allocator<SIMDBuffer>(info.total_size_bytes) :
        luisa::new_with_allocator<SIMDBuffer>(
            static_cast<std::byte *>(external_memory),
            info.total_size_bytes);
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->data();
    return info;
}

void SIMDDevice::destroy_buffer(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDBuffer *>(handle));
}

ResourceCreationInfo SIMDDevice::create_texture(
    PixelFormat, uint, uint, uint, uint, uint, void *, bool, bool) noexcept {
    return ResourceCreationInfo::make_invalid();
}

void SIMDDevice::destroy_texture(uint64_t) noexcept {}

ResourceCreationInfo SIMDDevice::create_bindless_array(
    size_t, BindlessSlotType) noexcept {
    return ResourceCreationInfo::make_invalid();
}

void SIMDDevice::destroy_bindless_array(uint64_t) noexcept {}

ResourceCreationInfo SIMDDevice::create_stream(StreamTag) noexcept {
    auto *stream = luisa::new_with_allocator<SIMDStream>();
    return {
        .handle = reinterpret_cast<uint64_t>(stream),
        .native_handle = stream->native_handle(),
    };
}

void SIMDDevice::destroy_stream(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDStream *>(handle));
}

void SIMDDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    reinterpret_cast<SIMDStream *>(stream_handle)->synchronize();
}

void SIMDDevice::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {
    reinterpret_cast<SIMDStream *>(stream_handle)->dispatch(
        std::move(list));
}

void SIMDDevice::set_stream_log_callback(
    uint64_t stream_handle,
    const StreamLogCallback &callback) noexcept {
    reinterpret_cast<SIMDStream *>(stream_handle)->set_log_callback(callback);
}

SwapchainCreationInfo SIMDDevice::create_swapchain(
    const SwapchainOption &, uint64_t) noexcept {
    SwapchainCreationInfo info{};
    info.invalidate();
    return info;
}

void SIMDDevice::destroy_swapchain(uint64_t) noexcept {}

void SIMDDevice::present_display_in_stream(
    uint64_t, uint64_t, uint64_t) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "The SIMD backend does not provide a display swapchain yet.");
}

ShaderCreationInfo SIMDDevice::create_shader(
    const ShaderOption &option, Function kernel) noexcept {
    Clock clock;
    auto block_size = kernel.block_size();
    auto *shader = luisa::new_with_allocator<SIMDShader>(
        option, kernel);
    LUISA_VERBOSE(
        "SIMD shader compilation took {} ms.", clock.toc());
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(shader);
    info.native_handle = reinterpret_cast<void *>(shader->native_handle());
    info.block_size = block_size;
    return info;
}

ShaderCreationInfo SIMDDevice::load_shader(
    luisa::string_view, luisa::span<const Type *const>) noexcept {
    return ShaderCreationInfo::make_invalid();
}

Usage SIMDDevice::shader_argument_usage(
    uint64_t handle, size_t index) noexcept {
    return reinterpret_cast<SIMDShader *>(handle)->argument_usage(index);
}

void SIMDDevice::destroy_shader(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDShader *>(handle));
}

ResourceCreationInfo SIMDDevice::create_event() noexcept {
    auto *event = luisa::new_with_allocator<SIMDEvent>();
    return {
        .handle = reinterpret_cast<uint64_t>(event),
        .native_handle = event->native_handle(),
    };
}

void SIMDDevice::destroy_event(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDEvent *>(handle));
}

void SIMDDevice::signal_event(
    uint64_t handle, uint64_t, uint64_t fence_value) noexcept {
    reinterpret_cast<SIMDEvent *>(handle)->signal(fence_value);
}

void SIMDDevice::wait_event(
    uint64_t handle, uint64_t, uint64_t fence_value) noexcept {
    reinterpret_cast<SIMDEvent *>(handle)->wait(fence_value);
}

bool SIMDDevice::is_event_completed(
    uint64_t handle, uint64_t fence_value) const noexcept {
    return reinterpret_cast<SIMDEvent *>(handle)->is_completed(fence_value);
}

void SIMDDevice::synchronize_event(
    uint64_t handle, uint64_t fence_value) noexcept {
    reinterpret_cast<SIMDEvent *>(handle)->wait(fence_value);
}

ResourceCreationInfo SIMDDevice::create_mesh(
    const AccelOption &) noexcept {
    return ResourceCreationInfo::make_invalid();
}

void SIMDDevice::destroy_mesh(uint64_t) noexcept {}

ResourceCreationInfo SIMDDevice::create_procedural_primitive(
    const AccelOption &) noexcept {
    return ResourceCreationInfo::make_invalid();
}

void SIMDDevice::destroy_procedural_primitive(uint64_t) noexcept {}

ResourceCreationInfo SIMDDevice::create_accel(
    const AccelOption &) noexcept {
    return ResourceCreationInfo::make_invalid();
}

void SIMDDevice::destroy_accel(uint64_t) noexcept {}

void SIMDDevice::set_name(
    Resource::Tag, uint64_t, luisa::string_view) noexcept {}

}// namespace luisa::compute::simd

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(
    luisa::compute::Context &&context,
    const luisa::compute::DeviceConfig *config) noexcept {
    return luisa::new_with_allocator<luisa::compute::simd::SIMDDevice>(
        std::move(context), config);
}

LUISA_EXPORT_API void destroy(
    luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}

LUISA_EXPORT_API void backend_device_names(
    luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    names.emplace_back(luisa::cpu_name());
}

#include "../../common/export_version.inl.h"
