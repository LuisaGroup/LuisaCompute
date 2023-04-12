//
// Created by Mike Smith on 2023/4/8.
//

#include <core/logging.h>
#include <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalDevice::MetalDevice(Context &&ctx, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(ctx)},
      _default_io{config == nullptr || config->binary_io == nullptr ?
                      luisa::make_unique<DefaultBinaryIO>(context(), "metal") :
                      nullptr},
      _io{config == nullptr || config->binary_io == nullptr ?
              _default_io.get() :
              config->binary_io} {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto device_index = config == nullptr ? 0u : config->device_index;
    auto all_devices = NS::TransferPtr(MTL::CopyAllDevices());
    auto device_count = all_devices->count();
    LUISA_ASSERT(device_index < device_count,
                 "Metal device index out of range.");
    _handle = all_devices->object<MTL::Device>(device_index);
}

MetalDevice::~MetalDevice() noexcept {
    _handle->release();
}

void *MetalDevice::native_handle() const noexcept {
    return _handle;
}

BufferCreationInfo MetalDevice::create_buffer(const Type *element, size_t elem_count) noexcept {
    return BufferCreationInfo();
}

BufferCreationInfo MetalDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
    return BufferCreationInfo();
}

void MetalDevice::destroy_buffer(uint64_t handle) noexcept {

}

ResourceCreationInfo MetalDevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_texture(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_bindless_array(size_t size) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_bindless_array(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_stream(StreamTag stream_tag) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_stream(uint64_t handle) noexcept {
}

void MetalDevice::synchronize_stream(uint64_t stream_handle) noexcept {
}

void MetalDevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
}

SwapChainCreationInfo MetalDevice::create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, bool vsync, uint back_buffer_size) noexcept {
    return SwapChainCreationInfo();
}

void MetalDevice::destroy_swap_chain(uint64_t handle) noexcept {
}

void MetalDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
}

ShaderCreationInfo MetalDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    return ShaderCreationInfo();
}

ShaderCreationInfo MetalDevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    return ShaderCreationInfo();
}

ShaderCreationInfo MetalDevice::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    return ShaderCreationInfo();
}

Usage MetalDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return Usage::NONE;
}

void MetalDevice::destroy_shader(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_event() noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_event(uint64_t handle) noexcept {
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
}

void MetalDevice::synchronize_event(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_mesh(const AccelOption &option) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_mesh(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
}

ResourceCreationInfo MetalDevice::create_accel(const AccelOption &option) noexcept {
    return ResourceCreationInfo();
}

void MetalDevice::destroy_accel(uint64_t handle) noexcept {
}

string MetalDevice::query(luisa::string_view property) noexcept {
    return DeviceInterface::query(property);
}

DeviceExtension *MetalDevice::extension(luisa::string_view name) noexcept {
    return DeviceInterface::extension(name);
}

void MetalDevice::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {
}

}// namespace luisa::compute::metal

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    return ::luisa::new_with_allocator<::luisa::compute::metal::MetalDevice>(std::move(ctx), config);
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    auto p_device = dynamic_cast<::luisa::compute::metal::MetalDevice *>(device);
    LUISA_ASSERT(p_device != nullptr, "Invalid device.");
    ::luisa::delete_with_allocator(p_device);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    auto all_devices = NS::TransferPtr(MTL::CopyAllDevices());
    if (auto n = all_devices->count()) {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        names.reserve(n);
        for (auto i = 0u; i < n; i++) {
            auto device = all_devices->object<MTL::Device>(i);
            names.emplace_back(device->name()->cString(
                NS::StringEncoding::UTF8StringEncoding));
            device->release();
        }
    }
}
