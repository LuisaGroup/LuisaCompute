//
// Created by Mike Smith on 2023/4/8.
//

#include <core/logging.h>

#ifdef LUISA_ENABLE_IR
#include <backends/metal/metal_codegen_ir.h>
#endif

#include <backends/metal/metal_codegen_ast.h>
#include <backends/metal/metal_compiler.h>
#include <backends/metal/metal_texture.h>
#include <backends/metal/metal_stream.h>
#include <backends/metal/metal_event.h>
#include <backends/metal/metal_bindless_array.h>
#include <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalDevice::MetalDevice(Context &&ctx, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(ctx)}, _io{nullptr} {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto device_index = config == nullptr ? 0u : config->device_index;
    auto all_devices = NS::TransferPtr(MTL::CopyAllDevices());
    auto device_count = all_devices->count();
    LUISA_ASSERT(device_index < device_count,
                 "Metal device index out of range.");
    _handle = all_devices->object<MTL::Device>(device_index);

    // create a default binary IO if none is provided
    if (config == nullptr || config->binary_io == nullptr) {
        _default_io = luisa::make_unique<DefaultBinaryIO>(context(), "metal");
        _io = _default_io.get();
    } else {
        _io = config->binary_io;
    }

    // create a compiler
    _compiler = luisa::make_unique<MetalCompiler>(this);

    // TODO: load built-in kernels
}

MetalDevice::~MetalDevice() noexcept {
    _handle->release();
}

void *MetalDevice::native_handle() const noexcept {
    return _handle;
}

[[nodiscard]] inline auto create_device_buffer(MTL::Device *device, size_t element_stride, size_t element_count) noexcept {
    auto buffer_size = element_stride * element_count;
    auto options = MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
    auto buffer = device->newBuffer(buffer_size, options);
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer;
    info.element_stride = element_stride;
    info.total_size_bytes = buffer_size;
    MTL::AccelerationStructureInstanceDescriptor desc{};
    return info;
}

BufferCreationInfo MetalDevice::create_buffer(const Type *element, size_t elem_count) noexcept {
    auto elem_size = MetalCodegenAST::type_size_bytes(element);
    return create_device_buffer(_handle, elem_size, elem_count);
}

BufferCreationInfo MetalDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
#ifdef LUISA_ENABLE_IR
    auto elem_size = MetalCodegenIR::type_size_bytes(element->get());
    return create_device_buffer(_handle, elem_size, elem_count);
#else
    LUISA_ERROR_WITH_LOCATION("IR is not enabled.");
#endif
}

void MetalDevice::destroy_buffer(uint64_t handle) noexcept {
    auto buffer = reinterpret_cast<MTL::Buffer *>(handle);
    buffer->release();
}

ResourceCreationInfo MetalDevice::create_texture(PixelFormat format, uint dimension,
                                                 uint width, uint height, uint depth,
                                                 uint mipmap_levels) noexcept {
    auto texture = new_with_allocator<MetalTexture>(
        _handle, format, dimension, width, height, depth, mipmap_levels);
    ResourceCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(texture);
    info.native_handle = texture;
    return info;
}

void MetalDevice::destroy_texture(uint64_t handle) noexcept {
    auto texture = reinterpret_cast<MetalTexture *>(handle);
    delete_with_allocator(texture);
}

ResourceCreationInfo MetalDevice::create_bindless_array(size_t size) noexcept {
    auto array = new_with_allocator<MetalBindlessArray>(this, size);
    ResourceCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(array);
    info.native_handle = array;
    return info;
}

void MetalDevice::destroy_bindless_array(uint64_t handle) noexcept {
    auto array = reinterpret_cast<MetalBindlessArray *>(handle);
    delete_with_allocator(array);
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
    auto event = new_with_allocator<MetalEvent>(_handle);
    ResourceCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(event);
    info.native_handle = event;
    return info;
}

void MetalDevice::destroy_event(uint64_t handle) noexcept {
    auto event = reinterpret_cast<MetalEvent *>(handle);
    delete_with_allocator(event);
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto event = reinterpret_cast<MetalEvent *>(handle);
    auto stream = reinterpret_cast<MetalStream *>(stream_handle);
    stream->signal(event);
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto event = reinterpret_cast<MetalEvent *>(handle);
    auto stream = reinterpret_cast<MetalStream *>(stream_handle);
    stream->wait(event);
}

void MetalDevice::synchronize_event(uint64_t handle) noexcept {
    auto event = reinterpret_cast<MetalEvent *>(handle);
    event->synchronize();
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
    LUISA_WARNING_WITH_LOCATION("Device extension \"{}\" is not supported on Metal.", name);
}

void MetalDevice::set_name(luisa::compute::Resource::Tag resource_tag,
                           uint64_t resource_handle, luisa::string_view name) noexcept {
    switch (resource_tag) {
        case Resource::Tag::BUFFER: {
            auto buffer = reinterpret_cast<MTL::Buffer *>(resource_handle);
            if (name.empty()) {
                buffer->setLabel(nullptr);
            } else {
                luisa::string mtl_name{name};
                auto autorelease_pool = NS::AutoreleasePool::alloc()->init();
                buffer->setLabel(NS::String::string(mtl_name.c_str(), NS::UTF8StringEncoding));
                autorelease_pool->release();
            }
            break;
        }
        case Resource::Tag::TEXTURE: {
            auto texture = reinterpret_cast<MetalTexture *>(resource_handle);
            texture->set_name(name);
            break;
        }
        case Resource::Tag::BINDLESS_ARRAY: break;
        case Resource::Tag::MESH: break;
        case Resource::Tag::PROCEDURAL_PRIMITIVE: break;
        case Resource::Tag::ACCEL: break;
        case Resource::Tag::STREAM: break;
        case Resource::Tag::EVENT: {
            auto event = reinterpret_cast<MetalEvent *>(resource_handle);
            event->set_name(name);
            break;
        }
        case Resource::Tag::SHADER: break;
        case Resource::Tag::RASTER_SHADER: break;
        case Resource::Tag::SWAP_CHAIN: break;
        case Resource::Tag::DEPTH_BUFFER: break;
    }
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
