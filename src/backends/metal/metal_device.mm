//
// Created by Mike Smith on 2021/3/17.
//

#if !__has_feature(objc_arc)
#error Compiling the Metal backend with ARC off.
#endif

#import <chrono>
#import <numeric>

#import <core/platform.h>
#import <core/hash.h>
#import <core/clock.h>
#import <runtime/context.h>
#import <runtime/bindless_array.h>

#import <backends/metal/metal_device.h>
#import <backends/metal/metal_bindless_array.h>
#import <backends/metal/metal_command_encoder.h>

namespace luisa::compute::metal {

uint64_t MetalDevice::create_buffer(size_t size_bytes) noexcept {
    Clock clock;
    auto buffer = [_handle newBufferWithLength:size_bytes
                                       options:MTLResourceStorageModePrivate |
                                               MTLResourceHazardTrackingModeUntracked];
    LUISA_VERBOSE_WITH_LOCATION(
        "Created buffer with size {} in {} ms.",
        size_bytes, clock.toc());
    return reinterpret_cast<uint64_t>((__bridge_retained void *)buffer);
}

void MetalDevice::destroy_buffer(uint64_t handle) noexcept {
    auto ptr = reinterpret_cast<void *>(handle);
    auto buffer = (__bridge_transfer id<MTLBuffer>)ptr;
    LUISA_VERBOSE_WITH_LOCATION("Destroyed buffer #{}.", handle);
}

uint64_t MetalDevice::create_stream() noexcept {
    Clock clock;
    auto max_command_buffer_count = _handle.isLowPower ? 8u : 16u;
    auto stream = new_with_allocator<MetalStream>(_handle, max_command_buffer_count);
    LUISA_VERBOSE_WITH_LOCATION("Created stream in {} ms.", clock.toc());
    return reinterpret_cast<uint64_t>(stream);
}

void MetalDevice::destroy_stream(uint64_t handle) noexcept {
    auto s = reinterpret_cast<MetalStream *>(handle);
    delete_with_allocator(s);
    LUISA_VERBOSE_WITH_LOCATION("Destroyed stream #{}.", handle);
}

MetalDevice::MetalDevice(const Context &ctx, uint32_t index) noexcept
    : Device::Interface{ctx} {

    auto devices = MTLCopyAllDevices();
    if (devices.count == 0u) {
        LUISA_ERROR_WITH_LOCATION(
            "No available devices found for Metal backend.");
    }
    if (auto count = devices.count; index >= count) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid Metal device index {}. Limited to max index {}.",
            index, count - 1u);
        index = static_cast<uint>(count - 1u);
    }
    luisa::vector<id<MTLDevice>> sorted_devices;
    sorted_devices.reserve(devices.count);
    for (id<MTLDevice> d in devices) { sorted_devices.emplace_back(d); }
    std::sort(sorted_devices.begin(), sorted_devices.end(), [](id<MTLDevice> lhs, id<MTLDevice> rhs) noexcept {
        if (lhs.isLowPower == rhs.isLowPower) { return lhs.registryID < rhs.registryID; }
        return static_cast<bool>(rhs.isLowPower);
    });
    _handle = sorted_devices[index];
    LUISA_INFO(
        "Created Metal device #{} with name: {}.",
        index, [_handle.name cStringUsingEncoding:NSUTF8StringEncoding]);
    _compiler = luisa::make_unique<MetalCompiler>(this);

    // initialize bindless array encoder
    // TODO: use SoA
    static constexpr auto src = @"#include <metal_stdlib>\n"
                                 "struct alignas(16) BindlessItem {"
                                 "  device const void *buffer;\n"
                                 "  metal::uint sampler2d;\n"
                                 "  metal::uint sampler3d;\n"
                                 "  metal::texture2d<float> tex2d;\n"
                                 "  metal::texture3d<float> tex3d;\n"
                                 "};\n"
                                 "[[kernel]] void k(device const BindlessItem *array) {}\n";
    NSError *error;
    auto library = [_handle newLibraryWithSource:src options:nullptr error:&error];
    if (error) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create bindless array encoder: {}.",
            [[error description] cStringUsingEncoding:NSUTF8StringEncoding]);
    }
    auto function = [library newFunctionWithName:@"k"];
    _bindless_array_encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (_bindless_array_encoder.encodedLength != MetalBindlessArray::slot_size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid bindless array buffer encoded size: {} (expected {}).",
            _bindless_array_encoder.encodedLength,
            MetalBindlessArray::buffer_slot_size);
    }
}

MetalDevice::~MetalDevice() noexcept {
    LUISA_INFO(
        "Destroyed Metal device with name: {}.",
        [_handle.name cStringUsingEncoding:NSUTF8StringEncoding]);
    _handle = nullptr;
}

void MetalDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    reinterpret_cast<MetalStream *>(stream_handle)->synchronize();
}

id<MTLDevice> MetalDevice::handle() const noexcept {
    return _handle;
}

MetalShader MetalDevice::compiled_kernel(uint64_t handle) const noexcept {
    auto ptr = reinterpret_cast<void *>(handle);
    return MetalShader{(__bridge id<MTLComputePipelineState>)ptr};
}

uint64_t MetalDevice::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels) noexcept {

    Clock clock;

    auto desc = [[MTLTextureDescriptor alloc] init];
    switch (dimension) {
        case 2u: desc.textureType = MTLTextureType2D; break;
        case 3u: desc.textureType = MTLTextureType3D; break;
        default: LUISA_ERROR_WITH_LOCATION("Invalid image dimension {}.", dimension); break;
    }
    desc.width = width;
    desc.height = height;
    desc.depth = depth;
    switch (format) {
        case PixelFormat::R8SInt: desc.pixelFormat = MTLPixelFormatR8Sint; break;
        case PixelFormat::R8UInt: desc.pixelFormat = MTLPixelFormatR8Uint; break;
        case PixelFormat::R8UNorm: desc.pixelFormat = MTLPixelFormatR8Unorm; break;
        case PixelFormat::RG8SInt: desc.pixelFormat = MTLPixelFormatRG8Sint; break;
        case PixelFormat::RG8UInt: desc.pixelFormat = MTLPixelFormatRG8Uint; break;
        case PixelFormat::RG8UNorm: desc.pixelFormat = MTLPixelFormatRG8Unorm; break;
        case PixelFormat::RGBA8SInt: desc.pixelFormat = MTLPixelFormatRGBA8Sint; break;
        case PixelFormat::RGBA8UInt: desc.pixelFormat = MTLPixelFormatRGBA8Uint; break;
        case PixelFormat::RGBA8UNorm: desc.pixelFormat = MTLPixelFormatRGBA8Unorm; break;
        case PixelFormat::R16SInt: desc.pixelFormat = MTLPixelFormatR16Sint; break;
        case PixelFormat::R16UInt: desc.pixelFormat = MTLPixelFormatR16Uint; break;
        case PixelFormat::R16UNorm: desc.pixelFormat = MTLPixelFormatR16Unorm; break;
        case PixelFormat::RG16SInt: desc.pixelFormat = MTLPixelFormatRG16Sint; break;
        case PixelFormat::RG16UInt: desc.pixelFormat = MTLPixelFormatRG16Uint; break;
        case PixelFormat::RG16UNorm: desc.pixelFormat = MTLPixelFormatRG16Unorm; break;
        case PixelFormat::RGBA16SInt: desc.pixelFormat = MTLPixelFormatRGBA16Sint; break;
        case PixelFormat::RGBA16UInt: desc.pixelFormat = MTLPixelFormatRGBA16Uint; break;
        case PixelFormat::RGBA16UNorm: desc.pixelFormat = MTLPixelFormatRGBA16Unorm; break;
        case PixelFormat::R32SInt: desc.pixelFormat = MTLPixelFormatR32Sint; break;
        case PixelFormat::R32UInt: desc.pixelFormat = MTLPixelFormatR32Uint; break;
        case PixelFormat::RG32SInt: desc.pixelFormat = MTLPixelFormatRG32Sint; break;
        case PixelFormat::RG32UInt: desc.pixelFormat = MTLPixelFormatRG32Uint; break;
        case PixelFormat::RGBA32SInt: desc.pixelFormat = MTLPixelFormatRGBA32Sint; break;
        case PixelFormat::RGBA32UInt: desc.pixelFormat = MTLPixelFormatRGBA32Uint; break;
        case PixelFormat::R16F: desc.pixelFormat = MTLPixelFormatR16Float; break;
        case PixelFormat::RG16F: desc.pixelFormat = MTLPixelFormatRG16Float; break;
        case PixelFormat::RGBA16F: desc.pixelFormat = MTLPixelFormatRGBA16Float; break;
        case PixelFormat::R32F: desc.pixelFormat = MTLPixelFormatR32Float; break;
        case PixelFormat::RG32F: desc.pixelFormat = MTLPixelFormatRG32Float; break;
        case PixelFormat::RGBA32F: desc.pixelFormat = MTLPixelFormatRGBA32Float; break;
    }
    desc.allowGPUOptimizedContents = YES;
    desc.storageMode = MTLStorageModePrivate;
    desc.hazardTrackingMode = MTLHazardTrackingModeTracked;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.mipmapLevelCount = mipmap_levels;

    id<MTLTexture> texture = nullptr;
    MetalBindlessArray *heap = nullptr;
    texture = [_handle newTextureWithDescriptor:desc];

    if (texture == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to allocate texture with description {}.",
            [desc.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    LUISA_VERBOSE_WITH_LOCATION(
        "Created image (with {} mipmap{}) in {} ms.",
        mipmap_levels, mipmap_levels <= 1u ? "" : "s",
        clock.toc());

    return reinterpret_cast<uint64_t>((__bridge_retained void *)texture);
}

void MetalDevice::destroy_texture(uint64_t handle) noexcept {
    auto ptr = reinterpret_cast<void *>(handle);
    auto texture = (__bridge_transfer id<MTLTexture>)ptr;
    LUISA_VERBOSE_WITH_LOCATION("Destroyed image #{}.", handle);
}

uint64_t MetalDevice::create_event() noexcept {
    Clock clock;
    auto event = new_with_allocator<MetalEvent>([_handle newEvent]);
    LUISA_VERBOSE_WITH_LOCATION("Created event in {} ms.", clock.toc());
    return reinterpret_cast<uint64_t>(event);
}

void MetalDevice::destroy_event(uint64_t handle) noexcept {
    auto event = reinterpret_cast<MetalEvent *>(handle);
    delete_with_allocator(event);
    LUISA_VERBOSE_WITH_LOCATION("Destroyed event #{}.", handle);
}

void MetalDevice::synchronize_event(uint64_t handle) noexcept {
    reinterpret_cast<MetalEvent *>(handle)->synchronize();
}

void MetalDevice::dispatch(uint64_t stream_handle, const CommandList &list) noexcept {
    @autoreleasepool {
        auto s = reinterpret_cast<MetalStream *>(stream_handle);
        MetalCommandEncoder encoder{this, s};
        for (auto command : list) { command->accept(encoder); }
        s->dispatch(encoder.command_buffer());
    }
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    @autoreleasepool {
        auto e = reinterpret_cast<MetalEvent *>(handle);
        auto s = reinterpret_cast<MetalStream *>(stream_handle);
        auto command_buffer = s->command_buffer();
        e->signal(command_buffer);
        s->dispatch(command_buffer);
    }
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    @autoreleasepool {
        auto e = reinterpret_cast<MetalEvent *>(handle);
        auto s = reinterpret_cast<MetalStream *>(stream_handle);
        auto command_buffer = s->command_buffer();
        e->wait(command_buffer);
        s->dispatch(command_buffer);
    }
}

uint64_t MetalDevice::create_mesh(
    uint64_t v_buffer_handle, size_t v_offset, size_t v_stride, size_t,
    uint64_t t_buffer_handle, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept {

#ifdef LUISA_METAL_RAYTRACING_ENABLED
    check_raytracing_supported();
    Clock clock;
    auto v_buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(v_buffer_handle));
    auto t_buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(t_buffer_handle));
    auto mesh = new_with_allocator<MetalMesh>(
        v_buffer, v_offset, v_stride,
        t_buffer, v_offset, t_count, hint);
    LUISA_VERBOSE_WITH_LOCATION("Created mesh in {} ms.", clock.toc());
    return reinterpret_cast<uint64_t>(mesh);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

void MetalDevice::destroy_mesh(uint64_t handle) noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto mesh = reinterpret_cast<MetalMesh *>(handle);
    delete_with_allocator(mesh);
    LUISA_VERBOSE_WITH_LOCATION("Destroyed mesh #{}.", handle);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

uint64_t MetalDevice::get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto mesh = reinterpret_cast<MetalMesh *>(mesh_handle);
    return reinterpret_cast<uint64_t>((__bridge void *)(mesh->vertex_buffer()));
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

uint64_t MetalDevice::get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto mesh = reinterpret_cast<MetalMesh *>(mesh_handle);
    return reinterpret_cast<uint64_t>((__bridge void *)(mesh->triangle_buffer()));
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

uint64_t MetalDevice::create_accel(AccelBuildHint hint) noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    check_raytracing_supported();
    Clock clock;
    auto accel = new_with_allocator<MetalAccel>(hint);
    LUISA_VERBOSE_WITH_LOCATION("Created accel in {} ms.", clock.toc());
    return reinterpret_cast<uint64_t>(accel);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

void MetalDevice::destroy_accel(uint64_t handle) noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto accel = reinterpret_cast<MetalAccel *>(handle);
    delete_with_allocator(accel);
    LUISA_VERBOSE_WITH_LOCATION("Destroyed accel #{}.", handle);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

void MetalDevice::emplace_back_instance_in_accel(uint64_t accel_handle, uint64_t mesh_handle, float4x4 transform, bool visible) noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto accel = reinterpret_cast<MetalAccel *>(accel_handle);
    auto mesh = reinterpret_cast<MetalMesh *>(mesh_handle);
    accel->add_instance(mesh, transform, visible);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

void MetalDevice::set_instance_transform_in_accel(uint64_t accel_handle, size_t index, float4x4 transform) noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto accel = reinterpret_cast<MetalAccel *>(accel_handle);
    accel->set_transform(index, transform);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

void MetalDevice::pop_back_instance_from_accel(uint64_t accel_handle) noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto accel = reinterpret_cast<MetalAccel *>(accel_handle);
    accel->pop_instance();
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

void MetalDevice::set_instance_in_accel(uint64_t accel_handle, size_t index, uint64_t mesh_handle, float4x4 transform, bool visible) noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto accel = reinterpret_cast<MetalAccel *>(accel_handle);
    auto mesh = reinterpret_cast<MetalMesh *>(mesh_handle);
    accel->set_instance(index, mesh, transform, visible);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

void MetalDevice::set_instance_visibility_in_accel(uint64_t accel_handle, size_t index, bool visible) noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto accel = reinterpret_cast<MetalAccel *>(accel_handle);
    accel->set_visibility(index, visible);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

bool MetalDevice::is_buffer_in_accel(uint64_t accel_handle, uint64_t buffer_handle) const noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto accel = reinterpret_cast<MetalAccel *>(accel_handle);
    return accel->uses_resource(buffer_handle);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

bool MetalDevice::is_mesh_in_accel(uint64_t accel_handle, uint64_t mesh_handle) const noexcept {
#ifdef LUISA_METAL_RAYTRACING_ENABLED
    auto accel = reinterpret_cast<MetalAccel *>(accel_handle);
    return accel->uses_resource(mesh_handle);
#else
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
#endif
}

uint64_t MetalDevice::create_bindless_array(size_t size) noexcept {
    Clock clock;
    auto array = new_with_allocator<MetalBindlessArray>(this, size);
    LUISA_VERBOSE_WITH_LOCATION("Created texture heap in {} ms.", clock.toc());
    return reinterpret_cast<uint64_t>(array);
}

void MetalDevice::destroy_bindless_array(uint64_t handle) noexcept {
    auto array = reinterpret_cast<MetalBindlessArray *>(handle);
    delete_with_allocator(array);
    LUISA_VERBOSE_WITH_LOCATION("Destroyed bindless array #{}.", handle);
}

uint64_t MetalDevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    if (kernel.raytracing()) { check_raytracing_supported(); }
    Clock clock;
    auto shader = _compiler->compile(kernel, meta_options);
    LUISA_VERBOSE_WITH_LOCATION("Compiled shader in {} ms.", clock.toc());
    auto p = (__bridge void *)(shader.handle());
    return reinterpret_cast<uint64_t>(p);
}

void MetalDevice::destroy_shader(uint64_t handle) noexcept {
    // nothing happens, the MetalCompiler class will manage all
    LUISA_VERBOSE_WITH_LOCATION("Destroyed shader #{}.", handle);
}

void MetalDevice::check_raytracing_supported() const noexcept {
    if (!_handle.supportsRaytracing) {
        LUISA_ERROR_WITH_LOCATION(
            "This device does not support raytracing: {}.",
            [_handle.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }
}

void *MetalDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

void *MetalDevice::texture_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

void *MetalDevice::native_handle() const noexcept {
    return (__bridge void *)_handle;
}

void *MetalDevice::stream_native_handle(uint64_t handle) const noexcept {
    auto stream = reinterpret_cast<MetalStream *>(handle);
    return (__bridge void *)(stream->handle());
}

void MetalDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    reinterpret_cast<MetalBindlessArray *>(array)->emplace_buffer(index, handle, offset_bytes);
}

void MetalDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    reinterpret_cast<MetalBindlessArray *>(array)->emplace_tex2d(index, handle, sampler);
}

void MetalDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    reinterpret_cast<MetalBindlessArray *>(array)->emplace_tex3d(index, handle, sampler);
}

void MetalDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<MetalBindlessArray *>(array)->remove_buffer(index);
}

void MetalDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<MetalBindlessArray *>(array)->remove_tex2d(index);
}

void MetalDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<MetalBindlessArray *>(array)->remove_tex3d(index);
}

bool MetalDevice::is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return reinterpret_cast<MetalBindlessArray *>(array)->has_buffer(handle);
}

bool MetalDevice::is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return reinterpret_cast<MetalBindlessArray *>(array)->has_texture(handle);
}

}

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, std::string_view properties) noexcept {
    auto p = nlohmann::json::parse(properties);
    auto index = p.value("index", 0);
    return luisa::new_with_allocator<luisa::compute::metal::MetalDevice>(ctx, index);
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    luisa::delete_with_allocator(device);
}
