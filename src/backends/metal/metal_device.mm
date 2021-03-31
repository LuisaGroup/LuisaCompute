//
// Created by Mike Smith on 2021/3/17.
//

#import <chrono>
#import <numeric>

#import <core/platform.h>
#import <core/hash.h>
#import <runtime/context.h>

#import <backends/metal/metal_device.h>
#import <backends/metal/metal_command_encoder.h>

namespace luisa::compute::metal {

uint64_t MetalDevice::_create_buffer(size_t size_bytes) noexcept {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto buffer = [_handle newBufferWithLength:size_bytes options:MTLResourceStorageModePrivate];
    auto t1 = std::chrono::high_resolution_clock::now();
    using namespace std::chrono_literals;
    auto dt = (t1 - t0) / 1ns * 1e-6;
    LUISA_VERBOSE_WITH_LOCATION("Created buffer with size {} in {} ms.", size_bytes, dt);
    std::scoped_lock lock{_buffer_mutex};
    if (_available_buffer_slots.empty()) {
        auto s = _buffer_slots.size();
        _buffer_slots.emplace_back(buffer);
        return s;
    }
    auto s = _available_buffer_slots.back();
    _available_buffer_slots.pop_back();
    _buffer_slots[s] = buffer;
    return s;
}

void MetalDevice::_dispose_buffer(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_buffer_mutex};
        _buffer_slots[handle] = nullptr;
        _available_buffer_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Disposed buffer #{}.", handle);
}

uint64_t MetalDevice::_create_stream() noexcept {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto stream = [_handle newCommandQueue];
    auto t1 = std::chrono::high_resolution_clock::now();
    using namespace std::chrono_literals;
    auto dt = (t1 - t0) / 1ns * 1e-6;
    LUISA_VERBOSE_WITH_LOCATION("Created stream in {} ms.", dt);
    std::scoped_lock lock{_stream_mutex};
    if (_available_stream_slots.empty()) {
        auto s = _stream_slots.size();
        _stream_slots.emplace_back(stream);
        return s;
    }
    auto s = _available_stream_slots.back();
    _available_stream_slots.pop_back();
    _stream_slots[s] = stream;
    return s;
}

void MetalDevice::_dispose_stream(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_stream_mutex};
        _stream_slots[handle] = nullptr;
        _available_stream_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Disposed stream #{}.", handle);
}

MetalDevice::MetalDevice(const Context &ctx, uint32_t index) noexcept
    : Device{ctx} {

    auto devices = MTLCopyAllDevices();
    if (auto count = devices.count; index >= count) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid Metal device index {} (#device = {}).",
            index, count);
    }
    _handle = devices[index];
    LUISA_INFO(
        "Created Metal device #{} with name: {}.",
        index, [_handle.name cStringUsingEncoding:NSUTF8StringEncoding]);

    _compiler = std::make_unique<MetalCompiler>(this);
    _argument_buffer_pool = std::make_unique<MetalArgumentBufferPool>(_handle);

    static constexpr auto initial_buffer_count = 64u;
    _buffer_slots.resize(initial_buffer_count, nullptr);
    _available_buffer_slots.resize(initial_buffer_count);
    std::iota(_available_buffer_slots.rbegin(), _available_buffer_slots.rend(), 0u);

    static constexpr auto initial_stream_count = 4u;
    _stream_slots.resize(initial_stream_count, nullptr);
    _available_stream_slots.resize(initial_stream_count);
    std::iota(_available_stream_slots.rbegin(), _available_stream_slots.rend(), 0u);

    static constexpr auto initial_texture_count = 16u;
    _texture_slots.resize(initial_texture_count, nullptr);
    _available_texture_slots.resize(initial_texture_count);
    std::iota(_available_texture_slots.rbegin(), _available_texture_slots.rend(), 0u);
}

MetalDevice::~MetalDevice() noexcept {
    auto name = fmt::format(
        "{}", [_handle.name cStringUsingEncoding:NSUTF8StringEncoding]);
    _handle = nullptr;
    LUISA_INFO("Destroyed Metal device with name: {}.", name);
}

void MetalDevice::_synchronize_stream(uint64_t stream_handle) noexcept {
    auto command_buffer = [stream(stream_handle) commandBuffer];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

id<MTLBuffer> MetalDevice::buffer(uint64_t handle) const noexcept {
    std::scoped_lock lock{_buffer_mutex};
    return _buffer_slots[handle];
}

id<MTLCommandQueue> MetalDevice::stream(uint64_t handle) const noexcept {
    std::scoped_lock lock{_stream_mutex};
    return _stream_slots[handle];
}

id<MTLDevice> MetalDevice::handle() const noexcept {
    return _handle;
}

void MetalDevice::_prepare_kernel(uint32_t uid) noexcept {
    _compiler->prepare(uid);
}

MetalCompiler::PipelineState MetalDevice::kernel(uint32_t uid) const noexcept {
    return _compiler->kernel(uid);
}

void MetalDevice::_dispatch(uint64_t stream_handle, CommandBuffer commands, std::function<void()> function) noexcept {
    auto command_buffer = [stream(stream_handle) commandBuffer];
    MetalCommandEncoder encoder{this, command_buffer};
    for (auto &&command : commands) { command->accept(encoder); }
    if (function) {
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
          auto f = std::move(function);
          f();
        }];
    }
    [command_buffer commit];
}

MetalArgumentBufferPool *MetalDevice::argument_buffer_pool() const noexcept {
    return _argument_buffer_pool.get();
}

uint64_t MetalDevice::_create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, bool is_bindless) {

    auto t0 = std::chrono::high_resolution_clock::now();
    auto desc = [[MTLTextureDescriptor alloc] init];
    switch (dimension) {
        case 2u: desc.textureType = MTLTextureType2D; break;
        case 3u: desc.textureType = MTLTextureType3D; break;
        default: LUISA_ERROR_WITH_LOCATION("Invalid texture dimension {}.", dimension); break;
    }
    desc.width = width;
    desc.height = height;
    desc.depth = depth;
    switch (format) {
        case PixelFormat::R8U: desc.pixelFormat = MTLPixelFormatR8Unorm; break;
        case PixelFormat::R8U_SRGB: desc.pixelFormat = MTLPixelFormatR8Unorm_sRGB; break;
        case PixelFormat::RG8U: desc.pixelFormat = MTLPixelFormatRG8Unorm; break;
        case PixelFormat::RG8U_SRGB: desc.pixelFormat = MTLPixelFormatRG8Unorm_sRGB; break;
        case PixelFormat::RGBA8U: desc.pixelFormat = MTLPixelFormatRGBA8Unorm; break;
        case PixelFormat::RGBA8U_SRGB: desc.pixelFormat = MTLPixelFormatRGBA8Unorm_sRGB; break;
        case PixelFormat::R32F: desc.pixelFormat = MTLPixelFormatR32Float; break;
        case PixelFormat::RG32F: desc.pixelFormat = MTLPixelFormatRG32Float; break;
        case PixelFormat::RGBA32F: desc.pixelFormat = MTLPixelFormatRGBA32Float; break;
    }
    desc.allowGPUOptimizedContents = true;
    desc.resourceOptions = MTLResourceStorageModePrivate;
    desc.usage = is_bindless ? MTLTextureUsageShaderRead
                             : MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.mipmapLevelCount = mipmap_levels;
    auto texture = [_handle newTextureWithDescriptor:desc];
    auto t1 = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    LUISA_VERBOSE_WITH_LOCATION(
        "Created texture in {} ms.",
        (t1 - t0) / 1ns * 1e-6);

    if (is_bindless) {
        // TODO: emplace into descriptor array...
    }
    
    std::scoped_lock lock{_texture_mutex};
    if (_available_texture_slots.empty()) {
        auto s = _texture_slots.size();
        _texture_slots.emplace_back(texture);
        return s;
    }
    auto s = _available_texture_slots.back();
    _available_texture_slots.pop_back();
    _texture_slots[s] = texture;
    return s;
}

void MetalDevice::_dispose_texture(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_texture_mutex};
        _texture_slots[handle] = nullptr;
        _available_texture_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Disposed texture #{}.", handle);
}

id<MTLTexture> MetalDevice::texture(uint64_t handle) const noexcept {
    std::scoped_lock lock{_texture_mutex};
    return _texture_slots[handle];
}

}

LUISA_EXPORT luisa::compute::Device *create(const luisa::compute::Context &ctx, uint32_t id) noexcept {
    return new luisa::compute::metal::MetalDevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device *device) noexcept {
    delete device;
}
