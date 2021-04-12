//
// Created by Mike Smith on 2021/3/17.
//

#import <chrono>
#import <numeric>

#import <core/platform.h>
#import <core/hash.h>
#import <core/clock.h>
#import <runtime/context.h>

#import <backends/metal/metal_device.h>
#import <backends/metal/metal_command_encoder.h>

namespace luisa::compute::metal {

uint64_t MetalDevice::create_buffer(size_t size_bytes) noexcept {
    Clock clock;
    auto buffer = [_handle newBufferWithLength:size_bytes options:MTLResourceStorageModePrivate];
    LUISA_VERBOSE_WITH_LOCATION(
        "Created buffer with size {} in {} ms.",
        size_bytes, clock.toc());
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

void MetalDevice::dispose_buffer(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_buffer_mutex};
        _buffer_slots[handle] = nullptr;
        _available_buffer_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Disposed buffer #{}.", handle);
}

uint64_t MetalDevice::create_stream() noexcept {
    Clock clock;
    auto stream = [_handle newCommandQueue];
    LUISA_VERBOSE_WITH_LOCATION("Created stream in {} ms.", clock.toc());
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

void MetalDevice::dispose_stream(uint64_t handle) noexcept {
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

    static constexpr auto initial_event_count = 4u;
    _event_slots.resize(initial_event_count, MetalEvent{nullptr});
    _available_event_slots.resize(initial_event_count);
    std::iota(_available_event_slots.rbegin(), _available_event_slots.rend(), 0u);
}

MetalDevice::~MetalDevice() noexcept {
    auto name = fmt::format(
        "{}", [_handle.name cStringUsingEncoding:NSUTF8StringEncoding]);
    _handle = nullptr;
    LUISA_INFO("Destroyed Metal device with name: {}.", name);
}

void MetalDevice::synchronize_stream(uint64_t stream_handle) noexcept {
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

void MetalDevice::compile_kernel(uint32_t uid) noexcept {
    static_cast<void>(_compiler->kernel(uid));
}

MetalCompiler::KernelItem MetalDevice::kernel(uint32_t uid) const noexcept {
    return _compiler->kernel(uid);
}

MetalArgumentBufferPool *MetalDevice::argument_buffer_pool() const noexcept {
    return _argument_buffer_pool.get();
}

uint64_t MetalDevice::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool is_bindless) {

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
        case PixelFormat::R8U:
        case PixelFormat::R8U_SRGB:
            desc.pixelFormat = MTLPixelFormatR8Unorm;
            break;
        case PixelFormat::RG8U:
        case PixelFormat::RG8U_SRGB:
            desc.pixelFormat = MTLPixelFormatRG8Unorm;
            break;
        case PixelFormat::RGBA8U:
        case PixelFormat::RGBA8U_SRGB:
            desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
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

    LUISA_VERBOSE_WITH_LOCATION(
        "Created image (with {} mipmap{}) in {} ms.",
        mipmap_levels, mipmap_levels <= 1u ? "" : "s",
        clock.toc());

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

void MetalDevice::dispose_texture(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_texture_mutex};
        _texture_slots[handle] = nullptr;
        _available_texture_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Disposed image #{}.", handle);
}

id<MTLTexture> MetalDevice::texture(uint64_t handle) const noexcept {
    std::scoped_lock lock{_texture_mutex};
    return _texture_slots[handle];
}

uint64_t MetalDevice::create_event() noexcept {
    Clock clock;
    MetalEvent event{[_handle newSharedEvent]};
    LUISA_VERBOSE_WITH_LOCATION("Created event in {} ms.", clock.toc());
    std::scoped_lock lock{_event_mutex};
    if (_available_event_slots.empty()) {
        auto s = _event_slots.size();
        _event_slots.emplace_back(event);
        return s;
    }
    auto s = _available_event_slots.back();
    _available_event_slots.pop_back();
    _event_slots[s] = event;
    return s;
}

void MetalDevice::dispose_event(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_event_mutex};
        _event_slots[handle] = MetalEvent{nullptr};
        _available_event_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Disposed event #{}.", handle);
}

void MetalDevice::synchronize_event(uint64_t handle) noexcept {

    auto [e, l, v] = [this, handle] {
        std::scoped_lock lock{_event_mutex};
        auto &&e = _event_slots[handle];
        if (e.listener == nullptr) {
            e.listener = [[MTLSharedEventListener alloc] init];
        }
        return std::make_tuple(e.handle, e.listener, e.counter);
    }();

    std::condition_variable cv;
    auto p = &cv;
    [e notifyListener:l
              atValue:v
                block:^(id<MTLSharedEvent>, uint64_t) { p->notify_one(); }];
    std::mutex mutex;
    std::unique_lock lock{mutex};
    cv.wait(lock);
}

void MetalDevice::dispatch(uint64_t stream_handle, CommandBuffer buffer) noexcept {
    auto command_buffer = [stream(stream_handle) commandBuffer];
    MetalCommandEncoder encoder{this, command_buffer};
    for (auto &&command : buffer) { command->accept(encoder); }
    [command_buffer commit];
}

void MetalDevice::signal_event(uint64_t handle, id<MTLCommandBuffer> cmd) noexcept {
    auto [event, value] = [this, handle] {
        std::scoped_lock lock{_event_mutex};
        auto &&e = _event_slots[handle];
        return std::make_pair(e.handle, ++e.counter);
    }();
    [cmd encodeSignalEvent:event value:value];
}

void MetalDevice::wait_event(uint64_t handle, id<MTLCommandBuffer> cmd) noexcept {
    auto [e, v] = [this, handle] {
        std::scoped_lock lock{_event_mutex};
        auto &&event = _event_slots[handle];
        return std::make_pair(event.handle, event.counter);
    }();
    [cmd encodeWaitForEvent:e value:v];
}

}

LUISA_EXPORT luisa::compute::Device *create(const luisa::compute::Context &ctx, uint32_t id) noexcept {
    return new luisa::compute::metal::MetalDevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device *device) noexcept {
    delete device;
}
