//
// Created by Mike Smith on 2021/3/17.
//

#if !__has_feature(objc_arc)
#error failed to compile the Metal backend with ARC off.
#endif

#import <chrono>
#import <numeric>

#import <core/platform.h>
#import <core/hash.h>
#import <core/clock.h>
#import <runtime/context.h>
#import <runtime/texture_heap.h>

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
    auto stream = std::make_unique<MetalStream>([_handle newCommandQueue]);
    LUISA_VERBOSE_WITH_LOCATION("Created stream in {} ms.", clock.toc());
    std::scoped_lock lock{_stream_mutex};
    if (_available_stream_slots.empty()) {
        auto s = _stream_slots.size();
        _stream_slots.emplace_back(std::move(stream));
        return s;
    }
    auto s = _available_stream_slots.back();
    _available_stream_slots.pop_back();
    _stream_slots[s] = std::move(stream);
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
    : Device::Interface{ctx} {

    auto devices = MTLCopyAllDevices();
    if (auto count = devices.count; index >= count) [[unlikely]] {
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
    _stream_slots.reserve(initial_stream_count);
    for (auto i = 0u; i < initial_stream_count; i++) { _stream_slots.emplace_back(nullptr); }
    _available_stream_slots.resize(initial_stream_count);
    std::iota(_available_stream_slots.rbegin(), _available_stream_slots.rend(), 0u);

    static constexpr auto initial_texture_count = 16u;
    _texture_slots.resize(initial_texture_count, nullptr);
    _available_texture_slots.resize(initial_texture_count);
    std::iota(_available_texture_slots.rbegin(), _available_texture_slots.rend(), 0u);

    static constexpr auto initial_heap_count = 4u;
    _heap_slots.reserve(initial_heap_count);
    for (auto i = 0u; i < initial_heap_count; i++) { _heap_slots.emplace_back(nullptr); }
    _available_heap_slots.resize(initial_heap_count);
    std::iota(_available_heap_slots.rbegin(), _available_heap_slots.rend(), 0u);

    static constexpr auto initial_event_count = 4u;
    _event_slots.reserve(initial_event_count);
    for (auto i = 0u; i < initial_event_count; i++) { _event_slots.emplace_back(nullptr); }
    _available_event_slots.resize(initial_event_count);
    std::iota(_available_event_slots.rbegin(), _available_event_slots.rend(), 0u);
}

MetalDevice::~MetalDevice() noexcept {
    LUISA_INFO(
        "Destroyed Metal device with name: {}.",
        [_handle.name cStringUsingEncoding:NSUTF8StringEncoding]);
    _handle = nullptr;
}

void MetalDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    stream(stream_handle)->synchronize();
}

id<MTLBuffer> MetalDevice::buffer(uint64_t handle) const noexcept {
    std::scoped_lock lock{_buffer_mutex};
    return _buffer_slots[handle];
}

MetalStream *MetalDevice::stream(uint64_t handle) const noexcept {
    std::scoped_lock lock{_stream_mutex};
    return _stream_slots[handle].get();
}

id<MTLDevice> MetalDevice::handle() const noexcept {
    return _handle;
}

void MetalDevice::compile(const detail::FunctionBuilder *kernel) noexcept {
    static_cast<void>(compiled_kernel(kernel));
}

MetalCompiler::KernelItem MetalDevice::compiled_kernel(Function kernel) const noexcept {
    return _compiler->compile(kernel);
}

MetalArgumentBufferPool *MetalDevice::argument_buffer_pool() const noexcept {
    return _argument_buffer_pool.get();
}

uint64_t MetalDevice::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels,
    uint64_t heap_handle,
    uint32_t index_in_heap) {

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

    auto from_heap = heap_handle != TextureHeap::invalid_handle;
    desc.allowGPUOptimizedContents = true;
    desc.resourceOptions = MTLResourceStorageModePrivate;
    desc.usage = from_heap ? MTLTextureUsageShaderRead
                           : MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.mipmapLevelCount = mipmap_levels;
    auto texture = [&] {
        if (from_heap) { return heap(heap_handle)->allocate_texture(desc, index_in_heap); }
        return [_handle newTextureWithDescriptor:desc];
    }();

    LUISA_VERBOSE_WITH_LOCATION(
        "Created image (with {} mipmap{}) in {} ms.",
        mipmap_levels, mipmap_levels <= 1u ? "" : "s",
        clock.toc());

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
        auto &&tex = _texture_slots[handle];
        if (tex.heap != nullptr) { [tex makeAliasable]; }
        tex = nullptr;
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
    auto event = std::make_unique<MetalEvent>([_handle newEvent]);
    LUISA_VERBOSE_WITH_LOCATION("Created event in {} ms.", clock.toc());
    std::scoped_lock lock{_event_mutex};
    if (_available_event_slots.empty()) {
        auto s = _event_slots.size();
        _event_slots.emplace_back(std::move(event));
        return s;
    }
    auto s = _available_event_slots.back();
    _available_event_slots.pop_back();
    _event_slots[s] = std::move(event);
    return s;
}

void MetalDevice::dispose_event(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_event_mutex};
        _event_slots[handle] = nullptr;
        _available_event_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Disposed event #{}.", handle);
}

void MetalDevice::synchronize_event(uint64_t handle) noexcept {
    event(handle)->synchronize();
}

void MetalDevice::dispatch(uint64_t stream_handle, CommandBuffer buffer) noexcept {
    auto s = stream(stream_handle);
    s->with_command_buffer([this,
                            &u = s->upload_ring_buffer(),
                            &d = s->download_ring_buffer(),
                            buffer = std::move(buffer)](id<MTLCommandBuffer> command_buffer) noexcept {
        MetalCommandEncoder encoder{this, command_buffer, u, d};
        for (auto &&command : buffer) { command->accept(encoder); }
    });
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto e = event(handle);
    stream(stream_handle)->with_command_buffer([e](auto buffer) noexcept { e->signal(buffer); });
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto e = event(handle);
    stream(stream_handle)->with_command_buffer([e](auto buffer) noexcept { e->wait(buffer); });
}

MetalEvent *MetalDevice::event(uint64_t handle) const noexcept {
    std::scoped_lock lock{_event_mutex};
    return _event_slots[handle].get();
}

uint64_t MetalDevice::create_mesh(uint64_t stream_handle,
                                  uint64_t vertex_buffer_handle, size_t vertex_buffer_offset_bytes, size_t vertex_count,
                                  uint64_t index_buffer_handle, size_t index_buffer_offset_bytes, size_t triangle_count) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void MetalDevice::dispose_mesh(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

uint64_t MetalDevice::create_accel(uint64_t stream_handle,
                                   uint64_t mesh_handle_buffer_handle, size_t mesh_handle_buffer_offset_bytes,
                                   uint64_t transform_buffer_handle, size_t transform_buffer_offset_bytes,
                                   size_t mesh_count) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void MetalDevice::dispose_accel(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

uint64_t MetalDevice::create_texture_heap(size_t size) noexcept {
    Clock clock;
    auto heap = std::make_unique<MetalTextureHeap>(_handle, size);
    LUISA_VERBOSE_WITH_LOCATION("Created texture heap in {} ms.", clock.toc());
    std::scoped_lock lock{_heap_mutex};
    if (_available_heap_slots.empty()) {
        auto s = _heap_slots.size();
        _heap_slots.emplace_back(std::move(heap));
        return s;
    }
    auto s = _available_heap_slots.back();
    _available_heap_slots.pop_back();
    _heap_slots[s] = std::move(heap);
    return s;
}

size_t MetalDevice::query_texture_heap_memory_usage(uint64_t handle) noexcept {
    return [heap(handle)->handle() usedSize];
}

void MetalDevice::dispose_texture_heap(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_heap_mutex};
        _heap_slots[handle] = nullptr;
        _available_heap_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Disposed heap #{}.", handle);
}

MetalTextureHeap *MetalDevice::heap(uint64_t handle) const noexcept {
    std::scoped_lock lock{_heap_mutex};
    return _heap_slots[handle].get();
}

}

LUISA_EXPORT luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, uint32_t id) noexcept {
    return new luisa::compute::metal::MetalDevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device::Interface *device) noexcept {
    delete device;
}
