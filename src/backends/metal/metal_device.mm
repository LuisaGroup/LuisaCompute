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
#import <runtime/heap.h>

#import <backends/metal/metal_device.h>
#import <backends/metal/metal_heap.h>
#import <backends/metal/metal_command_encoder.h>

namespace luisa::compute::metal {

uint64_t MetalDevice::create_buffer(size_t size_bytes, uint64_t heap_handle, uint32_t index_in_heap) noexcept {
    Clock clock;
    id<MTLBuffer> buffer = nullptr;
    MetalHeap *heap = nullptr;
    if (heap_handle == Heap::invalid_handle) {
        buffer = [_handle newBufferWithLength:size_bytes
                                      options:MTLResourceStorageModePrivate];
        LUISA_VERBOSE_WITH_LOCATION(
            "Created buffer with size {} in {} ms.",
            size_bytes, clock.toc());
    } else {
        heap = this->heap(heap_handle);
        buffer = heap->allocate_buffer(size_bytes);
        LUISA_VERBOSE_WITH_LOCATION(
            "Created buffer from heap #{} at index {} "
            "with size {} in {} ms.",
            heap_handle, index_in_heap,
            size_bytes, clock.toc());
    }

    auto buffer_handle = [&] {
        auto h = 0ull;
        std::scoped_lock lock{_buffer_mutex};
        if (_available_buffer_slots.empty()) {
            h = _buffer_slots.size();
            _buffer_slots.emplace_back(buffer);
        } else {
            h = _available_buffer_slots.back();
            _buffer_slots[h] = buffer;
            _available_buffer_slots.pop_back();
        }
        return h;
    }();
    if (heap != nullptr) { heap->emplace_buffer(index_in_heap, buffer_handle); }
    return buffer_handle | (heap_handle << 32u);
}

void MetalDevice::destroy_buffer(uint64_t handle) noexcept {
    {
        auto buffer_handle = handle & 0xffffffffu;
        if (auto heap_index = handle >> 32u; heap_index != 0xffffffffu) {
            heap(heap_index)->destroy_buffer(buffer_handle);
        }
        std::scoped_lock lock{_buffer_mutex};
        _buffer_slots[buffer_handle] = nullptr;
        _available_buffer_slots.emplace_back(buffer_handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed buffer #{}.", handle);
}

uint64_t MetalDevice::create_stream() noexcept {
    Clock clock;
    auto max_command_buffer_count = _handle.isLowPower ? 4u : 16u;
    auto stream = luisa::make_unique<MetalStream>(_handle, max_command_buffer_count);
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

void MetalDevice::destroy_stream(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_stream_mutex};
        _stream_slots[handle] = nullptr;
        _available_stream_slots.emplace_back(handle);
    }
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
    std::vector<id<MTLDevice>> sorted_devices;
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

#ifdef LUISA_METAL_RAYTRACING_ENABLED
    _compacted_size_buffer_pool = luisa::make_unique<MetalSharedBufferPool>(_handle, sizeof(uint), 4096u / sizeof(uint), false);
#endif

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

    static constexpr auto initial_shader_count = 16u;
    _shader_slots.resize(initial_shader_count, {});
    _available_shader_slots.resize(initial_shader_count);
    std::iota(_available_shader_slots.rbegin(), _available_shader_slots.rend(), 0u);

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
    return _buffer_slots[handle & 0xffffffffu];
}

MetalStream *MetalDevice::stream(uint64_t handle) const noexcept {
    std::scoped_lock lock{_stream_mutex};
    return _stream_slots[handle].get();
}

id<MTLDevice> MetalDevice::handle() const noexcept {
    return _handle;
}

MetalShader MetalDevice::compiled_kernel(uint64_t handle) const noexcept {
    std::scoped_lock lock{_shader_mutex};
    return _shader_slots[handle];
}

uint64_t MetalDevice::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels,
    TextureSampler sampler,
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

    auto from_heap = heap_handle != Heap::invalid_handle;
    desc.allowGPUOptimizedContents = YES;
    desc.storageMode = MTLStorageModePrivate;
    desc.hazardTrackingMode = from_heap ? MTLHazardTrackingModeUntracked
                                        : MTLHazardTrackingModeTracked;
    desc.usage = from_heap ? MTLTextureUsageShaderRead
                           : MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.mipmapLevelCount = mipmap_levels;

    id<MTLTexture> texture = nullptr;
    MetalHeap *heap = nullptr;
    if (from_heap) {
        heap = this->heap(heap_handle);
        texture = heap->allocate_texture(desc);
    } else {
        texture = [_handle newTextureWithDescriptor:desc];
    }

    if (texture == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to allocate texture with description {}.",
            [desc.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    LUISA_VERBOSE_WITH_LOCATION(
        "Created image (with {} mipmap{}) in {} ms.",
        mipmap_levels, mipmap_levels <= 1u ? "" : "s",
        clock.toc());

    auto texture_handle = [&] {
        auto h = 0ull;
        std::scoped_lock lock{_texture_mutex};
        if (_available_texture_slots.empty()) {
            h = _texture_slots.size();
            _texture_slots.emplace_back(texture);
        } else {
            h = _available_texture_slots.back();
            _texture_slots[h] = texture;
            _available_texture_slots.pop_back();
        }
        return h;
    }();
    if (heap != nullptr) {
        heap->emplace_texture(index_in_heap, texture_handle, sampler);
    }
    return texture_handle | (heap_handle << 32u);
}

void MetalDevice::destroy_texture(uint64_t handle) noexcept {
    {
        auto texture_handle = handle & 0xffffffffu;
        if (auto heap_handle = handle >> 32u; heap_handle != 0xffffffffu) {
            heap(heap_handle)->destroy_texture(texture_handle);
        }
        std::scoped_lock lock{_texture_mutex};
        auto &&tex = _texture_slots[texture_handle];
        if (tex.heap != nullptr) { [tex makeAliasable]; }
        tex = nullptr;
        _available_texture_slots.emplace_back(texture_handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed image #{}.", handle);
}

id<MTLTexture> MetalDevice::texture(uint64_t handle) const noexcept {
    std::scoped_lock lock{_texture_mutex};
    return _texture_slots[handle & 0xffffffffu];
}

uint64_t MetalDevice::create_event() noexcept {
    Clock clock;
    auto event = luisa::make_unique<MetalEvent>([_handle newEvent]);
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

void MetalDevice::destroy_event(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_event_mutex};
        _event_slots[handle] = nullptr;
        _available_event_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed event #{}.", handle);
}

void MetalDevice::synchronize_event(uint64_t handle) noexcept {
    event(handle)->synchronize();
}

void MetalDevice::dispatch(uint64_t stream_handle, CommandList cmd_list) noexcept {
    @autoreleasepool {
        auto s = stream(stream_handle);
        MetalCommandEncoder encoder{this, s};
        for (auto command : cmd_list) { command->accept(encoder); }
        s->dispatch(encoder.command_buffer());
    }
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    @autoreleasepool {
        auto e = event(handle);
        auto s = stream(stream_handle);
        auto command_buffer = s->command_buffer();
        e->signal(command_buffer);
        s->dispatch(command_buffer);
    }
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    @autoreleasepool {
        auto e = event(handle);
        auto s = stream(stream_handle);
        auto command_buffer = s->command_buffer();
        e->wait(command_buffer);
        s->dispatch(command_buffer);
    }
}

MetalEvent *MetalDevice::event(uint64_t handle) const noexcept {
    std::scoped_lock lock{_event_mutex};
    return _event_slots[handle].get();
}

#ifdef LUISA_METAL_RAYTRACING_ENABLED

uint64_t MetalDevice::create_mesh() noexcept {
    check_raytracing_supported();
    Clock clock;
    auto mesh = luisa::make_unique<MetalMesh>();
    LUISA_VERBOSE_WITH_LOCATION("Created mesh in {} ms.", clock.toc());
    std::scoped_lock lock{_mesh_mutex};
    if (_available_mesh_slots.empty()) {
        auto s = _mesh_slots.size();
        _mesh_slots.emplace_back(std::move(mesh));
        return s;
    }
    auto s = _available_mesh_slots.back();
    _available_mesh_slots.pop_back();
    _mesh_slots[s] = std::move(mesh);
    return s;
}

void MetalDevice::destroy_mesh(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_mesh_mutex};
        _mesh_slots[handle] = {};
        _available_mesh_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed mesh #{}.", handle);
}

uint64_t MetalDevice::create_accel() noexcept {
    check_raytracing_supported();
    Clock clock;
    auto accel = luisa::make_unique<MetalAccel>(this);
    LUISA_VERBOSE_WITH_LOCATION("Created accel in {} ms.", clock.toc());
    std::scoped_lock lock{_accel_mutex};
    if (_available_accel_slots.empty()) {
        auto s = _accel_slots.size();
        _accel_slots.emplace_back(std::move(accel));
        return s;
    }
    auto s = _available_accel_slots.back();
    _available_accel_slots.pop_back();
    _accel_slots[s] = std::move(accel);
    return s;
}

void MetalDevice::destroy_accel(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_accel_mutex};
        _accel_slots[handle] = {};
        _available_accel_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed accel #{}.", handle);
}

#else

uint64_t MetalDevice::create_mesh() noexcept {
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
}

void MetalDevice::destroy_mesh(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
}

uint64_t MetalDevice::create_accel() noexcept {
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
}

void MetalDevice::destroy_accel(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Raytracing is not enabled for Metal backend.");
}

#endif

uint64_t MetalDevice::create_heap(size_t size) noexcept {
    Clock clock;
    auto heap = luisa::make_unique<MetalHeap>(this, size);
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

size_t MetalDevice::query_heap_memory_usage(uint64_t handle) noexcept {
    return [heap(handle)->handle() usedSize];
}

void MetalDevice::destroy_heap(uint64_t handle) noexcept {
    auto heap = [&] {
        std::scoped_lock lock{_heap_mutex};
        auto h = std::move(_heap_slots[handle]);
        _available_heap_slots.emplace_back(handle);
        return h;
    }();
    // destroy all buffers
    {
        std::scoped_lock lock{_buffer_mutex};
        heap->traverse_buffers([&](auto b) noexcept {
            _buffer_slots[b] = nullptr;
            _available_buffer_slots.emplace_back(b);
        });
    }
    // destroy all textures
    {
        std::scoped_lock lock{_texture_mutex};
        heap->traverse_textures([&](auto t) noexcept {
            _texture_slots[t] = nullptr;
            _available_texture_slots.emplace_back(t);
        });
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed heap #{}.", handle);
}

MetalHeap *MetalDevice::heap(uint64_t handle) const noexcept {
    std::scoped_lock lock{_heap_mutex};
    return _heap_slots[handle].get();
}

uint64_t MetalDevice::create_shader(Function kernel) noexcept {
    if (kernel.raytracing()) { check_raytracing_supported(); }
    Clock clock;
    auto shader = _compiler->compile(kernel);
    LUISA_VERBOSE_WITH_LOCATION("Compiled shader in {} ms.", clock.toc());
    std::scoped_lock lock{_shader_mutex};
    if (_available_shader_slots.empty()) {
        auto shader_handle = _shader_slots.size();
        _shader_slots.emplace_back(shader);
        return shader_handle;
    }
    auto s = _available_shader_slots.back();
    _available_shader_slots.pop_back();
    _shader_slots[s] = shader;
    return s;
}

void MetalDevice::destroy_shader(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_shader_mutex};
        _shader_slots[handle] = {};
        _available_shader_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed shader #{}.", handle);
}

#ifdef LUISA_METAL_RAYTRACING_ENABLED

MetalMesh *MetalDevice::mesh(uint64_t handle) const noexcept {
    std::scoped_lock lock{_mesh_mutex};
    return _mesh_slots[handle].get();
}

MetalAccel *MetalDevice::accel(uint64_t handle) const noexcept {
    std::scoped_lock lock{_accel_mutex};
    return _accel_slots[handle].get();
}

MetalSharedBufferPool *MetalDevice::compacted_size_buffer_pool() const noexcept {
    return _compacted_size_buffer_pool.get();
}

#endif

void MetalDevice::check_raytracing_supported() const noexcept {
    if (!_handle.supportsRaytracing) {
        LUISA_ERROR_WITH_LOCATION(
            "This device does not support raytracing: {}.",
            [_handle.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }
}

void *MetalDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return (__bridge void *)buffer(handle);
}

void *MetalDevice::texture_native_handle(uint64_t handle) const noexcept {
    return (__bridge void *)texture(handle);
}

void *MetalDevice::native_handle() const noexcept {
    return (__bridge void *)_handle;
}

void *MetalDevice::stream_native_handle(uint64_t handle) const noexcept {
    return (__bridge void *)stream(handle)->handle();
}

}

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, uint32_t id) noexcept {
    return luisa::new_with_allocator<luisa::compute::metal::MetalDevice>(ctx, id);
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    luisa::delete_with_allocator(device);
}
