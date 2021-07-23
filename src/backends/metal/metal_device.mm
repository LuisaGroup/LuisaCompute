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

void MetalDevice::destroy_buffer(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_buffer_mutex};
        _buffer_slots[handle] = nullptr;
        _available_buffer_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed buffer #{}.", handle);
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
    if (auto count = devices.count; index >= count) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid Metal device index {} (#device = {}).",
            index, count);
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

    _compiler = std::make_unique<MetalCompiler>(this);
    _argument_buffer_pool = std::make_unique<MetalSharedBufferPool>(_handle, 4096u, 16u, true);
    _compacted_size_buffer_pool = std::make_unique<MetalSharedBufferPool>(_handle, sizeof(uint), 4096u / sizeof(uint), false);

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
    return _buffer_slots[handle];
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

MetalSharedBufferPool *MetalDevice::argument_buffer_pool() const noexcept {
    return _argument_buffer_pool.get();
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

    auto from_heap = heap_handle != TextureHeap::invalid_handle;
    desc.allowGPUOptimizedContents = YES;
    desc.resourceOptions = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeDefault;
    desc.usage = from_heap ? MTLTextureUsageShaderRead
                           : MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.mipmapLevelCount = mipmap_levels;
    auto texture = [&] {
        if (from_heap) { return heap(heap_handle)->allocate_texture(desc, index_in_heap, sampler); }
        return [_handle newTextureWithDescriptor:desc];
    }();

    if (texture == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to allocate texture with description {}.",
            [desc.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }

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

void MetalDevice::destroy_texture(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_texture_mutex};
        auto &&tex = _texture_slots[handle];
        if (tex.heap != nullptr) { [tex makeAliasable]; }
        tex = nullptr;
        _available_texture_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed image #{}.", handle);
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
    stream(stream_handle)->dispatch([this, cmd_list = std::move(cmd_list)](MetalStream *stream) noexcept {
        MetalCommandEncoder encoder{this, stream};
        auto command_index = 0u;
        for (auto command : cmd_list) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Encoding command at index {}.",
                command_index);
            command->accept(encoder);
            command_index++;
        }
        return encoder.command_buffer();
    });
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto e = event(handle);
    stream(stream_handle)->dispatch([e](auto s) noexcept {
        auto cb = s->command_buffer();
        e->signal(cb);
        return cb;
    });
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto e = event(handle);
    stream(stream_handle)->dispatch([e](auto s) noexcept {
        auto cb = s->command_buffer();
        e->wait(cb);
        return cb;
    });
}

MetalEvent *MetalDevice::event(uint64_t handle) const noexcept {
    std::scoped_lock lock{_event_mutex};
    return _event_slots[handle].get();
}

uint64_t MetalDevice::create_mesh() noexcept {
    Clock clock;
    auto mesh = std::make_unique<MetalMesh>(_handle);
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
    Clock clock;
    auto accel = std::make_unique<MetalAccel>(this);
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

uint64_t MetalDevice::create_texture_heap(size_t size) noexcept {
    Clock clock;
    auto heap = std::make_unique<MetalTextureHeap>(this, size);
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

void MetalDevice::destroy_texture_heap(uint64_t handle) noexcept {
    {
        std::scoped_lock lock{_heap_mutex};
        _heap_slots[handle] = nullptr;
        _available_heap_slots.emplace_back(handle);
    }
    LUISA_VERBOSE_WITH_LOCATION("Destroyed heap #{}.", handle);
}

MetalTextureHeap *MetalDevice::heap(uint64_t handle) const noexcept {
    std::scoped_lock lock{_heap_mutex};
    return _heap_slots[handle].get();
}

uint64_t MetalDevice::create_shader(Function kernel) noexcept {
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

MetalMesh *MetalDevice::mesh(uint64_t handle) const noexcept {
    std::scoped_lock lock{_mesh_mutex};
    return _mesh_slots[handle].get();
}

MetalAccel *MetalDevice::accel(uint64_t handle) const noexcept {
    std::scoped_lock lock{_accel_mutex};
    return _accel_slots[handle].get();
}

NSMutableArray<id<MTLAccelerationStructure>> *MetalDevice::mesh_handles(std::span<const uint64_t> handles) noexcept {
    auto array = [[NSMutableArray alloc] init];
    std::scoped_lock lock{_mesh_mutex};
    for (auto h : handles) {
        [array addObject:_mesh_slots[h]->handle()];
    }
    return array;
}

MetalSharedBufferPool *MetalDevice::compacted_size_buffer_pool() const noexcept {
    return _compacted_size_buffer_pool.get();
}

}

LUISA_EXPORT luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, uint32_t id) noexcept {
    return new luisa::compute::metal::MetalDevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device::Interface *device) noexcept {
    delete device;
}
