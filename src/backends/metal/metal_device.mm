//
// Created by Mike Smith on 2021/3/17.
//

#import <chrono>
#import <numeric>

#import <core/platform.h>
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
    _buffer_slots[handle] = nullptr;
    LUISA_VERBOSE_WITH_LOCATION("Disposed buffer #{}.", handle);
    std::scoped_lock lock{_buffer_mutex};
    _available_buffer_slots.emplace_back(handle);
}

uint64_t MetalDevice::_create_stream() noexcept {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto stream = [_handle newCommandQueueWithMaxCommandBufferCount:8u];
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
    _stream_slots[handle] = nullptr;
    LUISA_VERBOSE_WITH_LOCATION("Disposed stream #{}.", handle);
    std::scoped_lock lock{_stream_mutex};
    _available_stream_slots.emplace_back(handle);
}

MetalDevice::MetalDevice(uint32_t index) noexcept {

    auto devices = MTLCopyAllDevices();
    if (auto count = devices.count; index >= count) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid Metal device index {} (#device = {}).",
            index, count);
    }
    _handle = devices[index];
    LUISA_VERBOSE_WITH_LOCATION(
        "Created Metal device #{} with name: {}.",
        index, [_handle.name cStringUsingEncoding:NSUTF8StringEncoding]);

    static constexpr auto initial_buffer_count = 64u;
    _buffer_slots.resize(initial_buffer_count, nullptr);
    _available_buffer_slots.resize(initial_buffer_count);
    std::iota(_available_buffer_slots.rbegin(), _available_buffer_slots.rend(), 0u);

    static constexpr auto initial_stream_count = 4u;
    _stream_slots.resize(initial_stream_count, nullptr);
    _available_stream_slots.resize(initial_stream_count);
    std::iota(_available_stream_slots.rbegin(), _available_stream_slots.rend(), 0u);
}

MetalDevice::~MetalDevice() noexcept {
    auto name = fmt::format(
        "{}", [_handle.name cStringUsingEncoding:NSUTF8StringEncoding]);
    _handle = nullptr;
    LUISA_VERBOSE_WITH_LOCATION("Destroyed Metal device with name: {}.", name);
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

void MetalDevice::_dispatch(uint64_t stream_handle, std::unique_ptr<CommandBuffer> commands) noexcept {
    auto cb = [stream(stream_handle) commandBuffer];
    MetalCommandEncoder encoder{this, cb};
    for (auto &&command : *commands) { command->accept(encoder); }
    [cb commit];
}

id<MTLDevice> MetalDevice::handle() const noexcept {
    return _handle;
}

void MetalDevice::_dispatch(uint64_t stream_handle, std::function<void()> function) noexcept {
    auto command_buffer = [stream(stream_handle) commandBuffer];
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) { function(); }];
    [command_buffer commit];
}

}

LUISA_EXPORT luisa::compute::Device *create(uint32_t id) {
    return new luisa::compute::metal::MetalDevice{id};
}

LUISA_EXPORT void destroy(luisa::compute::Device *device) {
    delete device;
}
