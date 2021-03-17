//
// Created by Mike Smith on 2021/3/17.
//

#import <chrono>
#import <numeric>

#import <core/platform.h>
#import <backends/metal/metal_device.h>

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
    _stream_slots[handle] = nullptr;
    LUISA_VERBOSE_WITH_LOCATION("Disposed stream #{}.", handle);
    std::scoped_lock lock{_stream_mutex};
    _available_stream_slots.emplace_back(handle);
}

void MetalDevice::_dispatch(uint64_t stream_handle, BufferCopyCommand command) noexcept {
    auto stream = _stream_slots[stream_handle];
    auto command_buffer = [stream commandBuffer];
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:_buffer_slots[command.src_handle()]
                    sourceOffset:command.src_offset()
                        toBuffer:_buffer_slots[command.dst_handle()]
               destinationOffset:command.dst_offset()
                            size:command.size()];
    [blit_encoder endEncoding];
    [command_buffer commit];
}

void MetalDevice::_dispatch(uint64_t stream_handle, BufferUploadCommand command) noexcept {

    auto stream = _stream_slots[stream_handle];
    auto buffer = _buffer_slots[command.handle()];

    auto t0 = std::chrono::high_resolution_clock::now();
    auto temporary = [_handle newBufferWithBytes:command.data()
                                          length:command.size()
                                         options:MTLResourceStorageModeShared];
    auto t1 = std::chrono::high_resolution_clock::now();
    using namespace std::chrono_literals;
    LUISA_VERBOSE_WITH_LOCATION(
        "Allocated temporary shared buffer with size {} in {} ms.",
        command.size(), (t1 - t0) / 1ns * 1e-6);

    auto command_buffer = [stream commandBuffer];
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:temporary
                    sourceOffset:0u
                        toBuffer:buffer
               destinationOffset:command.offset()
                            size:command.size()];
    [blit_encoder endEncoding];
    [command_buffer commit];
}

void MetalDevice::_dispatch(uint64_t stream_handle, BufferDownloadCommand command) noexcept {

    auto stream = _stream_slots[stream_handle];
    auto buffer = _buffer_slots[command.handle()];

    auto t0 = std::chrono::high_resolution_clock::now();
    auto temporary = [_handle newBufferWithLength:command.size() options:MTLResourceStorageModeShared];
    auto t1 = std::chrono::high_resolution_clock::now();
    using namespace std::chrono_literals;
    LUISA_VERBOSE_WITH_LOCATION(
        "Allocated temporary shared buffer with size {} in {} ms.",
        command.size(), (t1 - t0) / 1ns * 1e-6);

    auto command_buffer = [stream commandBuffer];
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:buffer
                    sourceOffset:command.offset()
                        toBuffer:temporary
               destinationOffset:0u
                            size:command.size()];
    [blit_encoder endEncoding];
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
      std::memcpy(command.data(), temporary.contents, command.size());
    }];
    [command_buffer commit];
}

void MetalDevice::_dispatch(uint64_t stream_handle, KernelLaunchCommand command) noexcept {
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

void MetalDevice::_dispatch(uint64_t stream_handle, SynchronizeCommand) noexcept {
    auto stream = _stream_slots[stream_handle];
    auto command_buffer = [stream commandBuffer];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

}

LUISA_EXPORT luisa::compute::Device *create(uint32_t id) {
    return new luisa::compute::metal::MetalDevice{id};
}

LUISA_EXPORT void destroy(luisa::compute::Device *device) {
    delete device;
}
