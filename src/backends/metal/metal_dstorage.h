#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalPinnedMemory {

private:
    void *_host_ptr;
    size_t _size_bytes;
    size_t _offset;
    MTL::Buffer *_device_buffer;

public:
    MetalPinnedMemory(MTL::Device *device,
                      void *host_ptr,
                      size_t size_bytes) noexcept;
    ~MetalPinnedMemory() noexcept;
    // disable copy and move
    MetalPinnedMemory(MetalPinnedMemory &&) noexcept = delete;
    MetalPinnedMemory(const MetalPinnedMemory &) noexcept = delete;
    MetalPinnedMemory &operator=(MetalPinnedMemory &&) noexcept = delete;
    MetalPinnedMemory &operator=(const MetalPinnedMemory &) noexcept = delete;
    [[nodiscard]] auto valid() const noexcept { return _device_buffer != nullptr; }
    [[nodiscard]] auto host_pointer() const noexcept { return _host_ptr; }
    [[nodiscard]] auto size() const noexcept { return _size_bytes; }
    [[nodiscard]] auto device_buffer() const noexcept { return _device_buffer; }
    [[nodiscard]] auto device_buffer_offset() const noexcept { return _offset; }
    void set_name(luisa::string_view name) noexcept;
};

class MetalFileHandle {

private:
    MTL::Device *_device;
    NS::URL *_url;
    size_t _size_bytes;
    std::mutex _mutex;
    luisa::unordered_map<uint, MTL::IOFileHandle *> _handles;

public:
    MetalFileHandle(MTL::Device *device,
                    luisa::string_view path,
                    size_t size_bytes) noexcept;
    ~MetalFileHandle() noexcept;
    // disable copy and move
    MetalFileHandle(MetalFileHandle &&) noexcept = delete;
    MetalFileHandle(const MetalFileHandle &) noexcept = delete;
    MetalFileHandle &operator=(MetalFileHandle &&) noexcept = delete;
    MetalFileHandle &operator=(const MetalFileHandle &) noexcept = delete;
    [[nodiscard]] auto size() const noexcept { return _size_bytes; }
    [[nodiscard]] auto url() const noexcept { return _url; }
    [[nodiscard]] MTL::IOFileHandle *handle(DStorageCompression compression) noexcept;
    void set_name(luisa::string_view name) noexcept;
};

class MetalIOCommandEncoder;

class MetalIOStream : public MetalStream {

public:
    using CallbackContainer = luisa::vector<MetalCallbackContext *>;

private:
    MTL::IOCommandQueue *_io_queue;
    MTL::SharedEvent *_io_event;
    size_t _event_value{0ull};
    spin_mutex _event_mutex;

private:
    void _encode(MetalCommandEncoder &encoder, Command *command) noexcept override;

public:
    explicit MetalIOStream(MTL::Device *device) noexcept;
    ~MetalIOStream() noexcept override;
    [[nodiscard]] auto valid() const noexcept { return _io_queue != nullptr; }
    [[nodiscard]] auto io_queue() const noexcept { return _io_queue; }
    [[nodiscard]] auto io_event() const noexcept { return _io_event; }
    void barrier(MTL::CommandBuffer *command_buffer) noexcept;
    void signal(MetalEvent *event, uint64_t value) noexcept override;
    void wait(MetalEvent *event, uint64_t value) noexcept override;
    void synchronize() noexcept override;
    void dispatch(CommandList &&list) noexcept override;
    void set_name(luisa::string_view name) noexcept override;
};

class MetalDevice;

class MetalDStorageExt : public DStorageExt {

private:
    MetalDevice *_device;

protected:
    [[nodiscard]] DeviceInterface *device() const noexcept override;
    [[nodiscard]] ResourceCreationInfo create_stream_handle(const DStorageStreamOption &option) noexcept override;
    [[nodiscard]] FileCreationInfo open_file_handle(luisa::string_view path) noexcept override;
    [[nodiscard]] PinnedMemoryInfo pin_host_memory(void *ptr, size_t size_bytes) noexcept override;
    void close_file_handle(uint64_t handle) noexcept override;
    void unpin_host_memory(uint64_t handle) noexcept override;

public:
    explicit MetalDStorageExt(MetalDevice *device) noexcept;
    void compress(const void *data, size_t size_bytes,
                  Compression algorithm, CompressionQuality quality,
                  luisa::vector<std::byte> &result) noexcept override;
};

}// namespace luisa::compute::metal

