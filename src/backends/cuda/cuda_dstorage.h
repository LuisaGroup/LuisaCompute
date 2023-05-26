//
// Created by Mike on 5/26/2023.
//

#pragma once

#include <backends/ext/dstorage_ext_interface.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

class CUDAMappedFile {

private:
    void *_file_handle;
    void *_file_mapping;
    void *_mapped_pointer;
    CUdeviceptr _device_address;
    size_t _size_bytes;

public:
    explicit CUDAMappedFile(luisa::string_view path) noexcept;
    ~CUDAMappedFile() noexcept;
    CUDAMappedFile(CUDAMappedFile &&) noexcept = delete;
    CUDAMappedFile(const CUDAMappedFile &) noexcept = delete;
    CUDAMappedFile &operator=(CUDAMappedFile &&) noexcept = delete;
    CUDAMappedFile &operator=(const CUDAMappedFile &) noexcept = delete;
    [[nodiscard]] auto mapped_pointer() const noexcept { return _mapped_pointer; }
    [[nodiscard]] auto device_address() const noexcept { return _device_address; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
};

class CUDAPinnedMemory {

private:
    void *_host_pointer;
    CUdeviceptr _device_address;
    size_t _size_bytes;

public:
    CUDAPinnedMemory(void *p, size_t size) noexcept;
    ~CUDAPinnedMemory() noexcept;
    CUDAPinnedMemory(CUDAPinnedMemory &&) noexcept = delete;
    CUDAPinnedMemory(const CUDAPinnedMemory &) noexcept = delete;
    CUDAPinnedMemory &operator=(CUDAPinnedMemory &&) noexcept = delete;
    CUDAPinnedMemory &operator=(const CUDAPinnedMemory &) noexcept = delete;
    [[nodiscard]] auto host_pointer() const noexcept { return _host_pointer; }
    [[nodiscard]] auto device_address() const noexcept { return _device_address; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
};

class CUDADStorageExt : public DStorageExt {

private:
    CUDADevice *_device;

public:
    CUDADStorageExt(CUDADevice *device) noexcept : _device{device} {}

public:
    void compress(const void *data,
                  size_t size_bytes,
                  Compression algorithm,
                  CompressionQuality quality,
                  vector<std::byte> &result) noexcept override;

protected:
    [[nodiscard]] DeviceInterface *device() const noexcept override { return _device; }
    [[nodiscard]] ResourceCreationInfo create_stream_handle(const DStorageStreamOption &option) noexcept override;
    [[nodiscard]] FileCreationInfo open_file_handle(luisa::string_view path) noexcept override;
    void close_file_handle(uint64_t handle) noexcept override;
    [[nodiscard]] PinnedMemoryInfo pin_host_memory(void *ptr, size_t size_bytes) noexcept override;
    void unpin_host_memory(uint64_t handle) noexcept override;
};

}// namespace luisa::compute::cuda
