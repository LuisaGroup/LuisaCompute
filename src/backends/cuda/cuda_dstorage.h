//
// Created by Mike on 5/26/2023.
//

#pragma once

#include <backends/ext/dstorage_ext_interface.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

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
