//
// Created by Mike on 5/26/2023.
//

#include <backends/cuda/cuda_dstorage.h>

namespace luisa::compute::cuda {

void CUDADStorageExt::compress(const void *data, size_t size_bytes,
                               DStorageExt::Compression algorithm,
                               DStorageExt::CompressionQuality quality,
                               vector<std::byte> &result) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not supported.");
}

ResourceCreationInfo CUDADStorageExt::create_stream_handle(const DStorageStreamOption &option) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not supported.");
}

DStorageExt::FileCreationInfo CUDADStorageExt::open_file_handle(luisa::string_view path) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not supported.");
}

void CUDADStorageExt::close_file_handle(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not supported.");
}

DStorageExt::PinnedMemoryInfo CUDADStorageExt::pin_host_memory(void *ptr, size_t size_bytes) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not supported.");
}

void CUDADStorageExt::unpin_host_memory(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not supported.");
}

}// namespace luisa::compute::cuda
