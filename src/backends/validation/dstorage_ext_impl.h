#pragma once

#include <luisa/backends/ext/dstorage_ext_interface.h>
#include <luisa/vstl/common.h>

namespace lc::validation {

using namespace luisa::compute;

class DStorageExtImpl final : public DStorageExt, public vstd::IOperatorNewBase {
    DStorageExt *_impl;
    DeviceInterface *_self;

public:
    DeviceInterface *device() const noexcept override;
    FileCreationInfo open_file_handle(luisa::string_view path) noexcept override;
    void close_file_handle(uint64_t handle) noexcept override;
    PinnedMemoryInfo pin_host_memory(void *ptr, size_t size_bytes) noexcept override;
    void unpin_host_memory(uint64_t handle) noexcept override;
    ResourceCreationInfo create_stream_handle(const DStorageStreamOption &option) noexcept override;
    DStorageExtImpl(DStorageExt *ext, DeviceInterface *self);
    void compress(const void *data, size_t size_bytes,
                  Compression algorithm,
                  CompressionQuality quality,
                  luisa::vector<std::byte> &result) noexcept override;
};

}// namespace lc::validation

