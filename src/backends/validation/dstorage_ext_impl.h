#pragma once
#include <backends/ext/dstorage_ext_interface.h>
#include <vstl/common.h>
namespace lc::validation {
class Device;
using namespace luisa::compute;
class DStorageExtImpl : public DStorageExt, public vstd::IOperatorNewBase {
    DStorageExt *_impl;

public:
    DeviceInterface *device() const noexcept override;
    File open_file_handle(luisa::string_view path) noexcept override;
    void close_file_handle(uint64_t handle) noexcept override;
    ResourceCreationInfo create_stream_handle() noexcept override;
    void gdeflate_compress(
        luisa::span<std::byte const> input,
        CompressQuality quality,
        luisa::vector<std::byte> &result) noexcept override;
    DStorageExtImpl(DStorageExt *ext);
};
}// namespace lc::validation