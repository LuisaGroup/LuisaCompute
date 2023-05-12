#pragma once
#include <runtime/rhi/device_interface.h>
namespace luisa::compute {
class Stream;
class DStorageFile;
class DStorageExt : public DeviceExtension {
protected:
    ~DStorageExt() noexcept = default;

public:
    enum class CompressQuality : uint {
        Fastest,
        Default,
        Best
    };
    static constexpr luisa::string_view name = "DStorageExt";
    struct File : public ResourceCreationInfo {
        size_t size_bytes;
    };
    virtual DeviceInterface *device() const noexcept = 0;
    virtual File open_file_handle(luisa::string_view path) noexcept = 0;
    virtual void close_file_handle(uint64_t handle) noexcept = 0;
    virtual ResourceCreationInfo create_stream_handle() noexcept = 0;
    virtual void gdeflate_compress(
        luisa::span<std::byte const> input,
        CompressQuality quality,
        luisa::vector<std::byte> &result) noexcept = 0;
    [[nodiscard]] Stream create_stream() noexcept;
    [[nodiscard]] DStorageFile open_file(luisa::string_view path) noexcept;
};
}// namespace luisa::compute