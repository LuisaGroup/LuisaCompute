#pragma once
#include <runtime/device.h>
namespace luisa::compute {
class Stream;
class DStorageFile;
class DStorageExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "DStorageExt";
    struct File : public ResourceCreationInfo {
        size_t size_bytes;
    };

    virtual File open_file_handle(luisa::string_view path) noexcept = 0;
    virtual void close_file_handle(uint64_t handle) noexcept = 0;
    virtual std::pair<DeviceInterface *, ResourceCreationInfo> create_stream_handle() noexcept = 0;
    [[nodiscard]] Stream create_stream() noexcept;
    [[nodiscard]] DStorageFile open_file(luisa::string_view path) noexcept;
};
}// namespace luisa::compute