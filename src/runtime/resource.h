//
// Created by Mike Smith on 2021/7/30.
//

#pragma once

#include <type_traits>
#include <runtime/device.h>

namespace luisa::compute {

class LC_RUNTIME_API Resource : public luisa::enable_shared_from_this<Resource> {

public:
    enum struct Tag : uint32_t {
        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        MESH,
        ACCEL,
        STREAM,
        EVENT,
        SHADER,
        SWAP_CHAIN
    };

    using Handle = luisa::shared_ptr<Resource>;

private:
    Device::Handle _device{nullptr};
    uint64_t _handle{0u};
    Tag _tag{};

protected:
    void _destroy() noexcept;

public:
    Resource() noexcept = default;
    Resource(Device::Interface *device, Tag tag, uint64_t handle) noexcept;
    virtual ~Resource() noexcept { _destroy(); }
    Resource(Resource &&) noexcept = default;
    Resource(const Resource &) noexcept = delete;
    Resource &operator=(Resource &&) noexcept;
    Resource &operator=(const Resource &) noexcept = delete;
    [[nodiscard]] auto device() const noexcept { return _device.get(); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }
};

}// namespace luisa::compute
