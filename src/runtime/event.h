//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#include <runtime/rhi/command.h>
#include <runtime/rhi/resource.h>

namespace luisa::compute {

class Device;

class LC_RUNTIME_API Event final : public Resource {

public:
    struct Signal {
        uint64_t handle;
    };
    struct Wait {
        uint64_t handle;
    };

private:
    friend class Device;
    explicit Event(DeviceInterface *device) noexcept;

public:
    Event() noexcept = default;
    using Resource::operator bool;
    Event(Event &&) noexcept = default;
    Event(Event const &) noexcept = delete;
    Event &operator=(Event &&) noexcept = default;
    Event &operator=(Event const &) noexcept = delete;
    [[nodiscard]] auto signal() const noexcept { return Signal{handle()}; }
    [[nodiscard]] auto wait() const noexcept { return Wait{handle()}; }
    void synchronize() const noexcept;
};

}// namespace luisa::compute
