//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#include <runtime/command.h>
#include <runtime/resource.h>

namespace luisa::compute {

class LC_RUNTIME_API Device;

class Event final : public Resource {

public:
    struct Signal { uint64_t handle; };
    struct Wait { uint64_t handle; };
    
private:
    friend class Device;
    explicit Event(Device::Interface *device) noexcept;

public:
    Event() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto signal() const noexcept { return Signal{handle()}; }
    [[nodiscard]] auto wait() const noexcept { return Wait{handle()}; }
    void synchronize() const noexcept;
};

}// namespace luisa::compute
