//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#include <runtime/command.h>
#include <runtime/device.h>

namespace luisa::compute {

class Device;

class Event : concepts::Noncopyable {

public:
    struct Signal { uint64_t handle; };
    struct Wait { uint64_t handle; };

private:
    Device::Interface *_device;
    uint64_t _handle;
    
private:
    friend class Device;
    explicit Event(Device &device) noexcept;

public:
    ~Event() noexcept;
    
    Event(Event &&another) noexcept;
    Event &operator=(Event &&rhs) noexcept;
    
    [[nodiscard]] auto signal() const noexcept { return Signal{_handle}; }
    [[nodiscard]] auto wait() const noexcept { return Wait{_handle}; }
    void synchronize() const noexcept;
};

}// namespace luisa::compute
