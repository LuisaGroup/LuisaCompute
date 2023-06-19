//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/stream_event.h>

namespace luisa::compute {

class Device;

class LC_RUNTIME_API Event final : public Resource {

public:
    struct LC_RUNTIME_API Signal {
        uint64_t handle;
        void operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept;
    };
    struct LC_RUNTIME_API Wait {
        uint64_t handle;
        void operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept;
    };

private:
    friend class Device;
    explicit Event(DeviceInterface *device) noexcept;

public:
    Event() noexcept = default;
    ~Event() noexcept override;
    using Resource::operator bool;
    Event(Event &&) noexcept = default;
    Event(Event const &) noexcept = delete;
    Event &operator=(Event &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Event &operator=(Event const &) noexcept = delete;
    [[nodiscard]] auto signal() const noexcept { return Signal{handle()}; }
    [[nodiscard]] auto wait() const noexcept { return Wait{handle()}; }
    [[nodiscard]] bool is_complete() const noexcept;
    void synchronize() const noexcept;
};

LUISA_MARK_STREAM_EVENT_TYPE(Event::Signal)
LUISA_MARK_STREAM_EVENT_TYPE(Event::Wait)

}// namespace luisa::compute
