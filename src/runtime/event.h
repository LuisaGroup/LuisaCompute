//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#include <runtime/rhi/command.h>
#include <runtime/rhi/resource.h>
#include <runtime/rhi/stream_tag.h>

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
    void synchronize() const noexcept;
};
// std::true_type inheritage may break MSVC
template<>
struct StreamEvent<Event::Signal> : std::true_type {
    LC_RUNTIME_API static void execute(
        DeviceInterface *device,
        uint64_t stream_handle,
        const Event::Signal &signal) noexcept;
};
template<>
struct StreamEvent<Event::Wait> : std::true_type {
    LC_RUNTIME_API static void execute(
        DeviceInterface *device,
        uint64_t stream_handle,
        const Event::Wait &wait) noexcept;
};
}// namespace luisa::compute
