#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "remote_protocol.h"

namespace luisa::compute::remote {

struct Response {
    MessageKind request_kind{};
    Status status{Status::CONNECTION_CLOSED};
    luisa::string message;
    luisa::vector<std::byte> body;

    [[nodiscard]] explicit operator bool() const noexcept {
        return status == Status::OK;
    }
};

class Connection final {

public:
    using NotificationHandler = std::function<void(
        MessageKind, uint64_t, luisa::span<const std::byte>)>;
    using CloseHandler = std::function<void(luisa::string_view)>;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;

public:
    explicit Connection(ProtocolLimits limits = {}) noexcept;
    ~Connection() noexcept;
    Connection(Connection const &) = delete;
    Connection(Connection &&) = delete;
    Connection &operator=(Connection const &) = delete;
    Connection &operator=(Connection &&) = delete;

    [[nodiscard]] bool connect(
        luisa::string_view host,
        uint16_t port,
        std::chrono::milliseconds timeout,
        luisa::string &error) noexcept;

    [[nodiscard]] Response request(
        MessageKind kind,
        luisa::span<const std::byte> payload,
        std::chrono::milliseconds timeout) noexcept;

    [[nodiscard]] bool notify(
        MessageKind kind,
        uint64_t request_id,
        luisa::span<const std::byte> payload,
        luisa::string &error) noexcept;

    void set_notification_handler(NotificationHandler handler) noexcept;
    void set_close_handler(CloseHandler handler) noexcept;
    void close() noexcept;

    [[nodiscard]] bool connected() const noexcept;
    [[nodiscard]] luisa::string error() const noexcept;
};

}// namespace luisa::compute::remote
