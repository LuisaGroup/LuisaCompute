//
// Created by Mike Smith on 2021/11/19.
//

#pragma once

#include <memory>
#include <asio.hpp>

namespace luisa::compute {

class RenderServer;

class RenderClientSession : public std::enable_shared_from_this<RenderClientSession> {

private:
    RenderServer *_server;
    asio::ip::tcp::socket _socket;
    asio::system_timer _timer;
    uint32_t _last_render_id{std::numeric_limits<uint32_t>::max()};
    uint32_t _last_frame_count{};

private:
    static void _send(std::shared_ptr<RenderClientSession> self) noexcept;
    static void _receive(std::shared_ptr<RenderClientSession> self) noexcept;

public:
    explicit RenderClientSession(RenderServer *server) noexcept;
    ~RenderClientSession() noexcept;
    [[nodiscard]] auto &socket() noexcept { return _socket; }
    void run() noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return _socket.is_open(); }
    void close() noexcept;
};

}
