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

private:
    static void _send(std::shared_ptr<RenderClientSession> self) noexcept;
    static void _receive(std::shared_ptr<RenderClientSession> self) noexcept;

public:
    explicit RenderClientSession(RenderServer *server) noexcept;
    [[nodiscard]] auto &socket() noexcept { return _socket; }
    void run() noexcept;
};

}
