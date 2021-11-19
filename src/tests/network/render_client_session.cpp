//
// Created by Mike Smith on 2021/11/19.
//

#include <network/render_server.h>
#include <network/render_client_session.h>

namespace luisa::compute {

void RenderClientSession::_send(std::shared_ptr<RenderClientSession> self) noexcept {

}

void RenderClientSession::_receive(std::shared_ptr<RenderClientSession> self) noexcept {

}

RenderClientSession::RenderClientSession(RenderServer *server) noexcept
    : _server{server},
      _socket{server->context()},
      _timer{server->context()} {}

void RenderClientSession::run() noexcept {
    _receive(shared_from_this());
    _send(shared_from_this());
}

}
