//
// Created by Mike Smith on 2021/11/19.
//

#include <network/render_server.h>
#include <network/render_client_session.h>

namespace luisa::compute {

void RenderClientSession::_send(std::shared_ptr<RenderClientSession> self) noexcept {
    if (auto &&s = *self) {
        if (auto buffer = s._server->sending_buffer();
            buffer != nullptr &&
            (s._server->config()->render_id() != s._last_render_id ||
             s._server->frame_count() != s._last_frame_count)) {

            // update
            s._last_render_id = s._server->config()->render_id();
            s._last_frame_count = s._server->frame_count();

            // send
            auto asio_buffer = buffer->asio_buffer();
            asio::async_write(
                s._socket, asio_buffer,
                [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
                    if (error) {
                        LUISA_WARNING_WITH_LOCATION(
                            "Error when sending to client: {}.",
                            error.message());
                        self->close();
                    } else {
                        _send(std::move(self));
                    }
                });
        } else {
            using namespace std::chrono_literals;
            s._timer.expires_after(1ms);
            s._timer.async_wait([self = std::move(self)](asio::error_code error) mutable noexcept {
                if (error) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Error when waiting for timer: {}.",
                        error.message());
                    self->close();
                } else {
                    _send(std::move(self));
                }
            });
        }
    }
}

void RenderClientSession::_receive(std::shared_ptr<RenderClientSession> self) noexcept {
    if (auto &&s = *self) {
        BinaryBuffer buffer;
        buffer.write_skip(sizeof(RenderConfig));
        auto asio_buffer = buffer.asio_buffer();
        asio::async_read(
            s._socket, asio_buffer,
            [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
                if (error) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Error when reading from client: {}.",
                        error.message());
                    self->close();
                } else if (auto size = buffer.read_size(); size != sizeof(RenderConfig)) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Invalid render config size: {} (expected {}).",
                        size, sizeof(RenderConfig));
                    self->close();
                } else {
                    RenderConfig config;
                    buffer.read(config);
                    self->_server->set_config(config);
                    _receive(std::move(self));
                }
            });
    }
}

RenderClientSession::RenderClientSession(RenderServer *server) noexcept
    : _server{server},
      _socket{server->context()},
      _timer{server->context()} {}

void RenderClientSession::run() noexcept {
    _receive(shared_from_this());
    _send(shared_from_this());
}

RenderClientSession::~RenderClientSession() noexcept { close(); }

void RenderClientSession::close() noexcept {
    asio::error_code error;
    _socket.close(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when closing socket: {}.",
            error.message());
    }
    _timer.cancel(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when cancelling timer: {}.",
            error.message());
    }
}

}// namespace luisa::compute
