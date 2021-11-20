//
// Created by Mike Smith on 2021/9/24.
//

#include <network/binary_buffer.h>
#include <network/render_scheduler.h>
#include <network/render_worker_session.h>
#include <network/render_config.h>

namespace luisa::compute {

RenderWorkerSession::RenderWorkerSession(RenderScheduler *scheduler) noexcept
    : _socket{scheduler->context()},
      _timer{scheduler->context()},
      _scheduler{scheduler} {}

inline void RenderWorkerSession::_finish_tile(RenderTile tile, std::span<const std::byte> data) noexcept {
    if (auto iter = std::find(_rendering_tiles.begin(), _rendering_tiles.end(), tile);
        iter != _rendering_tiles.end()) {
        _rendering_tiles.erase(iter);
        _scheduler->accumulate(tile, data);
    }
}

inline void RenderWorkerSession::_receive(std::shared_ptr<RenderWorkerSession> self) noexcept {
    if (*self) {
        BinaryBuffer buffer;
        buffer.write_skip(sizeof(RenderTile));
        auto asio_buffer = buffer.asio_buffer();
        auto &&asio_socket = self->_socket;
        asio::async_read(
            asio_socket, asio_buffer,
            [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
                if (auto size = buffer.read_size();
                    error || size <= sizeof(RenderTile)) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Error when receiving message in RenderWorkerSession: {}.",
                        error.message());
                    self->_error_occurred = true;
                } else {// good tile, receive it
                    RenderTile tile;
                    buffer.read(tile);
                    buffer.write_skip(size - sizeof(RenderTile));
                    _receive_tile(std::move(self), tile, std::move(buffer));
                }
            });
    }
}

inline void RenderWorkerSession::_receive_tile(std::shared_ptr<RenderWorkerSession> self, RenderTile tile, BinaryBuffer buffer) noexcept {
    auto asio_buffer = buffer.asio_buffer_tail();
    auto &&asio_socket = self->_socket;
    asio::async_read(
        asio_socket, asio_buffer,
        [self = std::move(self), tile, buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
            if (error) {
                LUISA_WARNING_WITH_LOCATION(
                    "Error when receiving result tile in RenderWorkerSession: {}.",
                    error.message());
                self->_error_occurred = true;
            } else {
                self->_finish_tile(tile, buffer.tail());
                _receive(std::move(self));
            }
        });
}

void RenderWorkerSession::dispatch(const RenderConfig &config, RenderTile tile) noexcept {
    if (config.render_id() != _render_id) {
        _render_id = config.render_id();
        _pending_commands = {};
        _pending_commands.emplace(config);
    }
    _pending_commands.emplace(tile);
}

void RenderWorkerSession::close() noexcept {
    while (!_pending_commands.empty()) {
        auto command = _pending_commands.front();
        _pending_commands.pop();
        std::visit(
            [this](auto &&item) noexcept {
                using T = std::remove_cvref_t<decltype(item)>;
                if constexpr (std::is_same_v<T, RenderTile>) {
                    _scheduler->recycle(item);
                }
            },
            command);
    }
    for (auto t : _rendering_tiles) {
        _scheduler->recycle(t);
    }
    _rendering_tiles.clear();
    asio::error_code error;
    _socket.close(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when closing RenderWorkerSession: {}.",
            error.message());
    }
    _timer.cancel(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when cancelling timer: {}.",
            error.message());
    }
}

void RenderWorkerSession::run() noexcept {
    _send(shared_from_this());
    _receive(shared_from_this());
}

void RenderWorkerSession::_send(std::shared_ptr<RenderWorkerSession> self) noexcept {
    if (auto &&s = *self) {
        if (s._pending_commands.empty()) {
            // check later...
            using namespace std::chrono_literals;
            s._timer.expires_after(1ms);
            s._timer.async_wait([self = std::move(self)](asio::error_code error) mutable noexcept {
                if (error) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Error occurred while waiting for timer: {}.",
                        error.message());
                    self->_error_occurred = true;
                } else {
                    _send(std::move(self));
                }
            });
        } else {
            // encode command buffer
            auto item = s._pending_commands.front();
            s._pending_commands.pop();
            auto buffer = std::visit(
                [&s](auto &&item) noexcept {
                    using T = std::remove_cvref_t<decltype(item)>;
                    BinaryBuffer buffer;
                    if constexpr (std::is_same_v<T, RenderConfig>) {
                        std::array command{'C', 'O', 'N', 'F', 'I', 'G'};
                        buffer.write(command).write(item).write_size();
                    } else if constexpr (std::is_same_v<T, RenderTile>) {
                        std::array command{'R', 'E', 'N', 'D', 'E', 'R'};
                        s._rendering_tiles.emplace_back(item);
                        buffer.write(command).write(item).write_size();
                    } else {
                        static_assert(always_false_v<T>);
                    }
                    return buffer;
                },
                item);
            // send command
            auto asio_buffer = buffer.asio_buffer();
            asio::async_write(
                s._socket, asio_buffer,
                [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
                    if (error) {
                        LUISA_WARNING_WITH_LOCATION(
                            "Error occurred when sending work in RenderWorkerSession: {}.",
                            error.message());
                        self->_error_occurred = true;
                    } else {
                        _send(std::move(self));
                    }
                });
        }
    }
}

RenderWorkerSession::~RenderWorkerSession() noexcept {
    close();
}

}// namespace luisa::compute
