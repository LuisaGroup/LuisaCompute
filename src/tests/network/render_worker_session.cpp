//
// Created by Mike Smith on 2021/9/24.
//

#include <network/binary_buffer.h>
#include <network/render_scheduler.h>
#include <network/render_worker_session.h>

namespace luisa::compute {

RenderWorkerSession::RenderWorkerSession(RenderScheduler *scheduler) noexcept
    : _socket{scheduler->context()}, _scheduler{scheduler} {}

inline void RenderWorkerSession::_finish_tile(RenderTile tile, std::span<const std::byte> data) noexcept {
    if (auto iter = std::find(_working_tiles.begin(), _working_tiles.end(), tile);
        iter != _working_tiles.end()) {
        _working_tiles.erase(iter);
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
                    self->close();
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
                self->close();
            } else {
                self->_finish_tile(tile, buffer.tail());
                _receive(std::move(self));
            }
        });
}

void RenderWorkerSession::render(RenderTile tile) noexcept {
    BinaryBuffer buffer;
    buffer.write(tile).write_size();
    _working_tiles.emplace_back(tile);
    auto asio_buffer = buffer.asio_buffer();
    asio::async_write(
        _socket, asio_buffer,
        [self = shared_from_this(), tile, buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
            if (error) {
                LUISA_WARNING_WITH_LOCATION(
                    "Error occurred when sending work in RenderWorkerSession: {}.",
                    error.message());
                self->close();
            }
        });
}

void RenderWorkerSession::close() noexcept {
    for (auto t : _working_tiles) {
        _scheduler->recycle(t);
    }
    _working_tiles.clear();
    asio::error_code error;
    _socket.close(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when closing RenderWorkerSession: {}.",
            error.message());
    }
}

void RenderWorkerSession::run() noexcept {
    _receive(shared_from_this());
}

}// namespace luisa::compute
