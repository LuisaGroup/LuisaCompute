//
// Created by Mike Smith on 2021/9/24.
//

#pragma once

#include <queue>
#include <asio.hpp>

#include <network/render_tile.h>
#include <network/render_buffer.h>

namespace luisa::compute {

class RenderScheduler;

class RenderWorkerSession : public std::enable_shared_from_this<RenderWorkerSession> {

private:
    asio::ip::tcp::socket _socket;
    RenderScheduler *_scheduler;
    std::vector<RenderTile> _working_tiles;

private:
    static void _receive(std::shared_ptr<RenderWorkerSession> self) noexcept;
    static void _receive_tile(std::shared_ptr<RenderWorkerSession> self, RenderTile tile, BinaryBuffer buffer) noexcept;
    void _cancel_tile(RenderTile tile) noexcept;
    void _finish_tile(RenderTile tile, std::span<const std::byte> data) noexcept;

public:
    explicit RenderWorkerSession(RenderScheduler *scheduler) noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return _socket.is_open(); }
    [[nodiscard]] auto working_tile_count() const noexcept { return _working_tiles.size(); }
    [[nodiscard]] auto &socket() noexcept { return _socket; }
    void render(RenderTile tile) noexcept;
    void stop() noexcept;
    void close() noexcept;
};

}
