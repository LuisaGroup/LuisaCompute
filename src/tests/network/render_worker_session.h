//
// Created by Mike Smith on 2021/9/24.
//

#pragma once

#include <queue>
#include <variant>
#include <asio.hpp>

#include <network/render_tile.h>
#include <network/render_buffer.h>

namespace luisa::compute {

class RenderConfig;
class RenderScheduler;

class RenderWorkerSession : public std::enable_shared_from_this<RenderWorkerSession> {

public:
    static constexpr auto invalid_render_id = std::numeric_limits<uint32_t>::max();

private:
    asio::ip::tcp::socket _socket;
    asio::system_timer _timer;
    RenderScheduler *_scheduler;
    std::vector<RenderTile> _rendering_tiles;
    std::queue<std::variant<RenderConfig, RenderTile>> _pending_commands;
    uint32_t _render_id{invalid_render_id};
    bool _error_occurred{false};

private:
    static void _send(std::shared_ptr<RenderWorkerSession> self) noexcept;
    static void _receive(std::shared_ptr<RenderWorkerSession> self) noexcept;
    static void _receive_tile(std::shared_ptr<RenderWorkerSession> self, RenderTile tile, BinaryBuffer buffer) noexcept;
    void _finish_tile(RenderTile tile, std::span<const std::byte> data) noexcept;

public:
    explicit RenderWorkerSession(RenderScheduler *scheduler) noexcept;
    ~RenderWorkerSession() noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return !_error_occurred && _socket.is_open(); }
    [[nodiscard]] auto working_item_count() const noexcept { return _rendering_tiles.size() + _pending_commands.size(); }
    [[nodiscard]] auto &socket() noexcept { return _socket; }
    void dispatch(const RenderConfig &config, RenderTile tile) noexcept;
    void run() noexcept;
    void close() noexcept;
};

}// namespace luisa::compute
