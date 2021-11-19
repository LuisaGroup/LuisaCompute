//
// Created by Mike on 2021/11/19.
//

#pragma once

#include <memory>
#include <functional>
#include <queue>

#include <asio.hpp>

#include <core/basic_types.h>
#include <network/render_tile.h>
#include <network/binary_buffer.h>

namespace luisa::compute {

class RenderConfig;

class RenderWorker : public std::enable_shared_from_this<RenderWorker> {

public:
    using ConfigHandler = std::function<void(const RenderConfig &)>;
    using RenderHandler = std::function<void(const RenderTile &)>;

private:
    asio::io_context _context;
    asio::ip::tcp::socket _socket;
    asio::system_timer _timer;
    asio::ip::tcp::endpoint _server;
    std::mutex _sending_queue_mutex;
    std::queue<std::tuple<std::vector<float4>, RenderTile, uint2>> _sending_queue;
    ConfigHandler _config_handler;
    RenderHandler _render_handler;

private:
    static void _receive(std::shared_ptr<RenderWorker> self) noexcept;
    static void _send(std::shared_ptr<RenderWorker> self) noexcept;
    static void _receive_config_command(std::shared_ptr<RenderWorker> self, BinaryBuffer buffer) noexcept;
    static void _receive_render_command(std::shared_ptr<RenderWorker> self, BinaryBuffer buffer) noexcept;

public:
    RenderWorker(const std::string &server_ip, uint16_t server_port) noexcept;
    ~RenderWorker() noexcept;
    [[nodiscard]] static std::shared_ptr<RenderWorker> create(const std::string &server_ip, uint16_t server_port) noexcept;
    RenderWorker &set_config_handler(ConfigHandler handler) noexcept;
    RenderWorker &set_render_handler(RenderHandler handler) noexcept;
    void finish(const RenderTile &tile, std::vector<float4> result, uint2 tile_size) noexcept;
    void run() noexcept;
};

}// namespace luisa::compute
