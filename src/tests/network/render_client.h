//
// Created by Mike Smith on 2021/11/19.
//

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <array>

#include <asio.hpp>

#include <core/basic_types.h>

namespace luisa::compute {

class RenderConfig;
class BinaryBuffer;

class RenderClient : public std::enable_shared_from_this<RenderClient> {

public:
    using Pixel = std::array<uint8_t, 3u>;
    using DisplayHandler = std::function<void(
        const RenderConfig & /* config */,
        size_t /* frame_count */,
        std::span<Pixel> /* pixels */)>;

private:
    asio::io_context _context;
    asio::ip::tcp::socket _socket;
    asio::system_timer _timer;
    asio::ip::tcp::endpoint _server;
    DisplayHandler _display;
    std::unique_ptr<RenderConfig> _sending_config;
    std::mutex _mutex;

private:
    static void _receive(std::shared_ptr<RenderClient> self) noexcept;
    static void _receive_frame(std::shared_ptr<RenderClient> self, BinaryBuffer buffer, const RenderConfig &config, size_t frame_count) noexcept;
    static void _send(std::shared_ptr<RenderClient> self) noexcept;

public:
    RenderClient(const char *server_ip, uint16_t port) noexcept;
    ~RenderClient() noexcept;
    [[nodiscard]] static std::shared_ptr<RenderClient> create(const char *server_ip, uint16_t port) noexcept;
    RenderClient &set_display_handler(DisplayHandler handler) noexcept;
    RenderClient &set_config(const RenderConfig &config) noexcept;
    void run() noexcept;
};

}// namespace luisa::compute