//
// Created by Mike Smith on 2021/9/24.
//

#pragma once

#include <memory>
#include <vector>

#include <asio.hpp>

namespace luisa::compute {

class RenderBuffer;
class RenderConfig;
class RenderScheduler;
class BinaryBuffer;
class RenderWorkerSession;
class RenderClientSession;

class RenderServer : public std::enable_shared_from_this<RenderServer> {

private:
    asio::io_context _context;
    asio::system_timer _purge_timer;
    asio::ip::tcp::acceptor _worker_acceptor;
    asio::ip::tcp::acceptor _client_acceptor;
    std::shared_ptr<RenderScheduler> _scheduler;
    std::unique_ptr<RenderConfig> _config;
    std::vector<float4> _accum_buffer;
    uint32_t _render_id{};
    size_t _frame_count{};
    std::vector<std::shared_ptr<RenderClientSession>> _clients;
    std::shared_ptr<BinaryBuffer> _sending_buffer;
    size_t _sending_frame_count{};

private:
    static void _accept_workers(std::shared_ptr<RenderServer> self) noexcept;
    static void _accept_clients(std::shared_ptr<RenderServer> self) noexcept;
    static void _purge_clients(std::shared_ptr<RenderServer> self) noexcept;
    void _close() noexcept;
    void _purge() noexcept;

public:
    RenderServer(uint16_t worker_port, uint16_t client_port) noexcept;
    ~RenderServer() noexcept;
    [[nodiscard]] static std::shared_ptr<RenderServer> create(uint16_t worker_port, uint16_t client_port) noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return _worker_acceptor.is_open() && _client_acceptor.is_open(); }
    [[nodiscard]] auto &context() noexcept { return _context; }
    [[nodiscard]] auto &context() const noexcept { return _context; }
    void accumulate(RenderBuffer buffer) noexcept;
    void run() noexcept;
    [[nodiscard]] auto frame_count() const noexcept { return _frame_count; }
    [[nodiscard]] auto config() const noexcept { return _config.get(); }
    void set_config(const RenderConfig &config) noexcept;
    [[nodiscard]] std::shared_ptr<BinaryBuffer> sending_buffer()noexcept;
};

}// namespace luisa::compute
