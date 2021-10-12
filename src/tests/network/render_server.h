//
// Created by Mike Smith on 2021/9/24.
//

#pragma once

#include <memory>
#include <vector>

#include <asio.hpp>
#include <network/render_scheduler.h>

namespace luisa::compute {

class RenderBuffer;
class RenderWorkerSession;

class RenderServer : public std::enable_shared_from_this<RenderServer> {

private:
    asio::io_context _context;
    asio::ip::tcp::acceptor _worker_acceptor;
    asio::ip::tcp::acceptor _client_acceptor;
    RenderScheduler _scheduler;
    std::vector<float4> _accum_buffer;
    size_t _frame_count{};

private:
    static void _accept_workers(std::shared_ptr<RenderServer> self) noexcept;
    static void _accept_clients(std::shared_ptr<RenderServer> self) noexcept;
    void _close() noexcept;
    void _send_to_clients(std::shared_ptr<BinaryBuffer> buffer) noexcept;

public:
    RenderServer(uint16_t worker_port, uint16_t client_port) noexcept;
    ~RenderServer() noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return _worker_acceptor.is_open(); }
    [[nodiscard]] auto &context() noexcept { return _context; }
    [[nodiscard]] auto &context() const noexcept { return _context; }
    void process(size_t, RenderBuffer buffer) noexcept;
    void run() noexcept { _context.run(); }
};

}// namespace luisa::compute
