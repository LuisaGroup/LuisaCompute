//
// Created by Mike Smith on 2021/9/17.
//

#include <ctime>
#include <memory>
#include <span>
#include <queue>
#include <mutex>
#include <thread>
#include <deque>
#include <condition_variable>

#include <asio.hpp>

#include <core/basic_types.h>
#include <core/allocator.h>
#include <core/logging.h>

class RenderServer;

constexpr auto width = 512u;
constexpr auto height = 512u;

class RenderBuffer;
class RenderSchedule;

class RenderTile {

private:
    RenderBuffer *_render_buffer;
    size_t _index;
    std::span<uint8_t> _storage;
    luisa::uint2 _offset;
    luisa::uint2 _size;

private:
    friend class RenderBuffer;
    RenderTile(RenderBuffer *buffer, size_t index, std::span<uint8_t> storage, luisa::uint2 offset, luisa::uint2 size) noexcept
        : _render_buffer{buffer}, _index{index}, _storage{storage}, _offset{offset}, _size{size} {}

public:
    void finish() noexcept;
    void cancel() noexcept;
};

class RenderBuffer {

private:
    RenderSchedule *_schedule;
    std::vector<uint8_t> _framebuffer;// tiled
    luisa::uint2 _frame_size;
    luisa::uint2 _tile_size;
    size_t _total{0u};
    size_t _finished{0u};

private:
    friend class RenderTile;
    friend class RenderSchedule;
    RenderBuffer(RenderSchedule *schedule, luisa::uint2 frame_size, luisa::uint2 tile_size) noexcept;
    void finish(RenderTile tile) noexcept;
    void cancel(RenderTile tile) noexcept;

public:
    [[nodiscard]] auto done() const noexcept { return _finished == _total; }
};

void RenderTile::finish() noexcept { _render_buffer->finish(*this); }
void RenderTile::cancel() noexcept { _render_buffer->cancel(*this); }

class RenderSchedule {

private:
    luisa::uint2 _frame_size;
    luisa::uint2 _tile_size;
    size_t _sample_count;
    size_t _total_sample_count;
    std::deque<RenderTile> _tiles;
    std::queue<std::unique_ptr<RenderBuffer>> _buffers;

private:
    friend class RenderBuffer;
    void _emplace_front(RenderTile tile) noexcept { _tiles.emplace_front(tile); }
    void _emplace_back(RenderTile tile) noexcept { _tiles.emplace_back(tile); }

public:
    RenderSchedule(luisa::uint2 frame_size, luisa::uint2 tile_size = {64u, 64u}, size_t spp = 0u) noexcept
        : _frame_size{frame_size},
          _tile_size{tile_size},
          _sample_count{0u},
          _total_sample_count{spp} {}

    [[nodiscard]] auto fetch_tile() noexcept -> std::optional<RenderTile> {
        if (_tiles.empty()) {
            if (_total_sample_count != 0u &&
                _sample_count >= _total_sample_count) {
                // no more tiles...
                return std::nullopt;
            }
            _buffers.emplace(std::unique_ptr<RenderBuffer>{
                new RenderBuffer{this, _frame_size, _tile_size}});
        }
        auto tile = _tiles.front();
        _tiles.pop_front();
        return std::make_optional(tile);
    }

    [[nodiscard]] auto fetch_buffer() noexcept -> std::unique_ptr<RenderBuffer> {
        std::unique_ptr<RenderBuffer> buffer;
        if (!_buffers.empty() && _buffers.front()->done()) {
            buffer = std::move(_buffers.front());
            _buffers.pop();
        }
        return buffer;
    }
};

RenderBuffer::RenderBuffer(RenderSchedule *schedule, luisa::uint2 frame_size, luisa::uint2 tile_size) noexcept
    : _schedule{schedule},
      _framebuffer(frame_size.x * frame_size.y * 4u),
      _frame_size{frame_size}, _tile_size{tile_size} {
    auto tile_count = (frame_size + tile_size - 1u) / tile_size;
    auto bytes_per_tile = _tile_size.x * _tile_size.y * 4u;
    auto bytes_per_image = bytes_per_tile * tile_count.x * tile_count.y;
    _framebuffer.resize(bytes_per_image);
    auto p = _framebuffer.data();
    for (auto y = 0u; y < _frame_size.y; y += _tile_size.y) {
        for (auto x = 0u; x < _frame_size.x; x += _tile_size.x) {
            _schedule->_emplace_back(RenderTile{this, _total, {p, bytes_per_tile}, {x, y}, _tile_size});
            p += bytes_per_tile;
            _total++;
        }
    }
}

void RenderBuffer::finish(RenderTile tile) noexcept {
    _finished++;
}

void RenderBuffer::cancel(RenderTile tile) noexcept {
    _schedule->_emplace_front(tile);
}

class RenderWorkerSession : public std::enable_shared_from_this<RenderWorkerSession> {

private:
    asio::ip::tcp::socket _socket;
    asio::system_timer _dispatch_timer;
    RenderSchedule *_schedule;
    size_t _working_tile_count;
    size_t _max_working_tile_count;

    void _wait_for_work() noexcept {
        using namespace std::chrono_literals;
        _dispatch_timer.expires_after(1ms);
        _dispatch_timer.async_wait([this](asio::error_code error) noexcept {
            if (error) { LUISA_INFO("Error when waiting for work: {}.", error.message()); }
            _dispatch();
        });
    }

    void _dispatch() noexcept {
        while (*this && _working_tile_count < _max_working_tile_count) {
            if (auto tile = _schedule->fetch_tile()) {
                _send_work(*tile);
            } else {// no more tiles to render, check after some time
                _wait_for_work();
            }
        }
    }

    void _cancel_tile(RenderTile tile) noexcept {
        tile.cancel();
        _working_tile_count--;
    }

    void _send_work(RenderTile tile) noexcept {
        std::vector<std::byte> command;// TODO: encode command
        auto buffer = asio::buffer(command);
        asio::async_write(_socket, buffer, [self = shared_from_this(), tile, command = std::move(command)](asio::error_code error, size_t bytes_sent) mutable noexcept {
            if (error) {// broken connection, cancel this tile...
                LUISA_INFO("Failed to send work: {}.", error.message());
                self->_cancel_tile(tile);
            } else {// successfully send work, wait for result
                self->_receive_result(tile);
            }
        });
        _working_tile_count++;
    }

    void _receive_result(RenderTile tile) noexcept {
        std::vector<std::byte> response(4u);// TODO: size?
        auto buffer = asio::buffer(response);
        asio::async_read(_socket, buffer, [self = shared_from_this(), tile, response = std::move(response)](asio::error_code error, size_t bytes_received) noexcept {
            if (error) {
                LUISA_INFO("Error when receiving result: {}.", error.message());
                self->_cancel_tile(tile);
            } else {
                //  TODO: receive tile

                self->_working_tile_count--;
            }
            self->_dispatch();
        });
    }

public:
    RenderWorkerSession(asio::io_context &context, RenderSchedule *schedule, size_t max_tile_count = 4u) noexcept
        : _socket{context},
          _dispatch_timer{context},
          _schedule{schedule},
          _working_tile_count{0u},
          _max_working_tile_count{max_tile_count} {}
    [[nodiscard]] explicit operator bool() const noexcept { return _socket.is_open(); }
};

class RenderServer {

private:
    asio::io_context _context;
    asio::ip::tcp::acceptor _acceptor;
    asio::system_timer _sending_timer;
    std::vector<std::unique_ptr<asio::ip::tcp::socket>> _clients;
    std::vector<RenderWorkerSession> _workers;

    std::mutex _framebuffer_mutex;
    std::atomic_bool _framebuffer_dirty{false};
    std::vector<uint8_t> _sending_framebuffer;
    std::vector<uint8_t> _framebuffer;

private:
    void _purge_workers() noexcept {
        _workers.erase(
            std::remove_if(_workers.begin(), _workers.end(), [](auto &&w) noexcept {
                return !w;
            }),
            _workers.end());
    }

    void _purge_clients() noexcept {
        _clients.erase(
            std::remove_if(_clients.begin(), _clients.end(), [](auto &&c) noexcept {
                return !c->is_open();
            }),
            _clients.end());
    }

    void receive_from_one_worker(asio::ip::tcp::socket &worker) noexcept {
    }

    void _send_to_all_clients() noexcept {
        _sending_timer.async_wait([this](asio::error_code error) noexcept {
            if (error) {
                LUISA_INFO("Timer error: {}.", error.message());
            } else {
                _purge_clients();
                if (!_clients.empty() && _framebuffer_dirty.exchange(false)) {
                    // copy current framebuffer
                    {
                        std::scoped_lock lock{_framebuffer_mutex};
                        _sending_framebuffer = _framebuffer;
                    }
                    for (auto &&client : _clients) {
                        asio::async_write(*client, asio::buffer(_sending_framebuffer), [](asio::error_code error, size_t transferred_bytes) noexcept {
                            LUISA_INFO(
                                "Error after transferring {} byte{}: {}.",
                                transferred_bytes, transferred_bytes > 1u ? "s" : "", error.message());
                        });
                    }
                }
            }
            using namespace std::chrono_literals;
            _sending_timer.expires_after(1ms);
            _send_to_all_clients();
        });
    }

    void _accept() noexcept {
        auto socket = std::make_unique<asio::ip::tcp::socket>(_context);
        auto p_socket = socket.get();
        _acceptor.async_accept(*p_socket, [this, socket = std::move(socket)](asio::error_code error) mutable noexcept {
            if (error) {
                LUISA_INFO("Error: {}.", error.message());
            } else {
                std::array<char, 7> identifier{};
                asio::read(*socket, asio::buffer(identifier), error);
                if (!error) {
                    auto address = socket->remote_endpoint().address().to_string();
                    auto port = socket->remote_endpoint().port();
                    if (auto id = std::string_view{identifier.data()}; id == "client") {
                        LUISA_INFO(
                            "Connection from client {}:{}.",
                            address, port);
                        _clients.emplace_back(std::move(socket));
                    } else if (id == "worker") {
                        LUISA_INFO(
                            "Connection from worker {}:{}.",
                            address, port);
                        create_worker_session(*this, std::move(socket));
                    } else {
                        LUISA_INFO("Invalid connection from {}:{}. Closing.", address, port);
                    }
                }
            }
            _accept();
        });
    }

public:
    RenderServer() noexcept
        : _acceptor{_context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 13)},
          _sending_timer{_context},
          _framebuffer(width * height * 4u) {
        LUISA_INFO(
            "Listening to {}:{}.",
            _acceptor.local_endpoint().address().to_string(),
            _acceptor.local_endpoint().port());
        _accept();
        _send_to_all_clients();
    }
    [[nodiscard]] auto &context() noexcept { return _context; }

    [[nodiscard]] auto read_framebuffer(uint32_t &version, std::vector<uint8_t> &output) noexcept {
        auto should_update = version < _framebuffer_version.load();
        if (should_update) {
            std::scoped_lock lock{_framebuffer_mutex};
            output = _framebuffer;
            version = _framebuffer_version.load();
        }
        return should_update;
    }

    template<typename F>
    void update_framebuffer(F &&f) noexcept {
        {
            std::scoped_lock lock{_framebuffer_mutex};
            std::invoke(std::forward<F>(f), _framebuffer);
        }
        _framebuffer_version++;
    }

    template<typename F>
    void create_thread(F &&f) noexcept {
        _sessions.emplace_back(std::async(std::launch::async, std::forward<F>(f)));
    }

    void run() noexcept {
        asio::error_code error;
        _context.run(error);
        LUISA_INFO("Error: {}.", error.message());
    }
};

void create_client_session(RenderServer &server, std::shared_ptr<asio::ip::tcp::socket> socket) noexcept {
    server.create_thread([&server, socket = std::move(socket)] {
        asio::error_code error;
        auto remote = socket->remote_endpoint();
        auto framebuffer_version = 0u;
        std::vector<uint8_t> framebuffer_copy(width * height * 4u);
        while (!error) {
            if (server.read_framebuffer(framebuffer_version, framebuffer_copy)) {
                asio::write(*socket, asio::buffer(framebuffer_copy), error);
            } else {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(1ms);
            }
        }
        LUISA_INFO(
            "Disconnected from client {}:{} (error: {}).",
            remote.address().to_string(),
            remote.port(),
            error.message());
    });
}

void create_worker_session(RenderServer &server, std::shared_ptr<asio::ip::tcp::socket> socket) noexcept {
    server.create_thread([&server, socket = std::move(socket)] {
        asio::error_code error;
        auto remote = socket->remote_endpoint();
        std::vector<uint8_t> framebuffer(width * height * 4u);
        while (!error) {
            if (asio::read(*socket, asio::buffer(framebuffer), error); !error) {
                server.update_framebuffer([&framebuffer](auto &fb) { fb = framebuffer; });
            }
        }
        LUISA_INFO(
            "Disconnected from worker {}:{} (error: {}).",
            remote.address().to_string(),
            remote.port(),
            error.message());
    });
}

int main() {
    RenderServer server;
    server.run();
}
