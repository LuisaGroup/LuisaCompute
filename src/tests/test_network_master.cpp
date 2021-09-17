//
// Created by Mike Smith on 2021/9/17.
//

#include <ctime>
#include <memory>
#include <span>

#include <asio.hpp>

#include <core/basic_types.h>
#include <core/allocator.h>
#include <core/logging.h>

class RenderServer;

constexpr auto width = 512u;
constexpr auto height = 512u;

void create_client_session(RenderServer &server, std::shared_ptr<asio::ip::tcp::socket> socket) noexcept;
void create_worker_session(RenderServer &server, std::shared_ptr<asio::ip::tcp::socket> socket) noexcept;

class RenderServer {

private:
    asio::io_context _context;
    asio::ip::tcp::acceptor _acceptor;
    std::vector<std::future<void>> _sessions;
    std::mutex _framebuffer_mutex;
    std::atomic_uint _framebuffer_version{0u};
    std::vector<uint8_t> _framebuffer;

private:
    void _purge() noexcept {
        _sessions.erase(
            std::remove_if(_sessions.begin(), _sessions.end(), [](auto &&w) noexcept {
                using namespace std::chrono_literals;
                return w.wait_for(0ns) == std::future_status::ready;
            }),
            _sessions.end());
    }

    void _accept() noexcept {
        auto socket = std::make_shared<asio::ip::tcp::socket>(_context);
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
                        create_client_session(*this, std::move(socket));
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
            _purge();
            _accept();
        });
    }

public:
    RenderServer() noexcept
        : _acceptor{_context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 13)},
          _framebuffer(width * height * 4u) {
        LUISA_INFO(
            "Listening to {}:{}.",
            _acceptor.local_endpoint().address().to_string(),
            _acceptor.local_endpoint().port());
        _accept();
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
