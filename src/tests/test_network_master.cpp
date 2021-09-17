//
// Created by Mike Smith on 2021/9/17.
//

#include <ctime>
#include <memory>
#include <iostream>

#include <asio.hpp>

#include <core/allocator.h>
#include <core/logging.h>

class DisplayServer;

void create_session(DisplayServer &server, std::shared_ptr<asio::ip::tcp::socket> socket) noexcept;

class DisplayServer {

private:
    asio::io_context &_context;
    asio::ip::tcp::acceptor _acceptor;
    std::vector<std::future<void>> _workers;

private:
    void _purge() noexcept {
        _workers.erase(
            std::remove_if(_workers.begin(), _workers.end(), [](auto &&w) noexcept {
                using namespace std::chrono_literals;
                return w.wait_for(0ns) == std::future_status::ready;
            }),
            _workers.end());
    }

    void _accept() noexcept {
        auto socket = std::make_shared<asio::ip::tcp::socket>(_context);
        auto p_socket = socket.get();
        _acceptor.async_accept(*p_socket, [this, socket = std::move(socket)](asio::error_code error) mutable noexcept {
            if (error) {
                LUISA_INFO("Error: {}.", error.message());
            } else {
                LUISA_INFO(
                    "Connection from {}:{}.",
                    socket->remote_endpoint().address().to_string(),
                    socket->remote_endpoint().port());
                create_session(*this, std::move(socket));
            }
            _purge();
            _accept();
        });
    }

public:
    explicit DisplayServer(asio::io_context &io_context)
        : _context{io_context},
          _acceptor{io_context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 13)} {
        LUISA_INFO(
            "Listening to {}:{}.",
            _acceptor.local_endpoint().address().to_string(),
            _acceptor.local_endpoint().port());
        _accept();
    }
    [[nodiscard]] auto &context() noexcept { return _context; }

    template<typename F>
    void create_thread(F &&f) noexcept {
        _workers.emplace_back(std::async(std::launch::async, std::forward<F>(f)));
    }
};

void create_session(DisplayServer &server, std::shared_ptr<asio::ip::tcp::socket> socket) noexcept {
    server.create_thread([socket = std::move(socket)] {
        asio::error_code error;
        auto remote = socket->remote_endpoint();
        while (!error) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(100ms);
            auto now = time(nullptr);
            auto message = fmt::format("hello, now is {}", ctime(&now));
            message.pop_back();
            message.push_back('.');
            asio::write(*socket, asio::buffer(message), error);
        }
        LUISA_INFO(
            "Disconnected from {}:{} (error: {}).",
            remote.address().to_string(),
            remote.port(),
            error.message());
    });
}

int main() {
    asio::io_context io_context{1};
    DisplayServer server{io_context};
    asio::error_code error;
    io_context.run(error);
    LUISA_INFO("Error: {}.", error.message());
}
