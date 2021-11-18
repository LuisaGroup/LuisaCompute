//
// Created by Mike on 2021/11/19.
//

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <network/binary_buffer.h>
#include <network/render_tile.h>
#include <network/render_config.h>
#include <network/render_worker.h>

namespace luisa::compute {

inline RenderWorker::RenderWorker(const std::string &server_ip, uint16_t server_port) noexcept
    : _context{1u},
      _socket{_context},
      _timer{_context},
      _server{asio::ip::address_v4::from_string(server_ip), server_port} {}

std::shared_ptr<RenderWorker> RenderWorker::create(const std::string &server_ip, uint16_t server_port) noexcept {
    return std::make_shared<RenderWorker>(server_ip, server_port);
}

RenderWorker &RenderWorker::set_config_handler(RenderWorker::ConfigHandler handler) noexcept {
    _config_handler = std::move(handler);
    return *this;
}

RenderWorker &RenderWorker::set_render_handler(RenderWorker::RenderHandler handler) noexcept {
    _render_handler = std::move(handler);
    return *this;
}

void RenderWorker::_receive_config_command(std::shared_ptr<RenderWorker> self, BinaryBuffer buffer) noexcept {
    auto &&socket = self->_socket;
    auto asio_buffer = buffer.asio_buffer_tail();
    asio::async_read(
        socket, asio_buffer,
        [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
            if (error) {
                LUISA_ERROR_WITH_LOCATION(
                    "Error occurred while receiving config command: {}",
                    error.message());
            }
            RenderConfig config;
            buffer.read(config);
            self->_config_handler(config);
            _receive(std::move(self));
        });
}

void RenderWorker::_receive_render_command(std::shared_ptr<RenderWorker> self, BinaryBuffer buffer) noexcept {
    auto &&socket = self->_socket;
    auto asio_buffer = buffer.asio_buffer_tail();
    asio::async_read(
        socket, asio_buffer,
        [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
            if (error) {
                LUISA_ERROR_WITH_LOCATION(
                    "Error occurred while receiving render command: {}",
                    error.message());
            }
            RenderTile tile;
            buffer.read(tile);
            if (auto [result, size] = self->_render_handler(tile); !result.empty()) {
                static thread_local std::vector<std::byte> rgbe;
                rgbe.clear();
                rgbe.reserve(result.size() * 4u);
                stbi_write_hdr_to_func(
                    [](void *context, void *data, int n) noexcept {
                        auto &&rgbe = *static_cast<std::vector<std::byte> *>(context);
                        std::copy_n(static_cast<const std::byte *>(data), n, std::back_inserter(rgbe));
                    },
                    &rgbe, static_cast<int>(size.x), static_cast<int>(size.y), 4, &result.front().x);
                buffer.clear();
                buffer.write(tile).write(rgbe.data(), rgbe.size()).write_size();
                self->_sending_queue.emplace(std::move(buffer));
            }
            _receive(std::move(self));
        });
}

void RenderWorker::_receive(std::shared_ptr<RenderWorker> self) noexcept {
    BinaryBuffer buffer;
    buffer.write_skip(6);// command
    auto &&socket = self->_socket;
    auto asio_buffer = buffer.asio_buffer();
    asio::async_read(
        socket, asio_buffer,
        [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
            if (error) {
                LUISA_ERROR_WITH_LOCATION(
                    "Error occurred while receiving command: {}",
                    error.message());
            }
            auto size = buffer.read_size();
            auto command = [&buffer] {
                static thread_local std::array<char, 6> command{};
                buffer.read(command);
                return std::string_view{
                    command.data(),
                    command.size()};
            }();
            buffer.write_skip(size - 6u);// sizeof(command) == 6u
            if (command == "CONFIG") {
                _receive_config_command(std::move(self), std::move(buffer));
            } else if (command == "RENDER") {
                _receive_render_command(std::move(self), std::move(buffer));
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid command: {}.",
                    command);
            }
        });
}

void RenderWorker::run() noexcept {
    _socket.connect(_server);
    LUISA_INFO("Connected to server.");
    _receive(shared_from_this());
    _send(shared_from_this());
    _context.run();
}

void RenderWorker::_send(std::shared_ptr<RenderWorker> self) noexcept {
    if (self->_sending_queue.empty()) {
        using namespace std::chrono_literals;
        auto &&timer = self->_timer;
        timer.expires_after(1ms);
        timer.async_wait([self = std::move(self)](asio::error_code error) mutable noexcept {
            if (error) {
                LUISA_ERROR_WITH_LOCATION(
                    "Error occurred while waiting for timer: {}.",
                    error.message());
            }
            _send(std::move(self));
        });
    } else {
        auto buffer = std::move(self->_sending_queue.front());
        self->_sending_queue.pop();
        auto &&socket = self->_socket;
        auto asio_buffer = buffer.asio_buffer();
        asio::async_write(
            socket, asio_buffer,
            [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
                if (error) {
                    LUISA_ERROR_WITH_LOCATION(
                        "Error occurred while sending tile: {}.",
                        error.message());
                }
                _send(std::move(self));
            });
    }
}

RenderWorker::~RenderWorker() noexcept = default;

}// namespace luisa::compute
