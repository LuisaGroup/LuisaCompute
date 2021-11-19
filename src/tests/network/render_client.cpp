//
// Created by Mike Smith on 2021/11/19.
//

#include <stb/stb_image.h>

#include <network/binary_buffer.h>
#include <network/render_config.h>
#include <network/render_client.h>

namespace luisa::compute {

void RenderClient::_receive_frame(std::shared_ptr<RenderClient> self, BinaryBuffer buffer, const RenderConfig &config, size_t frame_count) noexcept {
    auto &&socket = self->_socket;
    auto asio_buffer = buffer.asio_buffer_tail();
    asio::async_read(
        socket, asio_buffer,
        [self = std::move(self), buffer = std::move(buffer), config, frame_count](asio::error_code error, size_t) mutable noexcept {
            if (error) {
                LUISA_ERROR_WITH_LOCATION(
                    "Error when receiving frame: {}.",
                    error.message());
            }
            auto width = 0;
            auto height = 0;
            auto channels = 0;
            std::unique_ptr<float4, void (*)(void *)> pixels{
                reinterpret_cast<float4 *>(
                    stbi_loadf_from_memory(
                        reinterpret_cast<const uint8_t *>(buffer.tail().data()),
                        static_cast<int>(buffer.tail().size_bytes()),
                        &width, &height, &channels, 4)),
                stbi_image_free};
            if (width != config.resolution().x || height != config.resolution().y) {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid tile: width = {}, height = {}, channels = {}.",
                    width, height, channels);
            }
            self->_display(
                config, frame_count,
                std::span{pixels.get(), config.resolution().x * config.resolution().y});
            _receive(std::move(self));
        });
}

void RenderClient::_receive(std::shared_ptr<RenderClient> self) noexcept {
    auto &&socket = self->_socket;
    BinaryBuffer buffer;
    buffer.write_skip(sizeof(RenderConfig)).write_skip(sizeof(size_t));
    auto asio_buffer = buffer.asio_buffer();
    asio::async_read(
        socket, asio_buffer,
        [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
            if (error) {
                LUISA_ERROR_WITH_LOCATION(
                    "Error when receiving frame meta-data: {}.",
                    error.message());
            }
            auto size = buffer.read_size();
            RenderConfig config;
            size_t frame_count;
            buffer.read(config).read(frame_count);
            buffer.write_skip(size - sizeof(RenderConfig) - sizeof(size_t));
            _receive_frame(std::move(self), std::move(buffer), config, frame_count);
        });
}

void RenderClient::_send(std::shared_ptr<RenderClient> self) noexcept {
    if (auto config = [&s = *self] {
            std::scoped_lock lock{s._mutex};
            std::unique_ptr<RenderConfig> c;
            c.swap(s._sending_config);
            return c;
        }()) {
        BinaryBuffer buffer;
        buffer.write(*config).write_size();
        auto asio_buffer = buffer.asio_buffer();
        auto &&socket = self->_socket;
        asio::async_write(
            socket, asio_buffer,
            [self = std::move(self), buffer = std::move(buffer)](asio::error_code error, size_t) mutable noexcept {
                if (error) {
                    LUISA_ERROR_WITH_LOCATION(
                        "Error when sending config: {}.",
                        error.message());
                }
                _send(std::move(self));
            });
    } else {
        using namespace std::chrono_literals;
        auto &&timer = self->_timer;
        timer.expires_after(5ms);
        timer.async_wait([self = std::move(self)](asio::error_code error) mutable noexcept {
            if (error) {
                LUISA_ERROR_WITH_LOCATION(
                    "Error when waiting for timer: {}.",
                    error.message());
            }
            _send(std::move(self));
        });
    }
}

inline RenderClient::RenderClient(const char *server_ip, uint16_t port) noexcept
    : _context{1u},
      _socket{_context},
      _timer{_context},
      _server{asio::ip::address_v4::from_string(server_ip), port} {}

RenderClient::~RenderClient() noexcept = default;

std::shared_ptr<RenderClient> RenderClient::create(const char *server_ip, uint16_t port) noexcept {
    return std::make_shared<RenderClient>(server_ip, port);
}

RenderClient &RenderClient::set_display_handler(RenderClient::DisplayHandler handler) noexcept {
    _display = std::move(handler);
    return *this;
}

void RenderClient::run() noexcept {
    _socket.connect(_server);
    LUISA_INFO("Connected to server.");
    _send(shared_from_this());
    _receive(shared_from_this());
    _context.run();
}

RenderClient &RenderClient::set_config(const RenderConfig &config) noexcept {
    std::scoped_lock lock{_mutex};
    _sending_config = std::make_unique<RenderConfig>(config);
    return *this;
}

}
