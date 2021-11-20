//
// Created by Mike Smith on 2021/11/19.
//

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
            self->_display(config, frame_count, buffer.tail());
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
            {
                std::scoped_lock lock{self->_mutex};
                if (auto curr_config = self->_config.get();
                    curr_config == nullptr || curr_config->render_id() < config.render_id()) {
                    *curr_config = config;
                    self->_config_dirty = false;
                }
            }
            buffer.write_skip(size - sizeof(RenderConfig) - sizeof(size_t));
            _receive_frame(std::move(self), std::move(buffer), config, frame_count);
        });
}

void RenderClient::_send(std::shared_ptr<RenderClient> self) noexcept {
    if (auto config = [&s = *self] {
            std::scoped_lock lock{s._mutex};
            auto c = s._config_dirty ?
                       std::optional{*s._config} :
                       std::nullopt;
            s._config_dirty = false;
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
    auto render_id = _config == nullptr ? 0u : _config->render_id() + 1u;
    _config = std::make_unique<RenderConfig>(
        render_id, config.scene(), config.resolution(), config.spp(),
        config.tile_size(), config.tile_spp(), config.tiles_in_flight());
    _config_dirty = true;
    return *this;
}

}// namespace luisa::compute
