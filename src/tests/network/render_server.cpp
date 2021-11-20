//
// Created by Mike Smith on 2021/9/24.
//

#include <stb/stb_image_write.h>

#include <core/clock.h>
#include <network/binary_buffer.h>
#include <network/render_buffer.h>
#include <network/render_client_session.h>
#include <network/render_worker_session.h>
#include <network/render_scheduler.h>
#include <network/render_server.h>

namespace luisa::compute {

using namespace std::chrono_literals;

inline RenderServer::RenderServer(uint16_t worker_port, uint16_t client_port) noexcept
    : _context{1u},
      _purge_timer{_context},
      _worker_acceptor{_context, asio::ip::tcp::endpoint{asio::ip::tcp::v4(), worker_port}},
      _client_acceptor{_context, asio::ip::tcp::endpoint{asio::ip::tcp::v4(), client_port}} {}

void RenderServer::accumulate(RenderBuffer buffer) noexcept {
    auto last_frame_count = _frame_count;
    if (_config->spp() == 0u) {
        _frame_count += _config->tile_spp();
        LUISA_INFO("Accumulating {}.", _frame_count);
    } else {
        _frame_count = std::min(
            _frame_count + _config->tile_spp(),
            static_cast<size_t>(_config->spp()));
        LUISA_INFO("Accumulating {}/{}.", _frame_count, _config->spp());
    }
    auto pixels = buffer.framebuffer().data();
    auto spp = _frame_count - last_frame_count;
    std::transform(
        _accum_buffer.cbegin(), _accum_buffer.cend(), pixels,
        _accum_buffer.begin(),
        [t = static_cast<float>(spp) / static_cast<float>(_frame_count)](auto lhs, auto rhs) noexcept {
            return make_float4(lerp(lhs, rhs, t).xyz(), 1.0f);
        });
}

void RenderServer::_accept_workers(std::shared_ptr<RenderServer> self) noexcept {
    if (auto &&s = *self) {
        auto worker = std::make_shared<RenderWorkerSession>(s._scheduler.get());
        auto &&socket = worker->socket();
        LUISA_INFO("Waiting for workers...");
        s._worker_acceptor.async_accept(socket, [self = std::move(self), worker = std::move(worker)](asio::error_code error) mutable noexcept {
            if (error) {
                LUISA_WARNING_WITH_LOCATION(
                    "Error when connecting to worker in RenderServer: {}.",
                    error.message());
            } else if (auto remote = worker->socket().remote_endpoint(error); !error) {
                LUISA_INFO(
                    "Connected to worker {}:{}.",
                    remote.address().to_string(), remote.port());
                self->_scheduler->add(std::move(worker));
            }
            _accept_workers(std::move(self));
        });
    }
}

void RenderServer::_accept_clients(std::shared_ptr<RenderServer> self) noexcept {
    if (auto &&s = *self) {
        auto client = std::make_shared<RenderClientSession>(&s);
        auto &&socket = client->socket();
        LUISA_INFO("Waiting for clients...");
        s._client_acceptor.async_accept(
            socket,
            [self = std::move(self), client = std::move(client)](asio::error_code error) mutable noexcept {
                if (error) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Error when connecting to client in RenderServer: {}.",
                        error.message());
                } else if (auto remote = client->socket().remote_endpoint(error); !error) {
                    LUISA_INFO(
                        "Connected to client {}:{}",
                        remote.address().to_string(), remote.port());
                    self->_purge();
                    self->_clients.emplace_back(std::move(client))->run();
                }
                _accept_clients(std::move(self));
            });
    }
}

void RenderServer::_close() noexcept {
    asio::error_code error;
    _worker_acceptor.close(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when closing worker acceptor in RenderServer: {}.",
            error.message());
    }
    _client_acceptor.close(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when closing client acceptor in RenderServer: {}.",
            error.message());
    }
    _scheduler->close();
}

std::shared_ptr<RenderServer> RenderServer::create(uint16_t worker_port, uint16_t client_port) noexcept {
    return std::make_shared<RenderServer>(worker_port, client_port);
}

void RenderServer::run() noexcept {
    _scheduler = std::make_shared<RenderScheduler>(this, 1ms);
    _scheduler->run();
    _accept_workers(shared_from_this());
    _accept_clients(shared_from_this());
    _purge_clients(shared_from_this());
    LUISA_INFO("RenderServer started.");
    asio::error_code error;
    _context.run(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error encountered in RenderServer: {}",
            error.message());
    }
}

void RenderServer::set_config(const RenderConfig &config) noexcept {
    _config = std::make_unique<RenderConfig>(
        ++_render_id, config.scene(), config.resolution(), config.spp(),
        config.tile_size(), config.tile_spp(), config.tiles_in_flight());
    _frame_count = 0u;
    _sending_buffer = nullptr;
    _sending_frame_count = 0u;
    _accum_buffer.resize(_config->resolution().x * _config->resolution().y);
}

std::shared_ptr<BinaryBuffer> RenderServer::sending_buffer() noexcept {
    if (_config == nullptr || _frame_count == 0u) { return nullptr; }
    if (_sending_frame_count < _frame_count) {// update sending buffer
        _sending_frame_count = _frame_count;
        _sending_buffer = std::make_shared<BinaryBuffer>();
        _sending_buffer->write(*_config).write(_sending_frame_count);
        static thread_local std::vector<std::array<uint8_t, 3u>> ldr;
        ldr.resize(_config->resolution().x * _config->resolution().y);
        static constexpr auto pow = [](float3 v, float p) noexcept {
            return make_float3(
                std::pow(v.x, p),
                std::pow(v.y, p),
                std::pow(v.z, p));
        };
        static constexpr auto aces = [](float3 x) noexcept {
            static constexpr auto a = 2.51f;
            static constexpr auto b = 0.03f;
            static constexpr auto c = 2.43f;
            static constexpr auto d = 0.59f;
            static constexpr auto e = 0.14f;
            return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
        };
        static constexpr auto linear_to_srgb = [](float4 v) noexcept {
            auto x = aces(make_float3(v));
            x = clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                             12.92f * x,
                             x <= 0.00031308f) *
                          255.0f,
                      0.0f, 255.0f);
            return std::array<uint8_t, 3u>{
                static_cast<uint8_t>(x.x),
                static_cast<uint8_t>(x.y),
                static_cast<uint8_t>(x.z)};
        };
        Clock clock;
        std::transform(_accum_buffer.cbegin(), _accum_buffer.cend(), ldr.begin(), linear_to_srgb);
        auto t1 = clock.toc();
        stbi_write_jpg_to_func(
            [](void *context, void *data, int n) noexcept {
                static_cast<BinaryBuffer *>(context)->write(data, n);
            },
            _sending_buffer.get(),
            static_cast<int>(_config->resolution().x),
            static_cast<int>(_config->resolution().y),
            3, ldr.data(), 75);
        auto t2 = clock.toc();
        LUISA_INFO("Updated sending buffer in {} ms (sRGB: {} ms, JPEG: {} ms).", t2, t1, t2 - t1);
        _sending_buffer->write_size();
    }
    return _sending_buffer;
}

void RenderServer::_purge_clients(std::shared_ptr<RenderServer> self) noexcept {
    if (auto &&s = *self) {
        using namespace std::chrono_literals;
        s._purge_timer.expires_after(1ms);
        s._purge_timer.async_wait([self = std::move(self)](asio::error_code error) mutable noexcept {
            if (error) {
                LUISA_WARNING_WITH_LOCATION(
                    "Error when performing periodic purging: {}.",
                    error.message());
            }
            self->_purge();
            _purge_clients(std::move(self));
        });
    }
}

void RenderServer::_purge() noexcept {
    for (auto &&c : _clients) {
        if (!*c) { c->close(); }
    }
    std::erase_if(_clients, [](auto &&c) noexcept { return !(*c); });
    if (_clients.empty()) {// no clients, stop rendering...
        _config = nullptr;
        _frame_count = 0u;
        _sending_buffer = nullptr;
        _sending_frame_count = 0u;
    }
}

RenderServer::~RenderServer() noexcept = default;

}// namespace luisa::compute
