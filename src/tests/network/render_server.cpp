//
// Created by Mike Smith on 2021/9/24.
//

#include <stb/stb_image_write.h>

#include <network/binary_buffer.h>
#include <network/render_buffer.h>
#include <network/render_worker_session.h>
#include <network/render_scheduler.h>
#include <network/render_server.h>

namespace luisa::compute {

using namespace std::chrono_literals;

inline RenderServer::RenderServer(uint16_t worker_port, uint16_t client_port) noexcept
    : _context{1u},
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

void RenderServer::_send_to_clients(std::shared_ptr<BinaryBuffer> buffer) noexcept {
}

std::shared_ptr<RenderServer> RenderServer::create(uint16_t worker_port, uint16_t client_port) noexcept {
    return std::make_shared<RenderServer>(worker_port, client_port);
}

void RenderServer::run() noexcept {
    static constexpr auto width = 1280u;
    static constexpr auto height = 720u;
    _config = std::make_unique<RenderConfig>(
        1u, "test",
        make_uint2(width, height),
        0u, make_uint2(256u, 256u),
        1u, 16u);
    _accum_buffer.resize(width * height);
    _scheduler = std::make_shared<RenderScheduler>(this, 1ms);
    _scheduler->run();
    _accept_workers(shared_from_this());
    _accept_clients(shared_from_this());
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
    _config = std::make_unique<RenderConfig>(config);
    _frame_count = 0u;
    _sending_buffer = nullptr;
    _sending_frame_count = 0u;
}

std::shared_ptr<BinaryBuffer> RenderServer::sending_buffer() noexcept {
    if (_config == nullptr || _frame_count == 0u) { return nullptr; }
    if (_sending_frame_count < _frame_count) {// update sending buffer
        _sending_frame_count = _frame_count;
        _sending_buffer = std::make_shared<BinaryBuffer>();
        _sending_buffer->write(*_config)
            .write(_sending_frame_count);
//        std::transform(
//            _accum_buffer.cbegin(), _accum_buffer.cend(),
//            reinterpret_cast<std::array<uint8_t, 4u> *>(image.data),
//            [](float4 p) noexcept {
//                auto cvt = [](float x) noexcept {
//                    return static_cast<uint8_t>(clamp(x * 255.0f, 0.0f, 255.0f));
//                };
//                return std::array{cvt(p.z), cvt(p.y), cvt(p.x), static_cast<uint8_t>(255u)};
//            });
    }
    return _sending_buffer;
}

RenderServer::~RenderServer() noexcept = default;

}// namespace luisa::compute
