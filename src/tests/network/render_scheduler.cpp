//
// Created by Mike Smith on 2021/9/24.
//

#include <random>

#include <core/mathematics.h>
#include <network/render_buffer.h>
#include <network/render_server.h>
#include <network/render_worker_session.h>
#include <network/render_scheduler.h>

namespace luisa::compute {

RenderScheduler::RenderScheduler(
    RenderServer *server,
    RenderScheduler::interval_type dispatch_interval) noexcept
    : _server{server}, _timer{server->context()},
      _interval{dispatch_interval},
      _render_id{},
      _frame_id{} {}

void RenderScheduler::start(std::shared_ptr<RenderConfig> config) noexcept {
    stop();
    _config = std::move(config);
    _render_id++;
    _frame_id = 0u;
    // TODO: set config...
    _dispatch(shared_from_this());
}

void RenderScheduler::stop() noexcept {
    std::error_code error;
    _timer.cancel(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when stopping dispatch timer in RenderScheduler: {}.",
            error.message());
    }
    for (auto &&w : _workers) { w->stop(); }
    _config.reset();
    _tiles = {};
    _recycled_tiles = {};
    _frames.clear();
    _frame_id = 0u;
}

void RenderScheduler::_dispatch(std::shared_ptr<RenderScheduler> self) noexcept {
    auto &&s = *self;
    s._timer.expires_after(s._interval);
    s._timer.async_wait([self = std::move(self)](asio::error_code error) mutable noexcept {
        if (error) {
            LUISA_WARNING_WITH_LOCATION(
                "Error when executing dispatch timer in RenderScheduler: {}.",
                error.message());
        } else {
            self->_purge();
            static thread_local std::default_random_engine random{std::random_device{}()};
            if (auto worker_count = self->_workers.size(); worker_count != 0u) {
                auto max_tile_count = self->_config->tiles_in_flight();
                auto start = std::uniform_int_distribution<size_t>{0u, self->_workers.size() - 1u}(random);
                for (auto i = 0u; i < worker_count; i++) {
                    if (auto &&w = *self->_workers[(start + i) % worker_count]; w.working_tile_count() != max_tile_count) {
                        if (auto tile = self->_next_tile()) { w.render(*tile); }
                    }
                }
            }
        }
        _dispatch(std::move(self));
    });
}

std::optional<RenderTile> RenderScheduler::_next_tile() noexcept {
    if (!_recycled_tiles.empty()) {
        auto tile = _recycled_tiles.front();
        _recycled_tiles.pop();
        return tile;
    }
    if (_tiles.empty()) {
        if (_config == nullptr || (_config->spp() != 0u && _frame_id == _config->spp())) { return std::nullopt; }
        _frames.emplace(_frame_id, RenderBuffer{_config->resolution(), _config->tile_size()});
        for (auto y = 0u; y < _config->resolution().y; y += _config->tile_size().y) {
            for (auto x = 0u; x < _config->resolution().x; x += _config->tile_size().x) {
                auto offset = make_uint2(x, y);
                _tiles.emplace(_render_id, _frame_id, offset);
            }
        }
        _frame_id += _config->tile_spp();
    }
    auto tile = _tiles.front();
    _tiles.pop();
    return tile;
}

void RenderScheduler::recycle(RenderTile tile) noexcept {
    if (tile.render_id() == _render_id) {
        _recycled_tiles.emplace(tile);
    }
}

void RenderScheduler::accumulate(RenderTile tile, std::span<const std::byte> data) noexcept {
    if (tile.render_id() != _render_id) { return; }
    if (auto iter = _frames.find(tile.frame_id()); iter != _frames.end()) {
        if (auto &&f = iter->second; f.accumulate(tile, data) && f.done()) {
            auto frame = std::move(f);
            _frames.erase(iter);
            _server->process(tile.frame_id(), std::move(frame));
        }
    }
}

asio::io_context &RenderScheduler::context() const noexcept {
    return _server->context();
}

void RenderScheduler::add(std::shared_ptr<RenderWorkerSession> worker) noexcept {
    if (*worker) { _workers.emplace_back(std::move(worker)); }
}

void RenderScheduler::_purge() noexcept {
    _workers.erase(std::remove_if(_workers.begin(), _workers.end(), [](auto &&p) noexcept {
                       return static_cast<bool>(*p);
                   }),
                   _workers.end());
}

void RenderScheduler::close() noexcept {
    stop();
    for (auto &&w : _workers) { w->close(); }
}

}// namespace luisa::compute
