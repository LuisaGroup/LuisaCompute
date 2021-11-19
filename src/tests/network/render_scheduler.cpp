//
// Created by Mike Smith on 2021/9/24.
//

#include <random>

#include <core/mathematics.h>
#include <network/render_buffer.h>
#include <network/render_config.h>
#include <network/render_server.h>
#include <network/render_worker_session.h>
#include <network/render_scheduler.h>

namespace luisa::compute {

RenderScheduler::RenderScheduler(
    RenderServer *server,
    RenderScheduler::interval_type dispatch_interval) noexcept
    : _server{server},
      _timer{server->context()},
      _interval{dispatch_interval},
      _render_id{invalid_render_id},
      _frame_id{} {}

void RenderScheduler::_dispatch(std::shared_ptr<RenderScheduler> self) noexcept {
    auto &&s = *self;
    s._timer.expires_after(s._interval);
    s._timer.async_wait([self = std::move(self)](asio::error_code error) mutable noexcept {
        if (error) {
            LUISA_WARNING_WITH_LOCATION(
                "Error when executing dispatch timer in RenderScheduler: {}.",
                error.message());
        } else {
            self->_purge();// remove dead workers
            if (auto config = self->_server->config();
                config == nullptr) {// no valid config, not rendering
                self->_reset(invalid_render_id);
            } else {
                if (auto render_id = config->render_id();
                    render_id != self->_render_id) {// config has changed, clean up
                    self->_reset(render_id);
                    LUISA_INFO(
                        "RenderConfig: scene = {}, render_id = {}, resolution = {}x{}, "
                        "spp = {}, tile_size = {}x{}, tile_spp = {}, max_tiles_in_flight = {}.",
                        config->scene(), render_id, config->resolution().x, config->resolution().y,
                        config->spp(), config->tile_size().x, config->tile_size().y, config->tile_spp(), config->tiles_in_flight());
                }
                // dispatch work
                static thread_local std::default_random_engine random{std::random_device{}()};
                auto worker_count = self->_workers.size();
                if (worker_count != 0u) {
                    auto max_tile_count = config->tiles_in_flight();
                    auto start = std::uniform_int_distribution<size_t>{0u, worker_count - 1u}(random);
                    auto any_available = true;
                    while (any_available) {
                        any_available = false;
                        for (auto i = 0u; i < worker_count; i++) {
                            auto worker_id = (start + i) % worker_count;
                            if (auto &&w = *self->_workers[worker_id];
                                w.working_item_count() < max_tile_count) {
                                if (auto tile = self->_next_tile(config)) {
                                    any_available = true;
                                    w.dispatch(*config, *tile);
                                }
                            }
                        }
                    }
                }
            }
        }
        _dispatch(std::move(self));
    });
}

std::optional<RenderTile> RenderScheduler::_next_tile(const RenderConfig *config) noexcept {
    // recycled tiles are dispatched first to minimize the latency
    if (!_recycled_tiles.empty()) {
        auto tile = _recycled_tiles.front();
        _recycled_tiles.pop();
        return tile;
    }
    // check pending tiles...
    if (_tiles.empty()) {
        // no more tiles to render
        if (auto spp = config->spp(); spp != 0u && _frame_id >= spp) {
            return std::nullopt;
        }
        // start next frame
        _frames.emplace(_frame_id, RenderBuffer{config->resolution(), config->tile_size()});
        for (auto y = 0u; y < config->resolution().y; y += config->tile_size().y) {
            for (auto x = 0u; x < config->resolution().x; x += config->tile_size().x) {
                _tiles.emplace(config->render_id(), _frame_id, make_uint2(x, y));
            }
        }
        // advance frame id
        _frame_id += config->tile_spp();
    }
    // return tile
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
        if (auto &&f = iter->second; f.accumulate(tile, data)) {
            if (f.done()) {
                auto frame = std::move(f);
                _frames.erase(iter);
                _server->accumulate(std::move(frame));
            }
        } else {
            recycle(tile);
        }
    }
}

asio::io_context &RenderScheduler::context() const noexcept {
    return _server->context();
}

void RenderScheduler::add(std::shared_ptr<RenderWorkerSession> worker) noexcept {
    if (*worker) { _workers.emplace_back(std::move(worker))->run(); }
}

void RenderScheduler::_purge() noexcept {
    _workers.erase(
        std::remove_if(_workers.begin(), _workers.end(), [](auto &&p) noexcept {
            return !(*p);
        }),
        _workers.end());
}

void RenderScheduler::close() noexcept {
    std::error_code error;
    _timer.cancel(error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Error when stopping dispatch timer in RenderScheduler: {}.",
            error.message());
    }
    for (auto &&w : _workers) {
        w->close();
    }
}

void RenderScheduler::_reset(uint32_t render_id) noexcept {
    _render_id = render_id;
    _frame_id = 0u;
    _tiles = {};
    _recycled_tiles = {};
    _frames.clear();
}

void RenderScheduler::run() noexcept {
    _dispatch(shared_from_this());
    LUISA_INFO("RenderScheduler started.");
}

}// namespace luisa::compute
