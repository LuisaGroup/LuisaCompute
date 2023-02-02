//
// Created by Mike on 2021/11/30.
//

#include <gui/framerate.h>

namespace luisa::compute {

Framerate::Framerate(size_t n) noexcept
    : _history_size{n} {
    _durations.reserve(n);
    _frames.reserve(n);
    _last = Clock::now();
}

void Framerate::clear() noexcept {
    _durations.clear();
    _frames.clear();
    _last = Clock::now();
}

double Framerate::duration() const noexcept {
    auto dt = Clock::now() - _last;
    using namespace std::chrono_literals;
    return static_cast<double>(dt / 1ns) * 1e-9;
}

void Framerate::record(size_t frame_count) noexcept {
    if (_durations.size() == _history_size) {
        _durations.erase(_durations.begin());
        _frames.erase(_frames.begin());
    }
    using namespace std::chrono_literals;
    _durations.emplace_back(duration());
    _frames.emplace_back(frame_count);
    _last = Clock::now();
}

double Framerate::report() const noexcept {
    if (_durations.empty()) { return 0.0; }
    auto total_duration = 0.0;
    auto total_frame_count = static_cast<size_t>(0u);
    for (auto i = 0u; i < _durations.size(); i++) {
        total_duration += _durations[i];
        total_frame_count += _frames[i];
    }
    return static_cast<double>(total_frame_count) / total_duration;
}

}
