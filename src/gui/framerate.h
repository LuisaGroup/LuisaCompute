//
// Created by Mike on 2021/11/30.
//

#pragma once

#include <chrono>

namespace luisa::compute {

class Framerate {

public:
    using Clock = std::chrono::steady_clock;
    using Timepoint = Clock::time_point;

private:
    std::vector<double> _durations;
    std::vector<size_t> _frames;
    Timepoint _last;
    size_t _history_size;

public:
    explicit Framerate(size_t n = 5) noexcept;
    void clear() noexcept;
    void record(size_t frame_count = 1u) noexcept;
    [[nodiscard]] double duration() const noexcept;
    [[nodiscard]] double report() const noexcept;
};

}
