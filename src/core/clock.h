//
// Created by Mike Smith on 2021/4/5.
//

#pragma once

#include <chrono>

namespace luisa {

class Clock {

    using SystemClock = std::chrono::high_resolution_clock;
    using Tick = std::chrono::high_resolution_clock::time_point;

private:
    Tick _last;

public:
    Clock() noexcept : _last{SystemClock::now()} {}
    void tic() noexcept { _last = SystemClock::now(); }
    [[nodiscard]] auto toc() const noexcept {
        auto curr = SystemClock::now();
        using namespace std::chrono_literals;
        return (curr - _last) / 1ns * 1e-6;
    }
};

}// namespace luisa
