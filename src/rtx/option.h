//
// Created by Mike Smith on 2023/2/8.
//

#pragma once

#include <cstdint>

namespace luisa::compute {

struct AccelCreateOption {

    enum struct UsageHint : uint8_t {
        FAST_TRACE,// build with best quality
        FAST_BUILD // optimize for frequent rebuild, maybe without compaction
    };

    UsageHint hint{UsageHint::FAST_BUILD};
    bool allow_compaction{false};
    bool allow_update{false};
};

}// namespace luisa::compute
