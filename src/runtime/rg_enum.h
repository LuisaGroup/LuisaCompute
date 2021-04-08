#pragma once

#include <cstdint>

namespace luisa::compute {

enum class RGNodeState : uint32_t {
    Preparing,
    InList,
    Executed
};

}// namespace luisa::compute
