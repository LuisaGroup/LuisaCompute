#pragma once

#include <cstdint>

namespace luisa::compute {

enum class StreamTag : uint8_t {
    GRAPHICS,// capable of graphics, compute, and copy commands
    COMPUTE, // capable of compute and copy commands
    COPY     // only copy commands
};

}
