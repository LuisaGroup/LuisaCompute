#pragma once
#include <stdint.h>
namespace luisa::compute {
enum class StreamTag : uint8_t {
    GRAPHICS,
    COMPUTE,
    COPY
};
};