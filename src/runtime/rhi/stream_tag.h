#pragma once

#include <cstdint>
#include <type_traits>

namespace luisa::compute {

enum class StreamTag : uint32_t {
    GRAPHICS,// capable of graphics, compute, and copy commands
    COMPUTE, // capable of compute and copy commands
    COPY,    // only copy commands,
    CUSTOM   // custom stream
};
template<typename T>
struct StreamEvent : std::false_type {};
}// namespace luisa::compute
