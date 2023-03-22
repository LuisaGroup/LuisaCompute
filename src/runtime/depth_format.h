#pragma once

#include <cstdint>

namespace luisa::compute {

enum class DepthFormat : uint32_t {
    None,
    D16,
    D24S8,
    D32,
    D32S8A24
};

}
