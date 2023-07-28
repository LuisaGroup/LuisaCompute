#pragma once

#include <cstdint>

namespace luisa::compute {

/** @file */
/// Flags of usage
enum struct Usage : uint32_t {
    NONE = 0u,
    READ = 0x01u,
    WRITE = 0x02u,
    READ_WRITE = READ | WRITE
};

}

