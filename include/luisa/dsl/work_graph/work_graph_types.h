#pragma once
#include <cstdint>
#include <luisa/core/basic_types.h>

namespace luisa::compute {

enum class WorkGraphLaunchType : uint8_t {
    BROADCASTING,
    THREAD,
    // add later, defining what "input" to COALESCING launch node should be
    // is much trickier than the other two types
    /* COALESCING */
};

template<typename T>
struct DispatchGrid { T value; };

struct WorkGraphEdgeConfig {
    uint max_records{1};
};

} // namespace luisa::compute