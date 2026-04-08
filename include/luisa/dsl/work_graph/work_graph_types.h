#pragma once
#include <cstdint>

namespace luisa::compute {

enum class WorkGraphLaunchType : uint8_t {
    BROADCASTING,
    THREAD,
    // add later, defining what "input" to COALESCING launch node should be
    // is much trickier than the other two types
    /* COALESCING */
};

// Note: you must put `size` member first in LUISA_STRUCT macro, otherwise the codegen
// to annotate it with `SV_DispatchGrid` will not work correctly
struct DispatchGridRecord { uint3 size; };

} // namespace luisa::compute