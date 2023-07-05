#pragma once

#include <cstdint>

namespace luisa::compute {

// X11 defines None as 0u which will conflict with our None enum value.
#if defined(None) && (None == 0u)
#define LUISA_NONE_DEFINED
#undef None
#endif

enum class DepthFormat : uint32_t {
    None,
    D16,
    D24S8,
    D32,
    D32S8A24
};

#ifdef LUISA_NONE_DEFINED
#define None 0u
#undef LUISA_NONE_DEFINED
#endif

}// namespace luisa::compute
