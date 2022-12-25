#pragma once

#include <cstdint>

namespace luisa::compute {

struct CustomStructBase {
    int32_t _placeholder;
};

#define LUISA_DECL_CUSTOM_STRUCT(name) \
    struct name : public CustomStructBase {}
LUISA_DECL_CUSTOM_STRUCT(DispatchArgs1D);
LUISA_DECL_CUSTOM_STRUCT(DispatchArgs2D);
LUISA_DECL_CUSTOM_STRUCT(DispatchArgs3D);
LUISA_DECL_CUSTOM_STRUCT(AABB);
#undef LUISA_DECL_CUSTOM_STRUCT

}// namespace luisa::compute
