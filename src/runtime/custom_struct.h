#pragma once

#include <core/stl/string.h>

namespace luisa::compute {

struct CustomStructBase {
    int32_t _placeholder;
    using is_custom_struct = void;
};

#define LUISA_DECL_CUSTOM_STRUCT(name) \
    struct name : public CustomStructBase { static constexpr luisa::string_view type_name = #name;}
LUISA_DECL_CUSTOM_STRUCT(DispatchArgs);
LUISA_DECL_CUSTOM_STRUCT(AABB);
#undef LUISA_DECL_CUSTOM_STRUCT

}// namespace luisa::compute
