#pragma once

#include <ast/type_registry.h>

namespace luisa::compute {

struct CustomStructBase {
    int32_t _placeholder;
    using is_custom_struct = void;
};

#define LUISA_DECL_CUSTOM_STRUCT(name)                               \
    struct name : public CustomStructBase {};                        \
    namespace detail {                                               \
    template<>                                                       \
    struct TypeDesc<name> {                                          \
        static constexpr luisa::string_view description() noexcept { \
            using namespace std::string_view_literals;               \
            return #name##sv;                                        \
        }                                                            \
    };                                                               \
    }// namespace detail
LUISA_DECL_CUSTOM_STRUCT(DispatchArgs);
LUISA_DECL_CUSTOM_STRUCT(AABB);
#undef LUISA_DECL_CUSTOM_STRUCT

}// namespace luisa::compute
