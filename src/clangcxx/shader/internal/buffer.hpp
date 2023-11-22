#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"


namespace luisa::shader {

template<typename Type, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("buffer")]] Buffer {
    [[callop("BUFFER_READ")]] Type load(uint3 loc);
    [[ignore]] Type operator[](uint2 loc) const { return load(uint3(loc, 0)); };

    [[callop("BUFFER_WRITE")]] void store(uint32 loc, Type value);
};

}// namespace luisa::shader