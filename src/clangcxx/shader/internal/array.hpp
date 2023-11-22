#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {

template<typename Type, uint64 size, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("array")]] Array {
    [[builtin("BUFFER_READ")]] Type load(uint3 loc);
    [[ignore]] Type operator[](uint2 loc) const { return load(uint3(loc, 0)); };

    [[builtin("BUFFER_WRITE")]] void store(uint32 loc, Type value);
};

}// namespace luisa::shader