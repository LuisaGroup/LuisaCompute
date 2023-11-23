#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {

template<typename Type, uint64 size, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("array")]] Array {
    [[expr("ACCESS")]] Type load(uint32 loc);
    [[ignore]] Type operator[](uint32 loc) const { return load(loc); };

    [[builtin("BUFFER_WRITE")]] void store(uint32 loc, Type value);

    Type v[size];
};

}// namespace luisa::shader