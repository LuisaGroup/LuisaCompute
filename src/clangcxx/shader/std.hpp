#pragma once
#include "internal/attributes.hpp"
#include "internal/type_traits.hpp"

namespace luisa::shader {

[[builtin("dispatch_id")]] extern uint3 dispatch_id();
[[builtin("sin")]] extern float sin(float rad);
[[builtin("cos")]] extern float cos(float rad);

template<typename Type = void, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("buffer")]] Buffer {
    [[builtin("BUFFER_READ")]] Type load(uint3 loc);
    [[ignore]] Type operator[](uint2 loc) const { return load(uint3(loc, 0)); };

    [[builtin("BUFFER_WRITE")]] void store(uint32 loc, Type value);
};

}// namespace luisa::shader