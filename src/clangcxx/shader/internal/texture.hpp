#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {
template<arithmetic_scalar T>
struct [[builtin("image")]] Image {
    [[callop("TEXTURE_READ")]] vec<T, 4> load(uint2 coord);
    [[callop("TEXTURE_WRITE")]] void store(uint2 coord, vec<T, 4> val);
    [[callop("TEXTURE_SIZE")]] uint2 size();
};
template<arithmetic_scalar T>
struct [[builtin("volume")]] Volume {
    [[callop("TEXTURE_READ")]] vec<T, 4> load(uint3 coord);
    [[callop("TEXTURE_WRITE")]] void store(uint3 coord, vec<T, 4> val);
    [[callop("TEXTURE_SIZE")]] uint3 size();
};
}// namespace luisa::shader