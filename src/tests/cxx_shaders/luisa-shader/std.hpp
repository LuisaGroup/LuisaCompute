#pragma once
#include "internal/attributes.hpp"
#include "internal/type_traits.hpp"
#include "internal/numerics.hpp"

#include "internal/array.hpp"
#include "internal/vec.hpp"
#include "internal/matrix.hpp"
#include "internal/resource.hpp"

#include "internal/functions.hpp"
#include "internal/ray_query.hpp"

struct zzSHADER_PRIMITIVES
{
    int i;
    short s;
    long l;
    long long ll;
    
    unsigned int ui;
    unsigned short us;
    unsigned long ul;
    unsigned long long ull;
    
    float f;
    double d;
};

namespace luisa::shader::mandelbrot {

template <typename Resource, typename T>
static void store_2d(Resource& r, uint32 row_pitch, uint2 pos, T val)
{
    using ResourceType = remove_cvref_t<Resource>;
    if constexpr (is_same_v<ResourceType, Buffer<T>>)
        r.store(pos.x + pos.y * row_pitch, val);
    else if constexpr (is_same_v<ResourceType, Image<typename scalar_type<T>>>)
        r.store(pos, val);
}
}
