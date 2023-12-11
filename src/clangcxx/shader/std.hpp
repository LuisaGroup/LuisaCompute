#pragma once
#include "internal/attributes.hpp"
#include "internal/type_traits.hpp"

#include "internal/functions.hpp"
#include "internal/array.hpp"
#include "internal/buffer.hpp"
#include "internal/texture.hpp"

#include "internal/accel.hpp"
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