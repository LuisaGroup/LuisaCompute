#pragma once

#include "rtweekend.h"

class ray {
public:
    ray() {}
    ray(const Float3 &origin, const Float3 &direction, Float time = 0.0f)
        : orig(origin), dir(direction), tm(time) {}

    Float3 origin() const { return orig; }
    Float3 direction() const { return dir; }
    Float time() const { return tm; }

    Float3 at(Float t) const {
        return orig + t * dir;
    }

public:
    Float3 orig;
    Float3 dir;
    Float tm;
};
