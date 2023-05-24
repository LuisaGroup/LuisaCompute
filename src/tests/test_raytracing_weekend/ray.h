#ifndef RAY_H
#define RAY_H

#include "rtweekend.h"

class ray {
    public:
        ray() {}
        ray(const Float3& origin, const Float3& direction)
            : orig(origin), dir(direction)
        {}

        Float3 origin() const  { return orig; }
        Float3 direction() const { return dir; }

        Float3 at(Float t) const {
            return orig + t*dir;
        }

    public:
        Float3 orig;
        Float3 dir;
};

#endif