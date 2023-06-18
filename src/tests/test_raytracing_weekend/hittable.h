#pragma once

#include "ray.h"

class material;

class hit_record {
public:
    Float3 p;
    Float3 normal;
    // shared_ptr<material> mat_ptr;
    UInt mat_index;
    Float t;
    Bool front_face;

    inline void set_face_normal(const ray &r, const Float3 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = select(-outward_normal, outward_normal, front_face);
    }
};

class hittable {
public:
    virtual Bool hit(const ray &r, Float t_min, Float t_max, hit_record &rec) const = 0;
};

