#pragma once

#include "hittable.h"

class sphere : public hittable {
public:
    sphere() {}
    sphere(float3 cen, float r, shared_ptr<material> m)
        : center(cen), radius(r) {
        for (uint i = 0; i < materials.size(); i++) {
            if (m == materials[i]) {
                mat_index = i;
                break;
            }
        }
    };

    virtual Bool hit(
        const ray &r, Float t_min, Float t_max, hit_record &rec) const override;

public:
    float3 center;
    float radius;
    // shared_ptr<material> mat_ptr;
    uint mat_index;
};

Bool sphere::hit(const ray &r, Float t_min, Float t_max, hit_record &rec) const {
    Bool ret = true;

    Float3 oc = r.origin() - center;
    Float a = length_squared(r.direction());
    Float half_b = dot(oc, r.direction());
    Float c = length_squared(oc) - radius * radius;

    Float discriminant = half_b * half_b - a * c;
    $if(discriminant < 0) {
        ret = false;
    }
    $else {
        Float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        Float root = (-half_b - sqrtd) / a;
        $if((root < t_min) | (t_max < root)) {
            root = (-half_b + sqrtd) / a;
            $if((root < t_min) | (t_max < root)) {
                ret = false;
            };
        };
        $if(ret) {
            rec.t = root;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            Float3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_index = mat_index;

            ret = true;
        };
    };
    return ret;
}

