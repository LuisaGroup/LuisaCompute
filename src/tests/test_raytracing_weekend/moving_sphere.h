#pragma once

#include "rtweekend.h"

#include "hittable.h"
#include "aabb.h"

class moving_sphere : public hittable {
public:
    moving_sphere() {}
    moving_sphere(
        float3 cen0, float3 cen1, float _time0, float _time1, float r, shared_ptr<material> m)
        : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m) {
        bool foundMat = false;
        for (uint i = 0; i < materials.size(); i++) {
            if (m == materials[i]) {
                mat_id = i;
                foundMat = true;
            }
        }
        if (!foundMat) {
            mat_id = materials.size();
            materials.push_back(m);
        }
    };

    virtual Bool hit(
        const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const override;

    virtual bool bounding_box(
        aabb &output_box) const override;

    Float3 center(Float time) const;

public:
    float3 center0, center1;
    float time0, time1;
    float radius;
    uint mat_id;
    shared_ptr<material> mat_ptr;
};

Float3 moving_sphere::center(Float time) const {
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

Bool moving_sphere::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    Bool ret = true;

    Float3 oc = r.origin() - center(r.time());
    Float a = length_squared(r.direction());
    Float half_b = dot(oc, r.direction());
    Float c = length_squared(oc) - radius * radius;

    Float discriminant = half_b * half_b - a * c;
    $if (discriminant < 0) {
        ret = false;
    }
    $else {
        Float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        Float root = (-half_b - sqrtd) / a;
        $if ((root < t_min) | (t_max < root)) {
            root = (-half_b + sqrtd) / a;
            $if ((root < t_min) | (t_max < root)) {
                ret = false;
            };
        };
        $if (ret) {
            rec.t = root;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;// TODO: useless?
            Float3 outward_normal = (rec.p - center(r.time())) / radius;
            rec.set_face_normal(r, outward_normal);
            // rec.mat_ptr = mat_ptr;
            rec.mat_id = mat_id;

            ret = true;
        };
    };
    return ret;
}

bool moving_sphere::bounding_box(aabb &output_box) const {
    aabb box0(
        center0 - float3(radius, radius, radius),
        center0 + float3(radius, radius, radius));
    aabb box1(
        center1 - float3(radius, radius, radius),
        center1 + float3(radius, radius, radius));
    output_box = surrounding_box(box0, box1);
    return true;
}