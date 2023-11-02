#pragma once

#include "hittable.h"

class sphere : public hittable {
public:
    sphere() {}
    sphere(float3 cen, float r, shared_ptr<material> m)
        : center(cen), radius(r), mat_ptr(m) {
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

    virtual bool bounding_box(aabb &output_box) const override;

private:
    static void get_sphere_uv(const Float3 &p, Float &u, Float &v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = acos(-p.y);
        auto phi = atan2(-p.z, p.x) + pi;

        u = phi / (2 * pi);
        v = theta / pi;
    }

public:
    float3 center;
    float radius;
    uint mat_id;
    shared_ptr<material> mat_ptr;
};

Bool sphere::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    Bool ret = true;

    Float3 oc = r.origin() - center;
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
            rec.normal = (rec.p - center) / radius;
            Float3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            get_sphere_uv(outward_normal, rec.u, rec.v);
            // rec.mat_ptr = mat_ptr;
            rec.mat_id = mat_id;

            ret = true;
        };
    };
    return ret;
}

bool sphere::bounding_box(aabb &output_box) const {
    output_box = aabb(
        center - float3(radius, radius, radius),
        center + float3(radius, radius, radius));
    return true;
}