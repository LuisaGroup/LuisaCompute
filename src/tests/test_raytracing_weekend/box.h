#pragma once

#include "rtweekend.h"

#include "aarect.h"
#include "hittable_list.h"

class box : public hittable {
public:
    box() {}
    box(const float3 &p0, const float3 &p1, shared_ptr<material> ptr);

    virtual Bool hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const override;

    virtual bool bounding_box(aabb &output_box) const override {
        output_box = aabb(box_min, box_max);
        return true;
    }

public:
    float3 box_min;
    float3 box_max;
    hittable_list sides;
};

box::box(const float3 &p0, const float3 &p1, shared_ptr<material> ptr) {
    box_min = p0;
    box_max = p1;

    sides.add(make_shared<xy_rect>(p0.x, p1.x, p0.y, p1.y, p1.z, ptr));
    sides.add(make_shared<xy_rect>(p0.x, p1.x, p0.y, p1.y, p0.z, ptr));

    sides.add(make_shared<xz_rect>(p0.x, p1.x, p0.z, p1.z, p1.y, ptr));
    sides.add(make_shared<xz_rect>(p0.x, p1.x, p0.z, p1.z, p0.y, ptr));

    sides.add(make_shared<yz_rect>(p0.y, p1.y, p0.z, p1.z, p1.x, ptr));
    sides.add(make_shared<yz_rect>(p0.y, p1.y, p0.z, p1.z, p0.x, ptr));
}

Bool box::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    return sides.hit(r, t_min, t_max, rec, seed);
}
