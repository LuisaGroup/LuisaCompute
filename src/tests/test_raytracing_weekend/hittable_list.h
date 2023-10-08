#pragma once

#include "hittable.h"
#include "aabb.h"

class hittable_list : public hittable {
public:
    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(shared_ptr<hittable> object) { objects.push_back(object); }

    virtual Bool hit(
        const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const override;

    virtual bool bounding_box(
        aabb &output_box) const override;

    void shuffle() {
        auto temp = objects[0];
        objects[0] = objects[3];
        objects[3] = temp;
    }

public:
    vector<shared_ptr<hittable>> objects;
};

Bool hittable_list::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    hit_record temp_rec;
    Bool hit_anything = false;
    Float closest_so_far = t_max;

    for (uint i = 0; i < objects.size(); i++) {
        auto &object = objects[i];
        $if (object->hit(r, t_min, closest_so_far, temp_rec, seed)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        };
    }

    return hit_anything;
}

bool hittable_list::bounding_box(aabb &output_box) const {
    if (objects.empty()) return false;

    aabb temp_box;
    bool first_box = true;

    for (const auto &object : objects) {
        if (!object->bounding_box(temp_box)) return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}