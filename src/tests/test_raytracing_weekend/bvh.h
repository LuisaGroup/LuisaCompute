#pragma once

#include <algorithm>

#include "rtweekend.h"

#include "hittable.h"
#include "hittable_list.h"

class bvh_node : public hittable {
public:
    bvh_node();

    bvh_node(const hittable_list &list)
        : bvh_node(list.objects, 0, list.objects.size()) {}

    bvh_node(
        const vector<shared_ptr<hittable>> &src_objects,
        size_t start, size_t end);

    virtual Bool hit(
        const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const override;

    virtual bool bounding_box(aabb &output_box) const override;

public:
    shared_ptr<hittable> left;
    shared_ptr<hittable> right;
    aabb box;
};

bool bvh_node::bounding_box(aabb &output_box) const {
    output_box = box;
    return true;
}

Bool bvh_node::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    Bool ret = true;

    $if (!box.hit(r, t_min, t_max, seed)) {
        ret = false;
    }
    $else {
        Bool hit_left = left->hit(r, t_min, t_max, rec, seed);
        Bool hit_right = right->hit(r, t_min, ite(hit_left, rec.t, t_max), rec, seed);

        ret = hit_left | hit_right;
    };

    return ret;
}

inline bool box_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis) {
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(box_a) || !b->bounding_box(box_b))
        LUISA_ERROR("No bounding box in bvh_node constructor.\n");

    return box_a.min()[axis] < box_b.min()[axis];
}

bool box_x_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return box_compare(a, b, 0);
}

bool box_y_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return box_compare(a, b, 1);
}

bool box_z_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return box_compare(a, b, 2);
}

bvh_node::bvh_node(
    const vector<shared_ptr<hittable>> &src_objects,
    size_t start, size_t end) {
    auto objects = src_objects;// Create a modifiable array of the source scene objects

    int axis = random_int(0, 2);
    auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare :
                                                                  box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    } else if (object_span == 2) {
        if (comparator(objects[start], objects[start + 1])) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            left = objects[start + 1];
            right = objects[start];
        }
    } else {
        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        auto mid = start + object_span / 2;
        left = make_shared<bvh_node>(objects, start, mid);
        right = make_shared<bvh_node>(objects, mid, end);
    }

    aabb box_left, box_right;

    if (!left->bounding_box(box_left) || !right->bounding_box(box_right))
        LUISA_ERROR("No bounding box in bvh_node constructor.\n");

    box = surrounding_box(box_left, box_right);
}