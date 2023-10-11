#pragma once

#include "rtweekend.h"

#include "hittable.h"

class xy_rect : public hittable {
public:
    xy_rect() {}

    xy_rect(float _x0, float _x1, float _y0, float _y1, float _k,
            shared_ptr<material> mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {
        bool foundMat = false;
        for (uint i = 0; i < materials.size(); i++) {
            if (mat == materials[i]) {
                mat_id = i;
                foundMat = true;
            }
        }
        if (!foundMat) {
            mat_id = materials.size();
            materials.push_back(mat);
        }
    };

    virtual Bool hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const override;

    virtual bool bounding_box(aabb &output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(float3(x0, y0, k - 0.0001f), float3(x1, y1, k + 0.0001f));
        return true;
    }

public:
    uint mat_id;
    shared_ptr<material> mp;
    float x0, x1, y0, y1, k;
};

Bool xy_rect::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    Bool ret = true;

    auto t = (k - r.origin().z) / r.direction().z;
    $if ((t < t_min) | (t > t_max)) {
        ret = false;
    }
    $else {
        auto x = r.origin().x + t * r.direction().x;
        auto y = r.origin().y + t * r.direction().y;
        $if ((x < x0) | (x > x1) | (y < y0) | (y > y1)) {
            ret = false;
        }
        $else {
            rec.u = (x - x0) / (x1 - x0);
            rec.v = (y - y0) / (y1 - y0);
            rec.t = t;
            auto outward_normal = Float3(0, 0, 1);
            rec.set_face_normal(r, outward_normal);
            // rec.mat_ptr = mp;
            rec.mat_id = mat_id;
            rec.p = r.at(t);
            ret = true;
        };
    };
    return ret;
}

class xz_rect : public hittable {
public:
    xz_rect() {}

    xz_rect(float _x0, float _x1, float _z0, float _z1, float _k,
            shared_ptr<material> mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {
        bool foundMat = false;
        for (uint i = 0; i < materials.size(); i++) {
            if (mat == materials[i]) {
                mat_id = i;
                foundMat = true;
            }
        }
        if (!foundMat) {
            mat_id = materials.size();
            materials.push_back(mat);
        }
    };

    virtual Bool hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const override;

    virtual bool bounding_box(aabb &output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Y
        // dimension a small amount.
        output_box = aabb(float3(x0, k - 0.0001f, z0), float3(x1, k + 0.0001f, z1));
        return true;
    }

public:
    uint mat_id;
    shared_ptr<material> mp;
    float x0, x1, z0, z1, k;
};

Bool xz_rect::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    Bool ret = true;

    auto t = (k - r.origin().y) / r.direction().y;
    $if ((t < t_min) | (t > t_max)) {
        ret = false;
    }
    $else {
        auto x = r.origin().x + t * r.direction().x;
        auto z = r.origin().z + t * r.direction().z;
        $if ((x < x0) | (x > x1) | (z < z0) | (z > z1)) {
            ret = false;
        }
        $else {
            rec.u = (x - x0) / (x1 - x0);
            rec.v = (z - z0) / (z1 - z0);
            rec.t = t;
            auto outward_normal = Float3(0, 1, 0);
            rec.set_face_normal(r, outward_normal);
            // rec.mat_ptr = mp;
            rec.mat_id = mat_id;
            rec.p = r.at(t);
            ret = true;
        };
    };
    return ret;
}

class yz_rect : public hittable {
public:
    yz_rect() {}

    yz_rect(float _y0, float _y1, float _z0, float _z1, float _k,
            shared_ptr<material> mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {
        bool foundMat = false;
        for (uint i = 0; i < materials.size(); i++) {
            if (mat == materials[i]) {
                mat_id = i;
                foundMat = true;
            }
        }
        if (!foundMat) {
            mat_id = materials.size();
            materials.push_back(mat);
        }
    };

    virtual Bool hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const override;

    virtual bool bounding_box(aabb &output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the X
        // dimension a small amount.
        output_box = aabb(float3(k - 0.0001f, y0, z0), float3(k + 0.0001f, y1, z1));
        return true;
    }

public:
    uint mat_id;
    shared_ptr<material> mp;
    float y0, y1, z0, z1, k;
};

Bool yz_rect::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    Bool ret = true;

    auto t = (k - r.origin().x) / r.direction().x;
    $if ((t < t_min) | (t > t_max)) {
        ret = false;
    }
    $else {
        auto y = r.origin().y + t * r.direction().y;
        auto z = r.origin().z + t * r.direction().z;
        $if ((y < y0) | (y > y1) | (z < z0) | (z > z1)) {
            ret = false;
        }
        $else {
            rec.u = (y - y0) / (y1 - y0);
            rec.v = (z - z0) / (z1 - z0);
            rec.t = t;
            auto outward_normal = Float3(1, 0, 0);
            rec.set_face_normal(r, outward_normal);
            // rec.mat_ptr = mp;
            rec.mat_id = mat_id;
            rec.p = r.at(t);
            ret = true;
        };
    };
    return ret;
}