#pragma once

#include "rtweekend.h"

#include "hittable.h"
#include "material.h"
#include "texture.h"

class constant_medium : public hittable {
public:
    constant_medium(shared_ptr<hittable> b, float d, shared_ptr<texture> a)
        : boundary(b),
          neg_inv_density(-1 / d) {
        auto mat = make_shared<isotropic>(a);
        phase_function = mat;

        mat_id = materials.size();
        materials.push_back(mat);
    }

    constant_medium(shared_ptr<hittable> b, float d, float3 c)
        : boundary(b),
          neg_inv_density(-1 / d) {
        auto mat = make_shared<isotropic>(c);
        phase_function = mat;

        mat_id = materials.size();
        materials.push_back(mat);
    }

    virtual Bool hit(
        const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const override;

    virtual bool bounding_box(aabb &output_box) const override {
        return boundary->bounding_box(output_box);
    }

public:
    shared_ptr<hittable> boundary;
    uint mat_id;
    shared_ptr<material> phase_function;
    float neg_inv_density;
};

Bool constant_medium::hit(const ray &r, Float t_min, Float t_max, hit_record &rec, UInt &seed) const {
    Bool ret = true;

    // Print occasional samples when debugging. To enable, set enableDebug true.
    // const bool enableDebug = false;
    // const bool debugging = enableDebug && random_float() < 0.00001f;

    hit_record rec1, rec2;

    $if (!boundary->hit(r, -infinity, infinity, rec1, seed)) {
        ret = false;
    }
    $else {

        $if (!boundary->hit(r, rec1.t + 0.0001f, infinity, rec2, seed)) {
            ret = false;
        }
        $else {

            // if (debugging) LUISA_INFO("\nt_min={}, t_max={}\n", rec1.t, rec2.t);

            $if (rec1.t < t_min) { rec1.t = t_min; };
            $if (rec2.t > t_max) { rec2.t = t_max; };

            $if (rec1.t >= rec2.t) {
                ret = false;
            }
            $else {

                $if (rec1.t < 0) {
                    rec1.t = 0;
                };

                const auto ray_length = length(r.direction());
                const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
                const auto hit_distance = neg_inv_density * log(frand(seed));

                $if (hit_distance > distance_inside_boundary) {
                    ret = false;
                }
                $else {

                    rec.t = rec1.t + hit_distance / ray_length;
                    rec.p = r.at(rec.t);

                    // if (debugging) {
                    //     LUISA_INFO("hit_distance = {}\nrec.t = {}\nrec.p = {}\n", hit_distance, rec.t, rec.p);
                    // }

                    rec.normal = Float3(1, 0, 0);// arbitrary
                    rec.front_face = true;       // also arbitrary
                    // rec.mat_ptr = phase_function;
                    rec.mat_id = mat_id;

                    ret = true;
                };
            };
        };
    };

    return ret;
}