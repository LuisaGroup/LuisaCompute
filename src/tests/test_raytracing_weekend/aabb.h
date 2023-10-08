#pragma once

#include "rtweekend.h"
#include "ray.h"

class aabb {
public:
    aabb() {}
    aabb(const float3 &a, const float3 &b) {
        minimum = a;
        maximum = b;
    }

    float3 min() const { return minimum; }
    float3 max() const { return maximum; }

    Bool hit(const ray &r, Float t_min, Float t_max, UInt &seed) const {
        Bool ret = true;

        for (uint a = 0; a < 3; a++) {
            $if (ret) {
                // old version
                // auto t0 = luisa::min((minimum[a] - r.origin()[a]) / r.direction()[a],
                //             (maximum[a] - r.origin()[a]) / r.direction()[a]);
                // auto t1 = luisa::max((minimum[a] - r.origin()[a]) / r.direction()[a],
                //             (maximum[a] - r.origin()[a]) / r.direction()[a]);

                auto invD = 1.0f / r.direction()[a];
                auto t0 = (def(min()[a]) - r.origin()[a]) * invD;
                auto t1 = (def(max()[a]) - r.origin()[a]) * invD;
                $if (invD < 0.0f) {
                    auto tmp = t0;
                    t0 = t1;
                    t1 = tmp;
                };

                t_min = luisa::compute::max(t0, t_min);
                t_max = luisa::compute::min(t1, t_max);
                $if (t_max <= t_min) {
                    ret = false;
                };
            };
        }

        return ret;
    }

    float3 minimum;
    float3 maximum;
};

aabb surrounding_box(aabb box0, aabb box1) {
    float3 small(fmin(box0.min().x, box1.min().x),
                 fmin(box0.min().y, box1.min().y),
                 fmin(box0.min().z, box1.min().z));

    float3 big(fmax(box0.max().x, box1.max().x),
               fmax(box0.max().y, box1.max().y),
               fmax(box0.max().z, box1.max().z));

    return aabb(small, big);
}