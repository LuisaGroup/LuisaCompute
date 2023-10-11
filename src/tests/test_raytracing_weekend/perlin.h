#pragma once

#include "rtweekend.h"

class perlin {
public:
    perlin(Device &device, Stream &stream) {
        for (int i = 0; i < point_count; ++i) {
            ranvec[i] = normalize(float3(random_float(-1, 1), random_float(-1, 1), random_float(-1, 1)));
        }

        perm_x = perlin_generate_perm();
        perm_y = perlin_generate_perm();
        perm_z = perlin_generate_perm();

        Rv = device.create_buffer<float3>(point_count);
        Px = device.create_buffer<int>(point_count);
        Py = device.create_buffer<int>(point_count);
        Pz = device.create_buffer<int>(point_count);
        stream << Rv.copy_from(ranvec.data())
               << Px.copy_from(perm_x.data())
               << Py.copy_from(perm_y.data())
               << Pz.copy_from(perm_z.data())
               << synchronize();
    }

    ~perlin() {}

    Float noise(const Float3 &p) const {
        auto u = p.x - floor(p.x);
        auto v = p.y - floor(p.y);
        auto w = p.z - floor(p.z);
        auto i = static_cast<Int>(floor(p.x));
        auto j = static_cast<Int>(floor(p.y));
        auto k = static_cast<Int>(floor(p.z));
        Float3 c[2][2][2];

        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    c[di][dj][dk] = Rv->read(
                        Px->read((i + di) & 255) ^
                        Py->read((j + dj) & 255) ^
                        Pz->read((k + dk) & 255));

        return perlin_interp(c, u, v, w);
    }

    Float turb(const Float3 &p, int depth = 7) const {
        Float accum = 0.0f;
        Float3 temp_p = p;
        Float weight = 1.0f;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(temp_p);
            weight *= 0.5f;
            temp_p *= 2.0f;
        }

        return abs(accum);
    }

private:
    static const int point_count = 256;
    std::array<float3, point_count> ranvec;
    std::array<int, point_count> perm_x;
    std::array<int, point_count> perm_y;
    std::array<int, point_count> perm_z;

    Buffer<float3> Rv;
    Buffer<int> Px, Py, Pz;

    static std::array<int, point_count> perlin_generate_perm() {
        std::array<int, point_count> p;

        for (int i = 0; i < perlin::point_count; i++)
            p[i] = i;

        permute(p, point_count);

        return p;
    }

    static void permute(std::array<int, point_count> &p, int n) {
        for (int i = n - 1; i > 0; i--) {
            int target = random_int(0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    static Float trilinear_interp(Float c[2][2][2], Float u, Float v, Float w) {
        Float accum = 0.0f;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    accum += (i * u + (1 - i) * (1 - u)) *
                             (j * v + (1 - j) * (1 - v)) *
                             (k * w + (1 - k) * (1 - w)) * c[i][j][k];

        return accum;
    }

    static Float perlin_interp(Float3 c[2][2][2], Float u, Float v, Float w) {
        auto uu = u * u * (3 - 2 * u);
        auto vv = v * v * (3 - 2 * v);
        auto ww = w * w * (3 - 2 * w);
        Float accum = 0.0f;

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    Float3 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) * (k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
                }

        return accum;
    }
};