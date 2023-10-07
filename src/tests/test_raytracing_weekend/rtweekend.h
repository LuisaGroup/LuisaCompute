#pragma once

#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>

#include <cstdlib>

using namespace luisa;
using namespace luisa::compute;

static constexpr float infinity = 1e10f;

inline float random_float() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_float(min, max + 1));
}

UInt tea(UInt v0, UInt v1) noexcept {
    Var s0 = 0u;
    for (uint n = 0u; n < 4u; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }
    return v0;
}

Float frand(UInt &state) noexcept {
    constexpr uint lcg_a = 1664525u;
    constexpr uint lcg_c = 1013904223u;
    state = lcg_a * state + lcg_c;
    return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
}

Float frand(UInt &state, Float min, Float max) noexcept {
    return min + (max - min) * frand(state);
}

Float3 random_float3(UInt &seed) {
    return make_float3(frand(seed), frand(seed), frand(seed));
}

Float3 random_float3(UInt &seed, Float min, Float max) {
    return make_float3(frand(seed, min, max), frand(seed, min, max), frand(seed, min, max));
}

Float3 random_in_unit_sphere(UInt &seed) {
    Float3 p;
    $loop {
        p = random_float3(seed, -1.0f, 1.0f);
        $if (length_squared(p) < 1.0f) { $break; };
    };
    return p;
}

Float3 random_unit_vector(UInt &seed) {
    return normalize(random_in_unit_sphere(seed));
}

Float3 random_in_unit_disk(UInt &seed) {
    Float3 p;
    $loop {
        Float x = frand(seed, -1, 1);
        Float y = frand(seed, -1, 1);
        p = make_float3(x, y, 0.0f);
        $if (length_squared(p) < 1) { $break; };
    };
    return p;
}

Bool near_zero(Float3 e) {
    // Return true if the vector is close to zero in all dimensions.
    const float s = 1e-8f;
    return (abs(e.x) < s) & (abs(e.y) < s) & (abs(e.z) < s);
}

Float3 ray_reflect(const Float3 &v, const Float3 &n) {
    return v - 2.0f * dot(v, n) * n;
}

Float3 ray_refract(const Float3 &uv, const Float3 &n, Float etai_over_etat) {
    Float cos_theta = min(dot(-uv, n), 1.0f);
    Float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Float3 r_out_parallel = -sqrt(abs(1.0f - length_squared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

class material;
vector<shared_ptr<material>> materials;
