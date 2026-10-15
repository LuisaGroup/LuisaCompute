// Procedural Texture Generator -- shared device-side noise helpers.
//
// Every helper is an inline factory returning a luisa::compute::Callable. The
// algorithm .cpp files instantiate them locally and capture them into their
// kernels (captures are transitive, so a kernel capturing a callable that
// captures another callable is fine). All functions are deterministic:
// f(position, seed) -- no tables, no atomics, no shared state.

#pragma once

#include <luisa/luisa-compute.h>

namespace procedural_tex {

using namespace luisa;
using namespace luisa::compute;

// Integer-lattice hash -> [0, 1). pcg-style avalanche on a linear congruential
// step; seed is XORed in by the callers so the same lattice point produces
// decorrelated values for different noise fields.
[[nodiscard]] inline auto make_hash01() noexcept {
    return Callable{[](UInt x, UInt y) noexcept {
        UInt h = x * 1664525u + y * 1013904223u;
        h = h ^ (h >> 16u);
        h = h * 2246822519u;
        h = h ^ (h >> 13u);
        return cast<float>(h & 0x00ffffffu) * (1.0f / 16777216.0f);
    }};
}

// Quintic Perlin fade 6t^5 - 15t^4 + 10t^3 (component-wise).
[[nodiscard]] inline auto make_fade_quintic() noexcept {
    return Callable{[](Float2 t) noexcept {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }};
}

// Smoothstep fade 3t^2 - 2t^3 (component-wise); cheaper than the quintic fade.
[[nodiscard]] inline auto make_fade_smoothstep() noexcept {
    return Callable{[](Float2 t) noexcept {
        return t * t * (3.0f - 2.0f * t);
    }};
}

// 2D value noise. `p` is in lattice units; returns values roughly in [0, 1].
[[nodiscard]] inline auto make_value_noise_2d() noexcept {
    auto hash = make_hash01();
    auto fade = make_fade_quintic();
    return Callable{[=](Float2 p, UInt seed) noexcept {
        Float2 i = floor(p);
        Float2 f = p - i;
        f = fade(f);
        UInt ix = cast<uint>(i.x);
        UInt iy = cast<uint>(i.y);
        Float v00 = hash(ix + 53u * seed, iy + 97u * seed);
        Float v10 = hash(ix + 1u + 53u * seed, iy + 97u * seed);
        Float v01 = hash(ix + 53u * seed, iy + 1u + 97u * seed);
        Float v11 = hash(ix + 1u + 53u * seed, iy + 1u + 97u * seed);
        return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
    }};
}

// 2D improved Perlin gradient noise. `p` is in lattice units; returns values
// roughly in [-1, 1]. Gradients are unit vectors hashed per lattice corner.
[[nodiscard]] inline auto make_perlin_2d() noexcept {
    auto hash = make_hash01();
    auto fade = make_fade_quintic();
    return Callable{[=](Float2 p, UInt seed) noexcept {
        Float2 i = floor(p);
        Float2 f = p - i;
        f = fade(f);
        UInt ix = cast<uint>(i.x);
        UInt iy = cast<uint>(i.y);
        auto grad = [&](UInt x, UInt y) noexcept {
            Float a = hash(x + 29u * seed, y + 17u * seed) * (2.0f * pi);
            return make_float2(cos(a), sin(a));
        };
        Float2 g00 = grad(ix, iy);
        Float2 g10 = grad(ix + 1u, iy);
        Float2 g01 = grad(ix, iy + 1u);
        Float2 g11 = grad(ix + 1u, iy + 1u);
        Float n00 = dot(g00, f);
        Float n10 = dot(g10, f - make_float2(1.0f, 0.0f));
        Float n01 = dot(g01, f - make_float2(0.0f, 1.0f));
        Float n11 = dot(g11, f - make_float2(1.0f, 1.0f));
        return lerp(lerp(n00, n10, f.x), lerp(n01, n11, f.x), f.y) * 1.41421f;
    }};
}

// 2D simplex noise (Perlin 2001). `p` is in simplex-lattice units; returns
// values roughly in [-1, 1].
[[nodiscard]] inline auto make_simplex_2d() noexcept {
    auto hash = make_hash01();
    return Callable{[=](Float2 p, UInt seed) noexcept {
        constexpr auto f2 = 0.36602540378f;// 0.5 * (sqrt(3) - 1)
        constexpr auto g2 = 0.21132486540f;// (3 - sqrt(3)) / 6
        Float s = (p.x + p.y) * f2;
        Float2 ip = floor(p + s);
        Float t = (ip.x + ip.y) * g2;
        Float2 x0 = p - (ip - t);
        UInt2 ii = make_uint2(cast<uint>(ip.x), cast<uint>(ip.y));
        UInt i1 = select(0u, 1u, x0.x > x0.y);
        UInt i2 = select(1u, 0u, x0.x > x0.y);
        Float2 x1 = x0 - make_float2(cast<float>(i1), cast<float>(i2)) + g2;
        Float2 x2 = x0 - 1.0f + 2.0f * g2;
        auto grad = [&](UInt x, UInt y) noexcept {
            Float a = hash(x + 11u * seed, y + 7u * seed) * (2.0f * pi);
            return make_float2(cos(a), sin(a));
        };
        Float n = def(0.0f);
        Float t0 = 0.5f - dot(x0, x0);
        Float t1 = 0.5f - dot(x1, x1);
        Float t2 = 0.5f - dot(x2, x2);
        $if (t0 > 0.0f) {
            t0 = t0 * t0;
            n = n + t0 * t0 * dot(grad(ii.x, ii.y), x0);
        };
        $if (t1 > 0.0f) {
            t1 = t1 * t1;
            n = n + t1 * t1 * dot(grad(ii.x + i1, ii.y + i2), x1);
        };
        $if (t2 > 0.0f) {
            t2 = t2 * t2;
            n = n + t2 * t2 * dot(grad(ii.x + 1u, ii.y + 1u), x2);
        };
        return n * 70.0f;
    }};
}

// 2D Worley (cellular) noise, Euclidean distance. `p` is in cell units.
// Returns (F1, F2) -- the first and second closest-feature-point distances,
// roughly in [0, ~1.4] for jitter <= 1. The caller maps F1/F2/F2-F1 to color.
[[nodiscard]] inline auto make_worley_2d() noexcept {
    auto hash = make_hash01();
    return Callable{[=](Float2 p, UInt seed, Float jitter) noexcept {
        Float2 ip = floor(p);
        Float2 fp = p - ip;
        UInt ix = cast<uint>(ip.x);
        UInt iy = cast<uint>(ip.y);
        Float f1 = def(1e9f);
        Float f2 = def(1e9f);
        for (auto oy = -1; oy <= 1; ++oy) {
            for (auto ox = -1; ox <= 1; ++ox) {
                Int gx = cast<int>(ip.x) + ox;
                Int gy = cast<int>(ip.y) + oy;
                UInt ux = cast<uint>(gx + 32768);
                UInt uy = cast<uint>(gy + 32768);
                Float2 o = make_float2(hash(ux + 3u * seed, uy + 5u * seed),
                                       hash(ux + 5u * seed, uy + 3u * seed));
                o = o * jitter;
                Float2 r = make_float2(cast<float>(ox), cast<float>(oy)) + o - fp;
                Float d = sqrt(dot(r, r));
                f2 = min(f2, max(f1, d));
                f1 = min(f1, d);
            }
        }
        return make_float2(f1, f2) * 1.2f;
    }};
}

// Remap x from [in_min, in_max] to [out_min, out_max], clamped to the output
// range.
[[nodiscard]] inline auto make_remap() noexcept {
    return Callable{[](Float x, Float in_min, Float in_max, Float out_min, Float out_max) noexcept {
        Float t = clamp((x - in_min) / (in_max - in_min), 0.0f, 1.0f);
        return lerp(out_min, out_max, t);
    }};
}

}// namespace procedural_tex
