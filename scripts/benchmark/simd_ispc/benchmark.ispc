// Independent ISPC controls for the Luisa SIMD comparison suite.
//
// Algorithm provenance:
// - Mandelbrot follows the public ISPC example's conventional recurrence.
// - masked_stream follows the intent of ISPC benchmark 08_masked_load_store.
// - aos_to_soa follows ISPC benchmarks 01_aossoa/02_soaaos but uses explicit
//   scalar fields so the compared external memory layout is unambiguous.
// - gemm matches the repository's row-major 256x256 DSL GEMM control.
// - path_trace is an asset-free analytic workload authored for this suite.
//
// ISPC is BSD-3-Clause licensed. No code from this file enters production.

#ifndef LUISA_ISPC_SUFFIX
#error LUISA_ISPC_SUFFIX must give this object a unique symbol suffix
#endif

#define LUISA_CONCAT_IMPL(a, b) a##b
#define LUISA_CONCAT(a, b) LUISA_CONCAT_IMPL(a, b)
#define LUISA_EXPORT(name) LUISA_CONCAT(name, LUISA_ISPC_SUFFIX)

typedef float<3> Float3;

static inline float dot3(Float3 a, Float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Float3 vec3(float x, float y, float z) {
    Float3 value = {x, y, z};
    return value;
}

static inline Float3 normalize3(Float3 v) {
    float length_squared = dot3(v, v);
    return v * (1.0f / sqrt(length_squared));
}

static inline uint32 lcg_next(uint32 &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static inline float lcg_float(uint32 &state) {
    return (float)(lcg_next(state) & 0x00ffffffu) *
           (1.0f / 16777216.0f);
}

static inline uint32 pixel_seed(uint32 x, uint32 y, uint32 frame) {
    uint32 state = x * 1973u + y * 9277u + frame * 26699u + 911u;
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

export void LUISA_EXPORT(luisa_ispc_mandelbrot)(
    uniform uint32 width, uniform uint32 height,
    uniform uint32 row_begin, uniform uint32 row_end,
    uniform uint32 output[]) {
    uniform uint32 max_iterations = 256u;
    uniform float x0 = -2.0f;
    uniform float y0 = -1.25f;
    uniform float dx = 3.0f / (float)width;
    uniform float dy = 2.5f / (float)height;
    for (uniform uint32 y = row_begin; y < row_end; y++) {
        foreach (x = 0 ... width) {
            float c_re = x0 + (float)x * dx;
            float c_im = y0 + (float)y * dy;
            float z_re = c_re;
            float z_im = c_im;
            uint32 iteration = 0u;
            while (iteration < max_iterations &&
                   z_re * z_re + z_im * z_im <= 4.0f) {
                float next_re = z_re * z_re - z_im * z_im + c_re;
                float next_im = 2.0f * z_re * z_im + c_im;
                z_re = next_re;
                z_im = next_im;
                iteration++;
            }
            output[y * width + x] = iteration;
        }
    }
}

export void LUISA_EXPORT(luisa_ispc_masked_stream)(
    uniform const float a[], uniform const float b[],
    uniform const uint32 mask[], uniform float output[],
    uniform uint32 begin, uniform uint32 end) {
    foreach (index = begin ... end) {
        if (mask[index] != 0u) {
            output[index] = b[index] - a[index];
        } else {
            output[index] = 0.0f;
        }
    }
}

export void LUISA_EXPORT(luisa_ispc_aos_to_soa)(
    uniform const float input[], uniform float output_x[],
    uniform float output_y[], uniform float output_z[],
    uniform float output_w[], uniform uint32 begin, uniform uint32 end) {
    foreach (index = begin ... end) {
        uint32 base = index * 4u;
        output_x[index] = input[base + 0u];
        output_y[index] = input[base + 1u];
        output_z[index] = input[base + 2u];
        output_w[index] = input[base + 3u];
    }
}

export void LUISA_EXPORT(luisa_ispc_gemm)(
    uniform const float lhs[], uniform const float rhs[],
    uniform float output[], uniform uint32 row_begin,
    uniform uint32 row_end) {
    uniform uint32 matrix_size = 256u;
    for (uniform uint32 row = row_begin; row < row_end; row++) {
        foreach (column = 0 ... matrix_size) {
            float sum = 0.0f;
            for (uniform uint32 inner = 0u; inner < matrix_size; inner++) {
                sum += lhs[row * matrix_size + inner] *
                       rhs[inner * matrix_size + column];
            }
            output[row * matrix_size + column] = sum;
        }
    }
}

struct PathHit {
    float t;
    Float3 normal;
    uint32 material;
};

static inline void intersect_sphere(
    Float3 origin, Float3 direction, uniform Float3 center,
    uniform float radius, uniform uint32 material, PathHit &hit) {
    Float3 oc = origin - center;
    float b = dot3(oc, direction);
    float c = dot3(oc, oc) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant > 0.0f) {
        float t = -b - sqrt(discriminant);
        if (t > 1.0e-4f && t < hit.t) {
            hit.t = t;
            hit.normal = (origin + direction * t - center) / radius;
            hit.material = material;
        }
    }
}

static inline Float3 trace_path(
    uint32 x, uint32 y, uniform uint32 width,
    uniform uint32 height, uniform uint32 frame) {
    float aspect = (float)width / (float)height;
    float px = (2.0f * ((float)x + 0.5f) / (float)width - 1.0f) * aspect;
    float py = 1.0f - 2.0f * ((float)y + 0.5f) / (float)height;
    Float3 origin = {0.0f, 0.6f, 4.5f};
    Float3 direction = normalize3(vec3(px, py, -1.6f));
    Float3 throughput = vec3(1.0f, 1.0f, 1.0f);
    Float3 radiance = vec3(0.0f, 0.0f, 0.0f);
    uint32 state = pixel_seed(x, y, frame);

    uniform Float3 sphere_center_0 = {-1.1f, 0.0f, -3.5f};
    uniform Float3 sphere_center_1 = {1.1f, -0.1f, -4.0f};
    uniform Float3 sphere_center_2 = {0.0f, 3.4f, -3.0f};
    uniform Float3 sphere_center_3 = {0.0f, 0.1f, -6.2f};

    for (uniform uint32 depth = 0u; depth < 8u; depth++) {
        PathHit hit;
        hit.t = 1.0e30f;
        hit.normal = vec3(0.0f, 0.0f, 0.0f);
        hit.material = 0xffffffffu;
        intersect_sphere(origin, direction, sphere_center_0,
                         1.0f, 0u, hit);
        intersect_sphere(origin, direction, sphere_center_1,
                         0.9f, 1u, hit);
        intersect_sphere(origin, direction, sphere_center_2,
                         0.75f, 2u, hit);
        intersect_sphere(origin, direction, sphere_center_3,
                         1.3f, 3u, hit);
        if (abs(direction.y) > 1.0e-6f) {
            float plane_t = (-1.0f - origin.y) / direction.y;
            if (plane_t > 1.0e-4f && plane_t < hit.t) {
                hit.t = plane_t;
                hit.normal = vec3(0.0f, 1.0f, 0.0f);
                hit.material = 4u;
            }
        }

        if (hit.material == 0xffffffffu) {
            float blend = 0.5f * (direction.y + 1.0f);
            Float3 sky = (1.0f - blend) * vec3(0.08f, 0.10f, 0.15f) +
                         blend * vec3(0.35f, 0.50f, 0.80f);
            radiance += throughput * sky;
            break;
        }
        if (hit.material == 2u) {
            radiance += throughput * vec3(7.0f, 6.5f, 5.5f);
            break;
        }

        Float3 albedo = {0.72f, 0.72f, 0.72f};
        if (hit.material == 0u) {
            albedo = vec3(0.82f, 0.20f, 0.16f);
        } else if (hit.material == 1u) {
            albedo = vec3(0.18f, 0.72f, 0.25f);
        } else if (hit.material == 3u) {
            albedo = vec3(0.20f, 0.30f, 0.82f);
        }
        throughput *= albedo;
        Float3 hit_position = origin + direction * hit.t;
        float random_x = 2.0f * lcg_float(state) - 1.0f;
        float random_y = 2.0f * lcg_float(state) - 1.0f;
        float random_z = 2.0f * lcg_float(state) - 1.0f;
        Float3 random_direction = {random_x, random_y, random_z};
        float random_length_squared = dot3(random_direction, random_direction);
        if (random_length_squared < 1.0e-12f) {
            random_direction = hit.normal;
        } else {
            random_direction *= 1.0f / sqrt(random_length_squared);
        }
        if (dot3(random_direction, hit.normal) < 0.0f) {
            random_direction = -random_direction;
        }
        origin = hit_position + hit.normal * 1.0e-3f;
        direction = normalize3(hit.normal + random_direction);

        if (depth >= 3u) {
            float survival = max(throughput.x,
                                 max(throughput.y, throughput.z));
            survival = clamp(survival, 0.1f, 0.95f);
            if (lcg_float(state) >= survival) {
                break;
            }
            throughput /= survival;
        }
    }
    return radiance;
}

export void LUISA_EXPORT(luisa_ispc_path_trace)(
    uniform uint32 width, uniform uint32 height, uniform uint32 frame,
    uniform uint32 row_begin, uniform uint32 row_end,
    uniform float output[]) {
    for (uniform uint32 y = row_begin; y < row_end; y++) {
        foreach (x = 0 ... width) {
            Float3 color = trace_path(x, y, width, height, frame);
            uint32 base = (y * width + x) * 4u;
            output[base + 0u] = color.x;
            output[base + 1u] = color.y;
            output[base + 2u] = color.z;
            output[base + 3u] = 1.0f;
        }
    }
}
