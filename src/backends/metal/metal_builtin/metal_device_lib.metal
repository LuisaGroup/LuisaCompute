#include <metal_stdlib>

using namespace metal;

template<typename... T>
[[nodiscard, gnu::always_inline]] inline auto make_float2x2(T... args) {
    return float2x2(args...);
}

[[nodiscard, gnu::always_inline]] inline auto make_float2x2(float3x3 m) {
    return float2x2(m[0].xy, m[1].xy);
}

[[nodiscard, gnu::always_inline]] inline auto make_float2x2(float4x4 m) {
    return float2x2(m[0].xy, m[1].xy);
}

template<typename... T>
[[nodiscard, gnu::always_inline]] inline auto make_float3x3(T... args) {
    return float3x3(args...);
}

[[nodiscard, gnu::always_inline]] inline auto make_float3x3(float2x2 m) {
    return float3x3(
        float3(m[0], 0.0f),
        float3(m[1], 0.0f),
        float3(0.0f, 0.0f, 1.0f));
}

[[nodiscard, gnu::always_inline]] inline auto make_float3x3(float4x4 m) {
    return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

template<typename... T>
[[nodiscard, gnu::always_inline]] inline auto make_float4x4(T... args) {
    return float4x4(args...);
}

[[nodiscard, gnu::always_inline]] inline auto make_float4x4(float2x2 m) {
    return float4x4(
        float4(m[0], 0.0f, 0.0f),
        float4(m[1], 0.0f, 0.0f),
        float4(0.0f, 0.0f, 1.0f, 0.0f),
        float4(0.0f, 0.0f, 0.0f, 1.0f));
}

[[nodiscard, gnu::always_inline]] inline auto make_float4x4(float3x3 m) {
    return float4x4(
        float4(m[0], 0.0f),
        float4(m[1], 0.0f),
        float4(m[2], 0.0f),
        float4(0.0f, 0.0f, 0.0f, 1.0f));
}

template<typename T, typename I>
[[nodiscard, gnu::always_inline]] inline auto buffer_read(const device T *buffer, I index) {
    return buffer[index];
}

template<typename T, typename I>
[[gnu::always_inline]] inline void buffer_write(device T *buffer, I index, T value) {
    buffer[index] = value;
}

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto address_of(thread T &x) {
    return &x;
}

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto address_of(threadgroup T &x) {
    return &x;
}

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto address_of(device T &x) {
    return &x;
}

namespace detail {
template<typename T>
inline auto vector_element_impl(T v) { return v.x; }
}// namespace detail

template<typename T>
struct vector_element {
    using type = decltype(detail::vector_element_impl(T{}));
};

template<typename T>
using vector_element_t = typename vector_element<T>::type;

template<uint index, typename T>
[[nodiscard, gnu::always_inline]] inline auto vector_element_ptr(thread T &v) {
    return reinterpret_cast<thread vector_element_t<T> *>(&v) + index;
}

template<uint index, typename T>
[[nodiscard, gnu::always_inline]] inline auto vector_element_ptr(threadgroup T &v) {
    return reinterpret_cast<threadgroup vector_element_t<T> *>(&v) + index;
}

template<uint index, typename T>
[[nodiscard, gnu::always_inline]] inline auto vector_element_ptr(device T &v) {
    return reinterpret_cast<device vector_element_t<T> *>(&v) + index;
}

template<typename T, access a>
[[nodiscard, gnu::always_inline]] inline auto texture_read(texture2d<T, a> t, uint2 uv) {
    return t.read(uv);
}

template<typename T, access a>
[[nodiscard, gnu::always_inline]] inline auto texture_read(texture3d<T, a> t, uint3 uvw) {
    return t.read(uvw);
}

template<typename T, access a, typename Value>
[[gnu::always_inline]] inline void texture_write(texture2d<T, a> t, uint2 uv, Value value) {
    t.write(value, uv);
}

template<typename T, access a, typename Value>
[[gnu::always_inline]] inline void texture_write(texture3d<T, a> t, uint3 uvw, Value value) {
    t.write(value, uvw);
}

[[nodiscard]] inline auto inverse(float2x2 m) {
    const auto one_over_determinant = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
    return float2x2(m[1][1] * one_over_determinant,
                    -m[0][1] * one_over_determinant,
                    -m[1][0] * one_over_determinant,
                    +m[0][0] * one_over_determinant);
}

[[nodiscard]] inline auto inverse(float3x3 m) {
    const auto one_over_determinant = 1.0f / (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
    return float3x3(
        (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
        (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
        (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
        (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
        (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
        (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
        (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
        (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
        (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
}

[[nodiscard]] inline auto inverse(float4x4 m) {
    const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    const auto fac0 = float4{coef00, coef00, coef02, coef03};
    const auto fac1 = float4{coef04, coef04, coef06, coef07};
    const auto fac2 = float4{coef08, coef08, coef10, coef11};
    const auto fac3 = float4{coef12, coef12, coef14, coef15};
    const auto fac4 = float4{coef16, coef16, coef18, coef19};
    const auto fac5 = float4{coef20, coef20, coef22, coef23};
    const auto Vec0 = float4{m[1].x, m[0].x, m[0].x, m[0].x};
    const auto Vec1 = float4{m[1].y, m[0].y, m[0].y, m[0].y};
    const auto Vec2 = float4{m[1].z, m[0].z, m[0].z, m[0].z};
    const auto Vec3 = float4{m[1].w, m[0].w, m[0].w, m[0].w};
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    constexpr auto sign_a = float4{+1.0f, -1.0f, +1.0f, -1.0f};
    constexpr auto sign_b = float4{-1.0f, +1.0f, -1.0f, +1.0f};
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m[0] * float4{inv_0.x, inv_1.x, inv_2.x, inv_3.x};
    const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    const auto one_over_determinant = 1.0f / dot1;
    return float4x4(inv_0 * one_over_determinant,
                    inv_1 * one_over_determinant,
                    inv_2 * one_over_determinant,
                    inv_3 * one_over_determinant);
}

[[gnu::always_inline]] inline void block_barrier() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device int &a) {
    return reinterpret_cast<device atomic_int *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device uint &a) {
    return reinterpret_cast<device atomic_uint *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device float &a) {
    return &a;
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup int &a) {
    return reinterpret_cast<threadgroup atomic_int *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup uint &a) {
    return reinterpret_cast<threadgroup atomic_uint *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup float &a) {
    return &a;
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device const int &a) {
    return reinterpret_cast<device const atomic_int *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device const uint &a) {
    return reinterpret_cast<device const atomic_uint *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device const float &a) {
    return &a;
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup const int &a) {
    return reinterpret_cast<threadgroup const atomic_int *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup const uint &a) {
    return reinterpret_cast<threadgroup const atomic_uint *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup const float &a) {
    return &a;
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange(device atomic_int *a, int cmp, int val, memory_order) {
    atomic_compare_exchange_weak_explicit(a, &cmp, val, memory_order_relaxed, memory_order_relaxed);
    return cmp;
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange(threadgroup atomic_int *a, int cmp, int val, memory_order) {
    atomic_compare_exchange_weak_explicit(a, &cmp, val, memory_order_relaxed, memory_order_relaxed);
    return cmp;
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange(device atomic_uint *a, uint cmp, uint val, memory_order) {
    atomic_compare_exchange_weak_explicit(a, &cmp, val, memory_order_relaxed, memory_order_relaxed);
    return cmp;
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange(threadgroup atomic_uint *a, uint cmp, uint val, memory_order) {
    atomic_compare_exchange_weak_explicit(a, &cmp, val, memory_order_relaxed, memory_order_relaxed);
    return cmp;
}

// atomic operations for floating-point values
[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange_float(device float *a, float cmp_in, float val, memory_order) {
    auto cmp = as_type<int>(cmp_in);
    atomic_compare_exchange_weak_explicit(reinterpret_cast<device atomic_int *>(a), &cmp, as_type<int>(val), memory_order_relaxed, memory_order_relaxed);
    return as_type<float>(cmp);
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange_float(threadgroup float *a, float cmp_in, float val, memory_order) {
    auto cmp = as_type<int>(cmp_in);
    atomic_compare_exchange_weak_explicit(reinterpret_cast<threadgroup atomic_int *>(a), &cmp, as_type<int>(val), memory_order_relaxed, memory_order_relaxed);
    return as_type<float>(cmp);
}

[[gnu::always_inline, nodiscard]] inline auto atomic_exchange_explicit_float(device float *a, float val, memory_order) {
    return as_type<float>(atomic_exchange_explicit(reinterpret_cast<device atomic_int *>(a), as_type<int>(val), memory_order_relaxed));
}

[[gnu::always_inline, nodiscard]] inline auto atomic_exchange_explicit_float(threadgroup float *a, float val, memory_order) {
    return as_type<float>(atomic_exchange_explicit(reinterpret_cast<threadgroup atomic_int *>(a), as_type<int>(val), memory_order_relaxed));
}

[[gnu::always_inline, nodiscard]] inline auto atomic_fetch_add_explicit_float(device float *a, float val, memory_order) {
#if __METAL_VERSION__ >= 300
    auto aa = reinterpret_cast<device atomic_float *>(a);
    return atomic_fetch_add_explicit(aa, val, memory_order_relaxed);
#else
    auto ok = false;
    auto old_val = 0.0f;
    while (!ok) {
        old_val = *a;
        auto new_val = old_val + val;
        ok = atomic_compare_exchange_weak_explicit(
            reinterpret_cast<device atomic_int *>(a), reinterpret_cast<thread int *>(&old_val),
            as_type<int>(new_val), memory_order_relaxed, memory_order_relaxed);
    }
    return old_val;
#endif
}

[[gnu::always_inline, nodiscard]] inline auto atomic_fetch_add_explicit_float(threadgroup float *a, float val, memory_order) {
    auto ok = false;
    auto old_val = 0.0f;
    while (!ok) {
        old_val = *a;
        auto new_val = old_val + val;
        ok = atomic_compare_exchange_weak_explicit(
            reinterpret_cast<threadgroup atomic_int *>(a), reinterpret_cast<thread int *>(&old_val),
            as_type<int>(new_val), memory_order_relaxed, memory_order_relaxed);
    }
    return old_val;
}

[[gnu::always_inline, nodiscard]] inline auto atomic_fetch_sub_explicit_float(device float *a, float val, memory_order) {
#if __METAL_VERSION__ >= 300
    auto aa = reinterpret_cast<device atomic_float *>(a);
    return atomic_fetch_sub_explicit(aa, val, memory_order_relaxed);
#else
    return atomic_fetch_add_explicit_float(a, -val, memory_order_relaxed);
#endif
}

[[gnu::always_inline, nodiscard]] inline auto atomic_fetch_sub_explicit_float(threadgroup float *a, float val, memory_order) {
    return atomic_fetch_add_explicit_float(a, -val, memory_order_relaxed);
}

[[gnu::always_inline, nodiscard]] inline auto atomic_fetch_min_explicit_float(device float *a, float val, memory_order) {
    for (;;) {
        if (auto old = *a;
            old <= val ||
            atomic_compare_exchange_float(a, old, val, memory_order_relaxed)) {
            return old;
        }
    }
}

[[gnu::always_inline, nodiscard]] inline auto atomic_fetch_min_explicit_float(threadgroup float *a, float val, memory_order) {
    for (;;) {
        if (auto old = *a;
            old <= val ||
            atomic_compare_exchange_float(a, old, val, memory_order_relaxed)) {
            return old;
        }
    }
}

[[gnu::always_inline, nodiscard]] inline auto atomic_fetch_max_explicit_float(device float *a, float val, memory_order) {
    for (;;) {
        if (auto old = *a;
            old >= val ||
            atomic_compare_exchange_float(a, old, val, memory_order_relaxed)) {
            return old;
        }
    }
}

[[gnu::always_inline, nodiscard]] inline auto atomic_fetch_max_explicit_float(threadgroup float *a, float val, memory_order) {
    for (;;) {
        if (auto old = *a;
            old >= val ||
            atomic_compare_exchange_float(a, old, val, memory_order_relaxed)) {
            return old;
        }
    }
}

[[gnu::always_inline, nodiscard]] inline auto is_nan(float x) {
    auto u = as_type<uint>(x);
    return (u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu);
}

[[gnu::always_inline, nodiscard]] inline auto is_nan(float2 v) {
    return bool2(is_nan(v.x), is_nan(v.y));
}

[[gnu::always_inline, nodiscard]] inline auto is_nan(float3 v) {
    return bool3(is_nan(v.x), is_nan(v.y), is_nan(v.z));
}

[[gnu::always_inline, nodiscard]] inline auto is_nan(float4 v) {
    return bool4(is_nan(v.x), is_nan(v.y), is_nan(v.z), is_nan(v.w));
}

[[gnu::always_inline, nodiscard]] inline auto is_inf(float x) {
    auto u = as_type<uint>(x);
    return (u & 0x7F800000u) == 0x7F800000u && !(u & 0x7FFFFFu);
}

[[gnu::always_inline, nodiscard]] inline auto is_inf(float2 v) {
    return bool2(is_inf(v.x), is_inf(v.y));
}

[[gnu::always_inline, nodiscard]] inline auto is_inf(float3 v) {
    return bool3(is_inf(v.x), is_inf(v.y), is_inf(v.z));
}

[[gnu::always_inline, nodiscard]] inline auto is_inf(float4 v) {
    return bool4(is_inf(v.x), is_inf(v.y), is_inf(v.z), is_inf(v.w));
}

template<typename T>
[[gnu::always_inline, nodiscard]] inline auto select(T f, T t, bool b) {
    return b ? t : f;
}

struct alignas(16) BindlessItem {
    device const void *buffer;
    metal::uint sampler2d;
    metal::uint sampler3d;
    metal::texture2d<float> handle2d;
    metal::texture3d<float> handle3d;
};

[[nodiscard, gnu::always_inline]] constexpr sampler get_sampler(uint code) {
    constexpr const array<sampler, 16u> samplers{
        sampler(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::none),
        sampler(coord::normalized, address::repeat, filter::nearest, mip_filter::none),
        sampler(coord::normalized, address::mirrored_repeat, filter::nearest, mip_filter::none),
        sampler(coord::normalized, address::clamp_to_zero, filter::nearest, mip_filter::none),
        sampler(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::none),
        sampler(coord::normalized, address::repeat, filter::linear, mip_filter::none),
        sampler(coord::normalized, address::mirrored_repeat, filter::linear, mip_filter::none),
        sampler(coord::normalized, address::clamp_to_zero, filter::linear, mip_filter::none),
        sampler(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear, max_anisotropy(1)),
        sampler(coord::normalized, address::repeat, filter::linear, mip_filter::linear, max_anisotropy(1)),
        sampler(coord::normalized, address::mirrored_repeat, filter::linear, mip_filter::linear, max_anisotropy(1)),
        sampler(coord::normalized, address::clamp_to_zero, filter::linear, mip_filter::linear, max_anisotropy(1)),
        sampler(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear, max_anisotropy(16)),
        sampler(coord::normalized, address::repeat, filter::linear, mip_filter::linear, max_anisotropy(16)),
        sampler(coord::normalized, address::mirrored_repeat, filter::linear, mip_filter::linear, max_anisotropy(16)),
        sampler(coord::normalized, address::clamp_to_zero, filter::linear, mip_filter::linear, max_anisotropy(16))};
    __builtin_assume(code < 16u);
    return samplers[code];
}

struct alignas(16) Ray {
    array<float, 3> m0;
    float m1;
    array<float, 3> m2;
    float m3;
};

struct alignas(16) Hit {
    uint m0;
    uint m1;
    float2 m2;
};

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_sample2d(device const BindlessItem *heap, uint index, float2 uv) {
    device const auto &t = heap[index];
    return t.handle2d.sample(get_sampler(t.sampler2d), uv);
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_sample3d(device const BindlessItem *heap, uint index, float3 uvw) {
    device const auto &t = heap[index];
    return t.handle3d.sample(get_sampler(t.sampler3d), uvw);
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_sample2d_level(device const BindlessItem *heap, uint index, float2 uv, float lod) {
    device const auto &t = heap[index];
    return t.handle2d.sample(get_sampler(t.sampler2d), uv, level(lod));
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_sample3d_level(device const BindlessItem *heap, uint index, float3 uvw, float lod) {
    device const auto &t = heap[index];
    return t.handle3d.sample(get_sampler(t.sampler3d), uvw, level(lod));
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_sample2d_grad(device const BindlessItem *heap, uint index, float2 uv, float2 dpdx, float2 dpdy) {
    device const auto &t = heap[index];
    return t.handle2d.sample(get_sampler(t.sampler2d), uv, gradient2d(dpdx, dpdy));
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_sample3d_grad(device const BindlessItem *heap, uint index, float3 uvw, float3 dpdx, float3 dpdy) {
    device const auto &t = heap[index];
    return t.handle3d.sample(get_sampler(t.sampler3d), uvw, gradient3d(dpdx, dpdy));
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_size2d(device const BindlessItem *heap, uint i) {
    return uint2(heap[i].handle2d.get_width(), heap[i].handle2d.get_height());
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_size3d(device const BindlessItem *heap, uint i) {
    return uint3(heap[i].handle3d.get_width(), heap[i].handle3d.get_height(), heap[i].handle3d.get_depth());
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_size2d_level(device const BindlessItem *heap, uint i, uint lv) {
    return uint2(heap[i].handle2d.get_width(lv), heap[i].handle2d.get_height(lv));
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_size3d_level(device const BindlessItem *heap, uint i, uint lv) {
    return uint3(heap[i].handle3d.get_width(lv), heap[i].handle3d.get_height(lv), heap[i].handle3d.get_depth(lv));
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_read2d(device const BindlessItem *heap, uint i, uint2 uv) {
    return heap[i].handle2d.read(uv);
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_read3d(device const BindlessItem *heap, uint i, uint3 uvw) {
    return heap[i].handle3d.read(uvw);
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_read2d_level(device const BindlessItem *heap, uint i, uint2 uv, uint lv) {
    return heap[i].handle2d.read(uv, lv);
}

[[nodiscard, gnu::always_inline]] inline auto bindless_texture_read3d_level(device const BindlessItem *heap, uint i, uint3 uvw, uint lv) {
    return heap[i].handle3d.read(uvw, lv);
}

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto bindless_buffer_read(device const BindlessItem *heap, uint buffer_index, uint i) {
    return static_cast<device const T *>(heap[buffer_index].buffer)[i];
}

using namespace metal::raytracing;

struct alignas(16) Instance {
    array<float, 12> transform;
    uint options;
    uint mask;
    uint intersection_function_offset;
    uint mesh_index;
};

static_assert(sizeof(Instance) == 64u, "");

struct Accel {
    instance_acceleration_structure handle;
    device Instance *__restrict__ instances;
};

[[nodiscard, gnu::always_inline]] constexpr auto intersector_closest() {
    intersector<triangle_data, instancing> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    i.accept_any_intersection(false);
    return i;
}

[[nodiscard, gnu::always_inline]] constexpr auto intersector_any() {
    intersector<triangle_data, instancing> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    i.accept_any_intersection(true);
    return i;
}

[[nodiscard, gnu::always_inline]] inline auto make_ray(Ray r_in) {
    auto o = float3(r_in.m0[0], r_in.m0[1], r_in.m0[2]);
    auto d = float3(r_in.m2[0], r_in.m2[1], r_in.m2[2]);
    return ray{o, d, r_in.m1, r_in.m3};
}

[[nodiscard, gnu::always_inline]] inline auto trace_closest(Accel accel, Ray r) {
    auto isect = intersector_closest().intersect(make_ray(r), accel.handle);
    return isect.type == intersection_type::none ?
               Hit{0xffffffffu, 0xffffffffu, float2(0.0f)} :
               Hit{isect.instance_id,
                   isect.primitive_id,
                   isect.triangle_barycentric_coord};
}

[[nodiscard, gnu::always_inline]] inline auto trace_any(Accel accel, Ray r) {
    auto isect = intersector_any().intersect(make_ray(r), accel.handle);
    return isect.type != intersection_type::none;
}

[[nodiscard, gnu::always_inline]] inline auto accel_instance_transform(Accel accel, uint i) {
    auto m = accel.instances[i].transform;
    return float4x4(
        m[0], m[1], m[2], 0.0f,
        m[3], m[4], m[5], 0.0f,
        m[6], m[7], m[8], 0.0f,
        m[9], m[10], m[11], 1.0f);
}

[[gnu::always_inline]] inline void accel_set_instance_transform(Accel accel, uint i, float4x4 m) {
    auto p = accel.instances[i].transform.data();
    p[0] = m[0][0];
    p[1] = m[0][1];
    p[2] = m[0][2];
    p[3] = m[1][0];
    p[4] = m[1][1];
    p[5] = m[1][2];
    p[6] = m[2][0];
    p[7] = m[2][1];
    p[8] = m[2][2];
    p[9] = m[3][0];
    p[10] = m[3][1];
    p[11] = m[3][2];
}

[[gnu::always_inline]] inline void accel_set_instance_visibility(Accel accel, uint i, bool v) {
    accel.instances[i].mask = v ? ~0u : 0u;
}
