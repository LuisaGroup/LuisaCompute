#pragma once

struct alignas(16) LCSurface {
    cudaSurfaceObject_t handle;
    lc_uint storage;
    unsigned short pixel_size_shift;
    unsigned short channel_count;
};

static_assert(sizeof(LCSurface) == 16);

template<typename A, typename B>
struct lc_is_same {
    static constexpr auto value = false;
};

template<typename A>
struct lc_is_same<A, A> {
    static constexpr auto value = true;
};

template<typename A, typename B>
constexpr auto lc_is_same_v = lc_is_same<A, B>::value;

template<typename...>
struct lc_always_false {
    static constexpr auto value = false;
};

template<typename... T>
constexpr auto lc_always_false_v = lc_always_false<T...>::value;

using lc_half = unsigned short;

template<typename T, size_t N>
class lc_array {

private:
    T _data[N];

public:
    template<typename... Elem>
    __device__ explicit constexpr lc_array(Elem... elem) noexcept: _data{elem...} {}
    [[nodiscard]] __device__ T &operator[](size_t i) noexcept { return _data[i]; }
    [[nodiscard]] __device__ T operator[](size_t i) const noexcept { return _data[i]; }
};

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_float(P x) noexcept {
    if constexpr (lc_is_same_v<P, char>) {
        return static_cast<unsigned char>(x) * (1.0f / 255.0f);
    } else if constexpr (lc_is_same_v<P, short>) {
        return static_cast<unsigned short>(x) * (1.0f / 65535.0f);
    } else if constexpr (lc_is_same_v<P, lc_half>) {
        return lc_half_to_float(x);
    } else if constexpr (lc_is_same_v<P, lc_float>) {
        return x;
    }
    return 0.0f;
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_int(P x) noexcept {
    if constexpr (lc_is_same_v<P, char>) {
        return static_cast<lc_int>(x);
    } else if constexpr (lc_is_same_v<P, short>) {
        return static_cast<lc_int>(x);
    } else if constexpr (lc_is_same_v<P, lc_int>) {
        return x;
    }
    return 0;
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_uint(P x) noexcept {
    if constexpr (lc_is_same_v<P, char>) {
        return static_cast<lc_uint>(static_cast<unsigned char>(x));
    } else if constexpr (lc_is_same_v<P, short>) {
        return static_cast<lc_uint>(static_cast<unsigned short>(x));
    } else if constexpr (lc_is_same_v<P, lc_int>) {
        return static_cast<lc_uint>(x);
    }
    return 0u;
}

template<typename T, typename P>
[[nodiscard]] __device__ inline auto lc_texel_read_convert(P p) noexcept {
    if constexpr (lc_is_same_v<T, lc_float>) {
        return lc_texel_to_float<P>(p);
    } else if constexpr (lc_is_same_v<T, lc_int>) {
        return lc_texel_to_int<P>(p);
    } else if constexpr (lc_is_same_v<T, lc_uint>) {
        return lc_texel_to_uint<P>(p);
    } else {
        static_assert(lc_always_false_v<T, P>);
    }
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_float_to_texel(lc_float x) noexcept {
    if constexpr (lc_is_same_v<P, char>) {
        return static_cast<char>(static_cast<unsigned char>(lc_round(lc_saturate(x) * 255.0f)));
    } else if constexpr (lc_is_same_v<P, short>) {
        return static_cast<short>(static_cast<unsigned short>(lc_round(lc_saturate(x) * 65535.0f)));
    } else if constexpr (lc_is_same_v<P, lc_half>) {
        return lc_float_to_half(x);
    } else if constexpr (lc_is_same_v<P, lc_float>) {
        return x;
    }
    return P{};
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_int_to_texel(int x) noexcept {
    if constexpr (lc_is_same_v<P, char>) {
        return static_cast<char>(x);
    } else if constexpr (lc_is_same_v<P, short>) {
        return static_cast<short>(x);
    } else if constexpr (lc_is_same_v<P, lc_int>) {
        return x;
    }
    return P{};
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_uint_to_texel(P x) noexcept {
    if constexpr (lc_is_same_v<P, char>) {
        return static_cast<char>(static_cast<unsigned char>(x));
    } else if constexpr (lc_is_same_v<P, short>) {
        return static_cast<short>(static_cast<unsigned short>(x));
    } else if constexpr (lc_is_same_v<P, lc_int>) {
        return static_cast<lc_int>(x);
    }
    return P{};
}

template<typename T, typename P>
[[nodiscard]] __device__ inline auto lc_texel_write_convert(T t) noexcept {
    if constexpr (lc_is_same_v<T, lc_float>) {
        return lc_float_to_texel<P>(t);
    } else if constexpr (lc_is_same_v<T, lc_int>) {
        return lc_int_to_texel<P>(t);
    } else if constexpr (lc_is_same_v<T, lc_uint>) {
        return lc_uint_to_texel<P>(t);
    } else {
        static_assert(lc_always_false_v<T, P>);
    }
}

template<typename T>
struct lc_vec4 {};

template<>
struct lc_vec4<lc_int> {
    using type = lc_int4;
};

template<>
struct lc_vec4<lc_uint> {
    using type = lc_uint4;
};

template<>
struct lc_vec4<lc_float> {
    using type = lc_float4;
};

template<typename T>
using lc_vec4_t = typename lc_vec4<T>::type;

template<typename T, typename P>
[[nodiscard]] __device__ inline auto lc_surf2d_read_impl(cudaSurfaceObject_t surf, lc_uint2 uv, lc_uint pixel_size_shift, lc_uint channel_count) noexcept {
    switch (channel_count) {
        case 1: {
            P x;
            surf2Dread(&x, surf, uv.x << pixel_size_shift, uv.y, cudaBoundaryModeZero);
            return lc_vec4_t<T>{
                lc_texel_read_convert<T, P>(x),
                static_cast<T>(0),
                static_cast<T>(0),
                static_cast<T>(0)};
        }
        case 2: {
            P x, y;
            surf2Dread(&x, surf, uv.x << pixel_size_shift, uv.y, cudaBoundaryModeZero);
            surf2Dread(&y, surf, (uv.x << pixel_size_shift) + sizeof(P), uv.y, cudaBoundaryModeZero);
            return lc_vec4_t<T>{
                lc_texel_read_convert<T, P>(x),
                lc_texel_read_convert<T, P>(y),
                static_cast<T>(0),
                static_cast<T>(0)};
        }
        default: {
            P x, y, z, w;
            surf2Dread(&x, surf, uv.x << pixel_size_shift, uv.y, cudaBoundaryModeZero);
            surf2Dread(&y, surf, (uv.x << pixel_size_shift) + sizeof(P), uv.y, cudaBoundaryModeZero);
            surf2Dread(&z, surf, (uv.x << pixel_size_shift) + sizeof(P) * 2, uv.y, cudaBoundaryModeZero);
            surf2Dread(&w, surf, (uv.x << pixel_size_shift) + sizeof(P) * 3, uv.y, cudaBoundaryModeZero);
            return lc_vec4_t<T>{
                lc_texel_read_convert<T, P>(x),
                lc_texel_read_convert<T, P>(y),
                lc_texel_read_convert<T, P>(z),
                lc_texel_read_convert<T, P>(w)};
        }
    }
}

template<typename T, typename P>
[[nodiscard]] __device__ inline auto lc_surf3d_read_impl(cudaSurfaceObject_t surf, lc_uint3 uvw, lc_uint pixel_size_shift, lc_uint channel_count) noexcept {
    switch (channel_count) {
        case 1: {
            P x;
            surf3Dread(&x, surf, uvw.x << pixel_size_shift, uvw.y, uvw.z, cudaBoundaryModeZero);
            return lc_vec4_t<T>{
                lc_texel_read_convert<T, P>(x),
                static_cast<T>(0),
                static_cast<T>(0),
                static_cast<T>(0)};
        }
        case 2: {
            P x, y;
            surf3Dread(&x, surf, uvw.x << pixel_size_shift, uvw.y, uvw.z, cudaBoundaryModeZero);
            surf3Dread(&y, surf, (uvw.x << pixel_size_shift) + sizeof(P), uvw.y, uvw.z, cudaBoundaryModeZero);
            return lc_vec4_t<T>{
                lc_texel_read_convert<T, P>(x),
                lc_texel_read_convert<T, P>(y),
                static_cast<T>(0),
                static_cast<T>(0)};
        }
        default: {
            P x, y, z, w;
            surf3Dread(&x, surf, uvw.x << pixel_size_shift, uvw.y, uvw.z, cudaBoundaryModeZero);
            surf3Dread(&y, surf, (uvw.x << pixel_size_shift) + sizeof(P), uvw.y, uvw.z, cudaBoundaryModeZero);
            surf3Dread(&z, surf, (uvw.x << pixel_size_shift) + sizeof(P) * 2, uvw.y, uvw.z, cudaBoundaryModeZero);
            surf3Dread(&w, surf, (uvw.x << pixel_size_shift) + sizeof(P) * 3, uvw.y, uvw.z, cudaBoundaryModeZero);
            return lc_vec4_t<T>{
                lc_texel_read_convert<T, P>(x),
                lc_texel_read_convert<T, P>(y),
                lc_texel_read_convert<T, P>(z),
                lc_texel_read_convert<T, P>(w)};
        }
    }
}

template<typename T, typename P, typename V>
[[nodiscard]] __device__ inline void lc_surf2d_write_impl(cudaSurfaceObject_t surf, V value, lc_uint2 uv, lc_uint pixel_size_shift, lc_uint channel_count) noexcept {
    switch (channel_count) {
        case 1:
            surf2Dwrite(lc_texel_write_convert<T, P>(value.x), surf, uv.x << pixel_size_shift, uv.y, cudaBoundaryModeZero);
            break;
        case 2:
            surf2Dwrite(lc_texel_write_convert<T, P>(value.x), surf, uv.x << pixel_size_shift, uv.y, cudaBoundaryModeZero);
            surf2Dwrite(lc_texel_write_convert<T, P>(value.y), surf, (uv.x << pixel_size_shift) + sizeof(P), uv.y, cudaBoundaryModeZero);
            break;
        default:
            surf2Dwrite(lc_texel_write_convert<T, P>(value.x), surf, uv.x << pixel_size_shift, uv.y, cudaBoundaryModeZero);
            surf2Dwrite(lc_texel_write_convert<T, P>(value.y), surf, (uv.x << pixel_size_shift) + sizeof(P), uv.y, cudaBoundaryModeZero);
            surf2Dwrite(lc_texel_write_convert<T, P>(value.z), surf, (uv.x << pixel_size_shift) + sizeof(P) * 2, uv.y, cudaBoundaryModeZero);
            surf2Dwrite(lc_texel_write_convert<T, P>(value.w), surf, (uv.x << pixel_size_shift) + sizeof(P) * 3, uv.y, cudaBoundaryModeZero);
            break;
    }
}

template<typename T, typename P, typename V>
[[nodiscard]] __device__ inline void lc_surf3d_write_impl(cudaSurfaceObject_t surf, V value, lc_uint3 uvw, lc_uint pixel_size_shift, lc_uint channel_count) noexcept {
    switch (channel_count) {
        case 1:
            surf3Dwrite(lc_texel_write_convert<T, P>(value.x), surf, uvw.x << pixel_size_shift, uvw.y, uvw.z, cudaBoundaryModeZero);
            break;
        case 2:
            surf3Dwrite(lc_texel_write_convert<T, P>(value.x), surf, uvw.x << pixel_size_shift, uvw.y, uvw.z, cudaBoundaryModeZero);
            surf3Dwrite(lc_texel_write_convert<T, P>(value.y), surf, (uvw.x << pixel_size_shift) + sizeof(P), uvw.y, uvw.z, cudaBoundaryModeZero);
            break;
        default:
            surf3Dwrite(lc_texel_write_convert<T, P>(value.x), surf, uvw.x << pixel_size_shift, uvw.y, uvw.z, cudaBoundaryModeZero);
            surf3Dwrite(lc_texel_write_convert<T, P>(value.y), surf, (uvw.x << pixel_size_shift) + sizeof(P), uvw.y, uvw.z, cudaBoundaryModeZero);
            surf3Dwrite(lc_texel_write_convert<T, P>(value.z), surf, (uvw.x << pixel_size_shift) + sizeof(P) * 2, uvw.y, uvw.z, cudaBoundaryModeZero);
            surf3Dwrite(lc_texel_write_convert<T, P>(value.w), surf, (uvw.x << pixel_size_shift) + sizeof(P) * 3, uvw.y, uvw.z, cudaBoundaryModeZero);
            break;
    }
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf2d_read(LCSurface surf, lc_uint2 uv) noexcept {
    switch (surf.storage) {
        case 0: return lc_surf2d_read_impl<T, char>(surf.handle, uv, surf.pixel_size_shift, surf.channel_count);
        case 1: return lc_surf2d_read_impl<T, short>(surf.handle, uv, surf.pixel_size_shift, surf.channel_count);
        case 2: return lc_surf2d_read_impl<T, int>(surf.handle, uv, surf.pixel_size_shift, surf.channel_count);
        case 3: return lc_surf2d_read_impl<T, unsigned short>(surf.handle, uv, surf.pixel_size_shift, surf.channel_count);
        default: return lc_surf2d_read_impl<T, float>(surf.handle, uv, surf.pixel_size_shift, surf.channel_count);
    }
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf3d_read(LCSurface surf, lc_uint3 uvw) noexcept {
    switch (surf.storage) {
        case 0: return lc_surf3d_read_impl<T, char>(surf.handle, uvw, surf.pixel_size_shift, surf.channel_count);
        case 1: return lc_surf3d_read_impl<T, short>(surf.handle, uvw, surf.pixel_size_shift, surf.channel_count);
        case 2: return lc_surf3d_read_impl<T, int>(surf.handle, uvw, surf.pixel_size_shift, surf.channel_count);
        case 3: return lc_surf3d_read_impl<T, unsigned short>(surf.handle, uvw, surf.pixel_size_shift, surf.channel_count);
        default: return lc_surf3d_read_impl<T, float>(surf.handle, uvw, surf.pixel_size_shift, surf.channel_count);
    }
}

template<typename T, typename V>
[[nodiscard]] __device__ inline void lc_surf2d_write(LCSurface surf, lc_uint2 uv, V value) noexcept {
    switch (surf.storage) {
        case 0: lc_surf2d_write_impl<T, char>(surf.handle, value, uv, surf.pixel_size_shift, surf.channel_count); break;
        case 1: lc_surf2d_write_impl<T, short>(surf.handle, value, uv, surf.pixel_size_shift, surf.channel_count); break;
        case 2: lc_surf2d_write_impl<T, int>(surf.handle, value, uv, surf.pixel_size_shift, surf.channel_count); break;
        case 3: lc_surf2d_write_impl<T, unsigned short>(surf.handle, value, uv, surf.pixel_size_shift, surf.channel_count); break;
        default: lc_surf2d_write_impl<T, float>(surf.handle, value, uv, surf.pixel_size_shift, surf.channel_count); break;
    }
}

template<typename T, typename V>
[[nodiscard]] __device__ inline auto lc_surf3d_write(LCSurface surf, lc_uint3 uvw, V value) noexcept {
    switch (surf.storage) {
        case 0: lc_surf3d_write_impl<T, char>(surf.handle, value, uvw, surf.pixel_size_shift, surf.channel_count); break;
        case 1: lc_surf3d_write_impl<T, short>(surf.handle, value, uvw, surf.pixel_size_shift, surf.channel_count); break;
        case 2: lc_surf3d_write_impl<T, int>(surf.handle, value, uvw, surf.pixel_size_shift, surf.channel_count); break;
        case 3: lc_surf3d_write_impl<T, unsigned short>(surf.handle, value, uvw, surf.pixel_size_shift, surf.channel_count); break;
        default: lc_surf3d_write_impl<T, float>(surf.handle, value, uvw, surf.pixel_size_shift, surf.channel_count); break;
    }
}
