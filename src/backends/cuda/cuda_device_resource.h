#pragma once

enum struct LCPixelStorage : lc_uint {

    BYTE1,
    BYTE2,
    BYTE4,

    SHORT1,
    SHORT2,
    SHORT4,

    INT1,
    INT2,
    INT4,

    HALF1,
    HALF2,
    HALF4,

    FLOAT1,
    FLOAT2,
    FLOAT4
};

struct alignas(16) LCSurface {
    cudaSurfaceObject_t handle;
    LCPixelStorage storage;
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

template<typename P, typename T>
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

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf2d_read(LCSurface surf, lc_uint2 p) noexcept {
    lc_vec4_t<T> result{0, 0, 0, 0};
    switch (surf.storage) {
        case LCPixelStorage::BYTE1: {
            auto v = surf2Dread<char>(surf.handle, p.x * sizeof(char), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::BYTE2: {
            auto v = surf2Dread<char2>(surf.handle, p.x * sizeof(char2), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::BYTE4: {
            auto v = surf2Dread<char4>(surf.handle, p.x * sizeof(char4), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        case LCPixelStorage::SHORT1: {
            auto v = surf2Dread<short>(surf.handle, p.x * sizeof(short), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::SHORT2: {
            auto v = surf2Dread<short2>(surf.handle, p.x * sizeof(short2), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::SHORT4: {
            auto v = surf2Dread<short4>(surf.handle, p.x * sizeof(short4), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        case LCPixelStorage::INT1: {
            auto v = surf2Dread<int>(surf.handle, p.x * sizeof(int), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::INT2: {
            auto v = surf2Dread<int2>(surf.handle, p.x * sizeof(int2), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::INT4: {
            auto v = surf2Dread<int4>(surf.handle, p.x * sizeof(int4), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        case LCPixelStorage::HALF1: {
            auto v = surf2Dread<unsigned short>(surf.handle, p.x * sizeof(unsigned short), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::HALF2: {
            auto v = surf2Dread<ushort2>(surf.handle, p.x * sizeof(ushort2), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::HALF4: {
            auto v = surf2Dread<ushort4>(surf.handle, p.x * sizeof(ushort4), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        case LCPixelStorage::FLOAT1: {
            auto v = surf2Dread<float>(surf.handle, p.x * sizeof(float), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            auto v = surf2Dread<float2>(surf.handle, p.x * sizeof(float2), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            auto v = surf2Dread<float4>(surf.handle, p.x * sizeof(float4), p.y, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        default: break;
    }
    return result;
}

template<typename T, typename V>
__device__ inline void lc_surf2d_write(LCSurface surf, lc_uint2 p, V value) noexcept {
    switch (surf.storage) {
        case LCPixelStorage::BYTE1: {
            char v = lc_texel_write_convert<char>(value.x);
            surf2Dwrite(v, surf.handle, p.x * sizeof(char), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::BYTE2: {
            char vx = lc_texel_write_convert<char>(value.x);
            char vy = lc_texel_write_convert<char>(value.y);
            surf2Dwrite(make_char2(vx, vy), surf.handle, p.x * sizeof(char2), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::BYTE4: {
            char vx = lc_texel_write_convert<char>(value.x);
            char vy = lc_texel_write_convert<char>(value.y);
            char vz = lc_texel_write_convert<char>(value.z);
            char vw = lc_texel_write_convert<char>(value.w);
            surf2Dwrite(make_char4(vx, vy, vz, vw), surf.handle, p.x * sizeof(char4), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::SHORT1: {
            short v = lc_texel_write_convert<short>(value.x);
            surf2Dwrite(v, surf.handle, p.x * sizeof(short), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::SHORT2: {
            short vx = lc_texel_write_convert<short>(value.x);
            short vy = lc_texel_write_convert<short>(value.y);
            surf2Dwrite(make_short2(vx, vy), surf.handle, p.x * sizeof(short2), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::SHORT4: {
            short vx = lc_texel_write_convert<short>(value.x);
            short vy = lc_texel_write_convert<short>(value.y);
            short vz = lc_texel_write_convert<short>(value.z);
            short vw = lc_texel_write_convert<short>(value.w);
            surf2Dwrite(make_short4(vx, vy, vz, vw), surf.handle, p.x * sizeof(short4), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::INT1: {
            int v = lc_texel_write_convert<int>(value.x);
            surf2Dwrite(v, surf.handle, p.x * sizeof(int), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::INT2: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            surf2Dwrite(make_int2(vx, vy), surf.handle, p.x * sizeof(int2), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::INT4: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            int vz = lc_texel_write_convert<int>(value.z);
            int vw = lc_texel_write_convert<int>(value.w);
            surf2Dwrite(make_int4(vx, vy, vz, vw), surf.handle, p.x * sizeof(int4), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::HALF1: {
            unsigned short v = lc_texel_write_convert<unsigned short>(value.x);
            surf2Dwrite(v, surf.handle, p.x * sizeof(unsigned short), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::HALF2: {
            unsigned short vx = lc_texel_write_convert<unsigned short>(value.x);
            unsigned short vy = lc_texel_write_convert<unsigned short>(value.y);
            surf2Dwrite(make_ushort2(vx, vy), surf.handle, p.x * sizeof(ushort2), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::HALF4: {
            unsigned short vx = lc_texel_write_convert<unsigned short>(value.x);
            unsigned short vy = lc_texel_write_convert<unsigned short>(value.y);
            unsigned short vz = lc_texel_write_convert<unsigned short>(value.z);
            unsigned short vw = lc_texel_write_convert<unsigned short>(value.w);
            surf2Dwrite(make_ushort4(vx, vy, vz, vw), surf.handle, p.x * sizeof(ushort4), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float v = lc_texel_write_convert<float>(value.x);
            surf2Dwrite(v, surf.handle, p.x * sizeof(float), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            surf2Dwrite(make_float2(vx, vy), surf.handle, p.x * sizeof(float2), p.y, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            float vz = lc_texel_write_convert<float>(value.z);
            float vw = lc_texel_write_convert<float>(value.w);
            surf2Dwrite(make_float4(vx, vy, vz, vw), surf.handle, p.x * sizeof(float4), p.y, cudaBoundaryModeZero);
            break;
        }
        default: break;
    }
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf3d_read(LCSurface surf, lc_uint3 p) noexcept {
    lc_vec4_t<T> result{0, 0, 0, 0};
    switch (surf.storage) {
        case LCPixelStorage::BYTE1: {
            auto v = surf3Dread<char>(surf.handle, p.x * sizeof(char), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::BYTE2: {
            auto v = surf3Dread<char2>(surf.handle, p.x * sizeof(char2), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::BYTE4: {
            auto v = surf3Dread<char4>(surf.handle, p.x * sizeof(char4), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        case LCPixelStorage::SHORT1: {
            auto v = surf3Dread<short>(surf.handle, p.x * sizeof(short), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::SHORT2: {
            auto v = surf3Dread<short2>(surf.handle, p.x * sizeof(short2), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::SHORT4: {
            auto v = surf3Dread<short4>(surf.handle, p.x * sizeof(short4), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        case LCPixelStorage::INT1: {
            auto v = surf3Dread<int>(surf.handle, p.x * sizeof(int), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::INT2: {
            auto v = surf3Dread<int2>(surf.handle, p.x * sizeof(int2), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::INT4: {
            auto v = surf3Dread<int4>(surf.handle, p.x * sizeof(int4), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        case LCPixelStorage::HALF1: {
            auto v = surf3Dread<unsigned short>(surf.handle, p.x * sizeof(unsigned short), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::HALF2: {
            auto v = surf3Dread<ushort2>(surf.handle, p.x * sizeof(ushort2), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::HALF4: {
            auto v = surf3Dread<ushort4>(surf.handle, p.x * sizeof(ushort4), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        case LCPixelStorage::FLOAT1: {
            auto v = surf3Dread<float>(surf.handle, p.x * sizeof(float), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            auto v = surf3Dread<float2>(surf.handle, p.x * sizeof(float2), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            auto v = surf3Dread<float4>(surf.handle, p.x * sizeof(float4), p.y, p.z, cudaBoundaryModeZero);
            result.x = lc_texel_read_convert<T>(v.x);
            result.y = lc_texel_read_convert<T>(v.y);
            result.z = lc_texel_read_convert<T>(v.z);
            result.w = lc_texel_read_convert<T>(v.w);
            break;
        }
        default: break;
    }
    return result;
}

template<typename T, typename V>
__device__ inline void lc_surf3d_write(LCSurface surf, lc_uint3 p, V value) noexcept {
    switch (surf.storage) {
        case LCPixelStorage::BYTE1: {
            char v = lc_texel_write_convert<char>(value.x);
            surf3Dwrite(v, surf.handle, p.x * sizeof(char), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::BYTE2: {
            char vx = lc_texel_write_convert<char>(value.x);
            char vy = lc_texel_write_convert<char>(value.y);
            surf3Dwrite(make_char2(vx, vy), surf.handle, p.x * sizeof(char2), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::BYTE4: {
            char vx = lc_texel_write_convert<char>(value.x);
            char vy = lc_texel_write_convert<char>(value.y);
            char vz = lc_texel_write_convert<char>(value.z);
            char vw = lc_texel_write_convert<char>(value.w);
            surf3Dwrite(make_char4(vx, vy, vz, vw), surf.handle, p.x * sizeof(char4), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::SHORT1: {
            short v = lc_texel_write_convert<short>(value.x);
            surf3Dwrite(v, surf.handle, p.x * sizeof(short), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::SHORT2: {
            short vx = lc_texel_write_convert<short>(value.x);
            short vy = lc_texel_write_convert<short>(value.y);
            surf3Dwrite(make_short2(vx, vy), surf.handle, p.x * sizeof(short2), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::SHORT4: {
            short vx = lc_texel_write_convert<short>(value.x);
            short vy = lc_texel_write_convert<short>(value.y);
            short vz = lc_texel_write_convert<short>(value.z);
            short vw = lc_texel_write_convert<short>(value.w);
            surf3Dwrite(make_short4(vx, vy, vz, vw), surf.handle, p.x * sizeof(short4), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::INT1: {
            int v = lc_texel_write_convert<int>(value.x);
            surf3Dwrite(v, surf.handle, p.x * sizeof(int), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::INT2: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            surf3Dwrite(make_int2(vx, vy), surf.handle, p.x * sizeof(int2), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::INT4: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            int vz = lc_texel_write_convert<int>(value.z);
            int vw = lc_texel_write_convert<int>(value.w);
            surf3Dwrite(make_int4(vx, vy, vz, vw), surf.handle, p.x * sizeof(int4), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::HALF1: {
            unsigned short v = lc_texel_write_convert<unsigned short>(value.x);
            surf3Dwrite(v, surf.handle, p.x * sizeof(unsigned short), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::HALF2: {
            unsigned short vx = lc_texel_write_convert<unsigned short>(value.x);
            unsigned short vy = lc_texel_write_convert<unsigned short>(value.y);
            surf3Dwrite(make_ushort2(vx, vy), surf.handle, p.x * sizeof(ushort2), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::HALF4: {
            unsigned short vx = lc_texel_write_convert<unsigned short>(value.x);
            unsigned short vy = lc_texel_write_convert<unsigned short>(value.y);
            unsigned short vz = lc_texel_write_convert<unsigned short>(value.z);
            unsigned short vw = lc_texel_write_convert<unsigned short>(value.w);
            surf3Dwrite(make_ushort4(vx, vy, vz, vw), surf.handle, p.x * sizeof(ushort4), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float v = lc_texel_write_convert<float>(value.x);
            surf3Dwrite(v, surf.handle, p.x * sizeof(float), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            surf3Dwrite(make_float2(vx, vy), surf.handle, p.x * sizeof(float2), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            float vz = lc_texel_write_convert<float>(value.z);
            float vw = lc_texel_write_convert<float>(value.w);
            surf3Dwrite(make_float4(vx, vy, vz, vw), surf.handle, p.x * sizeof(float4), p.y, p.z, cudaBoundaryModeZero);
            break;
        }
        default: break;
    }
}

struct alignas(16) LCBindlessItem {
    const void *buffer;
    const void *origin;
    const cudaTextureObject_t tex2d;
    const uint2 size2d;
    const cudaTextureObject_t tex3d;
    const ushort3 size3d;
};

static_assert(sizeof(LCBindlessItem) == 48);

template<typename T>
[[nodiscard]] inline __device__ auto lc_bindless_buffer_read(const LCBindlessItem *array, lc_uint index, lc_uint i) noexcept {
    auto buffer = reinterpret_cast<const T *>(array[index].buffer);
    return buffer[i];
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d(const LCBindlessItem *array, lc_uint index, lc_float2 p) noexcept {
    auto t = array[index].tex2d;
    auto v = tex2D<float4>(t, p.x, p.y);
    return lc_make_float4(v.x, v.y, v.z, v.w);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d(const LCBindlessItem *array, lc_uint index, lc_float3 p) noexcept {
    auto t = array[index].tex3d;
    auto v = tex3D<float4>(t, p.x, p.y, p.z);
    return lc_make_float4(v.x, v.y, v.z, v.w);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d_level(const LCBindlessItem *array, lc_uint index, lc_float2 p, float level) noexcept {
    auto t = array[index].tex2d;
    auto v = tex2DLod<float4>(t, p.x, p.y, level);
    return lc_make_float4(v.x, v.y, v.z, v.w);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d_level(const LCBindlessItem *array, lc_uint index, lc_float3 p, float level) noexcept {
    auto t = array[index].tex3d;
    auto v = tex3DLod<float4>(t, p.x, p.y, p.z, level);
    return lc_make_float4(v.x, v.y, v.z, v.w);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d_grad(const LCBindlessItem *array, lc_uint index, lc_float2 p, lc_float2 dx, lc_float2 dy) noexcept {
    auto t = array[index].tex2d;
    auto v = tex2DGrad<float4>(t, p.x, p.y, make_float2(dx.x, dx.y), make_float2(dy.x, dy.y));
    return lc_make_float4(v.x, v.y, v.z, v.w);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d_grad(const LCBindlessItem *array, lc_uint index, lc_float3 p, lc_float3 dx, lc_float3 dy) noexcept {
    auto t = array[index].tex3d;
    auto v = tex3DGrad<float4>(t, p.x, p.y, p.z, make_float4(dx.x, dx.y, dx.z, 1.0f), make_float4(dy.x, dy.y, dy.z, 1.0f));
    return lc_make_float4(v.x, v.y, v.z, v.w);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size2d(const LCBindlessItem *array, lc_uint index) noexcept {
    auto s = array[index].size2d;
    return lc_make_uint2(s.x, s.y);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size3d(const LCBindlessItem *array, lc_uint index) noexcept {
    auto s = array[index].size3d;
    return lc_make_uint3(s.x, s.y, s.z);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size2d_level(const LCBindlessItem *array, lc_uint index, lc_uint level) noexcept {
    auto s = array[index].size2d;
    return lc_max(lc_make_uint2(s.x, s.y) >> level, lc_make_uint2(1u));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size3d_level(const LCBindlessItem *array, lc_uint index, lc_uint level) noexcept {
    auto s = array[index].size3d;
    return lc_max(lc_make_uint3(s.x, s.y, s.z) >> level, lc_make_uint3(1u));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read2d(const LCBindlessItem *array, lc_uint index, lc_uint2 p) noexcept {
    auto s = lc_bindless_texture_size2d(array, index);
    auto pp = (lc_make_float2(p) + lc_make_float2(0.5f)) / lc_make_float2(s);
    return lc_bindless_texture_sample2d(array, index, pp);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read3d(const LCBindlessItem *array, lc_uint index, lc_uint3 p) noexcept {
    auto s = lc_bindless_texture_size3d(array, index);
    auto pp = (lc_make_float3(p) + lc_make_float3(0.5f)) / lc_make_float3(s);
    return lc_bindless_texture_sample3d(array, index, pp);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read2d_level(const LCBindlessItem *array, lc_uint index, lc_uint2 p, lc_uint level) noexcept {
    auto s = lc_bindless_texture_size2d_level(array, index, level);
    auto pp = (lc_make_float2(p) + lc_make_float2(0.5f)) / lc_make_float2(s);
    return lc_bindless_texture_sample2d_level(array, index, pp, static_cast<float>(level));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read3d_level(const LCBindlessItem *array, lc_uint index, lc_uint3 p, lc_uint level) noexcept {
    auto s = lc_bindless_texture_size3d_level(array, index, level);
    auto pp = (lc_make_float3(p) + lc_make_float3(0.5f)) / lc_make_float3(s);
    return lc_bindless_texture_sample3d_level(array, index, pp, static_cast<float>(level));
}
