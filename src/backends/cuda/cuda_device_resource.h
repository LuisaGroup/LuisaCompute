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
    [[nodiscard]] static constexpr auto value() noexcept { return false; };
};

template<typename A>
struct lc_is_same<A, A> {
    [[nodiscard]] static constexpr auto value() noexcept { return true; };
};

template<typename...>
struct lc_always_false {
    [[nodiscard]] static constexpr auto value() noexcept { return false; };
};

using lc_half = unsigned short;

template<typename T, size_t N>
class lc_array {

private:
    T _data[N];

public:
    template<typename... Elem>
    __device__ constexpr lc_array(Elem... elem) noexcept: _data{elem...} {}
    __device__ constexpr lc_array(lc_array &&) noexcept = default;
    __device__ constexpr lc_array(const lc_array &) noexcept = default;
    __device__ constexpr lc_array &operator=(lc_array &&) noexcept = default;
    __device__ constexpr lc_array &operator=(const lc_array &) noexcept = default;
    [[nodiscard]] __device__ T &operator[](size_t i) noexcept { return _data[i]; }
    [[nodiscard]] __device__ T operator[](size_t i) const noexcept { return _data[i]; }
};

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_float(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<unsigned char>(x) * (1.0f / 255.0f);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<unsigned short>(x) * (1.0f / 65535.0f);
    } else if constexpr (lc_is_same<P, lc_half>::value()) {
        return lc_half_to_float(x);
    } else if constexpr (lc_is_same<P, lc_float>::value()) {
        return x;
    }
    return 0.0f;
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_int(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<lc_int>(x);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<lc_int>(x);
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return x;
    }
    return 0;
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_uint(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<lc_uint>(static_cast<unsigned char>(x));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<lc_uint>(static_cast<unsigned short>(x));
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return static_cast<lc_uint>(x);
    }
    return 0u;
}

template<typename T, typename P>
[[nodiscard]] __device__ inline auto lc_texel_read_convert(P p) noexcept {
    if constexpr (lc_is_same<T, lc_float>::value()) {
        return lc_texel_to_float<P>(p);
    } else if constexpr (lc_is_same<T, lc_int>::value()) {
        return lc_texel_to_int<P>(p);
    } else if constexpr (lc_is_same<T, lc_uint>::value()) {
        return lc_texel_to_uint<P>(p);
    } else {
        static_assert(lc_always_false<T, P>::value());
    }
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_float_to_texel(lc_float x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(static_cast<unsigned char>(lc_round(lc_saturate(x) * 255.0f)));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(static_cast<unsigned short>(lc_round(lc_saturate(x) * 65535.0f)));
    } else if constexpr (lc_is_same<P, lc_half>::value()) {
        return lc_float_to_half(x);
    } else if constexpr (lc_is_same<P, lc_float>::value()) {
        return x;
    }
    return P{};
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_int_to_texel(int x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(x);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(x);
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return x;
    }
    return P{};
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_uint_to_texel(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(static_cast<unsigned char>(x));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(static_cast<unsigned short>(x));
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return static_cast<lc_int>(x);
    }
    return P{};
}

template<typename P, typename T>
[[nodiscard]] __device__ inline auto lc_texel_write_convert(T t) noexcept {
    if constexpr (lc_is_same<T, lc_float>::value()) {
        return lc_float_to_texel<P>(t);
    } else if constexpr (lc_is_same<T, lc_int>::value()) {
        return lc_int_to_texel<P>(t);
    } else if constexpr (lc_is_same<T, lc_uint>::value()) {
        return lc_uint_to_texel<P>(t);
    } else {
        static_assert(lc_always_false<T, P>::value());
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

struct alignas(16) LCRay {
    lc_array<float, 3> m0;// origin
    float m1;             // t_min
    lc_array<float, 3> m2;// direction
    float m3;             // t_max
};

struct alignas(16) LCHit {
    lc_uint m0;  // instance index
    lc_uint m1;  // primitive index
    lc_float2 m2;// barycentric coordinates
    LCHit() noexcept : m0{~0u}, m1{~0u}, m2{0.0f, 0.0f} {}
    LCHit(lc_uint inst, lc_uint prim, lc_float2 bary) noexcept
        : m0{inst}, m1{prim}, m2{bary} {}
};

#if LC_RAYTRACING_KERNEL

using LCAccel = unsigned long long;

template<lc_uint i>
inline void lc_set_payload(lc_uint x) noexcept {
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"(i), "r"(x) : );
}

[[nodiscard]] inline auto lc_get_primitive_index() noexcept {
    lc_uint u0;
    asm( "call (%0), _optix_read_primitive_idx, ();" : "=r"(u0) : );
    return u0;
}

[[nodiscard]] inline auto lc_get_instance_index() noexcept {
    lc_uint u0;
    asm( "call (%0), _optix_read_instance_idx, ();" : "=r"(u0) : );
    return u0;
}

[[nodiscard]] inline auto lc_get_bary_coords() noexcept {
    float f0, f1;
    asm( "call (%0, %1), _optix_get_triangle_barycentrics, ();" : "=f"(f0), "=f"(f1) : );
    return lc_make_float2(f0, f1);
}

[[nodiscard]] inline auto lc_is_tracing_any_hit() noexcept {
    lc_uint r0;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"(r0) : "r"(0) : );
    return r0 == 0u;
}

extern "C" __global__ void __closesthit__ch() {
    if (lc_is_tracing_any_hit()) {
        lc_set_payload<0u>(1u);
    } else {
        auto inst = lc_get_instance_index();
        auto prim = lc_get_primitive_index();
        auto bary = lc_get_bary_coords();
        lc_set_payload<0u>(inst);
        lc_set_payload<1u>(prim);
        lc_set_payload<2u>(__float_as_uint(bary.x));
        lc_set_payload<3u>(__float_as_uint(bary.y));
    }
}

template<lc_uint reg_count, lc_uint flags>
[[nodiscard]] inline auto lc_trace_impl(
    LCAccel accel, LCRay ray,
    lc_uint &r0, lc_uint &r1, lc_uint &r2, lc_uint &r3) noexcept {
    auto ox = ray.m0[0];
    auto oy = ray.m0[1];
    auto oz = ray.m0[2];
    auto dx = ray.m2[0];
    auto dy = ray.m2[1];
    auto dz = ray.m2[2];
    auto t_min = ray.m1;
    auto t_max = ray.m3;
    unsigned int p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25,
        p26, p27, p28, p29, p30, p31;
    asm volatile(
        "call"
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%"
        "29,%30,%31),"
        "_optix_trace_typed_32,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%"
        "59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(p4), "=r"(p5), "=r"(p6), "=r"(p7), "=r"(p8),
          "=r"(p9), "=r"(p10), "=r"(p11), "=r"(p12), "=r"(p13), "=r"(p14), "=r"(p15), "=r"(p16),
          "=r"(p17), "=r"(p18), "=r"(p19), "=r"(p20), "=r"(p21), "=r"(p22), "=r"(p23), "=r"(p24),
          "=r"(p25), "=r"(p26), "=r"(p27), "=r"(p28), "=r"(p29), "=r"(p30), "=r"(p31)
        : "r"(0u), "l"(accel), "f"(ox), "f"(oy), "f"(oz), "f"(dx), "f"(dy), "f"(dz), "f"(t_min),
          "f"(t_max), "f"(0.0f), "r"(0xffu), "r"(flags), "r"(0u), "r"(1u),
          "r"(0u), "r"(reg_count), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(p4), "r"(p5), "r"(p6),
          "r"(p7), "r"(p8), "r"(p9), "r"(p10), "r"(p11), "r"(p12), "r"(p13), "r"(p14), "r"(p15),
          "r"(p16), "r"(p17), "r"(p18), "r"(p19), "r"(p20), "r"(p21), "r"(p22), "r"(p23), "r"(p24),
          "r"(p25), "r"(p26), "r"(p27), "r"(p28), "r"(p29), "r"(p30), "r"(p31)
        : );
    (void)p4, (void)p5, (void)p6, (void)p7, (void)p8, (void)p9, (void)p10, (void)p11, (void)p12, (void)p13, (void)p14,
        (void)p15, (void)p16, (void)p17, (void)p18, (void)p19, (void)p20, (void)p21, (void)p22, (void)p23, (void)p24,
        (void)p25, (void)p26, (void)p27, (void)p28, (void)p29, (void)p30, (void)p31;
}

[[nodiscard]] inline auto lc_trace_closest(LCAccel accel, LCRay ray) noexcept {
    constexpr auto flags = 1u;// disable any hit
    auto r0 = ~0u;// also indicates trace_closest
    auto r1 = 0u;
    auto r2 = 0u;
    auto r3 = 0u;
    lc_trace_impl<4u, flags>(accel, ray, r0, r1, r2, r3);
    return LCHit{r0, r1, lc_make_float2(__uint_as_float(r2), __uint_as_float(r3))};
}

[[nodiscard]] inline auto lc_trace_any(LCAccel accel, LCRay ray) noexcept {
    constexpr auto flags = 1u | 4u;// disable any hit and terminate on first hit
    auto r0 = 0u;// also indicates trace_any
    auto r1 = 0u;
    auto r2 = 0u;
    auto r3 = 0u;
    lc_trace_impl<1u, flags>(accel, ray, r0, r1, r2, r3);
    return static_cast<bool>(r0);
}

[[nodiscard]] inline auto lc_rtx_dispatch_id() noexcept {
    lc_uint u0, u1, u2;
    asm( "call (%0), _optix_get_launch_index_x, ();" : "=r"(u0) : );
    asm( "call (%0), _optix_get_launch_index_y, ();" : "=r"(u1) : );
    asm( "call (%0), _optix_get_launch_index_z, ();" : "=r"(u2) : );
    return lc_make_uint3(u0, u1, u2);
}

[[nodiscard]] inline auto lc_rtx_dispatch_size() noexcept {
    lc_uint u0, u1, u2;
    asm( "call (%0), _optix_get_launch_dimension_x, ();" : "=r"(u0) : );
    asm( "call (%0), _optix_get_launch_dimension_y, ();" : "=r"(u1) : );
    asm( "call (%0), _optix_get_launch_dimension_z, ();" : "=r"(u2) : );
    return lc_make_uint3(u0, u1, u2);
}

#endif
