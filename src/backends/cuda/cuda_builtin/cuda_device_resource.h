#pragma once

[[nodiscard]] __device__ constexpr auto lc_infinity_float() noexcept { return __int_as_float(0x7f800000u); }
[[nodiscard]] __device__ constexpr auto lc_infinity_double() noexcept { return __longlong_as_double(0x7ff0000000000000ull); }

#if LC_NVRTC_VERSION < 110200
#define LC_CONSTANT const
#else
#define LC_CONSTANT constexpr
#endif

#if LC_NVRTC_VERSION < 110200
inline __device__ void lc_assume(bool) noexcept {}
#else
#define lc_assume(...) __builtin_assume(__VA_ARGS__)
#endif

template<typename T = void>
[[noreturn]] inline __device__ T lc_unreachable() noexcept {
#if LC_NVRTC_VERSION < 110300
    asm("trap;");
#else
    __builtin_unreachable();
#endif
}

#ifdef LUISA_DEBUG
#define lc_assert(x)                                    \
    do {                                                \
        if (!(x)) {                                     \
            printf("Assertion failed: %s [%s:%d:%s]\n", \
                   #x,                                  \
                   __FILE__,                            \
                   static_cast<int>(__LINE__),          \
                   __FUNCTION__);                       \
            asm("trap;");                               \
        }                                               \
    } while (false)
#define lc_check_in_bounds(size, max_size)                               \
    do {                                                                 \
        if (!((size) < (max_size))) {                                    \
            printf("Out of bounds: !(%s: %llu < %s: %llu) [%s:%d:%s]\n", \
                   #size, static_cast<size_t>(size),                     \
                   #max_size, static_cast<size_t>(max_size),             \
                   __FILE__, static_cast<int>(__LINE__),                 \
                   __FUNCTION__);                                        \
        }                                                                \
    } while (false)
#else
inline __device__ void lc_assert(bool) noexcept {}
#endif

struct lc_half {
    unsigned short bits;
};

struct alignas(4) lc_half2 {
    lc_half x, y;
};

struct alignas(8) lc_half4 {
    lc_half x, y, z, w;
};

[[nodiscard]] __device__ inline auto lc_half_to_float(lc_half x) noexcept {
    lc_float val;
    asm("{  cvt.f32.f16 %0, %1;}\n"
        : "=f"(val)
        : "h"(x.bits));
    return val;
}

[[nodiscard]] __device__ inline auto lc_float_to_half(lc_float x) noexcept {
    lc_half val;
    asm("{  cvt.rn.f16.f32 %0, %1;}\n"
        : "=h"(val.bits)
        : "f"(x));
    return val;
}

template<size_t alignment, size_t size>
struct alignas(alignment) lc_aligned_storage {
    unsigned char data[size];
};

struct alignas(16) LCIndirectHeader {
    lc_uint size;
};

struct alignas(16) LCIndirectDispatch {
    lc_uint3 block_size;
    lc_uint4 dispatch_size_and_kernel_id;
};

struct alignas(16) LCIndirectBuffer {
    void *__restrict__ data;
    size_t capacity;

    [[nodiscard]] auto header() const noexcept {
        return reinterpret_cast<LCIndirectHeader *>(data);
    }

    [[nodiscard]] auto dispatches() const noexcept {
        return reinterpret_cast<LCIndirectDispatch *>(reinterpret_cast<lc_ulong>(data) + sizeof(LCIndirectHeader));
    }
};

void lc_indirect_buffer_clear(const LCIndirectBuffer buffer) noexcept {
    buffer.header()->size = 0u;
}

void lc_indirect_buffer_emplace(LCIndirectBuffer buffer, lc_uint3 block_size, lc_uint3 dispatch_size, lc_uint kernel_id) noexcept {
    auto index = atomicAdd(&(buffer.header()->size), 1u);
#ifdef LUISA_DEBUG
    lc_check_in_bounds(index, buffer.capacity);
#endif
    buffer.dispatches()[index] = LCIndirectDispatch{
        block_size, lc_make_uint4(dispatch_size, kernel_id)};
}

template<typename T>
struct LCBuffer {
    T *__restrict__ ptr;
    size_t size_bytes;
};

template<typename T>
struct LCBuffer<const T> {
    const T *__restrict__ ptr;
    size_t size_bytes;
    LCBuffer(LCBuffer<T> buffer) noexcept
        : ptr{buffer.ptr}, size_bytes{buffer.size_bytes} {}
    LCBuffer() noexcept = default;
};

template<typename T>
[[nodiscard]] __device__ inline auto lc_buffer_size(LCBuffer<T> buffer) noexcept {
    return buffer.size_bytes / sizeof(T);
}

template<typename T, typename Index>
[[nodiscard]] __device__ inline auto lc_buffer_read(LCBuffer<T> buffer, Index index) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(index, lc_buffer_size(buffer));
#endif
    return buffer.ptr[index];
}

template<typename T, typename Index>
__device__ inline void lc_buffer_write(LCBuffer<T> buffer, Index index, T value) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(index, lc_buffer_size(buffer));
#endif
    buffer.ptr[index] = value;
}

enum struct LCPixelStorage {

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
    unsigned long long storage;
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
[[nodiscard]] __device__ inline auto lc_int_to_texel(lc_int x) noexcept {
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
[[nodiscard]] __device__ inline auto lc_uint_to_texel(lc_uint x) noexcept {
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
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int x;
            asm("suld.b.2d.b8.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            break;
        }
        case LCPixelStorage::BYTE2: {
            int x, y;
            asm("suld.b.2d.v2.b8.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            break;
        }
        case LCPixelStorage::BYTE4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b8.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            result.z = lc_texel_read_convert<T, char>(z);
            result.w = lc_texel_read_convert<T, char>(w);
            break;
        }
        case LCPixelStorage::SHORT1: {
            int x;
            asm("suld.b.2d.b16.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            break;
        }
        case LCPixelStorage::SHORT2: {
            int x, y;
            asm("suld.b.2d.v2.b16.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            break;
        }
        case LCPixelStorage::SHORT4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            result.z = lc_texel_read_convert<T, short>(z);
            result.w = lc_texel_read_convert<T, short>(w);
            break;
        }
        case LCPixelStorage::INT1: {
            int x;
            asm("suld.b.2d.b32.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            break;
        }
        case LCPixelStorage::INT2: {
            int x, y;
            asm("suld.b.2d.v2.b32.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            break;
        }
        case LCPixelStorage::INT4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            result.z = lc_texel_read_convert<T, int>(z);
            result.w = lc_texel_read_convert<T, int>(w);
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint x;
            asm("suld.b.2d.b16.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint x, y;
            asm("suld.b.2d.v2.b16.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            result.y = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(y)});
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint x, y, z, w;
            asm("suld.b.2d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            result.y = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(y)});
            result.z = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(z)});
            result.w = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(w)});
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float x;
            asm("suld.b.2d.b32.zero %0, [%1, {%2, %3}];"
                : "=f"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float x, y;
            asm("suld.b.2d.v2.b32.zero {%0, %1}, [%2, {%3, %4}];"
                : "=f"(x), "=f"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float x, y, z, w;
            asm("suld.b.2d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            result.z = lc_texel_read_convert<T, float>(z);
            result.w = lc_texel_read_convert<T, float>(w);
            break;
        }
        default: __builtin_unreachable();
    }
    return result;
}

template<typename T, typename V>
__device__ inline void lc_surf2d_write(LCSurface surf, lc_uint2 p, V value) noexcept {
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int v = lc_texel_write_convert<char>(value.x);
            asm volatile("sust.b.2d.b8.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE2: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            asm volatile("sust.b.2d.v2.b8.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE4: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            int vz = lc_texel_write_convert<char>(value.z);
            int vw = lc_texel_write_convert<char>(value.w);
            asm volatile("sust.b.2d.v4.b8.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT1: {
            int v = lc_texel_write_convert<short>(value.x);
            asm volatile("sust.b.2d.b16.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT2: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            asm volatile("sust.b.2d.v2.b16.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT4: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            int vz = lc_texel_write_convert<short>(value.z);
            int vw = lc_texel_write_convert<short>(value.w);
            asm volatile("sust.b.2d.v4.b16.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT1: {
            int v = lc_texel_write_convert<int>(value.x);
            asm volatile("sust.b.2d.b32.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT2: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            asm volatile("sust.b.2d.v2.b32.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT4: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            int vz = lc_texel_write_convert<int>(value.z);
            int vw = lc_texel_write_convert<int>(value.w);
            asm volatile("sust.b.2d.v4.b32.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint v = lc_texel_write_convert<lc_half>(value.x).bits;
            asm volatile("sust.b.2d.b16.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            asm volatile("sust.b.2d.v2.b16.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            lc_uint vz = lc_texel_write_convert<lc_half>(value.z).bits;
            lc_uint vw = lc_texel_write_convert<lc_half>(value.w).bits;
            asm volatile("sust.b.2d.v4.b16.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float v = lc_texel_write_convert<float>(value.x);
            asm volatile("sust.b.2d.b32.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float))), "r"(p.y), "f"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            asm volatile("sust.b.2d.v2.b32.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float2))), "r"(p.y), "f"(vx), "f"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            float vz = lc_texel_write_convert<float>(value.z);
            float vw = lc_texel_write_convert<float>(value.w);
            asm volatile("sust.b.2d.v4.b32.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float4))), "r"(p.y), "f"(vx), "f"(vy), "f"(vz), "f"(vw)
                         : "memory");
            break;
        }
        default: __builtin_unreachable();
    }
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf3d_read(LCSurface surf, lc_uint3 p) noexcept {
    lc_vec4_t<T> result{0, 0, 0, 0};
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int x;
            asm("suld.b.3d.b8.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            break;
        }
        case LCPixelStorage::BYTE2: {
            int x, y;
            asm("suld.b.3d.v2.b8.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert < T, char(x);
            result.y = lc_texel_read_convert < T, char(y);
            break;
        }
        case LCPixelStorage::BYTE4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b8.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            result.z = lc_texel_read_convert<T, char>(z);
            result.w = lc_texel_read_convert<T, char>(w);
            break;
        }
        case LCPixelStorage::SHORT1: {
            int x;
            asm("suld.b.3d.b16.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            break;
        }
        case LCPixelStorage::SHORT2: {
            int x, y;
            asm("suld.b.3d.v2.b16.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            break;
        }
        case LCPixelStorage::SHORT4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            result.z = lc_texel_read_convert<T, short>(z);
            result.w = lc_texel_read_convert<T, short>(w);
            break;
        }
        case LCPixelStorage::INT1: {
            int x;
            asm("suld.b.3d.b32.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            break;
        }
        case LCPixelStorage::INT2: {
            int x, y;
            asm("suld.b.3d.v2.b32.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            break;
        }
        case LCPixelStorage::INT4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            result.z = lc_texel_read_convert<T, int>(z);
            result.w = lc_texel_read_convert<T, int>(w);
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint x;
            asm("suld.b.3d.b16.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint x, y;
            asm("suld.b.3d.v2.b16.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            result.y = lc_texel_read_convert<T, lc_half>(y);
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint x, y, z, w;
            asm("suld.b.3d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            result.y = lc_texel_read_convert<T, lc_half>(y);
            result.z = lc_texel_read_convert<T, lc_half>(z);
            result.w = lc_texel_read_convert<T, lc_half>(w);
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float x;
            asm("suld.b.3d.b32.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=f"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float x, y;
            asm("suld.b.3d.v2.b32.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=f"(x), "=f"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float x, y, z, w;
            asm("suld.b.3d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            result.z = lc_texel_read_convert<T, float>(z);
            result.w = lc_texel_read_convert<T, float>(w);
            break;
        }
        default: __builtin_unreachable();
    }
    return result;
}

template<typename T, typename V>
__device__ inline void lc_surf3d_write(LCSurface surf, lc_uint3 p, V value) noexcept {
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int v = lc_texel_write_convert<char>(value.x);
            asm volatile("sust.b.3d.b8.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE2: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            asm volatile("sust.b.3d.v2.b8.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE4: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            int vz = lc_texel_write_convert<char>(value.z);
            int vw = lc_texel_write_convert<char>(value.w);
            asm volatile("sust.b.3d.v4.b8.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT1: {
            int v = lc_texel_write_convert<short>(value.x);
            asm volatile("sust.b.3d.b16.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT2: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            asm volatile("sust.b.3d.v2.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT4: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            int vz = lc_texel_write_convert<short>(value.z);
            int vw = lc_texel_write_convert<short>(value.w);
            asm volatile("sust.b.3d.v4.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT1: {
            int v = lc_texel_write_convert<int>(value.x);
            asm volatile("sust.b.3d.b32.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT2: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            asm volatile("sust.b.3d.v2.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT4: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            int vz = lc_texel_write_convert<int>(value.z);
            int vw = lc_texel_write_convert<int>(value.w);
            asm volatile("sust.b.3d.v4.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint v = lc_texel_write_convert<lc_half>(value.x).bits;
            asm volatile("sust.b.3d.b16.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            asm volatile("sust.b.3d.v2.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            lc_uint vz = lc_texel_write_convert<lc_half>(value.z).bits;
            lc_uint vw = lc_texel_write_convert<lc_half>(value.w).bits;
            asm volatile("sust.b.3d.v4.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float v = lc_texel_write_convert<float>(value.x);
            asm volatile("sust.b.3d.b32.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float))), "r"(p.y), "r"(p.z), "r"(0), "f"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            asm volatile("sust.b.3d.v2.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float2))), "r"(p.y), "r"(p.z), "r"(0), "f"(vx), "f"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            float vz = lc_texel_write_convert<float>(value.z);
            float vw = lc_texel_write_convert<float>(value.w);
            asm volatile("sust.b.3d.v4.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float4))), "r"(p.y), "r"(p.z), "r"(0), "f"(vx), "f"(vy), "f"(vz), "f"(vw)
                         : "memory");
            break;
        }
        default: __builtin_unreachable();
    }
}

template<typename T>
struct LCTexture2D {
    LCSurface surface;
};

template<typename T>
struct LCTexture3D {
    LCSurface surface;
};

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_size(LCTexture2D<T> tex) noexcept {
    lc_uint2 size;
    asm("suq.width.b32 %0, [%1];"
        : "=r"(size.x)
        : "l"(tex.surface.handle));
    asm("suq.height.b32 %0, [%1];"
        : "=r"(size.y)
        : "l"(tex.surface.handle));
    return size;
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_size(LCTexture3D<T> tex) noexcept {
    lc_uint3 size;
    asm("suq.width.b32 %0, [%1];"
        : "=r"(size.x)
        : "l"(tex.surface.handle));
    asm("suq.height.b32 %0, [%1];"
        : "=r"(size.y)
        : "l"(tex.surface.handle));
    asm("suq.depth.b32 %0, [%1];"
        : "=r"(size.z)
        : "l"(tex.surface.handle));
    return size;
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture2D<T> tex, lc_uint2 p) noexcept {
    return lc_surf2d_read<T>(tex.surface, p);
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture3D<T> tex, lc_uint3 p) noexcept {
    return lc_surf3d_read<T>(tex.surface, p);
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture2D<T> tex, lc_uint2 p, V value) noexcept {
    lc_surf2d_write<T>(tex.surface, p, value);
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture3D<T> tex, lc_uint3 p, V value) noexcept {
    lc_surf3d_write<T>(tex.surface, p, value);
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture2D<T> tex, lc_int2 p) noexcept {
    return lc_texture_read(tex, lc_make_uint2(p));
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture3D<T> tex, lc_int3 p) noexcept {
    return lc_texture_read(tex, lc_make_uint3(p));
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture2D<T> tex, lc_int2 p, V value) noexcept {
    lc_texture_write(tex, lc_make_uint2(p), value);
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture3D<T> tex, lc_int3 p, V value) noexcept {
    lc_texture_write(tex, lc_make_uint3(p), value);
}

struct alignas(16) LCBindlessSlot {
    const void *__restrict__ buffer;
    size_t buffer_size;
    cudaTextureObject_t tex2d;
    cudaTextureObject_t tex3d;
};

struct alignas(16) LCBindlessArray {
    const LCBindlessSlot *__restrict__ slots;
};

template<typename T = unsigned char>
[[nodiscard]] inline __device__ auto lc_bindless_buffer_size(LCBindlessArray array, lc_uint index) noexcept {
    lc_assume(__isGlobal(array.slots));
    return array.slots[index].buffer_size / sizeof(T);
}

[[nodiscard]] inline __device__ auto lc_bindless_buffer_size(LCBindlessArray array, lc_uint index, lc_uint stride) noexcept {
    lc_assume(__isGlobal(array.slots));
    return array.slots[index].buffer_size / stride;
}

template<typename T>
[[nodiscard]] inline __device__ auto lc_bindless_buffer_read(LCBindlessArray array, lc_uint index, lc_uint i) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const T *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(i, lc_bindless_buffer_size<T>(array, index));
#endif
    return buffer[i];
}

[[nodiscard]] inline __device__ auto lc_bindless_buffer_type(LCBindlessArray array, lc_uint index) noexcept {
    return 0ull;// TODO
}

template<typename T>
[[nodiscard]] inline __device__ auto lc_bindless_byte_address_buffer_read(LCBindlessArray array, lc_uint index, lc_uint offset) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const char *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(offset + sizeof(T), lc_bindless_buffer_size<char>(array, index));
#endif
    return *reinterpret_cast<const T *>(buffer + offset);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d(LCBindlessArray array, lc_uint index, lc_float2 p) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d(LCBindlessArray array, lc_uint index, lc_float3 p) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d_level(LCBindlessArray array, lc_uint index, lc_float2 p, float level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.level.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d_level(LCBindlessArray array, lc_uint index, lc_float3 p, float level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f), "f"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d_grad(LCBindlessArray array, lc_uint index, lc_float2 p, lc_float2 dx, lc_float2 dy) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.grad.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, {%9, %10};"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(dx.x), "f"(dx.y), "f"(dy.x), "f"(dy.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d_grad(LCBindlessArray array, lc_uint index, lc_float3 p, lc_float3 dx, lc_float3 dy) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.grad.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], {%9, %10, %11, %12}, {%13, %14, %15, 16};"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f),
          "f"(dx.x), "f"(dx.y), "f"(dx.z), "f"(0.f),
          "f"(dy.x), "f"(dy.y), "f"(dy.z), "f"(0.f));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size2d(LCBindlessArray array, lc_uint index) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto s = lc_make_uint2();
    asm("txq.width.b32 %0, [%1];"
        : "=r"(s.x)
        : "l"(t));
    asm("txq.height.b32 %0, [%1];"
        : "=r"(s.y)
        : "l"(t));
    return s;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size3d(LCBindlessArray array, lc_uint index) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto s = lc_make_uint3();
    asm("txq.width.b32 %0, [%1];"
        : "=r"(s.x)
        : "l"(t));
    asm("txq.height.b32 %0, [%1];"
        : "=r"(s.y)
        : "l"(t));
    asm("txq.depth.b32 %0, [%1];"
        : "=r"(s.z)
        : "l"(t));
    return s;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size2d_level(LCBindlessArray array, lc_uint index, lc_uint level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto s = lc_bindless_texture_size2d(array, index);
    return lc_max(s >> level, lc_make_uint2(1u));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size3d_level(LCBindlessArray array, lc_uint index, lc_uint level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto s = lc_bindless_texture_size3d(array, index);
    return lc_max(s >> level, lc_make_uint3(1u));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read2d(LCBindlessArray array, lc_uint index, lc_uint2 p) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read3d(LCBindlessArray array, lc_uint index, lc_uint3 p) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(p.z), "r"(0u));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read2d_level(LCBindlessArray array, lc_uint index, lc_uint2 p, lc_uint level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.level.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read3d_level(LCBindlessArray array, lc_uint index, lc_uint3 p, lc_uint level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.level.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(p.z), "r"(0u), "r"(level));
    return v;
}

struct alignas(16) LCRay {
    lc_array<float, 3> m0;// origin
    float m1;             // t_min
    lc_array<float, 3> m2;// direction
    float m3;             // t_max
};

struct alignas(8) LCTriangleHit {
    lc_uint m0;  // instance index
    lc_uint m1;  // primitive index
    lc_float2 m2;// barycentric coordinates
    lc_float m3; // t_hit
};

struct LCProceduralHit {
    lc_uint m0;// instance index
    lc_uint m1;// primitive index
};

enum struct LCHitType {
    MISS = 0,
    TRIANGLE = 1,
    PROCEDURAL = 2,
};

struct LCCommittedHit {
    lc_uint m0;  // instance index
    lc_uint m1;  // primitive index
    lc_float2 m2;// baricentric coordinates
    lc_uint m3;  // hit type
    lc_float m4; // t_hit
};
static_assert(sizeof(LCCommittedHit) == 24u, "LCCommittedHit size mismatch");
static_assert(alignof(LCCommittedHit) == 8u, "LCCommittedHit align mismatch");
enum LCInstanceFlags : unsigned int {
    LC_INSTANCE_FLAG_NONE = 0u,
    LC_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 0u,
    LC_INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1u << 1u,
    LC_INSTANCE_FLAG_DISABLE_ANYHIT = 1u << 2u,
    LC_INSTANCE_FLAG_ENFORCE_ANYHIT = 1u << 3u,
};

struct alignas(16) LCAccelInstance {
    lc_array<lc_float4, 3> m;
    lc_uint instance_id;
    lc_uint sbt_offset;
    lc_uint mask;
    lc_uint flags;
    lc_uint pad[4];
};

struct alignas(16u) LCAccel {
    unsigned long long handle;
    LCAccelInstance *instances;
};

[[nodiscard]] __device__ inline auto lc_accel_instance_transform(LCAccel accel, lc_uint instance_id) noexcept {
    lc_assume(__isGlobal(accel.instances));
    auto m = accel.instances[instance_id].m;
    return lc_make_float4x4(
        m[0].x, m[1].x, m[2].x, 0.0f,
        m[0].y, m[1].y, m[2].y, 0.0f,
        m[0].z, m[1].z, m[2].z, 0.0f,
        m[0].w, m[1].w, m[2].w, 1.0f);
}

__device__ inline void lc_accel_set_instance_transform(LCAccel accel, lc_uint index, lc_float4x4 m) noexcept {
    lc_assume(__isGlobal(accel.instances));
    lc_array<lc_float4, 3> p;
    p[0].x = m[0][0];
    p[0].y = m[1][0];
    p[0].z = m[2][0];
    p[0].w = m[3][0];
    p[1].x = m[0][1];
    p[1].y = m[1][1];
    p[1].z = m[2][1];
    p[1].w = m[3][1];
    p[2].x = m[0][2];
    p[2].y = m[1][2];
    p[2].z = m[2][2];
    p[2].w = m[3][2];
    accel.instances[index].m = p;
}

__device__ inline void lc_accel_set_instance_visibility(LCAccel accel, lc_uint index, lc_uint mask) noexcept {
    lc_assume(__isGlobal(accel.instances));
    accel.instances[index].mask = mask & 0xffu;
}

__device__ inline void lc_accel_set_instance_opacity(LCAccel accel, lc_uint index, bool opaque) noexcept {
    lc_assume(__isGlobal(accel.instances));
    auto flags = accel.instances[index].flags;
    // procedural primitives ignores the opaque flag, so only
    // apply the change when the instance is a triangle mesh
    if (flags & LC_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING) {
        flags &= ~(LC_INSTANCE_FLAG_DISABLE_ANYHIT |
                   LC_INSTANCE_FLAG_ENFORCE_ANYHIT);
        flags |= opaque ? LC_INSTANCE_FLAG_DISABLE_ANYHIT :
                          LC_INSTANCE_FLAG_ENFORCE_ANYHIT;
        accel.instances[index].flags = flags;
    }
}

__device__ inline float atomicCAS(float *a, float cmp, float v) noexcept {
    return __uint_as_float(atomicCAS(reinterpret_cast<lc_uint *>(a),
                                     __float_as_uint(cmp),
                                     __float_as_uint(v)));
}

__device__ inline float atomicSub(float *a, float v) noexcept {
    return atomicAdd(a, -v);
}

__device__ inline float atomicMin(float *a, float v) noexcept {
    for (;;) {
        if (auto old = *a;// read old
            old <= v /* no need to update */ ||
            atomicCAS(a, old, v) == old) { return old; }
    }
}

__device__ inline float atomicMax(float *a, float v) noexcept {
    for (;;) {
        if (auto old = *a;// read old
            old >= v /* no need to update */ ||
            atomicCAS(a, old, v) == old) { return old; }
    }
}

#define lc_atomic_exchange(atomic_ref, value) atomicExch(&(atomic_ref), value)
#define lc_atomic_compare_exchange(atomic_ref, cmp, value) atomicCAS(&(atomic_ref), cmp, value)
#define lc_atomic_fetch_add(atomic_ref, value) atomicAdd(&(atomic_ref), value)
#define lc_atomic_fetch_sub(atomic_ref, value) atomicSub(&(atomic_ref), value)
#define lc_atomic_fetch_min(atomic_ref, value) atomicMin(&(atomic_ref), value)
#define lc_atomic_fetch_max(atomic_ref, value) atomicMax(&(atomic_ref), value)
#define lc_atomic_fetch_and(atomic_ref, value) atomicAnd(&(atomic_ref), value)
#define lc_atomic_fetch_or(atomic_ref, value) atomicOr(&(atomic_ref), value)
#define lc_atomic_fetch_xor(atomic_ref, value) atomicXor(&(atomic_ref), value)

// static block size
[[nodiscard]] __device__ constexpr lc_uint3 lc_block_size() noexcept {
    return LC_BLOCK_SIZE;
}

#ifdef LUISA_ENABLE_OPTIX

enum LCPayloadTypeID : unsigned int {
    LC_PAYLOAD_TYPE_DEFAULT = 0u,
    LC_PAYLOAD_TYPE_ID_0 = 1u << 0u,
    LC_PAYLOAD_TYPE_ID_1 = 1u << 1u,
    LC_PAYLOAD_TYPE_ID_2 = 1u << 2u,
    LC_PAYLOAD_TYPE_ID_3 = 1u << 3u,
    LC_PAYLOAD_TYPE_ID_4 = 1u << 4u,
    LC_PAYLOAD_TYPE_ID_5 = 1u << 5u,
    LC_PAYLOAD_TYPE_ID_6 = 1u << 6u,
    LC_PAYLOAD_TYPE_ID_7 = 1u << 7u,
};

#define LC_PAYLOAD_TYPE_TRACE_CLOSEST (LC_PAYLOAD_TYPE_ID_0)
#define LC_PAYLOAD_TYPE_TRACE_ANY (LC_PAYLOAD_TYPE_ID_1)
#define LC_PAYLOAD_TYPE_RAY_QUERY (LC_PAYLOAD_TYPE_ID_2)

inline void lc_set_payload_types(LCPayloadTypeID type) noexcept {
    asm volatile("call _optix_set_payload_types, (%0);"
                 :
                 : "r"(type)
                 :);
}

template<lc_uint i>
inline void lc_set_payload(lc_uint x) noexcept {
    asm volatile("call _optix_set_payload, (%0, %1);"
                 :
                 : "r"(i), "r"(x)
                 :);
}

template<lc_uint i>
[[nodiscard]] auto lc_get_payload() noexcept {
    auto r = 0u;
    asm volatile("call (%0), _optix_get_payload, (%1);"
                 : "=r"(r)
                 : "r"(i)
                 :);
    return r;
}

[[nodiscard]] inline auto lc_get_primitive_index() noexcept {
    lc_uint u0;
    asm("call (%0), _optix_read_primitive_idx, ();"
        : "=r"(u0)
        :);
    return u0;
}

[[nodiscard]] inline auto lc_get_instance_index() noexcept {
    lc_uint u0;
    asm("call (%0), _optix_read_instance_idx, ();"
        : "=r"(u0)
        :);
    return u0;
}

[[nodiscard]] inline auto lc_get_bary_coords() noexcept {
    float f0, f1;
    asm("call (%0, %1), _optix_get_triangle_barycentrics, ();"
        : "=f"(f0), "=f"(f1)
        :);
    return lc_make_float2(f0, f1);
}

[[nodiscard]] inline auto lc_get_hit_distance() noexcept {
    float f0;
    asm("call (%0), _optix_get_ray_tmax, ();"
        : "=f"(f0)
        :);
    return f0;
}

#ifdef LUISA_ENABLE_OPTIX_TRACE_CLOSEST
extern "C" __global__ void __closesthit__trace_closest() {
    lc_set_payload_types(LC_PAYLOAD_TYPE_TRACE_CLOSEST);
    auto inst = lc_get_instance_index();
    auto prim = lc_get_primitive_index();
    auto bary = lc_get_bary_coords();
    auto t_hit = lc_get_hit_distance();
    lc_set_payload<0u>(inst);
    lc_set_payload<1u>(prim);
    lc_set_payload<2u>(__float_as_uint(bary.x));
    lc_set_payload<3u>(__float_as_uint(bary.y));
    lc_set_payload<4u>(__float_as_uint(t_hit));
}

extern "C" __global__ void __miss__trace_closest() {
    lc_set_payload_types(LC_PAYLOAD_TYPE_TRACE_CLOSEST);
    lc_set_payload<0u>(~0u);
}
#endif

#ifdef LUISA_ENABLE_OPTIX_TRACE_ANY
extern "C" __global__ void __miss__trace_any() {
    lc_set_payload_types(LC_PAYLOAD_TYPE_TRACE_ANY);
    lc_set_payload<0u>(~0u);
}
#endif

[[nodiscard]] inline auto lc_undef() noexcept {
    auto u0 = 0u;
    asm("call (%0), _optix_undef_value, ();"
        : "=r"(u0)
        :);
    return u0;
}

template<lc_uint ch_index, lc_uint miss_index, lc_uint reg_count, lc_uint flags>
[[nodiscard]] inline auto lc_trace_impl(
    lc_uint payload_type, LCAccel accel, LCRay ray, lc_uint mask,
    lc_uint &r0, lc_uint &r1, lc_uint &r2, lc_uint &r3, lc_uint &r4) noexcept {
    auto ox = ray.m0[0];
    auto oy = ray.m0[1];
    auto oz = ray.m0[2];
    auto dx = ray.m2[0];
    auto dy = ray.m2[1];
    auto dz = ray.m2[2];
    auto t_min = ray.m1;
    auto t_max = ray.m3;
    auto u = lc_undef();
    [[maybe_unused]] unsigned int
        p0 = 0u,
        p1 = 0u, p2 = 0u, p3 = 0u, p4 = 0u,
        p5, p6, p7, p8, p9, p10, p11, p12, p13,
        p14, p15, p16, p17, p18, p19, p20, p21, p22,
        p23, p24, p25, p26, p27, p28, p29, p30, p31;
    asm volatile(
        "call"
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31),"
        "_optix_trace_typed_32,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);"
        : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3), "=r"(p4), "=r"(p5), "=r"(p6), "=r"(p7), "=r"(p8),
          "=r"(p9), "=r"(p10), "=r"(p11), "=r"(p12), "=r"(p13), "=r"(p14), "=r"(p15), "=r"(p16),
          "=r"(p17), "=r"(p18), "=r"(p19), "=r"(p20), "=r"(p21), "=r"(p22), "=r"(p23), "=r"(p24),
          "=r"(p25), "=r"(p26), "=r"(p27), "=r"(p28), "=r"(p29), "=r"(p30), "=r"(p31)
        : "r"(payload_type), "l"(accel.handle), "f"(ox), "f"(oy), "f"(oz), "f"(dx), "f"(dy), "f"(dz), "f"(t_min),
          "f"(t_max), "f"(0.0f), "r"(mask & 0xffu), "r"(flags), "r"(ch_index), "r"(0u),
          "r"(miss_index), "r"(reg_count), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u)
        :);
    r0 = p0;
    r1 = p1;
    r2 = p2;
    r3 = p3;
    r4 = p4;
}

enum LCRayFlags : unsigned int {
    LC_RAY_FLAG_NONE = 0u,
    LC_RAY_FLAG_DISABLE_ANYHIT = 1u << 0u,
    LC_RAY_FLAG_ENFORCE_ANYHIT = 1u << 1u,
    LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1u << 2u,
    LC_RAY_FLAG_DISABLE_CLOSESTHIT = 1u << 3u,
    LC_RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 1u << 4u,
    LC_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 1u << 5u,
    LC_RAY_FLAG_CULL_DISABLED_ANYHIT = 1u << 6u,
    LC_RAY_FLAG_CULL_ENFORCED_ANYHIT = 1u << 7u,
};

[[nodiscard]] inline auto lc_accel_trace_closest(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT;
    auto r0 = lc_undef();
    auto r1 = lc_undef();
    auto r2 = lc_undef();
    auto r3 = lc_undef();
    auto r4 = lc_undef();
    lc_trace_impl<0u, 0u, 5u, flags>(LC_PAYLOAD_TYPE_TRACE_CLOSEST, accel, ray, mask, r0, r1, r2, r3, r4);
    return LCTriangleHit{r0, r1, lc_make_float2(__uint_as_float(r2), __uint_as_float(r3)), __uint_as_float(r4)};
}

[[nodiscard]] inline auto lc_accel_trace_any(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT |
                           LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                           LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    auto r0 = lc_undef();
    auto r1 = lc_undef();
    auto r2 = lc_undef();
    auto r3 = lc_undef();
    auto r4 = lc_undef();
    lc_trace_impl<0u, 2u, 1u, flags>(LC_PAYLOAD_TYPE_TRACE_ANY, accel, ray, mask, r0, r1, r2, r3, r4);
    return r0 != ~0u;
}

[[nodiscard]] inline auto lc_dispatch_id() noexcept {
    lc_uint u0, u1, u2;
    asm("call (%0), _optix_get_launch_index_x, ();"
        : "=r"(u0)
        :);
    asm("call (%0), _optix_get_launch_index_y, ();"
        : "=r"(u1)
        :);
    asm("call (%0), _optix_get_launch_index_z, ();"
        : "=r"(u2)
        :);
    return lc_make_uint3(u0, u1, u2);
}

[[nodiscard]] inline auto lc_dispatch_size() noexcept {
    lc_uint u0, u1, u2;
    asm("call (%0), _optix_get_launch_dimension_x, ();"
        : "=r"(u0)
        :);
    asm("call (%0), _optix_get_launch_dimension_y, ();"
        : "=r"(u1)
        :);
    asm("call (%0), _optix_get_launch_dimension_z, ();"
        : "=r"(u2)
        :);
    return lc_make_uint3(u0, u1, u2);
}

#define lc_kernel_id() static_cast<lc_uint>(params.ls_kid.w)

[[nodiscard]] inline auto lc_thread_id() noexcept {
    return lc_dispatch_id() % lc_block_size();
}

[[nodiscard]] inline auto lc_block_id() noexcept {
    return lc_dispatch_id() / lc_block_size();
}

// ray query
enum LCHitKind : lc_uint {
    LC_HIT_KIND_NONE = 0x00u,
    LC_HIT_KIND_PROCEDURAL = 0x01u,
    LC_HIT_KIND_PROCEDURAL_TERMINATED = 0x02u,
    LC_HIT_KIND_TRIANGLE_FRONT_FACE = 0xfeu,
    LC_HIT_KIND_TRIANGLE_BACK_FACE = 0xffu,
};

[[nodiscard]] inline auto lc_get_hit_kind() noexcept {
    auto u0 = 0u;
    asm("call (%0), _optix_get_hit_kind, ();"
        : "=r"(u0)
        :);
    return u0;
}

enum LCHitTypePrefix : lc_uint {
    LC_HIT_TYPE_PREFIX_TRIANGLE = 0x0u << 28u,
    LC_HIT_TYPE_PREFIX_PROCEDURAL = 0x1u << 28u,
    LC_HIT_TYPE_PREFIX_MASK = 0xfu << 28u,
};

template<bool terminate_on_first>
struct LCRayQuery {
    LCAccel accel;
    LCRay ray;
    lc_uint mask;
    LCCommittedHit hit;
};

using LCRayQueryAll = LCRayQuery<false>;
using LCRayQueryAny = LCRayQuery<true>;

[[nodiscard]] inline auto lc_ray_query_decode_hit(lc_uint u0, lc_uint u1, lc_uint u2, lc_uint u3, lc_uint u4) noexcept {
    LCCommittedHit hit;
    hit.m0 = u0 & ~LC_HIT_TYPE_PREFIX_MASK;
    hit.m1 = u1;
    hit.m2 = lc_make_float2(__uint_as_float(u2), __uint_as_float(u3));
    hit.m3 = ((u0 >> 28u) + 1u) & 0x3u;
    hit.m4 = __uint_as_float(u4);
    return hit;
}

template<bool terminate_on_first>
[[nodiscard]] inline auto lc_ray_query_trace(LCRayQuery<terminate_on_first> &q, lc_uint impl_tag, void *ctx) noexcept {
    constexpr auto flags = terminate_on_first ?
                               LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT :
                               LC_RAY_FLAG_NONE;
    auto p_ctx = reinterpret_cast<lc_ulong>(ctx);
    auto r0 = impl_tag;
    auto r1 = static_cast<lc_uint>(p_ctx >> 32u);
    auto r2 = static_cast<lc_uint>(p_ctx);
    auto r3 = lc_undef();
    auto r4 = lc_undef();
    lc_trace_impl<1u, 1u, 5u, flags>(LC_PAYLOAD_TYPE_RAY_QUERY, q.accel, q.ray, q.mask, r0, r1, r2, r3, r4);
    q.hit = lc_ray_query_decode_hit(r0, r1, r2, r3, r4);
}

[[nodiscard]] inline auto lc_accel_query_all(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    return LCRayQueryAll{accel, ray, mask, LCCommittedHit{}};
}

[[nodiscard]] inline auto lc_accel_query_any(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    return LCRayQueryAny{accel, ray, mask, LCCommittedHit{}};
}

template<bool terminate_on_first>
[[nodiscard]] inline auto lc_ray_query_committed_hit(LCRayQuery<terminate_on_first> q) noexcept {
    return q.hit;
}

[[nodiscard]] inline auto lc_ray_query_triangle_candidate() noexcept {
    auto inst = lc_get_instance_index();
    auto prim = lc_get_primitive_index();
    auto bary = lc_get_bary_coords();
    auto t_hit = lc_get_hit_distance();
    return LCTriangleHit{inst, prim, bary, t_hit};
}

[[nodiscard]] inline auto lc_ray_query_procedural_candidate() noexcept {
    auto inst = lc_get_instance_index();
    auto prim = lc_get_primitive_index();
    return LCProceduralHit{inst, prim};
}

[[nodiscard]] inline auto lc_ray_query_world_ray() noexcept {
    float ox, oy, oz, t_min, dx, dy, dz, t_max;
    // origin
    asm("call (%0), _optix_get_world_ray_origin_x, ();"
        : "=f"(ox)
        :);
    asm("call (%0), _optix_get_world_ray_origin_y, ();"
        : "=f"(oy)
        :);
    asm("call (%0), _optix_get_world_ray_origin_z, ();"
        : "=f"(oz)
        :);
    // t_min
    asm("call (%0), _optix_get_ray_tmin, ();"
        : "=f"(t_min)
        :);
    // direction
    asm("call (%0), _optix_get_world_ray_direction_x, ();"
        : "=f"(dx)
        :);
    asm("call (%0), _optix_get_world_ray_direction_y, ();"
        : "=f"(dy)
        :);
    asm("call (%0), _optix_get_world_ray_direction_z, ();"
        : "=f"(dz)
        :);
    // t_max
    asm("call (%0), _optix_get_ray_tmax, ();"
        : "=f"(t_max)
        :);
    LCRay ray{};
    ray.m0[0] = ox;
    ray.m0[1] = oy;
    ray.m0[2] = oz;
    ray.m1 = t_min;
    ray.m2[0] = dx;
    ray.m2[1] = dy;
    ray.m2[2] = dz;
    ray.m3 = t_max;
    return ray;
}

inline void lc_ray_query_report_intersection(lc_uint kind, lc_float t) noexcept {
    auto ret = 0u;
    asm volatile("call (%0), _optix_report_intersection_0"
                 ", (%1, %2);"
                 : "=r"(ret)
                 : "f"(t), "r"(kind)
                 :);
}

inline void lc_ray_query_ignore_intersection() noexcept {
    asm volatile("call _optix_ignore_intersection, ();");
}

inline void lc_ray_query_terminate() noexcept {
    asm volatile("call _optix_terminate_ray, ();");
}

#if LUISA_RAY_QUERY_IMPL_COUNT > 32
#error "LUISA_RAY_QUERY_IMPL_COUNT must be less than or equal to 32"
#endif

struct LCIntersectionResult {
    lc_float t_hit{};
    lc_bool committed{};
    lc_bool terminated{};
};

#define LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(index) \
    [[nodiscard]] inline LCIntersectionResult lc_ray_query_procedural_intersection_##index(void *ctx_in) noexcept

#define LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(index) \
    [[nodiscard]] inline LCIntersectionResult lc_ray_query_triangle_intersection_##index(void *ctx_in) noexcept

#define LC_RAY_QUERY_PROCEDURAL_CANDIDATE_HIT(q) lc_ray_query_procedural_candidate()
#define LC_RAY_QUERY_TRIANGLE_CANDIDATE_HIT(q) lc_ray_query_triangle_candidate()
#define LC_RAY_QUERY_WORLD_RAY(q) lc_ray_query_world_ray()
#define LC_RAY_QUERY_COMMIT_TRIANGLE(q) static_cast<void>(result.committed = true)
#define LC_RAY_QUERY_COMMIT_PROCEDURAL(q, t) \
    do {                                     \
        result.committed = true;             \
        result.t_hit = t;                    \
    } while (false)
#define LC_RAY_QUERY_TERMINATE(q) static_cast<void>(result.terminated = true)

// declare `lc_ray_query_intersection` for at most 32 implementations
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(0);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(1);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(2);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(3);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(4);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(5);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(6);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(7);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(8);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(9);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(10);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(11);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(12);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(13);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(14);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(15);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(16);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(17);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(18);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(19);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(20);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(21);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(22);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(23);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(24);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(25);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(26);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(27);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(28);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(29);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(30);
LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(31);

LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(0);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(1);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(2);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(3);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(4);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(5);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(6);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(7);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(8);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(9);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(10);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(11);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(12);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(13);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(14);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(15);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(16);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(17);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(18);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(19);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(20);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(21);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(22);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(23);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(24);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(25);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(26);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(27);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(28);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(29);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(30);
LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(31);

#ifdef LUISA_ENABLE_OPTIX_RAY_QUERY

extern "C" __global__ void __intersection__ray_query() {
#if LUISA_RAY_QUERY_IMPL_COUNT > 0
    lc_set_payload_types(LC_PAYLOAD_TYPE_RAY_QUERY);
    auto query_id = lc_get_payload<0u>();
    auto p_ctx_hi = lc_get_payload<1u>();
    auto p_ctx_lo = lc_get_payload<2u>();
    auto ctx = reinterpret_cast<void *>((static_cast<lc_ulong>(p_ctx_hi) << 32u) | p_ctx_lo);
    LCIntersectionResult r{};
    switch (query_id) {
#if LUISA_RAY_QUERY_IMPL_COUNT > 0
        case 0u: r = lc_ray_query_procedural_intersection_0(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 1
        case 1u: r = lc_ray_query_procedural_intersection_1(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 2
        case 2u: r = lc_ray_query_procedural_intersection_2(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 3
        case 3u: r = lc_ray_query_procedural_intersection_3(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 4
        case 4u: r = lc_ray_query_procedural_intersection_4(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 5
        case 5u: r = lc_ray_query_procedural_intersection_5(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 6
        case 6u: r = lc_ray_query_procedural_intersection_6(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 7
        case 7u: r = lc_ray_query_procedural_intersection_7(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 8
        case 8u: r = lc_ray_query_procedural_intersection_8(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 9
        case 9u: r = lc_ray_query_procedural_intersection_9(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 10
        case 10u: r = lc_ray_query_procedural_intersection_10(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 11
        case 11u: r = lc_ray_query_procedural_intersection_11(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 12
        case 12u: r = lc_ray_query_procedural_intersection_12(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 13
        case 13u: r = lc_ray_query_procedural_intersection_13(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 14
        case 14u: r = lc_ray_query_procedural_intersection_14(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 15
        case 15u: r = lc_ray_query_procedural_intersection_15(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 16
        case 16u: r = lc_ray_query_procedural_intersection_16(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 17
        case 17u: r = lc_ray_query_procedural_intersection_17(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 18
        case 18u: r = lc_ray_query_procedural_intersection_18(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 19
        case 19u: r = lc_ray_query_procedural_intersection_19(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 20
        case 20u: r = lc_ray_query_procedural_intersection_20(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 21
        case 21u: r = lc_ray_query_procedural_intersection_21(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 22
        case 22u: r = lc_ray_query_procedural_intersection_22(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 23
        case 23u: r = lc_ray_query_procedural_intersection_23(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 24
        case 24u: r = lc_ray_query_procedural_intersection_24(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 25
        case 25u: r = lc_ray_query_procedural_intersection_25(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 26
        case 26u: r = lc_ray_query_procedural_intersection_26(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 27
        case 27u: r = lc_ray_query_procedural_intersection_27(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 28
        case 28u: r = lc_ray_query_procedural_intersection_28(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 29
        case 29u: r = lc_ray_query_procedural_intersection_29(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 30
        case 30u: r = lc_ray_query_procedural_intersection_30(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 31
        case 31u: r = lc_ray_query_procedural_intersection_31(ctx); break;
#endif
        default: lc_unreachable();
    }
    if (r.committed) {
        lc_ray_query_report_intersection(
            r.terminated ?
                LC_HIT_KIND_PROCEDURAL_TERMINATED :
                LC_HIT_KIND_PROCEDURAL,
            r.t_hit);
    }
#endif
}

extern "C" __global__ void __closesthit__ray_query() {
#if LUISA_RAY_QUERY_IMPL_COUNT > 0
    lc_set_payload_types(LC_PAYLOAD_TYPE_RAY_QUERY);
    auto hit_kind = lc_get_hit_kind();
    auto prefix = (hit_kind == LC_HIT_KIND_TRIANGLE_FRONT_FACE ||
                   hit_kind == LC_HIT_KIND_TRIANGLE_BACK_FACE) ?
                      LC_HIT_TYPE_PREFIX_TRIANGLE :
                      LC_HIT_TYPE_PREFIX_PROCEDURAL;
    auto inst = lc_get_instance_index();
    auto prim = lc_get_primitive_index();
    auto bary = lc_get_bary_coords();
    auto t_hit = lc_get_hit_distance();
    lc_set_payload<0u>(prefix | inst);
    lc_set_payload<1u>(prim);
    lc_set_payload<2u>(__float_as_uint(bary.x));
    lc_set_payload<3u>(__float_as_uint(bary.y));
    lc_set_payload<4u>(__float_as_uint(t_hit));
#endif
}

extern "C" __global__ void __anyhit__ray_query() {
#if LUISA_RAY_QUERY_IMPL_COUNT > 0
    lc_set_payload_types(LC_PAYLOAD_TYPE_RAY_QUERY);
    auto hit_kind = lc_get_hit_kind();
    auto should_terminate = false;
    if (hit_kind == LC_HIT_KIND_TRIANGLE_FRONT_FACE ||
        hit_kind == LC_HIT_KIND_TRIANGLE_BACK_FACE) {// triangle
        auto query_id = lc_get_payload<0u>();
        auto p_ctx_hi = lc_get_payload<1u>();
        auto p_ctx_lo = lc_get_payload<2u>();
        auto ctx = reinterpret_cast<void *>((static_cast<lc_ulong>(p_ctx_hi) << 32u) | p_ctx_lo);
        LCIntersectionResult r{};
        switch (query_id) {
#if LUISA_RAY_QUERY_IMPL_COUNT > 0
            case 0u: r = lc_ray_query_triangle_intersection_0(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 1
            case 1u: r = lc_ray_query_triangle_intersection_1(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 2
            case 2u: r = lc_ray_query_triangle_intersection_2(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 3
            case 3u: r = lc_ray_query_triangle_intersection_3(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 4
            case 4u: r = lc_ray_query_triangle_intersection_4(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 5
            case 5u: r = lc_ray_query_triangle_intersection_5(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 6
            case 6u: r = lc_ray_query_triangle_intersection_6(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 7
            case 7u: r = lc_ray_query_triangle_intersection_7(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 8
            case 8u: r = lc_ray_query_triangle_intersection_8(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 9
            case 9u: r = lc_ray_query_triangle_intersection_9(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 10
            case 10u: r = lc_ray_query_triangle_intersection_10(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 11
            case 11u: r = lc_ray_query_triangle_intersection_11(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 12
            case 12u: r = lc_ray_query_triangle_intersection_12(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 13
            case 13u: r = lc_ray_query_triangle_intersection_13(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 14
            case 14u: r = lc_ray_query_triangle_intersection_14(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 15
            case 15u: r = lc_ray_query_triangle_intersection_15(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 16
            case 16u: r = lc_ray_query_triangle_intersection_16(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 17
            case 17u: r = lc_ray_query_triangle_intersection_17(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 18
            case 18u: r = lc_ray_query_triangle_intersection_18(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 19
            case 19u: r = lc_ray_query_triangle_intersection_19(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 20
            case 20u: r = lc_ray_query_triangle_intersection_20(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 21
            case 21u: r = lc_ray_query_triangle_intersection_21(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 22
            case 22u: r = lc_ray_query_triangle_intersection_22(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 23
            case 23u: r = lc_ray_query_triangle_intersection_23(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 24
            case 24u: r = lc_ray_query_triangle_intersection_24(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 25
            case 25u: r = lc_ray_query_triangle_intersection_25(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 26
            case 26u: r = lc_ray_query_triangle_intersection_26(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 27
            case 27u: r = lc_ray_query_triangle_intersection_27(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 28
            case 28u: r = lc_ray_query_triangle_intersection_28(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 29
            case 29u: r = lc_ray_query_triangle_intersection_29(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 30
            case 30u: r = lc_ray_query_triangle_intersection_30(ctx); break;
#endif
#if LUISA_RAY_QUERY_IMPL_COUNT > 31
            case 31u: r = lc_ray_query_triangle_intersection_31(ctx); break;
#endif
            default: lc_unreachable();
        }
        // ignore the intersection if not committed
        if (!r.committed) { lc_ray_query_ignore_intersection(); }
        should_terminate = r.terminated;
    } else {// procedural
        should_terminate = hit_kind == LC_HIT_KIND_PROCEDURAL_TERMINATED;
    }
    if (should_terminate) {
        lc_ray_query_terminate();
    }
#endif
}

extern "C" __global__ void __miss__ray_query() {
#if LUISA_RAY_QUERY_IMPL_COUNT > 0
    lc_set_payload_types(LC_PAYLOAD_TYPE_RAY_QUERY);
    lc_set_payload<0u>(~0u);
#endif
}

#endif

#else

#define lc_dispatch_size() lc_make_uint3(params.ls_kid)
#define lc_kernel_id() static_cast<lc_uint>(params.ls_kid.w)

[[nodiscard]] __device__ inline auto lc_thread_id() noexcept {
    return lc_make_uint3(lc_uint(threadIdx.x),
                         lc_uint(threadIdx.y),
                         lc_uint(threadIdx.z));
}

[[nodiscard]] __device__ inline auto lc_block_id() noexcept {
    return lc_make_uint3(lc_uint(blockIdx.x),
                         lc_uint(blockIdx.y),
                         lc_uint(blockIdx.z));
}

[[nodiscard]] __device__ inline auto lc_dispatch_id() noexcept {
    return lc_block_id() * lc_block_size() + lc_thread_id();
}

__device__ inline void lc_synchronize_block() noexcept {
    __syncthreads();
}

#endif

// autodiff
#define LC_GRAD_SHADOW_VARIABLE(x) auto x##_grad = lc_zero<decltype(x)>()
#define LC_MARK_GRAD(x, dx) x##_grad = dx
#define LC_GRAD(x) (x##_grad)
#define LC_ACCUM_GRAD(x_grad, dx) lc_accumulate_grad(&(x_grad), (dx))
#define LC_REQUIRES_GRAD(x) x##_grad = lc_zero<decltype(x##_grad)>()

template<typename T>
struct alignas(alignof(T) < 4u ? 4u : alignof(T)) LCPack {
    T value;
};

template<typename T>
__device__ inline void lc_pack_to(const T &x, LCBuffer<lc_uint> array, lc_uint idx) noexcept {
    constexpr lc_uint N = (sizeof(T) + 3u) / 4u;
    if constexpr (alignof(T) < 4u) {
        // too small to be aligned to 4 bytes
        LCPack<T> pack{};
        pack.value = x;
        auto data = reinterpret_cast<const lc_uint *>(&pack);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            array.ptr[idx + i] = data[i];
        }
    } else {
        // safe to reinterpret the pointer as lc_uint *
        auto data = reinterpret_cast<const lc_uint *>(&x);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            array.ptr[idx + i] = data[i];
        }
    }
}

template<typename T>
[[nodiscard]] __device__ inline T lc_unpack_from(LCBuffer<lc_uint> array, lc_uint idx) noexcept {
    if constexpr (alignof(T) <= 4u) {
        // safe to reinterpret the pointer as T *
        auto data = reinterpret_cast<const T *>(&array.ptr[idx]);
        return *data;
    } else {
        // copy to a temporary aligned buffer to avoid unaligned access
        constexpr lc_uint N = (sizeof(T) + 3u) / 4u;
        LCPack<T> x{};
        auto data = reinterpret_cast<lc_uint *>(&x);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            data[i] = array.ptr[idx + i];
        }
        return x.value;
    }
}
