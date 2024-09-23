#pragma once

[[nodiscard]] __device__ constexpr auto lc_infinity_half() noexcept { return __ushort_as_half(static_cast<unsigned short>(0x7c00u)); }
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

[[noreturn]] inline void lc_trap() noexcept { asm("trap;"); }

template<typename T = void>
[[noreturn]] inline __device__ T lc_unreachable(
    const char *file, int line) noexcept {
#if LC_NVRTC_VERSION < 110300 || defined(LUISA_DEBUG)
    printf("Unreachable code reached [%s:%d]\n", file, line);
    lc_trap();
#else
    __builtin_unreachable();
#endif
}

template<typename T = void>
[[noreturn]] inline __device__ T lc_unreachable_with_message(
    const char *file, int line, const char *msg) noexcept {
#if LC_NVRTC_VERSION < 110300 || defined(LUISA_DEBUG)
    printf("Unreachable code reached [%s:%d]\nMessage: %s\n", file, line, msg);
    lc_trap();
#else
    __builtin_unreachable();
#endif
}

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

#ifdef LUISA_DEBUG

#define lc_assert(x)                                                                     \
    do {                                                                                 \
        if (!(x)) {                                                                      \
            printf("Assertion failed: " #x " [" __FILE__ ":" STRINGIFY(__LINE__) "]\n"); \
            lc_trap();                                                                   \
        }                                                                                \
    } while (false)

#define lc_assert_with_message(x, msg)                                                                     \
    do {                                                                                                   \
        if (!(x)) {                                                                                        \
            printf("Assertion failed: " #x " [" __FILE__ ":" STRINGIFY(__LINE__) "]\nMessage: %s\n", msg); \
            lc_trap();                                                                                     \
        }                                                                                                  \
    } while (false)

#define lc_check_in_bounds(size, max_size)                               \
    do {                                                                 \
        if (!((size) < (max_size))) {                                    \
            printf("Out of bounds: !(%s: %llu < %s: %llu) [%s:%d:%s]\n", \
                   #size, static_cast<size_t>(size),                     \
                   #max_size, static_cast<size_t>(max_size),             \
                   __FILE__, static_cast<int>(__LINE__),                 \
                   __FUNCTION__);                                        \
            lc_trap();                                                   \
        }                                                                \
    } while (false)

#else
inline __device__ void lc_assert(bool) noexcept {}
inline __device__ void lc_assert_with_message(bool, const char *) noexcept {}
#endif

template<typename T>
[[nodiscard]] __device__ inline auto lc_address_of(T &object) noexcept {
    return reinterpret_cast<lc_ulong>(&object);
}

[[nodiscard]] __device__ inline auto lc_bits_to_half(lc_ushort bits) noexcept {
    return *reinterpret_cast<const lc_half *>(&bits);
}

[[nodiscard]] __device__ inline auto lc_half_to_bits(lc_half h) noexcept {
    return *reinterpret_cast<const lc_ushort *>(&h);
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
    lc_uint offset;
    lc_uint capacity;

    [[nodiscard]] auto header() const noexcept {
        return reinterpret_cast<LCIndirectHeader *>(data);
    }

    [[nodiscard]] auto dispatches() const noexcept {
        return reinterpret_cast<LCIndirectDispatch *>(reinterpret_cast<lc_ulong>(data) + sizeof(LCIndirectHeader));
    }
};

void lc_indirect_set_dispatch_count(const LCIndirectBuffer buffer, lc_uint count) noexcept {
#ifdef LUISA_DEBUG
    lc_check_in_bounds(buffer.offset + count, buffer.capacity + 1u);
#endif
    buffer.header()->size = count;
}

void lc_indirect_set_dispatch_kernel(const LCIndirectBuffer buffer, lc_uint index, lc_uint3 block_size, lc_uint3 dispatch_size, lc_uint kernel_id) noexcept {
#ifdef LUISA_DEBUG
    lc_check_in_bounds(index, buffer.header()->size);
    lc_check_in_bounds(index + buffer.offset, buffer.capacity);
#endif
    buffer.dispatches()[index + buffer.offset] = LCIndirectDispatch{block_size, lc_make_uint4(dispatch_size, kernel_id)};
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
[[nodiscard]] __device__ inline T lc_buffer_read(LCBuffer<T> buffer, Index index) noexcept {
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

template<typename T>
[[nodiscard]] __device__ inline auto lc_buffer_address(LCBuffer<T> buffer) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
    return reinterpret_cast<lc_ulong>(buffer.ptr);
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
        return static_cast<float>(x);
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
        return static_cast<lc_half>(x);
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
            result.x = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(x)));
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint x, y;
            asm("suld.b.2d.v2.b16.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(x)));
            result.y = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(y)));
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint x, y, z, w;
            asm("suld.b.2d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(x)));
            result.y = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(y)));
            result.z = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(z)));
            result.w = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(w)));
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
            lc_uint v = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.x));
            asm volatile("sust.b.2d.b16.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint vx = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.x));
            lc_uint vy = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.y));
            asm volatile("sust.b.2d.v2.b16.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint vx = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.x));
            lc_uint vy = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.y));
            lc_uint vz = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.z));
            lc_uint vw = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.w));
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
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
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
            result.x = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(x)));
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint x, y;
            asm("suld.b.3d.v2.b16.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(x)));
            result.y = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(y)));
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint x, y, z, w;
            asm("suld.b.3d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(x)));
            result.y = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(y)));
            result.z = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(z)));
            result.w = lc_texel_read_convert<T, lc_half>(lc_bits_to_half(static_cast<lc_ushort>(w)));
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
            lc_uint v = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.x));
            asm volatile("sust.b.3d.b16.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint vx = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.x));
            lc_uint vy = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.y));
            asm volatile("sust.b.3d.v2.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint vx = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.x));
            lc_uint vy = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.y));
            lc_uint vz = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.z));
            lc_uint vw = lc_half_to_bits(lc_texel_write_convert<lc_half>(value.w));
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
    void *__restrict__ buffer;
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
[[nodiscard]] inline __device__ T *lc_bindless_buffer(LCBindlessArray array, lc_uint index) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const T *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(i, lc_bindless_buffer_size<T>(array, index));
#endif
    return buffer;
}

template<typename T>
[[nodiscard]] inline __device__ T lc_bindless_buffer_read(LCBindlessArray array, lc_uint index, lc_ulong i) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const T *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(i, lc_bindless_buffer_size<T>(array, index));
#endif
    return buffer[i];
}

template<typename T>
[[nodiscard]] inline __device__ void lc_bindless_buffer_write(LCBindlessArray array, lc_uint index, lc_ulong i, T value) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<T *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(i, lc_bindless_buffer_size<T>(array, index));
#endif
    buffer[i] = value;
}

[[nodiscard]] inline __device__ auto lc_bindless_buffer_type(LCBindlessArray array, lc_uint index) noexcept {
    return 0ull;// TODO
}

[[nodiscard]] inline __device__ auto lc_bindless_buffer_address(LCBindlessArray array, lc_uint index) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const char *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
    return reinterpret_cast<lc_ulong>(buffer);
}

template<typename T>
[[nodiscard]] inline __device__ T lc_bindless_byte_buffer_read(LCBindlessArray array, lc_uint index, lc_ulong offset) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const char *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(offset + sizeof(T), lc_bindless_buffer_size<char>(array, index) + 1u);
#endif
    return *reinterpret_cast<const T *>(buffer + offset);
}

template<typename T>
inline __device__ void lc_bindless_byte_buffer_write(LCBindlessArray array, lc_uint index, lc_ulong offset, T value) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const char *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(offset + sizeof(T), lc_bindless_buffer_size<char>(array, index) + 1u);
#endif
    *reinterpret_cast<const T *>(buffer + offset) = value;
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
    BUILTIN = 1,
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

struct alignas(16) LCMotionSRT {
    lc_array<lc_float, 3> m0; // pivot
    lc_array<lc_float, 4> m1; // quaternion
    lc_array<lc_float, 3> m2; // scale
    lc_array<lc_float, 3> m3; // shear
    lc_array<lc_float, 3> m4; // translation
};

static_assert(sizeof(LCMotionSRT) == 64u, "LCMotionSRT size mismatch");
static_assert(alignof(LCMotionSRT) == 16u, "LCMotionSRT align mismatch");

struct LCMotionOptions {
    unsigned short count;
    unsigned short flags;
    float time_start;
    float time_end;
};

struct alignas(16) LCMotionTransformBuffer {
    unsigned long long child;
    LCMotionOptions options;
    unsigned int pad[3];
};

static_assert(sizeof(LCMotionTransformBuffer) == 32u, "LCMotionTransformBuffer size mismatch");

struct alignas(16) LCSRTData {
    float sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz;
};

static_assert(sizeof(LCSRTData) == 64u, "LCSRTData size mismatch");

struct alignas(16) LCMatrixData {
    float m[12];
};

static_assert(sizeof(LCMatrixData) == 48u, "LCMatrixData size mismatch");

enum LCInstanceFlags : lc_uint {
    LC_INSTANCE_FLAG_NONE = 0u,
    LC_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 0u,
    LC_INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1u << 1u,
    LC_INSTANCE_FLAG_DISABLE_ANYHIT = 1u << 2u,
    LC_INSTANCE_FLAG_ENFORCE_ANYHIT = 1u << 3u,
};

struct alignas(16) LCAccelInstance {
    lc_array<lc_float4, 3> m;
    lc_uint user_id;
    lc_uint sbt_offset;
    lc_uint mask;
    lc_uint flags;
    unsigned long long handle;
    lc_uint pad[2];
};

struct alignas(16u) LCAccel {
    unsigned long long handle;
    LCAccelInstance *instances;
};

template<typename T>
[[nodiscard]] __device__ T *lc_instance_motion_data(LCAccel accel, lc_uint inst_index, lc_uint key_index) noexcept {
    lc_assume(__isGlobal(accel.instances));
    auto handle = accel.instances[inst_index].handle;
    auto buffer = reinterpret_cast<LCMotionTransformBuffer *>(handle & ~0x0full);
#ifdef LUISA_DEBUG
    lc_check_in_bounds(key_index, buffer->options.count);
#endif
    return reinterpret_cast<T *>(buffer + 1) + key_index;
}

[[nodiscard]] __device__ lc_float4x4 lc_instance_motion_matrix(LCAccel accel, lc_uint inst_index, lc_uint key_index) noexcept {
    auto m = *lc_instance_motion_data<LCMatrixData>(accel, inst_index, key_index);
    return lc_make_float4x4(
        m.m[0], m.m[4], m.m[8], 0.0f,
        m.m[1], m.m[5], m.m[9], 0.0f,
        m.m[2], m.m[6], m.m[10], 0.0f,
        m.m[3], m.m[7], m.m[11], 1.0f);
}

[[nodiscard]] __device__ LCMotionSRT lc_instance_motion_srt(LCAccel accel, lc_uint inst_index, lc_uint key_index) noexcept {
    auto srt = *lc_instance_motion_data<LCSRTData>(accel, inst_index, key_index);
    LCMotionSRT result;
    result.m0[0] = srt.pvx;
    result.m0[1] = srt.pvy;
    result.m0[2] = srt.pvz;
    result.m1[0] = srt.qx;
    result.m1[1] = srt.qy;
    result.m1[2] = srt.qz;
    result.m1[3] = srt.qw;
    result.m2[0] = srt.sx;
    result.m2[1] = srt.sy;
    result.m2[2] = srt.sz;
    result.m3[0] = srt.tx;
    result.m3[1] = srt.ty;
    result.m3[2] = srt.tz;
    return result;
}

__device__ void lc_set_instance_motion_matrix(LCAccel accel, lc_uint inst_index, lc_uint key_index, lc_float4x4 m) noexcept {
    LCMatrixData data;
    data.m[0] = m[0][0];
    data.m[1] = m[1][0];
    data.m[2] = m[2][0];
    data.m[3] = m[3][0];
    data.m[4] = m[0][1];
    data.m[5] = m[1][1];
    data.m[6] = m[2][1];
    data.m[7] = m[3][1];
    data.m[8] = m[0][2];
    data.m[9] = m[1][2];
    data.m[10] = m[2][2];
    data.m[11] = m[3][2];
    *lc_instance_motion_data<LCMatrixData>(accel, inst_index, key_index) = data;
}

__device__ void lc_set_instance_motion_srt(LCAccel accel, lc_uint inst_index, lc_uint key_index, LCMotionSRT srt) noexcept {
    LCSRTData data;
    data.sx = srt.m2[0];
    data.sy = srt.m2[1];
    data.sz = srt.m2[2];
    data.pvx = srt.m0[0];
    data.pvy = srt.m0[1];
    data.pvz = srt.m0[2];
    data.qx = srt.m1[0];
    data.qy = srt.m1[1];
    data.qz = srt.m1[2];
    data.qw = srt.m1[3];
    data.tx = srt.m3[0];
    data.ty = srt.m3[1];
    data.tz = srt.m3[2];
    *lc_instance_motion_data<LCSRTData>(accel, inst_index, key_index) = data;
}

[[nodiscard]] __device__ inline auto lc_accel_instance_transform(LCAccel accel, lc_uint instance_id) noexcept {
    lc_assume(__isGlobal(accel.instances));
    auto m = accel.instances[instance_id].m;
    return lc_make_float4x4(
        m[0].x, m[1].x, m[2].x, 0.0f,
        m[0].y, m[1].y, m[2].y, 0.0f,
        m[0].z, m[1].z, m[2].z, 0.0f,
        m[0].w, m[1].w, m[2].w, 1.0f);
}

[[nodiscard]] __device__ inline auto lc_accel_instance_user_id(LCAccel accel, lc_uint instance_id) noexcept {
    lc_assume(__isGlobal(accel.instances));
    return accel.instances[instance_id].user_id;
}

[[nodiscard]] __device__ inline auto lc_accel_instance_visibility(LCAccel accel, lc_uint index) noexcept {
    lc_assume(__isGlobal(accel.instances));
    return accel.instances[index].mask;
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

__device__ inline void lc_accel_set_instance_user_id(LCAccel accel, lc_uint index, lc_uint user_id) noexcept {
    lc_assume(__isGlobal(accel.instances));
    accel.instances[index].user_id = user_id;
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

enum LCPayloadTypeID : lc_uint {
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

#define LC_PAYLOAD_TYPE_RAY_TRACE (LC_PAYLOAD_TYPE_ID_0)
#define LC_PAYLOAD_TYPE_RAY_QUERY (LC_PAYLOAD_TYPE_ID_1)

inline void lc_set_payload_types(LCPayloadTypeID type) noexcept {
    asm volatile("call _optix_set_payload_types, (%0);"
                 :
                 : "r"(type));
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

[[nodiscard]] inline auto lc_get_curve_parameter() noexcept {
    float f0;
    asm("call (%0), _optix_get_curve_parameter, ();"
        : "=f"(f0)
        :);
    return f0;
}

[[nodiscard]] inline auto lc_get_hit_distance() noexcept {
    float f0;
    asm("call (%0), _optix_get_ray_tmax, ();"
        : "=f"(f0)
        :);
    return f0;
}

[[nodiscard]] inline auto lc_undef() noexcept {
    auto u0 = 0u;
    asm("call (%0), _optix_undef_value, ();"
        : "=r"(u0)
        :);
    return u0;
}

inline void lc_shader_execution_reorder(lc_uint hint, lc_uint hint_bits) noexcept {
    asm volatile(
        "call (), _optix_hitobject_reorder, (%0,%1);"
        :
        : "r"(hint), "r"(hint_bits));
}

__device__ inline void lc_hit_object_reset() noexcept {
    asm volatile("call (), _optix_hitobject_make_nop, ();");
}

__device__ inline lc_float2 lc_hit_object_triangle_bary() noexcept {
    lc_uint u_bits, v_bits;
    asm volatile(
        "call (%0), _optix_hitobject_get_attribute, (%1);"
        : "=r"(u_bits)
        : "r"(0));
    asm volatile(
        "call (%0), _optix_hitobject_get_attribute, (%1);"
        : "=r"(v_bits)
        : "r"(1));
    return lc_make_float2(__int_as_float(u_bits), __int_as_float(v_bits));
}

__device__ inline lc_float lc_hit_object_curve_parameter() noexcept {
    lc_uint u_bits, v_bits;
    asm volatile(
        "call (%0), _optix_hitobject_get_attribute, (%1);"
        : "=r"(u_bits)
        : "r"(0));
    return __int_as_float(u_bits);
}

__device__ inline bool lc_hit_object_is_hit() noexcept {
    lc_uint result;
    asm volatile(
        "call (%0), _optix_hitobject_is_hit, ();"
        : "=r"(result)
        :);
    return result;
}

__device__ inline lc_uint lc_hit_object_instance_index() noexcept {
    lc_uint result;
    asm volatile(
        "call (%0), _optix_hitobject_get_instance_idx, ();"
        : "=r"(result)
        :);
    return result;
}

__device__ inline lc_uint lc_hit_object_primitive_index() noexcept {
    lc_uint result;
    asm volatile(
        "call (%0), _optix_hitobject_get_primitive_idx, ();"
        : "=r"(result)
        :);
    return result;
}

__device__ inline lc_float lc_hit_object_ray_t_max() noexcept {
    float result;
    asm volatile(
        "call (%0), _optix_hitobject_get_ray_tmax, ();"
        : "=f"(result)
        :);
    return result;
}

__device__ inline lc_uint lc_hit_object_hit_kind() noexcept {
    lc_uint result;
    asm volatile(
        "call (%0), _optix_hitobject_get_hitkind, ();"
        : "=r"(result)
        :);
    return result;
}

enum LCRayFlags : lc_uint {
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

// ray query
enum LCHitKind : lc_uint {
    LC_HIT_KIND_NONE = 0x00u,
    LC_HIT_KIND_PROCEDURAL = 0x01u,
    LC_HIT_KIND_PROCEDURAL_TERMINATED = 0x02u,
    LC_HIT_KIND_TRIANGLE_FRONT_FACE = 0xfeu,
    LC_HIT_KIND_TRIANGLE_BACK_FACE = 0xffu,
};

template<lc_uint payload_type, lc_uint sbt_offset, lc_uint reg_count = 0u>
inline void lc_ray_traverse(lc_uint flags, LCAccel accel,
                            LCRay ray, lc_float time, lc_uint mask,
                            lc_uint r0 = lc_undef(),
                            lc_uint r1 = lc_undef()) noexcept {
    static_assert(reg_count <= 2u, "Register count must be less than 2.");
    auto ox = ray.m0[0];
    auto oy = ray.m0[1];
    auto oz = ray.m0[2];
    auto dx = ray.m2[0];
    auto dy = ray.m2[1];
    auto dz = ray.m2[2];
    auto t_min = ray.m1;
    auto t_max = ray.m3;
    [[maybe_unused]] lc_uint
        p0,
        p1, p2, p3, p4, p5, p6, p7,
        p8, p9, p10, p11, p12, p13, p14, p15,
        p16, p17, p18, p19, p20, p21, p22, p23,
        p24, p25, p26, p27, p28, p29, p30, p31;
    auto u = lc_undef();
    // traverse without calling the closest or any hit programs
    asm volatile(
        "call"
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31),"
        "_optix_hitobject_traverse,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);"
        : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3), "=r"(p4), "=r"(p5), "=r"(p6), "=r"(p7), "=r"(p8),
          "=r"(p9), "=r"(p10), "=r"(p11), "=r"(p12), "=r"(p13), "=r"(p14), "=r"(p15), "=r"(p16),
          "=r"(p17), "=r"(p18), "=r"(p19), "=r"(p20), "=r"(p21), "=r"(p22), "=r"(p23), "=r"(p24),
          "=r"(p25), "=r"(p26), "=r"(p27), "=r"(p28), "=r"(p29), "=r"(p30), "=r"(p31)
        : "r"(payload_type), "l"(accel.handle), "f"(ox), "f"(oy), "f"(oz), "f"(dx), "f"(dy), "f"(dz), "f"(t_min),
          "f"(t_max), "f"(time), "r"(mask & 0xffu), "r"(flags), "r"(sbt_offset), "r"(0u),
          "r"(0u), "r"(reg_count), "r"(r0), "r"(r1), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u),
          "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u), "r"(u));
}

template<lc_uint flags>
[[nodiscard]] inline auto lc_accel_trace_closest_impl(LCAccel accel, LCRay ray, lc_float time, lc_uint mask) noexcept {
    // traverse
    lc_ray_traverse<LC_PAYLOAD_TYPE_RAY_TRACE, 0u>(flags, accel, ray, time, mask);
    // decode the hit
    auto hit = [] {
        auto inst = lc_hit_object_instance_index();
        auto prim = lc_hit_object_primitive_index();
#ifdef LUISA_ENABLE_OPTIX_CURVE
        auto hit_kind = lc_hit_object_hit_kind();
        auto bary = hit_kind == LC_HIT_KIND_TRIANGLE_FRONT_FACE ||
                            hit_kind == LC_HIT_KIND_TRIANGLE_BACK_FACE ?
                        lc_hit_object_triangle_bary() :
                        lc_make_float2(lc_hit_object_curve_parameter(), -1.f);
#else
        auto bary = lc_hit_object_triangle_bary();
#endif
        auto t = lc_hit_object_ray_t_max();
        return LCTriangleHit{inst, prim, bary, t};
    }();
    hit.m0 = lc_hit_object_is_hit() ? hit.m0 : ~0u;
    lc_hit_object_reset();
    return hit;
}

template<lc_uint flags>
[[nodiscard]] inline auto lc_accel_trace_any_impl(LCAccel accel, LCRay ray, lc_float time, lc_uint mask) noexcept {
    // traverse
    lc_ray_traverse<LC_PAYLOAD_TYPE_RAY_TRACE, 0u>(flags, accel, ray, time, mask);
    // decode if hit
    auto is_hit = lc_hit_object_is_hit();
    lc_hit_object_reset();
    return is_hit;
}

[[nodiscard]] inline auto lc_accel_trace_closest(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT |
                           LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return lc_accel_trace_closest_impl<flags>(accel, ray, 0.f, mask);
}

[[nodiscard]] inline auto lc_accel_trace_any(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT |
                           LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                           LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return lc_accel_trace_any_impl<flags>(accel, ray, 0.f, mask);
}

[[nodiscard]] inline auto lc_accel_trace_closest_motion_blur(LCAccel accel, LCRay ray, lc_float time, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT |
                           LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return lc_accel_trace_closest_impl<flags>(accel, ray, time, mask);
}

[[nodiscard]] inline auto lc_accel_trace_any_motion_blur(LCAccel accel, LCRay ray, lc_float time, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_ANYHIT |
                           LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                           LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return lc_accel_trace_any_impl<flags>(accel, ray, time, mask);
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

struct LCRayQuery {
    LCAccel accel;
    LCRay ray;
    lc_float time;
    lc_uint mask;
    lc_uint flags;
    LCCommittedHit hit;
};

using LCRayQueryAll = LCRayQuery;
using LCRayQueryAny = LCRayQuery;

[[nodiscard]] inline auto lc_ray_query_decode_hit() noexcept {
    auto hit = [] {// found closest hit
        auto inst = lc_hit_object_instance_index();
        auto prim = lc_hit_object_primitive_index();
        auto hit_kind = lc_hit_object_hit_kind();
#ifdef LUISA_ENABLE_OPTIX_CURVE
        auto bary = hit_kind == LC_HIT_KIND_TRIANGLE_FRONT_FACE ||
                            hit_kind == LC_HIT_KIND_TRIANGLE_BACK_FACE ?
                        lc_hit_object_triangle_bary() :
                        lc_make_float2(lc_hit_object_curve_parameter(), -1.f);
#else
        auto bary = lc_hit_object_triangle_bary();
#endif
        auto kind = hit_kind > 127u ?
                        static_cast<lc_uint>(LCHitType::BUILTIN) :
                        static_cast<lc_uint>(LCHitType::PROCEDURAL);
        auto dist = lc_hit_object_ray_t_max();
        return LCCommittedHit{inst, prim, bary, kind, dist};
    }();
    auto is_hit = lc_hit_object_is_hit();
    lc_hit_object_reset();
    hit.m0 = is_hit ? hit.m0 : ~0u;
    hit.m3 = is_hit ? hit.m3 : static_cast<lc_uint>(LCHitType::MISS);
    return hit;
}

inline void lc_ray_query_trace(LCRayQuery &q, lc_uint impl_tag, void *ctx) noexcept {
    auto p_ctx = reinterpret_cast<lc_ulong>(ctx);
    auto r0 = (impl_tag << 24u) | (static_cast<lc_uint>(p_ctx >> 32u) & 0xffffffu);
    auto r1 = static_cast<lc_uint>(p_ctx);
    // traverse
    lc_ray_traverse<LC_PAYLOAD_TYPE_RAY_QUERY, 5u, 2u>(q.flags, q.accel, q.ray, q.time, q.mask, r0, r1);
    q.hit = lc_ray_query_decode_hit();
}

[[nodiscard]] inline auto lc_accel_query_all(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return LCRayQueryAll{accel, ray, 0.f, mask, flags, LCCommittedHit{}};
}

[[nodiscard]] inline auto lc_accel_query_any(LCAccel accel, LCRay ray, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                           LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return LCRayQueryAny{accel, ray, 0.f, mask, flags, LCCommittedHit{}};
}

[[nodiscard]] inline auto lc_accel_query_all_motion_blur(LCAccel accel, LCRay ray, lc_float time, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return LCRayQueryAll{accel, ray, time, mask, flags, LCCommittedHit{}};
}

[[nodiscard]] inline auto lc_accel_query_any_motion_blur(LCAccel accel, LCRay ray, lc_float time, lc_uint mask) noexcept {
    constexpr auto flags = LC_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                           LC_RAY_FLAG_DISABLE_CLOSESTHIT;
    return LCRayQueryAny{accel, ray, time, mask, flags, LCCommittedHit{}};
}

[[nodiscard]] inline auto lc_ray_query_committed_hit(LCRayQuery q) noexcept {
    return q.hit;
}

[[nodiscard]] inline auto lc_ray_query_triangle_candidate() noexcept {
    auto inst = lc_get_instance_index();
    auto prim = lc_get_primitive_index();
#ifdef LUISA_ENABLE_OPTIX_CURVE
    auto kind = lc_get_hit_kind();
    auto bary = kind == LC_HIT_KIND_TRIANGLE_FRONT_FACE ||
                        kind == LC_HIT_KIND_TRIANGLE_BACK_FACE ?
                    lc_get_bary_coords() :
                    lc_make_float2(lc_get_curve_parameter(), -1.f);
#else
    auto bary = lc_get_bary_coords();
#endif
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
    auto query_id_and_p_ctx_hi = lc_get_payload<0u>();
    auto query_id = static_cast<lc_uint>(query_id_and_p_ctx_hi >> 24u);
    auto p_ctx_hi = static_cast<lc_uint>(query_id_and_p_ctx_hi & 0xffffffu);
    auto p_ctx_lo = lc_get_payload<1u>();
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
        default: lc_unreachable(__FILE__, __LINE__);
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

extern "C" __global__ void __anyhit__ray_query() {
#if LUISA_RAY_QUERY_IMPL_COUNT > 0
    lc_set_payload_types(LC_PAYLOAD_TYPE_RAY_QUERY);
    auto hit_kind = lc_get_hit_kind();
    auto should_terminate = false;
    if (hit_kind > 127u) {// triangle
        auto query_id_and_p_ctx_hi = lc_get_payload<0u>();
        auto query_id = static_cast<lc_uint>(query_id_and_p_ctx_hi >> 24u);
        auto p_ctx_hi = static_cast<lc_uint>(query_id_and_p_ctx_hi & 0xffffffu);
        auto p_ctx_lo = lc_get_payload<1u>();
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
            default: lc_unreachable(__FILE__, __LINE__);
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

#endif

#else

#define lc_dispatch_size() lc_make_uint3(params.ls_kid)
#define lc_kernel_id() static_cast<lc_uint>(params.ls_kid.w)

inline void lc_shader_execution_reorder(lc_uint hint, lc_uint hint_bits) noexcept {
    // do nothing since SER is not supported in plain CUDA
}

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

template<typename T>
[[nodiscard]] __device__ inline T lc_byte_buffer_read(LCBuffer<const lc_ubyte> buffer, lc_ulong offset) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
    auto address = reinterpret_cast<lc_ulong>(buffer.ptr + offset);
#ifdef LUISA_DEBUG
    lc_check_in_bounds(offset + sizeof(T), lc_buffer_size(buffer) + 1u);
    lc_assert(address % alignof(T) == 0u && "unaligned access");
#endif
    return *reinterpret_cast<T *>(address);
}

template<typename T>
__device__ inline void lc_byte_buffer_write(LCBuffer<lc_ubyte> buffer, lc_ulong offset, T value) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
    auto address = reinterpret_cast<lc_ulong>(buffer.ptr + offset);
#ifdef LUISA_DEBUG
    lc_check_in_bounds(offset + sizeof(T), lc_buffer_size(buffer) + 1u);
    lc_assert(address % alignof(T) == 0u && "unaligned access");
#endif
    *reinterpret_cast<T *>(address) = value;
}

[[nodiscard]] __device__ inline auto lc_byte_buffer_size(LCBuffer<const lc_byte> buffer) noexcept {
    return lc_buffer_size(buffer);
}

// warp intrinsics
[[nodiscard]] __device__ inline auto lc_warp_lane_id() noexcept {
    lc_uint ret;
    asm("mov.u32 %0, %laneid;"
        : "=r"(ret));
    return ret;
}

[[nodiscard]] __device__ constexpr auto lc_warp_size() noexcept {
    return static_cast<lc_uint>(warpSize);
}

#define LC_WARP_FULL_MASK 0xffff'ffffu
#define LC_WARP_ACTIVE_MASK __activemask()

[[nodiscard]] __device__ inline auto lc_warp_first_active_lane() noexcept {
    return __ffs(LC_WARP_ACTIVE_MASK) - 1u;
}

[[nodiscard]] __device__ inline auto lc_warp_is_first_active_lane() noexcept {
    return lc_warp_first_active_lane() == lc_warp_lane_id();
}

#if __CUDA_ARCH__ >= 700
[[nodiscard]] __device__ inline auto __match_all_sync(unsigned int mask, lc_half x, int *pred) noexcept {
    return __match_all_sync(mask, static_cast<float>(x), pred);
}
#define LC_WARP_ALL_EQ_SCALAR(T)                                                  \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T x) noexcept { \
        auto mask = LC_WARP_ACTIVE_MASK;                                          \
        auto pred = 0;                                                            \
        __match_all_sync(mask, x, &pred);                                         \
        return pred != 0;                                                         \
    }
#else
#define LC_WARP_ALL_EQ_SCALAR(T)                                                  \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T x) noexcept { \
        auto mask = LC_WARP_ACTIVE_MASK;                                          \
        auto first = __ffs(mask) - 1u;                                            \
        auto x0 = __shfl_sync(mask, x, first);                                    \
        return static_cast<bool>(__all_sync(mask, x == x0));                      \
    }
#endif

#define LC_WARP_ALL_EQ_VECTOR2(T)                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T##2 v) noexcept { \
        return lc_make_bool2(lc_warp_active_all_equal(v.x),                          \
                             lc_warp_active_all_equal(v.y));                         \
    }

#define LC_WARP_ALL_EQ_VECTOR3(T)                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T##3 v) noexcept { \
        return lc_make_bool3(lc_warp_active_all_equal(v.x),                          \
                             lc_warp_active_all_equal(v.y),                          \
                             lc_warp_active_all_equal(v.z));                         \
    }

#define LC_WARP_ALL_EQ_VECTOR4(T)                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T##4 v) noexcept { \
        return lc_make_bool4(lc_warp_active_all_equal(v.x),                          \
                             lc_warp_active_all_equal(v.y),                          \
                             lc_warp_active_all_equal(v.z),                          \
                             lc_warp_active_all_equal(v.w));                         \
    }

#define LC_WARP_ALL_EQ(T)     \
    LC_WARP_ALL_EQ_SCALAR(T)  \
    LC_WARP_ALL_EQ_VECTOR2(T) \
    LC_WARP_ALL_EQ_VECTOR3(T) \
    LC_WARP_ALL_EQ_VECTOR4(T)

LC_WARP_ALL_EQ(lc_bool)
LC_WARP_ALL_EQ(lc_short)
LC_WARP_ALL_EQ(lc_ushort)
LC_WARP_ALL_EQ(lc_int)
LC_WARP_ALL_EQ(lc_uint)
LC_WARP_ALL_EQ(lc_long)
LC_WARP_ALL_EQ(lc_ulong)
LC_WARP_ALL_EQ(lc_float)
LC_WARP_ALL_EQ(lc_half)
//LC_WARP_ALL_EQ(lc_double)// TODO

#undef LC_WARP_ALL_EQ_SCALAR
#undef LC_WARP_ALL_EQ_VECTOR2
#undef LC_WARP_ALL_EQ_VECTOR3
#undef LC_WARP_ALL_EQ_VECTOR4
#undef LC_WARP_ALL_EQ

template<typename T, typename F>
[[nodiscard]] __device__ inline auto lc_warp_active_reduce_impl(T x, F f) noexcept {
    auto mask = LC_WARP_ACTIVE_MASK;
    auto lane = lc_warp_lane_id();
    if (auto y = __shfl_xor_sync(mask, x, 0x10u); mask & (1u << (lane ^ 0x10u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x08u); mask & (1u << (lane ^ 0x08u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x04u); mask & (1u << (lane ^ 0x04u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x02u); mask & (1u << (lane ^ 0x02u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x01u); mask & (1u << (lane ^ 0x01u))) { x = f(x, y); }
    return x;
}

template<typename T>
[[nodiscard]] __device__ constexpr T lc_bit_and(T x, T y) noexcept { return x & y; }

template<typename T>
[[nodiscard]] __device__ constexpr T lc_bit_or(T x, T y) noexcept { return x | y; }

template<typename T>
[[nodiscard]] __device__ constexpr T lc_bit_xor(T x, T y) noexcept { return x ^ y; }

#define LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(op, T)                                     \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T x) noexcept { \
        return static_cast<lc_##T>(lc_warp_active_reduce_impl(                        \
            x, [](lc_##T a, lc_##T b) noexcept { return lc_bit_##op(a, b); }));       \
    }

#if __CUDA_ARCH__ >= 800
#define LC_WARP_REDUCE_BIT_SCALAR(op, T)                                              \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T x) noexcept { \
        return static_cast<lc_##T>(__reduce_##op##_sync(LC_WARP_ACTIVE_MASK,          \
                                                        static_cast<lc_uint>(x)));    \
    }
#else
#define LC_WARP_REDUCE_BIT_SCALAR(op, T) LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(op, T)
#endif

LC_WARP_REDUCE_BIT_SCALAR(and, uint)
LC_WARP_REDUCE_BIT_SCALAR(or, uint)
LC_WARP_REDUCE_BIT_SCALAR(xor, uint)
LC_WARP_REDUCE_BIT_SCALAR(and, int)
LC_WARP_REDUCE_BIT_SCALAR(or, int)
LC_WARP_REDUCE_BIT_SCALAR(xor, int)

LC_WARP_REDUCE_BIT_SCALAR(and, ushort)
LC_WARP_REDUCE_BIT_SCALAR(or, ushort)
LC_WARP_REDUCE_BIT_SCALAR(xor, ushort)
LC_WARP_REDUCE_BIT_SCALAR(and, short)
LC_WARP_REDUCE_BIT_SCALAR(or, short)
LC_WARP_REDUCE_BIT_SCALAR(xor, short)

LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(and, ulong)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(or, ulong)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(xor, ulong)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(and, long)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(or, long)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(xor, long)

#undef LC_WARP_REDUCE_BIT_SCALAR_FALLBACK
#undef LC_WARP_REDUCE_BIT_SCALAR

#define LC_WARP_REDUCE_BIT_VECTOR(op, T)                                                 \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T##2 v) noexcept { \
        return lc_make_##T##2(lc_warp_active_bit_##op(v.x),                              \
                              lc_warp_active_bit_##op(v.y));                             \
    }                                                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T##3 v) noexcept { \
        return lc_make_##T##3(lc_warp_active_bit_##op(v.x),                              \
                              lc_warp_active_bit_##op(v.y),                              \
                              lc_warp_active_bit_##op(v.z));                             \
    }                                                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T##4 v) noexcept { \
        return lc_make_##T##4(lc_warp_active_bit_##op(v.x),                              \
                              lc_warp_active_bit_##op(v.y),                              \
                              lc_warp_active_bit_##op(v.z),                              \
                              lc_warp_active_bit_##op(v.w));                             \
    }

LC_WARP_REDUCE_BIT_VECTOR(and, uint)
LC_WARP_REDUCE_BIT_VECTOR(or, uint)
LC_WARP_REDUCE_BIT_VECTOR(xor, uint)
LC_WARP_REDUCE_BIT_VECTOR(and, int)
LC_WARP_REDUCE_BIT_VECTOR(or, int)
LC_WARP_REDUCE_BIT_VECTOR(xor, int)

#undef LC_WARP_REDUCE_BIT_VECTOR

[[nodiscard]] __device__ inline auto lc_warp_active_bit_mask(bool pred) noexcept {
    return lc_make_uint4(__ballot_sync(LC_WARP_ACTIVE_MASK, pred), 0u, 0u, 0u);
}

[[nodiscard]] __device__ inline auto lc_warp_active_count_bits(bool pred) noexcept {
    return lc_popcount(__ballot_sync(LC_WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] __device__ inline auto lc_warp_active_all(bool pred) noexcept {
    return static_cast<lc_bool>(__all_sync(LC_WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] __device__ inline auto lc_warp_active_any(bool pred) noexcept {
    return static_cast<lc_bool>(__any_sync(LC_WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] __device__ inline auto lc_warp_prefix_mask() noexcept {
    lc_uint ret;
    asm("mov.u32 %0, %lanemask_lt;"
        : "=r"(ret));
    return ret;
}

[[nodiscard]] __device__ inline auto lc_warp_prefix_count_bits(bool pred) noexcept {
    return lc_popcount(__ballot_sync(LC_WARP_ACTIVE_MASK, pred) & lc_warp_prefix_mask());
}

#define LC_WARP_READ_LANE_SCALAR(T)                                                        \
    [[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_##T x, lc_uint i) noexcept { \
        return static_cast<lc_##T>(__shfl_sync(LC_WARP_ACTIVE_MASK, x, i));                \
    }

#define LC_WARP_READ_LANE_VECTOR2(T)                                                          \
    [[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_##T##2 v, lc_uint i) noexcept { \
        return lc_make_##T##2(lc_warp_read_lane(v.x, i),                                      \
                              lc_warp_read_lane(v.y, i));                                     \
    }

#define LC_WARP_READ_LANE_VECTOR3(T)                                                          \
    [[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_##T##3 v, lc_uint i) noexcept { \
        return lc_make_##T##3(lc_warp_read_lane(v.x, i),                                      \
                              lc_warp_read_lane(v.y, i),                                      \
                              lc_warp_read_lane(v.z, i));                                     \
    }

#define LC_WARP_READ_LANE_VECTOR4(T)                                                          \
    [[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_##T##4 v, lc_uint i) noexcept { \
        return lc_make_##T##4(lc_warp_read_lane(v.x, i),                                      \
                              lc_warp_read_lane(v.y, i),                                      \
                              lc_warp_read_lane(v.z, i),                                      \
                              lc_warp_read_lane(v.w, i));                                     \
    }

#define LC_WARP_READ_LANE(T)     \
    LC_WARP_READ_LANE_SCALAR(T)  \
    LC_WARP_READ_LANE_VECTOR2(T) \
    LC_WARP_READ_LANE_VECTOR3(T) \
    LC_WARP_READ_LANE_VECTOR4(T)

LC_WARP_READ_LANE(bool)
LC_WARP_READ_LANE(short)
LC_WARP_READ_LANE(ushort)
LC_WARP_READ_LANE(int)
LC_WARP_READ_LANE(uint)
LC_WARP_READ_LANE(long)
LC_WARP_READ_LANE(ulong)
LC_WARP_READ_LANE(float)
LC_WARP_READ_LANE(half)
//LC_WARP_READ_LANE(double)// TODO

#undef LC_WARP_READ_LANE_SCALAR
#undef LC_WARP_READ_LANE_VECTOR2
#undef LC_WARP_READ_LANE_VECTOR3
#undef LC_WARP_READ_LANE_VECTOR4
#undef LC_WARP_READ_LANE

[[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_float2x2 m, lc_uint i) noexcept {
    return lc_make_float2x2(lc_warp_read_lane(m[0], i),
                            lc_warp_read_lane(m[1], i));
}

[[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_float3x3 m, lc_uint i) noexcept {
    return lc_make_float3x3(lc_warp_read_lane(m[0], i),
                            lc_warp_read_lane(m[1], i),
                            lc_warp_read_lane(m[2], i));
}

[[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_float4x4 m, lc_uint i) noexcept {
    return lc_make_float4x4(lc_warp_read_lane(m[0], i),
                            lc_warp_read_lane(m[1], i),
                            lc_warp_read_lane(m[2], i),
                            lc_warp_read_lane(m[3], i));
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_read_first_active_lane(T x) noexcept {
    return lc_warp_read_lane(x, lc_warp_first_active_lane());
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_active_min_impl(T x) noexcept {
    return lc_warp_active_reduce_impl(x, [](T a, T b) noexcept { return lc_min(a, b); });
}
template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_active_max_impl(T x) noexcept {
    return lc_warp_active_reduce_impl(x, [](T a, T b) noexcept { return lc_max(a, b); });
}
template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_active_sum_impl(T x) noexcept {
    return lc_warp_active_reduce_impl(x, [](T a, T b) noexcept { return a + b; });
}
template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_active_product_impl(T x) noexcept {
    return lc_warp_active_reduce_impl(x, [](T a, T b) noexcept { return a * b; });
}

#define LC_WARP_ACTIVE_REDUCE_SCALAR(op, T)                                       \
    [[nodiscard]] __device__ inline auto lc_warp_active_##op(lc_##T x) noexcept { \
        return lc_warp_active_##op##_impl<lc_##T>(x);                             \
    }

#if __CUDA_ARCH__ >= 800
[[nodiscard]] __device__ inline auto lc_warp_active_min(lc_uint x) noexcept {
    return __reduce_min_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_max(lc_uint x) noexcept {
    return __reduce_max_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_sum(lc_uint x) noexcept {
    return __reduce_add_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_min(lc_int x) noexcept {
    return __reduce_min_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_max(lc_int x) noexcept {
    return __reduce_max_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_sum(lc_int x) noexcept {
    return __reduce_add_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_min(lc_ushort x) noexcept {
    return static_cast<lc_ushort>(__reduce_min_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_uint>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_max(lc_ushort x) noexcept {
    return static_cast<lc_ushort>(__reduce_max_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_uint>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_sum(lc_ushort x) noexcept {
    return static_cast<lc_ushort>(__reduce_add_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_uint>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_min(lc_short x) noexcept {
    return static_cast<lc_short>(__reduce_min_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_int>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_max(lc_short x) noexcept {
    return static_cast<lc_short>(__reduce_max_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_int>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_sum(lc_short x) noexcept {
    return static_cast<lc_short>(__reduce_add_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_int>(x)));
}
#else
LC_WARP_ACTIVE_REDUCE_SCALAR(min, uint)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, uint)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, uint)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, int)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, int)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, int)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, ushort)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, ushort)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, ushort)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, short)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, short)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, short)
#endif

LC_WARP_ACTIVE_REDUCE_SCALAR(product, uint)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, int)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, ushort)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, short)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, ulong)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, ulong)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, ulong)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, ulong)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, long)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, long)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, long)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, long)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, float)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, float)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, float)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, float)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, half)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, half)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, half)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, half)
// TODO: double
// LC_WARP_ACTIVE_REDUCE_SCALAR(min, double)
// LC_WARP_ACTIVE_REDUCE_SCALAR(max, double)
// LC_WARP_ACTIVE_REDUCE_SCALAR(sum, double)
// LC_WARP_ACTIVE_REDUCE_SCALAR(product, double)

#undef LC_WARP_ACTIVE_REDUCE_SCALAR

#define LC_WARP_ACTIVE_REDUCE_VECTOR2(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_active_##op(lc_##T##2 v) noexcept { \
        return lc_make_##T##2(lc_warp_active_##op(v.x),                              \
                              lc_warp_active_##op(v.y));                             \
    }

#define LC_WARP_ACTIVE_REDUCE_VECTOR3(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_active_##op(lc_##T##3 v) noexcept { \
        return lc_make_##T##3(lc_warp_active_##op(v.x),                              \
                              lc_warp_active_##op(v.y),                              \
                              lc_warp_active_##op(v.z));                             \
    }

#define LC_WARP_ACTIVE_REDUCE_VECTOR4(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_active_##op(lc_##T##4 v) noexcept { \
        return lc_make_##T##4(lc_warp_active_##op(v.x),                              \
                              lc_warp_active_##op(v.y),                              \
                              lc_warp_active_##op(v.z),                              \
                              lc_warp_active_##op(v.w));                             \
    }

#define LC_WARP_ACTIVE_REDUCE(T)              \
    LC_WARP_ACTIVE_REDUCE_VECTOR2(min, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR3(min, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR4(min, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR2(max, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR3(max, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR4(max, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR2(sum, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR3(sum, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR4(sum, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR2(product, T) \
    LC_WARP_ACTIVE_REDUCE_VECTOR3(product, T) \
    LC_WARP_ACTIVE_REDUCE_VECTOR4(product, T)

LC_WARP_ACTIVE_REDUCE(uint)
LC_WARP_ACTIVE_REDUCE(int)
LC_WARP_ACTIVE_REDUCE(ushort)
LC_WARP_ACTIVE_REDUCE(short)
LC_WARP_ACTIVE_REDUCE(ulong)
LC_WARP_ACTIVE_REDUCE(long)
LC_WARP_ACTIVE_REDUCE(float)
LC_WARP_ACTIVE_REDUCE(half)
//LC_WARP_ACTIVE_REDUCE(double)// TODO

#undef LC_WARP_ACTIVE_REDUCE_VECTOR2
#undef LC_WARP_ACTIVE_REDUCE_VECTOR3
#undef LC_WARP_ACTIVE_REDUCE_VECTOR4
#undef LC_WARP_ACTIVE_REDUCE

[[nodiscard]] __device__ inline auto lc_warp_prev_active_lane() noexcept {
    auto mask = 0u;
    asm("mov.u32 %0, %lanemask_lt;"
        : "=r"(mask));
    return (lc_warp_size() - 1u) - __clz(LC_WARP_ACTIVE_MASK & mask);
}

template<typename T, typename F>
[[nodiscard]] __device__ inline auto lc_warp_prefix_reduce_impl(T x, T unit, F f) noexcept {
    auto mask = LC_WARP_ACTIVE_MASK;
    auto lane = lc_warp_lane_id();
    x = __shfl_sync(mask, x, lc_warp_prev_active_lane());
    x = (lane == lc_warp_first_active_lane()) ? unit : x;
    if (auto y = __shfl_up_sync(mask, x, 0x01u); lane >= 0x01u && (mask & (1u << (lane - 0x01u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x02u); lane >= 0x02u && (mask & (1u << (lane - 0x02u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x04u); lane >= 0x04u && (mask & (1u << (lane - 0x04u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x08u); lane >= 0x08u && (mask & (1u << (lane - 0x08u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x10u); lane >= 0x10u && (mask & (1u << (lane - 0x10u)))) { x = f(x, y); }
    return x;
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_prefix_sum_impl(T x) noexcept {
    return lc_warp_prefix_reduce_impl(x, static_cast<T>(0), [](T a, T b) noexcept { return a + b; });
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_prefix_product_impl(T x) noexcept {
    return lc_warp_prefix_reduce_impl(x, static_cast<T>(1), [](T a, T b) noexcept { return a * b; });
}

#define LC_WARP_PREFIX_REDUCE_SCALAR(op, T)                                       \
    [[nodiscard]] __device__ inline auto lc_warp_prefix_##op(lc_##T x) noexcept { \
        return lc_warp_prefix_##op##_impl<lc_##T>(x);                             \
    }

#define LC_WARP_PREFIX_REDUCE_VECTOR2(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_prefix_##op(lc_##T##2 v) noexcept { \
        return lc_make_##T##2(lc_warp_prefix_##op(v.x),                              \
                              lc_warp_prefix_##op(v.y));                             \
    }

#define LC_WARP_PREFIX_REDUCE_VECTOR3(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_prefix_##op(lc_##T##3 v) noexcept { \
        return lc_make_##T##3(lc_warp_prefix_##op(v.x),                              \
                              lc_warp_prefix_##op(v.y),                              \
                              lc_warp_prefix_##op(v.z));                             \
    }

#define LC_WARP_PREFIX_REDUCE_VECTOR4(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_prefix_##op(lc_##T##4 v) noexcept { \
        return lc_make_##T##4(lc_warp_prefix_##op(v.x),                              \
                              lc_warp_prefix_##op(v.y),                              \
                              lc_warp_prefix_##op(v.z),                              \
                              lc_warp_prefix_##op(v.w));                             \
    }

#define LC_WARP_PREFIX_REDUCE(T)              \
    LC_WARP_PREFIX_REDUCE_SCALAR(sum, T)      \
    LC_WARP_PREFIX_REDUCE_SCALAR(product, T)  \
    LC_WARP_PREFIX_REDUCE_VECTOR2(sum, T)     \
    LC_WARP_PREFIX_REDUCE_VECTOR2(product, T) \
    LC_WARP_PREFIX_REDUCE_VECTOR3(sum, T)     \
    LC_WARP_PREFIX_REDUCE_VECTOR3(product, T) \
    LC_WARP_PREFIX_REDUCE_VECTOR4(sum, T)     \
    LC_WARP_PREFIX_REDUCE_VECTOR4(product, T)

LC_WARP_PREFIX_REDUCE(uint)
LC_WARP_PREFIX_REDUCE(int)
LC_WARP_PREFIX_REDUCE(ushort)
LC_WARP_PREFIX_REDUCE(short)
LC_WARP_PREFIX_REDUCE(ulong)
LC_WARP_PREFIX_REDUCE(long)
LC_WARP_PREFIX_REDUCE(float)
LC_WARP_PREFIX_REDUCE(half)
//LC_WARP_PREFIX_REDUCE(double)// TODO

#undef LC_WARP_PREFIX_REDUCE_SCALAR
#undef LC_WARP_PREFIX_REDUCE_VECTOR2
#undef LC_WARP_PREFIX_REDUCE_VECTOR3
#undef LC_WARP_PREFIX_REDUCE_VECTOR4
#undef LC_WARP_PREFIX_REDUCE

struct LCPrintBufferContent {
    lc_ulong size;
    lc_ubyte data[1];
};

struct LCPrintBuffer {
    lc_ulong capacity;
    LCPrintBufferContent *__restrict__ content;
};

#ifdef LUISA_ENABLE_OPTIX
#define LC_PRINT_BUFFER (params.print_buffer)
#else
#define LC_PRINT_BUFFER (print_buffer)
#endif

template<typename T>
__device__ inline void lc_print_impl(LCPrintBuffer buffer, T value) noexcept {
    if (buffer.capacity == 0u || buffer.content == nullptr) { return; }
    auto offset = atomicAdd(&buffer.content->size, sizeof(T));
    if (offset + sizeof(T) <= buffer.capacity) {
        auto ptr = buffer.content->data + offset;
        memcpy(ptr, &value, sizeof(T));
    }
}

#define LC_DECODE_STRING_FROM_ID(str_id) ((const char *)&(lc_string_data[lc_string_offsets[str_id]]))
