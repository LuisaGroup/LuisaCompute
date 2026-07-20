// Comprehensive test for 8-bit integer types (byte/int8, ubyte/uint8)
// Covers buffer R/W, unary ops, binary ops, and casts.

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>
#ifdef _WIN32
// see note in DX backend `Device.cpp` and `src/tests/integration/runtime/test_work_graph.cpp`
extern "C" __declspec(dllexport) const uint32_t D3D12SDKVersion = 619;
extern "C" __declspec(dllexport) const char *D3D12SDKPath = ".\\D3D12\\";
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Number of test slots per category
constexpr uint N = 64u;

enum Int8Slot : uint {
    I8_READ = 0,
    I8_WRITE,
    I8_NEG,
    I8_BIT_NOT,
    I8_ABS,
    I8_ADD,
    I8_SUB,
    I8_MUL,
    I8_DIV,
    I8_MOD,
    I8_MIN,
    I8_MAX,
    I8_AND,
    I8_OR,
    I8_XOR,
    I8_SHL,
    I8_SHR,
    I8_CAST_UBYTE,
    I8_CAST_INT,
    I8_CAST_BOOL_TRUE,
    I8_CAST_BOOL_FALSE,
    I8_SELECT,
    I8_EQ,
    I8_NE,
    I8_LT,
    I8_LE,
    I8_GT,
    I8_GE,
    I8_COUNT
};

enum UInt8Slot : uint {
    U8_READ = 0,
    U8_WRITE,
    U8_BIT_NOT,
    U8_CLZ,
    U8_CTZ,
    U8_POPCOUNT,
    U8_REVERSE,
    U8_ADD,
    U8_SUB,
    U8_MUL,
    U8_DIV,
    U8_MOD,
    U8_MIN,
    U8_MAX,
    U8_AND,
    U8_OR,
    U8_XOR,
    U8_SHL,
    U8_SHR,
    U8_CAST_BYTE,
    U8_CAST_INT,
    U8_CAST_BOOL_TRUE,
    U8_CAST_BOOL_FALSE,
    U8_SELECT,
    U8_EQ,
    U8_NE,
    U8_LT,
    U8_LE,
    U8_GT,
    U8_GE,
    U8_COUNT
};

static_assert(static_cast<uint>(I8_COUNT) <= N, "I8_COUNT exceeds N");
static_assert(static_cast<uint>(U8_COUNT) <= N, "U8_COUNT exceeds N");

int test_byte8(Device &device) {
    Stream stream = device.create_stream();

    Buffer<int8_t> i8_in = device.create_buffer<int8_t>(N);
    Buffer<int8_t> i8_out = device.create_buffer<int8_t>(N);
    Buffer<int> i8_wide_out = device.create_buffer<int>(N);
    Buffer<uint8_t> u8_in = device.create_buffer<uint8_t>(N);
    Buffer<uint8_t> u8_out = device.create_buffer<uint8_t>(N);
    Buffer<int> u8_wide_out = device.create_buffer<int>(N);
    Buffer<uint> u8_bits_out = device.create_buffer<uint>(N);

    // Host data
    luisa::vector<int8_t> i8_host(N);
    luisa::vector<uint8_t> u8_host(N);
    for (auto i = 0u; i < N; ++i) {
        i8_host[i] = static_cast<int8_t>((i * 7 + 3) & 0xFF);
        u8_host[i] = static_cast<uint8_t>((i * 13 + 5) & 0xFF);
    }
    // Deliberate values keep the division/remainder and widening checks from
    // degenerating to zero, the dividend, or an 8-bit round trip.
    i8_host[I8_ABS] = static_cast<int8_t>(-37);
    i8_host[I8_DIV] = static_cast<int8_t>(-101);
    i8_host[I8_MOD] = static_cast<int8_t>(7);
    i8_host[I8_MOD + 1u] = static_cast<int8_t>(3);
    i8_host[I8_CAST_UBYTE] = static_cast<int8_t>(-5);
    i8_host[I8_CAST_INT] = static_cast<int8_t>(-101);
    u8_host[U8_CLZ] = static_cast<uint8_t>(0x10u);
    u8_host[U8_CTZ] = static_cast<uint8_t>(0x28u);
    u8_host[U8_POPCOUNT] = static_cast<uint8_t>(0xadu);
    u8_host[U8_REVERSE] = static_cast<uint8_t>(0x96u);
    u8_host[U8_DIV] = static_cast<uint8_t>(201u);
    u8_host[U8_MOD] = static_cast<uint8_t>(13u);
    u8_host[U8_MOD + 1u] = static_cast<uint8_t>(5u);
    u8_host[U8_CAST_BYTE] = static_cast<uint8_t>(250u);
    u8_host[U8_CAST_INT] = static_cast<uint8_t>(201u);
    stream << i8_in.copy_from(luisa::span{i8_host});
    stream << u8_in.copy_from(luisa::span{u8_host});
    stream << synchronize();

    // Kernel for int8 tests
    Kernel1D i8_kernel = [&](BufferVar<int8_t> in_buf, BufferVar<int8_t> out_buf, BufferVar<int> wide_out_buf) {
        auto idx = dispatch_id().x;
        $if (idx == static_cast<uint>(I8_READ)) {
            out_buf.write(idx, in_buf.read(idx));
        }
        $elif (idx == static_cast<uint>(I8_WRITE)) {
            Var<int8_t> v = 42;
            out_buf.write(idx, v);
        }
        $elif (idx == static_cast<uint>(I8_NEG)) {
            out_buf.write(idx, cast<int8_t>(-in_buf.read(idx)));
        }
        $elif (idx == static_cast<uint>(I8_BIT_NOT)) {
            out_buf.write(idx, cast<int8_t>(~in_buf.read(idx)));
        }
        $elif (idx == static_cast<uint>(I8_ABS)) {
            out_buf.write(idx, cast<int8_t>(abs(in_buf.read(idx))));
        }
        $elif (idx == static_cast<uint>(I8_ADD)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) + in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_SUB)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) - in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_MUL)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) * in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_DIV)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) / in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_MOD)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) % in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_MIN)) {
            out_buf.write(idx, cast<int8_t>(min(in_buf.read(idx), in_buf.read(idx + 1u))));
        }
        $elif (idx == static_cast<uint>(I8_MAX)) {
            out_buf.write(idx, cast<int8_t>(max(in_buf.read(idx), in_buf.read(idx + 1u))));
        }
        $elif (idx == static_cast<uint>(I8_AND)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) & in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_OR)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) | in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_XOR)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) ^ in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_SHL)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) << 2));
        }
        $elif (idx == static_cast<uint>(I8_SHR)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) >> 2));
        }
        $elif (idx == static_cast<uint>(I8_CAST_UBYTE)) {
            wide_out_buf.write(idx, cast<int>(cast<uint8_t>(in_buf.read(idx))));
        }
        $elif (idx == static_cast<uint>(I8_CAST_INT)) {
            wide_out_buf.write(idx, cast<int>(in_buf.read(idx)));
        }
        $elif (idx == static_cast<uint>(I8_CAST_BOOL_TRUE)) {
            out_buf.write(idx, cast<int8_t>(cast<bool>(in_buf.read(idx))));
        }
        $elif (idx == static_cast<uint>(I8_CAST_BOOL_FALSE)) {
            out_buf.write(idx, cast<int8_t>(cast<bool>(static_cast<int8_t>(0))));
        }
        $elif (idx == static_cast<uint>(I8_SELECT)) {
            out_buf.write(idx, ite(in_buf.read(idx) > static_cast<int8_t>(0), static_cast<int8_t>(1), static_cast<int8_t>(-1)));
        }
        $elif (idx == static_cast<uint>(I8_EQ)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) == in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_NE)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) != in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_LT)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) < in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_LE)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) <= in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_GT)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) > in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(I8_GE)) {
            out_buf.write(idx, cast<int8_t>(in_buf.read(idx) >= in_buf.read(idx + 1u)));
        };
    };

    // Kernel for uint8 tests
    Kernel1D u8_kernel = [&](BufferVar<uint8_t> in_buf, BufferVar<uint8_t> out_buf, BufferVar<int> wide_out_buf, BufferVar<uint> bits_out_buf) {
        auto idx = dispatch_id().x;
        $if (idx == static_cast<uint>(U8_READ)) {
            out_buf.write(idx, in_buf.read(idx));
        }
        $elif (idx == static_cast<uint>(U8_WRITE)) {
            Var<uint8_t> v = 42;
            out_buf.write(idx, v);
        }
        $elif (idx == static_cast<uint>(U8_BIT_NOT)) {
            out_buf.write(idx, cast<uint8_t>(~in_buf.read(idx)));
        }
        $elif (idx == static_cast<uint>(U8_CLZ)) {
            out_buf.write(idx, cast<uint8_t>(clz(in_buf.read(idx))));
        }
        $elif (idx == static_cast<uint>(U8_CTZ)) {
            out_buf.write(idx, cast<uint8_t>(ctz(in_buf.read(idx))));
        }
        $elif (idx == static_cast<uint>(U8_POPCOUNT)) {
            out_buf.write(idx, cast<uint8_t>(popcount(in_buf.read(idx))));
        }
        $elif (idx == static_cast<uint>(U8_REVERSE)) {
            bits_out_buf.write(idx, reverse(in_buf.read(idx)));
        }
        $elif (idx == static_cast<uint>(U8_ADD)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) + in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_SUB)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) - in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_MUL)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) * in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_DIV)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) / in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_MOD)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) % in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_MIN)) {
            out_buf.write(idx, cast<uint8_t>(min(in_buf.read(idx), in_buf.read(idx + 1u))));
        }
        $elif (idx == static_cast<uint>(U8_MAX)) {
            out_buf.write(idx, cast<uint8_t>(max(in_buf.read(idx), in_buf.read(idx + 1u))));
        }
        $elif (idx == static_cast<uint>(U8_AND)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) & in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_OR)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) | in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_XOR)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) ^ in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_SHL)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) << 2));
        }
        $elif (idx == static_cast<uint>(U8_SHR)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) >> 2));
        }
        $elif (idx == static_cast<uint>(U8_CAST_BYTE)) {
            wide_out_buf.write(idx, cast<int>(cast<int8_t>(in_buf.read(idx))));
        }
        $elif (idx == static_cast<uint>(U8_CAST_INT)) {
            wide_out_buf.write(idx, cast<int>(in_buf.read(idx)));
        }
        $elif (idx == static_cast<uint>(U8_CAST_BOOL_TRUE)) {
            out_buf.write(idx, cast<uint8_t>(cast<bool>(in_buf.read(idx))));
        }
        $elif (idx == static_cast<uint>(U8_CAST_BOOL_FALSE)) {
            out_buf.write(idx, cast<uint8_t>(cast<bool>(static_cast<uint8_t>(0))));
        }
        $elif (idx == static_cast<uint>(U8_SELECT)) {
            out_buf.write(idx, ite(in_buf.read(idx) > static_cast<uint8_t>(0), static_cast<uint8_t>(1), static_cast<uint8_t>(0)));
        }
        $elif (idx == static_cast<uint>(U8_EQ)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) == in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_NE)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) != in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_LT)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) < in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_LE)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) <= in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_GT)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) > in_buf.read(idx + 1u)));
        }
        $elif (idx == static_cast<uint>(U8_GE)) {
            out_buf.write(idx, cast<uint8_t>(in_buf.read(idx) >= in_buf.read(idx + 1u)));
        };
    };

    auto i8_shader = device.compile(i8_kernel);
    auto u8_shader = device.compile(u8_kernel);

    stream << i8_shader(i8_in, i8_out, i8_wide_out).dispatch(N);
    stream << u8_shader(u8_in, u8_out, u8_wide_out, u8_bits_out).dispatch(N);
    stream << synchronize();

    luisa::vector<int8_t> i8_result(N);
    luisa::vector<int> i8_wide_result(N);
    luisa::vector<uint8_t> u8_result(N);
    luisa::vector<int> u8_wide_result(N);
    luisa::vector<uint> u8_bits_result(N);
    stream << i8_out.copy_to(luisa::span{i8_result});
    stream << i8_wide_out.copy_to(luisa::span{i8_wide_result});
    stream << u8_out.copy_to(luisa::span{u8_result});
    stream << u8_wide_out.copy_to(luisa::span{u8_wide_result});
    stream << u8_bits_out.copy_to(luisa::span{u8_bits_result});
    stream << synchronize();

    // CPU reference checks for int8
    auto i8_ref = [&](uint idx) -> int8_t {
        auto a = i8_host[idx];
        auto b = i8_host[idx + 1u];
        switch (static_cast<Int8Slot>(idx)) {
            case I8_READ: return a;
            case I8_WRITE: return 42;
            case I8_NEG: return static_cast<int8_t>(-static_cast<int>(a));
            case I8_BIT_NOT: return static_cast<int8_t>(~static_cast<int>(a));
            case I8_ABS: return static_cast<int8_t>(std::abs(static_cast<int>(a)));
            case I8_ADD: return static_cast<int8_t>(static_cast<int>(a) + static_cast<int>(b));
            case I8_SUB: return static_cast<int8_t>(static_cast<int>(a) - static_cast<int>(b));
            case I8_MUL: return static_cast<int8_t>(static_cast<int>(a) * static_cast<int>(b));
            case I8_DIV: return b == 0 ? static_cast<int8_t>(0) : static_cast<int8_t>(static_cast<int>(a) / static_cast<int>(b));
            case I8_MOD: return b == 0 ? static_cast<int8_t>(0) : static_cast<int8_t>(static_cast<int>(a) % static_cast<int>(b));
            case I8_MIN: return static_cast<int8_t>(std::min(static_cast<int>(a), static_cast<int>(b)));
            case I8_MAX: return static_cast<int8_t>(std::max(static_cast<int>(a), static_cast<int>(b)));
            case I8_AND: return static_cast<int8_t>(static_cast<int>(a) & static_cast<int>(b));
            case I8_OR: return static_cast<int8_t>(static_cast<int>(a) | static_cast<int>(b));
            case I8_XOR: return static_cast<int8_t>(static_cast<int>(a) ^ static_cast<int>(b));
            case I8_SHL: return static_cast<int8_t>(static_cast<int>(a) << 2);
            case I8_SHR: return static_cast<int8_t>(static_cast<int>(a) >> 2);
            case I8_CAST_UBYTE: return a;
            case I8_CAST_INT: return a;
            case I8_CAST_BOOL_TRUE: return a != 0 ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
            case I8_CAST_BOOL_FALSE: return static_cast<int8_t>(0);
            case I8_SELECT: return static_cast<int>(a) > 0 ? static_cast<int8_t>(1) : static_cast<int8_t>(-1);
            case I8_EQ: return static_cast<int8_t>(a == b ? 1 : 0);
            case I8_NE: return static_cast<int8_t>(a != b ? 1 : 0);
            case I8_LT: return static_cast<int8_t>(static_cast<int>(a) < static_cast<int>(b) ? 1 : 0);
            case I8_LE: return static_cast<int8_t>(static_cast<int>(a) <= static_cast<int>(b) ? 1 : 0);
            case I8_GT: return static_cast<int8_t>(static_cast<int>(a) > static_cast<int>(b) ? 1 : 0);
            case I8_GE: return static_cast<int8_t>(static_cast<int>(a) >= static_cast<int>(b) ? 1 : 0);
            default: return static_cast<int8_t>(0);
        }
    };

    // CPU reference checks for uint8
    auto u8_ref = [&](uint idx) -> uint8_t {
        auto a = u8_host[idx];
        auto b = u8_host[idx + 1u];
        switch (static_cast<UInt8Slot>(idx)) {
            case U8_READ: return a;
            case U8_WRITE: return 42;
            case U8_BIT_NOT: return static_cast<uint8_t>(~static_cast<unsigned>(a));
            case U8_CLZ: {
                // DSL clz() returns uint (32-bit); compute 32-bit CLZ
                unsigned x = a;
                if (x == 0) return static_cast<uint8_t>(32);
                int n = 0;
                while ((x & 0x80000000u) == 0) {
                    x <<= 1;
                    ++n;
                }
                return static_cast<uint8_t>(n);
            }
            case U8_CTZ: {
                // DSL ctz() returns uint (32-bit); compute 32-bit CTZ
                unsigned x = a;
                if (x == 0) return static_cast<uint8_t>(32);
                int n = 0;
                while ((x & 1u) == 0) {
                    x >>= 1;
                    ++n;
                }
                return static_cast<uint8_t>(n);
            }
            case U8_POPCOUNT: {
                unsigned x = a;
                int c = 0;
                for (int i = 0; i < 8; ++i) {
                    if (x & (1u << i)) ++c;
                }
                return static_cast<uint8_t>(c);
            }
            case U8_REVERSE: return static_cast<uint8_t>(0);// Validated separately without 8-bit truncation.
            case U8_ADD: return static_cast<uint8_t>(static_cast<unsigned>(a) + static_cast<unsigned>(b));
            case U8_SUB: return static_cast<uint8_t>(static_cast<unsigned>(a) - static_cast<unsigned>(b));
            case U8_MUL: return static_cast<uint8_t>(static_cast<unsigned>(a) * static_cast<unsigned>(b));
            case U8_DIV: return b == 0 ? static_cast<uint8_t>(0) : static_cast<uint8_t>(static_cast<unsigned>(a) / static_cast<unsigned>(b));
            case U8_MOD: return b == 0 ? static_cast<uint8_t>(0) : static_cast<uint8_t>(static_cast<unsigned>(a) % static_cast<unsigned>(b));
            case U8_MIN: return static_cast<uint8_t>(std::min(static_cast<unsigned>(a), static_cast<unsigned>(b)));
            case U8_MAX: return static_cast<uint8_t>(std::max(static_cast<unsigned>(a), static_cast<unsigned>(b)));
            case U8_AND: return static_cast<uint8_t>(static_cast<unsigned>(a) & static_cast<unsigned>(b));
            case U8_OR: return static_cast<uint8_t>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
            case U8_XOR: return static_cast<uint8_t>(static_cast<unsigned>(a) ^ static_cast<unsigned>(b));
            case U8_SHL: return static_cast<uint8_t>(static_cast<unsigned>(a) << 2);
            case U8_SHR: return static_cast<uint8_t>(static_cast<unsigned>(a) >> 2);
            case U8_CAST_BYTE: return a;
            case U8_CAST_INT: return a;
            case U8_CAST_BOOL_TRUE: return a != 0 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
            case U8_CAST_BOOL_FALSE: return static_cast<uint8_t>(0);
            case U8_SELECT: return static_cast<int>(a) > 0 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
            case U8_EQ: return static_cast<uint8_t>(a == b ? 1 : 0);
            case U8_NE: return static_cast<uint8_t>(a != b ? 1 : 0);
            case U8_LT: return static_cast<uint8_t>(static_cast<unsigned>(a) < static_cast<unsigned>(b) ? 1 : 0);
            case U8_LE: return static_cast<uint8_t>(static_cast<unsigned>(a) <= static_cast<unsigned>(b) ? 1 : 0);
            case U8_GT: return static_cast<uint8_t>(static_cast<unsigned>(a) > static_cast<unsigned>(b) ? 1 : 0);
            case U8_GE: return static_cast<uint8_t>(static_cast<unsigned>(a) >= static_cast<unsigned>(b) ? 1 : 0);
            default: return static_cast<uint8_t>(0);
        }
    };

    // Verify int8 results
    for (auto i = 0u; i < static_cast<uint>(I8_COUNT); ++i) {
        if (i == static_cast<uint>(I8_CAST_UBYTE) || i == static_cast<uint>(I8_CAST_INT)) { continue; }
        auto expected = i8_ref(i);
        auto actual = i8_result[i];
        expect(actual == expected)
            << "int8 slot " << i << " expected " << static_cast<int>(expected)
            << " got " << static_cast<int>(actual);
    }

    // Verify uint8 results
    for (auto i = 0u; i < static_cast<uint>(U8_COUNT); ++i) {
        if (i == static_cast<uint>(U8_REVERSE) || i == static_cast<uint>(U8_CAST_BYTE) || i == static_cast<uint>(U8_CAST_INT)) { continue; }
        auto expected = u8_ref(i);
        auto actual = u8_result[i];
        expect(actual == expected)
            << "uint8 slot " << i << " expected " << static_cast<int>(expected)
            << " got " << static_cast<int>(actual);
    }

    auto reverse_bits_32 = [](uint x) noexcept {
        auto reversed = 0u;
        for (auto bit = 0u; bit < 32u; ++bit) {
            if ((x & (1u << bit)) != 0u) { reversed |= 1u << (31u - bit); }
        }
        return reversed;
    };
    auto expected_reverse = reverse_bits_32(static_cast<uint>(u8_host[U8_REVERSE]));
    expect(u8_bits_result[U8_REVERSE] == expected_reverse)
        << "uint8 reverse expected " << expected_reverse
        << " got " << u8_bits_result[U8_REVERSE];

    auto expected_i8_to_u8 = static_cast<int>(static_cast<uint8_t>(i8_host[I8_CAST_UBYTE]));
    auto expected_i8_to_int = static_cast<int>(i8_host[I8_CAST_INT]);
    expect(i8_wide_result[I8_CAST_UBYTE] == expected_i8_to_u8)
        << "int8-to-uint8 widening expected " << expected_i8_to_u8
        << " got " << i8_wide_result[I8_CAST_UBYTE];
    expect(i8_wide_result[I8_CAST_INT] == expected_i8_to_int)
        << "int8-to-int widening expected " << expected_i8_to_int
        << " got " << i8_wide_result[I8_CAST_INT];

    auto expected_u8_to_i8 = static_cast<int>(static_cast<int8_t>(u8_host[U8_CAST_BYTE]));
    auto expected_u8_to_int = static_cast<int>(u8_host[U8_CAST_INT]);
    expect(u8_wide_result[U8_CAST_BYTE] == expected_u8_to_i8)
        << "uint8-to-int8 widening expected " << expected_u8_to_i8
        << " got " << u8_wide_result[U8_CAST_BYTE];
    expect(u8_wide_result[U8_CAST_INT] == expected_u8_to_int)
        << "uint8-to-int widening expected " << expected_u8_to_int
        << " got " << u8_wide_result[U8_CAST_INT];

    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_byte8(device);
}
