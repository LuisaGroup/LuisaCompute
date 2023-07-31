#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

struct Something {
    uint x;
    float3 v;
};

LUISA_STRUCT(Something, x, v){};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    Buffer<uint> buffer = device.create_buffer<uint>(4u);
    Kernel1D count_kernel = [&]() noexcept {
        Constant<uint> constant{1u};
        Var x = buffer->atomic(3u).fetch_add(constant[0]);
        if_(x == 0u, [&] {
            buffer->write(0u, 1u);
        });
    };
    Shader1D<> count = device.compile(count_kernel);

    uint4 host_buffer = make_uint4(0u);
    Stream stream = device.create_stream();

    Clock clock;
    clock.tic();
    stream << buffer.copy_from(&host_buffer)
           << count().dispatch(102400u)
           << buffer.copy_to(&host_buffer)
           << synchronize();
    double time = clock.toc();
    LUISA_INFO("Count: {} {}, Time: {} ms", host_buffer.x, host_buffer.w, time);
    LUISA_ASSERT(host_buffer.x == 1u && host_buffer.w == 102400u,
                 "Atomic operation failed.");

    Buffer<float> atomic_float_buffer = device.create_buffer<float>(1u);
    Kernel1D add_kernel = [&](BufferFloat buffer) noexcept {
        buffer.atomic(0u).fetch_sub(-1.f);
    };
    Shader1D<Buffer<float>> add_shader = device.compile(add_kernel);

    Kernel1D vector_atomic_kernel = [](BufferFloat3 buffer) noexcept {
        buffer.atomic(0u).x.fetch_add(1.f);
    };

    Kernel1D matrix_atomic_kernel = [](BufferFloat2x2 buffer) noexcept {
        buffer.atomic(0u)[1].x.fetch_add(1.f);
    };

    Kernel1D array_atomic_kernel = [](BufferVar<std::array<std::array<float4, 3u>, 5u>> buffer) noexcept {
        buffer.atomic(0u)[1][2][3].fetch_add(1.f);
    };

    Kernel1D struct_atomic_kernel = [](BufferVar<Something> buffer) noexcept {
        auto a = buffer.atomic(0u);
        a.v.x.fetch_max(1.f);
        Shared<float> s{16};
        s.atomic(0).compare_exchange(0.f, 1.f);
    };

    float result = 0.f;
    stream << atomic_float_buffer.copy_from(&result)
           << add_shader(atomic_float_buffer).dispatch(1024u)
           << atomic_float_buffer.copy_to(&result)
           << synchronize();
    LUISA_INFO("Atomic float result: {}.", result);
    LUISA_ASSERT(result == 1024.f, "Atomic float operation failed.");
}

