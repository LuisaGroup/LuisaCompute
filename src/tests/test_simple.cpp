//
// Created by Mike Smith on 2021/2/27.
//

#include <numeric>
#include <iostream>

#include <core/clock.h>
#include <core/uuid.h>
#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/image.h>
#include <runtime/stream.h>
#include <runtime/buffer.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

using namespace luisa;
using namespace luisa::compute;

struct Test {
    float a;
    float b;
    float array[16];
};

LUISA_STRUCT(Test, a, b, array);

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

#if defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    Callable load = [](BufferVar<float> buffer, Var<uint> index) noexcept {
        return buffer[index];
    };

    Callable store = [](BufferVar<float> buffer, Var<uint> index, Var<float> value) noexcept {
        buffer[index] = value;
    };

    Callable add = [](Var<float> a, Var<float> b) noexcept {
        return a + b;
    };

    Kernel1D kernel_def = [&](BufferVar<float> source, BufferVar<float> result, Var<float> x) noexcept {
        set_block_size(256u);
        auto index = dispatch_id().x;
        store(result, index, add(load(source, index), x));
    };
    auto kernel = device.compile(kernel_def);

    static constexpr auto n = 1024u * 1024u;

    auto stream = device.create_stream();
    auto buffer = device.create_buffer<float>(n);
    auto result_buffer = device.create_buffer<float>(n);

    std::vector<float> data(n);
    std::vector<float> results(n);
    std::iota(data.begin(), data.end(), 1.0f);

    Clock clock;
    stream << buffer.copy_from(data.data());
    auto command_buffer = stream.command_buffer();
    for (auto i = 0; i < 10; i++) {
        command_buffer << kernel(buffer, result_buffer, 3).dispatch(n);
    }
    command_buffer << commit();
    stream << result_buffer.copy_to(results.data());
    auto t1 = clock.toc();
    stream << synchronize();
    auto t2 = clock.toc();

    LUISA_INFO("Dispatched in {} ms. Finished in {} ms.", t1, t2);
    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[n - 2u], results[n - 1u]);

    auto u = uuid::generate();
    auto uuid_string = u.string();
    uuid_string.erase(std::remove(uuid_string.begin(),  uuid_string.end(), '-'), uuid_string.end());
    auto uuid2 = uuid::from(uuid_string);
    LUISA_INFO("uuid: {}", uuid2.string());
    LUISA_INFO("Hash: {:x}", Hash64{}(uuid{}));
    LUISA_INFO("Hash: {:x}", Hash64{}(std::string_view{}));
    auto v = std::vector<uint>{};
    LUISA_INFO("Hash: {:x}", Hash64{}(v));
    LUISA_INFO("Hash: {:x}", Hash64{}(0u));
    LUISA_INFO("Hash: {:x}", Hash64{}(0ull));
    LUISA_INFO("uuid: {}", uuid::from_hash64(hash64("hello")).string_without_dash(true));
}
