//
// Created by Mike Smith on 2021/4/6.
//

#include <stb/stb_image_write.h>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/syntax.h>
#include <dsl/printer.h>

using namespace luisa;
using namespace luisa::compute;

struct M {
    float3x3 m;
};

LUISA_STRUCT(M, m) {};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    auto device = context.create_device("cuda");

    auto m0 = make_float3x3(
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f);
    auto buffer = device.create_buffer<float3x3>(1u);
    auto stream = device.create_stream();
    auto scalar_buffer = device.create_buffer<float>(18u);

    Kernel1D test_kernel = [&](Float3x3 _, ArrayVar<float3x3, 1u> m) {
        set_block_size(1u, 1u, 1u);
        auto m2 = buffer.read(0u);
        auto one = def(1.f);
        auto s = make_float3x3(
            one, 0.f, 0.f,
            0.f, one * 2.f, 0.f,
            0.f, 0.f, one * 3.f);
        auto im2 = inverse(m2);
        auto m3 = make_float3x3(
            one, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f);
        for (auto i = 0u; i < 3u; i++) {
            for (auto j = 0u; j < 3u; j++) {
                scalar_buffer.write(i * 3u + j, m[0][i][j]);
                scalar_buffer.write(i * 3u + j + 9u, im2[i][j]);
            }
        }
    };
    auto test = device.compile(test_kernel);

    // dispatch
    std::array<float, 18u> download{};
    stream << buffer.copy_from(&m0)
           << test(m0, std::array{m0}).dispatch(1u)
           << scalar_buffer.copy_to(download.data())
           << synchronize();
    std::cout << "cbuffer:";
    for (auto i : std::span{download}.subspan(0u, 9u)) { std::cout << " " << i; }
    std::cout << "\nbuffer:";
    for (auto i : std::span{download}.subspan(9u)) { std::cout << " " << i; }
    std::cout << std::endl;
}
