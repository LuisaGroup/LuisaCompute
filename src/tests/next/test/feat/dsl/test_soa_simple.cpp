/**
 * @file test/feat/ast/test_soa_simple.cpp
 * @author sailing-innocent
 * @date 2023/11/04
 * @brief the dsl soa simple test case
*/

#include "common/config.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

#include <random>
using namespace luisa;
using namespace luisa::compute;
namespace luisa::test {

int test_soa_simple(Device &device) {
    const uint N = 1024u;
    auto soa = device.create_soa<float3>(N);
    auto rand = [](auto &engine) noexcept {
        std::uniform_real_distribution<float> dist{0.0f, 1.0f};
        return float3{dist(engine), dist(engine), dist(engine)};
    };
    luisa::vector<float3> host_upload(N);
    std::mt19937 engine{std::random_device{}()};
    for (auto i = 0u; i < N; i++) { host_upload[i] = rand(engine); }
    auto buffer_upload = device.create_buffer<float3>(N);
    auto buffer_download = device.create_buffer<float3>(N);
    auto stream = device.create_stream();
    auto shader_upload = device.compile<1u>([](SOAVar<float3> soa, BufferVar<float3> upload) noexcept {
        auto i = dispatch_x();
        soa.write(i, upload.read(i));
    });
    auto shader_download = device.compile<1u>([](SOAVar<float3> soa, BufferVar<float3> download) noexcept {
        auto i = dispatch_x();
        download.write(i, soa.read(i));
    });

    luisa::vector<float3> host_download(N);
    stream << buffer_upload.copy_from(host_upload.data())
           << shader_upload(soa, buffer_upload).dispatch(N)
           << shader_download(soa, buffer_download).dispatch(N)
           << buffer_download.copy_to(host_download.data())
           << synchronize();

    for (auto i = 0u; i < N; i++) {
        for (auto j = 0u; j < 3u; j++)
            CHECK_MESSAGE(host_upload[i][j] == host_download[i][j], "SOA upload/download mismatch at index {}\n Expected: {}\n  Actual:   {}", i, host_upload[i], host_download[i]);
    }
    return 0;
}

}// namespace luisa::test

TEST_SUITE("dsl") {
    LUISA_TEST_CASE_WITH_DEVICE("dsl_soa_simple", luisa::test::test_soa_simple(device) == 0);
}