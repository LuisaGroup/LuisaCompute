#include <random>
#include <luisa/luisa-compute.h>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_soa_simple(Device &device) {
    luisa::log_level_verbose();
    auto soa = device.create_soa<float3>(1024u);

    auto rand = [](auto &engine) noexcept {
        std::uniform_real_distribution<float> dist{0.0f, 1.0f};
        return float3{dist(engine), dist(engine), dist(engine)};
    };
    luisa::vector<float3> host_upload(1024u);
    std::mt19937 engine{std::random_device{}()};
    for (auto i = 0u; i < 1024u; i++) { host_upload[i] = rand(engine); }

    auto buffer_upload = device.create_buffer<float3>(1024u);
    auto buffer_download = device.create_buffer<float3>(1024u);

    auto stream = device.create_stream();
    auto shader_upload = device.compile<1u>([](SOAVar<float3> soa, BufferVar<float3> upload) noexcept {
        auto i = dispatch_x();
        soa.write(i, upload.read(i));
    });
    auto shader_download = device.compile<1u>([](SOAVar<float3> soa, BufferVar<float3> download) noexcept {
        auto i = dispatch_x();
        download.write(i, soa.read(i));
    });

    luisa::vector<float3> host_download(1024u);
    stream << buffer_upload.copy_from(luisa::span{host_upload})
           << shader_upload(soa, buffer_upload).dispatch(1024u)
           << shader_download(soa, buffer_download).dispatch(1024u)
           << buffer_download.copy_to(luisa::span{host_download})
           << synchronize();

    auto any_wrong = false;
    for (auto i = 0u; i < 1024u; i++) {
        if (any(host_upload[i] != host_download[i])) {
            LUISA_WARNING("SOA upload/download mismatch at index {}", i);
            any_wrong = true;
        }
    }
    expect(!any_wrong) << "soa_simple_upload_download_roundtrip";

    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_soa_simple(device);
}
