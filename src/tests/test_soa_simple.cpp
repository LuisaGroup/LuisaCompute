#include <random>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();

    auto context = Context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
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
    stream << buffer_upload.copy_from(host_upload.data())
           << shader_upload(soa, buffer_upload).dispatch(1024u)
           << shader_download(soa, buffer_download).dispatch(1024u)
           << buffer_download.copy_to(host_download.data())
           << synchronize();

    auto any_wrong = false;
    for (auto i = 0u; i < 1024u; i++) {
        if (any(host_upload[i] != host_download[i])) {
            LUISA_WARNING("SOA upload/download mismatch at index {}\n"
                          "  Expected: {}\n"
                          "  Actual:   {}",
                          i, host_upload[i], host_download[i]);
            any_wrong = true;
        }
    }
    if (any_wrong) {
        LUISA_ERROR("SOA upload/download mismatch.");
    } else {
        LUISA_INFO("SOA upload/download test passed.");
    }
}
