#include <random>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

struct RayRecord {
    Ray ray;
    float t;
    uint d1;
    uint d2;
    uint d3;
};

LUISA_STRUCT(RayRecord, ray, t, d1, d2, d3) {};

[[nodiscard]] inline auto operator==(const Ray &lhs, const Ray &rhs) noexcept {
    return lhs.compressed_origin[0] == rhs.compressed_origin[0] &&
           lhs.compressed_origin[1] == rhs.compressed_origin[1] &&
           lhs.compressed_origin[2] == rhs.compressed_origin[2] &&
           lhs.compressed_t_min == rhs.compressed_t_min &&
           lhs.compressed_direction[0] == rhs.compressed_direction[0] &&
           lhs.compressed_direction[1] == rhs.compressed_direction[1] &&
           lhs.compressed_direction[2] == rhs.compressed_direction[2] &&
           lhs.compressed_t_max == rhs.compressed_t_max;
}

[[nodiscard]] inline auto operator==(const RayRecord &lhs, const RayRecord &rhs) noexcept {
    return lhs.ray == rhs.ray &&
           lhs.t == rhs.t &&
           lhs.d1 == rhs.d1 &&
           lhs.d2 == rhs.d2 &&
           lhs.d3 == rhs.d3;
}

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();

    auto context = Context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    constexpr auto n = 1024u * 1024u;
    auto soa = device.create_soa<RayRecord>(n);

    auto rand = [](auto &engine) noexcept {
        std::uniform_real_distribution<float> dist{0.0f, 1.0f};
        RayRecord a;
        a.ray.compressed_origin[0] = dist(engine);
        a.ray.compressed_origin[1] = dist(engine);
        a.ray.compressed_origin[2] = dist(engine);
        a.ray.compressed_t_min = dist(engine);
        a.ray.compressed_direction[0] = dist(engine);
        a.ray.compressed_direction[1] = dist(engine);
        a.ray.compressed_direction[2] = dist(engine);
        a.ray.compressed_t_max = dist(engine);
        a.t = dist(engine);
        a.d1 = engine();
        a.d2 = engine();
        a.d3 = engine();
        return a;
    };
    luisa::vector<RayRecord> host_upload(n);
    std::mt19937 engine{std::random_device{}()};
    for (auto i = 0u; i < n; i++) {
        host_upload[i] = rand(engine);
    }

    auto buffer_upload = device.create_buffer<RayRecord>(n);
    auto buffer_download = device.create_buffer<RayRecord>(n);

    auto stream = device.create_stream();
    auto shader_upload = device.compile<1u>([&](BufferVar<RayRecord> upload) noexcept {
        auto i = dispatch_x();
        // soa passed to kernel by capture
        soa->write(i, upload.read(i));
    });
    auto shader_download = device.compile<1u>([](SOAVar<RayRecord> soa, BufferVar<RayRecord> download) noexcept {
        auto i = dispatch_x();
        // soa passed to kernel by argument
        download.write(i, soa.read(i));
    });

    constexpr auto subview_offset = 23333u;
    constexpr auto subview_size = 66666u;
    auto buffer_ray_download = device.create_buffer<Ray>(subview_size);
    auto shader_ray_download = device.compile<1u>([&](SOAVar<RayRecord> soa) noexcept {
        auto i = dispatch_x();
        buffer_ray_download->write(i, soa.ray->read(i));
    });

    luisa::vector<RayRecord> host_download(n);
    luisa::vector<Ray> host_ray_download(subview_size);
    stream << buffer_upload.copy_from(host_upload.data())
           << shader_upload(buffer_upload).dispatch(n)
           << shader_download(soa, buffer_download).dispatch(n)
           << buffer_download.copy_to(host_download.data())
           << shader_ray_download(soa.subview(subview_offset, subview_size)).dispatch(subview_size)
           << buffer_ray_download.copy_to(host_ray_download.data())
           << synchronize();

    auto any_wrong = false;
    for (auto i = 0u; i < n; i++) {
        if (host_upload[i] != host_download[i]) {
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

    auto any_wrong_ray = false;
    for (auto i = 0u; i < subview_size; i++) {
        if (host_upload[i + subview_offset].ray != host_ray_download[i]) {
            LUISA_WARNING("SOA subview download mismatch at index {}\n"
                          "  Expected: {}\n"
                          "  Actual:   {}",
                          i, host_upload[i + subview_offset].ray, host_ray_download[i]);
            any_wrong_ray = true;
        }
    }
    if (any_wrong_ray) {
        LUISA_ERROR("SOA subview download mismatch.");
    } else {
        LUISA_INFO("SOA subview download test passed.");
    }
}
