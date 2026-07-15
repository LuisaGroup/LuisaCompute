#include <random>
#include <luisa/luisa-compute.h>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct D {
    float3x3 m;
    float2x2 n;
};

LUISA_STRUCT(D, m, n) {};

struct A {
    float3 a;
    bool2 b;
    bool c;
    D d;
    std::array<std::array<int4, 1>, 1> e;
};

struct LightDistributionTreeNode {
    unsigned int left;
    unsigned int right;
    float leftContribution;
    float rightContribution;
};

struct LightDistributionCell {
    std::array<std::array<LightDistributionTreeNode, 256>, 2> nodes;
};

LUISA_STRUCT(LightDistributionTreeNode, left, right, leftContribution, rightContribution) {};
LUISA_STRUCT(LightDistributionCell, nodes) {};

LUISA_STRUCT(A, a, b, c, d, e) {};

[[nodiscard]] inline auto operator==(const D &lhs, const D &rhs) noexcept {
    for (auto i = 0u; i < 3u; i++) {
        if (any(lhs.m[i] != rhs.m[i])) {
            return false;
        }
    }
    for (auto i = 0u; i < 2u; i++) {
        if (any(lhs.n[i] != rhs.n[i])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline auto operator==(const A &lhs, const A &rhs) noexcept {
    return all(lhs.a == rhs.a) &&
           all(lhs.b == rhs.b) &&
           lhs.c == rhs.c &&
           lhs.d == rhs.d &&
           all(lhs.e[0][0] == rhs.e[0][0]);
}

int test_soa(Device &device) {
    luisa::log_level_verbose();

    constexpr auto n = 1357u;
    auto soa = device.create_soa<A>(n);

    auto rand = [](auto &engine) noexcept {
        std::uniform_real_distribution<float> dist{0.0f, 1.0f};
        A a{};
        a.a = make_float3(dist(engine), dist(engine), dist(engine));
        a.b = make_bool2(dist(engine) > 0.5f, dist(engine) > 0.5f);
        a.c = dist(engine) > 0.5f;
        a.d.m = make_float3x3(dist(engine), dist(engine), dist(engine),
                              dist(engine), dist(engine), dist(engine),
                              dist(engine), dist(engine), dist(engine));
        a.d.n = make_float2x2(dist(engine), dist(engine),
                              dist(engine), dist(engine));
        a.e[0][0] = make_int4(engine(), engine(), engine(), engine());
        return a;
    };
    luisa::vector<A> host_upload(n);
    std::mt19937 engine{std::random_device{}()};
    for (auto i = 0u; i < n; i++) {
        host_upload[i] = rand(engine);
    }

    auto buffer_upload = device.create_buffer<A>(n);
    auto buffer_download = device.create_buffer<A>(n);

    auto stream = device.create_stream();
    auto shader_upload = device.compile<1u>([&](BufferVar<A> upload) noexcept {
        auto i = dispatch_x();
        // soa passed to kernel by capture
        soa->write(i, upload.read(i));
    });
    auto shader_download = device.compile<1u>([](SOAVar<A> soa, BufferVar<A> download) noexcept {
        auto i = dispatch_x();
        // soa passed to kernel by argument
        download.write(i, soa.read(i));
    });

    luisa::vector<A> host_download(n);
    stream << buffer_upload.copy_from(luisa::span{host_upload})
           << shader_upload(buffer_upload).dispatch(n)
           << shader_download(soa, buffer_download).dispatch(n)
           << buffer_download.copy_to(luisa::span{host_download})
           << synchronize();

    auto any_wrong = false;
    for (auto i = 0u; i < n; i++) {
        if (host_upload[i] != host_download[i]) {
            LUISA_WARNING("SOA upload/download mismatch at index {}", i);
            any_wrong = true;
        }
    }
    expect(!any_wrong) << "soa_upload_download_roundtrip";

    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_soa(device);
}
