#include <random>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

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
    int4 e;
};

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
           all(lhs.e == rhs.e);
}

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();

    auto context = Context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

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
        a.e = make_int4(engine(), engine(), engine(), engine());
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
    stream << buffer_upload.copy_from(host_upload.data())
           << shader_upload(buffer_upload).dispatch(n)
           << shader_download(soa, buffer_download).dispatch(n)
           << buffer_download.copy_to(host_download.data())
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
}
