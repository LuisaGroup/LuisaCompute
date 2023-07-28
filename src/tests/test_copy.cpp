#include <random>
#include <array>
#include <numeric>
#include <algorithm>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

struct Test {
    int a;
    float3x3 b;
    float4 c;
    [[nodiscard]] auto operator==(const Test &rhs) const noexcept -> bool {
        return a == rhs.a &&
               all(b[0] == rhs.b[0]) &&
               all(b[1] == rhs.b[1]) &&
               all(b[2] == rhs.b[2]) &&
               all(c == rhs.c);
    }
    [[nodiscard]] static auto make_random(std::mt19937 &rand) noexcept {
        std::uniform_real_distribution<float> uniform;
        auto a = static_cast<int>(rand());
        auto b = make_float3x3(uniform(rand), uniform(rand), uniform(rand),
                               uniform(rand), uniform(rand), uniform(rand),
                               uniform(rand), uniform(rand), uniform(rand));
        auto c = make_float4(uniform(rand), uniform(rand), uniform(rand), uniform(rand));
        return Test{a, b, c};
    }
};

template<>
struct fmt::formatter<Test> {
    constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
    template<typename FormatContext>
    auto format(const Test &t, FormatContext &ctx) {
        return fmt::format_to(ctx.out(),
                              "Test{{"
                              "a={}, "
                              "b=(({}, {}, {}), ({}, {}, {}), ({}, {}, {})), "
                              "c=({}, {}, {}, {})}}",
                              t.a,
                              t.b[0].x, t.b[0].y, t.b[0].z,
                              t.b[1].x, t.b[1].y, t.b[1].z,
                              t.b[2].x, t.b[2].y, t.b[2].z,
                              t.c.x, t.c.y, t.c.z, t.c.w);
    }
};

template<typename T, typename Generate>
void test_buffer(Device &device, size_t size, const Generate &g) noexcept {
    LUISA_INFO("Testing {} elements with type '{}'",
               size, typeid(T).name());
    luisa::vector<T> host_input;
    luisa::vector<T> host_output;
    host_input.reserve(size);
    host_output.resize(size);
    for (auto i = 0u; i < size; i++) {
        host_input.emplace_back(g());
    }
    auto buffer = device.create_buffer<T>(size);
    Stream stream = device.create_stream();
    stream << buffer.copy_from(host_input.data())
           << buffer.copy_to(host_output.data())
           << synchronize();
    for (auto i = 0u; i < size; i++) {
        LUISA_ASSERT(host_input[i] == host_output[i],
                     "Element {} mismatch: {} != {}",
                     i, host_input[i], host_output[i]);
    }
}

template<typename T, typename Size>
void test_texture(Device &device, PixelStorage storage, Size size, std::mt19937 &rng) noexcept {
    luisa::vector<uint8_t> host_input;
    luisa::vector<uint8_t> host_output;
    auto size_bytes = [&] {
        if constexpr (std::is_same_v<Size, uint2>) {
            LUISA_INFO("Testing image with size {}x{} and storage {}.",
                       size.x, size.y, to_string(storage));
            return pixel_storage_size(storage, make_uint3(size, 1u));
        } else {
            LUISA_INFO("Testing volume with size {}x{}x{} and storage {}.",
                       size.x, size.y, size.z, to_string(storage));
            return pixel_storage_size(storage, size);
        }
    }();
    host_input.resize(size_bytes);
    host_output.resize(size_bytes);
    for (auto i = 0u; i < size_bytes; i++) {
        host_input[i] = static_cast<uint8_t>(rng());
    }
    auto texture = [&] {
        if constexpr (std::is_same_v<Size, uint2>) {
            return device.create_image<T>(storage, size);
        } else {
            return device.create_volume<T>(storage, size);
        }
    }();
    Stream stream = device.create_stream();
    stream << texture.copy_from(host_input.data())
           << texture.copy_to(host_output.data())
           << synchronize();
    for (auto i = 0u; i < size_bytes; i++) {
        LUISA_ASSERT(host_input[i] == host_output[i],
                     "Element {} mismatch: {} != {}",
                     i, host_input[i], host_output[i]);
    }
}

int main(int argc, char *argv[]) {

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }

    Context context{argv[0]};
    Device device = context.create_device(argv[1]);

    std::array sizes{static_cast<size_t>(1u),
                     static_cast<size_t>(233u),
                     1_k,
                     233_k,
                     17_M + 655_k + 13u};

    std::mt19937 rand{std::random_device{}()};

    for (auto size : sizes) {
        test_buffer<int>(device, size, [&] { return static_cast<int>(rand()); });
        test_buffer<float>(device, size, [&] {
            auto dist = std::uniform_real_distribution{-233.f, 666.f};
            return dist(rand);
        });
        test_buffer<Test>(device, size, [&] { return Test::make_random(rand); });
    }

    test_texture<float>(device, PixelStorage::BYTE1, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::BYTE2, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::BYTE4, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::SHORT1, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::SHORT2, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::SHORT4, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::HALF1, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::HALF2, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::HALF4, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::FLOAT1, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::FLOAT2, make_uint2(233u, 666u), rand);
    test_texture<float>(device, PixelStorage::FLOAT4, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::BYTE1, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::BYTE2, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::BYTE4, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::SHORT1, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::SHORT2, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::SHORT4, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::INT1, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::INT2, make_uint2(233u, 666u), rand);
    test_texture<int>(device, PixelStorage::INT4, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::BYTE1, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::BYTE2, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::BYTE4, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::SHORT1, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::SHORT2, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::SHORT4, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::INT1, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::INT2, make_uint2(233u, 666u), rand);
    test_texture<uint>(device, PixelStorage::INT4, make_uint2(233u, 666u), rand);

    test_texture<float>(device, PixelStorage::BYTE1, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::BYTE2, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::BYTE4, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::SHORT1, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::SHORT2, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::SHORT4, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::HALF1, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::HALF2, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::HALF4, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::FLOAT1, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::FLOAT2, make_uint3(233u, 666u, 45u), rand);
    test_texture<float>(device, PixelStorage::FLOAT4, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::BYTE1, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::BYTE2, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::BYTE4, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::SHORT1, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::SHORT2, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::SHORT4, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::INT1, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::INT2, make_uint3(233u, 666u, 45u), rand);
    test_texture<int>(device, PixelStorage::INT4, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::BYTE1, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::BYTE2, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::BYTE4, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::SHORT1, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::SHORT2, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::SHORT4, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::INT1, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::INT2, make_uint3(233u, 666u, 45u), rand);
    test_texture<uint>(device, PixelStorage::INT4, make_uint3(233u, 666u, 45u), rand);
}

