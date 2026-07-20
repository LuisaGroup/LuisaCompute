// Test exact host/device copies for buffers, 2D images, and 3D volumes.
//
// The cases deliberately cover scalar and aggregate buffers, odd dimensions,
// page-crossing element counts, integer/normalized/floating-point textures,
// and block-compressed 2D textures without turning the test into a memory or
// assertion-count stress test.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>

#include <array>
#include <random>
#include <type_traits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

struct TestValue {
    int a;
    float3x3 b;
    float4 c;

    [[nodiscard]] auto operator==(const TestValue &rhs) const noexcept -> bool {
        return a == rhs.a &&
               all(b[0] == rhs.b[0]) &&
               all(b[1] == rhs.b[1]) &&
               all(b[2] == rhs.b[2]) &&
               all(c == rhs.c);
    }

    [[nodiscard]] static auto make_random(std::mt19937 &rng) noexcept {
        std::uniform_int_distribution<int> integer_distribution;
        std::uniform_real_distribution<float> float_distribution{-233.0f, 666.0f};
        auto b = make_float3x3(
            float_distribution(rng), float_distribution(rng), float_distribution(rng),
            float_distribution(rng), float_distribution(rng), float_distribution(rng),
            float_distribution(rng), float_distribution(rng), float_distribution(rng));
        auto c = make_float4(
            float_distribution(rng), float_distribution(rng),
            float_distribution(rng), float_distribution(rng));
        return TestValue{integer_distribution(rng), b, c};
    }
};

struct CopyStats {
    size_t cases{};
    size_t bytes{};
    size_t failed_cases{};

    void record(size_t byte_count, bool passed) noexcept {
        cases++;
        bytes += byte_count;
        failed_cases += passed ? 0u : 1u;
    }

    [[nodiscard]] bool passed() const noexcept { return failed_cases == 0u; }
};

template<typename T>
[[nodiscard]] bool verify_exact(luisa::span<const T> expected,
                                luisa::span<const T> actual,
                                luisa::string_view label) noexcept {
    size_t mismatch_count = 0u;
    size_t first_mismatch = expected.size();
    if (expected.size() != actual.size()) {
        LUISA_WARNING("{}: size mismatch (expected {}, got {}).",
                      label, expected.size(), actual.size());
        return false;
    }
    for (size_t i = 0u; i < expected.size(); i++) {
        if (!(expected[i] == actual[i])) {
            if (mismatch_count == 0u) { first_mismatch = i; }
            mismatch_count++;
        }
    }
    if (mismatch_count != 0u) {
        LUISA_WARNING("{}: {} mismatches; first mismatch at element {}.",
                      label, mismatch_count, first_mismatch);
    }
    return mismatch_count == 0u;
}

template<typename T, typename Generate>
void test_buffer(Device &device, size_t size, Generate &&generate,
                 CopyStats &stats) noexcept {
    luisa::vector<T> host_input;
    host_input.reserve(size);
    for (size_t i = 0u; i < size; i++) {
        host_input.emplace_back(generate());
    }
    luisa::vector<T> host_output(size);

    auto buffer = device.create_buffer<T>(size);
    auto stream = device.create_stream();
    stream << buffer.copy_from(luisa::span{host_input})
           << buffer.copy_to(luisa::span{host_output})
           << synchronize();

    auto label = luisa::format("buffer ({} elements x {} bytes)", size, sizeof(T));
    auto passed = verify_exact<T>(host_input, host_output, label);
    stats.record(size * sizeof(T), passed);
}

template<typename T, typename Size>
void test_texture(Device &device, PixelStorage storage, Size size,
                  std::mt19937 &rng, CopyStats &stats) noexcept {
    auto size_bytes = [&] {
        if constexpr (std::is_same_v<Size, uint2>) {
            return pixel_storage_size(storage, make_uint3(size, 1u));
        } else {
            return pixel_storage_size(storage, size);
        }
    }();

    luisa::vector<uint8_t> host_input(size_bytes);
    luisa::vector<uint8_t> host_output(size_bytes);
    std::uniform_int_distribution<unsigned int> byte_distribution{0u, 255u};
    for (auto &byte : host_input) {
        byte = static_cast<uint8_t>(byte_distribution(rng));
    }

    auto texture = [&] {
        if constexpr (std::is_same_v<Size, uint2>) {
            return device.create_image<T>(storage, size);
        } else {
            return device.create_volume<T>(storage, size);
        }
    }();
    auto stream = device.create_stream();
    stream << texture.copy_from(luisa::span{host_input})
           << texture.copy_to(luisa::span{host_output})
           << synchronize();

    auto label = luisa::format("{} copy ({})",
                               std::is_same_v<Size, uint2> ? "image" : "volume",
                               to_string(storage));
    auto passed = verify_exact<uint8_t>(host_input, host_output, label);
    stats.record(size_bytes, passed);
}

void expect_group(const CopyStats &stats, luisa::string_view label) noexcept {
    expect(stats.passed())
        << luisa::format("{}: {} of {} cases failed ({} bytes checked exactly)",
                         label, stats.failed_cases, stats.cases, stats.bytes);
}

void test_copy(Device &device) {
    // Includes a single element, odd/non-power-of-two counts, and boundaries
    // just beyond 256 and 4096 elements. The final case is large enough to
    // exercise multi-page transfers while keeping peak memory below 8 MiB.
    constexpr std::array<size_t, 5u> buffer_sizes{1u, 3u, 257u, 4099u, 65537u};
    constexpr std::array float_storages{
        PixelStorage::BYTE1, PixelStorage::BYTE2, PixelStorage::BYTE4,
        PixelStorage::SHORT1, PixelStorage::SHORT2, PixelStorage::SHORT4,
        PixelStorage::HALF1, PixelStorage::HALF2, PixelStorage::HALF4,
        PixelStorage::FLOAT1, PixelStorage::FLOAT2, PixelStorage::FLOAT4};
    constexpr std::array integer_storages{
        PixelStorage::BYTE1, PixelStorage::BYTE2, PixelStorage::BYTE4,
        PixelStorage::SHORT1, PixelStorage::SHORT2, PixelStorage::SHORT4,
        PixelStorage::INT1, PixelStorage::INT2, PixelStorage::INT4};
    constexpr std::array compressed_storages{
        PixelStorage::BC1, PixelStorage::BC2, PixelStorage::BC3,
        PixelStorage::BC4, PixelStorage::BC5, PixelStorage::BC6,
        PixelStorage::BC7};

    std::mt19937 rng{0x4c435043u};
    std::uniform_int_distribution<int> integer_distribution;
    std::uniform_real_distribution<float> float_distribution{-233.0f, 666.0f};

    CopyStats buffer_stats;
    for (auto size : buffer_sizes) {
        test_buffer<int>(device, size, [&] { return integer_distribution(rng); }, buffer_stats);
        test_buffer<float>(device, size, [&] { return float_distribution(rng); }, buffer_stats);
        test_buffer<TestValue>(device, size, [&] { return TestValue::make_random(rng); }, buffer_stats);
    }
    expect_group(buffer_stats, "buffer copies");

    CopyStats image_stats;
    constexpr auto image_size = make_uint2(37u, 23u);
    for (auto storage : float_storages) {
        test_texture<float>(device, storage, image_size, rng, image_stats);
    }
    for (auto storage : integer_storages) {
        test_texture<int>(device, storage, image_size, rng, image_stats);
        test_texture<uint>(device, storage, image_size, rng, image_stats);
    }
    // HIP stores this otherwise unsupported native layout as one packed
    // 32-bit array channel, so cover both its UNorm and UInt views here.
    if (device.backend_name() == "hip") {
        test_texture<float>(device, PixelStorage::R10G10B10A2,
                            image_size, rng, image_stats);
        test_texture<uint>(device, PixelStorage::R10G10B10A2,
                           image_size, rng, image_stats);
    }
    expect_group(image_stats, "uncompressed 2D image copies");

    CopyStats volume_stats;
    constexpr auto volume_size = make_uint3(17u, 11u, 5u);
    for (auto storage : float_storages) {
        test_texture<float>(device, storage, volume_size, rng, volume_stats);
    }
    for (auto storage : integer_storages) {
        test_texture<int>(device, storage, volume_size, rng, volume_stats);
        test_texture<uint>(device, storage, volume_size, rng, volume_stats);
    }
    if (device.backend_name() == "hip") {
        test_texture<float>(device, PixelStorage::R10G10B10A2,
                            volume_size, rng, volume_stats);
        test_texture<uint>(device, PixelStorage::R10G10B10A2,
                           volume_size, rng, volume_stats);
    }
    expect_group(volume_stats, "uncompressed 3D volume copies");

    CopyStats compressed_stats;
    constexpr auto compressed_size = make_uint2(32u, 20u);
    for (auto storage : compressed_storages) {
        test_texture<float>(device, storage, compressed_size, rng, compressed_stats);
    }
    expect_group(compressed_stats, "block-compressed 2D image copies");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    test_copy(dc->device);
}
