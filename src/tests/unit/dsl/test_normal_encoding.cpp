// Test 32-bit octahedral normal encoding and device-side decoding.
//
// A deterministic set of random unit vectors plus poles, axes, diagonals, and
// fold-boundary cases is encoded on the host. Device results are checked both
// against the original vectors and against an independent double-precision CPU
// decoder.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

[[nodiscard]] uint oct_encode(float3 n) noexcept {
    auto inverse_l1 = 1.0f / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
    auto x = n.x * inverse_l1;
    auto y = n.y * inverse_l1;
    if (n.z < 0.0f) {
        auto old_x = x;
        auto old_y = y;
        x = (1.0f - std::abs(old_y)) * (old_x < 0.0f ? -1.0f : 1.0f);
        y = (1.0f - std::abs(old_x)) * (old_y < 0.0f ? -1.0f : 1.0f);
    }
    auto quantize = [](float v) noexcept {
        return static_cast<uint>(std::clamp(
            std::round((v * 0.5f + 0.5f) * 65535.0f), 0.0f, 65535.0f));
    };
    auto qx = quantize(x);
    auto qy = quantize(y);
    return qx | (qy << 16u);
}

[[nodiscard]] float3 oct_decode_reference(uint encoded) noexcept {
    auto x = static_cast<double>(encoded & 0xffffu) * (2.0 / 65535.0) - 1.0;
    auto y = static_cast<double>(encoded >> 16u) * (2.0 / 65535.0) - 1.0;
    auto z = 1.0 - std::abs(x) - std::abs(y);
    auto t = std::clamp(-z, 0.0, 1.0);
    x += x >= 0.0 ? -t : t;
    y += y >= 0.0 ? -t : t;
    auto inverse_length = 1.0 / std::sqrt(x * x + y * y + z * z);
    return make_float3(static_cast<float>(x * inverse_length),
                       static_cast<float>(y * inverse_length),
                       static_cast<float>(z * inverse_length));
}

[[nodiscard]] double length_double(float3 v) noexcept {
    auto x = static_cast<double>(v.x);
    auto y = static_cast<double>(v.y);
    auto z = static_cast<double>(v.z);
    return std::sqrt(x * x + y * y + z * z);
}

[[nodiscard]] double angular_error_degrees(float3 a, float3 b) noexcept {
    auto a_length = length_double(a);
    auto b_length = length_double(b);
    auto cosine = (static_cast<double>(a.x) * static_cast<double>(b.x) +
                   static_cast<double>(a.y) * static_cast<double>(b.y) +
                   static_cast<double>(a.z) * static_cast<double>(b.z)) /
                  (a_length * b_length);
    constexpr auto radians_to_degrees = 57.2957795130823208768;
    return std::acos(std::clamp(cosine, -1.0, 1.0)) * radians_to_degrees;
}

struct ErrorStats {
    size_t invalid_count{};
    double max_angular_error{};
    double sum_angular_error{};
    double max_reference_delta{};
    double max_unit_length_error{};
};

void test_normal_encoding(Device &device) {
    constexpr std::array special_normals{
        float3{1.0f, 0.0f, 0.0f},
        float3{-1.0f, 0.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f},
        float3{0.0f, -1.0f, 0.0f},
        float3{0.0f, 0.0f, 1.0f},
        float3{0.0f, 0.0f, -1.0f},
        float3{1.0f, 1.0f, 1.0f},
        float3{-1.0f, 1.0f, 1.0f},
        float3{1.0f, -1.0f, -1.0f},
        float3{-1.0f, -1.0f, -1.0f},
        float3{1.0f, 1.0f, 1.0e-6f},
        float3{1.0f, 1.0f, -1.0e-6f},
        float3{-1.0f, 1.0f, 1.0e-6f},
        float3{-1.0f, 1.0f, -1.0e-6f}};
    constexpr auto random_normal_count = 64u * 1024u;

    luisa::vector<float3> normals;
    normals.reserve(special_normals.size() + random_normal_count);
    for (auto v : special_normals) { normals.emplace_back(normalize(v)); }

    std::mt19937 random{0x4f435431u};
    std::uniform_real_distribution<float> distribution{-1.0f, 1.0f};
    while (normals.size() < special_normals.size() + random_normal_count) {
        auto v = make_float3(distribution(random), distribution(random), distribution(random));
        auto length_squared = dot(v, v);
        if (length_squared > 1.0e-8f && length_squared <= 1.0f) {
            normals.emplace_back(normalize(v));
        }
    }

    luisa::vector<uint> encoded_normals;
    encoded_normals.reserve(normals.size());
    for (auto normal : normals) { encoded_normals.emplace_back(oct_encode(normal)); }

    auto decoded_normal_buffer = device.create_buffer<float3>(normals.size());
    auto encoded_normal_buffer = device.create_buffer<uint>(normals.size());
    auto stream = device.create_stream();

    Kernel1D decode = [&] {
        auto oct_decode = [](Expr<uint> u) noexcept {
            Float2 p = make_float2(
                cast<float>(u & 0xffffu) * (1.0f / 65535.0f),
                cast<float>(u >> 16u) * (1.0f / 65535.0f));
            p = p * 2.0f - 1.0f;
            Float3 n = make_float3(p, 1.0f - abs(p.x) - abs(p.y));
            Float t = saturate(-n.z);
            return normalize(make_float3(
                n.xy() + select(t, -t, n.xy() >= 0.0f), n.z));
        };
        decoded_normal_buffer->write(
            dispatch_x(), oct_decode(encoded_normal_buffer->read(dispatch_x())));
    };

    auto shader = device.compile(decode);
    luisa::vector<float3> decoded_normals(normals.size());
    stream << encoded_normal_buffer.copy_from(luisa::span{encoded_normals})
           << shader().dispatch(static_cast<uint>(normals.size()))
           << decoded_normal_buffer.copy_to(luisa::span{decoded_normals})
           << synchronize();

    ErrorStats stats;
    for (size_t i = 0u; i < normals.size(); i++) {
        auto actual = decoded_normals[i];
        if (!std::isfinite(actual.x) || !std::isfinite(actual.y) ||
            !std::isfinite(actual.z)) {
            stats.invalid_count++;
            continue;
        }
        auto reference = oct_decode_reference(encoded_normals[i]);
        auto angular_error = angular_error_degrees(normals[i], actual);
        stats.max_angular_error = std::max(stats.max_angular_error, angular_error);
        stats.sum_angular_error += angular_error;
        stats.max_reference_delta = std::max(
            stats.max_reference_delta,
            std::max({std::abs(static_cast<double>(actual.x) - reference.x),
                      std::abs(static_cast<double>(actual.y) - reference.y),
                      std::abs(static_cast<double>(actual.z) - reference.z)}));
        stats.max_unit_length_error = std::max(
            stats.max_unit_length_error, std::abs(length_double(actual) - 1.0));
    }

    auto valid_count = normals.size() - stats.invalid_count;
    auto mean_angular_error = valid_count == 0u ? std::numeric_limits<double>::infinity() : stats.sum_angular_error / static_cast<double>(valid_count);
    LUISA_INFO("Oct16 normal roundtrip: {} samples, max/mean angular error = {}/{} deg, "
               "max CPU-reference delta = {}, max unit-length error = {}.",
               normals.size(), stats.max_angular_error, mean_angular_error,
               stats.max_reference_delta, stats.max_unit_length_error);

    expect(stats.invalid_count == 0u) << "decoded normals must all be finite";
    expect(stats.max_reference_delta <= 2.0e-5)
        << "device decoder must agree with independent double-precision CPU decoder";
    expect(stats.max_unit_length_error <= 2.0e-5)
        << "decoded normals must remain unit length";
    expect(stats.max_angular_error <= 0.01)
        << "oct16 maximum angular roundtrip error exceeds 0.01 degrees";
    expect(mean_angular_error <= 0.003)
        << "oct16 mean angular roundtrip error exceeds 0.003 degrees";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    test_normal_encoding(dc->device);
}
