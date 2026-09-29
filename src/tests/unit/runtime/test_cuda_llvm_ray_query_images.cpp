#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <bit>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto kWidth = 17u;
constexpr auto kHeight = 3u;
constexpr auto kDepth = 2u;

template<typename T>
struct Sample {
    T stored;
    float decoded;
};

template<uint Dimension, typename T, size_t N, typename Shader>
[[nodiscard]] bool check_storage(Device &device, Stream &stream, const Accel &scene,
                                 const Shader &shader, PixelStorage storage,
                                 const std::array<Sample<T>, N> &samples) {
    static_assert(Dimension == 2u || Dimension == 3u);
    constexpr auto pixel_count = kWidth * kHeight * (Dimension == 3u ? kDepth : 1u);
    auto create_texture = [&] {
        if constexpr (Dimension == 2u) {
            return device.create_image<float>(storage, kWidth, kHeight);
        } else {
            return device.create_volume<float>(storage, kWidth, kHeight, kDepth);
        }
    };
    auto source = create_texture();
    auto destination = create_texture();
    auto observed = device.create_buffer<float4>(pixel_count);
    auto hits = device.create_buffer<uint4>(pixel_count);
    luisa::vector<T> host_source(pixel_count * 4u);
    luisa::vector<T> host_destination(pixel_count * 4u, static_cast<T>(0));
    luisa::vector<float4> host_observed(pixel_count);
    luisa::vector<uint4> host_hits(pixel_count);
    for (auto pixel = 0u; pixel < pixel_count; pixel++) {
        for (auto channel = 0u; channel < 4u; channel++) {
            host_source[pixel * 4u + channel] = samples[(pixel + channel * 3u) % N].stored;
        }
    }
    auto dispatch = [&] {
        if constexpr (Dimension == 2u) {
            return shader(scene, source, destination, observed, hits).dispatch(kWidth, kHeight);
        } else {
            return shader(scene, source, destination, observed, hits).dispatch(kWidth, kHeight, kDepth);
        }
    }();
    LUISA_INFO("RTX {}D texture storage {}: validating {} texels.", Dimension, static_cast<uint>(storage), pixel_count);
    stream << source.copy_from(luisa::span{host_source})
           << destination.copy_from(luisa::span{host_destination})
           << std::move(dispatch)
           << destination.copy_to(luisa::span{host_destination})
           << observed.copy_to(luisa::span{host_observed})
           << hits.copy_to(luisa::span{host_hits}) << synchronize();
    auto hit_correct = true;
    auto read_correct = true;
    auto write_correct = true;
    auto expected_hit = make_uint4(static_cast<uint>(HitType::Surface), 0u, 0u, 0x3f800000u);
    for (auto pixel = 0u; pixel < pixel_count; pixel++) {
        hit_correct = hit_correct && all(host_hits[pixel] == expected_hit);
        for (auto channel = 0u; channel < 4u; channel++) {
            auto actual_read = std::bit_cast<uint32_t>(host_observed[pixel][channel]);
            auto expected_read = std::bit_cast<uint32_t>(samples[(pixel + channel * 3u) % N].decoded);
            auto read_valid = actual_read == expected_read;
            if constexpr (std::is_same_v<T, uint8_t>) {
                // UNORM conversion can round once after reciprocal scaling.
                // All inputs are nonnegative finite values; compare at 1 ULP.
                auto distance = actual_read > expected_read ? actual_read - expected_read : expected_read - actual_read;
                read_valid = distance <= 1u;
            }
            auto actual_write = host_destination[pixel * 4u + channel];
            auto expected_write = host_source[pixel * 4u + (3u - channel)];
            auto write_valid = actual_write == expected_write;
            if constexpr (std::is_same_v<T, float>) {
                write_valid = std::bit_cast<uint32_t>(actual_write) == std::bit_cast<uint32_t>(expected_write);
            }
            if (!read_valid || !write_valid) {
                LUISA_WARNING("RTX {}D texture storage {} pixel {} channel {}: read bits={:08x}/{:08x}, stored={}/{}, valid={}/{}.",
                              Dimension, static_cast<uint>(storage), pixel, channel, actual_read, expected_read,
                              actual_write, expected_write, read_valid, write_valid);
            }
            read_correct = read_correct && read_valid;
            write_correct = write_correct && write_valid;
        }
    }
    expect(hit_correct) << "RTX texture dispatch must retain the ray query and exact hit" << Dimension << static_cast<uint>(storage);
    expect(read_correct) << "dynamic texture storage read and signed-zero conversion" << Dimension << static_cast<uint>(storage);
    expect(write_correct) << "dynamic texture storage write preserves every expected stored bit" << Dimension << static_cast<uint>(storage);
    return hit_correct && read_correct && write_correct;
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device(argc, argv);
    auto &device = dc.device;
    auto stream = device.create_stream();
    const std::array vertices{make_float3(-2.0f, -2.0f, 0.0f),
                              make_float3(2.0f, -2.0f, 0.0f),
                              make_float3(0.0f, 2.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto scene = device.create_accel();
    scene.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build() << scene.build() << synchronize();

    // Both textures are unbound parameters. Reuse this one compiled kernel
    // across storage formats to exercise the runtime PixelStorage selection.
    Kernel2D kernel = [](AccelVar accel, ImageFloat source, ImageFloat destination,
                         BufferFloat4 observed, BufferUInt4 hits) noexcept {
        set_block_size(8u, 4u, 1u);
        auto coord = dispatch_id().xy();
        auto index = coord.y * dispatch_size().x + coord.x;
        Float4 value = source.read(coord);
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 2.0f);
        auto hit = accel.traverse_any(ray, {})
                       .on_surface_candidate([](SurfaceCandidate &candidate) noexcept { candidate.commit(); })
                       .trace();
        observed.write(index, value);
        hits.write(index, make_uint4(hit->hit_type, hit->inst, hit->prim, hit->distance().as<uint>()));
        $if (hit->hit_type == static_cast<uint>(HitType::Surface)) {
            destination.write(coord, make_float4(value.w, value.z, value.y, value.x));
        };
    };
    LUISA_INFO("RTX dynamic Image<float> read/write fixture: AST hash={:016x}.", kernel.function()->function().hash());
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = false});
    std::array<Sample<uint8_t>, 8u> byte_samples{};
    constexpr std::array<uint8_t, 8u> bytes{0u, 1u, 17u, 51u, 127u, 128u, 254u, 255u};
    for (auto i = 0u; i < bytes.size(); i++) {
        byte_samples[i] = {bytes[i], static_cast<float>(bytes[i]) / 255.0f};
    }
    const std::array half_samples{
        Sample<uint16_t>{0x0000u, 0.0f}, Sample<uint16_t>{0x8000u, -0.0f},
        Sample<uint16_t>{0x3400u, 0.25f}, Sample<uint16_t>{0xbc00u, -1.0f},
        Sample<uint16_t>{0x3e00u, 1.5f}, Sample<uint16_t>{0x7bffu, 65504.0f},
        Sample<uint16_t>{0x0400u, 0x1p-14f}, Sample<uint16_t>{0x0001u, 0x1p-24f}};
    const std::array float_samples{
        Sample<float>{0.0f, 0.0f}, Sample<float>{-0.0f, -0.0f},
        Sample<float>{0.125f, 0.125f}, Sample<float>{-2.0f, -2.0f},
        Sample<float>{1.5f, 1.5f}, Sample<float>{65504.0f, 65504.0f},
        Sample<float>{0x1p-14f, 0x1p-14f}, Sample<float>{0x1p-24f, 0x1p-24f}};
    auto byte_correct = check_storage<2u>(device, stream, scene, shader, PixelStorage::BYTE4, byte_samples);
    auto half_correct = check_storage<2u>(device, stream, scene, shader, PixelStorage::HALF4, half_samples);
    auto float_correct = check_storage<2u>(device, stream, scene, shader, PixelStorage::FLOAT4, float_samples);

    Kernel3D volume_kernel = [](AccelVar accel, VolumeFloat source, VolumeFloat destination,
                                BufferFloat4 observed, BufferUInt4 hits) noexcept {
        set_block_size(8u, 4u, 1u);
        auto coord = dispatch_id();
        auto size = dispatch_size();
        auto index = (coord.z * size.y + coord.y) * size.x + coord.x;
        Float4 value = source.read(coord);
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 2.0f);
        auto hit = accel.traverse_any(ray, {})
                       .on_surface_candidate([](SurfaceCandidate &candidate) noexcept { candidate.commit(); })
                       .trace();
        observed.write(index, value);
        hits.write(index, make_uint4(hit->hit_type, hit->inst, hit->prim, hit->distance().as<uint>()));
        $if (hit->hit_type == static_cast<uint>(HitType::Surface)) {
            destination.write(coord, make_float4(value.w, value.z, value.y, value.x));
        };
    };
    LUISA_INFO("RTX dynamic Volume<float> read/write fixture: AST hash={:016x}.", volume_kernel.function()->function().hash());
    auto volume_shader = device.compile(volume_kernel, ShaderOption{.enable_cache = false, .enable_fast_math = false});
    auto volume_byte_correct = check_storage<3u>(device, stream, scene, volume_shader, PixelStorage::BYTE4, byte_samples);
    auto volume_half_correct = check_storage<3u>(device, stream, scene, volume_shader, PixelStorage::HALF4, half_samples);
    auto volume_float_correct = check_storage<3u>(device, stream, scene, volume_shader, PixelStorage::FLOAT4, float_samples);
    return byte_correct && half_correct && float_correct &&
                   volume_byte_correct && volume_half_correct && volume_float_correct ?
               0 :
               1;
}
