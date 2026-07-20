// Exact signed narrow-texture I/O regression for the HIP LLVM backend.
//
// R8/R16 SInt share PixelStorage values with their UInt counterparts. This
// test therefore checks shader-side sign extension independently from image
// writes, and checks writes independently by downloading the raw narrow
// texels. Both direct 2D and direct 3D paths are covered.

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <limits>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto image_size = make_uint2(19u, 7u);
constexpr auto volume_size = make_uint3(7u, 5u, 3u);

constexpr std::array signed_byte_values{
    std::numeric_limits<byte>::min(), byte{-101}, byte{-17}, byte{-1},
    byte{0}, byte{1}, byte{23}, byte{96},
    std::numeric_limits<byte>::max()};

constexpr std::array signed_short_values{
    std::numeric_limits<short>::min(), short{-23117}, short{-1025}, short{-1},
    short{0}, short{1}, short{2049}, short{27123},
    std::numeric_limits<short>::max()};

template<typename T, size_t N>
[[nodiscard]] auto make_narrow_input(size_t count,
                                     const std::array<T, N> &values,
                                     size_t offset) noexcept {
    luisa::vector<T> result(count);
    for (auto i = 0u; i < count; i++) {
        result[i] = values[(i * 5u + offset) % values.size()];
    }
    return result;
}

template<typename T>
[[nodiscard]] auto widen(luisa::span<const T> values) noexcept {
    luisa::vector<int> result(values.size());
    for (auto i = 0u; i < values.size(); i++) {
        result[i] = static_cast<int>(values[i]);
    }
    return result;
}

template<typename T>
[[nodiscard]] bool exact_narrow(luisa::span<const T> actual,
                                luisa::span<const int> expected,
                                luisa::string_view label) noexcept {
    for (auto i = 0u; i < actual.size(); i++) {
        if (static_cast<int>(actual[i]) != expected[i]) {
            LUISA_WARNING("{} texel {} mismatch: got {}, expected {}.",
                          label, i, static_cast<int>(actual[i]), expected[i]);
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool exact_wide(luisa::span<const int> actual,
                              luisa::span<const int> expected,
                              luisa::string_view label) noexcept {
    for (auto i = 0u; i < actual.size(); i++) {
        if (actual[i] != expected[i]) {
            LUISA_WARNING("{} texel {} mismatch: got {}, expected {}.",
                          label, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

void test_signed_image_io(Device &device) {
    auto texel_count = static_cast<size_t>(image_size.x) * image_size.y;
    auto input8 = make_narrow_input(texel_count, signed_byte_values, 0u);
    auto input16 = make_narrow_input(texel_count, signed_short_values, 2u);
    auto narrow_write8 = make_narrow_input(
        texel_count, signed_byte_values, 4u);
    auto narrow_write16 = make_narrow_input(
        texel_count, signed_short_values, 6u);
    auto write8 = widen<byte>(narrow_write8);
    auto write16 = widen<short>(narrow_write16);

    auto image8 = device.create_image<int>(PixelStorage::BYTE1, image_size);
    auto image16 = device.create_image<int>(PixelStorage::SHORT1, image_size);
    auto write8_buffer = device.create_buffer<int>(texel_count);
    auto write16_buffer = device.create_buffer<int>(texel_count);
    auto read8_buffer = device.create_buffer<int>(texel_count);
    auto read16_buffer = device.create_buffer<int>(texel_count);

    Kernel2D kernel = [](ImageInt tex8, ImageInt tex16,
                         BufferInt write8_values,
                         BufferInt write16_values,
                         BufferInt read8_values,
                         BufferInt read16_values) noexcept {
        auto coord = dispatch_id().xy();
        auto index = coord.y * image_size.x + coord.x;
        read8_values.write(index, tex8.read(coord).x);
        read16_values.write(index, tex16.read(coord).x);
        tex8.write(coord, make_int4(write8_values.read(index)));
        tex16.write(coord, make_int4(write16_values.read(index)));
    };

    auto shader = device.compile(kernel);
    luisa::vector<int> read8(texel_count);
    luisa::vector<int> read16(texel_count);
    luisa::vector<byte> output8(texel_count);
    luisa::vector<short> output16(texel_count);
    auto stream = device.create_stream();
    stream << image8.copy_from(luisa::span{input8})
           << image16.copy_from(luisa::span{input16})
           << write8_buffer.copy_from(luisa::span{write8})
           << write16_buffer.copy_from(luisa::span{write16})
           << shader(image8, image16, write8_buffer, write16_buffer,
                     read8_buffer, read16_buffer)
                  .dispatch(image_size)
           << read8_buffer.copy_to(luisa::span{read8})
           << read16_buffer.copy_to(luisa::span{read16})
           << image8.copy_to(luisa::span{output8})
           << image16.copy_to(luisa::span{output16})
           << synchronize();

    auto expected_read8 = widen<byte>(input8);
    auto expected_read16 = widen<short>(input16);
    expect(exact_wide(read8, expected_read8, "2D R8SInt read"))
        << "R8SInt image reads must sign-extend every texel";
    expect(exact_wide(read16, expected_read16, "2D R16SInt read"))
        << "R16SInt image reads must sign-extend every texel";
    expect(exact_narrow<byte>(output8, write8, "2D R8SInt write"))
        << "R8SInt image writes must preserve representable signed values";
    expect(exact_narrow<short>(output16, write16, "2D R16SInt write"))
        << "R16SInt image writes must preserve representable signed values";
}

void test_signed_volume_io(Device &device) {
    auto texel_count = static_cast<size_t>(volume_size.x) *
                       volume_size.y * volume_size.z;
    auto input8 = make_narrow_input(texel_count, signed_byte_values, 1u);
    auto input16 = make_narrow_input(texel_count, signed_short_values, 3u);
    auto narrow_write8 = make_narrow_input(
        texel_count, signed_byte_values, 5u);
    auto narrow_write16 = make_narrow_input(
        texel_count, signed_short_values, 7u);
    auto write8 = widen<byte>(narrow_write8);
    auto write16 = widen<short>(narrow_write16);

    auto volume8 = device.create_volume<int>(PixelStorage::BYTE1, volume_size);
    auto volume16 = device.create_volume<int>(PixelStorage::SHORT1, volume_size);
    auto write8_buffer = device.create_buffer<int>(texel_count);
    auto write16_buffer = device.create_buffer<int>(texel_count);
    auto read8_buffer = device.create_buffer<int>(texel_count);
    auto read16_buffer = device.create_buffer<int>(texel_count);

    Kernel3D kernel = [](VolumeInt tex8, VolumeInt tex16,
                         BufferInt write8_values,
                         BufferInt write16_values,
                         BufferInt read8_values,
                         BufferInt read16_values) noexcept {
        auto coord = dispatch_id();
        auto index = (coord.z * volume_size.y + coord.y) *
                         volume_size.x +
                     coord.x;
        read8_values.write(index, tex8.read(coord).x);
        read16_values.write(index, tex16.read(coord).x);
        tex8.write(coord, make_int4(write8_values.read(index)));
        tex16.write(coord, make_int4(write16_values.read(index)));
    };

    auto shader = device.compile(kernel);
    luisa::vector<int> read8(texel_count);
    luisa::vector<int> read16(texel_count);
    luisa::vector<byte> output8(texel_count);
    luisa::vector<short> output16(texel_count);
    auto stream = device.create_stream();
    stream << volume8.copy_from(luisa::span{input8})
           << volume16.copy_from(luisa::span{input16})
           << write8_buffer.copy_from(luisa::span{write8})
           << write16_buffer.copy_from(luisa::span{write16})
           << shader(volume8, volume16, write8_buffer, write16_buffer,
                     read8_buffer, read16_buffer)
                  .dispatch(volume_size)
           << read8_buffer.copy_to(luisa::span{read8})
           << read16_buffer.copy_to(luisa::span{read16})
           << volume8.copy_to(luisa::span{output8})
           << volume16.copy_to(luisa::span{output16})
           << synchronize();

    auto expected_read8 = widen<byte>(input8);
    auto expected_read16 = widen<short>(input16);
    expect(exact_wide(read8, expected_read8, "3D R8SInt read"))
        << "R8SInt volume reads must sign-extend every texel";
    expect(exact_wide(read16, expected_read16, "3D R16SInt read"))
        << "R16SInt volume reads must sign-extend every texel";
    expect(exact_narrow<byte>(output8, write8, "3D R8SInt write"))
        << "R8SInt volume writes must preserve representable signed values";
    expect(exact_narrow<short>(output16, write16, "3D R16SInt write"))
        << "R16SInt volume writes must preserve representable signed values";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP signed narrow 2D image I/O preserves exact values"_test = [&] {
        test_signed_image_io(dc->device);
    };
    "HIP signed narrow 3D image I/O preserves exact values"_test = [&] {
        test_signed_volume_io(dc->device);
    };
}
